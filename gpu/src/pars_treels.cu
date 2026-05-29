// pars_treels.cu — GPU per-pattern parsimony for treels topologies
// Computes _pattern_pars[ptn] for each treels tree on GPU, replacing CPU computeParsimony().

#include "../include/pars_treels.cuh"
#include "../include/topo_helpers.cuh"
#include "../include/utils.cuh"

#include <cstdio>

namespace mpbootgpu
{

// ─── Kernel ──────────────────────────────────────────────────────────────────
// One block = one treels tree. One warp (32 threads) per block.
// Computes per-pattern Fitch parsimony for tree t:
//   pars_ptn[ptn] = #substitutions tree t requires for pattern ptn
//
// Layout reuse: d_parsVect[k × parsVectPerTree] is used as scratch.
//   Tips (nodes 1..N): identical across all K slots — read-only.
//   Inner nodes (N+1..2N-1): overwritten during traversal.

template <int STATES>
__global__ void treelsPatternParsKernel(
    int                         batch_start,
    int                         T_total,
    int                         N,
    int                         width,
    int                         nptn,
    int                         nptn_padded,
    int                         start_vf,        // = 0 (tip 1 vface)
    const int* __restrict__     d_treelsBackVf,  // [T_total × kMaxVFaces]
    parsimonyNumber* __restrict__ d_parsVect,    // [K × parsVectPerTree]
    size_t                      parsVectPerTree,
    uint16_t* __restrict__      d_out            // [T_total × nptn_padded]
)
{
    const int k    = blockIdx.x;              // batch slot (0..batch_size-1)
    const int t    = batch_start + k;         // global treels index
    if (t >= T_total) return;
    const int lane = threadIdx.x;             // 0..31

    const int num_inner = N - 1;

    // ── Shared memory layout ──────────────────────────────────────────────────
    // ti[3 × num_inner] + tiStack[256] only — pars_ptn lives in global d_out
    extern __shared__ char smem[];
    int* ti       = (int*)smem;
    int* tiStack  = ti + 3 * num_inner;

    // pars_ptn: write directly to output buffer (each lane owns pars_ptn[b*32..b*32+31])
    uint16_t* pars_ptn = d_out + (size_t)t * nptn_padded;

    // Init pars_ptn to 0
    for (int i = lane; i < nptn_padded; i += 32)
        pars_ptn[i] = 0;
    __syncwarp();

    const int* bvf = d_treelsBackVf + (size_t)t * kMaxVFaces;

    // ── Build traversal (lane 0) ───────────────────────────────────────────────
    // Pre-order DFS from inner node adjacent to tip start_vf.
    // Iterate ti[] in REVERSE to get post-order for newview.
    int tiSize = 0;
    if (lane == 0)
    {
        int top  = -1;
        int tis  = 0;
        int inner_start = bvf[start_vf];  // inner node vface adjacent to tip 1
        tiStack[++top] = inner_start;

        while (top >= 0)
        {
            int p_vf  = tiStack[top--];
            if (p_vf < N) continue;         // skip tips (safety guard)

            int pNext = vfNextFace(p_vf, N);
            int pNnxt = vfNnxtFace(p_vf, N);
            int q_vf  = bvf[pNext];
            int r_vf  = bvf[pNnxt];

            // Push inner children
            if (r_vf >= N) tiStack[++top] = r_vf;
            if (q_vf >= N) tiStack[++top] = q_vf;

            // Record node numbers (pre-order; reversed = post-order during newview)
            ti[tis++] = vfToNum(p_vf, N);
            ti[tis++] = vfToNum(q_vf, N);
            ti[tis++] = vfToNum(r_vf, N);
        }
        tiSize = tis;
    }
    __syncwarp();
    tiSize = __shfl_sync(0xFFFFFFFFu, tiSize, 0);

    const int n_inner = tiSize / 3;

    // parsVect scratch: slot k (tip data valid; inner nodes overwritten)
    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;

    // ── Fitch newview + per-pattern accumulation (post-order = reverse ti) ────
    for (int idx = n_inner - 1; idx >= 0; idx--)
    {
        const int p_num = ti[idx * 3    ];
        const int q_num = ti[idx * 3 + 1];
        const int r_num = ti[idx * 3 + 2];

        parsimonyNumber*       p_base = pars_tree + (size_t)p_num * width * STATES;
        const parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * STATES;
        const parsimonyNumber* r_base = pars_tree + (size_t)r_num * width * STATES;

        for (int b = lane; b < width; b += 32)
        {
            parsimonyNumber t_N = 0;
            parsimonyNumber isect_s[STATES], union_s[STATES];
            #pragma unroll
            for (int s = 0; s < STATES; s++)
            {
                const parsimonyNumber lv = q_base[(size_t)s * width + b];
                const parsimonyNumber rv = r_base[(size_t)s * width + b];
                isect_s[s] = lv & rv;
                union_s[s] = lv | rv;
                t_N |= isect_s[s];
            }
            t_N = ~t_N;  // bits where no state agrees → substitution needed

            // Update parsVect[p] = isect | (missed & union)
            #pragma unroll
            for (int s = 0; s < STATES; s++)
                p_base[(size_t)s * width + b] = isect_s[s] | (t_N & union_s[s]);

            // Per-pattern accumulation: lane b owns positions 32b..32b+31
            const int base = b << 5;  // b * 32
            unsigned int missed = (unsigned int)t_N;
            // Mask off bits beyond valid pattern range (last block only)
            if (base + 31 >= nptn && base < nptn)
                missed &= (2u << (nptn - base - 1)) - 1u;
            else if (base >= nptn)
                missed = 0;
            while (missed)
            {
                const int bit = __ffs(missed) - 1;
                pars_ptn[base + bit]++;
                missed &= missed - 1;  // clear lowest bit
            }
        }
    }
    __syncwarp();

    // ── Root edge: cross(tip_start, inner_adj_to_start) ───────────────────────
    {
        const int tip_num   = vfToNum(start_vf, N);          // = 1 (tip 1)
        const int inner_num = vfToNum(bvf[start_vf], N);     // inner adj to tip 1

        const parsimonyNumber* t_base = pars_tree + (size_t)tip_num   * width * STATES;
        const parsimonyNumber* i_base = pars_tree + (size_t)inner_num * width * STATES;

        for (int b = lane; b < width; b += 32)
        {
            parsimonyNumber ored = 0;
            #pragma unroll
            for (int s = 0; s < STATES; s++)
                ored |= t_base[(size_t)s * width + b] & i_base[(size_t)s * width + b];
            const int base = b << 5;
            unsigned int missed = (unsigned int)(~ored);
            // Mask off bits beyond valid pattern range
            if (base + 31 >= nptn && base < nptn)
                missed &= (2u << (nptn - base - 1)) - 1u;
            else if (base >= nptn)
                missed = 0;
            while (missed)
            {
                const int bit = __ffs(missed) - 1;
                pars_ptn[base + bit]++;
                missed &= missed - 1;
            }
        }
    }
    __syncwarp();

    // pars_ptn == d_out + t*nptn_padded — already written in place, no copy needed
}

// ─── Host launcher ───────────────────────────────────────────────────────────

bool gpuComputeTreelsPatternPars(
    GpuParsimonyMem* mem,
    int              n_treels,
    int              start_vf,
    int              nptn,
    int              nptn_padded,
    uint16_t*        d_treels_ptn_pars,
    cudaStream_t     stream
)
{
    if (n_treels <= 0 || mem->d_treelsBackVf == nullptr) return true;

    const int N      = mem->mxtips;
    const int width  = mem->width;
    const int states = mem->states;
    const int K      = mem->K;  // number of parsVect slots available as scratch

    // Only Fitch mode supported
    if (mem->d_cost_matrix != nullptr) return false;

    // Shared memory: ti[3*(N-1)] + tiStack[256] only — pars_ptn is in global d_out
    // For N≤800: (3*799+256)*4 = 10.4 KB — always within default 48 KB limit
    const size_t smem_bytes =
        (size_t)(3 * (N - 1) + 256) * sizeof(int);

    // Process in batches of K (reuse d_parsVect slots as scratch)
    for (int batch_start = 0; batch_start < n_treels; batch_start += K)
    {
        const int batch_size = std::min(K, n_treels - batch_start);
        const dim3 grid(batch_size);
        const dim3 block(32);  // one warp per tree

        if (states == 4)
        {
            treelsPatternParsKernel<4><<<grid, block, smem_bytes, stream>>>(
                batch_start, n_treels, N, width, nptn, nptn_padded, start_vf,
                mem->d_treelsBackVf, mem->d_parsVect, mem->parsVectPerTree,
                d_treels_ptn_pars
            );
        }
        else if (states == 20)
        {
            treelsPatternParsKernel<20><<<grid, block, smem_bytes, stream>>>(
                batch_start, n_treels, N, width, nptn, nptn_padded, start_vf,
                mem->d_treelsBackVf, mem->d_parsVect, mem->parsVectPerTree,
                d_treels_ptn_pars
            );
        }
        else
        {
            return false;  // unsupported states
        }
        CUDA_CHECK(cudaGetLastError());  // catch launch errors (e.g. smem > device max)
        cudaStreamSynchronize(stream);
    }

    return true;
}

}  // namespace mpbootgpu
