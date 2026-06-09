// pars_treels.cu — GPU per-pattern parsimony for treels topologies
// Computes _pattern_pars[ptn] for each treels tree on GPU, replacing CPU computeParsimony().

#include "../include/pars_treels.cuh"
#include "../include/topo_helpers.cuh"
#include "../include/utils.cuh"

#include <cstdio>

namespace mpbootgpu
{

// ─── Fitch kernel ────────────────────────────────────────────────────────────
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

            // Per-pattern accumulation: bit i of t_N → pattern b*32+i needs substitution.
            // Write pars_ptn[ptn] = 1 for each set bit (Fitch: 0 or 1 per pattern per edge).
            {
                unsigned int tN = (unsigned int)t_N;
                const int base_ptn = b * 32;
                for (int bit = 0; bit < 32; bit++) {
                    const int ptn = base_ptn + bit;
                    if (ptn < nptn)
                        pars_ptn[ptn] += (uint16_t)((tN >> bit) & 1u);
                }
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
            {
                unsigned int cross = (unsigned int)(~ored);
                const int base_ptn = b * 32;
                for (int bit = 0; bit < 32; bit++) {
                    const int ptn = base_ptn + bit;
                    if (ptn < nptn)
                        pars_ptn[ptn] += (uint16_t)((cross >> bit) & 1u);
                }
            }
        }
    }
    __syncwarp();

    // pars_ptn == d_out + t*nptn_padded — already written in place, no copy needed
}

// ─── Sankoff kernel ───────────────────────────────────────────────────────────
// One block = one treels tree. One warp (32 threads) per block.
// Computes per-pattern Sankoff parsimony for tree t.
//
// For Sankoff: parsVect[node][state][ptn] = cost uint32 (not a bitmask).
// width = nptn (number of informative patterns).
// pars_ptn[b] = min_ij(tip[i*width+b] + cost[i][j] + inner[j*width+b])
//   where (tip, inner) is the root edge (tip 1 and its adjacent inner node).
//
// Cost matrix loaded into shared memory for L1 caching.

template <int STATES>
__global__ void treelsPatternParsKernelSankoff(
    int                              batch_start,
    int                              T_total,
    int                              N,
    int                              width,        // = nptn for Sankoff
    int                              nptn_padded,
    int                              start_vf,
    const int* __restrict__          d_treelsBackVf,
    parsimonyNumber* __restrict__    d_parsVect,
    size_t                           parsVectPerTree,
    const unsigned int* __restrict__ d_cost_matrix,  // [STATES × STATES] device
    uint16_t* __restrict__           d_out
)
{
    const int k    = blockIdx.x;
    const int t    = batch_start + k;
    if (t >= T_total) return;
    const int lane = threadIdx.x;

    const int num_inner = N - 1;

    // Shared: ti[3*(N-1)] + tiStack[256] + cost_matrix[STATES*STATES]
    extern __shared__ char smem[];
    int*          ti         = (int*)smem;
    int*          tiStack    = ti + 3 * num_inner;
    unsigned int* sh_cost    = (unsigned int*)(tiStack + 256);

    // Load cost matrix into shared memory (all 32 lanes cooperate)
    const int cm_size = STATES * STATES;
    for (int i = lane; i < cm_size; i += 32)
        sh_cost[i] = d_cost_matrix[i];
    __syncwarp();

    // Init pars_ptn to 0
    uint16_t* pars_ptn = d_out + (size_t)t * nptn_padded;
    for (int i = lane; i < nptn_padded; i += 32)
        pars_ptn[i] = 0;
    __syncwarp();

    const int* bvf = d_treelsBackVf + (size_t)t * kMaxVFaces;

    // Build traversal (lane 0) — same DFS as Fitch kernel
    int tiSize = 0;
    if (lane == 0)
    {
        int top = -1, tis = 0;
        tiStack[++top] = bvf[start_vf];
        while (top >= 0)
        {
            int p_vf = tiStack[top--];
            if (p_vf < N) continue;
            int pNext = vfNextFace(p_vf, N);
            int pNnxt = vfNnxtFace(p_vf, N);
            int q_vf  = bvf[pNext];
            int r_vf  = bvf[pNnxt];
            if (r_vf >= N) tiStack[++top] = r_vf;
            if (q_vf >= N) tiStack[++top] = q_vf;
            ti[tis++] = vfToNum(p_vf, N);
            ti[tis++] = vfToNum(q_vf, N);
            ti[tis++] = vfToNum(r_vf, N);
        }
        tiSize = tis;
    }
    __syncwarp();
    tiSize = __shfl_sync(0xFFFFFFFFu, tiSize, 0);
    const int n_inner = tiSize / 3;

    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;

    // ── Sankoff newview (post-order) ─────────────────────────────────────────
    // p_cost[s][b] = min_j(q_cost[j][b] + cost[s][j]) + min_j(r_cost[j][b] + cost[s][j])
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
            #pragma unroll
            for (int s = 0; s < STATES; s++)
            {
                unsigned int best_q = kSankoffInf, best_r = kSankoffInf;
                #pragma unroll
                for (int u = 0; u < STATES; u++)
                {
                    unsigned int c  = sh_cost[s * STATES + u];
                    unsigned int qv = (unsigned int)q_base[(size_t)u * width + b] + c;
                    unsigned int rv = (unsigned int)r_base[(size_t)u * width + b] + c;
                    best_q = min(best_q, qv);
                    best_r = min(best_r, rv);
                }
                p_base[(size_t)s * width + b] = (parsimonyNumber)(best_q + best_r);
            }
        }
    }
    __syncwarp();

    // ── Root edge: pars_ptn[b] = min_ij(tip[i][b] + cost[i][j] + inner[j][b]) ─
    {
        const int tip_num   = vfToNum(start_vf, N);
        const int inner_num = vfToNum(bvf[start_vf], N);

        const parsimonyNumber* t_base = pars_tree + (size_t)tip_num   * width * STATES;
        const parsimonyNumber* i_base = pars_tree + (size_t)inner_num * width * STATES;

        for (int b = lane; b < width; b += 32)
        {
            unsigned int min_pars = kSankoffInf;
            #pragma unroll
            for (int ii = 0; ii < STATES; ii++)
            {
                unsigned int tv = (unsigned int)t_base[(size_t)ii * width + b];
                #pragma unroll
                for (int jj = 0; jj < STATES; jj++)
                {
                    unsigned int c  = sh_cost[ii * STATES + jj];
                    unsigned int iv = (unsigned int)i_base[(size_t)jj * width + b];
                    min_pars = min(min_pars, tv + c + iv);
                }
            }
            // Clamp to uint16_t range
            pars_ptn[b] = (uint16_t)(min_pars > 65535u ? 65535u : min_pars);
        }
    }
    __syncwarp();
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
    const int K      = mem->K;

    const bool is_sankoff = (mem->d_cost_matrix != nullptr);

    if (states != 4 && states != 20) return false;

    // Use K_ppars (>= K) blocks per launch to improve GPU occupancy.
    // d_ppars_parsVect provides scratch for K_ppars concurrent blocks.
    // Batches execute sequentially on the same stream — no data race.
    // Caller's cudaStreamSynchronize waits for all batches to finish.
    const int Kp = mem->K_ppars;
    parsimonyNumber* d_pv = mem->d_ppars_parsVect;

    if (is_sankoff)
    {
        // Sankoff kernel: shared = ti + tiStack + cost_matrix
        const size_t smem_bytes =
            (size_t)(3 * (N - 1) + 256) * sizeof(int)
            + (size_t)(states * states) * sizeof(unsigned int);

        for (int batch_start = 0; batch_start < n_treels; batch_start += Kp)
        {
            const int batch_size = std::min(Kp, n_treels - batch_start);
            if (states == 4)
                treelsPatternParsKernelSankoff<4><<<dim3(batch_size), dim3(32), smem_bytes, stream>>>(
                    batch_start, n_treels, N, width, nptn_padded, start_vf,
                    mem->d_treelsBackVf, d_pv, mem->parsVectPerTree,
                    mem->d_cost_matrix, d_treels_ptn_pars
                );
            else
                treelsPatternParsKernelSankoff<20><<<dim3(batch_size), dim3(32), smem_bytes, stream>>>(
                    batch_start, n_treels, N, width, nptn_padded, start_vf,
                    mem->d_treelsBackVf, d_pv, mem->parsVectPerTree,
                    mem->d_cost_matrix, d_treels_ptn_pars
                );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    else
    {
        // Fitch kernel: shared = ti + tiStack
        const size_t smem_bytes =
            (size_t)(3 * (N - 1) + 256) * sizeof(int);

        for (int batch_start = 0; batch_start < n_treels; batch_start += Kp)
        {
            const int batch_size = std::min(Kp, n_treels - batch_start);

            if (states == 4)
                treelsPatternParsKernel<4><<<dim3(batch_size), dim3(32), smem_bytes, stream>>>(
                    batch_start, n_treels, N, width, nptn, nptn_padded, start_vf,
                    mem->d_treelsBackVf, d_pv, mem->parsVectPerTree,
                    d_treels_ptn_pars
                );
            else
                treelsPatternParsKernel<20><<<dim3(batch_size), dim3(32), smem_bytes, stream>>>(
                    batch_start, n_treels, N, width, nptn, nptn_padded, start_vf,
                    mem->d_treelsBackVf, d_pv, mem->parsVectPerTree,
                    d_treels_ptn_pars
                );

            CUDA_CHECK(cudaGetLastError());
        }
    }

    return true;
}

}  // namespace mpbootgpu
