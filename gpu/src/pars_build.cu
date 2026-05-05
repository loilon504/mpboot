
#include <cassert>
#include <climits>
#include <stdexcept>
#include <string>

#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/topo_helpers.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

// back pointer accessors — these read/write topo->back_vf[]
__device__ __forceinline__ int vfBack(const GpuTopology* t, int vf)
{
    return t->back_vf[vf];
}
__device__ __forceinline__ int vfNextBack(const GpuTopology* t, int vf, int N)
{
    return t->back_vf[vfNextFace(vf, N)];
}
__device__ __forceinline__ int vfNnxtBack(const GpuTopology* t, int vf, int N)
{
    return t->back_vf[vfNnxtFace(vf, N)];
}

// GpuTopology-based hookup wrapper (lane 0 only)
__device__ __forceinline__ void gpuHookup(GpuTopology* t, int a, int b)
{
    t->back_vf[a] = b;
    t->back_vf[b] = a;
}

// ─── Shared memory per block ──────────────────────────────────────────────────
// Keeps small, frequently accessed data off global memory.
struct alignas(
    16
) BuildShared
{
    int perm[kMaxTaxa + 2];   // permutation 1..N  (~2.8 KB for N=700)
    int stack[kMaxTaxa * 2];  // DFS stack          (~5.6 KB)
    int ti[kMaxTaxa];         // traversalInfo      (~2.8 KB)
    int stackTop;
    int insertVf;           // best insert vface this round
    unsigned int bestPars;  // best parsimony found
    unsigned int bestHits;  // tie-breaking counter
    int startVf;            // tr->start vface (nodep[min(perm[1..3])])
    // per-iteration scalars broadcast from lane 0 to all lanes via shared memory
    int qnum;           // current inner node number being inserted
    int tipnum;         // current tip number being inserted
    int qf0, qf1, qf2;  // three vfaces of qnum
};

// ─── Kernel ───────────────────────────────────────────────────────────────────
// grid(K)  block(32)   shared = sizeof(BuildShared)
__global__ void buildParsimonyTreesKernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    const long* __restrict__ d_seeds,
    int width,
    int states,
    size_t parsVectPerTree,
    size_t parsScorePerTree
)
{
    extern __shared__ BuildShared sh_arr[];
    BuildShared& sh = sh_arr[0];

    const int k = blockIdx.x;
    const int lane = threadIdx.x;  // 0..31

    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    GpuTopology* topo = d_topos + k;
    const int N = topo->mxtips;

    // ── Phase 0: permutation + initial 3-tip tree (lane 0) ───────────────────
    if (lane == 0)
    {
        long seed = d_seeds[k];

        // Fisher-Yates, replicates makePermutationFast
        for (int i = 1; i <= N; i++)
        {
            sh.perm[i] = i;
        }
        for (int i = 1; i <= N; i++)
        {
            double d = gpuRandum(&seed);
            int k2 = (int)((double)(N + 1 - i) * d);
            int tmp = sh.perm[i];
            sh.perm[i] = sh.perm[i + k2];
            sh.perm[i + k2] = tmp;
        }

        // buildSimpleTree(perm[1], perm[2], perm[3]):
        //   hookupDefault(nodep[ip], nodep[iq])          ip<->iq
        //   inner1 = nodep[N+1], hookupDefault(tip_ir, inner1.face[2])
        //   insertParsimony(inner1, nodep[ip]):
        //     r = ip->back = iq
        //     hookupDefault(inner1.face[1], ip)
        //     hookupDefault(inner1.face[0], r=iq)
        int ip = sh.perm[1], iq = sh.perm[2], ir = sh.perm[3];

        int tip_ip = ip - 1;  // vface of tip ip
        int tip_iq = iq - 1;
        int tip_ir = ir - 1;
        // inner1 vfaces: face[0]=N, face[1]=N+1, face[2]=N+2
        int f0 = N, f1 = N + 1, f2 = N + 2;

        // After insertParsimony:
        gpuHookup(topo, tip_ip, f1);  // tip_ip <-> inner1.face[1]
        gpuHookup(topo, tip_iq, f0);  // tip_iq <-> inner1.face[0]
        gpuHookup(topo, tip_ir, f2);  // tip_ir <-> inner1.face[2]

        // tr->start = nodep[min(ip,iq,ir)]
        int startNum = ip < iq ? (ip < ir ? ip : ir) : (iq < ir ? iq : ir);
        sh.startVf = nodepVf(startNum, N);

        // Zero score for the initial inner node (tips are already 0)
        score_tree[N + 1] = 0;

        topo->nextnode = N + 2;  // next inner node number to allocate
        topo->ntips = 3;

        sh.bestPars = UINT_MAX;
        sh.bestHits = 1;
    }
    __syncwarp();

    // ── Initial newview: compute parsVect[N+1] from tips ip, iq ─────────────
    // inner1 (num=N+1): children are tip ip and tip iq
    {
        unsigned int partial = warpNewviewStep(
            pars_tree, N + 1, sh.perm[1], sh.perm[2], width, states
        );
        partial = warpReduceU32(partial);
        if (lane == 0)
        {
            score_tree[N + 1] = partial;  // score_tree[ip]=score_tree[iq]=0 for tips
        }
    }
    __syncwarp();

    // ── Phase 1: stepwise addition for tips 4..N ─────────────────────────────
    for (int nextsp = 4; nextsp <= N; nextsp++)
    {
        // Lane 0: allocate inner node q, link tip -> q.face[2],
        // store iteration scalars in shared memory for broadcast to all lanes.
        if (lane == 0)
        {
            sh.tipnum = sh.perm[nextsp];
            sh.qnum = topo->nextnode++;
            topo->ntips++;
            sh.qf0 = N + 3 * (sh.qnum - N - 1);
            sh.qf1 = sh.qf0 + 1;
            sh.qf2 = sh.qf0 + 2;

            // Link tip <-> q.face[2]
            gpuHookup(topo, sh.tipnum - 1, sh.qf2);

            sh.bestPars = UINT_MAX;
            sh.bestHits = 1;
            sh.insertVf = -1;

            // DFS stack: start with f->back (f = tr->start)
            sh.stack[0] = topo->back_vf[sh.startVf];
            sh.stackTop = 1;

            score_tree[sh.qnum] = 0;
        }
        __syncwarp();

        // All lanes read iteration scalars from shared memory (visible after syncwarp).
        const int q_num = sh.qnum;
        const int tip_num = sh.tipnum;
        const int q_f0 = sh.qf0;
        const int q_f1 = sh.qf1;
        const int q_f2 = sh.qf2;

        // ── DFS over candidate insertion edges ────────────────────────────────
        while (sh.stackTop > 0)
        {
            // Lane 0: pop q_edge_vf from stack, store in shared for broadcast.
            if (lane == 0)
            {
                int node = sh.stack[--sh.stackTop];
                int child = topo->back_vf[node];
                // Temporarily hook new inner node into edge (node, child)
                topo->back_vf[q_f1] = node;
                topo->back_vf[node] = q_f1;
                topo->back_vf[q_f0] = child;
                topo->back_vf[child] = q_f0;
                // Stash for lane use
                sh.stack[sh.stackTop] = node;  // borrow slot (stackTop not incremented)
                sh.stack[sh.stackTop + 1] = child;
            }
            __syncwarp();

            const int node = sh.stack[sh.stackTop];
            const int child = sh.stack[sh.stackTop + 1];
            const int node_num = vfToNum(node, N);
            const int child_num = vfToNum(child, N);

            // Compute parsVect[q_num] and temporarily set score.
            // Must include subtree scores of both edge endpoints, matching CPU:
            //   parsimonyScore[p] = cross(q_edge,r_edge) + parsimonyScore[q_edge] + parsimonyScore[r_edge]
            unsigned int partial = warpNewviewStep(
                pars_tree, q_num, node_num, child_num, width, states
            );
            partial = warpReduceU32(partial);
            if (lane == 0)
            {
                score_tree[q_num] = partial + score_tree[node_num] + score_tree[child_num];
            }
            __syncwarp();

            unsigned int mp = warpEvaluateScore(
                pars_tree, score_tree, q_num, tip_num, width, states
            );

            if (lane == 0)
            {
                printf("[DBG] score in stepwiseAddtion: %d\n", mp);
                if (mp < sh.bestPars)
                {
                    sh.bestPars = mp;
                    sh.bestHits = 1;
                    sh.insertVf = node;
                }
                else if (mp == sh.bestPars)
                {
                    sh.bestHits++;
                    // keep first best (no randum() tie-breaking for GPU build)
                }

                // Undo temporary hookup
                topo->back_vf[node] = child;
                topo->back_vf[child] = node;
                topo->back_vf[q_f1] = -1;
                topo->back_vf[q_f0] = -1;
                score_tree[q_num] = 0;

                // Push children if q_edge is an inner node that's been scored
                int eq_num = vfToNum(node, N);
                if (eq_num > N && score_tree[eq_num] > 0)
                {
                    int nxt = vfNextFace(node, N);
                    int nnxt = vfNnxtFace(node, N);
                    sh.stack[sh.stackTop++] = topo->back_vf[nxt];
                    sh.stack[sh.stackTop++] = topo->back_vf[nnxt];
                }
            }
            __syncwarp();
        }  // end DFS

        // ── Commit best insertion ─────────────────────────────────────────────
        int ins_num, r_num;
        if (lane == 0)
        {
            int ins_vf = sh.insertVf;
            int r_vf = topo->back_vf[ins_vf];
            gpuHookup(topo, q_f1, ins_vf);
            gpuHookup(topo, q_f0, r_vf);
            // store node numbers in shared for all-lane parsVect computation
            sh.stack[0] = vfToNum(ins_vf, N);
            sh.stack[1] = vfToNum(r_vf, N);
        }
        __syncwarp();
        ins_num = sh.stack[0];
        r_num = sh.stack[1];

        unsigned int partial = warpNewviewStep(pars_tree, q_num, ins_num, r_num, width, states);
        partial = warpReduceU32(partial);
        if (lane == 0)
        {
            score_tree[q_num] = partial + score_tree[ins_num] + score_tree[r_num];
        }
        __syncwarp();
    }  // end stepwise addition loop

    // ── Phase 2: nodeRectifier — set start_vface to nodep[1] ─────────────────
    if (lane == 0)
    {
        topo->start_vface = nodepVf(1, N);  // tr->start = nodep[1]
        topo->bestParsimony = sh.bestPars;
        // printf("[DBG] Pre-SPR score: %d\n", topo->bestParsimony);
    }
    __syncwarp();
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long* seeds,  // host array [K]
    cudaStream_t stream
)
{
    // Upload seeds to device
    long* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)mem->K * sizeof(long)));
    CUDA_CHECK(cudaMemcpyAsync(
        d_seeds, seeds, (size_t)mem->K * sizeof(long), cudaMemcpyHostToDevice, stream
    ));

    // Zero parsScore before building (tip scores are already 0)
    CUDA_CHECK(cudaMemsetAsync(
        mem->d_parsScore, 0, (size_t)mem->K * mem->parsScorePerTree * sizeof(unsigned int), stream
    ));

    const size_t sharedBytes = sizeof(BuildShared);
    dim3 grid(mem->K, 1, 1);
    dim3 block(kWarpSize, 1, 1);

    printf("[GPU] buildParsimonyTreesKernel: K=%d  shared=%.1f KB\n", mem->K, sharedBytes / 1024.0);

    buildParsimonyTreesKernel<<<grid, block, sharedBytes, stream>>>(
        mem->d_parsVect, mem->d_parsScore, mem->d_topos, d_seeds, mem->width, mem->states,
        mem->parsVectPerTree, mem->parsScorePerTree
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
