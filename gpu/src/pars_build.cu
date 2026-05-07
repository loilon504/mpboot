#include <climits>

#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/topo_helpers.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

// GpuTopology-based hookup wrapper (lane 0 only)
__device__ __forceinline__ void gpuHookup(
    GpuTopology* t, int a, int b
)
{
    t->back_vf[a] = b;
    t->back_vf[b] = a;
}

// ─── testInsert ───────────────────────────────────────────────────────────────
// Temporarily insert pruned node p at edge (q, q->back), evaluate full-tree
// parsimony, record if it beats sh.bestParsimony, then undo.
__device__ void testInsert(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    BuildShared& sh,
    int p,
    int q,
    int N,
    int width,
    int states,
    int lane
)
{
    if (lane == 0)
    {
        int r = topo->back_vf[q];
        sh.bcast[0] = r;
        gpuHookup(topo->back_vf, vfNextFace(p, N), q);
        gpuHookup(topo->back_vf, vfNnxtFace(p, N), r);
    }
    __syncwarp();

    createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, p, N, width, states, lane);
    unsigned int mp = createTiAndEvaluateParsimony(
        pars_tree, score_tree, topo, sh, p, N, false, width, states
    );
    __syncwarp();

    if (lane == 0)
    {
        if (mp < sh.bestParsimony)
        {
            sh.bestParsimony = mp;
            sh.bestHits      = 1;
            sh.bestRemoveVf  = p;
            sh.bestInsertVf  = q;
        }
        else if (mp == sh.bestParsimony)
        {
            sh.bestHits++;
            if (gpuRandum(&sh.seed) <= 1.0 / sh.bestHits)
            {
                sh.bestRemoveVf = p;
                sh.bestInsertVf = q;
            }
        }
        const int r = sh.bcast[0];
        gpuHookup(topo->back_vf, q, r);
        topo->back_vf[vfNextFace(p, N)] = -1;
        topo->back_vf[vfNnxtFace(p, N)] = -1;
    }
    __syncwarp();
}

// ─── doAddTraverse ────────────────────────────────────────────────────────────
// Iterative addTraverseParsimony(p, q_start, mintrav, maxtrav).
// sh.stack[] is reused as stackVf; sh.stackMint/stackMaxt hold trav counts.
__device__ void doAddTraverse(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    BuildShared& sh,
    int p,
    int q,
    int mintrav,
    int maxtrav,
    int N,
    int width,
    int states,
    int lane
)
{
    if (lane == 0)
    {
        sh.stack[0]     = q;
        sh.stackMint[0] = mintrav - 1;
        sh.stackMaxt[0] = maxtrav - 1;
        sh.stackTop     = 1;
    }
    __syncwarp();

    while (sh.stackTop > 0)
    {
        if (lane == 0)
        {
            int top      = --sh.stackTop;
            sh.bcast[1]  = sh.stack[top];
            sh.bcast[2]  = sh.stackMint[top];
            sh.bcast[3]  = sh.stackMaxt[top];
        }
        __syncwarp();
        const int cur_q = sh.bcast[1];
        const int mint  = sh.bcast[2];
        const int maxt  = sh.bcast[3];
        const int q_num = vfToNum(cur_q, N);

        if (mint <= 0)
            testInsert(pars_tree, score_tree, topo, sh, p, cur_q, N, width, states, lane);

        if (q_num > N && maxt > 0)
        {
            if (lane == 0)
            {
                int qn_vf  = topo->back_vf[vfNextFace(cur_q, N)];
                int qnn_vf = topo->back_vf[vfNnxtFace(cur_q, N)];
                sh.stack[sh.stackTop]     = qnn_vf;
                sh.stackMint[sh.stackTop] = mint - 1;
                sh.stackMaxt[sh.stackTop] = maxt - 1;
                sh.stackTop++;
                sh.stack[sh.stackTop]     = qn_vf;
                sh.stackMint[sh.stackTop] = mint - 1;
                sh.stackMaxt[sh.stackTop] = maxt - 1;
                sh.stackTop++;
            }
            __syncwarp();
        }
    }
}

// ─── applyMove ────────────────────────────────────────────────────────────────
// removeNodeParsimony(rm) + restoreTreeParsimony(rm, ins) + newview.
__device__ void applyMove(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    BuildShared& sh,
    int rm,
    int ins,
    int N,
    int width,
    int states,
    int lane
)
{
    if (lane == 0)
    {
        int pn  = vfNextFace(rm, N);
        int pnn = vfNnxtFace(rm, N);
        int p1  = topo->back_vf[pn];
        int p2  = topo->back_vf[pnn];
        gpuHookup(topo->back_vf, p1, p2);
        topo->back_vf[pn]  = -1;
        topo->back_vf[pnn] = -1;

        int r = topo->back_vf[ins];
        gpuHookup(topo->back_vf, pn,  ins);
        gpuHookup(topo->back_vf, pnn, r);
    }
    createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, rm, N, width, states, lane);
    __syncwarp();
}

// ─── Kernel ───────────────────────────────────────────────────────────────────
// grid(K)  block(32)   shared = sizeof(BuildShared)
// Phase 0-2: stepwise addition (build tree from scratch)
// Phase 3:   SPR hill-climbing (if sprDist > 0)
__global__ void buildParsimonyTreesKernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    const long* __restrict__ d_seeds,
    int width,
    int states,
    int sprDist,
    size_t parsVectPerTree,
    size_t parsScorePerTree
)
{
    extern __shared__ BuildShared sh_arr[];
    BuildShared& sh = sh_arr[0];

    const int k    = blockIdx.x;
    const int lane = threadIdx.x;

    parsimonyNumber* pars_tree  = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int*    score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    GpuTopology*     topo       = d_topos + k;
    const int        N          = topo->mxtips;

    // ── Phase 0: permutation + 3-tip tree (lane 0) ───────────────────────────
    if (lane == 0)
    {
        sh.seed = d_seeds[k];

        for (int i = 1; i <= N; i++) sh.perm[i] = i;
        for (int i = 1; i <= N; i++)
        {
            int k2      = (int)((double)(N + 1 - i) * gpuRandum(&sh.seed));
            int tmp     = sh.perm[i];
            sh.perm[i]      = sh.perm[i + k2];
            sh.perm[i + k2] = tmp;
        }

        int ip = sh.perm[1], iq = sh.perm[2], ir = sh.perm[3];
        int f0 = N, f1 = N + 1, f2 = N + 2;
        gpuHookup(topo, ip - 1, f1);
        gpuHookup(topo, iq - 1, f0);
        gpuHookup(topo, ir - 1, f2);

        sh.startVf = nodepVf(ip < iq ? (ip < ir ? ip : ir) : (iq < ir ? iq : ir), N);

        score_tree[N + 1] = 0;
        topo->nextnode    = N + 2;
        topo->ntips       = 3;

        sh.bestParsimony = UINT_MAX;
        sh.bestHits      = 1;
    }
    __syncwarp();

    // Initial newview for inner1 from the f2 face (children = tip_ip via f1, tip_iq via f0)
    createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, N + 2, N, width, states, lane);
    __syncwarp();

    // ── Phase 1: stepwise addition for tips 4..N ─────────────────────────────
    for (int nextsp = 4; nextsp <= N; nextsp++)
    {
        if (lane == 0)
        {
            sh.tipnum = sh.perm[nextsp];
            sh.qnum   = topo->nextnode++;
            topo->ntips++;
            sh.qf0 = N + 3 * (sh.qnum - N - 1);
            sh.qf1 = sh.qf0 + 1;
            sh.qf2 = sh.qf0 + 2;

            gpuHookup(topo, sh.tipnum - 1, sh.qf2);

            sh.bestParsimony = UINT_MAX;
            sh.bestHits      = 1;
            sh.insertVf      = -1;

            sh.stack[0] = topo->back_vf[sh.startVf];
            sh.stackTop = 1;

            score_tree[sh.qnum] = 0;
        }
        __syncwarp();

        const int q_f0 = sh.qf0;
        const int q_f1 = sh.qf1;
        const int q_f2 = sh.qf2;

        // ── DFS over candidate insertion edges ────────────────────────────────
        while (sh.stackTop > 0)
        {
            if (lane == 0)
            {
                int node  = sh.stack[--sh.stackTop];
                int child = topo->back_vf[node];
                topo->back_vf[q_f1] = node;
                topo->back_vf[node] = q_f1;
                topo->back_vf[q_f0] = child;
                topo->back_vf[child] = q_f0;
                sh.stack[sh.stackTop]     = node;   // borrow slot for broadcast
                sh.stack[sh.stackTop + 1] = child;
            }
            __syncwarp();

            const int node     = sh.stack[sh.stackTop];
            const int child    = sh.stack[sh.stackTop + 1];
            const int node_num = vfToNum(node, N);

            if (lane == 0)
            {
                sh.tiSize = 3;
                computeTraversalInfoParsimony(topo, sh, q_f2, N, false);
                sh.ti[1] = vfToNum(q_f2, N);
                sh.ti[2] = vfToNum(topo->back_vf[q_f2], N);
            }
            unsigned int mp = newviewParsimony(pars_tree, score_tree, sh, true, width, states);
            __syncwarp();

            if (lane == 0)
            {
                if (mp < sh.bestParsimony)
                {
                    sh.bestParsimony = mp;
                    sh.bestHits      = 1;
                    sh.insertVf      = node;
                }
                else if (mp == sh.bestParsimony)
                {
                    sh.bestHits++;
                }

                topo->back_vf[node]  = child;
                topo->back_vf[child] = node;

                if (node_num > N && score_tree[node_num] > 0)
                {
                    sh.stack[sh.stackTop++] = topo->back_vf[vfNextFace(node, N)];
                    sh.stack[sh.stackTop++] = topo->back_vf[vfNnxtFace(node, N)];
                }
            }
            __syncwarp();
        }

        // ── Commit best insertion ─────────────────────────────────────────────
        if (lane == 0)
        {
            int r_vf = topo->back_vf[sh.insertVf];
            gpuHookup(topo, q_f1, sh.insertVf);
            gpuHookup(topo, q_f0, r_vf);

            sh.tiSize = 3;
            computeTraversalInfoParsimony(topo, sh, q_f2, N, false);
        }
        newviewParsimony(pars_tree, score_tree, sh, false, width, states);
        __syncwarp();
    }

    // ── Phase 2: set start_vface ──────────────────────────────────────────────
    if (lane == 0)
    {
        topo->start_vface   = nodepVf(1, N);
        topo->bestParsimony = sh.bestParsimony;
    }
    __syncwarp();

    // ── Phase 3: SPR hill-climbing ────────────────────────────────────────────
    if (sprDist <= 0) return;

    int* const back_vf = topo->back_vf;

    if (lane == 0)
    {
        sh.randomMPHits         = 1;
        sh.randomMP             = sh.bestParsimony;
        topo->preSprParsimony   = sh.bestParsimony;
    }
    __syncwarp();

    unsigned int startMP;
    do
    {
        startMP = sh.randomMP;

        for (int i = 1; i <= 2 * N - 2; i++)
        {
            if (lane == 0)
            {
                sh.bestParsimony = sh.randomMP;
                sh.bestHits      = 1;
                sh.bestRemoveVf  = -1;
                sh.bestInsertVf  = -1;

                sh.bcast[4] = nodepVf(i, N);
                sh.bcast[5] = back_vf[sh.bcast[4]];
                sh.bcast[6] = vfToNum(sh.bcast[5], N);
            }
            __syncwarp();
            const int p     = sh.bcast[4];
            const int q     = sh.bcast[5];
            const int q_num = sh.bcast[6];

            createTiAndEvaluateParsimony(
                pars_tree, score_tree, topo, sh, p, N, true, width, states
            );

            // ── P-branch ─────────────────────────────────────────────────────
            if (i > N)
            {
                if (lane == 0)
                {
                    sh.bcast[7] = back_vf[vfNextFace(p, N)];
                    sh.bcast[8] = back_vf[vfNnxtFace(p, N)];
                }
                __syncwarp();
                const int p1     = sh.bcast[7];
                const int p2     = sh.bcast[8];
                const int p1_num = vfToNum(p1, N);
                const int p2_num = vfToNum(p2, N);

                if (p1_num > N || p2_num > N)
                {
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, p1, p2);
                        back_vf[vfNextFace(p, N)] = -1;
                        back_vf[vfNnxtFace(p, N)] = -1;
                        sh.tip_p_num = q_num;
                    }
                    __syncwarp();

                    if (p1_num > N)
                    {
                        doAddTraverse(pars_tree, score_tree, topo, sh, p,
                            back_vf[vfNextFace(p1, N)], 1, sprDist, N, width, states, lane);
                        doAddTraverse(pars_tree, score_tree, topo, sh, p,
                            back_vf[vfNnxtFace(p1, N)], 1, sprDist, N, width, states, lane);
                    }
                    if (p2_num > N)
                    {
                        doAddTraverse(pars_tree, score_tree, topo, sh, p,
                            back_vf[vfNextFace(p2, N)], 1, sprDist, N, width, states, lane);
                        doAddTraverse(pars_tree, score_tree, topo, sh, p,
                            back_vf[vfNnxtFace(p2, N)], 1, sprDist, N, width, states, lane);
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(p, N), p1);
                        gpuHookup(back_vf, vfNnxtFace(p, N), p2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony(
                        pars_tree, score_tree, topo, sh, p, N, width, states, lane);
                    __syncwarp();
                }
            }

            // ── Q-branch ─────────────────────────────────────────────────────
            if (q_num > N)
            {
                if (lane == 0)
                {
                    sh.bcast[7] = back_vf[vfNextFace(q, N)];
                    sh.bcast[8] = back_vf[vfNnxtFace(q, N)];
                }
                __syncwarp();
                const int q1     = sh.bcast[7];
                const int q2     = sh.bcast[8];
                const int q1_num = vfToNum(q1, N);
                const int q2_num = vfToNum(q2, N);

                bool q1_has_inner_gc = q1_num > N &&
                    (vfToNum(back_vf[vfNextFace(q1, N)], N) > N ||
                     vfToNum(back_vf[vfNnxtFace(q1, N)], N) > N);
                bool q2_has_inner_gc = q2_num > N &&
                    (vfToNum(back_vf[vfNextFace(q2, N)], N) > N ||
                     vfToNum(back_vf[vfNnxtFace(q2, N)], N) > N);

                if (q1_has_inner_gc || q2_has_inner_gc)
                {
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, q1, q2);
                        back_vf[vfNextFace(q, N)] = -1;
                        back_vf[vfNnxtFace(q, N)] = -1;
                        sh.tip_p_num = i;
                    }
                    __syncwarp();

                    if (q1_num > N)
                    {
                        doAddTraverse(pars_tree, score_tree, topo, sh, q,
                            back_vf[vfNextFace(q1, N)], 2, sprDist, N, width, states, lane);
                        doAddTraverse(pars_tree, score_tree, topo, sh, q,
                            back_vf[vfNnxtFace(q1, N)], 2, sprDist, N, width, states, lane);
                    }
                    if (q2_num > N)
                    {
                        doAddTraverse(pars_tree, score_tree, topo, sh, q,
                            back_vf[vfNextFace(q2, N)], 2, sprDist, N, width, states, lane);
                        doAddTraverse(pars_tree, score_tree, topo, sh, q,
                            back_vf[vfNnxtFace(q2, N)], 2, sprDist, N, width, states, lane);
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(q, N), q1);
                        gpuHookup(back_vf, vfNnxtFace(q, N), q2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony(
                        pars_tree, score_tree, topo, sh, q, N, width, states, lane);
                    __syncwarp();
                }
            }

            // ── Apply best move (if any) ──────────────────────────────────────
            if (lane == 0)
            {
                sh.bcast[9] = -1;
                if (sh.bestRemoveVf >= 0 && sh.bestInsertVf >= 0)
                {
                    bool apply = sh.bestParsimony < sh.randomMP;
                    if (apply)
                    {
                        sh.randomMPHits = 1;
                    }
                    else if (sh.bestParsimony == sh.randomMP)
                    {
                        sh.randomMPHits++;
                        apply = (gpuRandum(&sh.seed) <= 1.0 / sh.randomMPHits);
                    }
                    if (apply)
                    {
                        sh.bcast[9]  = sh.bestRemoveVf;
                        sh.bcast[10] = sh.bestInsertVf;
                    }
                }
            }
            __syncwarp();

            if (sh.bcast[9] >= 0)
            {
                applyMove(pars_tree, score_tree, topo, sh,
                    sh.bcast[9], sh.bcast[10], N, width, states, lane);
                if (lane == 0) sh.randomMP = sh.bestParsimony;
                __syncwarp();
            }
        }

    } while (sh.randomMP < startMP);

    if (lane == 0)
        topo->bestParsimony = sh.randomMP;
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem, const long* seeds, int sprDist, cudaStream_t stream
)
{
    long* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)mem->K * sizeof(long)));
    CUDA_CHECK(cudaMemcpyAsync(
        d_seeds, seeds, (size_t)mem->K * sizeof(long), cudaMemcpyHostToDevice, stream
    ));

    CUDA_CHECK(cudaMemsetAsync(
        mem->d_parsScore, 0,
        (size_t)mem->K * mem->parsScorePerTree * sizeof(unsigned int), stream
    ));

    const size_t sharedBytes = sizeof(BuildShared);
    dim3 grid(mem->K, 1, 1);
    dim3 block(kWarpSize, 1, 1);

    printf(
        "[GPU] buildParsimonyTreesKernel: K=%d  sprDist=%d  shared=%.1f KB\n",
        mem->K, sprDist, sharedBytes / 1024.0
    );

    buildParsimonyTreesKernel<<<grid, block, sharedBytes, stream>>>(
        mem->d_parsVect, mem->d_parsScore, mem->d_topos, d_seeds,
        mem->width, mem->states, sprDist,
        mem->parsVectPerTree, mem->parsScorePerTree
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
