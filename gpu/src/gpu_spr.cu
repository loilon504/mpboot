#include <cuda_runtime.h>

#include <climits>

#include "gpu/include/gpu_spr.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/topo_helpers.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

// ─── Shared memory layout ─────────────────────────────────────────────────────
// kMaxSprStack must be ≥ max(sprDist-based traversal, tree height for recompute).
// SPR DFS needs at most 2^sprDist ≈ 64 entries; recompute post-order DFS needs
// at most tree_height entries (≤ N-1 ≤ kMaxTaxa-1 for caterpillar trees).
static constexpr int kMaxSprStack = kMaxTaxa;  // 700, covers all cases

struct alignas(
    16
) SprShared
{
    int stkVf[kMaxSprStack];  // vface IDs (SPR DFS + recompute DFS)
    int stkMt[kMaxSprStack];  // mintrav (SPR) / child_index 0-2 (recompute)
    int stkMa[kMaxSprStack];  // maxtrav (SPR, max 64 entries used)
    int stkTop;

    // Global best parsimony for this tree (updated after each applied move)
    unsigned int randomMP;
    unsigned int randomMPHits;  // tie-breaking counter across iterations

    // Best move found during one rearrangeParsimony(p) call (both branches)
    unsigned int bestParsimony;
    unsigned int bestHits;  // tie-breaking within the current rearrangement
    int bestRemoveVf;
    int bestInsertVf;

    // "back" number of the pruned node used in testInsert evaluation
    int tip_p_num;

    // Per-block RNG seed (lane 0 only)
    long seed;

    // Broadcast slot: lane 0 writes, __syncwarp() ensures visibility to all
    int bcast[16];
};

// ─── testInsertParsimony ──────────────────────────────────────────────────────
// Temporarily inserts pruned node p at edge (q_edge_vf, q_edge_vf->back),
// evaluates total tree parsimony at edge (p_num, sh.tip_p_num), and records
// the move if it beats sh.bestParsimony.
//
// Invariant: parsVect and score_tree are valid for ALL nodes except p_num.
// After this call, parsVect[p_num] holds the last computed value (stale for
// p's original position, but overwritten again on the next call or restore).
__device__ void testInsert(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    const int* __restrict__ back_vf,
    SprShared& sh,
    int p_vf,
    int q_edge_vf,
    int N,
    int width,
    int states,
    int lane
)
{
    // Read edge endpoints (lane 0, then broadcast)
    if (lane == 0)
    {
        int r_vf = back_vf[q_edge_vf];
        sh.bcast[0] = vfToNum(q_edge_vf, N);
        sh.bcast[1] = vfToNum(r_vf, N);
        sh.bcast[2] = vfToNum(p_vf, N);
    }
    __syncwarp();
    const int q_edge_num = sh.bcast[0];
    const int r_num = sh.bcast[1];
    const int p_num = sh.bcast[2];

    // Compute parsVect[p_num] from (TIP, q_edge).
    // This matches CPU: after insertParsimony(p, q_cand), p's canonical face fA has
    // fA->back=TIP (unchanged) and fA->next->back=q_cand; evaluateParsimony uses
    // fA->nnxt (= face that connects to r) from children (TIP via fA, q_cand via fA->next).
    unsigned int partial = warpNewviewStep(
        pars_tree, p_num, sh.tip_p_num, q_edge_num, width, states
    );
    partial = warpReduceU32(partial);
    if (lane == 0)
    {
        score_tree[p_num] = partial + score_tree[sh.tip_p_num] + score_tree[q_edge_num];
    }
    __syncwarp();

    // Evaluate full-tree parsimony at edge (p_num, r_num).
    // mp = cross(p, r) + score[p] + score[r] = full tree parsimony.
    unsigned int mp = warpEvaluateScore(pars_tree, score_tree, p_num, r_num, width, states);

    // Lane 0 checks and records improvement
    if (lane == 0)
    {
        if (mp < sh.bestParsimony)
        {
            sh.bestParsimony = mp;
            sh.bestHits = 1;
            sh.bestRemoveVf = p_vf;
            sh.bestInsertVf = q_edge_vf;
        }
        else if (mp == sh.bestParsimony)
        {
            sh.bestHits++;
            if (gpuRandum(&sh.seed) <= 1.0 / sh.bestHits)
            {
                sh.bestRemoveVf = p_vf;
                sh.bestInsertVf = q_edge_vf;
            }
        }
    }
    __syncwarp();
}

// ─── doAddTraverse ────────────────────────────────────────────────────────────
// Iterative equivalent of addTraverseParsimony(p, q_start, mintrav, maxtrav).
// On entry: sh.stkTop must be 0 (caller's responsibility).
// Pushes (q_start_vf, mintrav-1, maxtrav-1) then loops until stack empty.
__device__ void doAddTraverse(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    const int* __restrict__ back_vf,
    SprShared& sh,
    int p_vf,
    int q_start_vf,
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
        sh.stkTop = 0;
        sh.stkVf[0] = q_start_vf;
        sh.stkMt[0] = mintrav - 1;  // pre-decrement mirrors CPU's --mintrav
        sh.stkMa[0] = maxtrav - 1;  // pre-decrement mirrors CPU's --maxtrav
        sh.stkTop = 1;
    }
    __syncwarp();

    while (sh.stkTop > 0)
    {
        // Pop one entry (lane 0) and broadcast
        if (lane == 0)
        {
            int top = --sh.stkTop;
            sh.bcast[3] = sh.stkVf[top];
            sh.bcast[4] = sh.stkMt[top];
            sh.bcast[5] = sh.stkMa[top];
        }
        __syncwarp();
        const int q_vf = sh.bcast[3];
        const int mt = sh.bcast[4];
        const int ma = sh.bcast[5];
        const int q_num = vfToNum(q_vf, N);

        // CPU: if (--mintrav <= 0) testInsert — only at inner nodes (PLL: q->number > mxtips)
        if (mt <= 0 && q_num > N)
        {
            testInsert(pars_tree, score_tree, back_vf, sh, p_vf, q_vf, N, width, states, lane);
        }

        // CPU: if (q is inner && --maxtrav > 0) recurse to q->next->back, q->nnxt->back
        if (q_num > N && ma > 0)
        {
            if (lane == 0)
            {
                int qn_vf = back_vf[vfNextFace(q_vf, N)];
                int qnn_vf = back_vf[vfNnxtFace(q_vf, N)];
                // Push right first so left is processed first (LIFO)
                sh.stkVf[sh.stkTop] = qnn_vf;
                sh.stkMt[sh.stkTop] = mt - 1;
                sh.stkMa[sh.stkTop] = ma - 1;
                sh.stkTop++;
                sh.stkVf[sh.stkTop] = qn_vf;
                sh.stkMt[sh.stkTop] = mt - 1;
                sh.stkMa[sh.stkTop] = ma - 1;
                sh.stkTop++;
            }
            __syncwarp();
        }
    }
}

// ─── applyMove ────────────────────────────────────────────────────────────────
// removeNodeParsimony(rm_vf) + restoreTreeParsimony(rm_vf, ins_vf) + newview.
__device__ void applyMove(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    int* __restrict__ back_vf,
    SprShared& sh,
    int rm_vf,
    int ins_vf,
    int N,
    int width,
    int states,
    int lane
)
{
    if (lane == 0)
    {
        // removeNodeParsimony(rm_vf)
        int pn_vf = vfNextFace(rm_vf, N);
        int pnn_vf = vfNnxtFace(rm_vf, N);
        int p1_vf = back_vf[pn_vf];
        int p2_vf = back_vf[pnn_vf];
        gpuHookup(back_vf, p1_vf, p2_vf);
        back_vf[pn_vf] = -1;
        back_vf[pnn_vf] = -1;

        // restoreTreeParsimony(rm_vf, ins_vf)
        int r_vf = back_vf[ins_vf];
        gpuHookup(back_vf, pn_vf, ins_vf);
        gpuHookup(back_vf, pnn_vf, r_vf);

        int rm_num = vfToNum(rm_vf, N);
        int ins_num = vfToNum(ins_vf, N);
        int r_num = vfToNum(r_vf, N);
        sh.bcast[6] = rm_num;
        sh.bcast[7] = ins_num;
        sh.bcast[8] = r_num;
    }
    __syncwarp();
    const int rm_num = sh.bcast[6];
    const int ins_num = sh.bcast[7];
    const int r_num = sh.bcast[8];

    // newviewParsimony(rm): recompute parsVect[rm_num] from ins and r
    unsigned int partial = warpNewviewStep(pars_tree, rm_num, ins_num, r_num, width, states);
    partial = warpReduceU32(partial);
    if (lane == 0)
    {
        score_tree[rm_num] = partial + score_tree[ins_num] + score_tree[r_num];
    }
    __syncwarp();
}

// ─── recomputeAllNodes ───────────────────────────────────────────────────────
// Iterative post-order DFS from start_vf's back.  Recomputes parsVect and
// score_tree for every inner node using the correct direction, then returns the
// full-tree parsimony at the (start_vf, start_vf->back) branch.
// Call before SPR so sh.randomMP is initialised to a correct full-tree value.
__device__ unsigned int recomputeAllNodes(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    const int* __restrict__ back_vf,
    SprShared& sh,
    int start_vf,
    int N,
    int width,
    int states,
    int lane
)
{
    // Lane 0 drives the iterative post-order DFS.
    // stkVf[i] = vface currently on stack; stkMt[i] = child_index (0/1/2).
    if (lane == 0)
    {
        sh.stkVf[0] = back_vf[start_vf];  // root: the inner node attached to tip start
        sh.stkMt[0] = 0;
        sh.stkTop = 1;
        if (blockIdx.x == 0)
        {
            sh.bcast[14] = 0;  // inner-node counter for debug
        }
    }
    __syncwarp();

    while (sh.stkTop > 0)
    {
        // Peek top (lane 0) and broadcast
        if (lane == 0)
        {
            int t = sh.stkTop - 1;
            sh.bcast[0] = sh.stkVf[t];
            sh.bcast[1] = sh.stkMt[t];
        }
        __syncwarp();
        const int top_vf = sh.bcast[0];
        const int top_idx = sh.bcast[1];
        const int top_num = vfToNum(top_vf, N);

        if (top_num <= N)
        {
            // Tip: pop immediately, no computation needed
            if (lane == 0)
            {
                sh.stkTop--;
            }
            __syncwarp();
            continue;
        }

        // Children of top_vf in the rooted sense (away from its "parent" back)
        const int c1_vf = back_vf[vfNextFace(top_vf, N)];
        const int c2_vf = back_vf[vfNnxtFace(top_vf, N)];

        if (top_idx == 0)
        {
            // First visit: push child1
            if (lane == 0)
            {
                sh.stkMt[sh.stkTop - 1] = 1;
                sh.stkVf[sh.stkTop] = c1_vf;
                sh.stkMt[sh.stkTop] = 0;
                sh.stkTop++;
            }
            __syncwarp();
        }
        else if (top_idx == 1)
        {
            // Child1 done: push child2
            if (lane == 0)
            {
                sh.stkMt[sh.stkTop - 1] = 2;
                sh.stkVf[sh.stkTop] = c2_vf;
                sh.stkMt[sh.stkTop] = 0;
                sh.stkTop++;
            }
            __syncwarp();
        }
        else
        {
            // Both children done: recompute parsVect and score for this node
            const int c1_num = vfToNum(c1_vf, N);
            const int c2_num = vfToNum(c2_vf, N);
            unsigned int partial = warpNewviewStep(
                pars_tree, top_num, c1_num, c2_num, width, states
            );
            partial = warpReduceU32(partial);
            if (lane == 0)
            {
                score_tree[top_num] = partial + score_tree[c1_num] + score_tree[c2_num];
                if (blockIdx.x == 0)
                {
                    sh.bcast[14]++;
                    // printf("[DFS k=0] #%d top=%d c1=%d(sc=%u) c2=%d(sc=%u) partial=%u ->
                    // sc[top]=%u\n", sh.bcast[14], top_num, c1_num, score_tree[c1_num], c2_num,
                    // score_tree[c2_num], partial, score_tree[top_num]);
                }
                sh.stkTop--;
            }
            __syncwarp();
        }
    }

    // Full-tree parsimony at the (start_vf, root) branch
    const int start_num = vfToNum(start_vf, N);
    const int root_num = vfToNum(back_vf[start_vf], N);
    if (lane == 0 && blockIdx.x == 0)
    {
        // printf("[DFS k=0] DFS done: inner_nodes_processed=%d (expected N-1=%d)
        // score_tree[root=%d]=%u\n", sh.bcast[14], N - 1, root_num, score_tree[root_num]);
        __syncwarp();
    }
    return warpEvaluateScore(pars_tree, score_tree, start_num, root_num, width, states);
}

// ─── GPU SPR kernel ───────────────────────────────────────────────────────────
// grid(K)  block(32)  shared = sizeof(SprShared)
// One block = one tree; one warp = topology (lane 0) + parsimony (all lanes).
__global__ void gpuSprKernel(
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
    extern __shared__ SprShared sh_arr[];
    SprShared& sh = sh_arr[0];

    const int k = blockIdx.x;
    const int lane = threadIdx.x;

    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    GpuTopology* topo = d_topos + k;
    int* const back_vf = topo->back_vf;
    const int N = topo->mxtips;

    if (lane == 0)
    {
        sh.randomMPHits = 1;
        sh.seed = d_seeds[k];

        // Debug: print node 589's neighbors for k==0
        if (k == 0)
        {
            int miss_num = 2 * N - 1;                  // 589 for N=295
            int base_vf = N + 3 * (miss_num - N - 1);  // vfaces of node 589
            printf(
                "[TOPO k=0] node_%d faces: vf[0]=%d vf[1]=%d vf[2]=%d\n", miss_num, base_vf,
                base_vf + 1, base_vf + 2
            );
            printf(
                "[TOPO k=0]   back_vf[%d]=%d (num=%d)\n", base_vf, back_vf[base_vf],
                vfToNum(back_vf[base_vf], N)
            );
            printf(
                "[TOPO k=0]   back_vf[%d]=%d (num=%d)\n", base_vf + 1, back_vf[base_vf + 1],
                vfToNum(back_vf[base_vf + 1], N)
            );
            printf(
                "[TOPO k=0]   back_vf[%d]=%d (num=%d)\n", base_vf + 2, back_vf[base_vf + 2],
                vfToNum(back_vf[base_vf + 2], N)
            );

            // Also show start_vface's back and its back-back
            int sv = topo->start_vface;
            int rv = back_vf[sv];
            int rv2 = back_vf[vfNextFace(rv, N)];
            int rv3 = back_vf[vfNnxtFace(rv, N)];
            printf(
                "[TOPO k=0] start_vf=%d -> root_vf=%d (num=%d) -> children vf=%d(num=%d) "
                "vf=%d(num=%d)\n",
                sv, rv, vfToNum(rv, N), rv2, vfToNum(rv2, N), rv3, vfToNum(rv3, N)
            );
        }
    }
    __syncwarp();

    // Initial full recompute: ensures parsVect/score_tree consistent with current
    // topology and sets sh.randomMP to the correct full-tree parsimony.
    {
        unsigned int fullMP = recomputeAllNodes(
            pars_tree, score_tree, back_vf, sh, topo->start_vface, N, width, states, lane
        );
        if (lane == 0)
        {
            sh.randomMP = fullMP;
            topo->preSprParsimony = fullMP;  // expose true pre-SPR parsimony for debug
            // Debug for tree 0 only
            if (k == 0)
            {
                int start_num = vfToNum(topo->start_vface, N);
                int root_num = vfToNum(back_vf[topo->start_vface], N);
                printf(
                    "[DBG k=0] start_vf=%d start_num=%d root_num=%d score_tree[root]=%u "
                    "fullMP=%u\n",
                    topo->start_vface, start_num, root_num, score_tree[root_num], fullMP
                );
            }
        }
        __syncwarp();
    }

    // ── Outer do-while: repeat until no improvement ───────────────────────────
    unsigned int startMP;
    do
    {
        startMP = sh.randomMP;
        __syncwarp();

        // ── Inner loop: rearrangeParsimony for each node i = 1..2N-2 ─────────
        for (int i = 1; i <= 2 * N - 2; i++)
        {
            // Initialise per-rearrangement best-move state (lane 0, broadcast)
            if (lane == 0)
            {
                sh.bestParsimony = sh.randomMP;
                sh.bestHits = 1;
                sh.bestRemoveVf = -1;
                sh.bestInsertVf = -1;

                int p_vf = nodepVf(i, N);
                int q_vf = back_vf[p_vf];  // p->back
                sh.bcast[9] = p_vf;
                sh.bcast[10] = q_vf;
                sh.bcast[11] = vfToNum(q_vf, N);  // q_num
            }
            __syncwarp();
            const int p_vf = sh.bcast[9];
            const int q_vf = sh.bcast[10];
            const int q_num = sh.bcast[11];
            // p_num = i (by construction of nodepVf)

            // ── P-branch: p must be an inner node ────────────────────────────
            if (i > N)
            {
                // p1 = p->next->back, p2 = p->next->next->back
                if (lane == 0)
                {
                    sh.bcast[12] = back_vf[vfNextFace(p_vf, N)];
                    sh.bcast[13] = back_vf[vfNnxtFace(p_vf, N)];
                }
                __syncwarp();
                const int p1_vf = sh.bcast[12];
                const int p2_vf = sh.bcast[13];
                const int p1_num = vfToNum(p1_vf, N);
                const int p2_num = vfToNum(p2_vf, N);

                if (p1_num > N || p2_num > N)
                {
                    // Replicate CPU line-2291: refresh score_tree[q_num] (= tip_p) from
                    // q's two children excluding p. recomputeAllNodes may have computed
                    // score_tree[q_num] from the DFS direction, which includes p's subtree
                    // if q is an ancestor of p in that DFS — causing testInsert to
                    // underestimate mp and accept bad moves.
                    if (q_num > N)
                    {
                        if (lane == 0)
                        {
                            sh.bcast[12] = vfToNum(back_vf[vfNextFace(q_vf, N)], N);
                            sh.bcast[13] = vfToNum(back_vf[vfNnxtFace(q_vf, N)], N);
                        }
                        __syncwarp();
                        const int qa_num = sh.bcast[12];
                        const int qb_num = sh.bcast[13];
                        unsigned int pq = warpNewviewStep(
                            pars_tree, q_num, qa_num, qb_num, width, states
                        );
                        pq = warpReduceU32(pq);
                        if (lane == 0)
                        {
                            score_tree[q_num] = pq + score_tree[qa_num] + score_tree[qb_num];
                        }
                        __syncwarp();
                    }

                    // removeNodeParsimony(p)
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, p1_vf, p2_vf);
                        back_vf[vfNextFace(p_vf, N)] = -1;
                        back_vf[vfNnxtFace(p_vf, N)] = -1;
                        sh.tip_p_num = q_num;  // p->back (unchanged)
                    }
                    __syncwarp();

                    // addTraverse into p1's subtree if p1 is inner
                    if (p1_num > N)
                    {
                        int qp1n_vf = back_vf[vfNextFace(p1_vf, N)];
                        int qp1nn_vf = back_vf[vfNnxtFace(p1_vf, N)];
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, p_vf, qp1n_vf, 1, sprDist, N, width,
                            states, lane
                        );
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, p_vf, qp1nn_vf, 1, sprDist, N,
                            width, states, lane
                        );
                    }
                    // addTraverse into p2's subtree if p2 is inner
                    if (p2_num > N)
                    {
                        int qp2n_vf = back_vf[vfNextFace(p2_vf, N)];
                        int qp2nn_vf = back_vf[vfNnxtFace(p2_vf, N)];
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, p_vf, qp2n_vf, 1, sprDist, N, width,
                            states, lane
                        );
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, p_vf, qp2nn_vf, 1, sprDist, N,
                            width, states, lane
                        );
                    }

                    // Restore p to original position
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(p_vf, N), p1_vf);
                        gpuHookup(back_vf, vfNnxtFace(p_vf, N), p2_vf);
                    }
                    __syncwarp();

                    // newviewParsimony(p): recompute parsVect[i] from p1, p2
                    unsigned int partial = warpNewviewStep(
                        pars_tree, i, p1_num, p2_num, width, states
                    );
                    partial = warpReduceU32(partial);
                    if (lane == 0)
                    {
                        score_tree[i] = partial + score_tree[p1_num] + score_tree[p2_num];
                    }
                    __syncwarp();
                }
            }

            // ── Q-branch: q must be inner and sprDist > 0 ────────────────────
            if (q_num > N && sprDist > 0)
            {
                // q1 = q->next->back, q2 = q->next->next->back
                // (using q_vf = p->back, which is the specific face of q)
                if (lane == 0)
                {
                    int q1_vf = back_vf[vfNextFace(q_vf, N)];
                    int q2_vf = back_vf[vfNnxtFace(q_vf, N)];
                    sh.bcast[12] = q1_vf;
                    sh.bcast[13] = q2_vf;
                }
                __syncwarp();
                const int q1_vf = sh.bcast[12];
                const int q2_vf = sh.bcast[13];
                const int q1_num = vfToNum(q1_vf, N);
                const int q2_num = vfToNum(q2_vf, N);

                // CPU condition: at least one qX is inner with ≥1 inner grandchild
                bool q1_has_inner_gc = false;
                bool q2_has_inner_gc = false;
                if (q1_num > N)
                {
                    int gca = vfToNum(back_vf[vfNextFace(q1_vf, N)], N);
                    int gcb = vfToNum(back_vf[vfNnxtFace(q1_vf, N)], N);
                    q1_has_inner_gc = (gca > N || gcb > N);
                }
                if (q2_num > N)
                {
                    int gca = vfToNum(back_vf[vfNextFace(q2_vf, N)], N);
                    int gcb = vfToNum(back_vf[vfNnxtFace(q2_vf, N)], N);
                    q2_has_inner_gc = (gca > N || gcb > N);
                }

                if (q1_has_inner_gc || q2_has_inner_gc)
                {
                    // Replicate CPU line-2291 for Q-branch: refresh score_tree[i] (= tip_q)
                    // from p's two children (p1, p2), excluding the edge toward q.
                    if (i > N)
                    {
                        if (lane == 0)
                        {
                            sh.bcast[12] = vfToNum(back_vf[vfNextFace(p_vf, N)], N);
                            sh.bcast[13] = vfToNum(back_vf[vfNnxtFace(p_vf, N)], N);
                        }
                        __syncwarp();
                        const int pi_c1 = sh.bcast[12];
                        const int pi_c2 = sh.bcast[13];
                        unsigned int pp = warpNewviewStep(
                            pars_tree, i, pi_c1, pi_c2, width, states
                        );
                        pp = warpReduceU32(pp);
                        if (lane == 0)
                        {
                            score_tree[i] = pp + score_tree[pi_c1] + score_tree[pi_c2];
                        }
                        __syncwarp();
                    }

                    // removeNodeParsimony(q) using q_vf perspective
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, q1_vf, q2_vf);
                        back_vf[vfNextFace(q_vf, N)] = -1;
                        back_vf[vfNnxtFace(q_vf, N)] = -1;
                        sh.tip_p_num = i;  // q->back = p (node i, unchanged)
                    }
                    __syncwarp();

                    // addTraverse with mintrav=2 (Q-branch)
                    if (q1_num > N)
                    {
                        int qq1n_vf = back_vf[vfNextFace(q1_vf, N)];
                        int qq1nn_vf = back_vf[vfNnxtFace(q1_vf, N)];
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, q_vf, qq1n_vf, 2, sprDist, N, width,
                            states, lane
                        );
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, q_vf, qq1nn_vf, 2, sprDist, N,
                            width, states, lane
                        );
                    }
                    if (q2_num > N)
                    {
                        int qq2n_vf = back_vf[vfNextFace(q2_vf, N)];
                        int qq2nn_vf = back_vf[vfNnxtFace(q2_vf, N)];
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, q_vf, qq2n_vf, 2, sprDist, N, width,
                            states, lane
                        );
                        doAddTraverse(
                            pars_tree, score_tree, back_vf, sh, q_vf, qq2nn_vf, 2, sprDist, N,
                            width, states, lane
                        );
                    }

                    // Restore q to original position
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(q_vf, N), q1_vf);
                        gpuHookup(back_vf, vfNnxtFace(q_vf, N), q2_vf);
                    }
                    __syncwarp();

                    // newviewParsimony(q): recompute parsVect[q_num] from q1, q2
                    unsigned int partial = warpNewviewStep(
                        pars_tree, q_num, q1_num, q2_num, width, states
                    );
                    partial = warpReduceU32(partial);
                    if (lane == 0)
                    {
                        score_tree[q_num] = partial + score_tree[q1_num] + score_tree[q2_num];
                    }
                    __syncwarp();
                }
            }

            // ── Apply the best move found (if any) ────────────────────────────
            if (lane == 0)
            {
                sh.bcast[14] = -1;  // sentinel: no move
                if (sh.bestRemoveVf >= 0 && sh.bestInsertVf >= 0)
                {
                    bool apply = false;
                    if (sh.bestParsimony < sh.randomMP)
                    {
                        sh.randomMPHits = 1;
                        apply = true;
                    }
                    else if (sh.bestParsimony == sh.randomMP)
                    {
                        sh.randomMPHits++;
                        apply = (gpuRandum(&sh.seed) <= 1.0 / sh.randomMPHits);
                    }
                    if (apply)
                    {
                        sh.bcast[14] = sh.bestRemoveVf;
                        sh.bcast[15] = sh.bestInsertVf;
                    }
                }
            }
            __syncwarp();

            if (sh.bcast[14] >= 0)
            {
                int rm_vf = sh.bcast[14];
                int ins_vf = sh.bcast[15];
                applyMove(
                    pars_tree, score_tree, back_vf, sh, rm_vf, ins_vf, N, width, states, lane
                );
                // Recompute all node scores to keep score_tree consistent with the
                // new topology. Without this, stale subtree scores from before the
                // move cause subsequent testInsert() calls to see artificially low
                // mp values, incorrectly accepting every move and cascading randomMP
                // toward 0.  Mirrors the CPU's lazy newview on the next evaluateParsimony.
                {
                    unsigned int fullMP = recomputeAllNodes(
                        pars_tree, score_tree, back_vf, sh, topo->start_vface, N, width, states,
                        lane
                    );
                    if (lane == 0)
                    {
                        sh.randomMP = fullMP;
                    }
                    __syncwarp();
                }
            }
        }  // end for i = 1..2N-2

    } while (sh.randomMP < startMP);

    // Store final parsimony back to topology
    if (lane == 0)
    {
        topo->bestParsimony = sh.randomMP;
    }
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void gpuSprBuildTrees(
    GpuParsimonyMem* mem, int sprDist, cudaStream_t stream
)
{
    if (sprDist <= 0)
    {
        return;
    }

    // Generate per-tree seeds (distinct from build seeds)
    long* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)mem->K * sizeof(long)));
    {
        long* h_seeds = new long[mem->K];
        for (int k = 0; k < mem->K; ++k)
        {
            h_seeds[k] = 98765L + (long)(k + 1) * 54321L;
        }
        CUDA_CHECK(cudaMemcpyAsync(
            d_seeds, h_seeds, (size_t)mem->K * sizeof(long), cudaMemcpyHostToDevice, stream
        ));
        delete[] h_seeds;
    }

    const size_t sharedBytes = sizeof(SprShared);
    dim3 grid(mem->K, 1, 1);
    dim3 block(kWarpSize, 1, 1);

    printf(
        "[GPU] gpuSprKernel: K=%d  sprDist=%d  shared=%.1f KB\n", mem->K, sprDist,
        sharedBytes / 1024.0
    );

    gpuSprKernel<<<grid, block, sharedBytes, stream>>>(
        mem->d_parsVect, mem->d_parsScore, mem->d_topos, d_seeds, mem->width, mem->states, sprDist,
        mem->parsVectPerTree, mem->parsScorePerTree
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
