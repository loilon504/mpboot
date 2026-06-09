#include <type_traits>

// Opt-5: causes pars_tree.cuh to define g_sankoff_cm here (not extern) — avoids NVCC redefinition.
#define PARS_BUILD_DEFINE_CM

#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/topo_helpers.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

void gpuUploadSankoffCostMatrix(const unsigned int* cm, int nstates)
{
    const size_t costBytes = (size_t)nstates * nstates * sizeof(unsigned int);
    CUDA_CHECK(cudaMemcpyToSymbol(g_sankoff_cm, cm, costBytes));
}

__device__ __forceinline__ void gpuHookup(
    GpuTopology* t, int a, int b
)
{
    t->back_vf[a] = b;
    t->back_vf[b] = a;
}

// ─── testInsert ───────────────────────────────────────────────────────────────
template <int STATES, typename SharedT>
__device__ void testInsert(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int q,
    int N,
    int width,
    int lane,
    unsigned int* treels_scores  = nullptr,
    int*          treels_back_vf = nullptr,
    int*          treels_filled  = nullptr,
    unsigned int* treels_cutoff  = nullptr,
    int           max_treels     = 0,
    unsigned int* treels_hashes  = nullptr
)
{
    if (lane == 0)
    {
        sh.bcast[0] = topo->back_vf[q];
        gpuHookup(topo->back_vf, vfNextFace(p, N), q);
        gpuHookup(topo->back_vf, vfNnxtFace(p, N), sh.bcast[0]);
    }
    __syncwarp();

    // Pre-refresh: refresh q, r, and tip_p subtrees WITHOUT computing parsVect[p] from face[2].
    // face0's children (for step 2) are tip_p and q — both must be fresh (xpars=1).
    // Without tip_p refresh, it degrades to xpars=0 across iterations → extra eval traversal.
    if (lane == 0)
    {
        const int r_vf = sh.bcast[0];
        const int tip_p_vf = topo->back_vf[p];  // back of remove node, unchanged by hookup
        sh.tiSize = 3;
        if (q >= N && !topo->xpars[q])
        {
            computeTraversalInfoParsimony(topo, sh, q, N, false);
        }
        if (r_vf >= N && !topo->xpars[r_vf])
        {
            computeTraversalInfoParsimony(topo, sh, r_vf, N, false);
        }
        if (tip_p_vf >= N && !topo->xpars[tip_p_vf])
        {
            computeTraversalInfoParsimony(topo, sh, tip_p_vf, N, false);
        }
    }
    __syncwarp();
    if (sh.tiSize > 3)
    {
        newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, false, width);
    }
    if (lane == 0)
    {
        if (q >= N)
        {
            topo->xpars[q] = 1;
        }
        if (sh.bcast[0] >= N)
        {
            topo->xpars[sh.bcast[0]] = 1;
        }
        const int tip_p_vf = topo->back_vf[p];
        if (tip_p_vf >= N)
        {
            topo->xpars[tip_p_vf] = 1;
        }
        topo->xpars[vfNnxtFace(p, N)] = 0;  // force step 2 to recompute p from face0
    }
    __syncwarp();

    // Step 2: evaluate parsimony at edge (face0, r). Always inlined to avoid if/else divergence.
    const int face0_ev = vfNnxtFace(p, N);
    unsigned int mp;
    {
        if (lane == 0)
        {
            const int r_vf_ev = topo->back_vf[face0_ev];
            sh.tiSize = 3;
            sh.ti[1] = vfToNum(face0_ev, N);
            sh.ti[2] = vfToNum(r_vf_ev, N);
            if (face0_ev >= N && !topo->xpars[face0_ev])
            {
                computeTraversalInfoParsimony(topo, sh, face0_ev, N, false);
            }
            if (r_vf_ev >= N && !topo->xpars[r_vf_ev])
            {
                computeTraversalInfoParsimony(topo, sh, r_vf_ev, N, false);
            }
        }
        __syncwarp();
        mp = newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, true, width);
    }

    if (lane == 0)
    {
        if (mp < sh.bestParsimony)
        {
            sh.bestParsimony = mp;
            sh.bestHits = 1;
            sh.bestRemoveVf = p;
            sh.bestInsertVf = q;
        }
        else if (mp == sh.bestParsimony && gpuRandum(&sh.seed) <= 1.0 / ++sh.bestHits)
        {
            sh.bestRemoveVf = p;
            sh.bestInsertVf = q;
        }
    }
    __syncwarp();

    // Save to treels if mp improves running best and passes cutoff.
    // Topology is currently "p inserted at q" (before rollback) — valid to copy directly.
    // Uses bcast[7] for slot index (safe: outer bcast[7] consumed before this call).
    if (treels_scores != nullptr)
    {
        if (lane == 0)
        {
            sh.bcast[7] = -1;
            bool pass_save_a = (sh.save_margin < 0.0f)
                               ? true
                               : (mp < (unsigned int)((float)sh.randomMP * (1.0f + sh.save_margin)));
            if (pass_save_a)
            {
                unsigned int cutoff = *((volatile unsigned int*)treels_cutoff);
                if (mp <= cutoff && *((volatile int*)treels_filled) < max_treels)
                {
                    int slot = atomicAdd(treels_filled, 1);
                    if (slot < max_treels)
                    {
                        treels_scores[slot] = mp;
                        sh.bcast[7] = slot;
                    }
                }
            }
        }
        __syncwarp();
        if (sh.bcast[7] >= 0)
        {
            int* dst = treels_back_vf + (size_t)sh.bcast[7] * kMaxVFaces;
            for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                dst[vf] = topo->back_vf[vf];
            if (lane == 0 && treels_hashes != nullptr)
            {
                unsigned int h = 0;
                for (int vf = 0; vf < topo->num_vfaces; vf++)
                    h = h * 2654435761u ^ (unsigned int)topo->back_vf[vf];
                treels_hashes[sh.bcast[7]] = h;
            }
            __syncwarp();
        }
    }

    if (lane == 0)
    {
        gpuHookup(topo->back_vf, q, sh.bcast[0]);
        topo->back_vf[vfNextFace(p, N)] = -1;
        topo->back_vf[vfNnxtFace(p, N)] = -1;
    }
    __syncwarp();
}

// ─── doAddTraverse ────────────────────────────────────────────────────────────
template <int STATES, typename SharedT>
__device__ void doAddTraverse(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int q,
    int mintrav,
    int maxtrav,
    int N,
    int width,
    int lane,
    unsigned int* treels_scores  = nullptr,
    int*          treels_back_vf = nullptr,
    int*          treels_filled  = nullptr,
    unsigned int* treels_cutoff  = nullptr,
    int           max_treels     = 0,
    unsigned int* treels_hashes  = nullptr
)
{
    if (lane == 0)
    {
        sh.stack[0] = q;
        sh.stackMint[0] = mintrav - 1;
        sh.stackMaxt[0] = maxtrav - 1;
        sh.stackTop = 1;
    }
    __syncwarp();

    // Opt-B+: tighter lower bound — add score_tree[tip_p] (computed once per call).
    // mp >= score_tree[tip_p] + score_tree[q_cand] + score_tree[r] (all cross terms >= 0).
    const unsigned int score_tip_p = score_tree[vfToNum(topo->back_vf[p], N)];

    while (sh.stackTop > 0)
    {
        if (lane == 0)
        {
            int top = --sh.stackTop;
            sh.bcast[1] = sh.stack[top];
            sh.bcast[2] = sh.stackMint[top];
            sh.bcast[3] = sh.stackMaxt[top];
        }
        __syncwarp();
        const int cur_q = sh.bcast[1], mint = sh.bcast[2], maxt = sh.bcast[3];
        const int q_num = vfToNum(cur_q, N);

        if (mint <= 0)
        {
            // Opt-B: lower bound prune — mp >= score_tree[cur_q] + score_tree[r] always.
            // If lb >= sh.randomMP, testInsert cannot improve → skip safely.
            const unsigned int lb = score_tip_p + score_tree[q_num]
                                    + score_tree[vfToNum(topo->back_vf[cur_q], N)];
            if (lb < sh.randomMP)
            {
                testInsert<STATES>(pars_tree, score_tree, topo, sh, p, cur_q, N, width, lane,
                    treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                    max_treels, treels_hashes);
            }
        }

        if (q_num > N && maxt > 0)
        {
            if (lane == 0)
            {
                int qn = topo->back_vf[vfNextFace(cur_q, N)];
                int qnn = topo->back_vf[vfNnxtFace(cur_q, N)];

                sh.stack[sh.stackTop] = qnn;
                sh.stackMint[sh.stackTop] = mint - 1;
                sh.stackMaxt[sh.stackTop] = maxt - 1;
                sh.stackTop++;
                sh.stack[sh.stackTop] = qn;
                sh.stackMint[sh.stackTop] = mint - 1;
                sh.stackMaxt[sh.stackTop] = maxt - 1;
                sh.stackTop++;
            }
            __syncwarp();
        }
    }
}

// ─── gpuNodeRectifierPars ────────────────────────────────────────────────────
// GPU equivalent of CPU nodeRectifierPars + reorderNodes (sprparsimony.cpp:2089).
// DFS from nodep[1]->back; reassigns nodep[N+1..2N-1] in DFS order without
// touching xpars or back_vf. Lane 0 only.
template <typename SharedT>
__device__ void gpuNodeRectifierPars(
    GpuTopology* topo, SharedT& sh, int N
)
{
    // Mirrors CPU nodeRectifierPars + reorderNodes (sprparsimony.cpp:2089).
    // DFS from nodep[1]->back; reassigns nodep[N+1..2N-1] in DFS order.
    // Does NOT touch xpars — tree DFS via children only, no cycle possible.
    int count = 0;
    int top = -1;
    sh.tiStack[++top] = topo->back_vf[topo->nodep[1]];
    topo->start_vface = topo->nodep[1];

    while (top >= 0)
    {
        int m_vf = sh.tiStack[top--];
        if (m_vf < N)
        {
            continue;  // null or tip
        }

        // CPU: tr->nodep[count + N + 1] = p  (the exact face encountered in DFS)
        topo->nodep[count + N + 1] = m_vf;
        count++;

        // CPU recurses p->next->back first, p->next->next->back second.
        // LIFO stack: push child2 first so child1 is popped (processed) first.
        int child1 = topo->back_vf[vfNextFace(m_vf, N)];
        int child2 = topo->back_vf[vfNnxtFace(m_vf, N)];
        if (child2 >= N)
        {
            sh.tiStack[++top] = child2;
        }
        if (child1 >= N)
        {
            sh.tiStack[++top] = child1;
        }
    }
}

// ─── applyMove ────────────────────────────────────────────────────────────────
template <int STATES, typename SharedT>
__device__ void applyMove(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int rm,
    int ins,
    int N,
    int width,
    int lane
)
{
    if (lane == 0)
    {
        int pn = vfNextFace(rm, N), pnn = vfNnxtFace(rm, N);
        gpuHookup(topo->back_vf, topo->back_vf[pn], topo->back_vf[pnn]);
        topo->back_vf[pn] = -1;
        topo->back_vf[pnn] = -1;

        int r = topo->back_vf[ins];
        gpuHookup(topo->back_vf, pn, ins);
        gpuHookup(topo->back_vf, pnn, r);
    }
    __syncwarp();
    createTiAndNewviewParsimony<SharedT, STATES>(
        pars_tree, score_tree, topo, sh, rm, N, width, lane
    );
    __syncwarp();
}

template <int STATES, typename SharedT>
__device__ void gpuSPRHillClimb(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int N,
    int sprDist,
    int width,
    int lane,
    unsigned int* treels_scores  = nullptr,
    int*          treels_back_vf = nullptr,
    int*          treels_filled  = nullptr,
    unsigned int* treels_cutoff  = nullptr,
    int           max_treels     = 0,
    unsigned int* treels_hashes  = nullptr
)
{
    int* const back_vf = topo->back_vf;

    // Skip gpuNodeRectifierPars on the first do-while iteration:
    // nodep[] is guaranteed fresh by the caller (runPhase3 calls it before gpuSPRHillClimb).
    bool first_iter = true;
    unsigned int startMP;
    if (lane == 0)
    {
        sh.bestParsimony = sh.randomMP;
    }
    __syncwarp();
    do
    {
        startMP = sh.randomMP;
        if (!first_iter)
        {
            if (lane == 0) { gpuNodeRectifierPars(topo, sh, N); }
            __syncwarp();
        }
        first_iter = false;

        for (int i = 1; i <= 2 * N - 2; i++)
        {
            if (lane == 0)
            {
                sh.bestHits = 1;
                sh.bestRemoveVf = -1;
                sh.bestInsertVf = -1;
                sh.bcast[4] = topo->nodep[i];
                sh.bcast[5] = back_vf[sh.bcast[4]];
            }
            __syncwarp();
            const int p = sh.bcast[4], q = sh.bcast[5];

            createTiAndEvaluateParsimony<SharedT, STATES>(
                pars_tree, score_tree, topo, sh, p, N, false, width
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
                const int p1 = sh.bcast[7];
                const int p2 = sh.bcast[8];

                if (p1 >= N || p2 >= N)
                {
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, p1, p2);
                        back_vf[vfNextFace(p, N)] = -1;
                        back_vf[vfNnxtFace(p, N)] = -1;
                    }
                    __syncwarp();

                    if (p1 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNextFace(p1, N)], 1,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p1, N)], 1,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                    }
                    if (p2 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNextFace(p2, N)], 1,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p2, N)], 1,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(p, N), p1);
                        gpuHookup(back_vf, vfNnxtFace(p, N), p2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony<SharedT, STATES>(
                        pars_tree, score_tree, topo, sh, p, N, width, lane
                    );
                    __syncwarp();
                }
            }

            // ── Q-branch ─────────────────────────────────────────────────────
            if (q >= N)
            {
                if (lane == 0)
                {
                    sh.bcast[7] = back_vf[vfNextFace(q, N)];
                    sh.bcast[8] = back_vf[vfNnxtFace(q, N)];
                }
                __syncwarp();
                const int q1 = sh.bcast[7], q2 = sh.bcast[8];

                bool q1_gc = q1 >= N
                             && (back_vf[vfNextFace(q1, N)] >= N
                                 || back_vf[vfNnxtFace(q1, N)] >= N);
                bool q2_gc = q2 >= N
                             && (back_vf[vfNextFace(q2, N)] >= N
                                 || back_vf[vfNnxtFace(q2, N)] >= N);

                if (q1_gc || q2_gc)
                {
                    if (lane == 0)
                    {
                        gpuHookup(back_vf, q1, q2);
                        back_vf[vfNextFace(q, N)] = -1;
                        back_vf[vfNnxtFace(q, N)] = -1;
                    }
                    __syncwarp();

                    if (q1 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNextFace(q1, N)], 2,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q1, N)], 2,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                    }
                    if (q2 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNextFace(q2, N)], 2,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q2, N)], 2,
                            sprDist, N, width, lane,
                            treels_scores, treels_back_vf, treels_filled, treels_cutoff,
                            max_treels, treels_hashes
                        );
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(q, N), q1);
                        gpuHookup(back_vf, vfNnxtFace(q, N), q2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony<SharedT, STATES>(
                        pars_tree, score_tree, topo, sh, q, N, width, lane
                    );
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
                        sh.bcast[9] = sh.bestRemoveVf;
                        sh.bcast[10] = sh.bestInsertVf;
                    }
                }
            }
            __syncwarp();

            if (sh.bcast[9] >= 0)
            {
                applyMove<STATES>(
                    pars_tree, score_tree, topo, sh, sh.bcast[9], sh.bcast[10], N, width, lane
                );
                __syncwarp();
                if (lane == 0)
                {
                    sh.randomMP = sh.bestParsimony;
                }
                __syncwarp();
            }

            // ── Treels: save best candidate per-node-i (regardless of improvement) ──
            // If improvement: back_vf already reflects candidate (applyMove done above).
            // If no improvement: temporarily apply → copy → undo via second applyMove.
            if (lane == 0)
            {
                sh.bcast[6] = -1;
                if (sh.bestRemoveVf >= 0 && sh.bestInsertVf >= 0 &&
                    treels_scores != nullptr &&
                    *((volatile int*)treels_filled) < max_treels)
                {
                    unsigned int cutoff = *((volatile unsigned int*)treels_cutoff);
                    if (sh.bestParsimony <= cutoff)
                    {
                        int slot = atomicAdd(treels_filled, 1);
                        if (slot < max_treels)
                        {
                            treels_scores[slot] = sh.bestParsimony;
                            sh.bcast[6] = slot;
                            if (sh.bcast[9] < 0)
                                sh.bcast[3] = topo->back_vf[vfNextFace(sh.bestRemoveVf, N)];
                        }
                    }
                }
            }
            __syncwarp();
            if (sh.bcast[6] >= 0)
            {
                if (sh.bcast[9] < 0)
                {
                    // Temporarily apply: back_vf does not yet reflect candidate
                    applyMove<STATES>(
                        pars_tree, score_tree, topo, sh,
                        sh.bestRemoveVf, sh.bestInsertVf, N, width, lane
                    );
                    __syncwarp();
                }
                int* dst = treels_back_vf + (size_t)sh.bcast[6] * kMaxVFaces;
                for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                    dst[vf] = topo->back_vf[vf];
                // Compute topology hash (lane 0) for dedup on CPU
                if (lane == 0 && treels_hashes != nullptr)
                {
                    unsigned int h = 0;
                    for (int vf = 0; vf < topo->num_vfaces; vf++)
                        h = h * 2654435761u ^ (unsigned int)topo->back_vf[vf];
                    treels_hashes[sh.bcast[6]] = h;
                }
                __syncwarp();
                if (sh.bcast[9] < 0)
                {
                    // Undo: re-apply rm to original p1 (saved in bcast[3])
                    // After temp apply, rm's old siblings (p1, p2) are hooked together,
                    // so applyMove(rm, p1) correctly re-inserts rm between p1 and p2.
                    applyMove<STATES>(
                        pars_tree, score_tree, topo, sh,
                        sh.bestRemoveVf, sh.bcast[3], N, width, lane
                    );
                    __syncwarp();
                }
            }
        }  // end for i

    } while (sh.randomMP < startMP);
}

// ─── gpuRandomNNIs ───────────────────────────────────────────────────────────
template <typename SharedT>
__device__ void gpuRandomNNIs(
    GpuTopology* topo, SharedT& sh, int N, int numNNI, int lane
)
{
    if (lane != 0)
    {
        __syncwarp();
        return;
    }

    // Store only canonical p_vf per inner node (N-1 entries).
    // q_vf is looked up dynamically from back_vf[p_vf] each iteration so it always
    // reflects the CURRENT topology — avoids stale-edge corruption after prior NNIs.
    int num_inner = 0;
    for (int p_num = N + 1; p_num <= 2 * N - 1; p_num++)
        sh.stack[num_inner++] = (int16_t)nodepVf(p_num, N);
    if (num_inner == 0) { __syncwarp(); return; }

    const int bitset_words = (2 * N) / 32 + 1;  // ≤ 51 ≤ kMaxSprStack=64
    for (int w = 0; w < bitset_words; w++)
        sh.stackMaxt[w] = 0;

    // CPU-style: random pick per NNI, reset bitset on conflict (mirrors doRandomNNIs).
    // Guarantees exactly numNNI loop iterations; skips an iteration only if q is a tip.
    for (int i = 0; i < numNNI; i++)
    {
        int idx = (int)(gpuRandum(&sh.seed) * num_inner);
        if (idx >= num_inner) idx = num_inner - 1;

        int p_vf  = sh.stack[idx];
        int q_vf  = topo->back_vf[p_vf];  // current neighbor — always up-to-date
        int p_num = vfToNum(p_vf, N);
        int q_num = vfToNum(q_vf, N);

        // NNI requires inner-inner edge; skip (don't count) if q is a tip
        if (q_num <= N) continue;

        int pw = (p_num - 1) >> 5, pb = (p_num - 1) & 31;
        int qw = (q_num - 1) >> 5, qb = (q_num - 1) & 31;

        // Conflict → reset bitset (CPU: usedNodes.clear()), then apply anyway
        if ((sh.stackMaxt[pw] & (1u << pb)) || (sh.stackMaxt[qw] & (1u << qb)))
            for (int w = 0; w < bitset_words; w++) sh.stackMaxt[w] = 0;

        int pf0 = vfNnxtFace(p_vf, N), qf1 = vfNextFace(q_vf, N), qf0 = vfNnxtFace(q_vf, N);
        int b = topo->back_vf[pf0];

        if ((int)(gpuRandum(&sh.seed) * 2) == 0)
        {
            int c = topo->back_vf[qf1];
            // Guards: -1 = null pointer; c==pf0 = self-loop at pf0
            if (b != -1 && c != -1 && c != pf0)
            {
                gpuHookup(topo->back_vf, pf0, c);
                gpuHookup(topo->back_vf, qf1, b);
            }
        }
        else
        {
            int d = topo->back_vf[qf0];
            // Guards: -1 = null pointer; d==pf0 = self-loop at pf0
            if (b != -1 && d != -1 && d != pf0)
            {
                gpuHookup(topo->back_vf, pf0, d);
                gpuHookup(topo->back_vf, qf0, b);
            }
        }
        sh.stackMaxt[pw] |= (1u << pb);
        sh.stackMaxt[qw] |= (1u << qb);
    }
    __syncwarp();
}


// ─── Phase 3 device helper (shared by buildParsimonyTreesKernel + buildPhase3Kernel) ──
// Runs exactly one iteration: pool restart → NNI or ratchet → SPR → pool insert → treels write.
// Worker parity (blockIdx.x % 2): even = NNI, odd = ratchet — fixed per worker, not per iteration.
// CPU calling loop controls how many times the kernel is launched.
template <int STATES, typename SharedT>
__device__ void runPhase3(
    parsimonyNumber* pars_tree,
    unsigned int* score_tree,
    GpuTopology* topo,
    SharedT& sh,
    unsigned int* sw_k,
    unsigned int* ratchet_k,
    int N,
    int sprDist,
    int numNNI,
    int lane,
    int k,
    int width,
    int pool_size,
    unsigned int* pool_scores,    // [pool_size] parsimony score per physical slot (UINT_MAX=empty)
    int* pool_back_vf,            // [pool_size * kMaxVFaces] topology per physical slot
    int* pool_filled,             // device ptr: number of filled slots (0..pool_size)
    int* pool_slot_locks,         // device ptr: per-slot spinlocks [pool_size]
    unsigned int* pool_hashes,    // [pool_size] topology hash per slot (0xFFFFFFFF=empty)
    unsigned int* global_best,    // device ptr: global best parsimony across all warps
    unsigned int* treels_scores,  // [max_treels] or nullptr — bootstrap output buffer
    int*   treels_back_vf,        // [max_treels × kMaxVFaces] or nullptr
    int*   treels_filled,         // atomic fill counter or nullptr
    unsigned int* treels_cutoff,  // score ≤ cutoff → write to treels; nullptr = disabled
    int    max_treels,            // treels buffer capacity
    unsigned int* treels_hashes,  // [max_treels] topology hash per slot; nullptr = disabled
    int    treels_writers         // only blockIdx.x < treels_writers write to treels
)
{
    // Writer flag: only the first treels_writers blocks may write to the treels buffer.
    // Non-writers still update the pool and global best, contributing to search diversity.
    const bool is_writer = (blockIdx.x < treels_writers);

    // ── Step 1: Pool restart ─────────────────────────────────────────────────
    // Writers: deterministic slot (blockIdx.x/2 % pool_size) ensures each pool tree
    // gets exactly one NNI worker + one ratchet worker exploring it per round.
    // Non-writers: random probe for diversity in pool updates.
    if (pool_scores != nullptr)
    {
        if (lane == 0)
        {
            sh.bcast[2] = -1;
            int filled = *pool_filled;
            if (filled > 0)
            {
                int start_slot;
                if (is_writer)
                {
                    // Deterministic: pair (0,1)→slot 0, pair (2,3)→slot 1, …
                    start_slot = (blockIdx.x / 2) % pool_size;
                }
                else
                {
                    // Random: linear probe from a hash of blockIdx.x
                    unsigned int rv = (unsigned int)(blockIdx.x * 2654435761u ^ 1013904223u);
                    rv = rv ^ (rv >> 16);
                    start_slot = (int)(rv % (unsigned int)pool_size);
                }
                for (int attempt = 0; attempt < pool_size; attempt++)
                {
                    int slot = (start_slot + attempt) % pool_size;
                    if (pool_scores[slot] != 0xFFFFFFFFu)
                    {
                        sh.bcast[2] = slot;
                        sh.randomMP = pool_scores[slot];
                        sh.randomMPHits = 1;
                        topo->bestParsimony = sh.randomMP;
                        break;
                    }
                }
            }
        }
        __syncwarp();

        if (sh.bcast[2] >= 0)
        {
            int phys_slot = sh.bcast[2];
            if (lane == 0)
            {
                while (atomicCAS(&pool_slot_locks[phys_slot], 0, 1) != 0) {}
            }
            __syncwarp();
            // Volatile reads bypass L1 cache: pool_back_vf may have been written by a
            // different SM in the previous round — L1 on this SM may be stale.
            const volatile int* src = (const volatile int*)(pool_back_vf + (size_t)phys_slot * kMaxVFaces);
            for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
            {
                topo->back_vf[vf] = src[vf];
                topo->xpars[vf] = 0;
            }
            __syncwarp();
            if (lane == 0)
            {
                atomicExch(&pool_slot_locks[phys_slot], 0);
            }
            __syncwarp();

            if (lane == 0)
            {
                gpuNodeRectifierPars(topo, sh, N);
            }
            __syncwarp();
        }
    }

    // ── Step 2: Even workers → NNI, odd workers → ratchet (fixed per worker) ─
    // Non-writers pass nullptr for treels to gpuSPRHillClimb, suppressing SAVE A writes.
    unsigned int* w_treels_scores  = is_writer ? treels_scores  : nullptr;
    int*          w_treels_back_vf = is_writer ? treels_back_vf : nullptr;
    int*          w_treels_filled  = is_writer ? treels_filled  : nullptr;
    unsigned int* w_treels_cutoff  = is_writer ? treels_cutoff  : nullptr;
    int           w_max_treels     = is_writer ? max_treels     : 0;
    unsigned int* w_treels_hashes  = is_writer ? treels_hashes  : nullptr;

    const bool iter_is_nni = (blockIdx.x % 2 == 0);
    if (iter_is_nni)
    {
        gpuRandomNNIs<SharedT>(topo, sh, N, numNNI, lane);

        unsigned int pm = createTiAndEvaluateParsimony<SharedT, STATES>(
            pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
        );
        if (lane == 0)
        {
            sh.randomMP = pm;
            sh.randomMPHits = 1;
        }
        __syncwarp();

        gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane,
            w_treels_scores, w_treels_back_vf, w_treels_filled, w_treels_cutoff, w_max_treels, w_treels_hashes);
    }
    else
    {
        if (lane == 0)
        {
            if (sh.use_sankoff)
            {
                // Sankoff: multiply original pattern frequencies by ratchet factor {1,2}
                // ratchet_k = temp buffer; sw_k (original freqs) stays untouched
                for (int b = 0; b < width; b++)
                    ratchet_k[b] = sw_k[b] * ((gpuRandum(&sh.seed) < 0.5) ? 2u : 1u);
                sh.site_weights = ratchet_k;
            }
            else
            {
                // Fitch: write {1,2} directly into sw_k (uniform baseline, no real freqs)
                for (int b = 0; b < width; b++)
                    sw_k[b] = (gpuRandum(&sh.seed) < 0.5) ? 2u : 1u;
                sh.site_weights = sw_k;
            }
        }
        __syncwarp();

        unsigned int pm1 = createTiAndEvaluateParsimony<SharedT, STATES>(
            pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
        );
        if (lane == 0)
        {
            sh.randomMP = pm1;
            sh.randomMPHits = 1;
        }
        __syncwarp();

        gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane,
            w_treels_scores, w_treels_back_vf, w_treels_filled, w_treels_cutoff, w_max_treels, w_treels_hashes);

        if (lane == 0)
            sh.site_weights = sh.use_sankoff ? sw_k : nullptr;  // restore original weights
        __syncwarp();

        unsigned int pm2 = createTiAndEvaluateParsimony<SharedT, STATES>(
            pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
        );
        if (lane == 0)
        {
            sh.randomMP = pm2;
            sh.randomMPHits = 1;
        }
        __syncwarp();

        gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane,
            w_treels_scores, w_treels_back_vf, w_treels_filled, w_treels_cutoff, w_max_treels, w_treels_hashes);
    }

    // ── Step 3: Stagnation tracking + pool insert with hash dedup ────────────
    if (lane == 0)
    {
        if (iter_is_nni)
        {
            topo->n_total_even++;
        }
        else
        {
            topo->n_total_odd++;
        }

        const unsigned int best_before = topo->bestParsimony;
        if (sh.randomMP < best_before)
        {
            topo->bestParsimony = sh.randomMP;
            if (iter_is_nni)
            {
                topo->n_improved_even++;
            }
            else
            {
                topo->n_improved_odd++;
            }
        }
    }
    __syncwarp();

    // Pool update: dedup via hash, then scan for worst slot, CAS-claim, lock for back_vf copy.
    if (lane == 0)
    {
        sh.bcast[4] = -1;
        if (pool_scores != nullptr)
        {
            // Compute topology hash (Knuth multiplicative hash over all back_vf entries)
            unsigned int new_hash = 0;
            for (int vf = 0; vf < topo->num_vfaces; vf++)
                new_hash = new_hash * 2654435761u ^ (unsigned int)topo->back_vf[vf];

            // Check for duplicate: skip insert if same hash already present with valid score.
            // Race-condition false negatives are acceptable — pool diversity may lose one slot.
            bool is_dup = false;
            if (pool_hashes != nullptr)
            {
                for (int i = 0; i < pool_size; i++)
                {
                    if (pool_hashes[i] == new_hash && pool_scores[i] != 0xFFFFFFFFu)
                    {
                        is_dup = true;
                        break;
                    }
                }
            }

            if (!is_dup)
            {
                // Find worst slot (max score) without holding lock
                int worst_slot = 0;
                unsigned int worst_score = pool_scores[0];
                for (int i = 1; i < pool_size; i++)
                {
                    unsigned int s = pool_scores[i];
                    if (s > worst_score)
                    {
                        worst_score = s;
                        worst_slot = i;
                    }
                }
                if (sh.randomMP < worst_score)
                {
                    unsigned int old = atomicCAS(
                        (unsigned int*)&pool_scores[worst_slot], worst_score, sh.randomMP
                    );
                    if (old == worst_score)
                    {
                        if (old == 0xFFFFFFFFu)
                        {
                            atomicAdd(pool_filled, 1);
                        }
                        sh.bcast[4] = worst_slot;
                        while (atomicCAS(&pool_slot_locks[worst_slot], 0, 1) != 0) {}
                        // Write hash inside lock so concurrent dedup checks see it atomically
                        if (pool_hashes != nullptr)
                            pool_hashes[worst_slot] = new_hash;
                    }
                }
            }
        }
    }
    __syncwarp();

    // All 32 lanes copy topology; lock still held by lane 0 when bcast[4] >= 0.
    if (sh.bcast[4] >= 0)
    {
        int* dst = pool_back_vf + (size_t)sh.bcast[4] * kMaxVFaces;
        for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
        {
            dst[vf] = topo->back_vf[vf];
        }
        // All 32 lanes fence their own stores: each lane wrote different vfaces
        // (0,32,64,... / 1,33,65,... / etc.), so each lane must flush its own
        // writes to L2.  A fence by lane 0 alone only guarantees lane 0's vfaces
        // are visible — lanes 1..31 stores may still be in write-back L1, causing
        // readers on other SMs to see garbage from cudaMalloc in the next round.
        __threadfence();
        __syncwarp();
    }
    __syncwarp();

    // Release per-slot spinlock after all stores are globally visible.
    if (sh.bcast[4] >= 0 && lane == 0)
    {
        atomicExch(&pool_slot_locks[sh.bcast[4]], 0);
    }
    __syncwarp();

    // ── Treels write: add current tree to bootstrap output buffer if score ≤ cutoff ──
    // No lock needed: each slot assigned uniquely via atomicAdd → no two warps collide.
    // Only writer blocks (blockIdx.x < treels_writers) may write here.
    if (is_writer && treels_scores != nullptr && treels_filled != nullptr && treels_cutoff != nullptr)
    {
        if (lane == 0)
        {
            sh.bcast[6] = -1;
            unsigned int cutoff = *((volatile unsigned int*)treels_cutoff);
            if (sh.randomMP <= cutoff)
            {
                int slot = atomicAdd(treels_filled, 1);
                if (slot < max_treels)
                {
                    treels_scores[slot] = sh.randomMP;
                    sh.bcast[6] = slot;
                }
            }
        }
        __syncwarp();
        if (sh.bcast[6] >= 0)
        {
            int* dst = treels_back_vf + (size_t)sh.bcast[6] * kMaxVFaces;
            for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                dst[vf] = topo->back_vf[vf];
            // Compute topology hash (lane 0) and store alongside back_vf
            if (lane == 0 && treels_hashes != nullptr)
            {
                unsigned int h = 0;
                for (int vf = 0; vf < topo->num_vfaces; vf++)
                    h = h * 2654435761u ^ (unsigned int)topo->back_vf[vf];
                treels_hashes[sh.bcast[6]] = h;
            }
            __syncwarp();
        }
        __syncwarp();
    }

    // Update global best
    if (lane == 0 && global_best != nullptr)
    {
        if (sh.randomMP < *global_best)
            atomicMin(global_best, sh.randomMP);
    }
    __syncwarp();
}

// ─── Kernel ───────────────────────────────────────────────────────────────────
template <int STATES, int NTAXA>
__global__ void buildParsimonyTreesKernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    unsigned int* __restrict__ d_siteWeights,
    const long* __restrict__ d_seeds,
    int width,
    int sprDist,
    unsigned int* d_postSprScores,  // write postSprParsimony[k] for Phase 3 threshold
    size_t parsVectPerTree,
    size_t parsScorePerTree,
    const unsigned int* d_cost_matrix  // nullptr = Fitch mode
)
{
    __shared__ BuildSharedT<NTAXA> sh;
    using SharedT = BuildSharedT<NTAXA>;

    const int k = blockIdx.x;
    const int lane = threadIdx.x;

    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    GpuTopology* topo = d_topos + k;
    unsigned int* sw_k = d_siteWeights ? (d_siteWeights + (size_t)k * width) : nullptr;
    const int N = topo->mxtips;

    // ── Phase 0: permutation + 3-tip tree ────────────────────────────────────
    if (lane == 0)
    {
        sh.seed = d_seeds[k];
        sh.use_sankoff = (d_cost_matrix != nullptr);
        sh.site_weights = sh.use_sankoff ? sw_k : nullptr;
        sh.save_margin = -1.0f;  // K1 doesn't write to treels

        for (int i = 1; i <= N; i++)
        {
            sh.perm[i] = i;
        }
        for (int i = 1; i <= N; i++)
        {
            int k2 = (int)((double)(N + 1 - i) * gpuRandum(&sh.seed));
            int tmp = sh.perm[i];
            sh.perm[i] = sh.perm[i + k2];
            sh.perm[i + k2] = tmp;
        }

        int ip = sh.perm[1] - 1, iq = sh.perm[2] - 1, ir = sh.perm[3] - 1;
        sh.startVf = ip < iq ? (ip < ir ? ip : ir) : (iq < ir ? iq : ir);
        gpuHookup(topo, ip, N + 1);
        gpuHookup(topo, iq, N);
        gpuHookup(topo, ir, N + 2);

        score_tree[N + 1] = 0;
        topo->nextnode = N + 2;
        sh.bestParsimony = UINT_MAX;
        sh.bestHits = 1;
        // nodep[] and xpars already initialized by cpuToGpuTopology on host before upload
    }
    __syncwarp();

    createTiAndNewviewParsimony<SharedT, STATES>(
        pars_tree, score_tree, topo, sh, N + 2, N, width, lane
    );
    __syncwarp();

    // ── Phase 1: stepwise addition ────────────────────────────────────────────
    for (int nextsp = 4; nextsp <= N; nextsp++)
    {
        if (lane == 0)
        {
            sh.bestParsimony = UINT_MAX;
            sh.tipnum = sh.perm[nextsp];
            sh.qnum = topo->nextnode++;
            sh.qf0 = N + 3 * (sh.qnum - N - 1);
            sh.qf1 = sh.qf0 + 1;
            sh.qf2 = sh.qf0 + 2;
            gpuHookup(topo, sh.tipnum - 1, sh.qf2);
            sh.stack[0] = topo->back_vf[sh.startVf];
            sh.stackTop = 1;
            score_tree[sh.qnum] = 0;
        }
        __syncwarp();

        const int q_f0 = sh.qf0, q_f1 = sh.qf1, q_f2 = sh.qf2;

        while (sh.stackTop > 0)
        {
            if (lane == 0)
            {
                int node = sh.stack[--sh.stackTop], child = topo->back_vf[node];
                topo->back_vf[q_f1] = node;
                topo->back_vf[node] = q_f1;
                topo->back_vf[q_f0] = child;
                topo->back_vf[child] = q_f0;
                sh.stack[sh.stackTop] = node;
                sh.stack[sh.stackTop + 1] = child;
            }
            __syncwarp();

            const int node = sh.stack[sh.stackTop], child = sh.stack[sh.stackTop + 1];
            const int node_num = vfToNum(node, N);

            if (lane == 0)
            {
                sh.tiSize = 3;
                computeTraversalInfoParsimony(topo, sh, q_f2, N, false);
                sh.ti[1] = vfToNum(q_f2, N);
                sh.ti[2] = vfToNum(topo->back_vf[q_f2], N);
            }
            unsigned int mp = newviewParsimony<SharedT, STATES>(
                pars_tree, score_tree, sh, true, width
            );
            __syncwarp();

            if (lane == 0)
            {
                if (mp < sh.bestParsimony)
                {
                    sh.bestParsimony = mp;
                    sh.bestHits = 1;
                    sh.insertVf = node;
                }
                else if (mp == sh.bestParsimony)
                {
                    sh.bestHits++;
                }
                topo->back_vf[node] = child;
                topo->back_vf[child] = node;
                if (node_num > N && score_tree[node_num] > 0)
                {
                    sh.stack[sh.stackTop++] = topo->back_vf[vfNextFace(node, N)];
                    sh.stack[sh.stackTop++] = topo->back_vf[vfNnxtFace(node, N)];
                }
            }
            __syncwarp();
        }

        if (lane == 0)
        {
            int r_vf = topo->back_vf[sh.insertVf];
            gpuHookup(topo, q_f1, sh.insertVf);
            gpuHookup(topo, q_f0, r_vf);
            sh.tiSize = 3;
            computeTraversalInfoParsimony(topo, sh, q_f2, N, false);
        }
        newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, false, width);
        __syncwarp();
    }

    if (lane == 0)
    {
        topo->bestParsimony = sh.bestParsimony;
    }
    __syncwarp();

    // ── Phase 2: initial SPR hill-climbing ────────────────────────────────────
    if (sprDist <= 0)
    {
        return;
    }

    if (lane == 0) { gpuNodeRectifierPars(topo, sh, N); }
    __syncwarp();
    if (lane == 0)
    {
        sh.randomMPHits = 1;
        sh.randomMP = sh.bestParsimony;
        topo->preSprParsimony = sh.bestParsimony;
    }
    __syncwarp();

    gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);

    if (lane == 0)
    {
        topo->postSprParsimony = sh.randomMP;
        topo->bestParsimony = sh.randomMP;
        // Save RNG state and score for Opt-G2 two-kernel Phase 3
        topo->savedSeed = sh.seed;
        if (d_postSprScores)
        {
            d_postSprScores[k] = topo->postSprParsimony;
        }
    }
    __syncwarp();
}

// ─── Opt-G2: Phase 3 only kernel (two-kernel selective Phase 3) ───────────────
template <int STATES, int NTAXA>
__global__ void buildPhase3Kernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    unsigned int* __restrict__ d_siteWeights,
    unsigned int* __restrict__ d_ratchetScratch,
    int width,
    int sprDist,
    int numNNI,
    size_t parsVectPerTree,
    size_t parsScorePerTree,
    int pool_size,
    unsigned int* pool_scores,    // [pool_size] population pool scores (UINT_MAX=empty)
    int* pool_back_vf,            // [pool_size * kMaxVFaces] population pool topologies
    int* pool_filled,             // device ptr: number of filled slots
    int* pool_slot_locks,         // device ptr: per-slot spinlocks [pool_size]
    unsigned int* pool_hashes,    // [pool_size] topology hash per slot (0xFFFFFFFF=empty)
    unsigned int* global_best,    // device ptr: global best parsimony across all warps
    unsigned int* treels_scores,
    int*   treels_back_vf,
    int*   treels_filled,
    unsigned int* treels_cutoff,
    int    max_treels,
    unsigned int* treels_hashes,       // [max_treels] topology hash per slot; nullptr = disabled
    const unsigned int* d_cost_matrix, // nullptr = Fitch mode
    float save_margin,                 // relative SAVE A margin: -1=save all, r≥0 → mp < randomMP*(1+r)
    int treels_writers                 // only blockIdx.x < treels_writers write to treels
)
{
    __shared__ BuildSharedT<NTAXA> sh;
    using SharedT = BuildSharedT<NTAXA>;

    const int k = blockIdx.x;
    const int lane = threadIdx.x;

    GpuTopology* topo = d_topos + k;
    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    unsigned int* sw_k      = d_siteWeights    ? (d_siteWeights    + (size_t)k * width) : nullptr;
    unsigned int* ratchet_k = d_ratchetScratch ? (d_ratchetScratch + (size_t)k * width) : nullptr;
    const int N = topo->mxtips;

    // Restore inter-kernel state from GpuTopology
    if (lane == 0)
    {
        sh.seed = topo->savedSeed;
        sh.randomMP = topo->postSprParsimony;
        sh.randomMPHits = 1;
        sh.use_sankoff = (d_cost_matrix != nullptr);
        sh.site_weights = sh.use_sankoff ? sw_k : nullptr;
        sh.bestParsimony = topo->bestParsimony;
        sh.save_margin = save_margin;
        (void)k;
    }
    __syncwarp();

    runPhase3<STATES>(
        pars_tree, score_tree, topo, sh, sw_k, ratchet_k, N, sprDist, numNNI, lane, k, width,
        pool_size, pool_scores, pool_back_vf, pool_filled, pool_slot_locks, pool_hashes,
        global_best, treels_scores, treels_back_vf, treels_filled, treels_cutoff, max_treels,
        treels_hashes, treels_writers
    );
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long* seeds,
    int k1_count,
    int sprDist,
    int numNNI,
    int poolSize,
    cudaStream_t stream,
    AfterK1Callback after_k1,
    AfterK2Callback after_k2,
    int k2_workers,
    int max_outer_iters
)
{
    const int K = mem->K;
    const bool skip_k1 = (k1_count == 0);  // 0 = K2-only (bootstrap rounds)
    if (!skip_k1 && (k1_count < 0 || k1_count > K)) k1_count = K;
    if (k2_workers <= 0 || k2_workers > K) k2_workers = K;
    long* d_seeds = nullptr;
    if (!skip_k1)
    {
        CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)k1_count * sizeof(long)));
        CUDA_CHECK(cudaMemcpyAsync(
            d_seeds, seeds, (size_t)k1_count * sizeof(long), cudaMemcpyHostToDevice, stream
        ));
        CUDA_CHECK(cudaMemsetAsync(
            mem->d_parsScore, 0, (size_t)K * mem->parsScorePerTree * sizeof(unsigned int), stream
        ));
    }

    const int states = mem->states;
    const int mxtips = mem->mxtips;

    // Dispatch kernels by compile-time STATES × NTAXA (Opt-P Layer 3)
    // STATES: 2=binary, 4=DNA, 20=protein, 32=fallback
    // NTAXA buckets: ≤128, ≤256, ≤384, ≤512, ≤800 (=kMaxTaxa)  [GPU_NTAXA_TEMPLATE]
    auto launch = [&](auto states_tag, auto ntaxa_tag)
    {
        constexpr int S = decltype(states_tag)::value;
        constexpr int NT = decltype(ntaxa_tag)::value;
        const size_t sharedBytes = sizeof(BuildSharedT<NT>);

        if (!skip_k1)
        {
            printf("\n[GPU] --------------------------------------------\n");
            printf("[GPU]   buildTreesKernel<STATES=%d,NTAXA=%d>\n", S, NT);
            printf(
                "[GPU]         K1=%d  k2=%d  sprDist=%d  shared=%.1f KB\n", k1_count, k2_workers, sprDist,
                sharedBytes / 1024.0
            );

            // Same stack limit needed for K1 (buildParsimonyTreesKernel<20> also > 1024 bytes).
            size_t k1_prev_stack = 0;
            CUDA_CHECK(cudaDeviceGetLimit(&k1_prev_stack, cudaLimitStackSize));
            if (k1_prev_stack < 4096)
            {
                CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
            }

            cudaEvent_t k1_start, k1_end;
            cudaEventCreate(&k1_start);
            cudaEventCreate(&k1_end);
            cudaEventRecord(k1_start, stream);
            buildParsimonyTreesKernel<S, NT><<<dim3(k1_count), dim3(kWarpSize), 0, stream>>>(
                mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights, d_seeds,
                mem->width, sprDist, mem->d_postSprScores, mem->parsVectPerTree, mem->parsScorePerTree,
                mem->d_cost_matrix
            );
            cudaEventRecord(k1_end, stream);
            CUDA_CHECK(cudaGetLastError());

            if (after_k1)
            {
                after_k1(stream, mem);
            }
            else
            {
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            if (k1_prev_stack < 4096)
            {
                CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, k1_prev_stack));
            }

            float k1ms = 0.f;
            cudaEventElapsedTime(&k1ms, k1_start, k1_end);
            cudaEventDestroy(k1_start);
            cudaEventDestroy(k1_end);
            printf("[GPU]         time: %.3f s\n", k1ms / 1e3);
        }

        if (max_outer_iters == 0) return;  // bootstrap mode: K1-only, skip K2

        if (!skip_k1) {
            printf("\n[GPU] --------------------------------------------\n");
            printf("[GPU]   hillClimbingKernel<STATES=%d,NTAXA=%d>\n", S, NT);
        }

        // STATES=20 kernel has 1264 bytes cumulative stack (register spills force non-inlined
        // calls); default CUDA per-thread stack is 1024 bytes → overflow → illegal access.
        // Set 4096 bytes to cover the gap.  Restore after sync to avoid wasting memory.
        size_t prev_stack = 0;
        CUDA_CHECK(cudaDeviceGetLimit(&prev_stack, cudaLimitStackSize));
        if (prev_stack < 4096)
        {
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));
        }

        // In bootstrap mode (max_treels > 0): restrict treels writes to pool_size*2 workers.
        // Writers (blockIdx.x < treels_writers) use deterministic pool assignment
        // (pair 0,1→slot 0; pair 2,3→slot 1; …) and write to treels.
        // Non-writers still explore and update the pool, but skip treels writes.
        const int treels_writers = (mem->max_treels > 0)
            ? std::min(k2_workers, poolSize * 2)
            : k2_workers;

        // Launch K2 async — AfterK2Callback (if any) runs CPU work while K2 executes
        cudaEvent_t k2_start, k2_end;
        cudaEventCreate(&k2_start);
        cudaEventCreate(&k2_end);
        cudaEventRecord(k2_start, stream);
        buildPhase3Kernel<S, NT><<<dim3(k2_workers), dim3(kWarpSize), 0, stream>>>(
            mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights,
            mem->d_ratchetScratch, mem->width,
            sprDist, numNNI, mem->parsVectPerTree, mem->parsScorePerTree, poolSize,
            mem->d_poolScores, mem->d_poolBackVf, mem->d_poolFilled, mem->d_poolSlotLocks,
            mem->d_poolHashes, mem->d_globalBest,
            mem->d_treelsScores, mem->d_treelsBackVf, mem->d_treelsFilled,
            mem->d_treelsCutoff, mem->max_treels, mem->d_treelsHashes,
            mem->d_cost_matrix, mem->save_margin, treels_writers
        );
        cudaEventRecord(k2_end, stream);
        CUDA_CHECK(cudaGetLastError());

        if (after_k2)
        {
            // Phase 2: CPU hill-climbing while K2 runs.
            // Contract: callback must call cudaStreamSynchronize(stream) before returning.
            after_k2(stream);
        }
        else
        {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        if (prev_stack < 4096)
        {
            CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, prev_stack));
        }

        float k2ms = 0.f;
        cudaEventElapsedTime(&k2ms, k2_start, k2_end);
        cudaEventDestroy(k2_start);
        cudaEventDestroy(k2_end);
        if (!skip_k1)
            printf("[GPU]         time: %.3f s\n\n", k2ms / 1e3);
    };

    // Dispatch on NTAXA bucket (Opt-P L3) then STATES.
    // GPU_NTAXA_TEMPLATE=ON  → 5 buckets (128/256/384/512/800), slow build, full speedup.
    // GPU_NTAXA_TEMPLATE=OFF → 800 only, fast build, no NTAXA speedup.
    auto dispatch_ntaxa = [&](auto states_tag)
    {
#ifdef GPU_NTAXA_TEMPLATE
        if (mxtips <= 128)
            launch(states_tag, std::integral_constant<int, 128>{});
        else if (mxtips <= 256)
            launch(states_tag, std::integral_constant<int, 256>{});
        else if (mxtips <= 384)
            launch(states_tag, std::integral_constant<int, 384>{});
        else if (mxtips <= 512)
            launch(states_tag, std::integral_constant<int, 512>{});
        else
            launch(states_tag, std::integral_constant<int, 800>{});
#else
        launch(states_tag, std::integral_constant<int, 800>{});
#endif
    };

    // Only DNA (4) and protein (20) — covers all real datasets.
    // Binary (2) and 32-state fallback removed to halve build time.
    if (states == 20)
    {
        dispatch_ntaxa(std::integral_constant<int, 20>{});
    }
    else
    {
        dispatch_ntaxa(std::integral_constant<int, 4>{});
    }

    if (d_seeds) CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
