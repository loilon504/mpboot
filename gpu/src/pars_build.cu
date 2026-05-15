#include <algorithm>
#include <climits>
#include <cmath>
#include <functional>
#include <type_traits>
#include <vector>

#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/topo_helpers.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

__device__ __forceinline__ void gpuHookup(
    GpuTopology* t, int a, int b
)
{
    t->back_vf[a] = b;
    t->back_vf[b] = a;
}

// ─── testInsert ───────────────────────────────────────────────────────────────
template<int STATES, typename SharedT>
__device__ void testInsert(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int q,
    int N,
    int width,
    int lane
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
        gpuHookup(topo->back_vf, q, sh.bcast[0]);
        topo->back_vf[vfNextFace(p, N)] = -1;
        topo->back_vf[vfNnxtFace(p, N)] = -1;
    }
    __syncwarp();
}

// ─── doAddTraverse ────────────────────────────────────────────────────────────
template<int STATES, typename SharedT>
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
    int lane
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
                testInsert<STATES>(pars_tree, score_tree, topo, sh, p, cur_q, N, width, lane);
            }
        }

        if (q_num > N && maxt > 0)
        {
            if (lane == 0)
            {
                int qn  = topo->back_vf[vfNextFace(cur_q, N)];
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
template<typename SharedT>
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
template<int STATES, typename SharedT>
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
    createTiAndNewviewParsimony<SharedT, STATES>(pars_tree, score_tree, topo, sh, rm, N, width, lane);
    __syncwarp();
}

template<int STATES, typename SharedT>
__device__ void gpuSPRHillClimb(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int N,
    int sprDist,
    int width,
    int lane
)
{
    int* const back_vf = topo->back_vf;

    // Skip gpuNodeRectifierPars on the first do-while iteration:
    // nodep[] is guaranteed fresh by the caller (runPhase3 calls it before gpuSPRHillClimb).
    bool first_iter = true;
    unsigned int startMP;
    do
    {
        startMP = sh.randomMP;
        if (!first_iter)
        {
            if (lane == 0) gpuNodeRectifierPars(topo, sh, N);
            __syncwarp();
        }
        first_iter = false;

        for (int i = 1; i <= 2 * N - 2; i++)
        {
            if (lane == 0)
            {
                sh.bestParsimony = sh.randomMP;
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
                            sprDist, N, width, lane
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p1, N)], 1,
                            sprDist, N, width, lane
                        );
                    }
                    if (p2 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNextFace(p2, N)], 1,
                            sprDist, N, width, lane
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p2, N)], 1,
                            sprDist, N, width, lane
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
                            sprDist, N, width, lane
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q1, N)], 2,
                            sprDist, N, width, lane
                        );
                    }
                    if (q2 >= N)
                    {
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNextFace(q2, N)], 2,
                            sprDist, N, width, lane
                        );
                        doAddTraverse<STATES>(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q2, N)], 2,
                            sprDist, N, width, lane
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
        }  // end for i

    } while (sh.randomMP < startMP);
}

// ─── gpuRandomNNIs ───────────────────────────────────────────────────────────
template<typename SharedT>
__device__ void gpuRandomNNIs(
    GpuTopology* topo, SharedT& sh, int N, int numNNI, int lane
)
{
    if (lane != 0)
    {
        __syncwarp();
        return;
    }

    int num_edges = 0;
    for (int p_num = N + 1; p_num <= 2 * N - 1; p_num++)
    {
        int p_vf = nodepVf(p_num, N), q_vf = topo->back_vf[p_vf];
        int q_num = vfToNum(q_vf, N);
        if (q_num > N && p_num < q_num)
        {
            sh.stack[num_edges * 2] = p_vf;
            sh.stack[num_edges * 2 + 1] = q_vf;
            num_edges++;
        }
    }

    for (int i = num_edges - 1; i > 0; i--)
    {
        int j = (int)(gpuRandum(&sh.seed) * (i + 1));
        if (j != i)
        {
            int tp = sh.stack[i * 2], tq = sh.stack[i * 2 + 1];
            sh.stack[i * 2] = sh.stack[j * 2];
            sh.stack[i * 2 + 1] = sh.stack[j * 2 + 1];
            sh.stack[j * 2] = tp;
            sh.stack[j * 2 + 1] = tq;
        }
    }

    int bitset_words = (2 * N) / 32 + 1;
    for (int w = 0; w < bitset_words; w++)
    {
        sh.stackMaxt[w] = 0;
    }

    int applied = 0;
    for (int i = 0; i < num_edges && applied < numNNI; i++)
    {
        int p_vf = sh.stack[i * 2], q_vf = sh.stack[i * 2 + 1];
        int p_num = vfToNum(p_vf, N), q_num = vfToNum(q_vf, N);
        int pw = (p_num - 1) >> 5, pb = (p_num - 1) & 31;
        int qw = (q_num - 1) >> 5, qb = (q_num - 1) & 31;

        if (!(sh.stackMaxt[pw] & (1u << pb)) && !(sh.stackMaxt[qw] & (1u << qb)))
        {
            int pf0 = vfNnxtFace(p_vf, N), qf1 = vfNextFace(q_vf, N), qf0 = vfNnxtFace(q_vf, N);
            int b = topo->back_vf[pf0];
            if ((int)(gpuRandum(&sh.seed) * 2) == 0)
            {
                int c = topo->back_vf[qf1];
                gpuHookup(topo->back_vf, pf0, c);
                gpuHookup(topo->back_vf, qf1, b);
            }
            else
            {
                int d = topo->back_vf[qf0];
                gpuHookup(topo->back_vf, pf0, d);
                gpuHookup(topo->back_vf, qf0, b);
            }
            sh.stackMaxt[pw] |= (1u << pb);
            sh.stackMaxt[qw] |= (1u << qb);
            applied++;
        }
    }
    __syncwarp();
}

// ─── Opt-C: warp-parallel topology fingerprint (used for sprdist=3 stagnation detection) ──
__device__ __forceinline__ uint32_t computeTopoFingerprint(
    const GpuTopology* __restrict__ topo, int lane
)
{
    uint32_t h = 0u;
    for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
        h ^= (uint32_t)((vf + 1) * 2654435761u) ^ (uint32_t)((topo->back_vf[vf] + 2) * 2246822519u);
    for (int offset = 16; offset > 0; offset >>= 1)
        h ^= __shfl_xor_sync(0xffffffff, h, offset);
    return h;
}

// ─── Phase 3 device helper (shared by buildParsimonyTreesKernel + buildPhase3Kernel) ──
template<int STATES, typename SharedT>
__device__ void runPhase3(
    parsimonyNumber* pars_tree,
    unsigned int* score_tree,
    GpuTopology* topo,
    SharedT& sh,
    unsigned int* sw_k,
    int N,
    int sprDist,
    int numSearchIter,
    int numNNI,
    int stopNoImprove,
    int lane,
    int k,
    int width
)
{
    if (lane == 0 && k == 0)
    {
        sh.t_p3_nni_spr = sh.t_p3_ratchet = 0;
        sh.n_p3_even = sh.n_p3_odd = 0;
    }
    if (lane == 0)
    {
        sh.last_odd_hash = 0;
        sh.restore_on_next_odd = 0;
        sh.sym_do_ratchet = 0;  // Symmetric path starts with NNI
    }
    __syncwarp();

    int no_improve_count = 0;
    for (int outer = 0; outer < numSearchIter; outer++)
    {
        unsigned int best_before = 0;
        if (lane == 0) { best_before = topo->bestParsimony; }

        bool iter_is_nni;  // track which type ran this iteration (for counters)

        if (sprDist == 3)
        {
            // ── Opt-C path: even=NNI, odd=Ratchet with stagnation detection ──────
            iter_is_nni = (outer % 2 == 0);

            if (iter_is_nni)
            {
                gpuRandomNNIs<SharedT>(topo, sh, N, numNNI, lane);

                unsigned int pm = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm; sh.randomMPHits = 1; }
                __syncwarp();

                {
                    long long _t0 = 0LL;
                    if (k == 0 && lane == 0) _t0 = clock64();
                    gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);
                    if (k == 0 && lane == 0) { sh.t_p3_nni_spr += clock64() - _t0; sh.n_p3_even++; }
                }
            }
            else
            {
                // Opt-C: conditional restore when Ratchet stagnates
                if (sh.restore_on_next_odd)
                {
                    for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                        topo->back_vf[vf] = topo->best_back_vf[vf];
                    __syncwarp();
                }

                if (lane == 0)
                {
                    for (int b = 0; b < width; b++)
                        sw_k[b] = (gpuRandum(&sh.seed) < 0.5) ? 2u : 1u;
                    sh.site_weights = sw_k;
                }
                __syncwarp();

                unsigned int pm1 = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm1; sh.randomMPHits = 1; }
                __syncwarp();

                gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);

                if (lane == 0) { sh.site_weights = nullptr; }
                __syncwarp();

                unsigned int pm2 = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm2; sh.randomMPHits = 1; }
                __syncwarp();

                {
                    long long _t0 = 0LL;
                    if (k == 0 && lane == 0) _t0 = clock64();
                    gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);
                    if (k == 0 && lane == 0) { sh.t_p3_ratchet += clock64() - _t0; sh.n_p3_odd++; }
                }

                // Stagnation detection: compare post-Ratchet topology with previous odd
                {
                    uint32_t cur_hash = computeTopoFingerprint(topo, lane);
                    if (lane == 0)
                    {
                        sh.restore_on_next_odd = (cur_hash == sh.last_odd_hash) ? 1 : 0;
                        sh.last_odd_hash = cur_hash;
                    }
                    __syncwarp();
                }
            }
        }
        else
        {
            // ── Symmetric path: adaptive NNI/Ratchet switching ───────────────────
            // sym_do_ratchet: 0=NNI, 1=Ratchet+restore, 2=NNI+restore
            const int was_ratchet = sh.sym_do_ratchet;
            iter_is_nni = (was_ratchet != 1);

            if (iter_is_nni)
            {
                // NNI+restore (was_ratchet==2): restore best before NNI
                if (was_ratchet == 2)
                {
                    for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                        topo->back_vf[vf] = topo->best_back_vf[vf];
                    __syncwarp();
                }

                gpuRandomNNIs<SharedT>(topo, sh, N, numNNI, lane);

                unsigned int pm = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm; sh.randomMPHits = 1; }
                __syncwarp();

                {
                    long long _t0 = 0LL;
                    if (k == 0 && lane == 0) _t0 = clock64();
                    gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);
                    if (k == 0 && lane == 0) { sh.t_p3_nni_spr += clock64() - _t0; sh.n_p3_even++; }
                }
            }
            else  // was_ratchet == 1: Ratchet, always restore best first
            {
                for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
                    topo->back_vf[vf] = topo->best_back_vf[vf];
                __syncwarp();

                if (lane == 0)
                {
                    for (int b = 0; b < width; b++)
                        sw_k[b] = (gpuRandum(&sh.seed) < 0.5) ? 2u : 1u;
                    sh.site_weights = sw_k;
                }
                __syncwarp();

                unsigned int pm1 = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm1; sh.randomMPHits = 1; }
                __syncwarp();

                gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);

                if (lane == 0) { sh.site_weights = nullptr; }
                __syncwarp();

                unsigned int pm2 = createTiAndEvaluateParsimony<SharedT, STATES>(
                    pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width
                );
                if (lane == 0) { sh.randomMP = pm2; sh.randomMPHits = 1; }
                __syncwarp();

                {
                    long long _t0 = 0LL;
                    if (k == 0 && lane == 0) _t0 = clock64();
                    gpuSPRHillClimb<STATES>(pars_tree, score_tree, topo, sh, N, sprDist, width, lane);
                    if (k == 0 && lane == 0) { sh.t_p3_ratchet += clock64() - _t0; sh.n_p3_odd++; }
                }
            }

            // Symmetric switching: stay if improved, switch if failed
            if (lane == 0)
            {
                const bool improved = (sh.randomMP < best_before);
                if (iter_is_nni)
                    sh.sym_do_ratchet = improved ? 0 : 1;
                else
                    sh.sym_do_ratchet = improved ? 1 : 2;
            }
            __syncwarp();
        }

        if (lane == 0)
        {
            if (iter_is_nni) topo->n_total_even++;
            else              topo->n_total_odd++;

            if (sh.randomMP < topo->bestParsimony)
            {
                topo->bestParsimony = sh.randomMP;
                if (iter_is_nni) topo->n_improved_even++;
                else              topo->n_improved_odd++;
                for (int vf = 0; vf < topo->num_vfaces; vf++)
                    topo->best_back_vf[vf] = topo->back_vf[vf];
            }

            if (sh.randomMP >= best_before) no_improve_count++;
            else                             no_improve_count = 0;
            sh.bcast[0] = (stopNoImprove > 0 && no_improve_count >= stopNoImprove) ? 1 : 0;
        }
        __syncwarp();
        if (sh.bcast[0]) break;
    }
}

// ─── Kernel ───────────────────────────────────────────────────────────────────
template<int STATES, int NTAXA>
__global__ void buildParsimonyTreesKernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    unsigned int* __restrict__ d_siteWeights,
    const long* __restrict__ d_seeds,
    int width,
    int sprDist,
    int numSearchIter,
    int numNNI,
    int stopNoImprove,
    unsigned int* d_postSprScores,  // Opt-G2: write postSprParsimony[k] here; nullptr=disabled
    size_t parsVectPerTree,
    size_t parsScorePerTree
)
{
    __shared__ BuildSharedT<NTAXA> sh;
    using SharedT = BuildSharedT<NTAXA>;

    const int k = blockIdx.x;
    const int lane = threadIdx.x;

    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    GpuTopology* topo = d_topos + k;
    unsigned int* sw_k = d_siteWeights + (size_t)k * width;
    const int N = topo->mxtips;

    // ── Phase 0: init timing + permutation + 3-tip tree ──────────────────────
    if (lane == 0)
    {
        sh.seed = d_seeds[k];
        sh.site_weights = nullptr;
        sh.t_build = sh.t_phase2 = sh.t_p3_nni_spr = sh.t_p3_ratchet = 0;
        sh.n_p3_even = sh.n_p3_odd = 0;

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

        // Init nodep[] — inner nodes overwritten by gpuNodeRectifierPars in Phase 3
        for (int num = 1; num <= N; num++)
        {
            topo->nodep[num] = num - 1;
        }
        for (int num = N + 1; num <= 2 * N - 1; num++)
        {
            int p = nodepVf(num, N);
            topo->nodep[num] = p;
            topo->xpars[p] = 1;
            topo->xpars[p - 1] = 0;
            topo->xpars[p - 2] = 0;
        }
    }
    __syncwarp();

    createTiAndNewviewParsimony<SharedT, STATES>(pars_tree, score_tree, topo, sh, N + 2, N, width, lane);
    __syncwarp();

    // ── Phase 1: stepwise addition ────────────────────────────────────────────
    long long _t_build = 0LL;
    if (k == 0 && lane == 0)
    {
        _t_build = clock64();
    }

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
            unsigned int mp = newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, true, width);
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
        for (int vf = 0; vf < topo->num_vfaces; vf++)
        {
            topo->best_back_vf[vf] = topo->back_vf[vf];
        }
        if (k == 0)
        {
            sh.t_build = clock64() - _t_build;
        }
    }
    __syncwarp();

    // ── Phase 2: initial SPR hill-climbing ────────────────────────────────────
    if (sprDist <= 0)
    {
        return;
    }

    long long _t_phase2 = 0LL;
    if (k == 0 && lane == 0)
    {
        _t_phase2 = clock64();
    }

    if (lane == 0)
    {
        gpuNodeRectifierPars(topo, sh, N);
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
        for (int vf = 0; vf < topo->num_vfaces; vf++)
        {
            topo->best_back_vf[vf] = topo->back_vf[vf];
        }
        if (k == 0)
        {
            sh.t_phase2 = clock64() - _t_phase2;
        }
        // Save RNG state and score for Opt-G2 two-kernel Phase 3
        topo->savedSeed = sh.seed;
        if (d_postSprScores)
        {
            d_postSprScores[k] = topo->postSprParsimony;
        }
    }
    __syncwarp();

    // ── Phase 3: iterative NNI + SPR×1 (even) / ratchet + SPR×2 (odd) ────────
    if (numSearchIter <= 0)
    {
        return;
    }

    runPhase3<STATES>(
        pars_tree, score_tree, topo, sh, sw_k, N, sprDist, numSearchIter, numNNI, stopNoImprove,
        lane, k, width
    );

    // Print timing for block 0
    if (k == 0 && lane == 0)
    {
        printf(
            "[TIMING] Phase1 build=%lld cyc\n"
            "[TIMING] Phase2 initial-SPR=%lld cyc\n"
            "[TIMING] Phase3 NNI+SPR (%d iters): spr=%lld cyc\n"
            "[TIMING] Phase3 Ratchet  (%d iters): total=%lld cyc\n",
            sh.t_build, sh.t_phase2,
            sh.n_p3_even, sh.t_p3_nni_spr,
            sh.n_p3_odd, sh.t_p3_ratchet
        );
    }
}

// ─── Opt-G2: Phase 3 only kernel (two-kernel selective Phase 3) ───────────────
template<int STATES, int NTAXA>
__global__ void buildPhase3Kernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    unsigned int* __restrict__ d_siteWeights,
    int width,
    int sprDist,
    int numSearchIter,
    int numNNI,
    int stopNoImprove,
    unsigned int phase3Threshold,  // absolute threshold: skip if postSprParsimony > threshold
    size_t parsVectPerTree,
    size_t parsScorePerTree
)
{
    __shared__ BuildSharedT<NTAXA> sh;
    using SharedT = BuildSharedT<NTAXA>;

    const int k = blockIdx.x;
    const int lane = threadIdx.x;

    GpuTopology* topo = d_topos + k;
    parsimonyNumber* pars_tree = d_parsVect + (size_t)k * parsVectPerTree;
    unsigned int* score_tree = d_parsScore + (size_t)k * parsScorePerTree;
    unsigned int* sw_k = d_siteWeights + (size_t)k * width;
    const int N = topo->mxtips;

    // Exact percentile skip (Opt-G2)
    if (topo->postSprParsimony > phase3Threshold)
    {
        return;
    }

    // Restore inter-kernel state from GpuTopology
    if (lane == 0)
    {
        sh.seed = topo->savedSeed;
        sh.randomMP = topo->postSprParsimony;
        sh.randomMPHits = 1;
        sh.site_weights = nullptr;
        sh.bestParsimony = topo->bestParsimony;
        // Zero timing accumulators (printed by runPhase3 for k=0)
        sh.t_build = 0;
        sh.t_phase2 = 0;
    }
    __syncwarp();

    // Hybrid: CPU-uploaded tree has stale parsVect and uninitialized nodep[].
    // Must: (1) init tip nodep[], (2) recompute parsVect from topology, (3) set inner nodep[].
    if (topo->needs_recompute)
    {
        // Step 1: tip nodep[] is always nodep[num] = num-1 (fixed for any tree with N tips)
        if (lane == 0)
        {
            for (int num = 1; num <= N; num++)
                topo->nodep[num] = num - 1;
        }
        __syncwarp();

        // Step 2: recompute parsVect from the CPU tree's back_vf topology (full traversal)
        unsigned int recomputed = createTiAndEvaluateParsimony<SharedT, STATES>(
            pars_tree, score_tree, topo, sh, topo->start_vface, N, /*full=*/true, width
        );

        // Step 3: set inner nodep[] so Phase 3 SPR loop can iterate topo->nodep[i]
        if (lane == 0)
        {
            gpuNodeRectifierPars(topo, sh, N);  // DFS from nodep[1]=0, sets nodep[N+1..2N-1]
            sh.randomMP      = recomputed;
            sh.randomMPHits  = 1;
            sh.bestParsimony = recomputed;
            topo->bestParsimony    = recomputed;
            topo->postSprParsimony = recomputed;
            topo->needs_recompute  = 0;
        }
        __syncwarp();
    }

    runPhase3<STATES>(
        pars_tree, score_tree, topo, sh, sw_k, N, sprDist, numSearchIter, numNNI, stopNoImprove,
        lane, k, width
    );
    // Print timing for block 0
    if (k == 0 && lane == 0)
    {
        printf(
            "[TIMING] Phase1 build=%lld cyc\n"
            "[TIMING] Phase2 initial-SPR=%lld cyc\n"
            "[TIMING] Phase3 NNI+SPR (%d iters): spr=%lld cyc\n"
            "[TIMING] Phase3 Ratchet  (%d iters): total=%lld cyc\n",
            sh.t_build, sh.t_phase2,
            sh.n_p3_even, sh.t_p3_nni_spr,
            sh.n_p3_odd, sh.t_p3_ratchet
        );
    }
}

// ─── Host wrapper ─────────────────────────────────────────────────────────────
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long* seeds,
    int sprDist,
    int numSearchIter,
    int numNNI,
    int stopNoImprove,
    float topPct,
    cudaStream_t stream,
    AfterK1Callback after_k1
)
{
    const int K = mem->K;
    long* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)K * sizeof(long)));
    CUDA_CHECK(
        cudaMemcpyAsync(d_seeds, seeds, (size_t)K * sizeof(long), cudaMemcpyHostToDevice, stream)
    );
    CUDA_CHECK(cudaMemsetAsync(
        mem->d_parsScore, 0, (size_t)K * mem->parsScorePerTree * sizeof(unsigned int), stream
    ));

    const int states = mem->states;
    const int mxtips = mem->mxtips;

    // Dispatch kernels by compile-time STATES × NTAXA (Opt-P Layer 3)
    // STATES: 2=binary, 4=DNA, 20=protein, 32=fallback
    // NTAXA buckets: ≤128, ≤256, ≤384, ≤512, ≤800 (=kMaxTaxa)
    auto launch = [&](auto states_tag, auto ntaxa_tag) {
        constexpr int S  = decltype(states_tag)::value;
        constexpr int NT = decltype(ntaxa_tag)::value;
        const size_t sharedBytes = sizeof(BuildSharedT<NT>);

        // Helper: time a single kernel launch with CUDA events
        auto ev_ms = [&](auto fn) -> float {
            cudaEvent_t ea, eb;
            cudaEventCreate(&ea); cudaEventCreate(&eb);
            cudaEventRecord(ea, stream);
            fn();
            cudaEventRecord(eb, stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float ms = 0.f;
            cudaEventElapsedTime(&ms, ea, eb);
            cudaEventDestroy(ea); cudaEventDestroy(eb);
            return ms;
        };

        if (topPct > 0.0f)
        {
            // ── Opt-G2: Two-kernel mode ───────────────────────────────────────
            printf("[GPU]   buildTreesKernel<STATES=%d,NTAXA=%d>\n", S, NT);
            printf("[GPU]         K=%d  sprDist=%d  top_pct=%.0f%%  shared=%.1f KB\n",
                   K, sprDist, topPct * 100.0f, sharedBytes / 1024.0);

            // Launch K1 async — do NOT sync here; callback (if any) does cudaStreamQuery loop
            cudaEvent_t k1_start, k1_end;
            cudaEventCreate(&k1_start); cudaEventCreate(&k1_end);
            cudaEventRecord(k1_start, stream);
            buildParsimonyTreesKernel<S, NT><<<dim3(K), dim3(kWarpSize), 0, stream>>>(
                mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights, d_seeds,
                mem->width, sprDist,
                /*numSearchIter=*/0, numNNI, stopNoImprove,
                mem->d_postSprScores,
                mem->parsVectPerTree, mem->parsScorePerTree
            );
            cudaEventRecord(k1_end, stream);
            CUDA_CHECK(cudaGetLastError());

            unsigned int threshold;
            if (after_k1)
            {
                // Hybrid: callback builds CPU trees via cudaStreamQuery, syncs, merges, uploads
                threshold = after_k1(stream, mem);
                // K1 is now fully done (callback called cudaStreamSynchronize)
            }
            else
            {
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<unsigned int> scores(K);
                CUDA_CHECK(cudaMemcpy(
                    scores.data(), mem->d_postSprScores, (size_t)K * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost
                ));
                std::sort(scores.begin(), scores.end());
                int top_k = max(1, (int)ceil((double)K * topPct));
                threshold = scores[top_k - 1];
            }

            // K1 timing (event already recorded; K1 is done by now)
            float k1ms = 0.f;
            cudaEventElapsedTime(&k1ms, k1_start, k1_end);
            cudaEventDestroy(k1_start); cudaEventDestroy(k1_end);
            printf("[GPU]         time: %.3f s\n", k1ms / 1e3);

            // Download scores for stats (K1 already synced)
            std::vector<unsigned int> scores(K);
            CUDA_CHECK(cudaMemcpy(
                scores.data(), mem->d_postSprScores, (size_t)K * sizeof(unsigned int),
                cudaMemcpyDeviceToHost
            ));
            std::sort(scores.begin(), scores.end());

            int actual_phase3 = 0;
            for (int i = 0; i < K; i++)
                if (scores[i] <= threshold) actual_phase3++;

            printf("[GPU]   hillClimbingKernel<STATES=%d,NTAXA=%d>\n", S, NT);
            printf("[GPU]         top_k=%d/%d  threshold=%u  actual=%d (%.1f%%)"
                   "  score_range=[%u, %u]\n",
                   max(1, (int)ceil((double)K * topPct)), K, threshold,
                   actual_phase3, 100.0 * actual_phase3 / K,
                   scores[0], scores[K - 1]);

            float k2ms = ev_ms([&] {
                buildPhase3Kernel<S, NT><<<dim3(K), dim3(kWarpSize), 0, stream>>>(
                    mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights,
                    mem->width, sprDist, numSearchIter, numNNI, stopNoImprove, threshold,
                    mem->parsVectPerTree, mem->parsScorePerTree
                );
            });
            printf("[GPU]         time: %.3f s\n", k2ms / 1e3);
        }
        else
        {
            // ── Single-kernel mode ────────────────────────────────────────────
            printf("[GPU]   buildTreesKernel<STATES=%d,NTAXA=%d>\n", S, NT);
            printf("[GPU]         K=%d  sprDist=%d  iters=%d  NNI=%d  stop=%d  shared=%.1f KB\n",
                   K, sprDist, numSearchIter, numNNI, stopNoImprove, sharedBytes / 1024.0);

            float k1ms = ev_ms([&] {
                buildParsimonyTreesKernel<S, NT><<<dim3(K), dim3(kWarpSize), 0, stream>>>(
                    mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights, d_seeds,
                    mem->width, sprDist, numSearchIter, numNNI, stopNoImprove,
                    /*d_postSprScores=*/nullptr,
                    mem->parsVectPerTree, mem->parsScorePerTree
                );
            });
            printf("[GPU]         time: %.3f s\n", k1ms / 1e3);
        }
    };

    // Dispatch on NTAXA bucket (Opt-P L3) then STATES.
    // GPU_NTAXA_TEMPLATE=ON  → 4 buckets (128/256/512/800), slow build, full speedup.
    // GPU_NTAXA_TEMPLATE=OFF → 800 only, fast build, no NTAXA speedup.
    auto dispatch_ntaxa = [&](auto states_tag) {
#ifdef GPU_NTAXA_TEMPLATE
        if      (mxtips <= 128) launch(states_tag, std::integral_constant<int,128>{});
        else if (mxtips <= 256) launch(states_tag, std::integral_constant<int,256>{});
        else if (mxtips <= 512) launch(states_tag, std::integral_constant<int,512>{});
        else                    launch(states_tag, std::integral_constant<int,800>{});
#else
        launch(states_tag, std::integral_constant<int,800>{});
#endif
    };

    // Only DNA (4) and protein (20) — covers all real datasets.
    // Binary (2) and 32-state fallback removed to halve build time.
    if (states == 20) dispatch_ntaxa(std::integral_constant<int,20>{});
    else              dispatch_ntaxa(std::integral_constant<int, 4>{});

    CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
