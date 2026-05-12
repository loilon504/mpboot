#include <climits>

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
    int lane,
    bool log = false
)
{
    if (lane == 0)
    {
        sh.bcast[0] = topo->back_vf[q];
        gpuHookup(topo->back_vf, vfNextFace(p, N), q);
        gpuHookup(topo->back_vf, vfNnxtFace(p, N), sh.bcast[0]);
    }
    __syncwarp();

    long long _t0 = 0LL, _t1 = 0LL;
    if (log) _t0 = clock64();

    // Pre-refresh: refresh q, r, and tip_p subtrees WITHOUT computing parsVect[p] from face[2].
    // face0's children (for step 2) are tip_p and q — both must be fresh (xpars=1).
    // Without tip_p refresh, it degrades to xpars=0 across iterations → extra eval traversal.
    if (lane == 0)
    {
        const int r_vf     = sh.bcast[0];
        const int tip_p_vf = topo->back_vf[p];   // back of remove node, unchanged by hookup
        sh.tiSize = 3;
        if (q >= N && !topo->xpars[q])
            computeTraversalInfoParsimony(topo, sh, q, N, false);
        if (r_vf >= N && !topo->xpars[r_vf])
            computeTraversalInfoParsimony(topo, sh, r_vf, N, false);
        if (tip_p_vf >= N && !topo->xpars[tip_p_vf])
            computeTraversalInfoParsimony(topo, sh, tip_p_vf, N, false);
    }
    __syncwarp();
    if (sh.tiSize > 3)
        newviewParsimony(pars_tree, score_tree, sh, false, width, states);
    if (lane == 0)
    {
        if (q >= N)             topo->xpars[q]           = 1;
        if (sh.bcast[0] >= N)   topo->xpars[sh.bcast[0]] = 1;
        const int tip_p_vf = topo->back_vf[p];
        if (tip_p_vf >= N)      topo->xpars[tip_p_vf]   = 1;
        topo->xpars[vfNnxtFace(p, N)] = 0;  // force step 2 to recompute p from face0
    }
    __syncwarp();

    if (log)
    {
        _t1 = clock64();
        sh.t_ti_newview += _t1 - _t0;
        sh.n_ti_newview_size += (sh.tiSize - 3) / 3;
    }

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
                computeTraversalInfoParsimony(topo, sh, face0_ev, N, false);
            if (r_vf_ev >= N && !topo->xpars[r_vf_ev])
                computeTraversalInfoParsimony(topo, sh, r_vf_ev, N, false);
        }
        __syncwarp();
        mp = newviewParsimony(pars_tree, score_tree, sh, true, width, states);
    }

    if (log)
    {
        sh.t_ti_eval += clock64() - _t1;
        sh.n_ti_eval_size += (sh.tiSize - 3) / 3;
        sh.n_testInsert++;
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
    int lane,
    bool log = false
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
            testInsert(pars_tree, score_tree, topo, sh, p, cur_q, N, width, states, lane, log);
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
__device__ void gpuNodeRectifierPars(
    GpuTopology* topo, BuildShared& sh, int N
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
        int pn = vfNextFace(rm, N), pnn = vfNnxtFace(rm, N);
        int p1 = topo->back_vf[pn], p2 = topo->back_vf[pnn];
        gpuHookup(topo->back_vf, p1, p2);
        topo->back_vf[pn] = -1;
        topo->back_vf[pnn] = -1;

        int r = topo->back_vf[ins];
        gpuHookup(topo->back_vf, pn, ins);
        gpuHookup(topo->back_vf, pnn, r);
    }
    __syncwarp();
    createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, rm, N, width, states, lane);
    __syncwarp();
}

__device__ void gpuSPRHillClimb(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    BuildShared& sh,
    int N,
    int sprDist,
    int width,
    int states,
    int lane,
    int maxDoWhile = INT_MAX,
    bool do_timing = false
)
{
    int* const back_vf = topo->back_vf;
    const bool log = do_timing && (blockIdx.x == 0) && (lane == 0);

    unsigned int startMP;
    int dowhile_count = 0;
    do
    {
        startMP = sh.randomMP;
        dowhile_count++;
        if (lane == 0)
        {
            gpuNodeRectifierPars(topo, sh, N);
        }
        __syncwarp();

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

            long long t0 = log ? clock64() : 0;
            createTiAndEvaluateParsimony(
                pars_tree, score_tree, topo, sh, p, N, false, width, states
            );
            if (log)
            {
                sh.t_line2291 += clock64() - t0;
            }

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

                    long long t1 = log ? clock64() : 0;
                    if (p1 >= N)
                    {
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNextFace(p1, N)], 1,
                            sprDist, N, width, states, lane, log
                        );
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p1, N)], 1,
                            sprDist, N, width, states, lane, log
                        );
                    }
                    if (p2 >= N)
                    {
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNextFace(p2, N)], 1,
                            sprDist, N, width, states, lane, log
                        );
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, p, back_vf[vfNnxtFace(p2, N)], 1,
                            sprDist, N, width, states, lane, log
                        );
                    }
                    if (log)
                    {
                        sh.t_search += clock64() - t1;
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(p, N), p1);
                        gpuHookup(back_vf, vfNnxtFace(p, N), p2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony(
                        pars_tree, score_tree, topo, sh, p, N, width, states, lane
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

                    long long t2 = log ? clock64() : 0;
                    if (q1 >= N)
                    {
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNextFace(q1, N)], 2,
                            sprDist, N, width, states, lane, log
                        );
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q1, N)], 2,
                            sprDist, N, width, states, lane, log
                        );
                    }
                    if (q2 >= N)
                    {
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNextFace(q2, N)], 2,
                            sprDist, N, width, states, lane, log
                        );
                        doAddTraverse(
                            pars_tree, score_tree, topo, sh, q, back_vf[vfNnxtFace(q2, N)], 2,
                            sprDist, N, width, states, lane, log
                        );
                    }
                    if (log)
                    {
                        sh.t_search += clock64() - t2;
                    }

                    if (lane == 0)
                    {
                        gpuHookup(back_vf, vfNextFace(q, N), q1);
                        gpuHookup(back_vf, vfNnxtFace(q, N), q2);
                    }
                    __syncwarp();
                    createTiAndNewviewParsimony(
                        pars_tree, score_tree, topo, sh, q, N, width, states, lane
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
                long long t3 = log ? clock64() : 0;
                applyMove(
                    pars_tree, score_tree, topo, sh, sh.bcast[9], sh.bcast[10], N, width, states,
                    lane
                );
                // Update nodep[] for new topology (CPU: nodeRectifierPars after each SPR move)
                // if (lane == 0)
                //     gpuNodeRectifierPars(topo, sh, N);
                __syncwarp();
                if (lane == 0)
                {
                    sh.randomMP = sh.bestParsimony;
                    if (log)
                    {
                        sh.t_apply += clock64() - t3;
                        sh.n_apply++;
                    }
                }
                __syncwarp();
            }
        }  // end for i

    } while (sh.randomMP < startMP && dowhile_count < maxDoWhile);
    if (log)
    {
        sh.n_dowhile = dowhile_count;
    }
}

// ─── gpuRandomNNIs ───────────────────────────────────────────────────────────
__device__ void gpuRandomNNIs(
    GpuTopology* topo, BuildShared& sh, int N, int numNNI, int lane
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

// ─── Kernel ───────────────────────────────────────────────────────────────────
__global__ void buildParsimonyTreesKernel(
    parsimonyNumber* __restrict__ d_parsVect,
    unsigned int* __restrict__ d_parsScore,
    GpuTopology* d_topos,
    unsigned int* __restrict__ d_siteWeights,
    const long* __restrict__ d_seeds,
    int width,
    int states,
    int sprDist,
    int numSearchIter,
    int numNNI,
    int maxDoWhile,  // Opt 3: cap do-while iterations per SPR call
    size_t parsVectPerTree,
    size_t parsScorePerTree
)
{
    extern __shared__ BuildShared sh_arr[];
    BuildShared& sh = sh_arr[0];

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
        sh.t_build = sh.t_phase2 = sh.t_p3_nni = sh.t_p3_nni_spr = sh.t_p3_ratchet = 0;
        sh.n_p3_even = sh.n_p3_odd = 0;
        sh.t_line2291 = 0;
        sh.t_search = 0;
        sh.t_apply = 0;
        sh.n_apply = 0;
        sh.n_dowhile = 0;
        sh.t_ti_newview = sh.t_ti_eval = 0;
        sh.n_testInsert = sh.n_ti_newview_size = sh.n_ti_eval_size = 0;

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

    createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, N + 2, N, width, states, lane);
    __syncwarp();

    // ── Phase 1: stepwise addition ────────────────────────────────────────────
    long long _t_build = 0LL;
    if (k == 0 && lane == 0) _t_build = clock64();

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
            unsigned int mp = newviewParsimony(pars_tree, score_tree, sh, true, width, states);
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
        newviewParsimony(pars_tree, score_tree, sh, false, width, states);
        __syncwarp();
    }

    if (lane == 0)
    {
        topo->bestParsimony = sh.bestParsimony;
        for (int vf = 0; vf < topo->num_vfaces; vf++)
            topo->best_back_vf[vf] = topo->back_vf[vf];
        if (k == 0) sh.t_build = clock64() - _t_build;
    }
    __syncwarp();

    // ── Phase 2: initial SPR hill-climbing ────────────────────────────────────
    if (sprDist <= 0)
    {
        return;
    }

    long long _t_phase2 = 0LL;
    if (k == 0 && lane == 0) _t_phase2 = clock64();

    if (lane == 0)
    {
        gpuNodeRectifierPars(topo, sh, N);
        sh.randomMPHits = 1;
        sh.randomMP = sh.bestParsimony;
        topo->preSprParsimony = sh.bestParsimony;
    }
    __syncwarp();

    gpuSPRHillClimb(pars_tree, score_tree, topo, sh, N, sprDist, width, states, lane);

    if (lane == 0)
    {
        topo->postSprParsimony = sh.randomMP;
        topo->bestParsimony = sh.randomMP;
        for (int vf = 0; vf < topo->num_vfaces; vf++)
            topo->best_back_vf[vf] = topo->back_vf[vf];
        if (k == 0) sh.t_phase2 = clock64() - _t_phase2;
    }
    __syncwarp();

    // ── Phase 3: iterative NNI + SPR×1 (even) / ratchet + SPR×2 (odd) ────────
    if (numSearchIter <= 0)
    {
        return;
    }

    // Reset Phase 3 timing accumulators
    if (lane == 0 && k == 0)
    {
        sh.t_p3_nni = sh.t_p3_nni_spr = sh.t_p3_ratchet = 0;
        sh.n_p3_even = sh.n_p3_odd = 0;
        sh.t_line2291 = sh.t_search = sh.t_apply = sh.n_apply = sh.n_dowhile = 0;
        sh.t_ti_newview = sh.t_ti_eval = 0;
        sh.n_testInsert = sh.n_ti_newview_size = sh.n_ti_eval_size = 0;
    }
    __syncwarp();

    for (int outer = 0; outer < numSearchIter; outer++)
    {
        if (outer % 2 == 0)
        {
            // ── NNI perturbation + SPR ────────────────────────────────────────
            long long _t0 = 0LL;
            if (k == 0 && lane == 0) _t0 = clock64();

            gpuRandomNNIs(topo, sh, N, numNNI, lane);

            if (lane == 0)
            {
                gpuNodeRectifierPars(topo, sh, N);
            }
            __syncwarp();
            unsigned int pm = createTiAndEvaluateParsimony(
                pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width, states
            );
            if (lane == 0)
            {
                sh.randomMP = pm;
                sh.randomMPHits = 1;
            }
            __syncwarp();

            long long _t1 = 0LL;
            if (k == 0 && lane == 0) { sh.t_p3_nni += (_t1 = clock64()) - _t0; }

            gpuSPRHillClimb(
                pars_tree, score_tree, topo, sh, N, sprDist, width, states, lane, maxDoWhile,
                /*timing=*/(k == 0)
            );

            if (k == 0 && lane == 0)
            {
                sh.t_p3_nni_spr += clock64() - _t1;
                sh.n_p3_even++;
            }
        }
        else
        {
            // ── Ratchet: weighted SPR (hclimb1) + normal SPR (hclimb2) ───────
            long long _t0 = 0LL;
            if (k == 0 && lane == 0) _t0 = clock64();

            if (lane == 0)
            {
                for (int b = 0; b < width; b++)
                {
                    sw_k[b] = (gpuRandum(&sh.seed) < 0.5) ? 2u : 1u;
                }
                sh.site_weights = sw_k;
            }
            __syncwarp();

            unsigned int pm1 = createTiAndEvaluateParsimony(
                pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width, states
            );
            if (lane == 0)
            {
                sh.randomMP = pm1;
                sh.randomMPHits = 1;
            }
            __syncwarp();

            gpuSPRHillClimb(
                pars_tree, score_tree, topo, sh, N, sprDist, width, states, lane, maxDoWhile, false
            );

            if (lane == 0)
            {
                sh.site_weights = nullptr;
            }
            __syncwarp();

            unsigned int pm2 = createTiAndEvaluateParsimony(
                pars_tree, score_tree, topo, sh, topo->start_vface, N, true, width, states
            );
            if (lane == 0)
            {
                sh.randomMP = pm2;
                sh.randomMPHits = 1;
            }
            __syncwarp();

            gpuSPRHillClimb(
                pars_tree, score_tree, topo, sh, N, sprDist, width, states, lane, maxDoWhile, false
            );

            if (k == 0 && lane == 0)
            {
                sh.t_p3_ratchet += clock64() - _t0;
                sh.n_p3_odd++;
            }
        }

        if (lane == 0 && sh.randomMP < topo->bestParsimony)
        {
            topo->bestParsimony = sh.randomMP;
            for (int vf = 0; vf < topo->num_vfaces; vf++)
                topo->best_back_vf[vf] = topo->back_vf[vf];
        }
        __syncwarp();
    }

    // Print timing for block 0
    if (k == 0 && lane == 0)
    {
        long long t_spr_total = sh.t_line2291 + sh.t_search + sh.t_apply;
        long long t_ti_total = sh.t_ti_newview + sh.t_ti_eval;
        int n = sh.n_testInsert;
        printf(
            "[TIMING k=0] Phase1 build=%lld cyc\n"
            "[TIMING k=0] Phase2 initial-SPR=%lld cyc\n"
            "[TIMING k=0] Phase3 NNI+SPR (%d iters): nni_setup=%lld cyc  spr=%lld cyc\n"
            "[TIMING k=0] Phase3 Ratchet  (%d iters): total=%lld cyc\n"
            "[TIMING k=0] Phase3 SPR breakdown: line2291=%lld (%.1f%%)  search=%lld (%.1f%%)"
            "  apply=%lld (%.1f%%)  moves=%d  dowhile=%d\n"
            "[TIMING k=0] testInsert (%d calls): newview=%lld (%.1f%%)"
            "  eval=%lld (%.1f%%)  total_ti=%lld vs search=%lld\n"
            "[TIMING k=0] testInsert avg traversal: newview=%.2f nodes  eval=%.2f nodes\n",
            sh.t_build,
            sh.t_phase2,
            sh.n_p3_even, sh.t_p3_nni, sh.t_p3_nni_spr,
            sh.n_p3_odd, sh.t_p3_ratchet,
            sh.t_line2291, t_spr_total > 0 ? 100.0 * sh.t_line2291 / t_spr_total : 0.0,
            sh.t_search,  t_spr_total > 0 ? 100.0 * sh.t_search  / t_spr_total : 0.0,
            sh.t_apply,   t_spr_total > 0 ? 100.0 * sh.t_apply   / t_spr_total : 0.0,
            sh.n_apply, sh.n_dowhile,
            n,
            sh.t_ti_newview, t_ti_total > 0 ? 100.0 * sh.t_ti_newview / t_ti_total : 0.0,
            sh.t_ti_eval,    t_ti_total > 0 ? 100.0 * sh.t_ti_eval    / t_ti_total : 0.0,
            t_ti_total, sh.t_search,
            n > 0 ? (float)sh.n_ti_newview_size / n : 0.0f,
            n > 0 ? (float)sh.n_ti_eval_size    / n : 0.0f
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
    int maxDoWhile,
    cudaStream_t stream
)
{
    long* d_seeds = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)mem->K * sizeof(long)));
    CUDA_CHECK(cudaMemcpyAsync(
        d_seeds, seeds, (size_t)mem->K * sizeof(long), cudaMemcpyHostToDevice, stream
    ));
    CUDA_CHECK(cudaMemsetAsync(
        mem->d_parsScore, 0, (size_t)mem->K * mem->parsScorePerTree * sizeof(unsigned int), stream
    ));

    const size_t sharedBytes = sizeof(BuildShared);
    printf(
        "[GPU] buildParsimonyTreesKernel: K=%d  sprDist=%d"
        "  numSearchIter=%d  numNNI=%d  maxDoWhile=%d  shared=%.1f KB\n",
        mem->K, sprDist, numSearchIter, numNNI, maxDoWhile, sharedBytes / 1024.0
    );

    buildParsimonyTreesKernel<<<dim3(mem->K), dim3(kWarpSize), sharedBytes, stream>>>(
        mem->d_parsVect, mem->d_parsScore, mem->d_topos, mem->d_siteWeights, d_seeds, mem->width,
        mem->states, sprDist, numSearchIter, numNNI, maxDoWhile, mem->parsVectPerTree,
        mem->parsScorePerTree
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_seeds));
}

}  // namespace mpbootgpu
