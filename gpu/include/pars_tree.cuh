#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "pllrepo/src/pll.h"
#include "topo_helpers.cuh"

namespace mpbootgpu
{

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr int kMaxTaxa = 800;
// Each tip: 1 noderec. Each inner node: 3 noderecs.
// Total noderecs = N + 3*(N-1) = 4N-3, but we index up to 4*N-2 to be safe.
static constexpr int kMaxVFaces = 4 * kMaxTaxa;  // vface IDs = offsets into nodeBaseAddress
// Node numbers go 1..2N-1
static constexpr int kMaxNodes = 2 * kMaxTaxa;
static constexpr int kMaxStates = 32;  // DNA
static constexpr int kWarpSize = 32;

// ─── GPU Topology ─────────────────────────────────────────────────────────────
// Mirrors PLL's node rings as plain integer arrays.
// A "vface" (virtual face) is an index into nodeBaseAddress[].
//
// Tips  (number = 1..N):        1 vface, next_vf = self, nnxt_vf = self
// Inner (number = N+1..2N-1):   3 vfaces sharing the same number,
//   nodep[i] = face[2], ring: face[2]->face[1]->face[0]->face[2]
//
// parsVect is indexed by node NUMBER (1..2N-1), not by vface.
struct GpuTopology
{
    // HOT: random-access in every testInsert / doAddTraverse call
    int back_vf[kMaxVFaces];       // back neighbor's vface (-1 = NULL) — current topology
    int xpars[kMaxVFaces];         // xPars flag (0 or 1)

    // Scalars (read a few times per SPR iteration)
    int mxtips;
    int ntips;
    int nextnode;  // next inner node slot to allocate
    unsigned int bestParsimony;
    unsigned int preSprParsimony;   // parsimony after stepwise addition (before initial SPR)
    unsigned int postSprParsimony;  // parsimony after initial SPR (before hill-climbing)
    int insert_vface;              // insertNode as vface (-1 = NULL)
    int start_vface;               // tr->start as vface
    int num_vfaces;                // = mxtips + 3*(mxtips-1)

    // MEDIUM: sequential access in SPR outer loop (gpuNodeRectifierPars + per-node lookup)
    int nodep[kMaxNodes];          // DFS-canonical vface per node (like CPU tr->nodep[])
                                   // tips: nodep[num] = num-1; inner: set by gpuNodeRectifierPars

    // COLD: only on bestParsimony update + topology download
    int best_back_vf[kMaxVFaces];  // back_vf[] snapshot at the time bestParsimony was achieved
    // NOTE: number[], next_vf[], nnxt_vf[] removed — kernel uses pure arithmetic
    //   (vfToNum, vfNextFace, vfNnxtFace) and CPU-side p->number/p->next are stable after init.
};

// ─── Parsimony memory for K trees ─────────────────────────────────────────────
// parsVect GPU layout: [treeId * nodesPerTree * width * states
//                       + nodeNumber * width * states
//                       + block * states
//                       + state]
// i.e. [tree][node][block][state]  — AoS at block level, coalesced per-thread
//
// CPU layout:  [node][state][block]
// → reorder on upload (reorderParsVectForGpu)
struct GpuParsimonyMem
{
    parsimonyNumber* d_parsVect;  // [K][2N+1][width][states]
    unsigned int* d_parsScore;    // [K][2N+1]
    GpuTopology* d_topos;         // [K] topology per tree (global mem)
    unsigned int* d_siteWeights;  // [K][width] per-block weights; 1=normal, 2=ratchet-doubled

    int K;  // number of trees
    int mxtips;
    int width;  // parsimonyLength (compressed blocks)
    int states;

    size_t nodesPerTree;        // = 2*mxtips + 1 (0-indexed; slot 0 unused)
    size_t parsVectPerTree;     // = nodesPerTree * width * states  (elements)
    size_t parsScorePerTree;    // = nodesPerTree                    (elements)
    size_t siteWeightsPerTree;  // = width (elements)
};

// ─── Shared memory per block ──────────────────────────────────────────────────
// Single struct shared by build phase AND SPR phase (executed sequentially).
// Total ≈ 25 KB — within A100's 48 KB default shared memory.
struct alignas(
    16
) BuildShared
{
    // ── Build-phase arrays ────────────────────────────────────────────────────
    int perm[kMaxTaxa + 2];  // permutation 1..N                       (~2.8 KB)
    // ── DFS stack: build uses as node-pair DFS; SPR reuses as stackVf ────────
    int stack[kMaxTaxa * 2];  // build DFS / SPR addTraverse stackVf    (~5.6 KB)
    // ── SPR addTraverse auxiliary stacks ─────────────────────────────────────
    int stackMint[kMaxTaxa];  // mintrav per entry / child_idx (recompute) (~2.8 KB)
    int stackMaxt[kMaxTaxa];  // maxtrav per entry              (~2.8 KB)
    int stackTop;
    // ── Traversal info (both phases) ─────────────────────────────────────────
    int ti[kMaxTaxa * 3];   // (p_num, q_num, r_num) tuples            (~8.4 KB)
    int tiStack[kMaxTaxa];  // DFS stack for computeTraversalInfo       (~2.8 KB)
    int tiSize;
    // ── Build-phase scalars ───────────────────────────────────────────────────
    int insertVf;       // best insert vface found this stepwise-addition round
    int startVf;        // tr->start vface = nodepVf(min(perm[1..3]))
    int qnum;           // inner node being inserted (build)
    int tipnum;         // tip being inserted (build)
    int qf0, qf1, qf2;  // vfaces of qnum (build)
    // ── Shared best parsimony (both phases use same fields) ──────────────────
    unsigned int bestParsimony;  // best parsimony found (build: per-edge; SPR: per-rearrangement)
    unsigned int bestHits;       // tie-breaking counter
    // ── SPR-phase scalars ─────────────────────────────────────────────────────
    long seed;                  // RNG seed (set from build, carried into SPR)
    unsigned int randomMP;      // global best parsimony for this tree (SPR do-while threshold)
    unsigned int randomMPHits;  // tie-breaking across outer SPR iterations
    int bestRemoveVf;           // best move: pruned node vface
    int bestInsertVf;           // best move: insertion edge vface
    // ── Broadcast slots (SPR phase) ──────────────────────────────────────────
    // [0]   testInsert: r = back_vf[q]
    // [1-3] doAddTraverse: vf, mint, maxt
    // [4-5] SPR rearrange loop: p, q (vfaces)
    // [7-8] P/Q branch children: p1(q1), p2(q2)
    // [9]   apply-move sentinel (-1 = no move) / bestRemoveVf
    // [10]  apply-move bestInsertVf
    int bcast[11];
    // ── Ratchet site weights (Phase 2b) ──────────────────────────────────────
    const unsigned int* site_weights;  // nullptr = uniform weight 1
    // ── Timing accumulators (block 0 lane 0 only) ─────────────────────────────
    long long t_build;       // Phase 1: stepwise addition (clock cycles)
    long long t_phase2;      // Phase 2: initial SPR (clock cycles)
    long long t_p3_nni;      // Phase 3 even iters: NNI+setup accumulated
    long long t_p3_nni_spr;  // Phase 3 even iters: SPR accumulated
    long long t_p3_ratchet;  // Phase 3 odd iters: total accumulated
    int n_p3_even;           // count of even Phase 3 iterations
    int n_p3_odd;            // count of odd Phase 3 iterations
    long long t_line2291;    // Phase 3 SPR: createTiAndEvaluateParsimony (line-2291)
    long long t_search;      // Phase 3 SPR: doAddTraverse (candidate search)
    long long t_apply;       // Phase 3 SPR: applyMove
    int n_apply;             // number of moves applied
    int n_dowhile;           // number of do-while passes
    // testInsert sub-timing (Phase 3 even iters, block 0 lane 0 only)
    long long t_ti_newview;    // createTiAndNewviewParsimony inside testInsert
    long long t_ti_eval;       // createTiAndEvaluateParsimony inside testInsert
    int n_testInsert;          // testInsert calls
    int n_ti_newview_size;     // total nodes traversed in newview (tiSize/3)
    int n_ti_eval_size;        // total nodes traversed in eval (tiSize/3)
};

// ─── Host API ─────────────────────────────────────────────────────────────────

// Convert pllInstance pointer-ring topology → GpuTopology integer arrays.
// Call this on the CPU; result can then be uploaded to device.
void cpuToGpuTopology(const pllInstance* tr, GpuTopology* out);

// Reverse: write integer arrays back to pllInstance pointer ring.
// tr must already have the same nodeBaseAddress layout.
void gpuTopoToCpu(const GpuTopology* in, pllInstance* tr);

// Allocate all GPU memory for K trees.
// mxtips, width, states must match the alignment.
GpuParsimonyMem* gpuParsimonyMemAlloc(int K, int mxtips, int width, int states);
void gpuParsimonyMemFree(GpuParsimonyMem* mem);

// Upload tip parsVect for ALL K trees (shared; tips are read-only).
// Reorders from CPU [node][state][block] → GPU [node][block][state].
// pr must be the pllInstance whose compressDNA has already been called.
void uploadTipParsVect(
    GpuParsimonyMem* mem, const pllInstance* tr, const partitionList* pr, cudaStream_t stream = 0
);

// Upload / download topology for tree k.
void uploadTopology(
    GpuParsimonyMem* mem, int k, const GpuTopology* h_topo, cudaStream_t stream = 0
);
void downloadTopology(
    const GpuParsimonyMem* mem, int k, GpuTopology* h_out, cudaStream_t stream = 0
);

// Download parsimonyScore for tree k (count = 2*mxtips+1 entries).
void downloadParsScore(
    const GpuParsimonyMem* mem, int k, unsigned int* h_out, cudaStream_t stream = 0
);

// ─── Validation kernel (Phase 1 test) ─────────────────────────────────────────
// Runs newview for all (p,q,r) triples in h_ti[0..tiCount-1]
// then accumulates parsimonyScore bottom-up.
// Uses the SAME ti[] for all K trees (good for validation against CPU).
void validateNewview(GpuParsimonyMem* mem, const int* h_ti, int tiCount, cudaStream_t stream = 0);

// ─── Device functions (header-inlined so any .cu can use them) ───────────────

// Warp-level sum reduction (result valid in lane 0).
__device__ __forceinline__ unsigned int warpReduceU32(
    unsigned int val
)
{
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename SharedT>
__device__ __forceinline__ void computeTraversalInfoParsimony(
    GpuTopology* topo, SharedT& sh, int node, int N, bool full
)
{
    int top = -1;
    sh.tiStack[++top] = node;
    while (top >= 0)
    {
        int p = sh.tiStack[top--];
        int pNext = vfNextFace(p, N);
        int pNnxt = vfNnxtFace(p, N);
        int q = topo->back_vf[pNext];
        int r = topo->back_vf[pNnxt];
        if (!topo->xpars[p])
        {
            if (topo->xpars[pNext])
            {
                topo->xpars[p] = 1;
                topo->xpars[pNext] = 0;
            }
            else if (topo->xpars[pNnxt])
            {
                topo->xpars[p] = 1;
                topo->xpars[pNnxt] = 0;
            }
        }

        if (full)
        {
            if (q >= N)
            {
                sh.tiStack[++top] = q;
            }
            if (r >= N)
            {
                sh.tiStack[++top] = r;
            }
        }
        else
        {
            if (q >= N && !topo->xpars[q])
            {
                sh.tiStack[++top] = q;
            }
            if (r >= N && !topo->xpars[r])
            {
                sh.tiStack[++top] = r;
            }
        }

        sh.ti[sh.tiSize++] = vfToNum(p, N);
        sh.ti[sh.tiSize++] = vfToNum(q, N);
        sh.ti[sh.tiSize++] = vfToNum(r, N);
    }
}

template <typename SharedT>
__device__ __forceinline__ unsigned int newviewParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    SharedT& sh,
    bool evaluate,
    int width,
    int states
)
{
    int lane = threadIdx.x & 31;
    // Ratchet: if sh.site_weights is set, multiply partial scores by site_weights[b].
    // Pointer is uniform across all lanes — no warp divergence on the check.
    const unsigned int* sw = sh.site_weights;

    if (lane == 0)
    {
        for (int i = 3; i < sh.tiSize; i += 3)
        {
            score_tree[sh.ti[i]] = 0;
        }
    }

    for (int i = sh.tiSize - 3; i >= 3; i -= 3)
    {
        int p_num = sh.ti[i];
        int q_num = sh.ti[i + 1];
        int r_num = sh.ti[i + 2];

        unsigned int score = 0;

        parsimonyNumber* p_base = pars_tree + (size_t)p_num * width * states;
        parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * states;
        parsimonyNumber* r_base = pars_tree + (size_t)r_num * width * states;

        for (int b = lane; b < width; b += kWarpSize)
        {
            parsimonyNumber t_N = 0;
            parsimonyNumber t_A[kMaxStates], o_A[kMaxStates];

            for (int s = 0; s < states; ++s)
            {
                parsimonyNumber lv = q_base[b * states + s];
                parsimonyNumber rv = r_base[b * states + s];
                t_A[s] = lv & rv;
                o_A[s] = lv | rv;
                t_N |= t_A[s];
            }
            t_N = ~t_N;

            for (int s = 0; s < states; ++s)
            {
                p_base[b * states + s] = t_A[s] | (t_N & o_A[s]);
            }

            score += sw ? sw[b] * __popc(t_N) : __popc(t_N);
        }
        score = warpReduceU32(score);

        if (lane == 0)
        {
            score_tree[p_num] = score + score_tree[q_num] + score_tree[r_num];
        }
    }
    if (!evaluate)
    {
        return score_tree[sh.ti[3]];
    }

    int q_num = sh.ti[1];
    int r_num = sh.ti[2];

    unsigned int score = 0;

    parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * states;
    parsimonyNumber* r_base = pars_tree + (size_t)r_num * width * states;

    for (int b = lane; b < width; b += kWarpSize)
    {
        parsimonyNumber t_N = 0;
        parsimonyNumber t_A[kMaxStates], o_A[kMaxStates];

        for (int s = 0; s < states; ++s)
        {
            parsimonyNumber lv = q_base[b * states + s];
            parsimonyNumber rv = r_base[b * states + s];
            t_A[s] = lv & rv;
            o_A[s] = lv | rv;
            t_N |= t_A[s];
        }
        t_N = ~t_N;

        score += sw ? sw[b] * __popc(t_N) : __popc(t_N);
    }
    score = warpReduceU32(score);

    if (lane == 0)
    {
        return score + score_tree[q_num] + score_tree[r_num];
    }
}

template <typename SharedT>
__device__ __forceinline__ void createTiAndNewviewParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int N,
    int width,
    int states,
    int lane
)
{
    if (p < N)
    {
        return;
    }

    if (lane == 0)
    {
        sh.tiSize = 3;
        computeTraversalInfoParsimony(topo, sh, p, N, false);
    }
    __syncwarp();
    newviewParsimony(pars_tree, score_tree, sh, false, width, states);
}

template <typename SharedT>
__device__ __forceinline__ unsigned int createTiAndEvaluateParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int N,
    bool full,
    int width,
    int states
)
{
    int q = topo->back_vf[p];
    int lane = threadIdx.x & 31;

    if (lane == 0)
    {
        sh.tiSize = 3;
        sh.ti[1] = vfToNum(p, N);
        sh.ti[2] = vfToNum(q, N);

        if (full)
        {
            if (p >= N)
            {
                computeTraversalInfoParsimony(topo, sh, p, N, full);
            }
            if (q >= N)
            {
                computeTraversalInfoParsimony(topo, sh, q, N, full);
            }
        }
        else
        {
            if (p >= N && !topo->xpars[p])
            {
                computeTraversalInfoParsimony(topo, sh, p, N, full);
            }
            if (q >= N && !topo->xpars[q])
            {
                computeTraversalInfoParsimony(topo, sh, q, N, full);
            }
        }
    }
    __syncwarp();  // ensure lane 0's ti[] writes are visible to all lanes before newviewParsimony

    return newviewParsimony(pars_tree, score_tree, sh, true, width, states);
}

}  // namespace mpbootgpu
