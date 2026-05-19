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
// Opt-P: reduced stack for SPR doAddTraverse + NNI bitset
// NNI bitset needs ceil(2*kMaxTaxa/32)+1 = 51 words; SPR stack depth ≤ 2*sprDist ≈ 12
static constexpr int kMaxSprStack = 64;  // >> max(51 bitset words, 12 SPR entries)

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
    long savedSeed;                  // RNG state saved at end of Phase 2; restored by buildPhase3Kernel
    int start_vface;               // tr->start as vface
    int num_vfaces;                // = mxtips + 3*(mxtips-1)

    // MEDIUM: sequential access in SPR outer loop (gpuNodeRectifierPars + per-node lookup)
    int nodep[kMaxNodes];          // DFS-canonical vface per node (like CPU tr->nodep[])
                                   // tips: nodep[num] = num-1; inner: set by gpuNodeRectifierPars

    // Phase 3 effectiveness counters (per-tree, written by kernel, read on host)
    int n_improved_even;  // iterations where even (NNI+SPR) improved bestParsimony
    int n_improved_odd;   // iterations where odd (Ratchet) improved bestParsimony
    int n_total_even;     // total even iterations run
    int n_total_odd;      // total odd iterations run

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
    unsigned int* d_postSprScores; // [K] postSprParsimony scores, filled by Phase 2; used by Opt-G2 two-kernel

    // Population pool for hill-climbing restarts (Phase 3)
    int pool_size;                // number of pool slots (runtime, from -gpu_pool_size)
    unsigned int* d_poolScores;   // [pool_size] parsimony score per pool slot (UINT_MAX = empty)
    int* d_poolBackVf;            // [pool_size][kMaxVFaces] topology snapshots
    int*     d_poolFilled;        // number of filled slots (0..pool_size)
    int*     d_poolSlotLocks;     // per-slot spinlocks [pool_size] (0=free, 1=held)
    unsigned int* d_poolHashes;   // [pool_size] topology hash per slot (0xFFFFFFFF = empty)
    unsigned int* d_globalBest;   // global best parsimony across all warps

    // Treels buffer (bootstrap round output): workers write here if score ≤ d_treelsCutoff
    int    max_treels;              // capacity; 0 = disabled
    unsigned int* d_treelsScores;   // [max_treels] parsimony score per slot (UINT_MAX = empty)
    int*   d_treelsBackVf;          // [max_treels × kMaxVFaces] back_vf snapshots
    int*   d_treelsFilled;          // atomic fill counter (0..max_treels)
    unsigned int* d_treelsCutoff;   // score ≤ cutoff → write to treels (UINT_MAX = all qualify)

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
// Opt-P Layer 3: Templated on NTAXA so array sizes track actual taxa count.
// Dispatch buckets: N≤128→128, N≤256→256, N≤384→384, N≤512→512, N>512→800.
// Using static __shared__ so size is compile-time; no dynamic shared mem needed.
// Layout: HOT int16_t arrays first (no padding between them) → int scalars.
// Grouping same-type arrays eliminates int16_t/int alignment padding.
// int16_t: values bounded by N (≤800) or ±sprDist — all fit in 16 bits.
// stackMaxt stays int: NNI bitset uses (1u<<pb) with pb=0..31, needs full 32 bits.
template<int NTAXA>
struct alignas(16) BuildSharedT
{
    // ── [HOT int16_t] Traversal info — accessed every newview/eval call ───────
    // node numbers 1..2N-1 ≤ 1599; vface IDs 0..4N-3 ≤ 3197 → fit int16_t
    int16_t ti[NTAXA * 3];   // (p_num, q_num, r_num) tuples; max entries = 3*N
    int16_t tiStack[NTAXA];  // DFS stack for computeTraversalInfo; worst case = N

    // ── [HOT int16_t] DFS / SPR addTraverse stack ────────────────────────────
    int16_t stack[NTAXA * 2];  // Phase 1: node-pair DFS; Phase 2-3: addTraverse stackVf
                               // Phase 3 NNI: edge list (p_vf, q_vf) pairs

    // ── [MEDIUM int16_t] Build/SPR mint stack (Opt-P L1: union saves 3.2 KB) ─
    // perm Phase 0-1 ONLY; stackMint Phase 2-3 ONLY → never overlap → safe
    union {
        int16_t perm[NTAXA + 2];  // Phase 0-1: Fisher-Yates permutation 1..N
        int16_t stackMint[NTAXA]; // Phase 2-3: mintrav per doAddTraverse stack entry
    };

    // ── [int] SPR maxtrav + NNI bitset (Opt-P L1: reduced kMaxTaxa→64) ───────
    // 32-bit required: NNI bitset sets individual bits via (1u << pb), pb=0..31
    int stackMaxt[kMaxSprStack];
    int stackTop;
    int tiSize;

    // ── [HOT int] SPR scalars — checked every testInsert / outer iteration ────
    // [0]   testInsert: r = back_vf[q]
    // [1-3] doAddTraverse: stackVf, mint, maxt popped per step
    // [4-5] SPR rearrange loop: p, q vfaces
    // [7-8] P/Q branch children: p1(q1), p2(q2)
    // [9]   apply-move sentinel (-1 = no move)
    // [10]  apply-move bestInsertVf
    int bcast[11];
    unsigned int bestParsimony;  // best parsimony this rearrangement scan
    unsigned int bestHits;       // tie-breaking counter
    unsigned int randomMP;       // do-while threshold (tree's current best parsimony)
    unsigned int randomMPHits;
    int bestRemoveVf;
    int bestInsertVf;
    long seed;                   // RNG state for gpuRandum

    // ── [MEDIUM] Ratchet / site weights ──────────────────────────────────────
    const unsigned int* site_weights;  // nullptr = uniform weight 1

    // ── [BUILD-ONLY] Phase 0-1 scalars (not touched during SPR) ─────────────
    int insertVf;
    int startVf;
    int qnum;
    int tipnum;
    int qf0, qf1, qf2;


};

// Default alias (NTAXA=kMaxTaxa=800): used as BuildShared throughout non-templated code
using BuildShared = BuildSharedT<kMaxTaxa>;


// ─── Host API ─────────────────────────────────────────────────────────────────

// Convert pllInstance pointer-ring topology → GpuTopology integer arrays.
// Call this on the CPU; result can then be uploaded to device.
void cpuToGpuTopology(const pllInstance* tr, GpuTopology* out);

// Reverse: write integer arrays back to pllInstance pointer ring.
// tr must already have the same nodeBaseAddress layout.
void gpuTopoToCpu(const GpuTopology* in, pllInstance* tr);

// Allocate all GPU memory for K trees.
// mxtips, width, states must match the alignment.
GpuParsimonyMem* gpuParsimonyMemAlloc(int K, int mxtips, int width, int states,
                                       int pool_size = 20, int max_treels = 0);
void gpuParsimonyMemFree(GpuParsimonyMem* mem);

// Reset treels buffer and set new cutoff threshold (call before each K2 bootstrap round).
void resetTreelsRound(GpuParsimonyMem* mem, unsigned int cutoff_pars);

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

// ─── Pool helpers (bootstrap rounds) ──────────────────────────────────────────

// Download all pool scores into h_out[pool_size]. Synchronous.
void downloadPoolScores(
    const GpuParsimonyMem* mem, unsigned int* h_out
);

// Download back_vf array for pool slot s into h_back_vf[kMaxVFaces]. Synchronous.
void downloadPoolBackVf(
    const GpuParsimonyMem* mem, int slot, int* h_back_vf
);

// Reset pool topology hashes to 0xFFFFFFFF (call before each K2 bootstrap round).
void resetPoolRound(GpuParsimonyMem* mem);

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

template <typename SharedT, int STATES>
__device__ __forceinline__ unsigned int newviewParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    SharedT& sh,
    bool evaluate,
    int width
)
{
    int lane = threadIdx.x & 31;
    // Ratchet: if sh.site_weights is set, multiply partial scores by site_weights[b].
    // Pointer is uniform across all lanes — no warp divergence on the check.
    const unsigned int* sw = sh.site_weights;

    for (int i = sh.tiSize - 3; i >= 3; i -= 3)
    {
        int p_num = sh.ti[i];
        int q_num = sh.ti[i + 1];
        int r_num = sh.ti[i + 2];

        unsigned int score = 0;

        parsimonyNumber* p_base = pars_tree + (size_t)p_num * width * STATES;
        parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * STATES;
        parsimonyNumber* r_base = pars_tree + (size_t)r_num * width * STATES;

        for (int b = lane; b < width; b += kWarpSize)
        {
            parsimonyNumber t_N = 0;
            parsimonyNumber t_A[STATES], o_A[STATES];

            #pragma unroll
            for (int s = 0; s < STATES; ++s)
            {
                parsimonyNumber lv = q_base[b * STATES + s];
                parsimonyNumber rv = r_base[b * STATES + s];
                t_A[s] = lv & rv;
                o_A[s] = lv | rv;
                t_N |= t_A[s];
            }
            t_N = ~t_N;

            #pragma unroll
            for (int s = 0; s < STATES; ++s)
            {
                p_base[b * STATES + s] = t_A[s] | (t_N & o_A[s]);
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

    parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * STATES;
    parsimonyNumber* r_base = pars_tree + (size_t)r_num * width * STATES;

    for (int b = lane; b < width; b += kWarpSize)
    {
        parsimonyNumber t_N = 0;
        parsimonyNumber t_A[STATES], o_A[STATES];

        #pragma unroll
        for (int s = 0; s < STATES; ++s)
        {
            parsimonyNumber lv = q_base[b * STATES + s];
            parsimonyNumber rv = r_base[b * STATES + s];
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

template <typename SharedT, int STATES>
__device__ __forceinline__ void createTiAndNewviewParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int N,
    int width,
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
    newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, false, width);
}

template <typename SharedT, int STATES>
__device__ __forceinline__ unsigned int createTiAndEvaluateParsimony(
    parsimonyNumber* __restrict__ pars_tree,
    unsigned int* __restrict__ score_tree,
    GpuTopology* topo,
    SharedT& sh,
    int p,
    int N,
    bool full,
    int width
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

    return newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, true, width);
}

}  // namespace mpbootgpu
