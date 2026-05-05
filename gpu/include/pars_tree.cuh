#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "pllrepo/src/pll.h"

namespace mpbootgpu
{

// ─── Constants ────────────────────────────────────────────────────────────────
static constexpr int kMaxTaxa = 700;
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
    int back_vf[kMaxVFaces];  // back neighbor's vface (-1 = NULL)
    int next_vf[kMaxVFaces];  // next face in ring (self for tips)
    int nnxt_vf[kMaxVFaces];  // next->next face in ring (self for tips)
    int number[kMaxVFaces];   // node->number  (parsVect index)
    int xpars[kMaxVFaces];    // xPars flag (0 or 1)

    // Scalars
    int mxtips;
    int ntips;
    int nextnode;  // next inner node slot to allocate
    unsigned int bestParsimony;
    unsigned int preSprParsimony;  // full-tree parsimony before SPR (set by gpuSprKernel init)
    int insert_vface;  // insertNode as vface (-1 = NULL)
    int start_vface;   // tr->start as vface
    int num_vfaces;    // = mxtips + 3*(mxtips-1)
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

    int K;  // number of trees
    int mxtips;
    int width;  // parsimonyLength (compressed blocks)
    int states;

    size_t nodesPerTree;      // = 2*mxtips + 1 (0-indexed; slot 0 unused)
    size_t parsVectPerTree;   // = nodesPerTree * width * states  (elements)
    size_t parsScorePerTree;  // = nodesPerTree                    (elements)
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

// Compute parsVect[p_num] from parsVect[q_num] and parsVect[r_num].
// Returns partial popcount score (call warpReduceU32 after to get full score).
__device__ __forceinline__ unsigned int warpNewviewStep(
    parsimonyNumber* __restrict__ pars_tree, int p_num, int q_num, int r_num, int width, int states
)
{
    int lane = threadIdx.x & 31;
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

        score += __popc(t_N);
    }
    return score;
}

// Compute parsimony score on branch (p_num – q_num).
// Assumes parsVect[p_num] and parsVect[q_num] are already up to date.
// Result valid in lane 0 after internal warp reduction.
__device__ __forceinline__ unsigned int warpEvaluateScore(
    const parsimonyNumber* __restrict__ pars_tree,
    const unsigned int* __restrict__ score_tree,
    int p_num,
    int q_num,
    int width,
    int states
)
{
    int lane = threadIdx.x & 31;

    const parsimonyNumber* p_base = pars_tree + (size_t)p_num * width * states;
    const parsimonyNumber* q_base = pars_tree + (size_t)q_num * width * states;

    unsigned int sum = 0;
    for (int b = lane; b < width; b += kWarpSize)
    {
        parsimonyNumber t_N = 0;
        for (int s = 0; s < states; ++s)
        {
            t_N |= (p_base[b * states + s] & q_base[b * states + s]);
        }
        t_N = ~t_N;
        sum += __popc(t_N);
    }
    sum = warpReduceU32(sum);

    if (lane == 0)
    {
        sum += score_tree[p_num] + score_tree[q_num];
    }

    return sum;
}

}  // namespace mpbootgpu
