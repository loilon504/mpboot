#pragma once
#include <cuda_runtime.h>
#include <functional>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// CPU-built tree data for hybrid mode: topology pre-converted to GPU format.
// needs_recompute is set to 1 so buildPhase3Kernel re-evaluates parsVect.
struct CpuTreeData
{
    GpuTopology  topo;
    unsigned int score;
};

// Hybrid callback: called right after Kernel 1 is launched (K1 still running on stream).
// Contract:
//   - Callback may call cudaStreamQuery(stream) to build CPU trees while K1 runs.
//   - Callback MUST call cudaStreamSynchronize(stream) before returning (to ensure K1 done).
//   - Returns the threshold to use for Kernel 2 (replacing standard top-pct computation).
using AfterK1Callback = std::function<unsigned int(cudaStream_t, GpuParsimonyMem*)>;

// Run stepwise-addition + SPR + iterative NNI+SPR search for all K trees.
// sprDist        = SPR radius (used for all phases)
// numSearchIter  = outer NNI+ratchet iterations (Phase 3)
// numNNI         = NNI moves per even outer iteration
// stopNoImprove  = stop Phase 3 after this many consecutive no-improve iters (0 = disabled)
// topPct         = Opt-G2 two-kernel: only top X% trees (lowest postSprParsimony) do Phase 3
//                  ≤0 = disabled (use single-kernel mode)
// after_k1       = optional hybrid callback (nullptr = standard mode)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem*  mem,
    const long*       seeds,
    int               sprDist,
    int               numSearchIter,
    int               numNNI,
    int               stopNoImprove,
    float             topPct,
    cudaStream_t      stream,
    AfterK1Callback   after_k1 = nullptr
);

}  // namespace mpbootgpu
