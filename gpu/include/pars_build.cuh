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
using AfterK1Callback = std::function<void(cudaStream_t, GpuParsimonyMem*)>;

// Phase 2 hybrid callback: called right after Kernel 2 is launched (K2 still running on stream).
// Contract:
//   - Callback may call cudaStreamQuery(stream) to do CPU work while K2 runs.
//   - Callback MUST call cudaStreamSynchronize(stream) before returning (to ensure K2 done).
using AfterK2Callback = std::function<void(cudaStream_t)>;

// Run stepwise-addition + SPR + iterative NNI+SPR search for all K trees.
// sprDist        = SPR radius (used for all phases)
// numNNI         = NNI moves per even outer iteration
// stopNoImprove  = stop Phase 3 after this many consecutive no-improve iters
// poolSize       = number of pool slots for population-based restarts (from -gpu_pool_size)
// after_k1       = optional Phase 1 hybrid callback (nullptr = standard mode)
// after_k2       = optional Phase 2 hybrid callback (nullptr = standard mode)
// k1_trees       = K' trees to build in K1 (≤ K); -1 or K = build all K (default)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem*  mem,
    const long*       seeds,
    int               sprDist,
    int               numNNI,
    int               stopNoImprove,
    int               poolSize,
    cudaStream_t      stream,
    AfterK1Callback   after_k1 = nullptr,
    AfterK2Callback   after_k2 = nullptr,
    int               k1_trees = -1
);

}  // namespace mpbootgpu
