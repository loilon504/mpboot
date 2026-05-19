#pragma once
#include <cuda_runtime.h>
#include <functional>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// CPU-built tree data for hybrid mode: topology pre-converted to GPU format.
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
// poolSize       = number of pool slots for population-based restarts (from -gpu_pool_size)
// poolStopThresh = stop K2 after this many outer iters with no global improvement (from -gpu_pool_stop)
// after_k1       = optional Phase 1 hybrid callback (nullptr = standard mode)
// after_k2       = optional Phase 2 hybrid callback (nullptr = standard mode)
// k1_count       = K1 blocks to build (= numpars); must be ≤ mem->K
// k2_workers     = K2 blocks (-1 = same as k1_count); mem->K = max(k1_count, k2_workers)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem*  mem,
    const long*       seeds,
    int               k1_count,
    int               sprDist,
    int               numNNI,
    int               poolSize,
    int               poolStopThresh,
    cudaStream_t      stream,
    AfterK1Callback   after_k1   = nullptr,
    AfterK2Callback   after_k2   = nullptr,
    int               k2_workers = -1
);

}  // namespace mpbootgpu
