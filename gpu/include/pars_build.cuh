#pragma once
#include <cuda_runtime.h>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Run stepwise-addition parsimony tree building for all K trees in mem.
// seeds[k]  = PLL-compatible randomNumberSeed for tree k.
// On return:
//   mem->d_topos[k]  — final (pre-SPR) topology for each tree k
//   mem->d_parsScore — subtree costs per node
// SPR hill-climbing is intentionally left to the CPU (download + CPU rearrange)
// so this kernel targets the dominant O(N²) stepwise-addition phase.
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long*      seeds,   // host array [K]
    cudaStream_t     stream
);

}  // namespace mpbootgpu
