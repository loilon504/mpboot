#pragma once
#include <cuda_runtime.h>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Run stepwise-addition + SPR hill-climbing for all K trees in parallel.
// seeds[k]  = PLL-compatible randomNumberSeed for tree k (also used as SPR RNG).
// sprDist   = SPR radius; pass 0 to skip SPR (build only).
// On return:
//   mem->d_topos[k].bestParsimony    — post-SPR best parsimony
//   mem->d_topos[k].preSprParsimony  — pre-SPR full-tree parsimony (set when sprDist>0)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long*      seeds,   // host array [K]
    int              sprDist,
    cudaStream_t     stream
);

}  // namespace mpbootgpu
