#pragma once
#include <cuda_runtime.h>

#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Runs parsimony SPR hill-climbing on all K trees in parallel (one block per
// tree, one warp per block).  On entry the parsVect / parsScore arrays in mem
// must be fully initialised (e.g. by gpuStepwiseBuildTrees).  On exit the
// topologies in d_topos are improved and topo->bestParsimony is updated.
void gpuSprBuildTrees(GpuParsimonyMem* mem, int sprDist, cudaStream_t stream = 0);

}  // namespace mpbootgpu
