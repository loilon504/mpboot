#pragma once
#include <cuda_runtime.h>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Run stepwise-addition + SPR + iterative NNI+SPR search for all K trees.
// sprDist       = SPR radius (used for all phases)
// numSearchIter = outer NNI+ratchet iterations (Phase 4)
// numNNI        = NNI moves per odd outer iteration
// maxDoWhile    = max do-while passes per SPR call (Opt 3: cap convergence)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long*      seeds,
    int              sprDist,
    int              numSearchIter,
    int              numNNI,
    int              maxDoWhile,
    cudaStream_t     stream
);

}  // namespace mpbootgpu
