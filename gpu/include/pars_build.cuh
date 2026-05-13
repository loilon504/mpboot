#pragma once
#include <cuda_runtime.h>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Run stepwise-addition + SPR + iterative NNI+SPR search for all K trees.
// sprDist        = SPR radius (used for all phases)
// numSearchIter  = outer NNI+ratchet iterations (Phase 3)
// numNNI         = NNI moves per even outer iteration
// stopNoImprove  = stop Phase 3 after this many consecutive no-improve iters (0 = disabled)
// phase3Margin   = selective Phase 3: skip if postSprParsimony > globalBest*(1+margin/100)
//                  UINT_MAX = disabled (all trees do Phase 3)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long*      seeds,
    int              sprDist,
    int              numSearchIter,
    int              numNNI,
    int              stopNoImprove,
    unsigned int     phase3Margin,
    cudaStream_t     stream
);

}  // namespace mpbootgpu
