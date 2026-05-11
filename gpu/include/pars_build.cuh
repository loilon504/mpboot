#pragma once
#include <cuda_runtime.h>
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// Run stepwise-addition + SPR + iterative NNI+SPR search for all K trees.
// sprDist      = SPR radius for Phase 3 (initial SPR after build)
// sprDist4     = SPR radius for Phase 4 outer NNI+ratchet iterations (Opt 2: smaller)
// numSearchIter = outer NNI+ratchet iterations (Phase 4)
// numNNI       = NNI moves per odd outer iteration
// maxDoWhile   = max do-while passes per SPR call (Opt 3: cap convergence)
void gpuStepwiseBuildTrees(
    GpuParsimonyMem* mem,
    const long*      seeds,
    int              sprDist,
    int              sprDist4,
    int              numSearchIter,
    int              numNNI,
    int              maxDoWhile,
    cudaStream_t     stream
);

}  // namespace mpbootgpu
