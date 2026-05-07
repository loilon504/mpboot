#include <cuda_runtime.h>
#include "gpu/include/gpu_spr.cuh"
#include "gpu/include/pars_tree.cuh"

namespace mpbootgpu
{

// SPR is now joined into buildParsimonyTreesKernel (pars_build.cu).
// This stub exists only to satisfy the linker; the real work is done in
// gpuStepwiseBuildTrees with sprDist > 0.
void gpuSprBuildTrees(
    GpuParsimonyMem* /*mem*/, int /*sprDist*/, cudaStream_t /*stream*/
)
{
    // no-op: SPR is executed inside buildParsimonyTreesKernel
}

}  // namespace mpbootgpu
