#ifndef MPBOOTGPU_UTILS_CUH_
#define MPBOOTGPU_UTILS_CUH_

#include <cuda_runtime.h>

namespace mpbootgpu
{

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                         \
    do                                                                                           \
    {                                                                                            \
        cudaError_t err = call;                                                                  \
        if (err != cudaSuccess)                                                                  \
        {                                                                                        \
            throw std::runtime_error(                                                            \
                std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " \
                + cudaGetErrorString(err)                                                        \
            );                                                                                   \
        }                                                                                        \
    } while (0)
#endif

}  // namespace mpbootgpu

#endif