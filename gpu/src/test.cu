#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "gpu/include/test.cuh"

namespace mpbootgpu
{
__global__ void vectorAdd(
    const float* A, const float* B, float* C, int n
)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void testGpu()
{
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    for (int i = 0; i < n; i++)
    {
        h_A[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    int errors = 0;
    float epsilon = 1e-5f;

    for (int i = 0; i < n; i++)
    {
        float cpu_result = h_A[i] + h_B[i];
        if (std::fabs(h_C[i] - cpu_result) > epsilon)
        {
            errors++;
            if (errors < 5)
            {
                std::cout << "Lỗi tại index " << i << ": GPU=" << h_C[i] << " CPU=" << cpu_result
                          << std::endl;
            }
        }
    }

    if (errors == 0)
    {
        std::cout << "SUCCESS: Toàn bộ " << n << " phần tử đều chính xác!" << std::endl;
    }
    else
    {
        std::cout << "FAILURE: Có " << errors << " lỗi trong tổng số " << n << " phần tử."
                  << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
}  // namespace mpbootgpu