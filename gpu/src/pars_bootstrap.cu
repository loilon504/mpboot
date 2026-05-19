#include <cstdio>
#include <stdexcept>
#include <string>
#include "gpu/include/pars_bootstrap.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

// ─── REPSKernel ───────────────────────────────────────────────────────────────
// Grid:  (B, 1, 1)  — one block per bootstrap replicate
// Block: (32, 1, 1) — one warp
//
// Each block computes:
//   rell[i] = sum_{p=0}^{P-1}  pattern_pars[p] * boot_samples[i][p]
//
// Access pattern:
//   - boot_samples row i: 32 lanes read boot[lane], boot[lane+32], ...
//     → 32 consecutive unsigned short per iteration = 64 bytes = 1 cache line (coalesced)
//   - pattern_pars: all B blocks read the same array → L2 broadcast (~10 KB, stays in cache)
__global__ void REPSKernel(
    const unsigned short* __restrict__ d_pattern_pars,  // [nunit]
    const unsigned short* __restrict__ d_boot_samples,  // [B × nunit], row-major
    int*                               d_rell,           // [B] output
    int P,
    int nunit
)
{
    const int i    = blockIdx.x;   // replicate index (0..B-1)
    const int lane = threadIdx.x;  // 0..31

    const unsigned short* boot = d_boot_samples + (size_t)i * nunit;

    unsigned int local_sum = 0;
    for (int p = lane; p < P; p += 32)
        local_sum += (unsigned int)d_pattern_pars[p] * (unsigned int)boot[p];

    // Warp reduction: accumulate across all 32 lanes
    for (int s = 16; s > 0; s >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, s);

    if (lane == 0)
        d_rell[i] = (int)local_sum;
}

// ─── Host API ─────────────────────────────────────────────────────────────────

GpuBootstrapMem* gpuBootstrapMemAlloc(int B, int P, int nunit)
{
    auto* mem = new GpuBootstrapMem();
    mem->B     = B;
    mem->P     = P;
    mem->nunit = nunit;

    CUDA_CHECK(cudaMalloc(&mem->d_boot_samples, (size_t)B * nunit * sizeof(unsigned short)));
    CUDA_CHECK(cudaMalloc(&mem->d_pattern_pars, (size_t)nunit    * sizeof(unsigned short)));
    CUDA_CHECK(cudaMalloc(&mem->d_rell,         (size_t)B        * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&mem->h_rell,     (size_t)B        * sizeof(int)));

    printf("[GPU Bootstrap] REPS memory: B=%d, P=%d, nunit=%d (%.2f MB boot_samples)\n",
           B, P, nunit, (double)B * nunit * sizeof(unsigned short) / (1 << 20));
    return mem;
}

void gpuBootstrapMemFree(GpuBootstrapMem* mem)
{
    if (!mem) return;
    cudaFree(mem->d_boot_samples);
    cudaFree(mem->d_pattern_pars);
    cudaFree(mem->d_rell);
    cudaFreeHost(mem->h_rell);
    delete mem;
}

void gpuUploadBootSamples(
    GpuBootstrapMem*                     mem,
    const std::vector<unsigned short*>&  boot_samples_pars
)
{
    const size_t row_bytes = (size_t)mem->nunit * sizeof(unsigned short);
    for (int i = 0; i < mem->B; ++i)
    {
        CUDA_CHECK(cudaMemcpy(
            mem->d_boot_samples + (size_t)i * mem->nunit,
            boot_samples_pars[i],
            row_bytes,
            cudaMemcpyHostToDevice
        ));
    }
    printf("[GPU Bootstrap] Uploaded %d bootstrap sample vectors (%.2f MB total)\n",
           mem->B, (double)mem->B * row_bytes / (1 << 20));
}

void gpuREPSEval(GpuBootstrapMem* mem, const unsigned short* pattern_pars)
{
    // Upload current candidate tree's per-pattern parsimony
    CUDA_CHECK(cudaMemcpy(
        mem->d_pattern_pars, pattern_pars,
        (size_t)mem->nunit * sizeof(unsigned short),
        cudaMemcpyHostToDevice
    ));

    // Launch B warps in parallel — one warp per bootstrap replicate
    REPSKernel<<<mem->B, 32>>>(
        mem->d_pattern_pars,
        mem->d_boot_samples,
        mem->d_rell,
        mem->P,
        mem->nunit
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronous download: B ints = B * 4 bytes
    CUDA_CHECK(cudaMemcpy(
        mem->h_rell, mem->d_rell,
        (size_t)mem->B * sizeof(int),
        cudaMemcpyDeviceToHost
    ));
}

}  // namespace mpbootgpu
