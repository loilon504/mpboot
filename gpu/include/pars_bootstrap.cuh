#pragma once
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
// Forward declaration for C++ TUs that include this header but don't compile with nvcc.
typedef struct CUstream_st* cudaStream_t;
#endif

namespace mpbootgpu
{

struct GpuBootstrapMem
{
    unsigned short* d_boot_samples;  // [B × nunit] device — uploaded once at init
    unsigned short* d_pattern_pars;  // [nunit] device — overwritten each candidate tree
    int*            d_rell;          // [B] device output (raw dot-product sums)
    int*            h_rell;          // [B] pinned host buffer for fast DMA download
    int B, P, nunit;

    // Batch REPS: process T trees in one kernel instead of T per-tree calls.
    // Allocated by gpuBatchREPSInit; freed by gpuBootstrapMemFree.
    unsigned short* d_batch_pars;   // [max_batch × nunit] device — compacted pattern_pars
    int*            d_batch_rell;   // [max_batch × B] device — output scores
    int*            h_batch_rell;   // [max_batch × B] pinned host — downloaded after kernel
    int             max_batch;      // allocated capacity (0 = not initialized)
};

// Allocate GPU bootstrap memory. Call once when bootstrap+GPU enabled.
// B     = gbo_replicates
// P     = getAlnNPattern()
// nunit = P + VCSIZE_USHORT  (SIMD-padded row size, matching CPU allocation)
GpuBootstrapMem* gpuBootstrapMemAlloc(int B, int P, int nunit);

void gpuBootstrapMemFree(GpuBootstrapMem* mem);

// Upload all B bootstrap sample vectors from CPU to device (call once).
// boot_samples_pars: vector of B CPU-side pointers, each pointing to nunit unsigned short values.
void gpuUploadBootSamples(
    GpuBootstrapMem*                     mem,
    const std::vector<unsigned short*>&  boot_samples_pars
);

// Compute REPS scores for all B replicates against the current candidate tree.
//   pattern_pars: CPU pointer to nunit unsigned short values (_pattern_pars in IQTree).
//   On return, mem->h_rell[i] = sum_p(pattern_pars[p] * boot_samples[i][p]), p in [0,P).
// Synchronous: blocks until results are in h_rell.
void gpuREPSEval(GpuBootstrapMem* mem, const unsigned short* pattern_pars);

// Allocate batch REPS buffers for up to max_batch trees per round.
// Call once after gpuBootstrapMemAlloc when max_treels is known.
void gpuBatchREPSInit(GpuBootstrapMem* mem, int max_batch);

// Grow batch REPS buffers if new_max > current capacity. No-op if already large enough.
void gpuBatchREPSGrow(GpuBootstrapMem* mem, int new_max);

// Batch REPS: compute scores for T trees × B replicates in one kernel launch.
//   h_batch_pars: CPU buffer [T × nunit] — pattern_pars for each tree (row-major).
//   stream: CUDA stream to use; defaults to null stream. Pass a dedicated non-default
//           stream to allow overlap with kernels running on other streams (e.g. ppars).
//   On return, mem->h_batch_rell[t * B + i] = REPS score for tree t, replicate i.
// Synchronous from the caller's perspective (cudaStreamSynchronize called internally).
void gpuBatchREPSEval(GpuBootstrapMem* mem,
                      const unsigned short* h_batch_pars, int T,
                      cudaStream_t stream = 0);

}  // namespace mpbootgpu
