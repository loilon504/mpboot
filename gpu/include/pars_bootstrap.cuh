#pragma once
#include <vector>

namespace mpbootgpu
{

struct GpuBootstrapMem
{
    unsigned short* d_boot_samples;  // [B × nunit] device — uploaded once at init
    unsigned short* d_pattern_pars;  // [nunit] device — overwritten each candidate tree
    int*            d_rell;          // [B] device output (raw dot-product sums)
    int*            h_rell;          // [B] pinned host buffer for fast DMA download
    int B, P, nunit;
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

}  // namespace mpbootgpu
