#pragma once
#include <cuda_runtime.h>
#include "pars_tree.cuh"

namespace mpbootgpu
{

// Compute per-pattern parsimony for all n_treels entries.
// Stores result in d_treels_ptn_pars[t × nptn_padded] (device).
// Fitch mode only (states in {4, 20}). Returns false if states not supported.
//
// Reuses d_parsVect[K] as scratch: tip data in slot 0 is broadcast; inner nodes overwritten.
// Process in batches of K if n_treels > K.
bool gpuComputeTreelsPatternPars(
    GpuParsimonyMem*   mem,
    int                n_treels,
    int                start_vf,     // = 0 (tip 1 vface, always)
    int                nptn,         // actual number of patterns
    int                nptn_padded,  // padded to VCSIZE_USHORT boundary
    uint16_t*          d_treels_ptn_pars,  // [n_treels × nptn_padded] device output
    cudaStream_t       stream
);

}  // namespace mpbootgpu
