#ifndef MPBOOTGPU_PLL_HPP_
#define MPBOOTGPU_PLL_HPP_

#include <cuda_runtime.h>
#include <vector>
#include "pllrepo/src/pll.h"

namespace mpbootgpu
{

struct GpuNode
{
    int nextIdx; 
    int backIdx; 
    int number;  
    int xPars;   
};

struct GpuTree
{
    GpuNode*      d_nodes        = nullptr;
    int*          d_ti           = nullptr;

    // --- parsVect layout [node][site][state] ---
    parsimonyNumber* d_parsVect  = nullptr;

    // --- Score arrays ---
    unsigned int* d_nodeScores     = nullptr;
    unsigned int* d_parsimonyScore = nullptr;

    size_t*       d_widths          = nullptr;
    size_t*       d_states          = nullptr;
    size_t*       d_parsVectOffset  = nullptr;

    cudaStream_t  stream            = nullptr;

    // --- Host-side metadata ---
    int    maxNodes      = 0;
    int    numPartitions = 0;
    size_t parsVectBytes = 0;
    size_t nodeScoresBytes = 0;
    size_t tiCapacity    = 0; 
    bool   parsVectUploaded = false;
};

}  // namespace mpbootgpu

#endif
