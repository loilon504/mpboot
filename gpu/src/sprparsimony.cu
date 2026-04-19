#include <queue>
#include <string>

#include "gpu/include/profiler.hpp"
#include "gpu/include/sprparsimony.hpp"
#include "gpu/include/utils.cuh"
#include "pllrepo/src/pll.h"

namespace mpbootgpu
{

#define BLOCK_SIZE 256
#define MAX_STATES 32

// Warp reduce
__device__ __forceinline__ unsigned int warpReduceSum(
    unsigned int val
)
{
    const unsigned int FULL_MASK = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Layout GPU: [node][site][state]  →  base[states*(width*node + i) + k]
// layout CPU: [node][state][site] = base[width*states*node + width*k + i])
__global__ void newviewParsimonyKernel(
    parsimonyNumber*            d_parsVect,       
    unsigned int*               d_nodeScores,     
    const int*  __restrict__    d_ti,             
    int                         tiCount,          
    const size_t* __restrict__  d_widths,         
    const size_t* __restrict__  d_states,         
    const size_t* __restrict__  d_parsVectOffset, 
    int                         numPartitions
)
{
    int model = blockIdx.y;
    int i     = blockIdx.x * blockDim.x + threadIdx.x;

    size_t width  = d_widths[model];
    size_t states = d_states[model];

    bool active = (i < (int)width);

    parsimonyNumber* base = d_parsVect + d_parsVectOffset[model];

    int laneId = threadIdx.x & 31;

    for (int index = 4; index < tiCount; index += 4)
    {
        size_t pNumber = (size_t)d_ti[index];
        size_t qNumber = (size_t)d_ti[index + 1];
        size_t rNumber = (size_t)d_ti[index + 2];

        unsigned int bits = 0;

        if (active)
        {
            parsimonyNumber* lBase = base + states * (width * qNumber + i);  // [q][i][0..states]
            parsimonyNumber* rBase = base + states * (width * rNumber + i);  // [r][i][0..states]
            parsimonyNumber* pBase = base + states * (width * pNumber + i);  // [p][i][0..states]

            parsimonyNumber t_N = 0;
            parsimonyNumber t_A[MAX_STATES], o_A[MAX_STATES];

            for (size_t k = 0; k < states; k++) {
                t_A[k] = lBase[k] & rBase[k];
                o_A[k] = lBase[k] | rBase[k];
                t_N   |= t_A[k];
            }

            t_N = ~t_N;

            for (size_t k = 0; k < states; k++)
                pBase[k] = t_A[k] | (t_N & o_A[k]);

            bits = (unsigned int)__popc(t_N);
        }

        unsigned int warpSum = warpReduceSum(bits);
        if (laneId == 0 && warpSum > 0)
            atomicAdd(&d_nodeScores[pNumber], warpSum);
    }
}


// =============================================================================
// reorderParsVect
// Change layout host [node][state][site] → layout GPU [node][site][state]
// host: parsVect[width*states*node + width*k + i]
// gpu:  parsVect[states*(width*node + i) + k]
// =============================================================================
static void reorderParsVect(
    const parsimonyNumber* src,  // layout host: [node][state][site]
    parsimonyNumber* dst,        // layout GPU:  [node][site][state]
    size_t numNodes,
    size_t width,
    size_t states
)
{
    for (size_t node = 0; node < numNodes; node++)
    {
        for (size_t i = 0; i < width; i++)
        {
            for (size_t k = 0; k < states; k++)
            {
                // src index: [node][state k][site i]
                size_t srcIdx = width * states * node + width * k + i;
                // dst index: [node][site i][state k]
                size_t dstIdx = states * (width * node + i) + k;
                dst[dstIdx] = src[srcIdx];
            }
        }
    }
}

struct NewviewGpuBuffers
{
    parsimonyNumber* d_parsVect = nullptr;
    unsigned int* d_nodeScores = nullptr;
    int* d_ti = nullptr;
    size_t* d_widths = nullptr;
    size_t* d_states = nullptr;
    size_t* d_parsVectOffset = nullptr;

    cudaStream_t stream = nullptr;
    int numPartitions = 0;
    size_t parsVectBytes = 0;
    size_t nodeScoresBytes = 0;
    size_t tiBytes = 0;
    bool parsVectUploaded = false;
};

static NewviewGpuBuffers d_buf;

void newviewGpuCleanup()
{
    if (d_buf.stream)
    {
        cudaStreamDestroy(d_buf.stream);
    }
    if (d_buf.d_parsVect)
    {
        cudaFree(d_buf.d_parsVect);
    }
    if (d_buf.d_nodeScores)
    {
        cudaFree(d_buf.d_nodeScores);
    }
    if (d_buf.d_ti)
    {
        cudaFree(d_buf.d_ti);
    }
    if (d_buf.d_widths)
    {
        cudaFree(d_buf.d_widths);
    }
    if (d_buf.d_states)
    {
        cudaFree(d_buf.d_states);
    }
    if (d_buf.d_parsVectOffset)
    {
        cudaFree(d_buf.d_parsVectOffset);
    }
    d_buf = {};
}

void newviewGpuInit(
    pllInstance* tr, partitionList* pr
)
{
    int numPartitions = pr->numberOfPartitions;
    size_t nodesInTree = (size_t)(2 * tr->mxtips + 1);

    std::vector<size_t> h_widths(numPartitions);
    std::vector<size_t> h_states(numPartitions);
    std::vector<size_t> h_parsVectOffset(numPartitions);
    size_t totalElems = 0;

    for (int m = 0; m < numPartitions; m++)
    {
        h_widths[m] = pr->partitionData[m]->parsimonyLength;
        h_states[m] = pr->partitionData[m]->states;
        h_parsVectOffset[m] = totalElems;
        totalElems += h_widths[m] * h_states[m] * nodesInTree;
    }

    size_t parsVectBytes = totalElems * sizeof(parsimonyNumber);
    size_t partitionBytes = numPartitions * sizeof(size_t);
    size_t nodeScoresBytes = nodesInTree * sizeof(unsigned int);

    cudaMalloc(&d_buf.d_parsVect, parsVectBytes);
    cudaMalloc(&d_buf.d_nodeScores, nodeScoresBytes);
    cudaMalloc(&d_buf.d_widths, partitionBytes);
    cudaMalloc(&d_buf.d_states, partitionBytes);
    cudaMalloc(&d_buf.d_parsVectOffset, partitionBytes);

    cudaStreamCreate(&d_buf.stream);

    cudaMemcpyAsync(
        d_buf.d_widths, h_widths.data(), partitionBytes, cudaMemcpyHostToDevice, d_buf.stream
    );
    cudaMemcpyAsync(
        d_buf.d_states, h_states.data(), partitionBytes, cudaMemcpyHostToDevice, d_buf.stream
    );
    cudaMemcpyAsync(
        d_buf.d_parsVectOffset, h_parsVectOffset.data(), partitionBytes, cudaMemcpyHostToDevice,
        d_buf.stream
    );
    // cudaStreamSynchronize(d_buf.stream);

    d_buf.numPartitions = numPartitions;
    d_buf.parsVectBytes = parsVectBytes;
    d_buf.nodeScoresBytes = nodeScoresBytes;
}

void newviewParsimonyGpu(
    pllInstance* tr, partitionList* pr
)
{
    int numPartitions = d_buf.numPartitions;
    size_t nodesInTree = (size_t)(2 * tr->mxtips + 1);
    int tiCount = tr->ti[0];

    // Upload parsVect only once
    // if (!d_buf.parsVectUploaded || d_buf.parsVectUploaded)
    if (!d_buf.parsVectUploaded)
    {
        std::vector<parsimonyNumber> h_reordered(d_buf.parsVectBytes / sizeof(parsimonyNumber));

        size_t offset = 0;
        for (int m = 0; m < numPartitions; m++)
        {
            size_t width = pr->partitionData[m]->parsimonyLength;
            size_t states = pr->partitionData[m]->states;
            size_t elems = width * states * nodesInTree;

            reorderParsVect(
                pr->partitionData[m]->parsVect,  // src: layout host
                h_reordered.data() + offset,     // dst: layout GPU
                nodesInTree, width, states
            );
            offset += elems;
        }

        cudaMemcpyAsync(
            d_buf.d_parsVect, h_reordered.data(), d_buf.parsVectBytes, cudaMemcpyHostToDevice,
            d_buf.stream
        );

        d_buf.parsVectUploaded = true;
    }

    // Upload ti[] (postorder traversal)
    size_t tiBytes = tiCount * sizeof(int);
    if (d_buf.tiBytes < tiBytes)
    {
        if (d_buf.d_ti)
        {
            cudaFree(d_buf.d_ti);
        }
        cudaMalloc(&d_buf.d_ti, tiBytes);
        d_buf.tiBytes = tiBytes;
    }
    cudaMemcpyAsync(d_buf.d_ti, tr->ti, tiBytes, cudaMemcpyHostToDevice, d_buf.stream);

    // Reset nodeScores
    cudaMemsetAsync(d_buf.d_nodeScores, 0, d_buf.nodeScoresBytes, d_buf.stream);

    // ------------------------------------------------------------------
    // Launch kernel
    //    Grid.x = ceil(maxWidth / BLOCK_SIZE)  — cover width của partition lớn nhất
    //    Grid.y = numPartitions
    // ------------------------------------------------------------------
    size_t maxWidth = 0;
    for (int m = 0; m < numPartitions; m++)
    {
        maxWidth = std::max(maxWidth, pr->partitionData[m]->parsimonyLength);
    }

    size_t blockSize = 32;
    dim3 block(blockSize);
    dim3 grid((maxWidth + blockSize - 1) / blockSize, numPartitions);

    newviewParsimonyKernel<<<grid, block, 0, d_buf.stream>>>(
        d_buf.d_parsVect, d_buf.d_nodeScores, d_buf.d_ti, tiCount, d_buf.d_widths, d_buf.d_states,
        d_buf.d_parsVectOffset, numPartitions
    );

    // Download parsVect
    // size_t offset = 0;
    // for (int m = 0; m < numPartitions; m++)
    // {
    //     size_t elems = pr->partitionData[m]->parsimonyLength * pr->partitionData[m]->states
    //                     * nodesInTree;
    //     cudaMemcpyAsync(
    //         pr->partitionData[m]->parsVect, (char*)d_buf.d_parsVect + offset,
    //         elems * sizeof(parsimonyNumber), cudaMemcpyDeviceToHost, d_buf.stream
    //     );
    //     offset += elems * sizeof(parsimonyNumber);
    // }

    // Download nodeScores → update parsimonyScore[]
    std::vector<unsigned int> h_nodeScores(nodesInTree);
    cudaMemcpyAsync(
        h_nodeScores.data(), d_buf.d_nodeScores, nodesInTree * sizeof(unsigned int),
        cudaMemcpyDeviceToHost, d_buf.stream
    );
    cudaStreamSynchronize(d_buf.stream);

    for (int index = 4; index < tiCount; index += 4)
    {
        size_t p = (size_t)tr->ti[index];
        size_t q = (size_t)tr->ti[index + 1];
        size_t r = (size_t)tr->ti[index + 2];
        tr->parsimonyScore[p] = h_nodeScores[p] + tr->parsimonyScore[q] + tr->parsimonyScore[r];
    }
}

// static void newviewParsimonyIterativeFast(
//     pllInstance* tr, partitionList* pr, int perSiteScores
// )
// {
//     if (pllCostMatrix)
//     {
//         return newviewSankoffParsimonyIterativeFast(tr, pr, perSiteScores);
//     }
//     int model, *ti = tr->ti, count = ti[0], index;

//     for (index = 4; index < count; index += 4)
//     {
//         unsigned int totalScore = 0;

//         size_t pNumber = (size_t)ti[index], qNumber = (size_t)ti[index + 1],
//                rNumber = (size_t)ti[index + 2];

//         for (model = 0; model < pr->numberOfPartitions; model++)
//         {
//             size_t k, states = pr->partitionData[model]->states,
//                       width = pr->partitionData[model]->parsimonyLength;

//             unsigned int i;

//             parsimonyNumber *left[32], *right[32], *cur[32];

//             parsimonyNumber o_A[32], t_A[32], t_N;

//             assert(states <= 32);

//             for (k = 0; k < states; k++)
//             {
//                 left[k] = &(
//                     pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]
//                 );
//                 right[k] = &(
//                     pr->partitionData[model]->parsVect[(width * states * rNumber) + width * k]
//                 );
//                 cur[k] = &(
//                     pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]
//                 );
//             }

//             for (i = 0; i < width; i++)
//             {
//                 t_N = 0;

//                 for (k = 0; k < states; k++)
//                 {
//                     t_A[k] = left[k][i] & right[k][i];
//                     o_A[k] = left[k][i] | right[k][i];
//                     t_N = t_N | t_A[k];
//                 }

//                 t_N = ~t_N;

//                 for (k = 0; k < states; k++)
//                 {
//                     cur[k][i] = t_A[k] | (t_N & o_A[k]);
//                 }

//                 totalScore += ((unsigned int)__builtin_popcount(t_N));
//             }
//         }

//         tr->parsimonyScore[pNumber] = totalScore + tr->parsimonyScore[rNumber]
//                                       + tr->parsimonyScore[qNumber];
//     }
// }

}  // namespace mpbootgpu