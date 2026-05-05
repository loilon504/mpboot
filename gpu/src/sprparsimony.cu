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
    parsimonyNumber* d_parsVect,
    unsigned int* d_parsimonyScore,
    const int* __restrict__ d_ti,
    int tiCount,
    const size_t* __restrict__ d_widths,
    const size_t* __restrict__ d_states,
    const size_t* __restrict__ d_parsVectOffset,
    int numPartitions
)
{
    int model = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    size_t width = d_widths[model];
    size_t states = d_states[model];

    bool active = (i < (int)width);

    parsimonyNumber* base = d_parsVect + d_parsVectOffset[model];

    int laneId = threadIdx.x & 31;

    if (i == 0 && model == 0)
    {
        for (int index = 4; index < tiCount; index += 4)
        {
            d_parsimonyScore[(size_t)d_ti[index]] = 0;
        }
    }

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

            for (size_t k = 0; k < states; k++)
            {
                t_A[k] = lBase[k] & rBase[k];
                o_A[k] = lBase[k] | rBase[k];
                t_N |= t_A[k];
            }

            t_N = ~t_N;

            for (size_t k = 0; k < states; k++)
            {
                pBase[k] = t_A[k] | (t_N & o_A[k]);
            }

            bits = (unsigned int)__popc(t_N);
        }

        unsigned int warpSum = warpReduceSum(bits);
        if (laneId == 0 && warpSum > 0)
        {
            atomicAdd(&d_parsimonyScore[pNumber], warpSum);
        }
    }
}

// Kernel that compute parsimonyScore[] to prevent download
__global__ void accumulateParsimonyScoreKernel(
    unsigned int* d_parsimonyScore, const int* __restrict__ d_ti, int tiCount
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }

    for (int index = 4; index < tiCount; index += 4)
    {
        size_t p = (size_t)d_ti[index];
        size_t q = (size_t)d_ti[index + 1];
        size_t r = (size_t)d_ti[index + 2];
        d_parsimonyScore[p] += d_parsimonyScore[q] + d_parsimonyScore[r];
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
    unsigned int* d_parsimonyScore = nullptr;
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
    if (d_buf.d_parsimonyScore)
    {
        cudaFree(d_buf.d_parsimonyScore);
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

    auto toMB = [](size_t bytes)
    {
        return (double)bytes / (1024.0 * 1024.0);
    };

    size_t totalGpuBytes = parsVectBytes + nodeScoresBytes
                           + partitionBytes * 3;  // d_widths + d_states + d_parsVectOffset

    printf("\n==== GPU Memory Usage (newviewGpuInit) ====\n");
    printf("  nodesInTree    : %zu\n", nodesInTree);
    printf("  numPartitions  : %d\n", numPartitions);
    printf("  d_parsVect     : %.2f MB  (%zu bytes)\n", toMB(parsVectBytes), parsVectBytes);
    printf("  d_parsimonyScore: %.2f MB (%zu bytes)\n", toMB(nodeScoresBytes), nodeScoresBytes);
    printf("  d_widths       : %.2f MB  (%zu bytes)\n", toMB(partitionBytes), partitionBytes);
    printf("  d_states       : %.2f MB  (%zu bytes)\n", toMB(partitionBytes), partitionBytes);
    printf("  d_parsVectOffset: %.2f MB (%zu bytes)\n", toMB(partitionBytes), partitionBytes);
    printf("  ─────────────────────────────────────────\n");
    printf("  Total (excl d_ti): %.2f MB  (%zu bytes)\n", toMB(totalGpuBytes), totalGpuBytes);
    printf("===========================================\n\n");

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&d_buf.d_parsVect, parsVectBytes));
    CUDA_CHECK(cudaMalloc(&d_buf.d_parsimonyScore, nodeScoresBytes));
    CUDA_CHECK(cudaMalloc(&d_buf.d_widths, partitionBytes));
    CUDA_CHECK(cudaMalloc(&d_buf.d_states, partitionBytes));
    CUDA_CHECK(cudaMalloc(&d_buf.d_parsVectOffset, partitionBytes));

    CUDA_CHECK(cudaStreamCreate(&d_buf.stream));

    CUDA_CHECK(cudaMemcpyAsync(
        d_buf.d_widths, h_widths.data(), partitionBytes, cudaMemcpyHostToDevice, d_buf.stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        d_buf.d_states, h_states.data(), partitionBytes, cudaMemcpyHostToDevice, d_buf.stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        d_buf.d_parsVectOffset, h_parsVectOffset.data(), partitionBytes, cudaMemcpyHostToDevice,
        d_buf.stream
    ));

    d_buf.numPartitions = numPartitions;
    d_buf.parsVectBytes = parsVectBytes;
    d_buf.nodeScoresBytes = nodeScoresBytes;
}

void uploadParsvect(
    pllInstance* tr, partitionList* pr
)
{
    int numPartitions = d_buf.numPartitions;
    size_t nodesInTree = (size_t)(2 * tr->mxtips + 1);

    std::vector<parsimonyNumber> h_reordered(d_buf.parsVectBytes / sizeof(parsimonyNumber));

    size_t offset = 0;
    for (int m = 0; m < numPartitions; m++)
    {
        size_t width = pr->partitionData[m]->parsimonyLength;
        size_t states = pr->partitionData[m]->states;
        size_t elems = width * states * nodesInTree;
        reorderParsVect(
            pr->partitionData[m]->parsVect, h_reordered.data() + offset, nodesInTree, width, states
        );
        offset += elems;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        d_buf.d_parsVect, h_reordered.data(), d_buf.parsVectBytes, cudaMemcpyHostToDevice,
        d_buf.stream
    ));
}

parsimonyNumber newviewParsimonyGpu(
    pllInstance* tr, partitionList* pr
)
{
    int numPartitions = d_buf.numPartitions;
    size_t nodesInTree = (size_t)(2 * tr->mxtips + 1);
    int tiCount = tr->ti[0];

    // Lazy upload parsVect
    // if (true)
    if (!d_buf.parsVectUploaded)
    {
        uploadParsvect(tr, pr);
        d_buf.parsVectUploaded = true;
    }

    // Upload ti[]
    size_t tiBytes = tiCount * sizeof(int);
    if (d_buf.tiBytes < tiBytes)
    {
        if (d_buf.d_ti)
        {
            CUDA_CHECK(cudaFree(d_buf.d_ti));
        }
        CUDA_CHECK(cudaMalloc(&d_buf.d_ti, tiBytes));
        d_buf.tiBytes = tiBytes;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_buf.d_ti, tr->ti, tiBytes, cudaMemcpyHostToDevice, d_buf.stream));

    // Kernel 1: compute cur[] and local score (parallel per site)
    size_t maxWidth = 0;
    for (int m = 0; m < numPartitions; m++)
    {
        maxWidth = std::max(maxWidth, pr->partitionData[m]->parsimonyLength);
    }
    std::cout << "MAX WIDTH: " << maxWidth << std::endl;

    dim3 block(BLOCK_SIZE);
    dim3 grid((maxWidth + BLOCK_SIZE - 1) / BLOCK_SIZE, numPartitions);

    {
        InlineProfilerTimer p("Kernel 1");
        newviewParsimonyKernel<<<grid, block, 0, d_buf.stream>>>(
            d_buf.d_parsVect, d_buf.d_parsimonyScore, d_buf.d_ti, tiCount, d_buf.d_widths,
            d_buf.d_states, d_buf.d_parsVectOffset, numPartitions
        );
        CUDA_CHECK(cudaStreamSynchronize(d_buf.stream));
    }
    {
        InlineProfilerTimer p("Kernel 2");
        // Kernel 2: compute parsimonyScore[] to prevent download
        accumulateParsimonyScoreKernel<<<1, 1, 0, d_buf.stream>>>(
            d_buf.d_parsimonyScore, d_buf.d_ti, tiCount
        );
        CUDA_CHECK(cudaStreamSynchronize(d_buf.stream));
    }

    {
        InlineProfilerTimer p("Download parsimonyScore");
        CUDA_CHECK(cudaMemcpyAsync(
            tr->parsimonyScore, d_buf.d_parsimonyScore, nodesInTree * sizeof(unsigned int),
            cudaMemcpyDeviceToHost, d_buf.stream
        ));
        CUDA_CHECK(cudaStreamSynchronize(d_buf.stream));
    }

    size_t rootNode = (size_t)tr->ti[tiCount - 4];
    return tr->parsimonyScore[rootNode];
}

void resetParsVect()
{
    d_buf.parsVectUploaded = false;
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