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

__global__ void fitchLevelKernel(
    parsimonyNumber* d_parsVect,        // đọc left/right, ghi cur (non-const)
    unsigned int* d_levelScores,        // output: totalScore per node [numNodes]
    const int* __restrict__ d_triples,  // {p, q, r} x numNodes
    const size_t* __restrict__ d_widths,
    const size_t* __restrict__ d_states,
    const size_t* __restrict__ d_parsVectOffset,
    int numNodes,
    int numPartitions
)
{
    int nodeIdx = blockIdx.x;
    int model = blockIdx.y;

    if (nodeIdx >= numNodes || model >= numPartitions)
    {
        return;
    }

    size_t width = d_widths[model];
    size_t states = d_states[model];

    size_t pNumber = (size_t)d_triples[nodeIdx * 3 + 0];
    size_t qNumber = (size_t)d_triples[nodeIdx * 3 + 1];
    size_t rNumber = (size_t)d_triples[nodeIdx * 3 + 2];

    parsimonyNumber* base = d_parsVect + d_parsVectOffset[model];

    const parsimonyNumber* left[MAX_STATES];
    const parsimonyNumber* right[MAX_STATES];
    parsimonyNumber* cur[MAX_STATES];

    for (size_t k = 0; k < states; k++)
    {
        left[k] = base + (width * states * qNumber) + width * k;
        right[k] = base + (width * states * rNumber) + width * k;
        cur[k] = base + (width * states * pNumber) + width * k;
    }

    extern __shared__ unsigned int s_score[];
    s_score[threadIdx.x] = 0;

    unsigned int localScore = 0;

    for (unsigned int i = threadIdx.x; i < (unsigned int)width; i += blockDim.x)
    {
        parsimonyNumber t_N = 0;
        parsimonyNumber t_A[MAX_STATES], o_A[MAX_STATES];

        for (size_t k = 0; k < states; k++)
        {
            t_A[k] = left[k][i] & right[k][i];
            o_A[k] = left[k][i] | right[k][i];
            t_N |= t_A[k];
        }

        t_N = ~t_N;

        for (size_t k = 0; k < states; k++)
        {
            cur[k][i] = t_A[k] | (t_N & o_A[k]);
        }

        localScore += (unsigned int)__popc(t_N);
    }

    s_score[threadIdx.x] = localScore;
    __syncthreads();

    // Block reduction
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            s_score[threadIdx.x] += s_score[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_levelScores[nodeIdx], s_score[0]);
    }
}

struct BfsParsimonyBuffers
{
    parsimonyNumber* d_parsVect = nullptr;
    unsigned int* d_levelScores = nullptr;  // score per node trong 1 level
    int* d_triples = nullptr;               // {p,q,r} của 1 level
    size_t* d_widths = nullptr;
    size_t* d_states = nullptr;
    size_t* d_parsVectOffset = nullptr;

    cudaStream_t stream = nullptr;
    int numPartitions = 0;
    size_t triplesBytes = 0;     // capacity hiện tại của d_triples
    size_t levelScoreBytes = 0;  // capacity hiện tại của d_levelScores
    size_t parsVectBytes = 0;
};

static BfsParsimonyBuffers d_buf;

// =============================================================================
// bfsParsimonyGpuCleanup
// =============================================================================
void bfsParsimonyGpuCleanup()
{
    if (d_buf.stream)
    {
        cudaStreamDestroy(d_buf.stream);
    }
    if (d_buf.d_parsVect)
    {
        cudaFree(d_buf.d_parsVect);
    }
    if (d_buf.d_levelScores)
    {
        cudaFree(d_buf.d_levelScores);
    }
    if (d_buf.d_triples)
    {
        cudaFree(d_buf.d_triples);
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

// init constant data for pll in gpu
void parsimonyGpuInit(
    pllInstance* tr, partitionList* pr
)
{
    int numPartitions = pr->numberOfPartitions;
    size_t nodesInTree = (size_t)(2 * tr->mxtips) + 1;

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

    cudaMalloc(&d_buf.d_parsVect, parsVectBytes);
    cudaMalloc(&d_buf.d_widths, partitionBytes);
    cudaMalloc(&d_buf.d_states, partitionBytes);
    cudaMalloc(&d_buf.d_parsVectOffset, partitionBytes);

    cudaStreamCreate(&d_buf.stream);

    // Upload dữ liệu tĩnh
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

    cudaStreamSynchronize(d_buf.stream);

    d_buf.parsVectBytes = parsVectBytes;
    d_buf.numPartitions = numPartitions;
}

std::vector<std::vector<NodeTriple>> computeTraversalInfoBFS(
    nodeptr root, int maxTips, pllBoolean full, size_t& numNodes
)
{
    std::vector<std::vector<NodeTriple>> levels;

    std::queue<std::pair<nodeptr, int>> bfsQueue;
    bfsQueue.push({root, 0});

    while (!bfsQueue.empty())
    {
        auto [p, level] = bfsQueue.front();
        bfsQueue.pop();

        nodeptr q = p->next->back;
        nodeptr r = p->next->next->back;

        if (!p->xPars)
        {
            nodeptr s;

            if ((s = p->next)->xPars || (s = s->next)->xPars)
            {
                p->xPars = s->xPars;
                s->xPars = 0;
            }
        }

        if ((int)levels.size() <= level)
        {
            levels.resize(level + 1);
        }

        levels[level].push_back({p->number, q->number, r->number});
        numNodes++;

        bool visitQ = (q->number > maxTips) && (full || !q->xPars);
        bool visitR = (r->number > maxTips) && (full || !r->xPars);

        if (visitQ)
        {
            bfsQueue.push({q, level + 1});
        }
        if (visitR)
        {
            bfsQueue.push({r, level + 1});
        }
    }

    std::reverse(levels.begin(), levels.end());
    return std::move(levels);
}

static bool initialized = false;

void newviewParsimonyGpu(
    pllInstance* tr, partitionList* pr, std::vector<std::vector<NodeTriple>>& levels, size_t nNodes
)
{
    if (!initialized)
    {
        // Upload parsVect only one time
        size_t nodesInTree = (size_t)(2 * tr->mxtips) + 1;
        size_t offset = 0;
        for (int m = 0; m < d_buf.numPartitions; m++)
        {
            size_t elems = pr->partitionData[m]->parsimonyLength * pr->partitionData[m]->states
                           * nodesInTree;
            size_t bytes = elems * sizeof(parsimonyNumber);
            cudaMemcpyAsync(
                (char*)d_buf.d_parsVect + offset, pr->partitionData[m]->parsVect, bytes,
                cudaMemcpyHostToDevice, d_buf.stream
            );
            offset += bytes;
        }

        initialized = true;
    }
    // size_t nodesInTree = (size_t)(2 * tr->mxtips) + 1;
    // size_t offset = 0;
    // for (int m = 0; m < d_buf.numPartitions; m++)
    // {
    //     size_t elems = pr->partitionData[m]->parsimonyLength * pr->partitionData[m]->states
    //                    * nodesInTree;
    //     size_t bytes = elems * sizeof(parsimonyNumber);
    //     cudaMemcpyAsync(
    //         (char*)d_buf.d_parsVect + offset, pr->partitionData[m]->parsVect, bytes,
    //         cudaMemcpyHostToDevice, d_buf.stream
    //     );
    //     offset += bytes;
    // }

    int numPartitions = d_buf.numPartitions;

    std::vector<unsigned int> h_allScores(nNodes, 0);

    size_t curNode = 0;
    for (const auto& level : levels)
    {
        int numNodes = (int)level.size();

        // Upload triples of this level
        size_t triplesBytes = numNodes * 3 * sizeof(int);
        if (d_buf.triplesBytes < triplesBytes)
        {
            if (d_buf.d_triples)
            {
                cudaFree(d_buf.d_triples);
            }
            cudaMalloc(&d_buf.d_triples, triplesBytes);
            d_buf.triplesBytes = triplesBytes;
        }

        std::vector<int> h_triples(numNodes * 3);
        for (int n = 0; n < numNodes; n++)
        {
            h_triples[n * 3 + 0] = level[n].p;
            h_triples[n * 3 + 1] = level[n].q;
            h_triples[n * 3 + 2] = level[n].r;
        }
        cudaMemcpyAsync(
            d_buf.d_triples, h_triples.data(), triplesBytes, cudaMemcpyHostToDevice, d_buf.stream
        );

        // Upload d_levelScores
        size_t levelScoreBytes = numNodes * sizeof(unsigned int);
        if (d_buf.levelScoreBytes < levelScoreBytes)
        {
            if (d_buf.d_levelScores)
            {
                cudaFree(d_buf.d_levelScores);
            }
            cudaMalloc(&d_buf.d_levelScores, levelScoreBytes);
            d_buf.levelScoreBytes = levelScoreBytes;
        }
        cudaMemsetAsync(d_buf.d_levelScores, 0, levelScoreBytes, d_buf.stream);

        // Launch kernel
        const size_t blockSize = 256;
        dim3 grid(numNodes, numPartitions);
        dim3 block(blockSize);
        size_t sharedBytes = blockSize * sizeof(unsigned int);

        fitchLevelKernel<<<grid, block, sharedBytes, d_buf.stream>>>(
            d_buf.d_parsVect, d_buf.d_levelScores, d_buf.d_triples, d_buf.d_widths, d_buf.d_states,
            d_buf.d_parsVectOffset, numNodes, numPartitions
        );
        cudaStreamSynchronize(d_buf.stream);

        // Download scores of this level
        std::vector<unsigned int> h_levelScores(numNodes);
        cudaMemcpy(
            h_levelScores.data(), d_buf.d_levelScores, levelScoreBytes, cudaMemcpyDeviceToHost
        );

        // Save into h_allScores
        for (int n = 0; n < numNodes; n++)
        {
            h_allScores[curNode++] = h_levelScores[n];
        }
    }

    // Download parsVect
    // {
    //     size_t nodesInTree = (size_t)(2 * tr->mxtips) + 1;
    //     size_t offset = 0;
    //     for (int m = 0; m < numPartitions; m++)
    //     {
    //         size_t elems = pr->partitionData[m]->parsimonyLength * pr->partitionData[m]->states
    //                        * nodesInTree;
    //         size_t bytes = elems * sizeof(parsimonyNumber);
    //         cudaMemcpyAsync(
    //             pr->partitionData[m]->parsVect, (char*)d_buf.d_parsVect + offset, bytes,
    //             cudaMemcpyDeviceToHost, d_buf.stream
    //         );
    //         offset += bytes;
    //     }
    //     cudaStreamSynchronize(d_buf.stream);
    // }

    // Update tr->parsimonyScore[]
    curNode = 0;
    for (const auto& level : levels)
    {
        for (const auto& triple : level)
        {
            size_t p = triple.p, q = triple.q, r = triple.r;
            tr->parsimonyScore[p] = h_allScores[curNode++] + tr->parsimonyScore[q]
                                    + tr->parsimonyScore[r];
        }
    }
}

static void newviewParsimonyIterativeFast(
    pllInstance* tr, partitionList* pr, int perSiteScores
)
{
    if (pllCostMatrix)
    {
        return newviewSankoffParsimonyIterativeFast(tr, pr, perSiteScores);
    }
    int model, *ti = tr->ti, count = ti[0], index;

    for (index = 4; index < count; index += 4)
    {
        unsigned int totalScore = 0;

        size_t pNumber = (size_t)ti[index], qNumber = (size_t)ti[index + 1],
               rNumber = (size_t)ti[index + 2];

        for (model = 0; model < pr->numberOfPartitions; model++)
        {
            size_t k, states = pr->partitionData[model]->states,
                      width = pr->partitionData[model]->parsimonyLength;

            unsigned int i;

            parsimonyNumber *left[32], *right[32], *cur[32];

            parsimonyNumber o_A[32], t_A[32], t_N;

            assert(states <= 32);

            for (k = 0; k < states; k++)
            {
                left[k] = &(
                    pr->partitionData[model]->parsVect[(width * states * qNumber) + width * k]
                );
                right[k] = &(
                    pr->partitionData[model]->parsVect[(width * states * rNumber) + width * k]
                );
                cur[k] = &(
                    pr->partitionData[model]->parsVect[(width * states * pNumber) + width * k]
                );
            }

            for (i = 0; i < width; i++)
            {
                t_N = 0;

                for (k = 0; k < states; k++)
                {
                    t_A[k] = left[k][i] & right[k][i];
                    o_A[k] = left[k][i] | right[k][i];
                    t_N = t_N | t_A[k];
                }

                t_N = ~t_N;

                for (k = 0; k < states; k++)
                {
                    cur[k][i] = t_A[k] | (t_N & o_A[k]);
                }

                totalScore += ((unsigned int)__builtin_popcount(t_N));
            }
        }

        tr->parsimonyScore[pNumber] = totalScore + tr->parsimonyScore[rNumber]
                                      + tr->parsimonyScore[qNumber];
    }
}

}  // namespace mpbootgpu