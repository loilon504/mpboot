#include <queue>
#include <string>

#include "gpu/include/profiler.hpp"
#include "gpu/include/pll.hpp"
#include "gpu/include/utils.cuh"
#include "pllrepo/src/pll.h"

namespace mpbootgpu
{
inline void flattenNodes(
    pllInstance*          tr,
    std::vector<GpuNode>& h_nodes   // output, size = maxNodes
)
{
    int maxNodes = 2 * tr->mxtips + 1;
    h_nodes.resize(maxNodes);

    // Map pointer → index using PLL's number field.
    // PLL guarantees nodep[i]->number == i.
    auto ptrToIdx = [&](nodeptr p) -> int {
        if (!p) return -1;
        return p->number;  // index trong nodep[]
    };

    for (int i = 1; i < maxNodes; i++)
    {
        nodeptr p = tr->nodep[i];
        if (!p) { h_nodes[i] = {-1, -1, i, 0}; continue; }

        h_nodes[i].number  = p->number;
        h_nodes[i].backIdx = ptrToIdx(p->back);
        h_nodes[i].xPars   = p->xPars;

        // next: PLL uses a 3-node ring for inner nodes; leaves point to themselves.
        h_nodes[i].nextIdx = ptrToIdx(p->next);
    }
}

inline void gpuTreeInit(
    pllInstance*  tr,
    partitionList* pr,
    GpuTree&      gt
)
{
    int    numPartitions = pr->numberOfPartitions;
    int    maxNodes      = 2 * tr->mxtips + 1;
    size_t nodesInTree   = (size_t)maxNodes;

    // Compute parsVect layout.
    std::vector<size_t> h_widths(numPartitions);
    std::vector<size_t> h_states(numPartitions);
    std::vector<size_t> h_offsets(numPartitions);
    size_t totalElems = 0;

    for (int m = 0; m < numPartitions; m++) {
        h_widths[m]  = pr->partitionData[m]->parsimonyLength;
        h_states[m]  = pr->partitionData[m]->states;
        h_offsets[m] = totalElems;
        totalElems  += h_widths[m] * h_states[m] * nodesInTree;
    }

    size_t parsVectBytes   = totalElems * sizeof(parsimonyNumber);
    size_t partitionBytes  = numPartitions * sizeof(size_t);
    size_t nodeScoresBytes = nodesInTree * sizeof(unsigned int);
    size_t nodesBytes      = maxNodes * sizeof(GpuNode);
    size_t tiInitBytes     = maxNodes * 4 * sizeof(int);  // initial capacity

    // Print memory usage.
    auto toMB = [](size_t b) { return (double)b / (1024.0 * 1024.0); };
    printf("\n==== GPU Memory (gpuTreeInit) ====\n");
    printf("  d_parsVect      : %.2f MB\n", toMB(parsVectBytes));
    printf("  d_nodes         : %.2f MB\n", toMB(nodesBytes));
    printf("  d_nodeScores    : %.2f MB\n", toMB(nodeScoresBytes));
    printf("  d_parsimonyScore: %.2f MB\n", toMB(nodeScoresBytes));
    printf("  d_ti (initial)  : %.2f MB\n", toMB(tiInitBytes));
    printf("  partition meta  : %.2f MB\n", toMB(partitionBytes * 3));
    printf("  Total           : %.2f MB\n",
           toMB(parsVectBytes + nodesBytes + nodeScoresBytes * 2
                + tiInitBytes + partitionBytes * 3));
    printf("==================================\n\n");

    // Allocate GPU buffers.
    CUDA_CHECK(cudaMalloc(&gt.d_parsVect,        parsVectBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_nodes,           nodesBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_nodeScores,      nodeScoresBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_parsimonyScore,  nodeScoresBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_widths,          partitionBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_states,          partitionBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_parsVectOffset,  partitionBytes));
    CUDA_CHECK(cudaMalloc(&gt.d_ti,              tiInitBytes));

    CUDA_CHECK(cudaStreamCreate(&gt.stream));

    // Upload partition metadata (constant across runs).
    CUDA_CHECK(cudaMemcpyAsync(gt.d_widths,  h_widths.data(),  partitionBytes, cudaMemcpyHostToDevice, gt.stream));
    CUDA_CHECK(cudaMemcpyAsync(gt.d_states,  h_states.data(),  partitionBytes, cudaMemcpyHostToDevice, gt.stream));
    CUDA_CHECK(cudaMemcpyAsync(gt.d_parsVectOffset, h_offsets.data(), partitionBytes, cudaMemcpyHostToDevice, gt.stream));

    gt.maxNodes       = maxNodes;
    gt.numPartitions  = numPartitions;
    gt.parsVectBytes  = parsVectBytes;
    gt.nodeScoresBytes = nodeScoresBytes;
    gt.tiCapacity     = tiInitBytes;
    gt.parsVectUploaded = false;
}

inline void gpuTreeUploadNodes(
    pllInstance* tr,
    GpuTree&     gt
)
{
    std::vector<GpuNode> h_nodes;
    flattenNodes(tr, h_nodes);

    CUDA_CHECK(cudaMemcpyAsync(
        gt.d_nodes, h_nodes.data(),
        gt.maxNodes * sizeof(GpuNode),
        cudaMemcpyHostToDevice, gt.stream
    ));
}

inline void gpuTreeUploadTi(
    pllInstance* tr,
    GpuTree&     gt
)
{
    int    tiCount = tr->ti[0];
    size_t tiBytes = tiCount * sizeof(int);

    // Resize if needed.
    if (gt.tiCapacity < tiBytes) {
        CUDA_CHECK(cudaFree(gt.d_ti));
        CUDA_CHECK(cudaMalloc(&gt.d_ti, tiBytes));
        gt.tiCapacity = tiBytes;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        gt.d_ti, tr->ti, tiBytes,
        cudaMemcpyHostToDevice, gt.stream
    ));
}

inline void gpuTreeUploadParsVect(
    pllInstance*  tr,
    partitionList* pr,
    GpuTree&      gt
)
{
    if (gt.parsVectUploaded) return;

    size_t nodesInTree = (size_t)gt.maxNodes;
    std::vector<parsimonyNumber> h_reordered(gt.parsVectBytes / sizeof(parsimonyNumber));

    size_t offset = 0;
    for (int m = 0; m < gt.numPartitions; m++) {
        size_t width  = pr->partitionData[m]->parsimonyLength;
        size_t states = pr->partitionData[m]->states;
        size_t elems  = width * states * nodesInTree;

        const parsimonyNumber* src = pr->partitionData[m]->parsVect;
        parsimonyNumber*       dst = h_reordered.data() + offset;

        // Reorder: [node][state][site] → [node][site][state]
        for (size_t node = 0; node < nodesInTree; node++)
            for (size_t i = 0; i < width; i++)
                for (size_t k = 0; k < states; k++)
                    dst[states * (width * node + i) + k] =
                        src[width * states * node + width * k + i];

        offset += elems;
    }

    CUDA_CHECK(cudaMemcpyAsync(
        gt.d_parsVect, h_reordered.data(), gt.parsVectBytes,
        cudaMemcpyHostToDevice, gt.stream
    ));
    gt.parsVectUploaded = true;
}

inline void gpuTreeCleanup(GpuTree& gt)
{
    if (gt.stream)            cudaStreamDestroy(gt.stream);
    if (gt.d_parsVect)        cudaFree(gt.d_parsVect);
    if (gt.d_nodes)           cudaFree(gt.d_nodes);
    if (gt.d_nodeScores)      cudaFree(gt.d_nodeScores);
    if (gt.d_parsimonyScore)  cudaFree(gt.d_parsimonyScore);
    if (gt.d_ti)              cudaFree(gt.d_ti);
    if (gt.d_widths)          cudaFree(gt.d_widths);
    if (gt.d_states)          cudaFree(gt.d_states);
    if (gt.d_parsVectOffset)  cudaFree(gt.d_parsVectOffset);
    gt = {};
}

}  // namespace mpbootgpu