#ifndef MPBOOTGPU_SPRPARSIMONY_CUH_
#define MPBOOTGPU_SPRPARSIMONY_CUH_

// #include "iqtree.h"
#include "pllrepo/src/pll.h"

namespace mpbootgpu
{

struct NodeTriple
{
    int p, q, r;
};

std::vector<std::vector<NodeTriple>> computeTraversalInfoBFS(
    nodeptr root, int maxTips, pllBoolean full, size_t& nNodes
);

void parsimonyGpuInit(pllInstance* tr, partitionList* pr);

void newviewParsimonyGpu(
    pllInstance* tr, partitionList* pr, std::vector<std::vector<NodeTriple>>& levels, size_t nNodes
);

}  // namespace mpbootgpu

#endif
