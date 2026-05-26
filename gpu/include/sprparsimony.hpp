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

void newviewGpuInit(pllInstance* tr, partitionList* pr);

parsimonyNumber newviewParsimonyGpu(
    pllInstance* tr, partitionList* pr
);

void resetParsVect();

}  // namespace mpbootgpu

#endif
