#pragma once
#include <string>
#include <vector>

#include "iqtree.h"
#include "pllrepo/src/pll.h"

// Forward declarations to avoid including heavy headers here
struct Params;
class IQTree;

namespace mpbootgpu
{

// Build numInitTrees-1 parsimony trees on GPU using stepwise addition + SPR.
// Results go into candidateTrees[1..numInitTrees-1] as Newick strings.
// Trees are also registered into iqtree.candidateTrees (via update + setBestTree)
// using CPU-recomputed parsimony scores.
//
// Returns the number of trees successfully built.
int mpbootGpu(
    const Params& params,
    IQTree& iqtree,
    int numInitTrees,
    std::vector<std::string>& candidateTrees  // out, index 1..numInitTrees-1
);

}  // namespace mpbootgpu
