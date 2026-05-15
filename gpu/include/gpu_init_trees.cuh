#pragma once
#include <string>
#include <vector>
#include "pllrepo/src/pll.h"

// Forward declarations to avoid including heavy headers here
struct Params;
class IQTree;

namespace mpbootgpu
{

// Build numInitTrees-1 parsimony trees on GPU using stepwise addition.
// Results go into candidateTrees[1..numInitTrees-1] as Newick strings
// (same layout as the CPU OpenMP path in initCandidateTreeSet).
//
// SPR hill-climbing is NOT done on GPU — after this call the caller must
// run rearrangeParsimony on each downloaded tree (or skip SPR for GPU trees).
//
// Returns the number of distinct trees actually built.
int mpbootGpu(
    const Params&        params,
    IQTree&              iqtree,
    int                  numInitTrees,
    std::vector<std::string>& candidateTrees   // out, index 1..numInitTrees-1
);

}  // namespace mpbootgpu
