#pragma once
#include <string>
#include <vector>

#include "iqtree.h"
#include "pllrepo/src/pll.h"

// Forward declarations to avoid pulling CUDA headers into C++ compilation units
namespace mpbootgpu {
    struct GpuParsimonyMem;
    void gpuParsimonyMemFree(GpuParsimonyMem* mem);
}

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
// out_mem: if non-null and bootstrap enabled, mem is NOT freed and *out_mem is set.
// Caller is responsible for calling gpuParsimonyMemFree(*out_mem) when done.
int mpbootGpu(
    const Params& params,
    IQTree& iqtree,
    int numInitTrees,
    std::vector<std::string>& candidateTrees,  // out, index 1..numInitTrees-1
    GpuParsimonyMem** out_mem = nullptr
);

// GPU iterative hill-climbing loop. Runs K2 rounds of SPR+NNI search from the pool.
//   Bootstrap (-bb): treels → saveCurrentTree (REPS eval), convergence check.
//   Non-bootstrap:   treels → candidateTrees, stops when no improvement.
// Preconditions:
//   - mem: GPU parsimony memory (pool populated from K1 run in mpbootGpu)
//   - Bootstrap: iqtree.gpu_boot_mem_ != nullptr (bootstrap samples uploaded)
//   - _allocateParsimonyDataStructures called
void gpuHillClimbing(
    const Params& params,
    IQTree&        iqtree,
    GpuParsimonyMem* mem
);

}  // namespace mpbootgpu
