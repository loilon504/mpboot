---
description: GPU pipeline overview — entry point, 7 steps from CPU alloc to Newick output, and key file map
alwaysApply: true
---

## GPU Pipeline Overview

MPBoot's `makeParsimonyTreeFast` builds `numInitTrees` trees (stepwise-addition + SPR).
CPU runs them serially; GPU runs **K trees in parallel** — one CUDA block per tree, one warp (32 lanes) per block.

**Entry point**: `gpuInitCandidateTrees()` in `gpu/src/gpu_init_trees.cu`
Called from `IQTree::initCandidateTreesParsimony()` when `--use_gpu` is passed.

### Pipeline steps (as printed at runtime)

| Step | Function | What happens |
|------|----------|-------------|
| [1] | `_allocateParsimonyDataStructures` | CPU allocs `parsVect` buffers, compresses alignment into Fitch bit-vectors per partition |
| [2] | `gpuParsimonyMemAlloc` | GPU allocs `d_parsVect[K][2N+1][width][states]`, `d_parsScore[K][2N+1]`, `d_topos[K]` |
| [3] | `uploadTipParsVect` | Reorders tip parsVect CPU→GPU layout and uploads (tips are **read-only**, shared across all K trees) |
| [4] | `cpuToGpuTopology` + `uploadTopology` | Converts PLL pointer-ring → flat int arrays (`GpuTopology`), uploads one initial topology to all K trees |
| [5] | `buildParsimonyTreesKernel` | Each CUDA block builds one tree via **stepwise addition** on GPU |
| [6] | `gpuSprKernel` | Each CUDA block runs **SPR hill-climbing** on its tree |
| [7] | `downloadTopology` + `gpuTopoToCpu` + `pllTreeToNewick` | Download K topologies, convert back to PLL pointer-rings, emit Newick strings |

### Key file map

| File | Purpose |
|------|---------|
| `gpu/src/gpu_init_trees.cu` | Entry point: orchestrates the full pipeline |
| `gpu/src/pars_build.cu` | Stepwise-addition kernel (`buildParsimonyTreesKernel`) |
| `gpu/src/gpu_spr.cu` | SPR hill-climbing kernel (`gpuSprKernel`) |
| `gpu/src/pars_tree.cu` | Memory alloc, topology conversion, upload/download |
| `gpu/include/pars_tree.cuh` | `GpuTopology`, `GpuParsimonyMem`, `warpNewviewStep`, `warpEvaluateScore` |
| `gpu/include/topo_helpers.cuh` | `vfToNum`, `nodepVf`, `vfNextFace`, `vfNnxtFace`, `gpuRandum` |
| `mpboot/sprparsimony.cpp` | CPU reference: `makeParsimonyTreeFast`, `rearrangeParsimony`, `testInsertParsimony`, `newviewParsimonyIterativeFast` |
