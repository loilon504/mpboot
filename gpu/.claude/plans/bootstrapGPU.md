# GPU Bootstrap Search (`-use_gpu -bb N`)

## Entry point

`gpuBootstrapSearch(params, iqtree, boot_gpu_mem)` in `gpu/src/gpu_init_trees.cu`  
Called from `runBasicMpbootGpu` in `phyloanalysis.cpp` after `mpbootGpu` finishes.

---

## Pipeline (per round)

1. Compute `cutoff_pars` from `iqtree.logl_cutoff` (from prev round); call `resetTreelsRound(mem, cutoff_pars)` — resets `d_treelsFilled=0`, uploads new cutoff to GPU.
2. `resetPoolRound(mem)` — reset `d_poolStop`, `d_globalBest`
3. `gpuStepwiseBuildTrees(mem, k1_count=0, ..., k2_workers, round_size)` — run K2 kernel only (skip K1). Workers write to treels buffer when `score <= cutoff_pars`.
4. `downloadTopology(mem, 0, &base_topo)` — re-download base topology each round (Bug B10 fix)
5. Download treels buffer: `cudaMemcpy(d_treelsFilled → n_treels)`, then scores + back_vf for `n_treels` slots.
6. For each treels slot `s` (0..n_treels-1):
   - Reconstruct `slot_topo` from `h_treels_bvf[s]`
   - `gpuTopoToCpu(&slot_topo, pllInst)` → load into PLL
   - `pllTreeToNewick(...)` → Newick string
   - `readTreeString(newick)` → load into IQTree
   - `initializeAllPartialPars()` + `clearAllPartialLH()`
   - `computeParsimony()` → fills `_pattern_pars`
   - `params->spr_parsimony = false` temporarily → bypass `pllComputePatternParsimony`
   - `saveCurrentTree(-(double)pars)` → GPU REPS eval via `gpuREPSEval(gpu_boot_mem_, _pattern_pars)`
   - `params->spr_parsimony` restored
7. `setCurIt(curIt + max(1, evaluated))`
8. If parsimony improved this round: `stop_rule.addImprovedIteration(curIt)`
9. Update `logl_cutoff` from `treels_logl` (top 10%, activate when size > 200)
10. Every `step_iter_rounds` rounds: `summarizeBootstrap` + `computeBootstrapCorrelation`

---

## Stop condition

```
SC_UNSUCCESS_ITERATION: curIt > lastImproved + unsuccess_iteration
```

For N=295: `unsuccess_iteration = ((295-1)/100 + 1)*100 = 300` (set in `IQTree::init()`).

**Loop condition** (`gpu_init_trees.cu`):
```cpp
while (getCurIt() < gbo_replicates ||
       !stop_rule.meetStopCondition(getCurIt(), cur_correlation))
```
- Phase 1 (`curIt < gbo_replicates`): always continue
- Phase 2 (`curIt >= gbo_replicates`): stop when `curIt > lastImproved + 300`

`addImprovedIteration` is called **only when** `best_pars_seen` improves (not every round).

---

## Treels buffer (GPU-side)

Added to `GpuParsimonyMem`:
```cpp
int    max_treels;              // = k2_workers (e.g. 400)
unsigned int* d_treelsScores;   // [max_treels] scores
int*   d_treelsBackVf;          // [max_treels × kMaxVFaces] topologies
int*   d_treelsFilled;          // atomic fill counter
unsigned int* d_treelsCutoff;   // score ≤ this → write to treels
```

K2 kernel (`runPhase3`) writes to treels at end of each outer iteration (after pool update):
- Lane 0: `if sh.randomMP <= *treels_cutoff: slot = atomicAdd(treels_filled, 1); treels_scores[slot] = sh.randomMP`
- All 32 lanes: copy `back_vf` to `treels_back_vf[slot]`
- No lock needed (atomicAdd gives unique slot per warp)

---

## Fixed bugs

### Bug 1 — SIGSEGV (NULL `perSitePartialPars`)

`_pllFreeParsimonyDataStructures` frees `perSitePartialPars` in `mpbootGpu`. Use IQTree path instead:
```
pllTreeToNewick → readTreeString → initializeAllPartialPars
→ clearAllPartialLH → computeParsimony()   ← fills _pattern_pars correctly
```
Set `params->spr_parsimony = false` to bypass `pllComputePatternParsimony` inside `saveCurrentTree`.

### Bug 2 — Infinite loop (`addImprovedIteration` called every slot)

Fix: only call when `slot_score < best_pars_seen`. Add `curIt < gbo_replicates` to while condition.

### Bug 3 — Pool eval: low diversity, slow convergence

Old: evaluate pool (top-20 globally) → 50 rounds to reach 1000 evaluations.  
Fix: GPU treels buffer → workers write all trees within cutoff → 3 rounds suffice.

---

## Key constraints

- `perSitePartialPars` is **NULL** throughout bootstrap. Do NOT call `pllComputePatternParsimony`.
- `computeParsimony()` fills `_pattern_pars` via IQTree's `partial_pars` system.
- `saveCurrentTree` with `spr_parsimony=false` skips PLL parsimony but runs GPU REPS (line 3429).
- treels write uses `sh.bcast[6]` (BuildSharedT::bcast[11], safe).

---

## Benchmark result (tree1.phy, N=295, -bb 1000, -sprdist 6, seed=1, gpu_device=1)

### With treels buffer (current):
- **3 rounds, 21 seconds wall time**
- Round 1: 400 trees (cutoff=UINT_MAX → all qualify → logl_cutoff set to -6663)
- Round 2: 400 trees (cutoff=-6662)
- Round 3: 304 trees (cutoff=-6662, filter active)
- Total evaluations: 1105
- `bestPars=6662`, `.splits.nex` + `.contree` ✓

### Previous (pool eval, 20 slots/round):
- 50 rounds × 4.4s ≈ **3:42 wall time** (after poolStopThresh=2 optimization)
- `bestPars=6662`

**Speedup: ~7.8×** (treels + early logl_cutoff + curIt += evaluated).
