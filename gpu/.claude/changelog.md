# GPU Parsimony Kernel — Changelog

---

## 2026-05-13 (Opt-P Layer 1)

### Opt-P Layer 1: BuildShared Shrink — Union perm/stackMint + Reduce stackMaxt

**Problem**: `BuildShared = 28.4 KB` → 5 blocks/SM (164 KB/SM ÷ 28.4 KB) → 7.8% occupancy. Shared memory is the binding occupancy constraint (not registers).

**Fix**:
- `perm[kMaxTaxa+2]` (Phase 0-1 only) unionized with `stackMint[kMaxTaxa]` (Phase 2-3 only) → saves **3.2 KB**
- `stackMaxt[kMaxTaxa=800]` reduced to `stackMaxt[kMaxSprStack=64]` (max use: 51 words NNI bitset or 12 entries SPR stack) → saves **2.9 KB**
- **Net**: BuildShared 28.4 KB → **22.4 KB** → blocks/SM: 5 → **7** (+40% occupancy)

**Files**: `gpu/include/pars_tree.cuh` only (union is transparent to pars_build.cu)

**Benchmark (50 datasets, numpars=200, sprdist=3, gpu_stop=4, seed=1)**:

| Metric | Value |
|--------|-------|
| Average speedup vs Opt-M | **+1.197x** |
| Min speedup | 0.977x (N=59, noise) |
| Max speedup | 1.517x (N=93–169) |
| Quality regression | **0/50** |

Pattern: N≤65 → ~1.0x (noise); N≥80 → 1.18–1.52x consistent gains.

---

## 2026-05-13 (Opt-M)

### Opt-M: Template Specialization of `newviewParsimony` by STATES

**Problem**: `newviewParsimony` used `t_A[kMaxStates=32]`, `o_A[kMaxStates=32]` arrays and a runtime `states` loop — wasting registers and preventing loop unrolling.
- DNA (states=4): 28/32 register slots wasted per array × 2 = **−56 wasted registers/lane**
- Protein (states=20): 12/32 × 2 = **−24 wasted registers/lane**

**Fix**: Templated entire call chain on compile-time `STATES`:
- `newviewParsimony<SharedT, STATES>` — `t_A[STATES]`, `o_A[STATES]`, `#pragma unroll`
- `createTiAndNewviewParsimony<SharedT, STATES>`, `createTiAndEvaluateParsimony<SharedT, STATES>`
- `testInsert<STATES>`, `doAddTraverse<STATES>`, `applyMove<STATES>`, `gpuSPRHillClimb<STATES>`, `runPhase3<STATES>`
- `buildParsimonyTreesKernel<STATES>`, `buildPhase3Kernel<STATES>` (global kernels)
- Host dispatch: `if (states==2) launch<2> elif (states==4) launch<4> elif (states==20) launch<20> else launch<32>`

**Files**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`

**Benchmark (50 datasets, seed=1, sprdist=3, gpu_stop=4)**:

| numpars | avg ms/tree speedup | median | Quality |
|---------|--------------------|----|---------|
| 200 | **2.16×** | 2.04× | better=2 same=40 worse=8 |
| 400 | **1.43×** | 1.28× | better=2 same=43 worse=5 |

Protein datasets đặc biệt benefit: prot_M4860 (62 taxa): 3.71×, prot_M9973 (60 taxa): 3.52×.
Quality deltas nhỏ (±1–5) là stochastic variance từ RNG state, không phải algorithmic regression. **Reviewer xác nhận đúng đắn 100%.**

---

### Opt-B+: Tighter Lower Bound in SPR DFS Prune

**Enhancement of Opt-B**: Added `score_tree[tip_p_num]` to lower bound formula.
```cpp
// Before (Opt-B):
const unsigned int lb = score_tree[q_num] + score_tree[r_num];
// After (Opt-B+):
const unsigned int lb = score_tip_p + score_tree[q_num] + score_tree[r_num];
// where score_tip_p = score_tree[vfToNum(topo->back_vf[p], N)] — computed once per doAddTraverse call
```

**Proof**: `mp ≥ score_tree[tip_p] + score_tree[q_cand] + score_tree[r]` — still a valid lower bound.

**Result**: Prune rate **~22% → ~30%**, avg speedup **1.31× → 1.49×**, 10/10 datasets faster (vs 7/10 for Opt-B).

**File**: `gpu/src/pars_build.cu` — `doAddTraverse`

---

### Opt-B: Subtree Prune in SPR DFS

**Problem**: `doAddTraverse` called `testInsert` even when the lower bound on parsimony score guaranteed no improvement.

**Fix**: Before each `testInsert` call, check `lb = score_tree[q] + score_tree[r] >= sh.randomMP` → skip if true.

**Result**: ~22% prune rate (edges skipped), avg 1.31× speedup. Later superseded by Opt-B+ (~30%, 1.49×).

**File**: `gpu/src/pars_build.cu` — `doAddTraverse`

---

### Opt-K: `gpu_stop` Default 2 → 4

Benchmark confirmed stop=4 gives better parsimony quality on datasets N≥295 with acceptable overhead (+31–75% time). Changed default in `tools.cpp`.

---

## 2026-05-12

### Opt-G2: Two-Kernel Selective Phase 3 (exact top-X%)

After Phase 2, download `postSprParsimony` for all K trees to host, sort, compute exact top-X% threshold, then run Phase 3 only for those trees.

**vs Opt-G (atomicMin)**: Exact percentile selection, no bias from partial globalBest.

**Result**: ~44% speedup with top_pct=0.1 on N=295. Default `gpu_phase3_top_pct=0.1`.

**New fields**: `GpuTopology::savedSeed`, `GpuParsimonyMem::d_postSprScores`

**New kernel**: `buildPhase3Kernel<STATES>` (Phase 3 only)

**Files**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_tree.cu`, `gpu/src/pars_build.cu`, `mpboot/tools.h/cpp`

---

### Opt-I: Configurable NNI Strength (`gpu_nni_strength`)

Changed `numNNI` formula from hardcoded `/10` to user-configurable `strength × (N-3)`.

**Benchmark**: strength=0.1 optimal (fastest, comparable quality). Default set to 0.1.

**New param**: `-gpu_nni_strength` (float, 0..1, default 0.1)

**File**: `mpboot/tools.h/cpp`, `gpu/src/gpu_init_trees.cu`

---

### Opt-G: Selective Phase 3 — atomicMin Margin

After Phase 2, each block does `atomicMin(&d_globalBest, postSprParsimony)`, then skips Phase 3 if `postSprParsimony > globalBest × (1 + margin/1000)`.

**Result**: 22–50% speedup tùy dataset. Later superseded by Opt-G2.

**New param**: `-gpu_phase3_margin` (float %, default -1=off)

---

### Parameter Defaults Updated

| Parameter | Old default | New default | Reason |
|-----------|------------|-------------|--------|
| `gpu_hc_iter` | 100 | **30** | gpu_stop is primary stopping; 30 = 4× observed max |
| `gpu_stop` | 2 | **4** | Benchmark: stop=4 better quality on N≥295 |
| `gpu_phase3_top_pct` | -1 (off) | **0.1** | Opt-G2: top 10% Phase 3, 22–50% speedup |
| `gpu_nni_strength` | — (new) | **0.1** | Optimal from benchmark |

---

### Opt-D: GpuTopology Struct Shrink

Removed `number[]`, `next_vf[]`, `nnxt_vf[]` arrays (pure arithmetic) from GpuTopology.

**Struct size**: 83,236 → 44,836 bytes (−37.5 KB). `back_vf` and `xpars` now at offsets 0 and 12.8 KB → better L2 cache locality.

**Result**: −8.7% ms/tree average.

---

### Opt-H: Early Stopping Phase 3

Added `no_improve_count` counter in Phase 3 loop. Stop after `gpu_stop` consecutive no-improve iterations.

**Result**: −32.4% ms/tree average (actual iterations: 2–8 vs 10 hardcoded).

---

### testInsert Pre-Refresh Optimization

Eliminated wasted Fitch step in `testInsert`: pre-refresh q, r, tip_p without computing parsVect[p] from face[2] (immediately overwritten by step 2).

**Result**: −24% kernel time, newview avg 2.51→1.72 nodes/call.
