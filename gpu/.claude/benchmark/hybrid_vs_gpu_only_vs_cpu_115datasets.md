# Benchmark: Hybrid vs GPU-Only vs CPU — 115 Datasets × 5 Configs

**Date**: 2026-05-16  
**Binary**: `/build/mpboot-avx` (hybrid4 — [7] loop refactor, no pllClone, candidateTrees.update inside mpbootGpu)  
**Baseline**: `cpu_d6` — 115 datasets, `-seed 1 -sprdist 6 -numpars 100` (default)  
**GPU-Only**: `gpu_only_n200d{3,4,5,5,6}s{6,6,4,6,4}` — same 115 datasets, same 5 configs, no CPU HC  
**Hybrid**: `hybrid_n200d{3,4,5,5,6}s{6,6,4,6,4}` — GPU + CPU concurrent Hill-Climbing (hybrid_cb2)

---

## 1. Config Matrix

| Dir | numpars | sprdist | gpu_stop | GPU device |
|-----|---------|---------|----------|------------|
| n200d3s6 | 200 | 3 | 6 | 1 |
| n200d4s6 | 200 | 4 | 6 | 2 |
| n200d5s4 | 200 | 5 | 4 | 3 |
| n200d5s6 | 200 | 5 | 6 | 4 |
| n200d6s4 | 200 | 6 | 4 | 5 |

All runs: `-gpu_top_pct 0.1 -seed 1`

---

## 2. Score Quality (115 datasets)

**All 3 modes produce identical final parsimony scores** for all 115 × 5 combinations.

| Comparison | H_better | G_better | Equal |
|-----------|---------|---------|-------|
| Hybrid vs cpu_d6 | 0 | 0 | 115/115 |
| GPU-only vs cpu_d6 | 0 | 0 | 115/115 |
| Hybrid vs GPU-only | 0 | 0 | 115/115 |

**Insight**: Final MPBoot parsimony score is identical regardless of initialization strategy. GPU and CPU trees both feed the same `candidateTrees` pool, and MPBoot's own post-processing converges to the same local minimum.

---

## 3. Total Time (sum over 115 datasets)

### With all 115 datasets (includes 3 extreme outliers — see §5):

| Config | Hybrid (s) | GPU-only (s) | CPU (s) | H/C speedup | G/C speedup | H/G ratio |
|--------|-----------|-------------|---------|------------|------------|---------|
| n200d3s6 | 2737 | 1242 | 6520 | **2.38×** | 5.25× | 2.20× |
| n200d4s6 | 2912 | 1486 | 6520 | **2.24×** | 4.39× | 1.96× |
| n200d5s4 | 2894 | 1444 | 6520 | **2.25×** | 4.51× | 2.00× |
| n200d5s6 | 3347 | 1887 | 6520 | **1.95×** | 3.46× | 1.77× |
| n200d6s4 | 3240 | 1821 | 6520 | **2.01×** | 3.58× | 1.78× |

### Without 3 extreme outliers (112 datasets — see §5):

| Config | Hybrid (s) | GPU-only (s) | CPU (s) | H/C speedup | G/C speedup | H/G ratio |
|--------|-----------|-------------|---------|------------|------------|---------|
| n200d3s6 | 1340 | 1221 | 6440 | **4.81×** | 5.27× | **1.10×** |
| n200d4s6 | 1563 | 1459 | 6440 | **4.12×** | 4.41× | **1.07×** |
| n200d5s4 | 1520 | 1419 | 6440 | **4.24×** | 4.54× | **1.07×** |
| n200d5s6 | 1960 | 1857 | 6440 | **3.29×** | 3.47× | **1.06×** |
| n200d6s4 | 1898 | 1791 | 6440 | **3.39×** | 3.60× | **1.06×** |

**Key insight**: When the 3 extreme outliers are excluded, hybrid adds only **6–10% overhead** over GPU-only — the concurrent CPU HC runs almost for free.

---

## 4. CPU Contribution Metrics (hybrid-specific)

Aggregated over 115 datasets, config n200d3s6 (representative):

### 4a. Upload ratio: CPU→GPU memory slots

```
CPU built: ~1463 trees total (across 115 datasets)
Uploaded to GPU slots: 0 (0%)
```

CPU-built trees are **never** inserted back into GPU memory (no GPU topology replacement). They go directly into `candidateTrees` via `candidateTrees.update()`. The "upload" counter counts GPU topology slot replacement only.

### 4b. CPU found-better events

"CPU found better tree at iteration N" = CPU's NNI produced a tree with lower parsimony than current `iqtree.bestScore` at that moment.

| Config | Total events | Datasets with ≥1 event |
|--------|------------|----------------------|
| n200d3s6 | 227 | 64/115 |
| n200d4s6 | 221 | 64/115 |
| n200d5s4 | 192 | 54/115 |
| n200d5s6 | 203 | 54/115 |
| n200d6s4 | 190 | 55/115 |

~1.5–2 "found better" events per participating dataset on average. For N>400, 7–8 events per dataset.

### 4c. CPU wins vs GPU wins (best_cpu_tree vs best_gpu_tree at [7])

Config n200d3s6 — who found the better initialization tree?

| N-group | n_ds | CPU wins | GPU wins | Tie |
|---------|------|---------|---------|-----|
| N≤100 | 36 | 3 | 3 | 30 |
| 100<N≤200 | 9 | 0 | 0 | 9 |
| 200<N≤400 | 62 | 30 | 9 | 23 |
| **N>400** | **8** | **7** | **0** | **1** |
| **Total** | **115** | **40** | **12** | **63** |

Across all 5 configs: CPU wins 24–40/115 datasets; GPU wins 9–14/115.

**When CPU wins**: average gap = 4–7 parsimony points lower than GPU best.  
**When GPU wins**: average gap = 2–7 parsimony points.

**Key insight for large N**: For N>400, CPU concurrent HC almost always beats GPU (7/8). The CPU NNI perturbation + search is highly effective for large complex trees, while the GPU's fixed SPR radius and parallel structure gives less benefit per iteration.

### 4d. CPU Hill-Climbing iterations (n200d3s6)

Per-dataset CPU HC ran 1–168 NNI iterations during the GPU kernel window. Datasets with many iterations (e.g., 168 iter) are those where the GPU kernel takes long (N>300), giving the CPU more time to work.

---

## 5. Critical Issue: 3 Extreme Overhead Outliers

Three datasets cause disproportionate hybrid slowdown:

| Dataset | N | Sites | hybrid | gpu_only | cpu_d6 | cpu_iters |
|---------|---|-------|--------|---------|--------|---------|
| dna_M14678_225_2673 | 225 | 2673 | **626.9s** | 5.4s | 14.1s | 1 |
| dna_M5731_242_9626 | 242 | 9626 | **579.1s** | 8.7s | 36.5s | 1 |
| dna_M6134_219_5158 | 219 | 5158 | **191.4s** | 6.5s | 29.5s | 1 |

**The GPU finished in ~5–9s but hybrid took 191–627s** — overhead of 6300–11500%.

**Root cause**: `doNNISearch()` in `hybrid_cb2` (line 401, `gpu_init_trees.cu`) is called once and takes extremely long for these medium-N, high-site datasets. The CPU Hill-Climbing iteration does **1 NNI step** but spends 191–627 seconds. Suspected cause: `doNNISearch()` invokes IQ-TREE ML-style NNI with branch length optimization (`optimizeAllBranches()`) internally for these specific datasets, not pure parsimony NNI. 

**Scope**: 3/115 datasets (2.6%). These datasets share medium N (219–242) with high site count (2673–9626).

**Impact on total hybrid time**: These 3 outliers contribute 627+579+191 = **1397 seconds** out of hybrid's total 2737s for config n200d3s6 — **51% of total hybrid time from 2.6% of datasets**.

**TODO**: Investigate `doNNISearch()` path in hybrid_cb2. Should replace with `rearrangeParsimony()` or time-bounded HC, or add per-iteration timeout.

---

## 6. Speedup by N-Group (config n200d3s6)

| N-group | n | H_mean | G_mean | C_mean | H/C | G/C | H/G |
|---------|---|--------|--------|--------|-----|-----|-----|
| N≤100 | 36 | 3.2s | 2.8s | 11.7s | 2.83× | 3.16× | 0.97× |
| 100<N≤200 | 9 | 7.9s | 7.0s | 55.6s | 5.01× | 5.47× | 0.96× |
| 200<N≤400 | 62 | 32.4s | 9.3s | 37.1s | 2.89× | 3.24× | 0.90× |
| N>400 | 8 | 67.8s | 62.7s | 412.7s | 5.66× | 6.22× | 0.93× |

Note: 200<N≤400 mean is heavily skewed by the 3 outlier datasets.

---

## 7. Summary / Key Takeaways

1. **Score: all methods equal** — GPU, hybrid, and CPU all converge to the same final parsimony. The initialization strategy doesn't affect quality.

2. **Hybrid overhead is minimal when healthy** — excluding 3 buggy outliers, hybrid adds only 6–10% overhead vs GPU-only while doing concurrent CPU NNI hill-climbing.

3. **CPU concurrent HC helps for large N** — For N>400, CPU wins the best tree in 7/8 cases. The CPU's full SPR/NNI search is more effective than the GPU's fixed-radius parallel search at large scale.

4. **3 critical outliers break hybrid** — datasets N=219–242 with high site counts trigger a pathological `doNNISearch()` path that takes 190–627s (vs GPU 5–9s). These alone account for ~51% of hybrid's total time overhead. **This bug must be fixed before hybrid can be claimed to be competitive.**

5. **GPU-only is the safe baseline** — GPU-only achieves 3.46–5.27× speedup over CPU with no risk of outlier regression. Hybrid achieves similar speedup when the outlier bug is absent.
