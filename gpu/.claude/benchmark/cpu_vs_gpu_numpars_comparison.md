# CPU vs GPU: numpars 200 / 300 / 400 — Full Comparison

**GPU settings**: `-seed 1 -sprdist 3 -gpu_stop 4` + defaults (`gpu_hc_iter=30`, `gpu_phase3_top_pct=0.1`, `gpu_nni_strength=0.1`)  
**CPU settings**: standard mpboot CPU (serial)  
**Datasets**: 115 datasets (DNA + protein, 50–767 taxa)  
**Date**: 2026-05-13  
**Logs**: `output/gpu_opt/` (np=200), `output/gpu_opt_300/` (np=300), `output/gpu_opt_400/` (np=400)  
**Excel**: `output/results_cpu_vs_gpu_opt.xlsx`, `results_cpu_vs_gpu_opt_400.xlsx`

---

## Summary

### ms/tree (GPU efficiency per tree)

| numpars | avg ms/tree | vs np=200 |
|---------|------------|-----------|
| 200     | 150.6 ms   | baseline  |
| 300     | 93.1 ms    | −38%      |
| **400** | **73.4 ms**| **−51%**  |

### Speedup vs CPU (CPU_elapsed / GPU_elapsed)

| Metric | np=200 | np=300 | np=400 |
|--------|--------|--------|--------|
| avg speedup | **1.29×** | 1.27× | 1.25× |
| median speedup | 1.10× | **1.13×** | 1.06× |
| GPU faster (>1×) | 63/115 | **64/115** | 62/115 |
| GPU slower (<1×) | 52/115 | 51/115 | 53/115 |
| Best speedup | 3.51× | 3.18× | 3.27× |
| Worst speedup | 0.42× | 0.34× | 0.35× |

### Quality (GPU hc_best vs CPU best_pars, Δ>0 = GPU better)

| | np=200 | np=300 | np=400 |
|--|--------|--------|--------|
| GPU better | 47/115 | 48/115 | **49/115** |
| Same | 60/115 | 61/115 | 60/115 |
| GPU worse | 8/115 | 6/115 | **6/115** |

### Total kernel time (avg across 115 datasets)

| numpars | avg kernel total |
|---------|-----------------|
| 200 | 30.0 s |
| **300** | **27.8 s** |
| 400 | 29.3 s |

---

## numpars 200 vs 300 vs 400 (GPU-only quality)

| | np=300 vs np=200 | np=400 vs np=200 | np=400 vs np=300 |
|--|-----------------|-----------------|-----------------|
| better | 12/115 | 21/115 | 11/115 |
| same   | 103/115 | 92/115 | 102/115 |
| worse  | 0/115  | 2/115  | 2/115  |

---

## Kết luận & Khuyến nghị

| Mục tiêu | Recommended numpars |
|----------|---------------------|
| Best ms/tree (per-tree efficiency) | **400** |
| Best wall-clock vs CPU | **300** (avg kernel 27.8s, median speedup 1.13×) |
| Best parsimony quality | **400** (49/115 better than CPU, 21/115 better than np=200) |
| Balanced | **300** hoặc **400** |

### Khi nào GPU tốt hơn CPU
- **Taxa ≥ 300** hoặc **sites ≥ 5000**: GPU nhanh hơn rõ ràng
- **Taxa < 200** + **sites < 2000**: CPU thường nhanh hơn (GPU overhead > compute benefit)
- Protein datasets nhỏ (< 100 taxa, < 1000 sites): GPU chậm hơn (overhead dominates)

### np=300 là sweet spot về speedup
- Best speedup thắng nhiều nhất (41/115 datasets)
- Avg total kernel thấp nhất (27.8s)
- Median speedup cao nhất (1.13×)

### np=400 là sweet spot về quality
- ms/tree thấp nhất (73.4ms, −51% vs np=200)
- GPU better parsimony nhiều nhất (49/115)
- GPU worse parsimony ít nhất (6/115)

---

## Notes

- Opt-B (subtree prune): **22% prune rate** on doAddTraverse edges — implemented và xác nhận hiệu quả
- top_pct=0.1: chỉ ~10–22% trees (including ties) làm Phase 3 → giảm overhead numpars lớn
- gpu_stop=4: quality tốt hơn rõ rệt vs default 2, overhead hợp lý
