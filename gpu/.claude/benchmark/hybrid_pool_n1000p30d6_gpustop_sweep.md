# Benchmark: Hybrid Pool — gpu_stop Sweep (stop=4/6/8/12/14)

**Date**: 2026-05-17  
**Binary**: `/build/mpboot-avx` (sau dead code removal: K1=96 regs, K2=151 regs)  
**Baseline**: `cpu_d6` — 115 datasets, `-seed 1` (CPU default, no GPU)  
**Config cố định**: `numpars=1000, sprdist=6, gpu_pool_size=30, seed=1`  
**Datasets**: 115 datasets treebase, N=50–767 taxa  
**Devices**: s4→device 1, s6→device 2, s8→device 3, s12→device 4, s14→device 5  
**Output dirs**: `output/hybrid_pool_n1000p30d6s{4,6,8,12,14}/`

> **Lưu ý**: s6 ban đầu bị mixed data (48 log từ run cũ dùng `hardtrees/` + `-gpu_top_pct 0.1`).
> Đã xóa và chạy lại sạch. Kết quả dưới đây là từ run sạch (115 logs từ `data_treebase/`).

---

## Kết quả tổng hợp

| gpu_stop | Mean spd | Median spd | Total spd | Total GPU time (s) | GPU faster (>1×) | GPU better score | CPU better score | Tie |
|----------|----------|------------|-----------|-------------------|-----------------|-----------------|-----------------|-----|
| 4 | 2.70× | 2.42× | 4.36× | 1497 | 107/115 | 7/115 | 24/115 | 84/115 |
| **6** | **2.77×** | 2.38× | **4.55×** | **1434** | **111/115** | 8/115 | **21/115** | 86/115 |
| 8 | 2.61× | 2.17× | 4.25× | 1534 | 107/115 | 8/115 | 22/115 | 85/115 |
| 12 | 2.57× | 2.16× | 4.10× | 1590 | 107/115 | 8/115 | 24/115 | 83/115 |
| 14 | 2.52× | 2.08× | 4.06× | 1605 | 105/115 | 9/115 | 22/115 | 83/115 |

**Total CPU time (baseline)**: 6520.5s (tất cả configs chia sẻ cùng baseline)

---

## Speedup theo taxa group (Mean speedup)

| Taxa | stop=4 | stop=6 | stop=8 | stop=12 | stop=14 |
|------|--------|--------|--------|---------|---------|
| 0–99 (N=34) | 1.58× | 1.59× | 1.51× | 1.48× | 1.41× |
| 100–199 (N=11) | 2.52× | 2.54× | 2.41× | 2.39× | 2.33× |
| 200–299 (N=50) | 2.66× | **2.78×** | 2.56× | 2.49× | 2.49× |
| 300–399 (N=12) | 3.64× | **3.74×** | 3.57× | 3.60× | 3.56× |
| 400–499 (N=3) | 4.44× | **4.42×** | 4.21× | 3.89× | 3.75× |
| 500–599 (N=2) | 5.23× | **5.45×** | 5.33× | 5.42× | 5.18× |
| 600–699 (N=2) | **10.96×** | 10.75× | 10.55× | 10.95× | 10.89× |
| 700–799 (N=1) | 6.93× | 6.91× | 6.83× | 6.49× | 6.32× |

---

## Score quality (avg margin khi thắng)

| gpu_stop | GPU avg margin (khi GPU thắng) | CPU avg margin (khi CPU thắng) |
|----------|-------------------------------|-------------------------------|
| 4 | 1.9 | 5.8 |
| **6** | 1.6 | **6.8** |
| 8 | 1.5 | 6.3 |
| 12 | 2.4 | 5.9 |
| 14 | 2.3 | 6.5 |

---

## Kết luận

**gpu_stop=6 là sweet spot** trên tất cả metrics:
- **Nhanh nhất**: total speedup 4.55×, total GPU time 1434s (thấp nhất)
- **Score tốt nhất**: ít dataset CPU thắng nhất (21/115), GPU faster nhiều nhất (111/115)
- **Pattern rõ**: tăng gpu_stop > 6 chỉ làm chậm hơn, không cải thiện score

**Monotone pattern**: gpu_stop tăng → total GPU time tăng đều (1434 → 1605s), speedup giảm đều.  
Exception: stop=4 nhanh (1497s) nhưng score tệ hơn stop=6 (24/115 CPU thắng vs 21/115).

**Recommended config**: `numpars=1000, sprdist=6, gpu_stop=6, gpu_pool_size=30`  
→ 4.55× total speedup, 111/115 GPU faster, 86/115 tie score với CPU.
