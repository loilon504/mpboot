# CPU vs GPU_OPT: 115 Datasets Full Benchmark

**GPU settings**: `-seed 1 -numpars 200 -sprdist 3 -gpu_stop 4` + defaults (`gpu_hc_iter=30`, `gpu_phase3_top_pct=0.1`)  
**CPU settings**: standard mpboot CPU  
**Date**: 2026-05-13  
**Script**: `build/bench_gpu_opt.sh` → logs in `output/gpu_opt/`  
**Excel**: `output/results_cpu_vs_gpu_opt.xlsx`

---

## Tổng kết speedup (CPU_elapsed / GPU_elapsed)

| Metric | Giá trị |
|--------|---------|
| Average speedup | **1.29×** |
| Median speedup | 1.10× |
| Best | **3.51×** — prot_M10372_169_22426 (169 taxa, 22426 sites) |
| Worst | 0.42× — prot_M10236_59_164 (59 taxa, 164 sites) |
| GPU nhanh hơn (>1×) | **63/115** (55%) |
| GPU chậm hơn (<1×) | 52/115 (45%) |
| Speedup >2× | 19/115 |
| Speedup >5× | 0/115 |

## Tổng kết quality (Δ = CPU_best − GPU_best, dương = GPU tốt hơn)

| | Count |
|--|-------|
| GPU tốt hơn (Δ>0) | **47/115** — avg +5.0 units, max +61 |
| Bằng nhau (Δ=0) | 60/115 |
| GPU tệ hơn (Δ<0) | 8/115 — đều rất nhỏ (1–4 units) |

## Top 10 speedup nhanh nhất

| Dataset | N | Sites | CPU_s | GPU_s | Speedup | Δbest |
|---------|---|-------|-------|-------|---------|-------|
| prot_M10372_169_22426 | 169 | 22426 | 255.8 | 72.9 | **3.51×** | 0 |
| prot_M11012_55_11500 | 55 | 11500 | 25.6 | 7.5 | 3.40× | 0 |
| prot_M5379_60_7776 | 60 | 7776 | 16.6 | 5.0 | 3.32× | 0 |
| prot_M7078_77_12457 | 77 | 12457 | 34.5 | 10.6 | 3.26× | 0 |
| prot_M10273_169_11009 | 169 | 11009 | 130.5 | 41.9 | 3.11× | 0 |
| prot_M11013_55_8741 | 55 | 8741 | 19.3 | 6.9 | 2.78× | 0 |
| dna_M12051_699_6914 | 699 | 6914 | 718.5 | 260.4 | 2.76× | +11 |
| prot_M4539_59_12428 | 59 | 12428 | 33.5 | 12.5 | 2.68× | 0 |
| prot_M11740_138_4427 | 138 | 4427 | 54.0 | 20.7 | 2.61× | 0 |
| dna_M14582_372_61199 | 372 | 61199 | 659.3 | 252.3 | 2.61× | 0 |

## Datasets GPU chậm hơn CPU (speedup < 1×)

52 datasets, chủ yếu là:
- Protein datasets nhỏ (< 100 taxa, < 1000 sites) — GPU overhead > compute benefit
- DNA datasets nhỏ (200–300 taxa, < 2000 sites) với ít computation

Các worst cases:
| Dataset | N | Sites | Speedup | Pattern |
|---------|---|-------|---------|---------|
| prot_M10236_59_164 | 59 | 164 | 0.42× | Protein, rất ít sites |
| prot_M3810_55_271 | 55 | 271 | 0.42× | Protein nhỏ |
| prot_M12103_97_199 | 97 | 199 | 0.44× | Protein nhỏ |
| dna_M2534_207_976 | 207 | 976 | 0.54× | DNA, ít sites |

## Phân tích theo kích thước

**GPU có lợi rõ rệt khi**:
- Taxa ≥ 300, hoặc
- Sites ≥ 5000

**GPU không có lợi khi**:
- Protein < 100 taxa với < 1000 sites
- DNA < 250 taxa với < 2000 sites

**Lý do GPU chậm trên dataset nhỏ**:
- Kernel launch overhead + memory transfer chiếm % lớn
- Với `top_pct=0.1`, Kernel A (Phase 0-2) + sort + Kernel B (Phase 3 top 10%) có latency cố định
- Dataset nhỏ: compute quá ít để amortize overhead

## Ghi chú về settings

- GPU dùng `gpu_phase3_top_pct=0.1` (Opt-G2): chỉ top 10% trees (actual 10–25% do ties) làm Phase 3
- `actual_phase3` dao động 10–22% trên datasets thực tế
- GPU quality thường tốt hơn hoặc bằng CPU (47+60 = 107/115 datasets)
