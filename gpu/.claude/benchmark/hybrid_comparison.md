# Hybrid Configs — Bảng Tổng Hợp (2026-05-16)

**Binary**: build/mpboot-avx (`-use_gpu`)  
**Datasets**: 115 (data_treebase, N=50–767), seed=1  
**CPU baseline**: cpu_d6 (sprdist=6, tổng: **108m40s**)  
**Metric**: `/usr/bin/time -v` elapsed wall clock

---

## Giải thích

- **K** = numpars − 1 (số GPU blocks = số trees song song)
- **Bet/Sam/Wor** = Hybrid better / same / worse vs CPU baseline (BEST SCORE FOUND)
- **AvgΔ** = avg(CPU_score − Hybrid_score) — âm = CPU tốt hơn; càng gần 0 càng tốt
- **ms/tree** = avg thời gian GPU per tree trên 115 datasets
- **Avg spd** = mean(CPU_elapsed / Hybrid_elapsed) per dataset
- **Total GPU** = tổng elapsed tất cả 115 datasets
- **Tot spd** = CPU_total / Hybrid_total (108m40s / Total GPU)

---

## Bảng tổng hợp tất cả hybrid configs vs cpu_d6

| Config | K | sprdist | gpu_stop | Bet | Sam | Wor | AvgΔ | ms/tree | Avg spd | Total GPU | Tot spd |
|--------|---|---------|----------|-----|-----|-----|------|---------|---------|-----------|---------|
| n200d3s6  | 199 | 3 | 6  |  1 | 66 | 48 | −4.37 | 38.5ms | 3.62× | 20m20s | **5.34×** |
| n200d4s6  | 199 | 4 | 6  |  2 | 78 | 35 | −2.51 | 48.7ms | 2.92× | 24m20s | 4.47× |
| n200d5s4  | 199 | 5 | 4  |  4 | 75 | 36 | −1.84 | 46.4ms | 2.98× | 23m26s | 4.64× |
| n200d5s6  | 199 | 5 | 6  |  3 | 81 | 31 | −1.53 | 64.2ms | 2.40× | 30m13s | 3.60× |
| n200d6s4  | 199 | 6 | 4  |  4 | 78 | 33 | −1.57 | 60.9ms | 2.43× | 29m01s | 3.75× |
| n200d7s4  | 199 | 7 | 4  |  3 | 79 | 33 | −1.50 | 77.2ms | 2.02× | 35m13s | 3.08× |
| n400d3s6  | 399 | 3 | 6  |  3 | 74 | 38 | −3.58 | 22.9ms | 2.74× | 26m22s | 4.12× |
| n400d4s6  | 399 | 4 | 6  |  4 | 86 | 25 | −1.63 | 28.3ms | 2.29× | 30m36s | 3.55× |
| n400d5s4  | 399 | 5 | 4  |  4 | 79 | 32 | −1.60 | 28.0ms | 2.30× | 30m20s | 3.58× |
| n400d5s6  | 399 | 5 | 6  |  6 | 84 | 25 | −1.25 | 37.1ms | 1.89× | 37m21s | 2.91× |
| n400d6s4  | 399 | 6 | 4  |  5 | 80 | 30 | −1.07 | 36.2ms | 1.88× | 36m57s | 2.94× |
| n400d3s10 | 399 | 3 | 10 |  4 | 79 | 32 | −2.77 | 31.8ms | 2.18× | 33m12s | 3.27× |
| n400d4s10 | 399 | 4 | 10 |  4 | 84 | 27 | −1.48 | 41.8ms | 1.77× | 40m51s | 2.66× |
| n400d5s10 | 399 | 5 | 10 |  7 | 86 | 22 | −1.10 | 57.4ms | 1.40× | 52m45s | 2.06× |
| n400d6s10 | 399 | 6 | 10 |  9 | 87 | 19 | **−0.66** | 76.0ms | 1.11× | 66m58s | 1.62× |
| n1000d3s4 | 999 | 3 | 4  |  3 | 74 | 38 | −3.50 |  9.5ms | 1.90× | 36m28s | 2.98× |
| n1000d3s6 | 999 | 3 | 6  |  4 | 80 | 31 | −2.68 | 12.6ms | 1.67× | 42m33s | 2.55× |
| n1000d4s4 | 999 | 4 | 4  |  2 | 81 | 32 | −1.90 | 12.5ms | 1.65× | 42m05s | 2.58× |
| n1000d4s6 | 999 | 4 | 6  |  4 | 86 | 25 | −1.35 | 15.8ms | 1.44× | 48m09s | 2.26× |
| n1000d5s6 | 999 | 5 | 6  |  6 | 88 | 21 | **−1.04** | 20.8ms | 1.21× | 58m03s | 1.87× |

---

## Phân tích

### Tác động của từng tham số

**sprdist** (SPR radius GPU): tăng sprdist → AvgΔ cải thiện (gần CPU hơn) nhưng ms/tree tăng, Tot spd giảm.
- d3: nhanh nhất, score tệ nhất (AvgΔ ≈ −2.8 to −4.4)
- d6: chậm nhất nhóm, score tốt nhất (AvgΔ ≈ −0.66 to −1.57)

**gpu_stop** (early stopping iters): tăng gpu_stop → AvgΔ cải thiện ~0.3–0.5 units, thời gian tăng ~15–40%.
- s4 vs s6: +0.2–0.4 AvgΔ, +15–25% thời gian
- s6 vs s10: +0.4–0.6 AvgΔ, +30–80% thời gian

**K (numpars)**: tăng K → ms/tree giảm (better GPU occupancy), nhưng total time tăng vì K2 phải xử lý nhiều trees hơn.
- K=199 → K=399: ms/tree giảm ~40%, total time tăng ~30–60%
- K=399 → K=999: ms/tree giảm ~60–65%, total time tăng ~10–40%; AvgΔ cải thiện nhẹ ~0.1–0.2 units
- Ví dụ cùng d4s6: K=199 (48.7ms, 4.47×) → K=399 (28.3ms, 3.55×) → K=999 (15.8ms, 2.26×)

### Top configs theo mục tiêu

| Mục tiêu | Config tốt nhất | Giá trị |
|-----------|----------------|---------|
| Nhanh nhất (Tot spd) | **n200d3s6** | 5.34× |
| Chất lượng tốt nhất (AvgΔ) | **n400d6s10** | −0.66 |
| Cân bằng tốt nhất | **n400d5s4** hoặc **n400d4s6** | AvgΔ≈−1.6, Tot spd≈3.55× |
| Ít tệ hơn CPU nhất (Wor) | **n1000d5s6** | 21/115 |
| ms/tree thấp nhất | **n1000d3s4** | 9.5ms |

### Frontier speed–quality

Các configs nằm trên frontier Pareto (không config nào tốt hơn cả 2 chiều):

| Config | Tot spd | AvgΔ |
|--------|---------|------|
| n200d3s6  | **5.34×** | −4.37 |
| n400d3s6  | 4.12× | −3.58 |
| n200d5s4  | 4.64× | −1.84 |
| n400d4s6  | 3.55× | −1.63 |
| n400d5s4  | 3.58× | −1.60 |
| n400d5s6  | 2.91× | −1.25 |
| n400d6s4  | 2.94× | −1.07 |
| n400d5s10 | 2.06× | −1.10 |
| n400d6s10 | 1.62× | **−0.66** |

---

## Output directories

| Config | Directory | Excel |
|--------|-----------|-------|
| n200d3s6  | `output/hybrid_n200d3s6/`  | `results_d6base_n200d3s6.xlsx` |
| n200d4s6  | `output/hybrid_n200d4s6/`  | `results_d6base_n200d4s6.xlsx` |
| n200d5s4  | `output/hybrid_n200d5s4/`  | `results_d6base_n200d5s4.xlsx` |
| n200d5s6  | `output/hybrid_n200d5s6/`  | `results_d6base_n200d5s6.xlsx` |
| n200d6s4  | `output/hybrid_n200d6s4/`  | `results_d6base_n200d6s4.xlsx` |
| n200d7s4  | `output/hybrid_n200d7s4/`  | `results_d6base_n200d7s4.xlsx` |
| n400d3s6  | `output/hybrid_n400d3s6/`  | `results_d6base_n400d3s6.xlsx` |
| n400d4s6  | `output/hybrid_n400d4s6/`  | `results_d6base_n400d4s6.xlsx` |
| n400d5s4  | `output/hybrid_n400d5s4/`  | `results_d6base_n400d5s4.xlsx` |
| n400d5s6  | `output/hybrid_n400d5s6/`  | `results_d6base_n400d5s6.xlsx` |
| n400d6s4  | `output/hybrid_n400d6s4/`  | `results_d6base_n400d6s4.xlsx` |
| n400d3s10 | `output/hybrid_n400d3s10/` | `results_d6base_n400d3s10.xlsx` |
| n400d4s10 | `output/hybrid_n400d4s10/` | `results_d6base_n400d4s10.xlsx` |
| n400d5s10 | `output/hybrid_n400d5s10/` | `results_d6base_n400d5s10.xlsx` |
| n400d6s10 | `output/hybrid_n400d6s10/` | `results_d6base_n400d6s10.xlsx` |
| n1000d3s4 | `output/hybrid_n1000d3s4/` | `results_d6base_n1000d3s4.xlsx` |
| n1000d3s6 | `output/hybrid_n1000d3s6/` | `results_d6base_n1000d3s6.xlsx` |
| n1000d4s4 | `output/hybrid_n1000d4s4/` | `results_d6base_n1000d4s4.xlsx` |
| n1000d4s6 | `output/hybrid_n1000d4s6/` | `results_d6base_n1000d4s6.xlsx` |
| n1000d5s6 | `output/hybrid_n1000d5s6/` | `results_d6base_n1000d5s6.xlsx` |
| CPU baseline d6 | `output/cpu_d6/` | — |
