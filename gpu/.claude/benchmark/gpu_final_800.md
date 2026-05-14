# GPU Final Benchmark — 7 Config Parameter Sweep vs CPU (2026-05-14)

**Binary**: build/mpboot-avx (tất cả optimizations: Opt-P, Opt-M, Opt-Q1+Q2, Opt-R)  
**Datasets**: 115 (data_treebase, N=50–767), seed=1, numpars=400  
**Metric time**: `/usr/bin/time -v` (elapsed wall clock)  
**Ký hiệu**: d=sprdist, s=gpu_stop, p=gpu_top_pct

## Bảng 1: Tổng hợp 7 config vs CPU

| Metric | d3 s4 p0.1 | d3 s4 p0.3 | d3 s8 p0.1 | d3 s8 p0.3 | d4 s4 p0.1 | d5 s4 p0.1 | d6 s4 p0.1 |
|--------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Avg speedup | **3.33×** | 3.04× | 2.76× | 2.39× | 2.87× | 2.40× | 1.93× |
| Max speedup | 11.61× | 9.94× | 9.07× | 8.02× | 9.05× | 7.42× | 5.91× |
| GPU better | 40 (35%) | 44 (38%) | 47 (41%) | 48 (42%) | 49 (43%) | **50 (43%)** | **50 (43%)** |
| GPU same | 55 (48%) | 58 (50%) | 59 (51%) | 59 (51%) | 57 (50%) | 61 (53%) | 61 (53%) |
| GPU worse | 20 (17%) | 13 (11%) | 9 (8%) | 8 (7%) | 9 (8%) | **4 (3%)** | **4 (3%)** |
| Avg Δparsimony | −0.22 | −0.57 | −1.44 | −1.64 | −2.63 | −2.97 | **−3.90** |
| Avg ms/tree | 21.7ms | 19.6ms | 21.4ms | 25.9ms | 21.7ms | 28.4ms | 39.0ms |

## Bảng 2: Speedup theo nhóm taxa

| N group | d3 s4 p0.1 | d3 s4 p0.3 | d3 s8 p0.1 | d3 s8 p0.3 | d4 s4 p0.1 | d5 s4 p0.1 | d6 s4 p0.1 |
|---------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| N≤99 | 3.25× | 2.79× | 2.41× | 2.05× | 2.66× | 2.27× | 1.78× |
| N=100–199 | **4.44×** | 3.80× | 3.43× | 2.71× | 3.64× | 3.08× | 2.54× |
| N=200–299 | 2.94× | 2.62× | 2.37× | 2.07× | 2.43× | 2.03× | 1.64× |
| N=300–399 | 3.34× | 3.15× | 2.88× | 2.61× | 3.02× | 2.54× | 2.10× |
| N≥400 | 4.55× | 5.54× | 5.59× | 5.04× | 5.19× | 4.13× | 3.25× |

**Nhận xét**: N≥400 — stop=8 và pct lớn hiệu quả hơn vì đủ thời gian hội tụ.

## Bảng 3: Dataset GPU tệ hơn CPU trong nhiều config

| Dataset | N | #/7 | d3s4p0.1 | d3s4p0.3 | d3s8p0.1 | d3s8p0.3 | d4s4p0.1 | d5s4p0.1 | d6s4p0.1 |
|---------|---|-----|---------|---------|---------|---------|---------|---------|---------|
| dna_M11113_344_9778 | 344 | **7/7** | +4 | +3 | +4 | +3 | +5 | +3 | +3 |
| dna_M14678_225_2673 | 225 | 6/7 | +1 | +1 | +1 | +1 | +2 | ✅ | +1 |
| dna_M14582_372_61199 | 372 | 6/7 | +2 | +2 | +1 | +1 | +1 | ✅ | +1 |
| dna_M7964_640_25260 | 640 | 5/7 | +20 | +12 | +12 | +9 | ✅ | +4 | ✅ |
| dna_M5078_265_9768 | 265 | 5/7 | +3 | +3 | +1 | +1 | ✅ | +1 | ✅ |
| dna_M10933_229_2696 | 229 | 5/7 | +11 | +4 | +2 | +2 | +1 | ✅ | ✅ |
| prot_M8461_89_5699 | 89 | 4/7 | +5 | +5 | +4 | +4 | ✅ | ✅ | ✅ |
| dna_M1838_228_1131 | 228 | 4/7 | +1 | +1 | +1 | +1 | ✅ | ✅ | ✅ |

## Bảng 4: Best config theo tiêu chí

| Tiêu chí | Config tốt nhất | Giá trị |
|---------|----------------|---------|
| Nhanh nhất | d3 s4 p0.1 | avg **3.33×** speedup |
| Chất lượng tốt nhất | d6 s4 p0.1 | avg Δ = **−3.90** |
| Ít worse nhất | d5 s4 p0.1 | chỉ **4/115** datasets worse |

## Kết luận quan trọng

### 1. sprdist tác động lớn hơn stop
- d5 (4 worse) tốt hơn d3s8 (9 worse) — tăng SPR radius hiệu quả hơn tăng số vòng lặp
- Nhưng d5 (2.40×) chậm hơn d3s4 (3.33×) vì mỗi SPR search tốn hơn

### 2. Dataset "cứng" nhất
`dna_M11113_344_9778` (N=344) — GPU thua **7/7 configs**, không config nào giúp được.  
→ Cần research sâu hơn về topology structure của dataset này.

`dna_M7964_640_25260` (N=640) — thua 5/7, nhưng **d4 và d6 giải được** (✅) → vấn đề SPR radius.

### 3. Khuyến nghị theo use case

| Use case | Config | Lý do |
|---------|--------|-------|
| **Thesis benchmark** | d5 s4 p0.1 | Chỉ 4 worse, Δ=−2.97, cân bằng tốt |
| **Nhanh nhất** | d3 s4 p0.1 | 3.33× speedup |
| **Chất lượng cao** | d6 s4 p0.1 | Δ=−3.90, 4 worse, nhưng 1.93× (chậm) |
| **Protein nặng (N=100-200)** | d3 s4 p0.1 | Max speedup 11.61× nhờ width lớn |
| **DNA lớn (N≥400)** | d5/d6 s4 p0.1 | SPR radius đủ để thoát local optima |

### 4. Trade-off tổng quát
```
sprdist↑ → chất lượng↑, tốc độ↓
stop↑    → chất lượng↑ nhỉnh, tốc độ↓
pct↑     → chất lượng↑ nhỉnh, tốc độ↓ nhỉnh
```
Tăng sprdist từ 3→5: chất lượng tốt hơn đáng kể (20→4 worse), tốc độ giảm từ 3.33→2.40× (~28%).

## Files Excel kết quả

| File | Config |
|------|--------|
| `results_gpu_final_800_vs_cpu.xlsx` | d3, s4, p0.1 |
| `results_gpu_final_800_0.3_vs_cpu.xlsx` | d3, s4, p0.3 |
| `results_gpu_final_800_stop8_vs_cpu.xlsx` | d3, s8, p0.1 |
| `results_gpu_final_800_stop8_0.3_vs_cpu.xlsx` | d3, s8, p0.3 |
| `results_gpu_final_800_dist4_stop4_0.1_vs_cpu.xlsx` | d4, s4, p0.1 |
| `results_gpu_final_800_dist5_stop4_0.1_vs_cpu.xlsx` | d5, s4, p0.1 |
| `results_gpu_final_800_dist6_stop4_0.1_vs_cpu.xlsx` | d6, s4, p0.1 |
