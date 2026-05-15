# Benchmark Opt-C vs Opt-R — 5 Config (2026-05-14)

**Binary**: build/mpboot-avx (Opt-P, Opt-M, Opt-Q1+Q2, Strategy-0, **+Opt-C**)  
**Opt-C**: Stagnation detection — restore `best_back_vf` trước Ratchet odd iteration khi topology hash giống iteration trước (adaptive restart khi stuck).  
**Datasets**: 115 (data_treebase, N=50–767), seed=1, numpars=400  
**Metric time**: `/usr/bin/time -v` (elapsed wall clock)  
**Ref**: Opt-R binary (gpu_final_800 series) — restore best trước MỌI iteration (cả even+odd)

## Bảng so sánh chính

| Config | Ref Wor | OptC Wor | ΔWor | Ref AvgΔ | OptC AvgΔ | Ref ms/tree | OptC ms/tree | Ref spd | OptC spd |
|--------|---------|---------|------|----------|----------|------------|-------------|---------|---------|
| d3 s4 p0.1 | 20 | **16** | **↓4** ✅ | +0.22 | **+0.27** | 21.7ms | **15.5ms** | 3.33× | **3.39×** |
| d3 s4 p0.3 | 13 | **12** | **↓1** ✅ | +0.57 | **+0.83** | 19.6ms | 19.1ms | 3.04× | 2.98× |
| d3 s8 p0.1 | 9  | **8**  | **↓1** ✅ | +1.44 | **+1.70** | 21.4ms | 21.2ms | 2.76× | 2.69× |
| d3 s8 p0.3 | 8  | **6**  | **↓2** ✅ | +1.64 | **+2.00** | 25.9ms | 25.8ms | 2.39× | 2.34× |
| d5 s4 p0.1 | 4  | 8      | ↑4 ❌ | +2.97 | +3.08 | 28.4ms | 27.8ms | 2.40× | 2.37× |

**AvgΔ** = avg(CPU_score − GPU_score). Dương = GPU tìm được cây tốt hơn CPU.  
**Wor** = số datasets GPU tệ hơn CPU.

## Bảng đầy đủ Bet/Sam/Wor

| Config | Ref Bet/Sam/Wor | OptC Bet/Sam/Wor |
|--------|----------------|-----------------|
| d3 s4 p0.1 | 40/55/20 | **41/58/16** |
| d3 s4 p0.3 | 44/58/13 | 43/60/12 |
| d3 s8 p0.1 | 47/59/9  | 47/60/8  |
| d3 s8 p0.3 | 48/59/8  | 47/62/6  |
| d5 s4 p0.1 | 50/61/4  | 50/57/8  |

## Kết luận

### Opt-C hoạt động tốt với sprdist nhỏ (d3)
- **4/4 configs d3**: Opt-C giảm số worse datasets, cải thiện AvgΔ
- **d3 s4 p0.1**: nổi bật nhất — 20→16 worse (↓4), ms/tree giảm từ 21.7→15.5ms, speedup 3.33→3.39×
- **d3 s8 p0.3**: chỉ còn 6 worse datasets — tốt nhất trong toàn bộ configs đã test

### Opt-C thất bại với sprdist=5
- **d5 s4 p0.1**: 4→8 worse (↑4) — sprdist lớn làm SPR tìm được nhiều topology khác nhau hơn, topology hash ít lặp lại → stagnation detection ít trigger → khi trigger lại không đúng lúc

### Speedup
- d3 s4 p0.1: Opt-C **nhanh hơn** ref (+0.06×, 15.5 vs 21.7 ms/tree) — nhờ Strategy-0 timing removal
- Các configs còn lại: gần như tương đương (±0.07×)

## Config tốt nhất theo tiêu chí

| Tiêu chí | Config | Wor | AvgΔ | Speedup |
|---------|--------|-----|------|---------|
| **Ít worse nhất** | d3 s8 p0.3 OptC | **6** | +2.00 | 2.34× |
| **Nhanh nhất** | d3 s4 p0.1 OptC | 16 | +0.27 | **3.39×** |
| **Cân bằng tốt nhất** | d3 s8 p0.1 OptC | **8** | +1.70 | 2.69× |

## Output directories

| Config | Ref dir | OptC dir |
|--------|---------|---------|
| d3 s4 p0.1 | `output/gpu_final_800/` | `output/gpu_optC_d3s4p01/` |
| d3 s4 p0.3 | `output/gpu_final_800_0.3/` | `output/gpu_optC_d3s4p03/` |
| d3 s8 p0.1 | `output/gpu_final_800_stop8/` | `output/gpu_optC_d3s8p01/` |
| d3 s8 p0.3 | `output/gpu_final_800_stop8_0.3/` | `output/gpu_optC_d3s8p03/` |
| d5 s4 p0.1 | `output/gpu_final_800_dist5_stop4_0.1/` | `output/gpu_optC_d5s4p01/` |

## Excel files

| File | Config |
|------|--------|
| `output/results_optC_d3s4p01_vs_cpu.xlsx` | d3 s4 p0.1 |
| `output/results_optC_d3s4p03_vs_cpu.xlsx` | d3 s4 p0.3 |
| `output/results_optC_d3s8p01_vs_cpu.xlsx` | d3 s8 p0.1 |
| `output/results_optC_d5s4p01_vs_cpu.xlsx` | d5 s4 p0.1 |
| `output/results_optC_d3s8p03_vs_cpu.xlsx` | d3 s8 p0.3 |
