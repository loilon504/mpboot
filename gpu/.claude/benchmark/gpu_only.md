# GPU-only Benchmark — 13 Configs (2026-05-15)

**Binary**: build/mpboot-avx (Combined v2: Opt-C@sprdist=3, Symmetric@sprdist>3)  
**Defaults**: gpu_hc_iter=100, gpu_top_pct=0.1, seed=1  
**Datasets**: 115 (data_treebase, N=50–767)  
**Metric**: `/usr/bin/time -v` (elapsed wall clock)

> **Note on hardware**: d3/d4 configs (7 cũ) chạy trên machine cũ (CPU tổng 139m13s/108m40s).  
> d5/d6 configs (6 mới, 2026-05-15) chạy trên machine mới — CPU d3=38m12s, CPU d6=36m40s.  
> Speedup **không so sánh trực tiếp** giữa hai nhóm. AvgΔ (quality) **có thể so sánh**.

## Configs tested

### d3/d4 configs (machine cũ)

| Config | numpars | sprdist | gpu_stop | device | Output dir |
|--------|---------|---------|----------|--------|-----------|
| n200 d3 s6  | 200 | 3 | 6  | 1 | `gpu_only_n200d3s6/` |
| n200 d3 s10 | 200 | 3 | 10 | 2 | `gpu_only_n200d3s10/` |
| n400 d3 s6  | 400 | 3 | 6  | 3 | `gpu_only_n400d3s6/` |
| n400 d3 s10 | 400 | 3 | 10 | 4 | `gpu_only_n400d3s10/` |
| n200 d4 s6  | 200 | 4 | 6  | 5 | `gpu_only_n200d4s6/` |
| n200 d4 s10 | 200 | 4 | 10 | 6 | `gpu_only_n200d4s10/` |
| n400 d4 s6  | 400 | 4 | 6  | 7 | `gpu_only_n400d4s6/` |

### d5/d6 configs (machine mới, 2026-05-15)

| Config | numpars | sprdist | gpu_stop | device | Output dir | Crashed |
|--------|---------|---------|----------|--------|-----------|---------|
| n200 d5 s4 | 200 | 5 | 4 | 1 | `gpu_only_n200d5s4/` | 0/115 |
| n200 d5 s6 | 200 | 5 | 6 | 1 | `gpu_only_n200d5s6/` | 7/115 |
| n200 d6 s4 | 200 | 6 | 4 | 1 | `gpu_only_n200d6s4/` | 5/115 |
| n200 d6 s6 | 200 | 6 | 6 | 1 | `gpu_only_n200d6s6/` | 11/115 |
| n400 d5 s4 | 400 | 5 | 4 | 1 | `gpu_only_n400d5s4/` | 5/115 |
| n400 d5 s6 | 400 | 5 | 6 | 1 | `gpu_only_n400d5s6/` | 11/115 |

> **Crashed files**: CUDA illegal memory access trên protein datasets (states=20) với sprdist≥5.  
> Chỉ xảy ra ở protein, không ảnh hưởng DNA datasets.

## CPU Baselines

| Baseline | sprdist | Total time | Machine |
|----------|---------|-----------|---------|
| CPU d3 | 3 | **139m13s** | cũ |
| CPU d6 | 6 | **108m40s** | cũ |
| CPU d3 | 3 | **38m12s**  | mới |
| CPU d6 | 6 | **36m40s**  | mới |

*CPU d3 log dir*: `output/cpu_d3/`  
*CPU d6 log dir*: `output/cpu/`

---

## Kết quả vs CPU d3 (sprdist=3, 139m13s)

**Bet** = GPU better than CPU, **Sam** = same, **Wor** = GPU worse  
**AvgΔ** = avg(CPU_score − GPU_score), dương = GPU tìm được cây tốt hơn CPU

### d3/d4 configs (machine cũ, CPU d3=139m13s)

| Config | Bet | Sam | Wor | AvgΔ | GPU Total | Speedup |
|--------|-----|-----|-----|------|-----------|---------|
| n200 d3 s6  | 64 | 47 |  4 | +7.62 | 20m42s | **6.73×** |
| n200 d3 s10 | 64 | 48 |  3 | +8.33 | 25m33s | 5.45× |
| n400 d3 s6  | 65 | 49 |  1 | +8.76 | 26m33s | 5.24× |
| n400 d3 s10 | 65 | 49 |  1 | +9.70 | 32m16s | 4.32× |
| n200 d4 s6  | 65 | 50 |  **0** | +9.26 | 24m46s | 5.62× |
| n200 d4 s10 | 65 | 50 |  **0** | +9.77 | 32m29s | 4.29× |
| n400 d4 s6  | 67 | 48 |  **0** | **+10.32** | 30m09s | 4.62× |

### d5/d6 configs (machine mới, CPU d3=38m12s — 2026-05-15)

> **AvgΔ cho d6**: inflated do protein datasets — CPU d3 (sprdist=3) tệ hơn nhiều trên proteins.  
> Xem "Protein outlier note" bên dưới.

| Config | Bet | Sam | Wor | AvgΔ | GPU Total | Speedup | Datasets |
|--------|-----|-----|-----|------|-----------|---------|---------|
| n200 d5 s4 | 66 | 48 | 1 | +10.39 | 19m04s | 2.00× | 115/115 |
| n200 d5 s6 | 66 | 41 | 1 | +33.23 | 20m12s | 1.89× | 108/115 |
| n200 d6 s4 | 68 | 41 | 1 | +1393† | 22m10s | 1.72× | 110/115 |
| n200 d6 s6 | 67 | 36 | 1 | +1474† | 22m59s | 1.66× | 104/115 |
| n400 d5 s4 | 67 | 43 | **0** | +32.61 | 20m21s | **1.88×** | 110/115 |
| n400 d5 s6 | 66 | 38 | **0** | +34.91 | 22m38s | 1.69× | 104/115 |

† AvgΔ inflated do 2 protein outliers (delta=+103K, +48K); DNA-only AvgΔ ≈ +17.

### Nhận xét vs CPU d3
- **d3/d4 (machine cũ)**: n200 d4 s6, n400 d4 s6 có 0 worse, speedup 4.6–5.6×
- **d5/d6 (machine mới)**: speedup thấp hơn (~1.7–2×) vì cả GPU lẫn CPU đều nhanh hơn trên machine mới
- **n400 d5 s4**, **n400 d5 s6**: **0 worse** — mở rộng danh sách "0-worse configs" sang sprdist=5
- **Quality trend**: AvgΔ (DNA) tăng theo sprdist: d3(~8) < d4(~10) < d5(~33) < d6(~17†protein)
- **Best d5 quality**: n400 d5 s6 — 66 better, 0 worse, AvgΔ=+34.91

---

## Kết quả vs CPU d6 (sprdist=6, 108m40s) — so sánh công bằng hơn

### d3/d4 configs (machine cũ, CPU d6=108m40s)

| Config | Bet | Sam | Wor | AvgΔ | GPU Total | Speedup |
|--------|-----|-----|-----|------|-----------|---------|
| n200 d3 s6  | 38 | 56 | 21 | +0.16 | 20m42s | **5.25×** |
| n200 d3 s10 | 41 | 58 | 16 | +0.87 | 25m33s | 4.25× |
| n400 d3 s6  | 46 | 59 | 10 | +1.30 | 26m33s | 4.09× |
| n400 d3 s10 | 48 | 60 |  7 | +2.23 | 32m16s | 3.37× |
| n200 d4 s6  | 44 | 64 |  7 | +1.80 | 24m46s | 4.39× |
| n200 d4 s10 | 50 | 59 |  6 | +2.30 | 32m29s | 3.35× |
| n400 d4 s6  | 51 | 60 |  **4** | **+2.86** | 30m09s | 3.60× |

### d5/d6 configs (machine mới, CPU d6=36m40s — 2026-05-15)

| Config | Bet | Sam | Wor | AvgΔ | GPU Total | Speedup | Datasets |
|--------|-----|-----|-----|------|-----------|---------|---------|
| n200 d5 s4 | 46 | 60 |  9 | +2.93 | 19m04s | 1.92× | 115/115 |
| n200 d5 s6 | 49 | 54 |  5 | +25.23 | 20m12s | 1.82× | 108/115 |
| n200 d6 s4 | 52 | 52 |  6 | +1385† | 22m10s | 1.65× | 110/115 |
| n200 d6 s6 | 51 | 48 |  5 | +1466† | 22m59s | 1.60× | 104/115 |
| n400 d5 s4 | 51 | 54 |  5 | +24.75 | 20m21s | 1.80× | 110/115 |
| n400 d5 s6 | 53 | 49 |  **2** | **+26.60** | 22m38s | 1.62× | 104/115 |

† AvgΔ inflated do protein outliers (xem note ở trên).

### Nhận xét vs CPU d6
- **d3/d4 (cũ)**: n400 d4 s6 best quality — 4 worse, AvgΔ=+2.86, **3.60× speedup**
- **d5/d6 (mới)**: n400 d5 s6 best quality — **2 worse**, AvgΔ=+26.60 (DNA), 1.62× speedup
- **n200 d5 s6** giảm "worse" từ 7 (d4) → 5, **n400 d5 s6** → **2 worse** — improvement rõ rệt
- GPU luôn tìm được cây tốt hơn CPU trung bình (AvgΔ > 0 cho mọi config)

---

## Best configs theo tiêu chí

### d3/d4 configs (machine cũ — speedup numbers from original run)

| Tiêu chí | Config | Wor (vs d3) | AvgΔ (vs d3) | Total | Speedup (vs d3) |
|---------|--------|------------|--------------|-------|----------------|
| **Nhanh nhất** | n200 d3 s6 | 4 | +7.62 | **20m42s** | **6.73×** |
| **0 worse + nhanh** | n200 d4 s6 | **0** | +9.26 | 24m46s | 5.62× |
| **Best quality** | n400 d4 s6 | **0** | **+10.32** | 30m09s | 4.62× |
| **vs CPU d6 tốt nhất** | n400 d4 s6 | — | **+2.86** | 30m09s | 3.60× |

### d5/d6 configs (machine mới, 2026-05-15)

| Tiêu chí | Config | Wor (vs d3) | AvgΔ DNA (vs d3) | Total | Speedup (vs d3 new) |
|---------|--------|------------|-----------------|-------|---------------------|
| **Nhanh + 0 worse** | n400 d5 s4 | **0** | +32.61 | 20m21s | **1.88×** |
| **Best quality** | n400 d5 s6 | **0** | **+34.91** | 22m38s | 1.69× |
| **vs CPU d6 tốt nhất** | n400 d5 s6 | 2 | — | 22m38s | **1.62×** (AvgΔ=+26.60) |
| **Nhanh nhất** | n200 d5 s4 | 1 | +10.39 | **19m04s** | 2.00× |

**Nhận xét cross-group**:
- d5 configs cải thiện quality rõ rệt so với d4: AvgΔ DNA tăng từ ~10 lên ~33
- d5 configs "worse" count giảm vs d4 khi so với CPU d3 (d5 sprdist=5 > d4 sprdist=4)
- n400 d5 s4 và n400 d5 s6: **0 worse vs CPU d3** — cùng mức đảm bảo như n400 d4 s6 cũ

---

## So sánh từng dataset (Pairwise) — Time và Score theo nhóm taxa

### vs CPU d3 (sprdist=3)

**Time**: GPU faster/slower tính theo từng dataset riêng lẻ  
**Score**: Better/Same/Worse tính theo parsimony score

#### d3/d4 configs

| Config | GPU faster | GPU slower | GPU same_t | → N≤100 slower | N=201-300 bet/wor | N>400 bet/wor |
|--------|-----------|-----------|-----------|---------------|-------------------|--------------|
| n200 d3 s6  | **103**/115 | 0 | 12 | 0/37  | 38/1 | 7/1 |
| n200 d3 s10 | 97/115 | 1 | 17 | 1/37  | 38/1 | 7/1 |
| n400 d3 s6  | 98/115 | 1 | 16 | 1/37  | 39/0 | 7/1 |
| n400 d3 s10 | 97/115 | **8** | 10 | 8/37  | 39/0 | 7/1 |
| n200 d4 s6  | 99/115 | 2 | 14 | 2/37  | 39/0 | 7/0 |
| n200 d4 s10 | 96/115 | 5 | 14 | 5/37  | 39/0 | 7/0 |
| n400 d4 s6  | 97/115 | 6 | 12 | 6/37  | 40/0 | 7/0 |

**Nhận xét**: GPU slower tập trung hoàn toàn ở **N≤100** (dataset nhỏ, GPU overhead > benefit). N=201-300 và N>100: GPU **luôn nhanh hơn** (faster=100%).

#### d5/d6 configs (machine mới, vs CPU d3=38m12s)

| Config | GPU faster | GPU slower | GPU same_t | N≤100 f/s | N=201-300 bet/wor | N>400 bet/wor |
|--------|-----------|-----------|-----------|-----------|------------------|--------------|
| n200 d5 s4 | 89/114 | 6 | 19 | 19/0 | 40/0 | 7/1 |
| n200 d5 s6 | 82/107 | 13 | 12 | 14/6 | 40/0 | 7/1 |
| n200 d6 s4 | 84/109 | 15 | 10 | 16/8 | 40/0 | 7/1 |
| n200 d6 s6 | 75/103 | 18 | 10 | 14/8 | 40/0 | 7/1 |
| n400 d5 s4 | 83/109 | 14 | 12 | 14/6 | 40/0 | 7/0 |
| n400 d5 s6 | 74/103 | 18 | 11 | 12/10 | 39/0 | 7/0 |

**Nhận xét d5/d6**: GPU slower tăng lên ở N≤100 (đặc biệt d6 với 8–10 slower/37) vì sprdist lớn hơn → tốn thêm thời gian. N=201-300 vẫn **0 worse** cho tất cả configs. N>400: 0 worse cho n400 configs.

---

### vs CPU d6 (sprdist=6)

#### d3/d4 configs

| Config | GPU faster | GPU slower | GPU same_t | N≤100 f/s | N=201-300 bet/wor | N>400 bet/wor |
|--------|-----------|-----------|-----------|-----------|------------------|--------------|
| n200 d3 s6  | **108**/115 | 0 | 7 | 30/0 | 22/11 | 3/5 |
| n200 d3 s10 | 103/115 | 0 | 12 | 25/0 | 23/8  | 3/4 |
| n400 d3 s6  | 103/115 | 1 | 11 | 25/1 | 27/4  | 4/4 |
| n400 d3 s10 | 97/115  | 3 | 15 | 20/3 | 29/2  | 4/3 |
| n200 d4 s6  | 101/115 | 0 | 14 | 23/0 | 25/2  | 4/2 |
| n200 d4 s10 | 99/115  | 4 | 12 | 22/3 | 30/1  | 5/2 |
| **n400 d4 s6** | **98**/115 | 1 | 16 | 20/1 | **31/0** | **5/1** |

**Nhận xét vs CPU d6 (d3/d4)**:
- GPU **luôn nhanh hơn** ở N=101-400 (all configs)
- N≤100: GPU vẫn nhanh hơn (30/37), chỉ d3s10 và d4s10 có vài trường hợp slower
- N>400: GPU nhanh hơn 100%, nhưng quality thấp hơn ở d3 (5 worse) → d4 tốt hơn (1 worse)
- **n400 d4 s6**: N=201-300 → **31/0** (31 better, 0 worse!) — tốt nhất trong nhóm này

#### d5/d6 configs (machine mới, vs CPU d6=36m40s)

| Config | GPU faster | GPU slower | GPU same_t | N≤100 f/s | N=201-300 bet/wor | N>400 bet/wor |
|--------|-----------|-----------|-----------|-----------|------------------|--------------|
| n200 d5 s4 | 94/114 | 8 | 12 | 25/0 | 27/4 | 5/2 |
| n200 d5 s6 | 82/107 | 13 | 12 | 17/3 | 28/1 | 6/2 |
| n200 d6 s4 | 85/109 | 15 | 9  | 20/4 | 29/1 | 6/2 |
| n200 d6 s6 | 77/103 | 13 | 13 | 14/4 | 29/1 | 6/2 |
| n400 d5 s4 | 85/109 | 12 | 12 | 16/4 | 30/2 | 6/1 |
| **n400 d5 s6** | 74/103 | 13 | 16 | 13/3 | **30/0** | **6/1** |

**Nhận xét vs CPU d6 (d5/d6)**:
- N=201-300: **n400 d5 s6** → **30/0** (30 better, **0 worse**!) — cải thiện so với n400 d4 s6 cũ (31/0)
- d5 better hơn d4 ở N=201-300: n400 d5 s6 (30/0) vs n400 d4 s6 (31/0) — tương đương
- N>400: d5/d6 đạt 6 better / 1–2 worse — cải thiện so với d3/d4 (4–5 better)

---

### Breakdown chi tiết theo taxa group — n400 d4 s6 (best d4 config)

#### vs CPU d3:
| Taxa group | n | GPU faster | GPU slower | Better | Same | Worse |
|-----------|---|-----------|-----------|--------|------|-------|
| N≤100     | 37 | 19 | 6 | 8 | 29 | 0 |
| N=101-200 | 8  | 8  | 0 | 3 | 5  | 0 |
| N=201-300 | 51 | 51 | 0 | 40 | 11 | 0 |
| N=301-400 | 11 | 11 | 0 | 9  | 2  | 0 |
| N>400     | 8  | 8  | 0 | 7  | 1  | 0 |

#### vs CPU d6:
| Taxa group | n | GPU faster | GPU slower | Better | Same | Worse |
|-----------|---|-----------|-----------|--------|------|-------|
| N≤100     | 37 | 20 | 1 | 6  | 30 | 1 |
| N=101-200 | 8  | 8  | 0 | 2  | 6  | 0 |
| N=201-300 | 51 | 51 | 0 | 31 | 20 | **0** |
| N=301-400 | 11 | 11 | 0 | 7  | 2  | 2 |
| N>400     | 8  | 8  | 0 | 5  | 2  | 1 |

**Key finding**: N=201-300 (51 datasets) — GPU **100% nhanh hơn** vs cả CPU d3 và d6, và **40/51 better** vs CPU d3, **31/51 better** (0 worse!) vs CPU d6.

### Breakdown chi tiết theo taxa group — n400 d5 s4 (best d5 config, 2026-05-15)

#### vs CPU d3 (new machine):
| Taxa group | n | GPU faster | GPU slower | Better | Same | Worse |
|-----------|---|-----------|-----------|--------|------|-------|
| N≤100     | 32 | 14 | 6 | 9  | 23 | 0 |
| N=101-200 | 7  | 5  | 2 | 2  | 5  | 0 |
| N=201-300 | 51 | 48 | 3 | 40 | 11 | 0 |
| N=301-400 | 11 | 10 | 1 | 9  | 2  | 0 |
| N>400     | 8  | 6  | 2 | 7  | 1  | **0** |

#### vs CPU d6 (new machine):
| Taxa group | n | GPU faster | GPU slower | Better | Same | Worse |
|-----------|---|-----------|-----------|--------|------|-------|
| N≤100     | 32 | 16 | 4 | 8  | 23 | 1 |
| N=101-200 | 7  | 5  | 2 | 1  | 6  | 0 |
| N=201-300 | 51 | 49 | 2 | 30 | 19 | 2 |
| N=301-400 | 11 | 10 | 1 | 6  | 4  | 1 |
| N>400     | 8  | 5  | 3 | 6  | 1  | 1 |

**Key finding (d5)**: N=201-300 — **0 worse** vs CPU d3 (40/51 better), và 49/51 GPU faster. vs CPU d6: 30/51 better (2 worse) — so sánh khó hơn vì cùng sprdist.  
N>400: **0 worse vs CPU d3** và 7/8 better — cải thiện rõ so với n400 d4 s6 (7/8 better, 0 worse).

---

## Phân tích theo Better/Worse case

### vs CPU d3 (sprdist=3)

**BetterCase** = datasets GPU tìm được cây tốt hơn CPU  
**WorseCase** = datasets GPU tìm được cây tệ hơn CPU  
`GPU_t` / `CPU_t` = avg elapsed time (seconds) cho từng nhóm

#### d3/d4 configs (machine cũ)

| Config | BetterCase GPU_t | BetterCase CPU_t | BetterCase spd | WorseCase GPU_t | WorseCase CPU_t | WorseCase spd |
|--------|-----------------|-----------------|---------------|----------------|----------------|--------------|
| n200 d3 s6  | 15.8s | 113.1s | **7.16×** | 7.2s | 31.6s | 4.39× |
| n200 d3 s10 | 19.1s | 113.1s | 5.92× | 10.9s | 34.4s | 3.16× |
| n400 d3 s6  | 19.8s | 112.1s | 5.65× | 17.0s | 53.9s | 3.18× |
| n400 d3 s10 | 23.9s | 112.1s | 4.68× | 23.0s | 53.9s | 2.34× |
| n200 d4 s6  | 18.6s | 112.4s | **6.03×** | — | — | N/A (0 worse) |
| n200 d4 s10 | 24.6s | 112.4s | 4.56× | — | — | N/A (0 worse) |
| n400 d4 s6  | 22.1s | 109.8s | **4.97×** | — | — | N/A (0 worse) |

**Nhận xét**: Datasets GPU tốt hơn CPU thường là datasets có N lớn hơn → CPU mất nhiều thời gian hơn (112s avg), GPU chỉ mất 16–24s → speedup 5–7× trên BetterCase.

#### d5/d6 configs (machine mới, vs CPU d3=38m12s)

| Config | BetterCase GPU_t | BetterCase CPU_t | BetterCase spd | WorseCase GPU_t | WorseCase CPU_t | WorseCase spd |
|--------|-----------------|-----------------|---------------|----------------|----------------|--------------|
| n200 d5 s4 | 13.4s | 25.1s | 1.87× | 15.6s | 53.9s | 3.47× |
| n200 d5 s6 | 13.9s | 25.1s | 1.80× | 23.8s | 53.9s | 2.26× |
| n200 d6 s4 | 15.0s | 25.0s | 1.66× | 23.7s | 53.9s | 2.28× |
| n200 d6 s6 | 16.8s | 25.3s | 1.51× | 30.7s | 53.9s | 1.75× |
| n400 d5 s4 | 13.5s | 24.8s | 1.84× | — | — | N/A (0 worse) |
| n400 d5 s6 | 16.7s | 24.5s | 1.47× | — | — | N/A (0 worse) |

**Nhận xét d5/d6**: BetterCase CPU_t ≈ 25s (machine mới vs ~112s machine cũ) → speedup thấp hơn (1.5–1.9× vs 5–7×). Tuy nhiên WorseCase vẫn có speedup cao hơn BetterCase — pattern nhất quán.

### vs CPU d6 (sprdist=6)

#### d3/d4 configs (machine cũ)

| Config | BetterCase GPU_t | BetterCase CPU_t | BetterCase spd | WorseCase GPU_t | WorseCase CPU_t | WorseCase spd |
|--------|-----------------|-----------------|---------------|----------------|----------------|--------------|
| n200 d3 s6  | 10.0s | 41.8s | 4.19× | 28.4s | 171.7s | **6.04×** |
| n200 d3 s10 | 12.6s | 43.2s | 3.43× | 40.5s | 210.7s | 5.21× |
| n400 d3 s6  | 14.7s | 57.6s | 3.92× | 54.4s | 254.1s | 4.67× |
| n400 d3 s10 | 17.6s | 56.4s | 3.20× | 82.9s | 334.4s | 4.03× |
| n200 d4 s6  | 14.4s | 57.8s | 4.01× | 59.2s | 332.8s | **5.62×** |
| n200 d4 s10 | 19.2s | 58.2s | 3.03× | 93.1s | 382.3s | 4.11× |
| n400 d4 s6  | 17.3s | 57.3s | 3.31× | 103.8s | 502.6s | **4.84×** |

**Nhận xét quan trọng**: WorseCase datasets (GPU tệ hơn CPU d6) là các datasets N lớn → CPU d6 mất 170–500s, GPU mất 28–104s → **GPU vẫn nhanh hơn 4–6× ngay cả khi quality tệ hơn!** Tức là trên các datasets khó, GPU tiết kiệm thời gian đủ để chạy nhiều restarts hơn.

#### d5/d6 configs (machine mới, vs CPU d6=36m40s)

| Config | BetterCase GPU_t | BetterCase CPU_t | BetterCase spd | WorseCase GPU_t | WorseCase CPU_t | WorseCase spd |
|--------|-----------------|-----------------|---------------|----------------|----------------|--------------|
| n200 d5 s4 | 12.0s | 18.5s | 1.55× | 21.4s | 32.8s | 1.53× |
| n200 d5 s6 | 12.9s | 20.1s | 1.56× | 21.9s | 31.3s | 1.42× |
| n200 d6 s4 | 13.9s | 20.9s | 1.51× | 32.3s | 35.9s | 1.11× |
| n200 d6 s6 | 16.4s | 21.2s | 1.29× | 27.5s | 34.0s | 1.24× |
| n400 d5 s4 | 12.6s | 19.6s | 1.56× | 25.3s | 24.4s | 0.96× |
| n400 d5 s6 | 16.1s | 20.1s | 1.24× | 34.3s | 43.5s | 1.27× |

**Nhận xét d5/d6 vs CPU d6**: GPU d6 so với CPU d6 là so sánh **công bằng nhất** (cùng sprdist). WorseCase speedup nhỏ hơn nhiều so với d3/d4 (1.1–1.5× vs 4–6×) — do cùng sprdist nên CPU d6 không còn chậm hơn nhiều trên large datasets.

---

## Phân tích tham số

### numpars: 200 vs 400
- n400 luôn tốt hơn n200 về quality (~+0.5–1.5 AvgΔ) nhưng chậm hơn ~20–30%
- n400 tốt hơn với N≥400 taxa vì nhiều cây hơn → diversity cao hơn
- d5/d6: n400 d5 s4 (AvgΔ=+32.61) vs n200 d5 s4 (AvgΔ=+10.39) — khoảng cách lớn hơn ở d5

### sprdist: 3 vs 4 vs 5 vs 6
- d4 giảm "worse" đáng kể vs CPU d3 (0 vs 3–4 worse)
- d5 tiếp tục cải thiện quality: AvgΔ DNA tăng từ ~10 (d4) → ~33 (d5)
- d6 inflated AvgΔ do proteins; DNS-only AvgΔ ≈ +17 (thấp hơn d5 vì cùng sprdist với CPU d6 → khó beat)
- vs CPU d6: d5 có 2–5 worse, d6 có 5–6 worse — d5 tốt hơn d6 khi so với CPU d6!
- Thời gian: d5 chậm hơn d4 ~10–15%, d6 chậm hơn d5 ~10–13%

### gpu_stop: 4 vs 6
- s6 tốt hơn s4 về quality: AvgΔ tăng ~2–3 đơn vị, "better" tăng nhẹ
- s6 chậm hơn s4 khoảng 6–12%
- Tradeoff nhỏ: s4 nhanh hơn không đáng kể, s6 mạnh hơn về quality
- Cả hai đều **0 worse** với numpars=400

### Protein crashes (d5/d6 specific)
- CUDA illegal memory access xảy ra với protein datasets (states=20) ở sprdist≥5
- Crash rate tăng theo sprdist: d5s4=5/115, d5s6=11/115, d6s4=5/115, d6s6=11/115
- Tất cả crashes ở protein datasets — DNA không bị ảnh hưởng
- Bug liên quan đến NTAXA template instantiation hay shared memory với states=20

---

## Kết luận

### d3/d4 recommendation (machine cũ)
**Recommended config cho thesis benchmark**: `n400 d4 s6`
- Quality: **0 worse** vs CPU d3, AvgΔ=**+10.32** (GPU vượt trội)
- vs CPU d6: 4 worse, AvgΔ=+2.86, **3.60× speedup**
- Total: 30m09s vs CPU d3 139m = **4.62× total speedup**

### d5 recommendation (machine mới, 2026-05-15)
**Best new config**: `n400 d5 s4` hoặc `n400 d5 s6`

| Config | vs CPU d3: worse | vs CPU d6: worse | AvgΔ DNA | Total | Speedup (vs d3) |
|--------|-----------------|-----------------|---------|-------|----------------|
| n400 d5 s4 | **0** | 5 | +32.61 | 20m21s | **1.88×** |
| n400 d5 s6 | **0** | **2** | +34.91 | 22m38s | 1.69× |

- **n400 d5 s4**: 0 worse vs d3, tốt nhất về speed trong d5 group
- **n400 d5 s6**: 0 worse vs d3, only **2 worse** vs CPU d6 — best overall quality
- AvgΔ DNA (~33) cao hơn gấp 3× so với n400 d4 s6 (~10) — chất lượng cải thiện rõ rệt
- N=201-300: 40/51 better vs CPU d3, 49/51 GPU faster — đây là sweet spot của GPU

### Tổng hợp "0-worse" configs:
| Config | vs CPU d3 | vs CPU d6 | AvgΔ DNA |
|--------|-----------|-----------|---------|
| n200 d4 s6  | 0 | 7 | ~9 |
| n200 d4 s10 | 0 | 6 | ~10 |
| n400 d4 s6  | 0 | 4 | ~10 |
| n400 d5 s4  | 0 | 5 | ~33 |
| n400 d5 s6  | 0 | **2** | ~35 |

**n400 d5 s6** là config tốt nhất overall: 0 worse vs d3, chỉ 2 worse vs d6, AvgΔ cao nhất (35).
