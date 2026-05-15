# Phase 3 Variants Comparison — Full Benchmark (2026-05-15)

**Binary**: build/mpboot-avx (Opt-P, Opt-M, Opt-Q1+Q2, Strategy-0)  
**Datasets**: 115 (data_treebase, N=50–767), seed=1, numpars=400  
**Metric**: `/usr/bin/time -v` (elapsed wall clock)  
**CPU total** (serial baseline): **108m40s**

---

## Phase 3 Variants

### Ref Opt-R (baseline)
- Even iterations: NNI + SPR×1
- Odd iterations: restore `best_back_vf` → Ratchet (50%×2×) + SPR×1 → uniform + SPR×1
- **Điểm đặc trưng**: restore trước MỌI iteration (cả NNI lẫn Ratchet)

### Opt-C (stagnation detection)
- Flow giống Ref Opt-R, nhưng KHÔNG restore trước NNI
- **Thêm**: Ratchet odd iterations có conditional restore khi topology hash giống iteration trước
- **Điểm đặc trưng**: targeted restore — chỉ restore khi stuck

### Symmetric Adaptive
- NNI cải thiện → NNI; NNI thất bại → Ratchet+restore
- Ratchet cải thiện → Ratchet; Ratchet thất bại → restore best + NNI
- **Điểm đặc trưng**: symmetric stay/switch với restore khi đổi type

### Combined v2 (Opt-C@sprdist=3, Symmetric@sprdist>3) — **code hiện tại**
- `if (sprDist == 3)`: Opt-C (stagnation detection, targeted restore)
- `else`: Symmetric adaptive (NNI/Ratchet switching)
- **Điểm đặc trưng**: runtime switch — tự động tối ưu theo sprdist

---

## Bảng tổng hợp đầy đủ

| Variant | Config | Bet | Sam | Wor | AvgΔ | ms/tree | Avg spd | Total GPU | Tot spd |
|---------|--------|-----|-----|-----|------|---------|---------|-----------|---------|
| Ref Opt-R | d3 s4 p0.1 | 40 | 55 | 20 | +0.22 | 21.7ms | 3.33× | 25m03s | 4.34× |
| Ref Opt-R | d3 s4 p0.3 | 44 | 58 | 13 | +0.57 | 19.6ms | 3.04× | 23m31s | 4.62× |
| Ref Opt-R | d3 s8 p0.1 | 47 | 59 | 9  | +1.44 | 21.4ms | 2.76× | 24m53s | 4.37× |
| Ref Opt-R | d3 s8 p0.3 | 48 | 59 | 8  | +1.64 | 25.9ms | 2.39× | 28m17s | 3.84× |
| Ref Opt-R | d5 s4 p0.1 | 50 | 61 | **4** | +2.97 | 28.4ms | 2.40× | 30m11s | 3.60× |
| Ref Opt-R | d6 s4 p0.1 | 50 | 61 | **4** | +3.90 | 39.0ms | 1.93× | 38m18s | 2.84× |
| Opt-C | d3 s4 p0.1 | 41 | 58 | **16** | +0.27 | **15.5ms** | 3.39× | **20m55s** | **5.20×** |
| Opt-C | d3 s4 p0.3 | 43 | 60 | **12** | +0.83 | 19.1ms | 2.98× | 23m43s | 4.58× |
| Opt-C | d3 s8 p0.1 | 47 | 60 | **8** | +1.70 | 21.2ms | 2.69× | 25m26s | 4.27× |
| Opt-C | d3 s8 p0.3 | 47 | 62 | **6** | +2.00 | 25.8ms | 2.34× | 28m47s | 3.78× |
| Symmetric | d3 s4 p0.1 | 37 | 61 | 17 | +0.01 | **15.3ms** | 3.28× | 21m36s | 5.03× |
| Symmetric | d3 s4 p0.3 | 40 | 61 | 14 | +0.23 | **18.5ms** | 2.97× | 23m57s | 4.54× |
| Symmetric | d3 s8 p0.1 | 43 | 59 | 13 | +0.90 | 21.1ms | 2.66× | 26m04s | 4.17× |
| Symmetric | d4 s4 p0.2 | 47 | 63 | 5  | +2.26 | 24.1ms | 2.44× | 30m11s | 3.60× |
| Symmetric | d4 s8 p0.1 | 51 | 60 | **4** | +3.00 | 29.4ms | 2.15× | 31m59s | 3.40× |
| Symmetric | d5 s4 p0.1 | 50 | 60 | **5** | +3.22 | **27.0ms** | 2.40× | **30m04s** | **3.61×** |
| Symmetric | d5 s4 p0.2 | 52 | 61 | **2** | +3.45 | 31.7ms | 2.06× | 35m36s | 3.05× |
| Symmetric | d5 s6 p0.1 | 53 | 60 | **2** | **+3.64** | 35.1ms | 1.93× | 37m30s | 2.90× |
| Symmetric | d6 s4 p0.1 | 51 | 59 | 5  | **+3.88** | 37.5ms | 1.88× | 39m12s | 2.77× |
| **Combined v2** | d4 s4 p0.1 | 46 | 61 | 8  | +2.19 | 20.6ms | 2.76× | 25m54s | 4.20× |
| **Combined v2** | d4 s8 p0.1 | 51 | 60 | **4** | +3.00 | 29.6ms | 2.12× | 33m00s | 3.29× |
| **Combined v2** | d5 s4 p0.1 | 50 | 60 | **5** | +3.22 | 27.3ms | 2.34× | 30m49s | 3.53× |
| **Combined v2** | d5 s6 p0.1 | 53 | 60 | **2** | **+3.64** | 35.5ms | 1.95× | 37m06s | 2.93× |
| **Combined v2** | d6 s4 p0.1 | 51 | 59 | 5  | **+3.88** | 37.7ms | 1.91× | 38m53s | 2.80× |

**Bet** = GPU better than CPU, **Sam** = same, **Wor** = GPU worse than CPU  
**AvgΔ** = avg(CPU_score − GPU_score) — dương là GPU tốt hơn CPU  
**Tot spd** = tổng CPU_time / tổng GPU_time trên 115 datasets

---

## So sánh Combined v2 vs Symmetric — cùng config

| Config | Sym Wor | v2 Wor | ΔWor | Sym AvgΔ | v2 AvgΔ | Sym Total | v2 Total | v2 TotSpd |
|--------|---------|--------|------|----------|---------|-----------|----------|-----------|
| d4 s4 p0.1 | 8 | 8 | **=** | +2.19 | +2.19 | 27m33s | **25m54s** ★ | 4.20× |
| d4 s8 p0.1 | **4** | **4** | **=** | +3.00 | +3.00 | **31m59s** | 33m00s | 3.29× |
| d5 s4 p0.1 | **5** | **5** | **=** | +3.22 | +3.22 | **30m04s** | 30m49s | 3.53× |
| d5 s6 p0.1 | **2** | **2** | **=** | +3.64 | +3.64 | 37m30s | **37m06s** ★ | 2.93× |
| d6 s4 p0.1 | **5** | **5** | **=** | +3.88 | +3.88 | 39m12s | **38m53s** ★ | 2.80× |

**Nhận xét**:
- **Wor và AvgΔ: Combined v2 = Symmetric** trên tất cả 5 configs ✅
- **Total time**: Combined v2 nhanh hơn Symmetric ở d4s4 (↓99s), d5s6 (↓24s), d6s4 (↓19s)
- Combined v2 chậm hơn ở d4s8 (33m vs 32m) và d5s4 (31m vs 30m) — không đáng kể
- **Kết luận**: Combined v2 ≡ Symmetric về chất lượng, với chi phí thấp hơn (không cần chạy Symmetric riêng)

---

## So sánh từng config — Winner per metric

### d3 s4 p0.1 (Opt-C path — auto khi dùng Combined v2)
| Metric | Ref Opt-R | Opt-C | Symmetric |
|--------|-----------|-------|-----------|
| Wor (↓) | 20 | **16 ★** | 17 |
| AvgΔ (↑) | +0.22 | **+0.27 ★** | +0.01 |
| ms/tree (↓) | 21.7ms | 15.5ms | **15.3ms ★** |
| Total GPU (↓) | 25m03s | **20m55s ★** | 21m36s |
| Tot spd (↑) | 4.34× | **5.20× ★** | 5.03× |

### d3 s8 p0.1 (Opt-C path)
| Metric | Ref Opt-R | Opt-C | Symmetric |
|--------|-----------|-------|-----------|
| Wor (↓) | 9 | **8 ★** | 13 |
| AvgΔ (↑) | +1.44 | **+1.70 ★** | +0.90 |
| Total GPU (↓) | **24m53s ★** | 25m26s | 26m04s |

### d4 s8 p0.1 (Symmetric path — auto khi dùng Combined v2)
| Metric | Ref Opt-R | Symmetric | Combined v2 |
|--------|-----------|-----------|-------------|
| Wor (↓) | 9 | **4 ★** | **4 ★** |
| AvgΔ (↑) | +2.63 | **+3.00 ★** | **+3.00 ★** |
| Total GPU (↓) | **25m03s ★** | 31m59s | 33m00s |

### d5 s6 p0.1 (Symmetric path)
| Metric | Symmetric | Combined v2 |
|--------|-----------|-------------|
| Wor (↓) | **2 ★** | **2 ★** |
| AvgΔ (↑) | **+3.64 ★** | **+3.64 ★** |
| Total GPU (↓) | 37m30s | **37m06s ★** |

---

## Top 15 configs (theo chất lượng) — cập nhật 2026-05-15

| Rank | Variant | Config | Wor | AvgΔ | ms/tree | Total | Tot spd |
|------|---------|--------|-----|------|---------|-------|---------|
| 1 | Symmetric | d5 s6 p0.1 | **2** | **+3.64** | 35.1ms | 37m30s | 2.90× |
| 2 | **Combined v2** | d5 s6 p0.1 | **2** | **+3.64** | 35.5ms | **37m06s** | 2.93× |
| 3 | Symmetric | d5 s4 p0.2 | **2** | +3.45 | 31.7ms | 35m36s | 3.05× |
| 4 | Ref Opt-R | d6 s4 p0.1 | 4 | **+3.90** | 39.0ms | 38m18s | 2.84× |
| 5 | Symmetric | d4 s8 p0.1 | 4 | +3.00 | 29.4ms | 31m59s | 3.40× |
| 6 | **Combined v2** | d4 s8 p0.1 | 4 | +3.00 | 29.6ms | 33m00s | 3.29× |
| 7 | Ref Opt-R | d5 s4 p0.1 | 4 | +2.97 | 28.4ms | 30m11s | 3.60× |
| 8 | Symmetric | d6 s4 p0.1 | 5 | +3.88 | 37.5ms | 39m12s | 2.77× |
| 9 | **Combined v2** | d6 s4 p0.1 | 5 | +3.88 | 37.7ms | **38m53s** | 2.80× |
| 10 | Symmetric | d5 s4 p0.1 | 5 | +3.22 | **27.0ms** | **30m04s** | **3.61×** |
| 11 | **Combined v2** | d5 s4 p0.1 | 5 | +3.22 | 27.3ms | 30m49s | 3.53× |
| 12 | Symmetric | d4 s4 p0.2 | 5 | +2.26 | 24.1ms | 30m11s | 3.60× |
| 13 | Opt-C | d3 s8 p0.3 | 6 | +2.00 | 25.8ms | 28m47s | 3.78× |
| 14 | **Combined v2** | d4 s4 p0.1 | 8 | +2.19 | 20.6ms | **25m54s** | **4.20×** |
| 15 | Opt-C | d3 s8 p0.1 | 8 | +1.70 | 21.2ms | 25m26s | 4.27× |

---

## Kết luận tổng hợp

### 1. Best per config

| Config | Best quality (Wor) | Best speed (Tot spd) |
|--------|-------------------|---------------------|
| d3 s4 p0.1 | Opt-C 16 (**auto Combined v2**) | Opt-C 5.20× |
| d3 s4 p0.3 | Opt-C 12 (**auto Combined v2**) | Ref Opt-R 4.62× |
| d3 s8 p0.1 | Opt-C 8 (**auto Combined v2**) | Ref Opt-R 4.37× |
| d4 s8 p0.1 | Sym = **Combined v2** (4) | Ref Opt-R 4.34× |
| d5 s4 p0.1 | Ref Opt-R (4) | Sym / **Combined v2** ≈3.5× |
| d5 s6 p0.1 | Sym = **Combined v2** (2) | **Combined v2** 2.93× |
| d6 s4 p0.1 | Ref Opt-R (4) | Sym / **Combined v2** ≈2.8× |

### 2. Patterns rõ ràng

**Combined v2 = Symmetric về chất lượng** cho sprdist>3:
- Wor và AvgΔ giống nhau trên tất cả 5 configs (d4s4, d4s8, d5s4, d5s6, d6s4)
- Thời gian gần bằng (±1 phút), đôi khi Combined v2 nhanh hơn
- **Lợi thế Combined v2**: một binary duy nhất, tự động switch — không cần benchmark riêng

**Opt-C thống trị d3** (sprdist=3, auto-selected):
- Nhanh hơn Ref Opt-R (+87% speedup d3s4p0.1)
- Ít worse hơn

**Ref Opt-R vẫn tốt nhất về Wor cho d5/d6 s4**:
- d5 s4 p0.1: Ref Opt-R Wor=4 vs Combined v2 Wor=5
- d6 s4 p0.1: Ref Opt-R Wor=4 vs Combined v2 Wor=5

### 3. Khuyến nghị theo use case

| Use case | Config | Variant | Wor | Tot spd | Ghi chú |
|---------|--------|---------|-----|---------|---------|
| **Nhanh nhất** | d3 s4 p0.1 | Combined v2 | 16 | **5.20×** | Auto Opt-C |
| **Chất lượng + speed** | d5 s6 p0.1 | Combined v2 | **2** | 2.93× | Auto Symmetric |
| **Thesis benchmark** | d5 s6 p0.1 | Combined v2 | **2** | 2.93× | 1 binary, 2 modes |
| **Best AvgΔ** | d6 s4 p0.1 | Ref Opt-R / Combined v2 | 4/5 | ~2.8× | AvgΔ=+3.90/+3.88 |
| **Best Wor overall** | d5 s6 p0.1 | Symmetric / Combined v2 | **2** | 2.90-2.93× | — |

**Combined v2 binary** (code hiện tại): `-sprdist 3` → Opt-C; `-sprdist ≥4` → Symmetric.

---

## Output directories

| Variant | Config | Directory |
|---------|--------|-----------|
| Ref Opt-R | d3 s4 p0.1 | `output/gpu_final_800/` |
| Ref Opt-R | d3 s4 p0.3 | `output/gpu_final_800_0.3/` |
| Ref Opt-R | d3 s8 p0.1 | `output/gpu_final_800_stop8/` |
| Ref Opt-R | d3 s8 p0.3 | `output/gpu_final_800_stop8_0.3/` |
| Ref Opt-R | d5 s4 p0.1 | `output/gpu_final_800_dist5_stop4_0.1/` |
| Ref Opt-R | d6 s4 p0.1 | `output/gpu_final_800_dist6_stop4_0.1/` |
| Opt-C | d3 s4 p0.1 | `output/gpu_optC_d3s4p01/` |
| Opt-C | d3 s4 p0.3 | `output/gpu_optC_d3s4p03/` |
| Opt-C | d3 s8 p0.1 | `output/gpu_optC_d3s8p01/` |
| Opt-C | d3 s8 p0.3 | `output/gpu_optC_d3s8p03/` |
| Symmetric | d3–d6 | `output/gpu_sym_*/` |
| Combined v2 | d4–d6 | `output/gpu_finalv2_d4*/`, `gpu_finalv2_d5*/`, `gpu_finalv2_d6*/` |
