# GPU Final Benchmark — 3 Config vs CPU (2026-05-14)

**Binary**: build/mpboot-avx (tất cả optimizations: Opt-P, Opt-M, Opt-Q1+Q2, Opt-R)  
**Datasets**: 115 (data_treebase, N=50–767), seed=1, numpars=400, sprdist=3  
**CPU binary**: copy-build/mpboot-avx (baseline serial)  
**Metric time**: `/usr/bin/time -v` (elapsed wall clock)

## Tóm tắt 3 config

| Metric | stop=4, pct=0.1 | stop=4, pct=0.3 | stop=8, pct=0.1 |
|--------|----------------|----------------|----------------|
| Output dir | `gpu_final_800` | `gpu_final_800_0.3` | `gpu_final_800_stop8` |
| Avg speedup vs CPU | **3.33×** | 3.04× | 2.76× |
| Min speedup | 1.01× | 0.99× | 0.69× |
| Max speedup | **11.61×** | 9.94× | 9.07× |
| GPU better quality | 40/115 (35%) | 44/115 (38%) | **47/115 (41%)** |
| Same quality | 55/115 (48%) | 58/115 (50%) | 59/115 (51%) |
| GPU worse quality | 20/115 (17%) | 13/115 (11%) | **9/115 (8%)** |
| Avg Δparsimony (GPU−CPU) | −0.22 | −0.57 | **−1.44** |

## Speedup theo nhóm taxa

| N range | stop=4 pct=0.1 | stop=4 pct=0.3 | stop=8 pct=0.1 |
|---------|---------------|---------------|---------------|
| N≤99 | 3.25× (n=34) | 2.79× | 2.41× |
| N=100–199 | **4.44×** (n=11) | 3.80× | 3.43× |
| N=200–299 | 2.94× (n=50) | 2.62× | 2.37× |
| N=300–399 | 3.34× (n=12) | 3.15× | 2.88× |
| N≥400 | 4.55× (n=8) | 5.54× | **5.59×** |

**Nhận xét**: N≥400 — stop=8 và pct=0.3 đều tốt hơn pct=0.1 (đủ thời gian converge)

## Trade-off

- **Muốn nhanh**: stop=4, pct=0.1 → avg **3.33×** speedup
- **Muốn cân bằng**: stop=4, pct=0.3 → ít worse hơn (13 vs 20), Δ tốt hơn (−0.57)
- **Muốn chất lượng**: stop=8, pct=0.1 → ít worse nhất (9), Δ tốt nhất (−1.44)

## Top 5 speedup (nhất quán cả 3 config)

| Dataset | N | stop=4 pct=0.1 | stop=4 pct=0.3 | stop=8 pct=0.1 |
|---------|---|---|---|---|
| prot_M10273_169_11009 | 169 | 11.61× | 9.94× | 8.70× |
| dna_M12051_699_6914 | 699 | 5.45× | 8.69× | 9.07× |
| dna_M7024_767_5814 | 767 | 7.88× | 7.25× | 7.93× |
| prot_M10372_169_22426 | 169 | 7.64× | 6.77× | 6.13× |
| prot_M8630_50_21154 | 50 | 6.69× | 6.84× | 4.44× |

## Dataset GPU thua CPU trong TẤT CẢ 3 config

| Dataset | N | Δ pct=0.1 | Δ pct=0.3 | Δ stop=8 |
|---------|---|---|---|---|
| dna_M7964_640_25260 | 640 | +20 | +12 | +12 |
| dna_M10933_229_2696 | 229 | +11 | +4 | +2 |
| dna_M5381_413_3632 | 413 | +8 | +5 | +2 |
| prot_M8461_89_5699 | 89 | +5 | +5 | +4 |
| dna_M11113_344_9778 | 344 | +4 | +3 | +4 |
| dna_M5078_265_9768 | 265 | +3 | +3 | +1 |
| dna_M14582_372_61199 | 372 | +2 | +2 | +1 |
| dna_M14678_225_2673 | 225 | +1 | +1 | +1 |
| dna_M1838_228_1131 | 228 | +1 | +1 | +1 |

**Khó nhất**: N=640 (dna_M7964) — tất cả params đều GPU thua. Cần sprdist lớn hơn hoặc numpars cao hơn.

## Files Excel

- `output/results_gpu_final_800_vs_cpu.xlsx` — stop=4, pct=0.1
- `output/results_gpu_final_800_0.3_vs_cpu.xlsx` — stop=4, pct=0.3
- `output/results_gpu_final_800_stop8_vs_cpu.xlsx` — stop=8, pct=0.1
