# Numpars Scaling Benchmark: 5 Datasets × 3 Numpars

**Settings**: margin=off, sprdist=3, gpu_stop=4, gpu_hc_iter=10, seed=101/102 (best of 2 reps)  
**Date**: 2026-05-12

## Raw data

```
taxa   numpars  time_ms      ms_tree    best    
--------------------------------------------------------------
201    200      10857.8      54.56      11358   
201    300      19116.6      63.94      11358   
201    400      21976.6      55.08      11358   

295    200      29859.3      150.05     6663    
295    300      34144.2      114.19     6663    
295    400      39873.8      99.93      6663    

413    200      71908.8      361.35     48530   
413    300      76794.8      256.84     48530   
413    400      72756.8      182.35     48532   

504    200      68526.8      344.36     138016  
504    300      85831.2      287.06     137998  
504    400      97719.5      244.91     138016  

640    200      310922.7     1562.43    256916  
640    300      351650.4     1176.09    256918  
640    400      367375.7     920.74     256917  
```

## Datasets used

| taxa | file | sites |
|------|------|-------|
| 201  | dna_M4720_201_2899.phy  | 2899  |
| 295  | dna_M214_295_1836.phy   | 1836  |
| 413  | dna_M5381_413_3632.phy  | 3632  |
| 504  | dna_M9915_504_2757.phy  | 2757  |
| 640  | dna_M7964_640_25260.phy | 25260 |

## ms/tree improvement (numpars 200 → 400)

| taxa | 200 ms/tree | 300 ms/tree | 400 ms/tree | Improvement | Total time ×factor |
|------|-------------|-------------|-------------|-------------|-------------------|
| 201  | 54.6        | 63.9 ↑      | 55.1        | ~flat       | ×2.02             |
| 295  | 150.1       | 114.2       | 99.9        | −33%        | ×1.34             |
| 413  | 361.4       | 256.8       | 182.4       | **−50%**    | **×1.01**         |
| 504  | 344.4       | 287.1       | 244.9       | −29%        | ×1.43             |
| 640  | 1562.4      | 1176.1      | 920.7       | **−41%**    | ×1.18             |

## Key findings

1. **ms/tree giảm khi tăng numpars** — GPU latency hiding: nhiều blocks → SM pipeline đầy hơn
2. **413 taxa sweet spot**: numpars=400 NHANH hơn numpars=300 trong tổng thời gian (72756 < 76794 ms)
3. **Scaling sub-linear**: 2× cây chỉ tốn 1.01×–2.0× thời gian
4. **640 taxa rất chậm** (25260 sites) — cần Opt-G để có speedup
5. **201 taxa**: numpars=300 chậm hơn numpars=200 và 400 (ms/tree: 63.9 vs 54.6/55.1) — nhỏ quá, GPU không saturate đều

## GPU capacity context

- A100: 108 SMs × 5 blocks/SM = **540 simultaneous blocks** max (bottleneck: shared mem 28.4 KB/block)
- K=199, 299, 399 đều < 540 → tất cả chạy đồng thời (không queuing)
- numpars=400 tốt hơn numpars=200 vì many blocks → better SM pipelining

## Recommendations

| taxa range | Recommended numpars | Reason |
|------------|--------------------|----|
| ≤ 200      | 200                | Không gain từ more trees |
| 200–400    | 400                | 2× cây chỉ +34% time, ms/tree −33% |
| 400–640    | 400                | Near-free scaling (×1.01–1.18), ms/tree −41–50% |
| > 640      | 400+               | Cần test thêm; latency hiding tiếp tục scale |
