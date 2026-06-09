# Benchmark — K' < K (k1_ratio=0.2 vs k1_ratio=1.0)

**Ngày**: 2026-05-17  
**Hardware**: NVIDIA A100-SXM4-80GB (devices 0–7)  
**Config**: numpars=1000, pool_size=30, seed=1, 115 datasets (`data_treebase/`)  
**Comparison**: K'=0.2K (`-gpu_k1_ratio 0.2`) vs K'=K (baseline, default ratio=1.0)

## Kết quả tổng hợp

| Config | sprdist | gpu_stop | K'=K speedup | K'=0.2K speedup | Δtotal | K1 Δ | K2 Δ | GPU wins (K'=K→0.2K) |
|--------|---------|----------|-------------|----------------|--------|------|------|----------------------|
| d3s10 | 3 | 10 | 8.04× | **10.11×** | **+25.8%** | −59.7% | +15.6% | 57→64/115 |
| d4s6  | 4 | 6  | 6.70× | **8.21×**  | **+22.5%** | −59.6% | +16.2% | 53→64/115 |
| d4s20 | 4 | 20 | 5.82× | **7.41×**  | **+27.4%** | −59.8% | +0.7%  | 52→65/115 |
| d5s6  | 5 | 6  | 5.46× | **6.61×**  | **+21.1%** | −59.4% | +15.2% | 48→58/115 |
| d6s6  | 6 | 6  | 4.55× | **5.35×**  | **+17.6%** | −58.8% | +18.5% | 45→61/115 |

K'=0.2K **vượt trội K'=K trên tất cả 5 configs**, không config nào tệ hơn.

## Output directories

- K'=0.2K: `output/hybrid_pool_lessk1_n1000p30d{3s10,4s6,4s20,5s6,6s6}/`
- K'=K baseline: `output/hybrid_pool_n1000p30d{3s10,4s6,4s20,5s6,6s6}/`

## Nhận xét

- **K1 giảm ~60%** trên tất cả configs (consistent với K'/K = 0.2 trong 1-wave regime)
- **K2 tăng 15–19%** ở phần lớn configs: shorter K1 → CPU builds more trees → better pool → K2 ít stagnate hơn
- **d4s20 exception** (K2 chỉ +0.7%): stop=20 là barrier → K2 đã run đủ iterations với K'=K; improvement từ pool không cắt ngắn được
- **GPU wins tăng 7–13 datasets** (6–25%): pool tốt hơn → K2 tìm được cây tốt hơn

## Benchmark scripts

```bash
# K'=0.2K (Opt-LessK1)
bash run_bench_k1ratio.sh 3 10 0.2 0 /output/hybrid_pool_lessk1_n1000p30d3s10
bash run_bench_k1ratio.sh 4 6  0.2 1 /output/hybrid_pool_lessk1_n1000p30d4s6
bash run_bench_k1ratio.sh 4 20 0.2 2 /output/hybrid_pool_lessk1_n1000p30d4s20
bash run_bench_k1ratio.sh 5 6  0.2 3 /output/hybrid_pool_lessk1_n1000p30d5s6
bash run_bench_k1ratio.sh 6 6  0.2 4 /output/hybrid_pool_lessk1_n1000p30d6s6

# K'=K baseline (gpu_k1_ratio=1.0 default)
bash run_bench.sh 3 10 5 /output/hybrid_pool_n1000p30d3s10
bash run_bench.sh 4 6  6 /output/hybrid_pool_n1000p30d4s6
bash run_bench.sh 4 20 2 /output/hybrid_pool_n1000p30d4s20  # pre-existing
bash run_bench.sh 5 6  7 /output/hybrid_pool_n1000p30d5s6
bash run_bench.sh 6 6  0 /output/hybrid_pool_n1000p30d6s6   # pre-existing
```
