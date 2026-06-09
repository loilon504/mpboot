# NCU Profiling — 3 GPU Kernels (Bootstrap Mode)

**Ngày**: 2026-05-19  
**Hardware**: NVIDIA A100-SXM4-80GB, device 1, CC 8.0  
**Dataset**: `data_debug/tree1.phy` — N=295 taxa, DNA  
**Command**:
```bash
ncu --launch-count 5 \
    --metrics gpu__time_duration.sum,launch__registers_per_thread, \
              launch__shared_memory_per_block_static, \
              sm__maximum_warps_per_active_cycle_pct, \
              sm__warps_active.avg.pct_of_peak_sustained_active, \
              dram__throughput.avg.pct_of_peak_sustained_elapsed, \
              smsp__cycles_active.avg.pct_of_peak_sustained_elapsed, \
              l1tex__t_sector_hit_rate.pct \
    ./mpboot-avx -s ../data_debug/tree1.phy \
    -use_gpu -seed 1 -sprdist 6 -gpu_device 1 -gpu_worker 200 \
    -bb 1000 -gpu_pool_size 20 -gpu_nni_strength 0.5
```

**Report file**: `/tmp/ncu_all3_v2.ncu-rep`

---

## Launch order quan sát được (5 launches)

1. `buildParsimonyTreesKernel<4,800>` — 1 lần (K1, initial build)
2. `buildPhase3Kernel<4,800>` — 1 lần (K2, 1st outer iteration)
3. `REPSKernel` × 3 — bootstrap scoring (chạy nhiều lần trong outer loop)

---

## Raw metrics

### K1 — `buildParsimonyTreesKernel<4,800>`

Grid: **(100, 1, 1)** × (32, 1, 1) — k1_count=100 (numpars=101)

| Metric | Value |
|--------|-------|
| `gpu__time_duration.sum` | **1.79 s** |
| `launch__registers_per_thread` | **96** reg/thread |
| `launch__shared_memory_per_block_static` | (!) n/a |
| `sm__maximum_warps_per_active_cycle_pct` | **20.31%** (theor. occ) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **1.56%** (achieved occ) |
| `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` | 16.06% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 0.09% |
| `l1tex__t_sector_hit_rate.pct` | **95.64%** |

---

### K2 — `buildPhase3Kernel<4,800>`

Grid: **(200, 1, 1)** × (32, 1, 1) — gpu_worker=200

| Metric | Value |
|--------|-------|
| `gpu__time_duration.sum` | **1.79 s** (1st launch) |
| `launch__registers_per_thread` | **115** reg/thread |
| `launch__shared_memory_per_block_static` | (!) n/a |
| `sm__maximum_warps_per_active_cycle_pct` | **20.31%** (theor. occ) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **2.67%** (achieved occ) |
| `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` | 26.45% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 0.16% |
| `l1tex__t_sector_hit_rate.pct` | **92.76%** |

---

### K3 — `REPSKernel` (avg 3 launches)

Grid: **(1000, 1, 1)** × (32, 1, 1) — B=1000 bootstrap replicates

| Metric | Value |
|--------|-------|
| `gpu__time_duration.sum` | **~10 µs** per launch |
| `launch__registers_per_thread` | **30** reg/thread |
| `launch__shared_memory_per_block_static` | (!) n/a |
| `sm__maximum_warps_per_active_cycle_pct` | **50%** (theor. occ) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **13.16%** (achieved occ) |
| `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` | **69.60%** |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 13.99% |
| `l1tex__t_sector_hit_rate.pct` | 54.27% |

---

## Bảng so sánh

| | K1 (build) | K2 (phase3) | K3 (REPS) |
|-|------------|-------------|-----------|
| Grid | 100 blocks | 200 blocks | 1000 blocks |
| Duration | 1.79 s | 1.79 s (1st) | ~10 µs |
| Registers | 96 | **115** | 30 |
| Theor. Occ | 20.31% | 20.31% | **50%** |
| Achieved Occ | **1.56%** | **2.67%** | 13.16% |
| Compute% | 16.06% | 26.45% | **69.60%** |
| DRAM% | 0.09% | 0.16% | 13.99% |
| L1 hit rate | **95.64%** | **92.76%** | 54.27% |

---

## Phân tích

### K1/K2 — Achieved Occ cực thấp (1.56% / 2.67%)

Không phải kernel bug — là hệ quả của grid nhỏ:
- A100: 108 SMs × 13 blocks/SM (smem-limited) = **1404 blocks capacity**
- K1 dùng 100/1404 = **7% GPU capacity**
- K2 dùng 200/1404 = **14% GPU capacity**

GPU không saturated. Để đạt ≥1 wave cần ≥1404 blocks:
- K1: numpars ≥ 1405 → `k1_count = numpars - 1 ≥ 1404`
- K2: gpu_worker ≥ 1404

Với config hiện tại (numpars=200, gpu_worker=200), K1/K2 chỉ dùng ~1/7 GPU. Đây là trade-off có chủ đích: numpars nhỏ → ít cây → nhanh hơn per-replicate, nhưng lãng phí GPU.

### K2 registers = 115 (không phải 128)

Trước đây đo được 128 sau pool restart simplification (NCU 2026-05-18). Có thể do:
- Bootstrap mode compile khác path (template instantiation khác)
- Config khác (gpu_pool_size=20 thay vì 30)
- Cần xác nhận lại với `ncu --set full` hoặc `ptxas -v`

### K3 REPSKernel — không phải bottleneck

- Duration ~10 µs per launch (cực nhanh)
- Theor. occ **50%** (tốt — ít registers, ít smem)
- Compute% **69.6%** — compute-bound, không phải memory-bound
- Không cần optimize

### `shared_memory_per_block_static = (!) n/a`

NCU không đọc được smem static. Nguyên nhân có thể: `BuildSharedT<NTAXA=800>` được allocated dưới dạng dynamic shared memory (`extern __shared__`) hoặc NCU không resolve được template smem size.

Cần thêm metric `launch__shared_memory_per_block_dynamic` để xác nhận. Từ ptxas trước: smem_static = 11.584 KB/block → 13 blocks/SM → theor. occ = 20.31% (nhất quán với kết quả trên).

---

## nsys Profiling — Timeline & Kernel Instances

**Ngày**: 2026-05-19  
**Command**:
```bash
nsys profile --output /tmp/nsys_boot_bb1000 --trace cuda,nvtx,osrt --stats true \
    ./mpboot-avx -s ../data_debug/tree1.phy \
    -use_gpu -seed 1 -sprdist 6 -gpu_device 1 -gpu_worker 200 \
    -bb 1000 -gpu_pool_size 20
```

**Report file**: `/tmp/nsys_boot_bb1000.nsys-rep`  
**Wall clock**: 20.5 s | `runBasicMpbootGpu`: 19.96 s | BEST SCORE: 6662

### Kernel instances (toàn bộ run)

| Kernel | Instances | Total GPU time | % GPU | Avg/call | Min | Max |
|--------|-----------|---------------|-------|---------|-----|-----|
| K2 `buildPhase3Kernel<4,800>` | **10** | 12.19 s | **89.1%** | 1.22 s | 1.11 s | 1.48 s |
| K1 `buildParsimonyTreesKernel<4,800>` | 1 | 1.48 s | 10.8% | 1.48 s | — | — |
| K3 `REPSKernel` | **581** | ~3 ms | **~0%** | 5.18 µs | 5.02 µs | 8.99 µs |

### CUDA API summary

| API | Calls | Total time | Avg |
|-----|-------|-----------|-----|
| `cudaStreamSynchronize` | 41 | 12.19 s | 297 ms |
| `cudaMemcpy` | 2214 | 33.7 ms | 15.2 µs |
| `cudaMemcpyAsync` | 516 | 8.2 ms | 15.8 µs |
| `cudaLaunchKernel` | 592 | 6.2 ms | 10.5 µs |

### Phân tích nsys

**K2 là bottleneck duy nhất (89% GPU time)**:
- 10 outer iterations × avg 1.22s — variation lớn (1.11–1.48s/call) → số SPR steps mỗi round không cố định
- Grid nhỏ (200 blocks / 1404 capacity) → mỗi round kết thúc nhanh nhưng GPU không saturated

**REPSKernel hoàn toàn không phải bottleneck**:
- 581 instances = 581 unique candidate trees được scored qua B=1000 replicates
- 581 / 10 rounds ≈ **58 cây mới/round** được thêm vào `treels`
- Tổng REPS time ~3ms / 13.7s GPU = **0.02%** — bỏ qua được

**CPU overhead ~6.8s** (wall 20.5s − GPU active 13.7s):
- Treels management, `boot_trees_parsimony` update, topology download per round

**`cudaStreamSynchronize` 41 calls**: 10 K2 + 1 K1 + vài sync khác (alloc/free/upload) = khớp.

---

## nsys Profiling — N=504 (`dna_M9915_504_2757.phy`)

**Ngày**: 2026-05-19  
**Command**:
```bash
nsys profile --output /tmp/nsys_M9915_504 --trace cuda,nvtx,osrt --stats true \
    ./mpboot-avx -s ../data_treebase/dna_M9915_504_2757.phy \
    -use_gpu -seed 1 -sprdist 6 -gpu_device 1 -gpu_worker 200 \
    -bb 1000 -gpu_pool_size 20
```

**Report file**: `/tmp/nsys_M9915_504.nsys-rep`  
**Wall clock**: 50.8 s | `runBasicMpbootGpu`: 49.58 s | BEST SCORE: 137924

### Kernel instances (toàn bộ run)

| Kernel | Instances | Total GPU time | % GPU | Avg/call | Min | Max |
|--------|-----------|---------------|-------|---------|-----|-----|
| K2 `buildPhase3Kernel<4,800>` | **8** | 29.61 s | **86.3%** | 3.70 s | 3.07 s | 4.23 s |
| K1 `buildParsimonyTreesKernel<4,800>` | 1 | 4.68 s | 13.6% | 4.68 s | — | — |
| K3 `REPSKernel` | **977** | ~6.7 ms | **~0%** | 6.91 µs | 6.69 µs | 12.9 µs |

### So sánh N=295 vs N=504

| Metric | N=295 | N=504 | Ratio |
|--------|-------|-------|-------|
| Wall clock | 20.5 s | 50.8 s | 2.5× |
| K2 outer iterations | 10 | **8** | — |
| K2 avg/call | 1.22 s | **3.70 s** | **3.0×** |
| K1 duration | 1.48 s | 4.68 s | 3.2× |
| REPS instances | 581 | **977** | 1.7× |
| REPS avg/call | 5.18 µs | 6.91 µs | 1.3× |
| GPU active | ~13.7 s | ~34.3 s | 2.5× |
| CPU overhead | **~6.8 s** | **~15.6 s** | **2.3×** |

### Phân tích

**K2 avg/call tăng 3× khi N: 295→504**:
- Work per block tăng tuyến tính (nhiều inner nodes hơn, SPR traverse dài hơn)
- `BuildSharedT<NTAXA=800>` đủ chứa cả 2 dataset → smem footprint không đổi, chỉ số iteration tăng

**CPU overhead tăng 2.3× (6.8s → 15.6s)**:
- 977 REPS calls (vs 581) → nhiều `saveCurrentTree()` + `boot_trees_parsimony` update hơn
- Newick string dài hơn với N=504 → string comparison trong `treels` nặng hơn

**K2 chỉ 8 outer iterations** (vs 10 ở N=295):
- Stopping criterion trigger sớm hơn — K=200 workers không đủ diverse với N lớn

**REPS avg tăng nhẹ (5.18→6.91 µs)**: do P (số patterns) lớn hơn với alignment dài hơn (2757 sites vs 1836 sites).

---

## TODO / Next steps

- [ ] Thêm `launch__shared_memory_per_block_dynamic` để confirm smem size
- [ ] Đo K2 với numpars=1000, gpu_worker=1000 để so sánh achieved occ khi grid lớn
- [ ] Xác nhận K2 registers=115 vs 128 trước đây (ptxas -v hoặc `--set full`)
