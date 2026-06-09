# GPU Profiling Analysis (ncu)

**Dataset**: `data_debug/tree1.phy` — N=295 taxa, 1836 cols, 1400 patterns  
**Config**: numpars=200, sprdist=4, gpu_stop=6, gpu_top_pct=0.1, pool_size=30, device=6 (A100)  
**Tool**: `ncu --set basic -o /tmp/ncu_tree1`  
**Date**: 2026-05-17

---

## Kernels profiled

| Kernel | Duration (ncu, 1 pass) | Actual runtime |
|--------|------------------------|----------------|
| `buildParsimonyTreesKernel` (K1) | 1.38s | 15.56s |
| `buildPhase3Kernel` (K2/Phase3) | 12.37s | 124.37s |

> ncu replays kernel 10× để thu thập metrics → `duration × 10 ≈ actual runtime`. Actual lấy từ log GPU.

---

## Giải thích các section metrics

### GPU Speed Of Light Throughput

"Speed of Light" = hiệu suất tối đa lý thuyết. Các % = bạn đang dùng được bao nhiêu % công suất tối đa.

| Metric | Ý nghĩa |
|--------|---------|
| DRAM/SM Frequency | Tốc độ xung clock phần cứng — cố định, không phải hiệu suất |
| Elapsed Cycles | Tổng clock cycles kernel chạy |
| **Compute (SM) Throughput** | % sức tính toán CUDA cores đang dùng — chỉ số quan trọng nhất |
| Memory Throughput | % băng thông bộ nhớ tổng hợp (L1+L2+DRAM) |
| DRAM Throughput | Riêng băng thông RAM ngoài (HBM) |
| L1/TEX Cache Throughput | Băng thông L1 cache trong mỗi SM |
| L2 Cache Throughput | Băng thông L2 cache (chia sẻ giữa SMs) |
| SM Active Cycles | Số cycles trung bình mỗi SM thực sự có warp chạy |

**Kết quả**:
- K1: Compute 4.19%, Memory 5.66%, DRAM 0.56%
- K2: Compute **0.65%**, Memory 0.83%, DRAM **0.02%**

→ Kernel **không bị giới hạn bởi memory hay compute** — idle hầu hết thời gian.

---

### Launch Statistics

Thông tin về cách kernel được launch.

| Metric | K1/K2 | Ý nghĩa |
|--------|-------|---------|
| Block Size | 32 | Mỗi block = 1 warp = 1 cây (thiết kế cơ bản) |
| Grid Size | 200 | Tổng blocks = K = numpars |
| **Registers Per Thread** | 155 (K1) / 129 (K2) | Số register mỗi thread dùng — rất cao, đây là bottleneck |
| Static Shared Memory Per Block | 11.63 KB | Struct `sh` (SharedMemT) — cố định lúc compile |
| # SMs | 108 | A100 có 108 SM |
| **Waves Per SM** | 0.15 | Số "đợt" block chạy qua mỗi SM — < 1 nghĩa là GPU chưa được lấp đầy |

**Waves**: GPU chia blocks thành "sóng". 200 blocks / 108 SMs = 1.85 blocks/SM, nhưng register limit chỉ cho 12 blocks/SM tối đa → waves = 200/(108×12) = **0.15**. Chỉ 15% GPU được dùng.

---

### Occupancy

Occupancy = % warp slots trên GPU đang có warp chạy.

A100: 108 SMs × 64 warp slots = 6,912 tổng. Ta có 200 warps → tối đa 200/6912 ≈ 2.9%.

| Metric | K1 | K2 | Ý nghĩa |
|--------|----|----|---------|
| Block Limit SM | 32 | 32 | Phần cứng: tối đa 32 blocks/SM |
| **Block Limit Registers** | 12 | 12 | Do register: 155 regs × 32 threads = 4,960 regs/block; SM có 65,536 regs → 13 blocks max → 12 |
| Block Limit Shared Mem | 13 | 13 | Do shared mem: 11.63KB × 13 ≈ 151KB < 167.94KB config |
| Block Limit Warps | 64 | 64 | Phần cứng thuần |
| Theoretical Active Warps/SM | 12 | 12 | min(12, 13, 32, 64) = 12 — bottleneck là register |
| **Theoretical Occupancy** | 18.75% | 18.75% | 12/64 = 18.75% — ceiling tuyệt đối do register pressure |
| **Achieved Occupancy** | 2.76% | 2.15% | Thực tế đạt được — thấp hơn ceiling vì chỉ có 200 warps |
| Achieved Active Warps/SM | 1.77 | 1.38 | Mỗi SM trung bình chỉ có ~1.5 warp chạy (trong khi chứa được 12) |

**Tại sao Achieved < Theoretical?** 200 warps / 108 SMs = 1.85 warps/SM → không đủ block để schedule lên GPU.

---

### GPU and Memory Workload Distribution

Phân tích phân phối công việc giữa các SM.

**K1 — imbalance nhẹ**:
- Max SM active +22.75% / Min -27.70% so với trung bình
- Nguyên nhân: phân phối block không đồng đều (các cây có độ phức tạp khác nhau)

**K2 — imbalance rất nặng**:
- Max SM active **+84.88%** / Min **-82.08%** so với trung bình
- Nguyên nhân: Phase 3 có early stop (`stopNoImprove=6`) → warp nào hội tụ nhanh dừng sớm, SM chứa chúng idle trong khi SM khác còn chạy hết `numSearchIter=29`

---

## Tổng kết — 3 bottleneck gốc rễ

### 1. K=200 quá nhỏ cho 108 SMs (vấn đề chính)
```
200 blocks / 108 SMs = 1.85 blocks/SM → 0.15 waves → ~97% SM idle
```
Fix: tăng numpars. Cần ít nhất K ≈ 108×12 = **1,296 warps** để đạt theoretical max.  
Với K=200: chỉ dùng ~2% sức mạnh GPU.

### 2. Register pressure cao (155 regs K1, 129 regs K2)
```
Ceiling occupancy = 18.75% — không thể vượt dù có vô hạn block
```
Fix: rất khó — cần refactor kernel để giảm live variables, hoặc dùng `__launch_bounds__` để force compiler tối ưu register.

### 3. Load imbalance trong Phase 3
```
SM max active = +84.88% vs SM min active = -82.08% → phần lớn SM idle sớm
```
Fix: khó về mặt thuật toán — đây là bản chất của early stop.

---

---

## So sánh K=200 vs K=1000 (tree1.phy, N=295)

**Config thêm**: numpars=1000, sprdist=4, gpu_stop=6, gpu_top_pct=0.1, pool_size=30, device=6  
**Report**: `/tmp/ncu_tree1_n1000.ncu-rep`

### Throughput & Occupancy

| Metric | K=200 K1 | K=1000 K1 | K=200 K2 | K=1000 K2 |
|--------|----------|-----------|----------|-----------|
| Compute SM Throughput | 4.19% | **9.64%** | 0.65% | **2.44%** |
| Memory Throughput | 5.66% | **13.08%** | 0.83% | **3.10%** |
| Achieved Occupancy | 2.76% | **13.07%** | 2.15% | **8.90%** |
| Achieved Warps/SM | 1.77 | **8.37** | 1.38 | **5.70** |
| Waves Per SM | 0.15 | **0.77** | 0.15 | **0.77** |
| SM imbalance (max) | +22.75% | **+10.79%** | +84.88% | **+66.31%** |

### Actual Runtime (ms, từ log GPU)

| Kernel | K=200 | K=1000 | Ghi chú |
|--------|-------|--------|---------|
| K1 | 15.56s | 31.60s | +2.0x — thêm cây thêm việc |
| K2 | 124.37s | **65.79s** | **−1.9x** — K2 nhanh hơn dù 5× nhiều cây hơn |
| **Total** | 140.38s | **98.41s** | **−1.43x** |
| **ms/tree** | 701.9 ms | **98.4 ms** | **7.1× hiệu quả hơn** |

**Tại sao K2 nhanh hơn tuyệt đối?** K=1000 warp đủ để GPU scheduler ẩn memory latency (khi 1 warp chờ, SM switch sang warp khác). K=200 warp quá ít → SM phải idle chờ.

**Kết luận thực tế**: Cùng ngân sách thời gian 140s → K=200 ra 200 cây, K=1000 ra 1000 cây trong 98s. Tăng numpars vừa cải thiện chất lượng vừa tăng GPU utilization.

---

## Implications cho benchmark

- DRAM throughput 0.02% trong K2 (K=200): parsVect fit hoàn toàn trong L1/L2, không đọc DRAM
- Compute thấp không phải vì branch divergence đơn thuần — SM đơn giản là idle vì không đủ warps
- Cần K ≥ 1,296 để đạt theoretical ceiling. K=1000 đạt 77% (0.77 waves); K1 occupancy 69.7% ceiling, K2 chỉ 47.5% do early-stop imbalance
- Register ceiling 18.75% là giới hạn cứng — dù K lớn đến đâu cũng không vượt được nếu không giảm register

---

## Lệnh chạy ncu

```bash
# Profile cả 2 kernels, lưu report
cd /raid/home/loinguyen/workspace/mpboot-gpu/build
ncu --set basic -o /tmp/ncu_out --target-processes all ./mpboot-avx \
    -s ../data_debug/tree1.phy -use_gpu -seed 1 -numpars 200 \
    -sprdist 4 -gpu_stop 6 -gpu_top_pct 0.1 -gpu_pool_size 30 -gpu_device 6

# Đọc report (in ra terminal)
ncu --import /tmp/ncu_out.ncu-rep

# Với K=1000 (để so sánh utilization)
ncu --set basic -o /tmp/ncu_out_n1000 --target-processes all ./mpboot-avx \
    -s ../data_debug/tree1.phy -use_gpu -seed 1 -numpars 1000 \
    -sprdist 4 -gpu_stop 6 -gpu_top_pct 0.1 -gpu_pool_size 30 -gpu_device 6
```

> `--set basic` đủ để lấy throughput, occupancy, launch stats. Không cần `--set full` (bị block bởi hardware counter permissions trừ khi có admin rights).
