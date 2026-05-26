---
name: nsys-profiling
description: Hướng dẫn profile GPU kernels bằng nsys và ncu trên server này, bao gồm giới hạn permissions và cách đọc kết quả.
---

## Quyền hạn trên server này

| Tool | Trạng thái | Ghi chú |
|------|-----------|---------|
| `nsys` (CUDA timeline) | ✅ Chạy được | user-level |
| `nsys --gpu-metrics-devices` | ❌ Bị chặn | cần perf counters |
| `ncu` (bất kỳ option nào) | ❌ Bị chặn | `ERR_NVGPUCTRPERM` |

Để bật ncu/gpu-metrics: admin cần chạy:
```bash
sudo sh -c 'echo 0 > /proc/driver/nvidia/params/RmProfilingAdminOnly'
```

---

## Kernel names trong mpboot-gpu

| Kernel | Tên CUDA đầy đủ |
|--------|----------------|
| K1 (build + SPR) | `mpbootgpu::buildParsimonyTreesKernel<STATES, NTAXA>` |
| K2 (hill-climbing) | `mpbootgpu::buildPhase3Kernel<STATES, NTAXA>` |

---

## Lệnh nsys chuẩn (user-level, tối đa detail)

Chạy từ `build/`:

```bash
nsys profile \
    --output=../output/profile/nsys_<tag> \
    --trace=cuda,osrt,nvtx \
    --backtrace=dwarf \
    --cudabacktrace=all:0 \
    --cuda-memory-usage=true \
    --sample=process-tree \
    --force-overwrite=true \
    ./mpboot-avx \
        -s <dataset.phy> \
        -use_gpu -seed 1 \
        -numpars <N> -sprdist <D> -gpu_stop <S> -gpu_top_pct 0.1 \
        -gpu_device <DEV> \
    > ../output/profile/nsys_<tag>_run.log 2>&1
```

Xem kết quả:
```bash
nsys stats ../output/profile/nsys_<tag>.nsys-rep
```

---

## Kết quả đã biết (baseline: N=295, sprdist=5, gpu_stop=6, top_pct=10%)

### K=199 (numpars=200)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 1.47s | 10.1% | 7.4ms |
| K2 (~20 trees) | 13.13s | 89.9% | ~656ms |

### K=399 (numpars=400)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 1.97s | 12.2% | 4.9ms |
| K2 (~40 trees) | 14.16s | 87.8% | ~354ms |

### K=499 (numpars=500)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 2.29s | 13.9% | 4.6ms |
| K2 (~50 trees) | 14.24s | 86.1% | ~285ms |

### K=999 (numpars=1000)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 3.25s | 19.4% | 3.3ms |
| K2 (~100 trees) | 13.54s | 80.6% | ~135ms |

### K=1999 (numpars=2000)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 6.13s | 24.6% | 3.07ms |
| K2 (~200 trees) | 18.83s | 75.4% | ~94ms |

### K=2999 (numpars=3000)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 8.83s | 28.2% | 2.94ms |
| K2 (~300 trees) | 22.47s | 71.8% | ~75ms |

### K=3456 (numpars=3457) ← điểm cân bằng cục bộ
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 9.77s | 31.3% | 2.83ms |
| K2 (~346 trees) | 21.48s | 68.7% | ~62ms |

### K=4999 (numpars=5000)
| Kernel | Time | % GPU | ms/tree |
|--------|------|-------|---------|
| K1 | 13.59s | 34.1% | 2.72ms |
| K2 (~500 trees) | 26.31s | 65.9% | ~53ms |

**Pattern và phân tích**:
- K2 chiếm ~65–90% GPU time; % giảm dần khi K tăng.
- K1 ms/tree giảm dần nhưng **chưa bão hòa** tại K=5000 — tốc độ cải thiện ~4% mỗi lần tăng 50% K. Saturation thực tế ước tính K~10000+.
- **K=3456 là điểm cân bằng cục bộ tối ưu**: total time (31.2s) ≈ K=2999 (31.3s) nhưng có 15% trees nhiều hơn. K2 tại K=3456 thực ra GIẢM (21.48s vs 22.47s) vì top-10% trees chất lượng cao hơn → K2 converge nhanh hơn.
- Beyond K=3456: total time tăng mạnh (39.9s tại K=4999) — lợi ích K1 ms/tree không bù được chi phí K2.
- **Kết luận**: `numpars=3457` là điểm sweet spot giữa saturation và balance cho N=295 trên A100.

---

## Kết quả N=767 (dna_M7024_767_5814.phy, sprdist=4, gpu_stop=6, top_pct=10%)

| K (numpars) | K1 time | K1 % | K1 ms/tree | K2 time | K2 % | K2 ms/tree | Total |
|------------|---------|------|-----------|---------|------|-----------|-------|
| 199 (200)  | 10.74s  | 13.9% | 54.0ms  | 66.35s  | 86.1% | ~3317ms  | 77.1s |
| 999 (1000) | 21.95s  | 18.9% | 22.0ms  | 93.99s  | 81.1% | ~940ms   | 115.9s |
| 1999 (2000)| 40.02s  | 22.8% | 20.0ms  | 135.18s | 77.2% | ~676ms   | 175.2s |
| 3456 (3457)| 63.00s  | 29.0% | 18.2ms  | 154.06s | 71.0% | ~446ms   | 217.1s |
| 4999 (5000)| 86.32s  | —    | 17.3ms  | —(OOM nsys) | —  | —       | — |

**So sánh N=295 vs N=767** (cùng K=3456):
- K1 ms/tree: 2.83ms vs 18.2ms → **6.4× chậm hơn** (N tăng 2.6×, sites tăng ~4.9×)
- K2 ms/tree: 62ms vs 446ms → **7.2× chậm hơn**
- Cả hai kernel scale theo O(N × sites), không chỉ O(N)

**Pattern N=767**:
- Không có local minimum trong total time (khác N=295) — total time tăng đều theo K.
- K1 ms/tree bão hòa chậm hơn: 54→22→20→18ms (improvement rate ~9%/tăng gấp đôi K).
- Không có "sweet spot" rõ ràng như N=295. Lựa chọn K theo time budget:
  - K=199 (~77s): baseline nhanh
  - K=999 (~116s): hiệu quả nhất (trees/second tăng mạnh từ 199)
  - K=1999 (~175s): diminishing returns bắt đầu
  - K=3456 (~217s): K1 ms/tree tốt hơn nhưng K2 dominate hoàn toàn

**Kết luận**: Với N lớn (N>500), K=999–1999 là vùng sweet spot thực tế — K1 ms/tree đã giảm ~60% từ K=199 nhưng total time chưa tăng quá nhiều. Tăng K thêm (K>2000) cho lợi ích K1 <10% nhưng tăng total time 20-50%.

---

## Cách đọc kết quả nsys stats

### CUDA GPU Kernel Summary — quan trọng nhất
```
Time (%)  Total Time (ns)  Instances  Avg (ns)  Name
  87.8    14,157,532,369       1      14.16s    buildPhase3Kernel     ← K2
  12.2     1,972,494,597       1       1.97s    buildParsimonyTreesKernel ← K1
```

### CUDA API Summary
- `cudaStreamSynchronize`: thời gian CPU block chờ GPU — phải ≈ K2 duration
- `cudaLaunchKernel`: 2 calls (K1 + K2), thời gian CPU-side overhead (~ms)
- `cudaMemcpyAsync`: upload/download topologies và parsVect

### OS Runtime Summary (osrt_sum)
- `pthread_join`: CPU block chờ monitor thread (hybrid_cb2 stop_search) — phải ≈ K2 duration
- `sem_wait`/`poll`: background threads, thường chiếm nhiều wall-clock nhưng không phải bottleneck

### CUDA GPU MemOps Summary
- H→D: upload parsVect (tips, read-only) + topologies
- D→H: download best topologies sau K2
- Memcpy thường < 0.1% tổng time → **không phải bottleneck**

---

## Lý thuyết SM Occupancy (A100, 1 warp/block)

A100: 108 SMs, max 64 warps/SM, shared memory 164 KB/SM.
Kernel dùng 1 warp/block (32 threads), shared ≈ 11.4 KB/block.

| K (blocks) | Blocks/SM | Warps/SM | Occupancy |
|-----------|-----------|----------|-----------|
| 199 | 1–2 | 1–2 | ~3% |
| 399 | 3–4 | 3–4 | ~6% |
| 999 | 9–10 | 9–10 | ~15% |
| 1999 | 18–19 | 18–19 | ~28% |
| 2999 | 27–28 | 27–28 | ~43% |
| 3456 (32×108) | 32 | 32 | **50%** (lý thuyết max với 1 warp/block) |
| 4999 | >32 → cap | 32 (max) | **50%** (không tăng thêm) |

**Lưu ý thực tế**: Dù lý thuyết max là K=3456 (32 blocks/SM), K1 ms/tree vẫn tiếp tục giảm ở K=4999.
Nguyên nhân có thể là shared memory (11.4 KB/block × N blocks/SM ≤ 164 KB → max ~14 blocks/SM thực tế),
hoặc warp scheduling overhead. Saturation thực sự ước tính tại K~10000+.

Với numpars=400 (K=399): occupancy ~6% — lý do ms/tree còn cao.
**K=3456 (numpars=3457): sweet spot** — total time cực tiểu cục bộ, K2 converge nhanh hơn do trees chất lượng cao hơn.

---

## Profile output directory

Lưu tất cả vào `output/profile/`:
- `nsys_<tag>.nsys-rep` — binary report (mở bằng Nsight Systems GUI)
- `nsys_<tag>.sqlite` — database (tự động tạo khi chạy `nsys stats`)
- `nsys_<tag>_run.log` — stdout/stderr của mpboot khi profile
