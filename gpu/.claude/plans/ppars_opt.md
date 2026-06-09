# Kế hoạch tối ưu ppars pipeline

**Ngày**: 2026-06-01  
**Dataset benchmark**: M10467 (N=202, nptn=4074), 8 rounds, seed=1, gpu_worker=200

---

## Trạng thái hiện tại (sau session này)

| Tối ưu | Ngày | Kết quả M10467 |
|--------|------|----------------|
| A1-A4: Fix Fitch kernel, PASS1 fast copy, max_reps=20000 | 2026-06-01 | baseline |
| B: K_ppars=1000 (dynamic alloc từ free VRAM) | 2026-06-01 | ppars 8239→6620ms R1 |
| C: Pin h_treels_* với cudaMallocHost | 2026-06-01 | D2H R1: 2493→388ms |

**Wall-clock hiện tại**: 68.3s (M10467, bench_pinned)

---

## Phân tích D2H

### Các mảng được download mỗi round (step [7])

| Mảng | Size (n=19k, M10467) | Cần? |
|------|---------------------|------|
| `h_treels_bvf` (back_vf) | n × 12.8 KB = **246 MB** | YES – newickFromBackVf() |
| `h_treels_ptn_pars` | n × nptn_padded × 2B = **157 MB** | YES – REPS batch fast path |
| `h_treels_scores` | n × 4B = 77 KB | MAYBE (dedup filter) |
| `h_treels_hashes` | n × 4B = 77 KB | MAYBE (bootstrap dedup) |

### Root cause D2H chậm: pageable memory page faults

```
std::vector<int> h_treels_bvf;       // ← lazy alloc: 2.56 GB chưa commit
h_treels_bvf.resize(max_treels × 3200);  // chỉ reserve địa chỉ ảo
```

Lần đầu cudaMemcpy ghi vào → OS phải fault 60k+ pages → ~200 MB/s.

```
R1: 1980ms (200 MB/s cold)  →  R7: 247ms (6 GB/s hot)
```

### Fix đã thực hiện: `cudaMallocHost` (pinned memory)

```cpp
// gpu_init_trees.cu lines 640-667
CUDA_CHECK(cudaMallocHost(&h_treels_bvf,    max_treels × kMaxVFaces × 4));
CUDA_CHECK(cudaMallocHost(&h_treels_scores,  max_treels × 4));
CUDA_CHECK(cudaMallocHost(&h_treels_hashes,  max_treels × 4));
CUDA_CHECK(cudaMallocHost(&h_treels_ptn_pars, max_treels × nptn_padded × 2));
```

Cleanup ở lines 1387-1391:
```cpp
if (h_treels_ptn_pars) cudaFreeHost(h_treels_ptn_pars);
if (h_treels_bvf)      cudaFreeHost(h_treels_bvf);
if (h_treels_scores)   cudaFreeHost(h_treels_scores);
if (h_treels_hashes)   cudaFreeHost(h_treels_hashes);
```

Kết quả: D2H R1: 2493ms → 388ms (**6.4×**), ổn định từ round 1.

---

## Tối ưu tiếp theo: K2_N ∥ ppars_{N-1}

### Pipeline hiện tại (per round N)

```
[1] reset treels + launch K2_N (stream)  ← non-blocking
[2] PASS1_{N-1} concurrent với K2_N     ← đọc h_treels từ D2H_{N-1}
[3] sync K2_N
[4] small D2H: n_treels count + pool scores
[5] launch ppars_N (stream, async)
[6] REPS_{N-1} + PASS2_{N-1} concurrent với ppars_N (stream_reps + CPU)
[7] sync ppars_N → BIG D2H_N (back_vf + ptn_pars)
[8] stats/log → loop
```

Critical path mỗi round:
```
K2(1240ms) + ppars(6620ms) + D2H(388ms) = 8248ms
```

PASS1 (901ms) ẩn trong K2 (1240ms > 901ms → fully hidden).

### Ý tưởng: chạy K2_N song song với ppars_{N-1}

**ppars_{N-1}** không cần kết quả K2_N. **K2_N** không cần kết quả ppars_{N-1}.

→ Về logic: có thể chạy song song.

### Conflict chính: d_treelsBackVf

```
ppars_{N-1}  READS:  d_treelsBackVf[0 .. n_{N-1} - 1]
K2_N         WRITES: d_treelsBackVf[0 .. n_N - 1]  (sau resetTreelsRound reset d_treelsFilled=0)
```

K2_N sẽ overwrite dữ liệu ppars_{N-1} đang đọc → corrupt.

Tương tự cho `d_treelsScores` và `d_treelsHashes`.

### Fix: D2D scratch buffer

Sau sync K2_{N-1}, trước khi launch ppars_{N-1}:

```cpp
// Async D2D copy trên ppars_stream (trước ppars launch)
cudaMemcpyAsync(d_bvf_scratch, d_treelsBackVf,
    n_{N-1} × kMaxVFaces × 4, cudaMemcpyDeviceToDevice, ppars_stream);

// ppars_{N-1} đọc từ scratch (không còn conflict với K2_N)
gpuComputeTreelsPatternPars(mem, n_{N-1}, d_bvf_scratch, ..., ppars_stream);

// Ngay sau: reset treels → launch K2_N trên original stream → CONCURRENT!
resetTreelsRound(mem, cutoff, stream);
gpuStepwiseBuildTrees(..., stream);  // K2_N chạy song song với ppars_{N-1}
```

**Tốc độ D2D copy**: 244 MB (19k trees × 12.8 KB) tại HBM2 bandwidth ~2 TB/s = **~0.12ms** (negligible).

**GPU memory extra**: `d_bvf_scratch` = max_treels × kMaxVFaces × 4 = 200,000 × 12,800 B = **2.56 GB**.
Với A100 (80 GB), hiện dùng ~7.74 GB → còn ~72 GB free → fit dễ dàng.

### Pipeline mới với overlap

```
ppars_stream: [D2D_0 | ppars_0(6620ms)] → [D2D_1 | ppars_1(6547ms)] → ...
stream:       [K2_0] → (K2_0 done) → [K2_1(1240ms)] → [K2_2] → ...
                                        ↑ concurrent với ppars_0
```

K2_{i+1} launch ngay sau K2_i done (~1240ms interval).
ppars_i chạy ~6620ms (ẩn toàn bộ K2 thời gian).

### Ước tính lợi ích

**Tổng ppars time** (M10467, 8 rounds từ bench_pinned log):
```
6620+6547+4708+4622+4128+3272+2611+1847 = 34355ms ≈ 34.4s
```

**Tổng K2 time** (8 rounds × 1240ms):
```
8 × 1240 = 9920ms ≈ 9.9s  ← fully hidden trong ppars với overlap
```

**Wall time hiện tại**: 68.3s

**Wall time ước tính sau overlap**:
- ppars_stream dominates: ~34.4s
- D2H sau mỗi ppars (nếu sync blocking): +8 × 300ms = +2.4s
- PASS1+REPS+PASS2 per round: ~1.1s, ẩn trong ppars_{i+1} (6s >> 1.1s)
- Overhead (copy, sync, etc.): ~1s
- **Ước tính**: ~34.4 + 2.4 + 1 ≈ **38s** (→ ~44% speedup từ 68.3s)

**Nếu D2H async** (cudaMemcpyAsync trên stream riêng sau pinned fix):
- D2H (388ms) chạy song song với ppars_{i+1} (6547ms) → D2H fully hidden
- Wall time → ~34.4 + 1 ≈ **35-36s** (→ ~48% speedup)

### Độ phức tạp implementation

| Thay đổi | File | Ước tính |
|----------|------|---------|
| Thêm `d_bvf_scratch` vào GpuParsimonyMem | pars_tree.cuh, pars_tree.cu | ~20 dòng |
| Tách ppars sang `stream_ppars` mới | gpu_init_trees.cu | ~30 dòng |
| D2D copy trước ppars launch | gpu_init_trees.cu | ~15 dòng |
| K2 launch mà không đợi ppars trước | gpu_init_trees.cu | refactor pipeline ~100 dòng |
| D2H async (optional, thêm lợi ích) | gpu_init_trees.cu | ~50 dòng |
| Sync events mới (ppars done, D2H done) | gpu_init_trees.cu | ~20 dòng |
| Cũng cần scratch cho scores/hashes | pars_tree.cu | ~10 dòng |

**Tổng**: ~250 dòng, rủi ro cao (race condition, sync sai thứ tự).

### Lưu ý quan trọng

1. **PASS1 bị dời**: Hiện PASS1_{N-1} chạy trong K2_N (concurrent). Với overlap, PASS1 phải đợi D2H_{N-1} (sau ppars_{N-1}). Nhưng PASS1 (901ms) < ppars_{i+1} (>4s), nên PASS1 vẫn ẩn.

2. **REPS timing**: R1 REPS=6659ms (lớn, do build model lần đầu), R2+ REPS=125ms. Với overlap, R1 vẫn là round đặc biệt.

3. **Bottleneck thực sự**: Sau khi overlap, bottleneck là tổng ppars time (~34s). Để cải thiện thêm cần tăng tốc ppars kernel (memory bandwidth limited - 200MB→3.4GB >> A100 L2 40MB).

---

## Tóm tắt tất cả tối ưu đã làm và sắp làm

| # | Tối ưu | Status | Speedup |
|---|--------|--------|---------|
| A1 | Fix Fitch kernel pars_ptn indexing | ✅ Done | Correctness fix |
| A2 | newickFromBackVf use_int_ids | ✅ Done | Correctness fix |
| A3 | PASS1 fast copy (GPU ppars → h_batch_pars) | ✅ Done | ~4ms/tree CPU saved |
| A4 | max_reps_per_round 2000→20000 | ✅ Done | Nhiều tree hơn/round |
| B  | K_ppars=1000 (dynamic alloc free VRAM) | ✅ Done | ppars ~1.24× |
| C  | Pin h_treels_* (cudaMallocHost) | ✅ Done | D2H R1: 6.4× faster |
| D  | K2_N ∥ ppars_{N-1} (D2D scratch buffer) | 🔲 Planned | ~44-48% total |
