# Kernel Review & Optimization Plan (2026-05-12)

## Trạng thái hiện tại

| Optimization | Status | Kết quả |
|---|---|---|
| testInsert pre-refresh (loại bỏ Fitch dư) | ✅ Done | −24% kernel time |
| **Opt-D: GpuTopology struct shrink** | ✅ Done | −8.7% ms/tree |

---

## Opt-D: GpuTopology Struct Shrink ✅ DONE (2026-05-12)

### Phát hiện

`vfToNum`, `vfNextFace`, `vfNnxtFace` đều là **pure arithmetic** — kernel KHÔNG BAO GIỜ đọc
`topo->number[]`, `topo->next_vf[]`, `topo->nnxt_vf[]`. Ba mảng này chỉ phục vụ CPU-side
`cpuToGpuTopology` / `gpuTopoToCpu`.

⇒ **Xoá luôn 3 mảng** khỏi struct:
- `number[]` 12.8 KB — CPU dùng `p->number` trực tiếp (stable sau init, không cần lưu GPU)
- `next_vf[]` 12.8 KB — thay bằng `vfNextFace(vf, N)` arithmetic trong `gpuTopoToCpu`
- `nnxt_vf[]` 12.8 KB — không được đọc trong `gpuTopoToCpu`

Thêm `__host__` vào `vfNextFace`/`vfNnxtFace` để gọi được từ CPU code.

**Struct: 83,236 → 44,836 bytes (−37.5 KB)**

### Layout sau Opt-D

```
back_vf[0]       offset=0        HOT  12.8 KB
xpars[0]         offset=12.8 KB  HOT  12.8 KB  (cũ: cách back_vf 64 KB!)
scalars          offset=25.6 KB  36 bytes
nodep[0]         offset=25.6 KB  MEDIUM 6.4 KB  (sequential, outer loop)
best_back_vf[0]  offset=32.0 KB  COLD 12.8 KB  (chỉ khi update + download)
```

Toàn bộ "live" topology data (back_vf + xpars + nodep) = ~32 KB → fit tốt hơn vào L2 cache per-SM.

### Benchmark kết quả (seed=1, numpars=200, gpu_hc_iter=10, sprdist=3)

| Dataset (N) | Before | After | Δ |
|---|---|---|---|
| prot_M6416 (N=57) | 17.65 ms/tree | 16.91 ms/tree | −4.2% |
| prot_M12104 (N=93) | 34.75 ms/tree | 32.07 ms/tree | −7.7% |
| prot_M13804 (N=78) | 45.76 ms/tree | 41.88 ms/tree | −8.5% |
| dna_M7210 (N=204) | 58.76 ms/tree | 53.49 ms/tree | −9.0% |
| prot_M11013 (N=55) | 76.40 ms/tree | 71.54 ms/tree | −6.4% |
| dna_M8012 (N=213) | 91.29 ms/tree | 80.65 ms/tree | −11.7% |
| dna_M3031 (N=276) | 105.71 ms/tree | 95.76 ms/tree | −9.4% |
| dna_M1838 (N=228) | 117.53 ms/tree | 101.16 ms/tree | −13.9% |
| dna_M8692 (N=395) | 145.66 ms/tree | 130.41 ms/tree | −10.5% |
| prot_M3114 (N=77) | 199.66 ms/tree | 188.29 ms/tree | −5.7% |
| **Trung bình** | | | **−8.7%** |

Parsimony quality: 8/10 identical, 1 better (3299→3297), 1 worse by 2 (2938→2940) — stochasticity bình thường.

### Files đã sửa

| File | Thay đổi |
|------|---------|
| `gpu/include/pars_tree.cuh` | Xoá `number[]`, `next_vf[]`, `nnxt_vf[]`; reorder `back_vf`, `xpars` lên đầu |
| `gpu/include/topo_helpers.cuh` | Thêm `__host__` vào `vfNextFace`, `vfNnxtFace` |
| `gpu/src/pars_tree.cu` | `cpuToGpuTopology`: bỏ 3 write; `gpuTopoToCpu`: thay `next_vf[vf]` bằng `vfNextFace(vf, N)` |

---

## Opt-E: Sub-Warp Parallel Evaluation (Future)

Thay vì 32 lanes làm 1 candidate, chia thành 8 nhóm × 4 lanes, mỗi nhóm làm 1 candidate.
- Lý thuyết: 8× throughput
- Thực tế: 2-4× sau overhead
- Độ khó: RẤT CAO (cần redesign BuildShared, warp reduction, DFS stack per sub-warp)
- **Dành cho future version.**

---

## Hướng tiếp theo

1. **Profile với nvprof/nsight** — xác nhận L2 cache hit rate cải thiện sau Opt-D; tìm bottleneck tiếp theo
2. **Full benchmark** — chạy `bench_gpu.sh` toàn bộ 460 datasets
3. **Opt-E** — nếu muốn tăng tốc lớn hơn nữa

---

## Verification

```bash
cd /raid/home/loinguyen/workspace/mpboot-gpu/build
make -j4

./mpboot-avx -s ../data_treebase/dna_M214_295_1836.phy \
    -use_gpu -seed 1 -numpars 100 -gpu_hc_iter 10 -sprdist 3 \
    2>&1 | grep -E "Post-Hill|TIMING|GPU.*kernel"

# Full benchmark:
bash bench_gpu.sh && python3 ../output/summarize.py
```
