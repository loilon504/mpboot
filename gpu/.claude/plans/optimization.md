# GPU Parsimony Optimizations — Tổng quan (2026-05-14)
<!-- Lưu tại: /raid/home/loinguyen/workspace/mpboot-gpu/mpboot/gpu/.claude/plans/optimization.md -->

## Đã hoàn thành

| Opt | Tên | Kết quả |
|-----|-----|---------|
| testInsert pre-refresh | Loại bỏ Fitch step thừa | −24% kernel time |
| Opt-D | GpuTopology struct shrink | −8.7% ms/tree |
| Opt-H | Early stopping Phase 3 (gpu_stop) | −32.4% ms/tree |
| ~~Opt-G~~ | ~~Selective Phase 3 atomicMin margin~~ | **Xoá — disabled by default, code phức tạp** |
| Opt-G2 (→ -gpu_top_pct) | Selective Phase 3 two-kernel exact top-X% | default=10%, giảm Phase 3 overhead |
| Opt-I | NNI strength configurable (gpu_nni_strength); rewrite on-the-fly q_vf + CPU-style bitset reset | strength=0.5 works; all EXIT=0; no regression |
| Opt-K | gpu_stop default 2 → 4 | Quality tốt hơn trên N≥295 |
| Opt-B | Subtree prune trong SPR DFS (per-edge lb check) | ~22% prune rate, avg 1.31× speedup |
| Opt-B+ | Tighter lb: thêm score_tree[tip_p] | **~30% prune rate, avg 1.49× speedup, 10/10 faster** |
| Opt-M | Template specialization newviewParsimony<STATES> (4, 20 only) | **avg 2.16× ms/tree (np=200, 50 datasets)** |
| Opt-P L1 | BuildSharedT: union perm/stackMint + stackMaxt[64] | **+1.197× vs Opt-M (50 datasets)** |
| Opt-P L2b | BuildSharedT: int16_t arrays + reorder hot fields | **+2.0% vs L1 (reorder); int16_t net-negative alone** |
| Opt-P L3 | BuildSharedT<NTAXA> template (128/256/512/800) + GPU_NTAXA_TEMPLATE flag | Implemented; speedup when K>756 |
| Opt-Q1 | score_tree zero-init loop removal (dead code) | **−3.4% avg** (10 datasets) |
| Opt-Q2 | Lazy gpuNodeRectifierPars: skip first do-while iteration | **−13.0% avg** (10/10 datasets) |
| **Opt-R** | **Restore best_back_vf trước mỗi Phase 3 perturbation** | **avg +60% speedup vs CPU, −38% ms/tree** |
| **Reseed** | **Reseed bad GPU slots từ threshold pool (hybrid_cb Step 3.5)** | **0/37 regressions, 14/37 improved, +14.5% time (N≈200)** |
| **Dead code removal** | **Xóa timing fields, best_back_vf, numSearchIter, gpu_hc_iter** | **K1: 155→96 regs; GpuTopology −12.8 KB; API đơn giản hơn** |
| **K' < K (Opt-LessK1)** | **Build K'=max(pool,0.2K) trees trong K1; K2 vẫn K blocks** | **−60% K1 time; +17–28% total speedup; GPU wins +13–25%** |
| **Pool restart simplify** | **Lane-0 O(pool_size²) selection-sort, unsigned long long bitmask (pool_size ≤ 60); accessible=10+outer** | **K2 regs 151→128; simpler code** |
| **Opt-S: Per-slot locks** | **pool_lock→pool_slot_locks[pool_size]; thundering herd fix** | **Contention 1000→50 blocks/lock (pool=20)** |

---

## K' < K: Giảm số cây K1 (Opt-LessK1, 2026-05-17)

**Phát hiện quan trọng**: `hybrid_cb` (callback sau K1) đã reseed **toàn bộ K slots** từ pool
(Step 5: `for (k=0; k<Kc_full; k++)` với `Kc_full = mem->K`). K2 đọc `needs_recompute=1` path
— không dùng K1 topology trực tiếp. K1 trees chỉ cần đủ để populate pool (Step 3).

**Tối ưu**: Build K'=max(pool_size, K×ratio) trees trong K1 thay vì K:
- `k1_trees = max(30, (int)(K × gpu_k1_ratio))` với `gpu_k1_ratio=0.2` → K'=200 (K=999)
- K1 launch: `dim3(k1_trees)` (thay vì `dim3(K)`)
- Callback split: `Kc_scores = k1_trees` (Step 2/3), `Kc_full = K` (Step 5 không đổi)
- K2 launch: `dim3(K)` không đổi

**Win-win**: K1 ngắn hơn → CPU builds ~50% more trees trong callback window → pool tốt hơn
→ K2 cả nhanh hơn (16%) lẫn chất lượng cao hơn (GPU wins tăng 20–25%).

**Benchmark** (K=999, pool=30, k1_ratio=0.2, 115 datasets per config, A100-SXM4-80GB):

| Config (sprdist, stop) | K1 Δ | K2 Δ | Δtotal speedup | GPU wins (K'=K → K'=0.2K) |
|------------------------|------|------|----------------|--------------------------|
| d3, s10 | −59.7% | +15.6% | **+25.8%** (8.04×→10.11×) | 57→64 |
| d4, s6  | −59.6% | +16.2% | **+22.5%** (6.70×→8.21×)  | 53→64 |
| d4, s20 | −59.8% | +0.7%  | **+27.4%** (5.82×→7.41×)  | 52→65 |
| d5, s6  | −59.4% | +15.2% | **+21.1%** (5.46×→6.61×)  | 48→58 |
| d6, s6  | −58.8% | +18.5% | **+17.6%** (4.55×→5.35×)  | 45→61 |

**Files**: `mpboot/tools.h` (gpu_k1_ratio field), `mpboot/tools.cpp` (default+parse),
`gpu/include/pars_build.cuh` (k1_trees param), `gpu/src/pars_build.cu` (K1 dim3),
`gpu/src/gpu_init_trees.cu` (compute k1_trees, Kc_scores/Kc_full split).

→ Full logs: `output/hybrid_pool_{lessk1,n1000p30}d{3s10,4s6,4s20,5s6,6s6}/`

---

## Profiling Findings

### NCU 2026-05-18 — w400p20d6s3, 6 hard datasets (N=219–504), A100

**Method**: `ncu --launch-count 2 --set default` per dataset trên device 1–6
→ Reports: `/output/ncu/w400p20d6s3/*.ncu-rep`

| Kernel | Grid | Regs | Occ limit (regs) | Occ limit (smem) | Waves/SM | Theor. Occ% |
|--------|------|------|-----------------|-----------------|----------|-------------|
| K1 `buildParsimonyTreesKernel<4,800>` | 200 | **96** | 20/SM | **13/SM** | 0.14 | **20.31%** |
| K2 `buildPhase3Kernel<4,800>` | 400 | **128** | 16/SM | **13/SM** | 0.28 | **20.31%** |

**Key findings**:
- Bottleneck là **shared memory** (11.584 KB/block → 13 blocks/SM), không phải registers
- K2 reg limit = 16/SM sau pool restart simplify (regs 151→128), nhưng smem vẫn stricter
- w400: waves/SM = 0.28 → **GPU chưa saturated**; cần ≥ 1404 blocks (gpu_worker ≥ 1400) cho 1 wave
- NTAXA=800 template cố định → smem không phụ thuộc N thực → kết quả giống hệt nhau trên tất cả datasets
- Để tăng lên 14 blocks/SM (21.9% occ): cần cắt ~1.3 KB shared memory (11.584 → ≤10.25 KB)

### NCU 2026-05-17 — sau dead code removal, K=200, dna_M10434 (544 taxa)

**Method**: NCU `--set basic --launch-count 2` trên A100-SXM4-80GB

| Kernel | Regs/thread | Block Limit Reg | Block Limit Smem | Theor. Occ | Achieved Occ (K=200/600/1000) |
|--------|------------|-----------------|-----------------|-----------|-------------------------------|
| K1 `buildParsimonyTreesKernel` | **96** (↓ từ 155) | 20 | 13 | **20.31%** | 2.83% / 7.98% / 13.20% |
| K2 `buildPhase3Kernel` | **151** → **128**¹ | 12 → **16**¹ | 13 | 18.75% → **20.31%**¹ | 2.34% / 5.64% / 9.22% |

¹ Sau pool restart simplification (2026-05-18)

**Scaling với K** (K1 / K2):
- Duration: K=200→7.31s/9.63s; K=600→10.80s/12.11s; K=1000→13.53s/14.03s (sublinear — waves overlap)
- Compute throughput K=1000: K1=9.39%, K2=4.71% — K2 stall nhiều hơn (divergence trong NNI/SPR loop)

→ Full NCU reports: `/tmp/ncu_both_reg.ncu-rep`, `/tmp/ncu_K600.ncu-rep`, `/tmp/ncu_K1000.ncu-rep`

---

---

## Pool Restart Simplification (2026-05-18)

**Vấn đề**: Scan block trong pool restart dùng warp-parallel k-th min (phức tạp, scattered lane work).

**Fix**: Lane-0 chạy O(pool_size²) selection-sort với `unsigned long long used` bitmask:
- pool_size ≤ 60 → bitmask fits in 1 `unsigned long long` (64-bit); dùng `1ULL <<`
- (Ban đầu `uint32_t` chỉ hỗ trợ ≤ 32; đổi sang `unsigned long long` để hỗ trợ ≤ 60)
- Setup block (bcast[3]=accessible, bcast[5]=order_idx) giữ nguyên
- Scan block thay toàn bộ bằng lane-0 sequential sort

**accessible = 10 + outer** (thay `*pool_accessible`): window tăng tuyến tính theo iteration, không cần atomic counter sync. Tất cả warps đều thực thi cùng nhau → không cần phân tán window.

**Effect**: K2 registers **151 → 128** (theo NCU); Block Limit Reg 12→16.
Theor. Occ vẫn **20.31%** vì shared memory là bottleneck (11.584 KB/block → 13 blocks/SM < 16).

**File**: `gpu/src/pars_build.cu` — pool restart scan block.

---

## Opt-S: Per-slot Pool Spinlocks (2026-05-18)

**Vấn đề**: 1 `pool_lock` int → 1000 blocks thundering herd khi cần copy pool slot (12.8 KB).

**Fix**: `pool_slot_locks[pool_size]` — lock riêng cho mỗi slot trong pool:
```cpp
// Pool restart — lock slot đang đọc:
while (atomicCAS(&pool_slot_locks[phys_slot], 0, 1) != 0) {}
// ... copy 12.8 KB ...
atomicExch(&pool_slot_locks[phys_slot], 0);

// Pool insert — lock slot đang ghi:
while (atomicCAS(&pool_slot_locks[worst_slot], 0, 1) != 0) {}
// ... copy topology to pool ...
atomicExch(&pool_slot_locks[sh.bcast[4]], 0);
```

**Tại sao an toàn**: Lock A (restart) và Lock B (insert) dùng đúng slot ID → không deadlock.
Hai block cùng restart từ slot 3 vẫn contend 1 lock (pool_slot_locks[3]), nhưng xác suất thấp.

**Memory delta**: +pool_size×4 bytes = +80 bytes (pool_size=20).

**NCU** (worker=1000, N=295, A100, sau tất cả thay đổi):
| Metric | Giá trị |
|--------|---------|
| K2 Regs | 128 |
| Theor. Occ | 20.31% (shared-mem limited) |
| Achieved Occ | 13.29% |
| Waves/SM | 0.71 |
| Compute% | 6.30% (GPU mostly stalling) |

**Files**: `pars_tree.cuh`, `pars_tree.cu`, `pars_build.cu` (2 sites), `gpu_init_trees.cu`.

---

## Benchmark 4 configs (2026-05-18, numpars=200, pool=20, 115 datasets)

CPU baseline: `cpu_d6` (sprdist=6, numpars=200), avg 56.70s, avg_score=29022.9.

| Config | avg_time | avg_speedup | wins/115 | GPU≤CPU |
|--------|----------|------------|---------|---------|
| w400_d6_s3 | 12.13s | 2.97× | 52 | 106 |
| w1000_d6_s2 | 15.88s | 2.27× | 50.5 | 102 |
| **w400_d4_s3** | **9.68s** | **3.84×** | 48 | 98 |
| w1000_d4_s2 | 10.51s | 3.63× | 49 | 97 |

→ Full results: `output/compare_4configs.txt`; xlsx: `output/worker_*/results.xlsx`

---

## Còn lại — theo độ khó và rủi ro

### 🟢 Dễ, rủi ro thấp

#### ✅ Opt-P Layer 1: BuildShared Shrink — Tăng Occupancy (DONE 2026-05-13)

**Thay đổi**:
- `perm[kMaxTaxa+2]` union với `stackMint[kMaxTaxa]` (Phase 0-1 vs Phase 2-3, non-overlapping) → saves 3.2 KB
- `stackMaxt[kMaxTaxa=800]` → `stackMaxt[kMaxSprStack=64]` (NNI bitset ≤51 words, SPR stack ≤12) → saves 2.9 KB
- BuildShared: 28.4 KB → **22.4 KB** → blocks/SM: 5 → **7** (+40% occupancy)

**Kết quả** (50 datasets, numpars=200, sprdist=3, gpu_stop=4):
- Average speedup vs Opt-M: **+1.197x** (N≥80: 1.18–1.52x; N≤65: ~1.0x noise)
- Quality: **0/50 regression** ✅

**File**: `gpu/include/pars_tree.cuh`

---

#### ✅ Opt-P Layer 3: NTAXA Templating (DONE 2026-05-13)

**Thay đổi**: `BuildSharedT<NTAXA>` template, buckets **128/256/512/800** (bỏ 384).  
`GPU_NTAXA_TEMPLATE=OFF` (default): chỉ compile NTAXA=800 → build nhanh ~5 phút.  
`GPU_NTAXA_TEMPLATE=ON`: 4 buckets × 2 STATES × 2 kernels = **16 kernels** → build ~16 phút.

**Giới hạn thực tế**: Với numpars=200 (K=199), tất cả blocks đã chạy song song trong 1 wave ngay cả ở NTAXA=800 (capacity=756 > 199). Speedup chỉ có khi K > 756 (numpars > 757).

**Occupancy phân tích** (A100, 108 SMs):
- NTAXA=800 (22.4 KB): 7 blocks/SM → capacity 756; register limit 13 blocks/SM
- NTAXA=512: 14 KB → 11 blocks/SM; NTAXA=256: ~7 KB → 23 blocks/SM; NTAXA=128: ~3.5 KB → 47 blocks/SM

**File**: `pars_tree.cuh`, `pars_build.cu`, `gpu/CMakeLists.txt`

---

### 🟡 Trung bình, rủi ro vừa

#### Opt-N: Thread coarsening — xử lý 2 parsimony blocks per lane

**Vấn đề hiện tại**:

`newviewParsimony` inner loop:
```cpp
for (int b = lane; b < width; b += kWarpSize)  // mỗi lane xử lý 1 block/iteration
```
Với `width=40` (N=295, DNA) và `kWarpSize=32`: mỗi lane xử lý `ceil(40/32)=2` iterations,
nhưng lần 2 chỉ có 8 lanes active (`b=32..39`) → 24 lanes idle. **Warp utilization thấp.**

Ngoài ra, giữa mỗi `newview` node là `warpReduceU32(score)` + `__syncwarp()` — overhead sync
tỉ lệ với số node trong `ti[]`.

**Ý tưởng**:

Mỗi lane xử lý **2 blocks** per iteration (`b = lane` và `b = lane + kWarpSize`), tích lũy score:
```cpp
// Trước: 1 block/lane
for (int b = lane; b < width; b += kWarpSize) { ... score += __popc(t_N); }

// Sau: 2 blocks/lane (loop unroll factor 2)
for (int b = lane; b < width; b += kWarpSize * 2) {
    // block b
    ... score += __popc(t_N_b);
    // block b + kWarpSize (nếu còn trong width)
    if (b + kWarpSize < width) { ... score += __popc(t_N_b2); }
}
```

**Lợi ích kỳ vọng**:
- Giảm số `warpReduceU32` calls xuống còn 1 per newview node (thay vì 2 với width=40)
- Tăng arithmetic intensity: mỗi lane làm nhiều FP work hơn giữa 2 sync
- Với width=40: hiện 2 iters (32 lanes active + 8 lanes active) → sau: 1 iter (8 lanes active, mỗi làm 2 blocks)

**Rủi ro**:
- Tăng register pressure (`t_A[]`, `o_A[]` cần 2 sets — nhưng có thể reuse)
- Với STATES=20: `t_A[20]` + `o_A[20]` = 40 registers/set → 2 sets = 80 extra regs → likely spill
- **An toàn hơn với STATES=4 (DNA)**: 2×(4+4) = 16 extra regs, không spill

**Kế hoạch thực hiện**:
1. Thêm template param `UNROLL` vào `newviewParsimony<SharedT, STATES, UNROLL=1>`
2. `UNROLL=2` chỉ cho STATES=4 (DNA); protein giữ UNROLL=1
3. Dispatch: `if (states==4) launch<4, 2> else launch<20, 1>`
4. Benchmark DNA datasets (width lớn) trước

**File**: `gpu/include/pars_tree.cuh` (newviewParsimony, warpEvaluateScore)  
**Effort**: 4–6 giờ + benchmark

---

## Phase 3 Effectiveness Analysis (2026-05-13)

10 datasets, numpars=200, gpu_stop=4:

| Phase | Avg improvement rate |
|-------|---------------------|
| NNI+SPR (even iters) | 19.2% |
| Ratchet (odd iters) | **24.5%** |

Ratchet hiệu quả hơn NNI+SPR trong 6/10 dataset. Đặc biệt rõ ở N≥699 (Ratchet ~52–56% vs NNI+SPR ~28–36%).

## Phase 3 Micro-optimizations (2026-05-14)

| Opt | Tên | Kết quả |
|-----|-----|---------|
| Opt-Q1 | score_tree zero-init loop removal | **−3.4% avg** (10 datasets, cùng env) |
| Opt-Q2 | Lazy gpuNodeRectifierPars (skip first do-while) | **−13.0% avg** (10/10 datasets cải thiện) |

### Opt-Q1: Xóa score_tree zero-init loop (pars_tree.cuh line 308)
Loop `score_tree[p_num] = 0` là dead code — main loop dùng assignment (`=`), không phải `+=`.
Xóa loop giúp giảm overhead sequential write lane-0.

### Opt-Q2: Skip first `gpuNodeRectifierPars` trong `gpuSPRHillClimb`
`gpuNodeRectifierPars` bị gọi redundant ở đầu do-while iter đầu tiên vì runPhase3 đã gọi ngay trước đó.
Fix: `bool first_iter = true;` → skip rectify khi `first_iter`, set false sau.
Rectify vẫn được gọi ở iter 2+ (khi topology thực sự thay đổi sau move).
**File**: `gpu/src/pars_build.cu` — `gpuSPRHillClimb` (lines 322-338)

---

## Thứ tự ưu tiên đề xuất

1. ~~Opt-K~~ ✅  2. ~~Opt-B / Opt-B+~~ ✅  3. ~~Opt-M~~ ✅  4. ~~Opt-P L1+L2b+L3~~ ✅  5. ~~Opt-G xoá~~ ✅
6. ~~Opt-Q1+Q2~~ ✅ (Phase 3 micro-opts, −13% avg)
7. ~~Opt-R~~ ✅ (Restore best_back_vf, **−38% ms/tree, +60% speedup vs CPU**)
8. ~~Reseed~~ ✅ (Reseed bad slots từ threshold pool, 0 regressions / +14.5% time N≈200)
9. ~~Dead code removal~~ ✅ (K1: 155→96 regs, GpuTopology −12.8 KB, API simplification)
10. ~~K' < K (Opt-LessK1)~~ ✅ (**−60% K1 time; +17–28% total speedup; GPU wins +13–25%**)
11. ~~Pool restart simplify~~ ✅ (K2 regs 151→128, simpler lane-0 selection-sort)
12. ~~Opt-S: Per-slot pool locks~~ ✅ (thundering herd fix, pool_size=20 → ~50 blocks/lock)
13. **Opt-N** (thread coarsening UNROLL=2 cho STATES=4): estimate +10-20% cho DNA datasets
14. **gpuRandomNNIs parallelize** (31 lanes idle → all 32 active): estimate 10–20× NNI phase
13. ~~Opt-O~~ — hủy: applyMove chiếm <0.1% thời gian, không đáng optimize
14. ~~Opt-E~~ — hủy: không khả thi, redesign quá lớn

---

## Phase 3 Algorithm Experiments (2026-05-14 → 2026-05-15)

### Current Defaults (2026-05-17, updated post-LessK1)
- `gpu_stop = 6` — **sweet spot** (benchmark 115 datasets: 4.55× total speedup K'=K, 5.35× với K'=0.2K)
- `gpu_top_pct = 0.1`
- `gpu_pool_size = 30`, `numpars = 1000`, `sprdist = 6` — recommended production config
- `gpu_k1_ratio = 0.2` — **production default** (K'=max(30, 200)=200 với K=999); 1.0 = backward compat
- `-gpu_hc_iter` đã bị xóa (2026-05-17); `gpu_stop` là stopping criterion duy nhất

### Output Formatting (2026-05-15)
- Kernel times now in **seconds** (%.3f s) instead of ms
- Individual timing for buildTreesKernel + hillClimbingKernel
- Aligned columns, `═══` borders
- `setbuf(stdout, NULL)` for immediate flush in GPU section

### ✅ Hybrid CPU-GPU — DONE (2026-05-15 → 2026-05-16)

- **hybrid_cb1**: CPU build cây (`_pllMakeParsimonyTreeFast` + `computeParsimony()`) trong khi GPU Kernel 1 chạy async. Push vào `candidateTrees` khi score đủ tốt.
- **hybrid_cb2**: CPU alternates NNI (even) / Ratchet (odd) perturbation trong khi GPU `hillClimbingKernel` chạy. Dùng `PhyloTree::computeParsimony()` — không PLL.
- **[7] redesign**: Xóa `pllInstanceClone`/`pllPartitionsClone` — reuse `tr`/`pr` trực tiếp. `candidateTrees.update()` chạy trong [7] loop, post-loop trong `phyloanalysis.cpp` đã bị xóa.
- **Print format**: `best CPU tree: X  best GPU tree: Y` thay `built = K / K trees`.

**Bug fixes trong quá trình implement**:
- Bug #10: ODR violation `sizeof(SearchInfo)` khác nhau giữa CXX TU (clang++) và CUDA TU (gcc) → fix bằng `#if __cplusplus >= 201103L` trong `tools.h`
- Bug #11: `hybrid_cb2` 5 crash độc lập do PLL state conflict + unsafe function calls → fix bằng loại bỏ toàn bộ PLL parsimony, dùng `PhyloTree::computeParsimony()` only