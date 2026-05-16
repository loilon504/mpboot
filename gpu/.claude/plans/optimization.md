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
| Opt-I | NNI strength configurable (gpu_nni_strength) | Baseline alignment |
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

---

## Profiling Findings (2026-05-13)

**Method**: ptxas compile-time analysis (`--ptxas-options=-v`)

| Kernel | STATES | Registers/thread | Spills |
|--------|--------|-----------------|--------|
| buildParsimonyTreesKernel | 32 | 173 | 0 |
| buildParsimonyTreesKernel | 20 | 158 | 0 |
| buildPhase3Kernel | 32 | 166 | 0 |
| buildPhase3Kernel | 20 | 134 | 0 |
| buildPhase3Kernel | 4 | 148 | 0 |

**Occupancy**: `BuildShared = 28.4 KB` là bottleneck → **5 blocks/SM** (164 KB ÷ 28.4 KB) → **7.8% theoretical occupancy** (5/64 warps).  
Register limit = 11–15 blocks/SM — NOT the bottleneck.  
**Opt-M speedup đến từ ILP/loop unrolling, KHÔNG từ occupancy.**

→ Full analysis: [benchmark/profile_results_postOptM.md](../benchmark/profile_results_postOptM.md)

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

#### Opt-O: Incremental parsimony update sau SPR move

**Vấn đề hiện tại**:

Sau `applyMove(rm, ins)`, code gọi `createTiAndNewviewParsimony(rm, ...)` — đây là DFS từ `rm`
đi qua tất cả descendants với `xpars=0`. Với xPars system, chỉ một số ít node thực sự stale,
nhưng cây sau NNI perturbation (odd iter) hoặc nhiều applyMove liên tiếp có nhiều node stale.

**Ý tưởng — Incremental path update**:

Sau SPR move (rm, ins), chỉ node `rm` và các ancestors của `rm` đến root bị stale.
Path lên root có độ dài O(depth) ≈ O(log N) cho balanced tree, tệ nhất O(N) cho caterpillar.

```
Trước move:
    A --- B --- C --- rm --- D
                      |
                     tip_p

Sau applyMove(rm, ins):
    A --- B --- C --- (trống)    rm re-inserted tại ins
                \
                 D (kết nối trực tiếp với C)
```

Path bị ảnh hưởng: chỉ `rm` và path từ nơi rm được chèn vào lên đến tổ tiên chung.

**Phân tích độ khó**:

GPU không có con trỏ parent → cần DFS ngược. Với `ti[]` được tạo bởi
`computeTraversalInfoParsimony`, thứ tự là bottom-up (post-order). Incremental update cần biết
ancestor path → **cần lưu thêm parent[] array** hoặc traverse lại.

**Option A** (simpler): Sau applyMove, chỉ mark `xpars[rm]=0` và tất cả ancestors dọc đường
lên `start_vface`. Walk up: bắt đầu từ `back_vf[rm]` (parent of rm sau move), tiếp tục theo
`back_vf` của từng face đến khi reach `start_vface`. Mark tất cả stale. Cost: O(depth) marks,
sau đó `computeTraversalInfoParsimony` lazy-refresh đúng những gì cần.

**Option B** (aggressive): Tính lại parsVect và score_tree NGAY trên path từ rm lên root, không
cần `computeTraversalInfoParsimony`. Tiết kiệm DFS overhead nhưng phức tạp hơn.

**Tác động thực tế**:
- Sau testInsert (undo-based): KHÔNG cần incremental — undo restore topology, xpars được
  quản lý trong pre-refresh block. Opt-O chỉ liên quan đến `applyMove` (khi tìm được move tốt).
- `applyMove` chỉ gọi khi `sh.bestParsimony < sh.randomMP` → ít lần/iteration
- Timing data: `apply=0.05%` của SPR search time → **Opt-O không đáng để optimize**

**Kết luận**: Opt-O không nên implement. `applyMove` chiếm <0.1% thời gian, lợi ích gần như zero.
Đề xuất: **hủy Opt-O**.

---

### 🔴 Phức tạp, rủi ro cao

#### Opt-J: parsVect caching vào shared memory *(bị bác)*
Giảm occupancy, phức tạp, rủi ro chưa rõ benefit vượt chi phí.

---

## Opt-R: Restore best_back_vf trước mỗi Phase 3 perturbation (2026-05-14)

**Phát hiện**: Iteration i bắt đầu từ end-state của iteration i-1 (có thể tệ hơn best), gây "topology drift". Fix: warp-parallel copy `best_back_vf → back_vf` đầu mỗi even (NNI) VÀ odd (Ratchet) iteration.

```cpp
for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
    topo->back_vf[vf] = topo->best_back_vf[vf];
__syncwarp();
```

**Kết quả** (10 datasets, −gpu_top_pct -1):
- Avg speedup vs baseline: **−38.0%** ms/tree (tất cả 10 datasets cải thiện)
- Ratchet improvement rate: 35% → 52% (N=295), 52% → 70% (N=699)

**File**: `gpu/src/pars_build.cu` — `runPhase3`

---

## GPU vs CPU Benchmark tổng hợp — gpu_final_800 (2026-05-14)

**Config**: numpars=400, sprdist=3, gpu_stop=4, gpu_top_pct=0.1, seed=1, 115 datasets  
**Includes**: Tất cả optimizations (Opt-P, Opt-M, Opt-Q1+Q2, Opt-R)  
→ Full results: `output/gpu_final_800/`

| Metric | gpu_np400 (trước Opt-R) | gpu_final_800 (Opt-R) |
|--------|------------------------|----------------------|
| Avg speedup vs CPU | 2.28× | **3.66×** |
| Max speedup | 6.69× | **12.88×** |
| N≥400: speedup | 3.5–6.7× | **5–10×** |
| GPU quality better | 33/115 (29%) | **40/115 (35%)** |
| GPU worse quality | 25/115 (22%) | **20/115 (17%)** |
| Avg Δparsimony (GPU−CPU) | −0.13 | **−0.22** |

## GPU vs CPU Benchmark tổng hợp — gpu_np400 (2026-05-13, trước Opt-R)

**Config**: numpars=400, sprdist=3, gpu_stop=4, seed=1, 115 datasets (N=50–767)  
→ Full results: [benchmark/gpu_vs_cpu_np400_115datasets.md](../benchmark/gpu_vs_cpu_np400_115datasets.md)

| Metric | Value |
|--------|-------|
| Avg speedup vs CPU | **2.28×** |
| N≥400: speedup | **3.5–6.7×** |
| N≤60: speedup | 0.7–1.3× (overhead dominant) |
| GPU quality better | 33/115 (29%) |
| Same quality | 57/115 (50%) |
| Avg Δparsimony | **−0.13** (GPU nhỉnh hơn) |

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
8. **Opt-N** (thread coarsening UNROLL=2 cho STATES=4): estimate +10-20% cho DNA datasets
9. ~~Opt-O~~ — hủy: applyMove chiếm <0.1% thời gian, không đáng optimize
10. ~~Opt-E~~ — hủy: không khả thi, redesign quá lớn

---

## Phase 3 Algorithm Experiments (2026-05-14 → 2026-05-15)

### Strategy 0: Remove hot-loop timing fields
- Removed fine-grained timing vars from `BuildSharedT` (t_line2291, t_search, etc.)
- Result: 148 → **136 regs** → 15 blocks/SM (was 13)
- **File**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`

### Phase 3 Variants Explored

| Variant | Description | Result |
|---------|-------------|--------|
| Opt-C | Stagnation detection: hash post-Ratchet topology, conditional restore | **Best for sprdist=3** |
| Symmetric Adaptive | NNI/Ratchet switching with restore-on-switch | Best for sprdist≥4 |
| Combined v2 | Opt-C@sprdist=3, Symmetric@sprdist>3 (runtime switch) | **CURRENT DEFAULT** |

### Combined v2 Benchmark (115 datasets, seed=1, numpars=400, top_pct=0.1)

| Config | Wor | AvgΔ | Total | Tot spd |
|--------|-----|------|-------|---------|
| d3 s4 p0.1 (Opt-C auto) | 16 | +0.27 | ~21m | 5.20× |
| d5 s6 p0.1 (Sym auto) | **2** | +3.64 | 37m | 2.93× |
| d6 s4 p0.1 (Sym auto) | 5 | +3.88 | 39m | 2.78× |

→ Details: `/gpu/.claude/benchmark/phase3_variants_comparison.md`

### Current Defaults (2026-05-15)
- `gpu_hc_iter = 100` (was 30)
- `gpu_stop = 6` (was 4)
- `gpu_top_pct = 0.1`

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

---

## Hybrid Benchmark — hybrid4 (2026-05-16)

**Setup**: 20 datasets (N=50–640, 7 protein + 13 DNA), seed=1, 5 configs × run trên device 1–5 (A100).

| Config | numpars | sprdist | gpu_stop | Mean speedup | Quality vs CPU |
|--------|---------|---------|----------|-------------|----------------|
| n200d3s4 | 200 | 3 | 4 | **3.62×** | 8↑ / 10= / 2↓ |
| n200d4s6 | 200 | 4 | 6 | 2.64× | **12↑ / 7= / 1↓** |
| n200d5s6 | 200 | 5 | 6 | 2.12× | **12↑ / 7= / 1↓** |
| n400d4s6 | 400 | 4 | 6 | 1.99× | **12↑ / 7= / 1↓** |
| n400d5s6 | 400 | 5 | 6 | 1.66× | **12↑ / 7= / 1↓** |

**Insights**:
- **Sweet spot**: `n200d4s6` — 2.64× speedup, 95% win/tie vs CPU. sprdist=5 cho cùng quality nhưng chậm hơn 25%.
- **Speed vs quality**: `n200d3s4` nhanh nhất (3.62×) nhưng chỉ 73% win/tie — sprdist=3 không đủ cho N>300.
- **n400 không worth it**: 2× trees nhưng quality tăng ≤2 điểm, mất 40–50% tốc độ.
- **Timing regression nhỏ vs hybrid3**: [7] loop mới thêm K lần `computeParsimony` — overhead ~20% cho N≤100, không đáng kể cho N≥200. Trade-off chấp nhận được vì đổi lấy tính đúng đắn.
- **dna_M7964 (N=640)**: GPU thua CPU ở d3s4 (~20 điểm), cần sprdist≥4. n400d5s6 xấp xỉ bằng CPU.

→ Full logs: `output/hybrid4_{n200d3s4,n200d4s6,n200d5s6,n400d4s6,n400d5s6}/`
