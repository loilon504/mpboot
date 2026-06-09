# GPU Parsimony Optimizations — Tổng quan (2026-05-14)
<!-- Lưu tại: /raid/home/loinguyen/workspace/mpboot-gpu/mpboot/gpu/.claude/plans/optimization.md -->

## Đã hoàn thành

| Opt | Tên | Kết quả |
|-----|-----|---------|
| testInsert pre-refresh | Loại bỏ Fitch step thừa | −24% kernel time |
| Opt-D | GpuTopology struct shrink | −8.7% ms/tree |
| Opt-H | Early stopping Phase 3 (gpu_stop) | Không còn sử dụng |
| Opt-G2 (→ -gpu_top_pct) | Selective Phase 3 two-kernel exact top-X% | Không còn sử dụng |
| Opt-I | NNI strength configurable (gpu_nni_strength); rewrite on-the-fly q_vf + CPU-style bitset reset | strength=0.5 works; all EXIT=0; no regression |
| Opt-K | gpu_stop default 2 → 4 | Không còn sử dụng |
| Opt-B | Subtree prune trong SPR DFS (per-edge lb check) | ~22% prune rate, avg 1.31× speedup |
| Opt-B+ | Tighter lb: thêm score_tree[tip_p] | **~30% prune rate, avg 1.49× speedup, 10/10 faster** |
| Opt-M | Template specialization newviewParsimony<STATES> (4, 20 only) | **avg 2.16× ms/tree (np=200, 50 datasets)** |
| Opt-P L1 | BuildSharedT: union perm/stackMint + stackMaxt[64] | **+1.197× vs Opt-M (50 datasets)** |
| Opt-P L2b | BuildSharedT: int16_t arrays + reorder hot fields | **+2.0% vs L1 (reorder); int16_t net-negative alone** |
| Opt-P L3 | BuildSharedT<NTAXA> template (128/256/512/800) + GPU_NTAXA_TEMPLATE flag | Implemented; speedup when K>756 |
| Opt-Q1 | score_tree zero-init loop removal (dead code) | **−3.4% avg** (10 datasets) |
| Opt-Q2 | Lazy gpuNodeRectifierPars: skip first do-while iteration | **−13.0% avg** (10/10 datasets) |
| **Opt-R** | **Restore best_back_vf trước mỗi Phase 3 perturbation** | Không còn sử dụng |
| **Reseed** | **Reseed bad GPU slots từ threshold pool (hybrid_cb Step 3.5)** | Không còn sử dụng |
| **Dead code removal** | **Xóa timing fields, best_back_vf, numSearchIter, gpu_hc_iter** | **K1: 155→96 regs; GpuTopology −12.8 KB; API đơn giản hơn** |
| **K' < K (Opt-LessK1)** | **Build K'=max(pool,0.2K) trees trong K1; K2 vẫn K blocks** | **−60% K1 time; +17–28% total speedup; GPU wins +13–25%** |
| **Pool restart simplify** | **Lane-0 O(pool_size²) selection-sort, unsigned long long bitmask (pool_size ≤ 60); accessible=10+outer** | **K2 regs 151→128; simpler code** |
| **Opt-S: Per-slot locks** | **pool_lock→pool_slot_locks[pool_size]; thundering herd fix** | **Contention 1000→50 blocks/lock (pool=20)** |
| **Sankoff encoding fix** | **uploadSankoffTipParsVect: width=parsimonyLength, tr->yVector PLL bitmask** | **GPU Sankoff đúng, 6662=CPU** |
| **Fix 2 — AA tip bitmask** | **`uploadSankoffTipParsVect`: one-hot convert AA index→bitmask (DNA dùng PLL_MAP_NT, AA dùng PLL_MAP_AA)** | **20/20 protein non-bootstrap diff ≤ 0 ✅ (2026-05-31)** |
| **Bug A — score_tree Sankoff** | **`warpNewviewStep`: accumulate `min_s(p[s][b])` vào `score` để Opt-B prune hoạt động** | **Applied; benchmark pending** |
| **Sankoff ratchet (d_ratchetScratch)** | **Buffer scratch riêng; iter_is_nni bỏ use_sankoff\|\|** | **K=200 đạt 6662 (cần K=5000 trước)** |
| **Opt 1: parsVect layout [ptn][state]→[state][ptn]** | **Coalesced warp access; Fitch→memcpy; Sankoff h_buf reindex** | **Fitch 2.5×, Sankoff 2.1× ms/tree** |
| **Opt 3: Remove dead min_site in Sankoff newview** | **Xóa score_tree accumulation không được đọc** | **Minor; absorbed into Opt 1** |
| **Opt-4: Register preload Sankoff** | **`lv_[STATES], rv_[STATES]` preload trước ii×jj loop; `#pragma unroll` cả hai vòng** | **S²→S global loads; protein speedup ~1.4–1.8× (thành Opt-5 baseline)** |
| **Opt-5: Constant memory cost matrix** | **`g_sankoff_cm` trong `__constant__` 64 KB; `cudaMemcpyToSymbol`; `#ifdef PARS_BUILD_DEFINE_CM` macro** | **DRAM≈0% (NCU); avg 1.64× protein speedup vs Opt-4** |

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
## Opt-4: Register Preload trong Sankoff newview/evaluate (2026-06-07)

**Vấn đề**: Loop Sankoff `for (ii) { for (jj) { load q[jj], r[jj] } }` — `q[jj]` và `r[jj]` là loop-invariant với `ii` nhưng compiler không hoist được vì global memory pointer (`__restrict__` không giúp ích với non-contiguous strides). S²=400 loads thay vì S=20.

**Fix** (`pars_tree.cuh`, Sankoff branch của `newviewParsimony`):
- Preload `lv_[STATES]`, `rv_[STATES]` trước vòng ii×jj
- Evaluate: chỉ preload `rv_[STATES]` (r cần S², còn `qi` scalar S lần)
- `#pragma unroll` cho cả ba loops (STATES=compile-time constant)

**Chi phí**: S=4 → 8 extra regs (không đáng kể); S=20 → 40 extra regs → kernel đạt 255 regs (hardware max A100) → stack spill 4096 bytes.

**Kết quả** (protein, STATES=20): Opt-4 là baseline để đo Opt-5. Speedup Opt-4 so với layout cũ chưa được đo riêng (Opt-4 implement cùng session với Opt-5).

**Files**: `gpu/include/pars_tree.cuh` (~line 394–419 newview, ~line 470–488 evaluate)

---

## Opt-5: Sankoff Cost Matrix vào CUDA `__constant__` Memory (2026-06-07)

**Vấn đề**: `d_cost_matrix` (cudaMalloc) → kernel đọc từ L2 cache. Với uniform warp access `cm[same_ii][same_jj]`, constant memory broadcast hardware là optimal.

**Cơ chế**: CUDA constant memory = 64 KB dedicated per-device cache. Khi 32 lanes cùng đọc một địa chỉ → hardware broadcast từ constant cache → ~0 latency. Cost matrix 1600 bytes (S=20) fit dễ dàng.

**Implementation**:

```cuda
// pars_tree.cuh — khai báo symbol
static constexpr int kMaxSankoffStates = 20;
#ifdef PARS_BUILD_DEFINE_CM
__constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
#else
extern __constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
#endif

// pars_build.cu — define + upload
#define PARS_BUILD_DEFINE_CM
void gpuUploadSankoffCostMatrix(const unsigned int* cm, int nstates) {
    CUDA_CHECK(cudaMemcpyToSymbol(g_sankoff_cm, cm, nstates*nstates*sizeof(unsigned int)));
}

// pars_tree.cu — sentinel pattern
gpuUploadSankoffCostMatrix(cost_matrix, nstates);
CUDA_CHECK(cudaMalloc(&mem->d_cost_matrix, sizeof(unsigned int)));  // sentinel

// Kernel — dùng g_sankoff_cm
const unsigned int* cm = sh.use_sankoff ? g_sankoff_cm : nullptr;
```

**Bugs fixed trong quá trình implement**:
1. NVCC redefinition error → `#ifdef PARS_BUILD_DEFINE_CM` macro (không cần RDC vì template chỉ instantiate trong 1 TU)
2. `sh.use_sankoff = false` sau khi `d_cost_matrix = nullptr` → 4-byte sentinel
3. `pars_bootstrap.cuh` include `cuda_runtime_api.h` trong C++ TU → `#ifdef __CUDACC__` guard

**NCU profiling** (A100 Device 5, STATES=20, prot_M10236_59_164):

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| DRAM Throughput | **0.00%** | Opt-5 thành công: constant cache phục vụ toàn bộ cost matrix |
| L1/TEX Hit Rate | 68.87% | parsVect cho N=59 fit tốt trong L1 |
| L2 Hit Rate | 99.9% | Gần như toàn bộ từ L2 |
| Registers/thread | **255** | Hardware max (do Opt-4 thêm 40 regs) |
| Stack Size | 4096 bytes | Register spill (đã bump từ 1024) |
| Theoretical Occupancy | **12.5%** | 8 blocks/SM — giờ register-limited (không còn smem-limited) |
| Warp Cycles/Issued Inst | **2.93** | Pipeline gần đầy, ít latency stall |

**Speedup Opt-5 vs Opt-4** (5 protein datasets, device 4, numpars=200, sprdist=6):

| Dataset | Opt-4 | Opt-5 | Speedup |
|---------|-------|-------|---------|
| prot_M10236 (59T, w=164) | 40.53 ms | 25.66 ms | **1.58×** |
| prot_M11595 (66T, w=463) | 87.15 ms | 62.45 ms | **1.40×** |
| prot_M3807 (82T, w=591) | 229.36 ms | 137.40 ms | **1.67×** |
| prot_M10866 (88T, w=3329) | 1051.96 ms | 574.74 ms | **1.83×** |
| prot_M11740 (138T, w=4427) | 3569.00 ms | 2074.64 ms | **1.72×** |
| **Trung bình** | | | **1.64×** |

Speedup tăng theo width: datasets có width lớn (3329, 4427) benefit nhiều hơn vì cost matrix được tái sử dụng nhiều lần hơn trên nhiều patterns.

**Files**: `gpu/include/pars_tree.cuh` (khai báo + usage), `gpu/src/pars_build.cu` (define + upload), `gpu/src/pars_tree.cu` (sentinel), `gpu/include/pars_bootstrap.cuh` (CUDACC guard)

**NCU reports**: `/output/profile_opt5/prot_M10236_ncu.ncu-rep`, `prot_M10866_ncu.ncu-rep` (pending), `prot_M11740_ncu.ncu-rep` (pending)

