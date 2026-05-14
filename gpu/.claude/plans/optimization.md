# GPU Parsimony Optimizations — Tổng quan (2026-05-13)
<!-- Lưu tại: /raid/home/loinguyen/workspace/mpboot-gpu/mpboot/gpu/.claude/plans/optimization.md -->

## Đã hoàn thành

| Opt | Tên | Kết quả |
|-----|-----|---------|
| testInsert pre-refresh | Loại bỏ Fitch step thừa | −24% kernel time |
| Opt-D | GpuTopology struct shrink | −8.7% ms/tree |
| Opt-H | Early stopping Phase 3 (gpu_stop) | −32.4% ms/tree |
| Opt-G | Selective Phase 3 atomicMin margin | −22–50% tùy dataset |
| Opt-G2 | Selective Phase 3 two-kernel exact top-X% | −44% trên N=295 |
| Opt-I | NNI strength configurable (gpu_nni_strength) | Baseline alignment |
| Opt-K | gpu_stop default 2 → 4 | Quality tốt hơn trên N≥295 |
| Opt-B | Subtree prune trong SPR DFS (per-edge lb check) | ~22% prune rate, avg 1.31× speedup |
| Opt-B+ | Tighter lb: thêm score_tree[tip_p] | **~30% prune rate, avg 1.49× speedup, 10/10 faster** |
| Opt-M | Template specialization newviewParsimony<STATES> | **avg 2.16× ms/tree (np=200, 50 datasets)** |

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

#### Opt-P Layer 3: NTAXA Templating (planned)

**Ý tưởng**: Template `BuildShared` theo NTAXA bucket (128, 256, 384, 512, 800) để array sizes thu nhỏ theo N thực tế. Dataset N≤128 có thể đạt 41 blocks/SM (64% occupancy).

**Buckets**: N≤128 → NTAXA=128; N≤256 → 256; N≤384 → 384; N≤512 → 512; N>512 → 800.  
**Expected savings** (NTAXA=128): BuildShared ~4 KB → 41 blocks/SM!

**File**: `gpu/include/pars_tree.cuh` (BuildSharedT<NTAXA> template), `gpu/src/pars_build.cu` (dispatch), `gpu/src/gpu_init_trees.cu` (NTAXA dispatch added to STATES dispatch).  
**Effort**: 1–2 ngày.

---

### 🟡 Trung bình, rủi ro vừa

#### Opt-N: Thread coarsening — xử lý nhiều parsimony blocks per lane

**Ý tưởng**: Mỗi lane hiện xử lý `b = lane, lane+32, lane+64, ...` (stride kWarpSize). Tăng lên 2 blocks/lane giúp giảm `__syncwarp()` overhead và tăng arithmetic intensity.

**Rủi ro**: Tăng register pressure, cần benchmark cẩn thận.

---

#### Opt-O: Incremental parsimony update sau SPR move

**Ý tưởng**: Sau `applyMove`, chỉ update path từ insert point lên đến LCA của old/new subtrees thay vì full lazy refresh.

**File**: `pars_build.cu` — `applyMove`.  
**Rủi ro**: Cần chứng minh correctness.

---

### 🔴 Phức tạp, rủi ro cao

#### Opt-J: parsVect caching vào shared memory *(bị bác)*
Giảm occupancy, phức tạp, rủi ro chưa rõ benefit vượt chi phí.

#### Opt-E: Sub-warp parallel candidate evaluation *(future)*
4× speedup lý thuyết nhưng redesign hoàn toàn kernel. Effort 1-2 tuần.

---

## Thứ tự ưu tiên đề xuất

1. ~~Opt-K~~ ✅  2. ~~Opt-B / Opt-B+~~ ✅  3. ~~Opt-M~~ ✅
4. Profile Nsight để xác nhận bottleneck tiếp theo
5. **Opt-N** (thread coarsening): thử nếu profile cho thấy memory-bound
6. **Opt-E** (sub-warp parallel): khi tất cả opt nhỏ xong
