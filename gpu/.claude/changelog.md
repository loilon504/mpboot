# GPU Parsimony Changelog

## 2026-05-31 (Priority 5 — Treels Diversity)

### Fix 3 — Near-optimal treels save margin (`pars_build.cu`, `gpu_init_trees.cu`)
- **Bug**: Sau Fix 2, tất cả K2 workers converge về T* (optimal) → treels bị fill bởi T* → hash dedup loại hết → REPS chọn T* cho mọi bootstrap replicate → star consensus (nsplits=1).
- **Root cause**: Point A (testInsert treels save) chỉ lưu khi `mp < sh.randomMP`. Tại T*, không candidate nào có mp < T* → A saves nothing. Point B và C là con của A → cũng không có diversity.
- **Fix**:
  - `pars_build.cu` testInsert (~line 140): `mp < sh.randomMP` → `mp < sh.randomMP + sh.save_margin`
  - `pars_tree.cuh` BuildSharedT: thêm `unsigned int save_margin;`
  - `pars_tree.cuh` GpuParsimonyMem: thêm `unsigned int save_margin;`
  - `pars_build.cu` buildPhase3Kernel: thêm `save_margin` param, init `sh.save_margin = save_margin`
  - `gpu_init_trees.cu`: set `mem->save_margin = params.gpu_treels_margin`; margin-based `logl_cutoff`
  - `tools.h`/`tools.cpp`: thêm `-gpu_treels_margin N` (default=10)
- **Cơ chế**: `testInsert` lưu "p at q_cand" topologies với mp ≤ sh.randomMP + margin. Tại T* (sh.randomMP=22376, margin=10): lưu tất cả candidates với parsimony 22376-22385 → O(N) diverse topologies per SPR pass.
- **Status**: Build OK, đang verify với aa/138.

## 2026-05-31

### Fix 2 — AA tip bitmask bug (`gpu_init_trees.cu`, `uploadSankoffTipParsVect`) ✅ VERIFIED
- **Bug**: `bitmask = (unsigned int)nuc` dùng AA state index (0=Ala…) như Fitch bitmask →
  Alanine (index=0) cho `bitmask=0` → tất cả states = kSankoffInf → tip rỗng → tree search sai.
- **Root cause**: DNA dùng `PLL_MAP_NT` (nuc IS bitmask); AA dùng `PLL_MAP_AA` (nuc là index →
  cần one-hot convert).
- **Fix** (`gpu_init_trees.cu` line ~87):
  ```cpp
  const bool is_bitmask_coded = (undetermined == ((1u << states) - 1u));
  unsigned int bitmask = is_bitmask_coded
      ? (unsigned int)nuc
      : (nuc < (unsigned char)states ? (1u << nuc) : (1u << states) - 1u);
  ```
- **Verification**: 20 protein datasets (treebase non-bootstrap) re-run → 20/20 diff ≤ 0
  (16 match CPU, 4 GPU tốt hơn CPU). DNA không bị ảnh hưởng (PLL_MAP_NT vẫn đúng).

### Bug A fix — score_tree = 0 Sankoff (`pars_tree.cuh`) ✅ VERIFIED
- **Bug**: Sankoff branch trong `warpNewviewStep` không accumulate `score` → `score_tree[p_num] = 0`
  với mọi inner node → Opt-B lower-bound prune vô hiệu → protein SPR chậm hơn cần thiết.
- **Fix** (`pars_tree.cuh` Sankoff branch ~line 378–392): thêm `min_cost_b` tracking inline:
  ```cpp
  unsigned int min_cost_b = kSankoffInf;
  // inside ii loop:
  unsigned int val = best_left + best_right;
  p_base[...] = (parsimonyNumber)val;
  min_cost_b = min(min_cost_b, val);
  // after ii loop:
  score += (min_cost_b < kSankoffInf) ? min_cost_b : 0u;
  ```
- **Impact**: performance only (score_tree dùng cho Opt-B prune, không ảnh hưởng evaluate).
- **Benchmark** (6 protein datasets, `output/treebase_gpu_bugA/`):

  | Dataset | N | Baseline (Fix2) | Bug A | Speedup | Score Δ |
  |---------|---|-----------------|-------|---------|---------|
  | M1118_137 | 137 | 311.2s | 279.6s | +11% | 0 |
  | M11341_100 | 100 | 409.8s | 382.0s | +7% | 0 |
  | M4249_153 | 153 | 1269.4s | 910.9s | **+28%** | 0 |
  | M4318_78 | 78 | 991.9s | 857.8s | +14% | 0 |
  | M4780_90 | 90 | 427.7s | 363.7s | +15% | +1* |
  | M8569_164 | 164 | 459.1s | 404.1s | +12% | 0 |

  *M4780 +1: stochastic variation do Opt-B thay đổi candidate evaluation order — bình thường với heuristic search. Lower bound vẫn valid.
  **Kết quả: 5/6 score unchanged; speedup 7–28%, trung bình ~14%.**

### Priority 4 — AA bootstrap re-benchmark (`pandit_gpu_bb_non_new2/aa/`) ✅ VERIFIED
- **Setup**: 5 pandit AA datasets, `-bb 1000 -cost prot.cost`, Fix 2 + Bug A binary.
- **Kết quả**: 5/5 GPU score ≤ CPU ref (diff -2…0); GPU cũ (pre-Fix2): diff +1…+22.
- **Speedup**: 2.74–4.65× vs CPU (mean 3.56×).
- **Không còn star tree**: cor=0.0 ban đầu, tăng dần — Fix 2 đã sửa AA tip bug hoàn toàn.

---

## 2026-05-18

### Opt-S: Per-slot pool spinlocks (thundering herd fix)
- **Trước**: 1 `pool_lock` global int bị tranh chấp bởi tất cả K=1000 blocks → thundering herd
- **Sau**: `pool_slot_locks[pool_size]` — mỗi slot có lock riêng (pool_size=20 → max 50 blocks/lock)
- Files: `pars_tree.cuh` (field rename), `pars_tree.cu` (alloc/free), `pars_build.cu` (2 sites), `gpu_init_trees.cu` (hybrid_cb reset)
- Build: 0 errors; test tree1.phy N=295: GPU=6662, CPU=6669 ✅

### Pool restart simplification (pool_size ≤ 32)
- **Trước**: Warp-parallel k-th min selection (scattered lane work, complex sync)
- **Sau**: Lane-0 O(pool_size²) selection-sort với `uint32_t used` bitmask — đủ vì pool_size ≤ 32
- Setup block (bcast[3]/bcast[5]) giữ nguyên; chỉ thay scan block
- Effect: K2 regs **151 → 128** (đo bằng NCU)

### NCU profiling (A100, N=295, tree1.phy, worker=1000/600)
- K2 `buildPhase3Kernel`: 128 regs/thread, Theoretical Occ=20.31%, Achieved Occ=13.29% (worker=1000)
- Shared memory là bottleneck chiếm dụng (không phải registers): 12.9 KB/block → 12 blocks/SM
- Bottleneck xác định: `gpuRandomNNIs` — 31 lanes idle (lane-0 only)

### Benchmark 8 configs (115 datasets, numpars=200)
- 4 configs đã xong: w400/1000 × d4/6 × stop2/3 → avg speedup 2.27–3.84× vs CPU (cpu_d6 baseline)
- Sweet spot: `w400_d4_s3` — 3.84× avg, 9.68s avg, 48 wins / 115 datasets
- 4 configs mới đang chạy: w600 × d4/6 × s3; w400 × d4/6 × s4

---


## 2026-05-14

### Strategy 0: timing fields removal
- Removed hot-loop timing from `BuildSharedT` (t_line2291, t_search, etc.)
- 148 → **136 regs**, 13 → **15 blocks/SM**

### Opt-P (BuildSharedT NTAXA templating)
- `BuildSharedT<NTAXA>` template: 128/256/512/800 buckets
- `GPU_NTAXA_TEMPLATE=OFF` flag (default, fast build)

### Opt-M (STATES dispatch)
- Only DNA (4) and Protein (20) STATES in dispatch

### Opt-Q1+Q2
- Q1: removed dead `score_tree[p_num]=0` zero-init loop
- Q2: skip first `gpuNodeRectifierPars` in `gpuSPRHillClimb` (redundant)

### Parameter additions
- `-gpu_device N` (default 1)
---

## 2026-05-12 → 2026-05-13

### Major optimizations
- **Opt-B / Opt-B+**: Lower-bound pruning in `doAddTraverse` — skip testInsert when lb ≥ threshold
- **Opt-D**: Remove `number[]`, `next_vf[]`, `nnxt_vf[]` from `GpuTopology` — pure arithmetic via `vfToNum`, `vfNextFace`, `vfNnxtFace` → struct −37.5 KB
- **Opt-K**: `testInsert` pre-refresh — eliminate redundant Fitch step

### Joined kernel
- Merged Phase 1+2+3 into single `buildParsimonyTreesKernel` (no separate SPR kernel)
- `gpu_spr.cu` reduced to no-op stub

### Fixed bugs
- `vfNnxtFace(q,N)` → `vfNnxtFace(p,N)` in testInsert (#A)
- `sh.randomMP` uninitialized (#B)
- Missing `__syncwarp()` (#C, #6)
- `q_num >= N` → `> N` in doAddTraverse (#7)
- `node_num >= N` → `> N` in stepwise DFS (#8)
