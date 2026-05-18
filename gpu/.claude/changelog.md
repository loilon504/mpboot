# GPU Parsimony Changelog

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

## 2026-05-15

### Output formatting
- `gpu_init_trees.cu`: print times in seconds (%.3f s), aligned columns, `═══` borders
- `pars_build.cu`: rename kernel labels → `buildTreesKernel` / `hillClimbingKernel`; params on separate line; individual timing per kernel using CUDA events
- `gpu_init_trees.cu`: `setbuf(stdout, NULL)` for immediate flush, restored on exit via RAII

### Default parameter changes
- `gpu_hc_iter`: 30 → **100**
- `gpu_stop`: 4 → **6**

### Combined v2 Phase 3 (runtime switch)
- `runPhase3` in `pars_build.cu`: `if (sprDist == 3)` → Opt-C path; `else` → Symmetric Adaptive path
- Added `last_odd_hash`, `restore_on_next_odd`, `sym_do_ratchet` to `BuildSharedT`
- Benchmark: 115 datasets — Combined v2 d5s6 = **2 worse**, 2.93× total speedup
- Full results: `/gpu/.claude/benchmark/phase3_variants_comparison.md`

### CPU baseline benchmarks
- `output/cpu_d3/`: 115 datasets with sprdist=3, seed=1 → total 139m13s
- `output/gpu_only_*/`: 7 configs (n200/n400 × d3/d4 × s6/s10) running

---

## 2026-05-14

### Phase 3 algorithm exploration
- Tested Opt-C (stagnation detection), Symmetric Adaptive, Combined v1/v2
- All results in `/gpu/.claude/benchmark/phase3_variants_comparison.md`

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

### Opt-R (restore best_back_vf)
- Restore `best_back_vf → back_vf` before every Phase 3 perturbation
- Result: **−38% ms/tree**, +60% speedup vs CPU

### Parameter additions
- `-gpu_device N` (default 1)
- `-gpu_top_pct X` (default 0.1)
- `-gpu_stop N` (was 4, now 6)

---

## 2026-05-12 → 2026-05-13

### Major optimizations
- **Opt-B / Opt-B+**: Lower-bound pruning in `doAddTraverse` — skip testInsert when lb ≥ threshold
- **Opt-D**: Remove `number[]`, `next_vf[]`, `nnxt_vf[]` from `GpuTopology` — pure arithmetic via `vfToNum`, `vfNextFace`, `vfNnxtFace` → struct −37.5 KB
- **Opt-K**: `testInsert` pre-refresh — eliminate redundant Fitch step
- **Opt-H**: Early stopping Phase 3 — `no_improve_count ≥ gpu_stop` → break

### Joined kernel
- Merged Phase 1+2+3 into single `buildParsimonyTreesKernel` (no separate SPR kernel)
- `gpu_spr.cu` reduced to no-op stub

### Fixed bugs
- `vfNnxtFace(q,N)` → `vfNnxtFace(p,N)` in testInsert (#A)
- `sh.randomMP` uninitialized (#B)
- Missing `__syncwarp()` (#C, #6)
- `q_num >= N` → `> N` in doAddTraverse (#7)
- `node_num >= N` → `> N` in stepwise DFS (#8)
