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
