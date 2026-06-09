# MPBoot / UFBoot2 — Paper Summary & Bootstrap Code Analysis

**Paper**: Hoang et al. (2018) "UFBoot2: Improving the ultrafast bootstrap approximation"
*Molecular Biology and Evolution* 35(2):518–522 — DOI 10.1093/molbev/msx281
(BMC Evol Bio DOI 10.1186/s12862-018-1131-3 redirects to this MBE paper)

---

## 1. What is MPBoot?

MPBoot là phiên bản **maximum parsimony** của UFBoot. Thay vì chạy full tree search per bootstrap replicate (O(B × search_cost)), MPBoot:

1. **Amortize** tree search — chỉ build cây 1 lần, collect pool of candidate trees `C`.
2. **REPS evaluation** — mỗi candidate tree đánh giá nhanh trên tất cả B replicates bằng dot product (`_pattern_pars · boot_samples_pars[i]`): không cần tree traversal per replicate.
3. **Online update** — sau mỗi candidate tree mới, update `boot_trees[i]` (best tree index per replicate).

**Speedup vs standard parsimony bootstrap**: ~100–1000× (REPS evaluation thay full search).

---

## 2. Algorithm (Pseudocode)

```
# --- PRE-INIT ---
boot_samples_pars[B][P]    ← multinomial resampling, P = #patterns
best_pars[B]               ← -INF
best_tree[B]               ← -1
C = {}                     ← candidate tree pool

# --- MAIN LOOP (curIt = 1..max) ---
for curIt = 1, 2, ... until STOP:

    # Perturbation (chọn 1 trong 2)
    if ratchet_iter triggered:
        T = random from C
        A' = perturb alignment (upweight ratchet_percent% informative sites ×2)
        T' = NNI search on T under A'
        T_start = NNI search on T' under original A   # post-ratchet

    else:
        T_start = random from C (or best tree)
        T_start = random-NNI perturb(T_start)         # IQP perturbation

    # Hill-climbing (đây là phần GPU hiện tại thực hiện)
    T_new = SPR + NNI search on T_start under A

    # Compute per-pattern parsimony (Fitch traversal, O(N × P))
    _pattern_pars[p] = parsimony cost at pattern p for T_new

    # --- REPS evaluation (BOTTLE NECK #2, GPU target) ---
    for i = 0..B-1:
        rell = sum_p( _pattern_pars[p] * boot_samples_pars[i][p] )
        if rell > best_pars[i]:
            best_pars[i] = rell
            best_tree[i] = tree_index

    # Convergence check every step_iterations=100
    if split_correlation(prev, now) >= 0.99 AND no_improve >= unsuccess_iter:
        STOP

# --- POST-PROCESSING (optional) ---
if -opt_btree:
    for i = 0..B-1:
        switch alignment to boot_samples_pars[i]     # modifyPatternFreq()
        T_best[i] = SPR/NNI optimize(best_tree[i])   # per-replicate search!

# --- OUTPUT ---
compute majority-rule consensus with support = count(best_tree[i] contains split) / B
```

---

## 3. UFBoot2 Key Improvements (vs UFBoot1)

### a. Polytomy correction (epsilon_boot = 0.5)
UFBoot1: chọn 1 tree max-RELL per replicate → overconfident on short branches.
UFBoot2: chọn **tất cả trees** trong window ± epsilon_boot của max → giảm overconfidence.
MPBoot tương đương: `-mulhits` flag lưu `boot_trees_parsimony[i]` là IntegerSet (nhiều tied trees).

### b. NNI correction (`-opt_btree`)
Sau main loop, với từng replicate i:
- Đổi alignment sang `boot_samples_pars[i]` (via `modifyPatternFreq()`)
- SPR/NNI search trên `best_tree[i]` → tree tối ưu hơn cho replicate đó
- 2× chậm hơn base MPBoot, nhưng chính xác hơn khi model violation

### c. SIMD vectorization
REPS loop dùng `VectorClassUShort` — load 8 ushort đồng thời, compute dot-product per segment.
Precomputed `boot_samples_pars_remain_bounds[i][seg]` → early exit nếu partial sum đã thua current best.

---

## 4. Key Parameters

| Parameter | Default | Ý nghĩa |
|-----------|---------|---------|
| `gbo_replicates` (B) | 1000 | Số bootstrap replicates (`-bb 1000`) |
| `step_iterations` | 100 | Check convergence mỗi N iterations |
| `min_correlation` | 0.99 | Split correlation threshold để stop |
| `unsuccess_iteration` | ceil((N-1)/100)×100 | Min extra iters sau last improvement |
| `max_iterations` | 10 × N_taxa | Hard cap |
| `ufboot_epsilon` | 0.5 | RELL tolerance cho polytomy correction |
| `ratchet_iter` | user | Ratchet fire mỗi N iters (-1 = off) |
| `ratchet_percent` | user | % informative sites upweighted |
| `popSize` | user | Candidate pool size |
| `cutoff_percent` | 10 | Chỉ giữ top 10% trees vào candidate pool |

---

## 5. Data Flow

```
Khởi tạo (1 lần):
  boot_samples_pars[B][P]  ← multinomial resampling
  parsVect[2N+1][P][STATES] ← Fitch vectors, allocated once

Mỗi main-loop iteration (1 candidate tree):
  _pattern_pars[P]         ← Fitch traversal (per-pattern parsimony)
  REPS loop: B dot-products ← score candidate vs all replicates

Per-replicate (optional, -opt_btree):
  alignment reweighted      ← modifyPatternFreq(boot_samples_pars[i])
  SPR/NNI search            ← full search on replicate alignment
```

**Điểm quan trọng**: `parsVect` (Fitch bit-vectors) **không rebuild per replicate** — chỉ rebuild khi topology thay đổi. Bootstrap chỉ đổi weights, không đổi vectors.

---

## 6. Bootstrap Code Locations (CPU side)

| File | Lines | Function | Mô tả |
|------|-------|----------|-------|
| `iqtree.cpp` | 2379 | `pllInitUFBootData()` | Alloc `boot_samples_pars[B][P]`, `boot_logl[B]`, `boot_trees[B]` |
| `iqtree.cpp` | 214–322 | constructor | Generate B bootstrap replicates (Poisson resampling per site) |
| `iqtree.cpp` | 3281–3459 | `saveCurrentTree()` | **REPS loop**: `rell += _pattern_pars[p] × boot_samples_pars[i][p]` |
| `iqtree.cpp` | 3428–3459 | inside `saveCurrentTree` | Update `best_pars[i]`, `best_tree[i]` for each replicate |
| `iqtree.cpp` | 3803 | `doSegmenting()` | Chia patterns thành segments, precompute remain bounds |
| `iqtree.cpp` | 2485, 2530 | `optimizeBootTrees()` | Post-hoc NNI/SPR per replicate (`-opt_btree`) |
| `iqtree.cpp` | 4323 | `computeBootstrapCorrelation()` | Convergence check (split correlation) |
| `iqtree.cpp` | 4260 | `pllConvertUFBootData2IQTree()` | Convert PLL → IQTree format sau GPU build |
| `sprparsimony.cpp` | 3039 | `_allocateParsimonyDataStructures()` | Alloc parsVect (1 lần, reused across replicates) |
| `sprparsimony.cpp` | 3376–3379 | `makeParsimonyTreeFast()` | Call site of alloc; also called on ratchet |
| `alignment.cpp` | 118 | `modifyPatternFreq()` | Đổi pattern weights cho per-replicate optimization |
| `phyloanalysis.cpp` | 1417 | `runBasicMpbootGpu()` | GPU entry point (hiện chỉ dùng cho initial trees) |

---

## 7. Phân tích: GPU cần làm gì để hỗ trợ Bootstrap?

### Hiện trạng
```
CPU flow với -bb:
  [1] Generate boot_samples_pars[B][P]    (CPU, 1 lần)
  [2] MAIN LOOP iterations:
        [2a] Perturb + SPR/NNI tree search (CPU, per iteration)
               └─ GPU hiện làm bước này (K1+K2 kernel) cho INITIAL trees
                  nhưng sau đó CPU tiếp tục main bootstrap loop
        [2b] _pattern_pars[P] = Fitch traversal of new tree
        [2c] REPS loop: B dot-products (CPU, SIMD vectorized)
        [2d] Update boot_trees[B]          (CPU)
  [3] Optional: optimize best trees per replicate (CPU, per replicate)
  [4] Compute consensus + support values   (CPU)
```

### GPU hiện tại chỉ cover
GPU K1 + K2 thực hiện **initial candidate tree building** (bước 2a của vòng lặp đầu tiên), sau đó trả về CPU. Main bootstrap loop vẫn chạy **hoàn toàn trên CPU**.

### Để tích hợp bootstrap lên GPU cần:

#### Option A — GPU REPS evaluation (ưu tiên cao, dễ nhất)
Sau mỗi K2 iteration build ra K trees, thay vì dùng CPU REPS loop:

```
GPU kernel: REPSKernel<<<B, 32>>>
  Input:  _pattern_pars[P] (mới tính từ current candidate tree)
          d_boot_samples[B × P] (pre-uploaded, constant)
  Output: best_pars[B], best_tree_idx[B] (update nếu better)

  Per warp: tính rell = sum_p(pattern_pars[p] × boot_samples[i][p])
  One warp per replicate (i = blockIdx.x)
  32 lanes chia P patterns: lane j handles p = j, j+32, j+64, ...
  Warp reduction (warpReduceU32) → lane 0 cập nhật best_pars[i]
```

- **B=1000 warps** → 1000 blocks → GPU throughput tốt
- **P patterns** typically 1000–5000 → width = P/32 per lane
- **Memory**: `d_boot_samples` = B × P × 2 bytes = 1000 × 5000 × 2 = 10 MB (fits on GPU)
- **No smem needed** (registers + L2 cache)

#### Option B — GPU bootstrap tree optimization (`-opt_btree`)
Với mỗi replicate i, chạy SPR/NNI search trên bootstrap alignment:

```
Reuse K2 kernel (buildPhase3Kernel) với:
  - d_siteWeights[k] = boot_samples_pars[i] (uploaded per replicate)
  - K = B replicates run in parallel (1 block per replicate)
  - Pool = single best tree per replicate (không cần pool)
```

**Khó hơn** vì:
- Mỗi replicate có tree topology khác nhau → upload B topologies
- Site weights khác nhau per replicate → B × width ints upload
- Feasible nếu B ≤ gpu_worker (currently 400), hoặc batch by B/gpu_worker

#### Option C — GPU main loop (tham vọng nhất)
Port toàn bộ main bootstrap loop lên GPU: perturbation + SPR + REPS update per iteration.
- Về cơ bản là K1+K2 hiện tại nhưng thêm REPS evaluation sau mỗi K2 outer iteration
- Cần pass `d_boot_samples` vào kernel, tích hợp REPS vào `runPhase3`

### Priority order

| Option | Độ khó | Speedup tiềm năng | Rủi ro |
|--------|--------|-------------------|--------|
| A — REPS kernel (separate) | Dễ | Moderate (REPS << search cost) | Thấp |
| B — per-replicate optimization | Trung bình | High (nếu -opt_btree used) | Trung bình |
| C — full GPU main loop | Khó | Cao nhất | Cao |

**Recommendation**: Option A first — standalone `REPSKernel` đánh giá B dot-products per candidate tree, thay thế `saveCurrentTree()` REPS loop. Minimal invasive, verifiable, no change to existing K1/K2.

---

## 8. Cách dùng bootstrap trong CLI

```bash
# Basic: ultrafast bootstrap 1000 replicates
./mpboot -s alignment.phy -bb 1000

# Với NNI correction (post-hoc per-replicate optimization)
./mpboot -s alignment.phy -bb 1000 -opt_btree

# Với ratchet perturbation
./mpboot -s alignment.phy -bb 1000 -ratchet_iter 10 -ratchet_percent 25

# Default (-use_gpu): GPU chỉ build initial trees, bootstrap vẫn CPU
./mpboot -s alignment.phy -bb 1000 -use_gpu
```

**Lưu ý**: `-use_gpu` và `-bb` có thể kết hợp, nhưng GPU hiện chỉ cover initial tree building. REPS evaluation và convergence check vẫn trên CPU.
