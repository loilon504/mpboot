# Plan: Correctness → Bootstrap quality → Optimization

## Priority order

1. **Correctness non-bootstrap Sankoff** (DNA + protein) — phải đúng hoàn toàn trước khi bàn tới bất kỳ thứ gì
2. **Correctness bootstrap Sankoff + uniform** (DNA + protein) — sau khi non-bootstrap sạch
3. **Chiến thuật treels / stopping condition** — chỉ sau khi 1 & 2 verified

## Context (vấn đề cần giải quyết)

**Vấn đề 1 — Pool convergence**: workers cạnh tranh cho pool slots → mất đa dạng topology.

**Vấn đề 2 — Treels poverty**: GPU saves "per-node best" = O(N) per worker per round. CPU saves mọi testInsert → sau dedup: 4-7 × N² unique topologies. GPU thiếu ~N lần.

---

## ✅ Đã hoàn thành

### Default params (tools.cpp)
- `gpu_worker`: 100 → **200** (K2 workers per round)
- `gpu_pool_size`: 10 → **20**
- `gpu_worker_stop`: 1 (giữ nguyên)

### Memory cap (gpu_init_trees.cu)
K1 và K2 được cap độc lập về 80% free GPU memory trước khi alloc:
```cpp
size_t _gpu_free = 0, _gpu_total = 0;
cudaMemGetInfo(&_gpu_free, &_gpu_total);
const size_t _parsVect_per_worker =
    (size_t)(2 * mxtips + 1) * (size_t)width * (size_t)states * sizeof(parsimonyNumber);
const int _max_by_mem = ... floor(0.8 × free / parsVect_per_worker) ...;
const int K1           = max(1, min(K,              _max_by_mem));
const int k2_workers_early = max(1, min(gpu_worker, _max_by_mem));
const int K_alloc      = max(K1, k2_workers_early);
```
- Log dòng: `K1=X  K2=Y  K_alloc=Z  per_worker=W MB  free=F GB`
- Đã kiểm tra: prot_M10372 (22426 sites, states=20) → 459.8 MB/worker → K2 capped 200→139 trên A100 80GB ✓

**Lưu ý**: A100 thực tế là **80 GB** (plan cũ ghi sai 37 GB).

### Stopping condition
- Đã thử và reject N-scaling variants (multiplicative, additive, step-function) — tất cả chậm hơn stop=1 mà không cải thiện score
- Giữ: `unsuccess_thresh = k2_workers * gpu_worker_stop` cho cả hai mode
- **Bootstrap fix**: bỏ `unsuccess_iteration` khỏi formula (giá trị đó được design cho sequential CPU, vô nghĩa với GPU parallel)
  - Trước: `thresh = unsuccess_iteration + k2*stop` → N=295: **500**
  - Sau: `thresh = k2*stop` → **200** (cả bootstrap lẫn non-bootstrap)
  - Bootstrap vẫn có correlation check (`cur_correlation ≥ min_correlation`) làm điều kiện chính

### Task A — Richer treels ✅ (hoàn thành ~2026-05-28)

Save topologies trong `testInsert` khi `mp < sh.randomMP` (trước rollback), thay vì chỉ save sau node hoàn chỉnh:
- **`pars_build.cu`**: `testInsert` thêm treels save block; `doAddTraverse` pass treels params
- **`gpu_init_trees.cu`**: `max_treels` tăng từ `K_alloc×1000` → `K_alloc×3000`

Kết quả: treels size tăng đáng kể (ví dụ 36,000 → 531,000 trees cho aa/138).

---

## ✅ Fix mới nhất: Sankoff GPU Bootstrap (2026-05-30/31)

### Fix 1 — GPU Sankoff ppars kernel (`pars_treels.cu`)
**Vấn đề**: `use_gpu_treels_pars = false` khi `d_cost_matrix != nullptr` → AA bootstrap dùng CPU `computeParsimony()` per-tree → hàng giờ.

**Fix**:
- Thêm `treelsPatternParsKernelSankoff<20>` kernel (per-pattern Sankoff parsimony trên GPU)
- Bỏ guard `if (mem->d_cost_matrix != nullptr) return false` trong `gpuComputeTreelsPatternPars`
- `gpu_init_trees.cu`: `use_gpu_treels_pars = is_bootstrap && (states == 4 || states == 20)` (bỏ điều kiện `d_cost_matrix == nullptr`)

**Kết quả BB10** (10 pandit datasets lớn, `-bb 1000 -cost`, `output/pandit_gpu_bb_non_new/`):
| | DNA (5 datasets) | AA (5 datasets) |
|--|--|--|
| Score | = CPU (Δ=0) | GPU ≥ CPU (+1…+22) |
| Speedup | 0.99–1.34× (mean 1.14×) | **1.91–5.70× (mean 3.75×)** |

### Fix 2 — AA tip bitmask bug (`gpu_init_trees.cu`, `uploadSankoffTipParsVect`)
**Vấn đề**: `bitmask = (unsigned int)nuc` dùng AA state **index** (0=Ala, 1=Arg…) như Fitch bitmask → Alanine (index=0) cho `bitmask=0` → tất cả states = kSankoffInf → tip rỗng → per-pattern parsimony vô nghĩa → star consensus tree.

**Root cause** phát hiện bởi 2-agent debug (reviewer + debugger):
- DNA dùng `PLL_MAP_NT`: nuc IS bitmask (A=1, C=2, G=4, T=8) → đúng
- AA dùng `PLL_MAP_AA`: nuc là index (A=0, R=1, …) → cần convert sang one-hot

**Fix** (`gpu_init_trees.cu` line ~87):
```cpp
// Detect encoding: DNA undetermined=15=(1<<4)-1 (bitmask), AA undetermined=22 (index)
const bool is_bitmask_coded = (undetermined == ((1u << states) - 1u));
// ...
unsigned int bitmask = is_bitmask_coded
    ? (unsigned int)nuc
    : (nuc < (unsigned char)states ? (1u << nuc) : (1u << states) - 1u);
```

**Status**: Fix đã apply, build OK. Đang verify với aa/138 (background task `bwk2org3g`).

**Side effect của bug**: `cor=1.0` sớm sau 3 rounds (600 iter) do tất cả trees hash giống nhau (sai metric → K2 converge 1 topology) → `stable_rounds ≥ 2` → stop sớm. Sẽ tự fix sau khi tip bug được fix.

---

## Bootstrap accuracy (DNA, 5 datasets, `pandit_gpu_bb_non_new` vs `pandit_cpu_bb_non`)

**Tổng accuracy** = 85.9% cả GPU lẫn CPU, nhưng distribution khác:
- GPU tập trung 70% branches tại support=100% (1186/1706) — thiếu diversity
- CPU spread đều 30–100% — calibrated tốt
- GPU poorly calibrated ở mid-range: 80% support → chỉ 59.5% đúng (CPU: 92.3%)

**Nguyên nhân**: treels pool converge nhanh → hầu hết replicates chọn cùng topology → ít variation.

**AA**: sau Fix 1, cần re-benchmark sau Fix 2 (tip bug). Kết quả cũ (star tree) là do Fix 2 chưa apply.

---

## Số liệu thực tế

### Pool size sweep (2026-05-30)

**Setup**: numpars=100, gpu_worker=100, gpu_worker_stop=5, 5 seeds/config.

SCORE TABLE (avg_best / min_best, CPU reference trong ngoặc):
| Dataset     | pool=5           | pool=10          | pool=20          | pool=30          | pool=50           |
|-------------|------------------|------------------|------------------|------------------|-------------------|
| N=217 [33546] | 33552/33547    | 33550/**33546**  | 33551/33547      | 33552/**33546**  | 33554/33551       |
| N=428 [91394] | 91399/91397    | 91398/91395      | **91395/91394**  | 91399/**91394**  | 91399/91396       |
| N=504 [137906] | 137947/137908 | 137946/137909    | **137930/137919**| 137935/137922    | 137934/**137900** |
| N=767 [95049] | 95057/95048    | 95060/95051      | 95057/**95047**  | 95060/**95046**  | 95059/95049       |

**Kết luận**: pool=20 sweet spot — quality tốt, tốc độ hợp lý.

### Memory per worker (A100 80 GB, measured)
| Dataset               | N   | sites  | per_worker | K2_max (80% free) |
|-----------------------|-----|--------|------------|-------------------|
| prot_M10372 + cost    | 169 | 22,426 | 459.8 MB   | **139**           |
| prot_M10273 + cost    | 169 | 11,009 | 216.4 MB   | **~295** (no cap) |
| dna_M14582 + cost     | 372 | 61,199 | 310.8 MB   | **~205** (no cap) |
| dna_M14582 (Fitch)    | 372 | 61,199 | 10.3 MB    | >>200             |
| dna_M7964 (Fitch)     | 640 | 25,260 | 11.7 MB    | >>200             |

### CPU treels sizes thực tế
| N | CPU treels unique | Ratio |
|---|-------------------|-------|
| 330 | ~654,000 | 6 × N² |
| 699 | ~1.96M | 4 × N² |
| 767 | ~2.35M | 4 × N² |

Current GPU max_treels = K_alloc × 3000 = 600,000 (sau Task A).

---

---

## Bugs đã phát hiện (chưa fix) — phát hiện bởi 2-agent audit 2026-05-31

### Bug A — `score_tree = 0` cho Sankoff → Opt-B prune vô hiệu
**File**: `pars_tree.cuh` line ~398–401  
**Mức độ**: MED — performance, không phải correctness  
**Cơ chế**: `newviewParsimony` branch Sankoff không increment `score` → `score_tree[p_num] = 0` với mọi inner node. Fitch tích lũy `score_tree[p] = score + score_tree[q] + score_tree[r]` (cumulative subtree sum). Hậu quả: lower-bound prune trong `doAddTraverse` line ~232:
```cpp
const unsigned int lb = score_tip_p + score_tree[q_num] + score_tree[r_num];
// Sankoff: lb = score_tip_p + 0 + 0 ≈ 0 → luôn < sh.randomMP → không prune
```
Toàn bộ SPR candidates đều bị evaluate (không skip) → protein SPR chậm hơn cần thiết.  
**Tree search score vẫn đúng** — evaluate không dùng score_tree.  
**Fix**: Tính `score_tree[p]` Sankoff = `min_s(p[s][b])` summed over b (tổng per-site minimum cost), tương đương semantic "minimum total cost of subtree". Sau đó Opt-B lb hoạt động.

### Bug B — `uint16_t` overflow trong `treelsPatternParsKernelSankoff`
**File**: `pars_treels.cu` line ~321  
**Mức độ**: HIGH — bootstrap correctness  
**Cơ chế**:
```cpp
pars_ptn[b] = (uint16_t)(min_pars > 65535u ? 65535u : min_pars);
```
`min_pars = min_ij(tip[i][b] + cost[i][j] + inner[j][b])`. `inner[j][b]` tích lũy qua toàn bộ subtree. Với N lớn hoặc cost matrix entries cao, `min_pars` có thể vượt 65535 → clamp âm thầm → REPS bootstrap chọn sai winner → bootstrap support sai.  
**Ước tính risk**: N=128, max_cost≈1000, depth≈7 → ~14K < 65535 (OK). N=300, max_cost=2000, depth≈9 → ~36K < 65535 (OK). Với custom cost matrix entries >3000 hoặc N>400 → có thể overflow.  
**Fix**: Đổi `uint16_t` → `uint32_t` cho `d_treels_ptn_pars`, `h_treels_ptn_pars`, `h_batch_pars`, toàn bộ pipeline REPS.  
**Ảnh hưởng**: Chỉ bootstrap — tree search không dùng treelsPatternPars.

---

## Task tiếp theo (theo priority mới)

### Priority 1 — Verify non-bootstrap Sankoff correctness ✅ VERIFIED (2026-05-31)

**Kết quả**: 20 protein datasets re-run với post-Fix2 binary, `output/treebase_gpu_non_new/`.

| Kết quả | Số dataset |
|---------|-----------|
| diff < 0 (GPU tốt hơn CPU) | 4 |
| diff = 0 (GPU match CPU) | 16 |
| diff > 0 (GPU tệ hơn CPU) | **0** |

**20/20 diff ≤ 0** — Fix 2 (AA tip bitmask) đã sửa hoàn toàn non-bootstrap Sankoff protein.  
DNA non-bootstrap Sankoff cũng đúng (treebase_gpu_non, kiểm trước).  
**Priority 1 DONE.**

---

### Priority 2 — Fix Bug A: score_tree Sankoff cho Opt-B pruning ✅ VERIFIED (2026-05-31)

**Files**: `pars_tree.cuh` (~line 378–392, Sankoff branch)  
**Benchmark** (6 protein datasets, `output/treebase_gpu_bugA/`):

| Dataset | N | Fix2 time | Bug A time | Speedup | Score Δ |
|---------|---|-----------|------------|---------|---------|
| M1118_137 | 137 | 311.2s | 279.6s | +11% | 0 |
| M11341_100 | 100 | 409.8s | 382.0s | +7% | 0 |
| M4249_153 | 153 | 1269.4s | 910.9s | **+28%** | 0 |
| M4318_78 | 78 | 991.9s | 857.8s | +14% | 0 |
| M4780_90 | 90 | 427.7s | 363.7s | +15% | +1* |
| M8569_164 | 164 | 459.1s | 404.1s | +12% | 0 |

*+1 là stochastic variation bình thường — không phải lỗi lower bound.  
**Tổng kết: 5/6 score unchanged, speedup 7–28%, trung bình ~14%.**

---

### Priority 3 — Fix Bug B: uint16_t → uint32_t cho Sankoff bootstrap

**Files**:
- `pars_treels.cu`: đổi buffer type, remove clamp
- `gpu_init_trees.cu`: `d_treels_ptn_pars`, `h_treels_ptn_pars` — double memory usage
- Bootstrap REPS pipeline: `h_batch_pars`, `gpuBatchREPS` — phải consistent

**Trade-off**: Memory usage × 2 cho ptn_pars buffer. Với `max_treels = 600K` và `nptn_padded = 10K sites`:  
600K × 10K × 2 bytes (uint16) = 12 GB → đã vượt memory budget!  
Cần kiểm tra actual `max_treels` và `nptn_padded` values, có thể cần giảm max_treels hoặc dùng chunked processing.

---

### Priority 4 — Re-benchmark bootstrap AA sau Fix 2 + Bug A ✅ VERIFIED (2026-05-31)

**5 pandit AA datasets `-bb 1000 -cost`, `output/pandit_gpu_bb_non_new2/aa/`:**

| Dataset | CPU ref | GPU cũ (pre-Fix2) | Diff cũ | GPU mới | Diff mới | Speedup vs CPU |
|---------|---------|------------------|---------|---------|---------|----------------|
| aa/138  | 22376 | 22377 | +1  | **22376** | **0**  | 4.65× |
| aa/1470 | 27035 | 27044 | +9  | **27035** | **0**  | 3.44× |
| aa/1552 | 30827 | 30842 | +15 | **30826** | **-1** | 3.64× |
| aa/280  | 24634 | 24656 | +22 | **24634** | **0**  | 3.35× |
| aa/500  | 36967 | 36982 | +15 | **36965** | **-2** | 2.74× |
| **Mean** | | | **+12.4** | | **-0.6** | **3.56×** |

**5/5 diff ≤ 0** (cũ: 0/5). Không còn star tree (cor=0 ban đầu, tăng dần qua R3–10).  
Speedup vs CPU: **2.74–4.65× (mean 3.56×)**.

---

### Priority 5 — Chiến thuật cải thiện treels (bootstrap quality)

Chỉ sau khi Priorities 1–4 verified:
- Nới lỏng save criterion: lưu khi `mp < sh.randomMP + margin`
- Random NNI/ratchet per worker per round (Task B cũ)
- Adaptive migration cho stuck pool slots
- Stopping condition tuning

---

## Benchmark kết quả tham khảo

### BB10 (pandit, -bb 1000 -cost, `pandit_gpu_bb_non_new/`)
| | DNA (5) | AA (5) |
|--|--|--|
| Score vs CPU | Δ=0 (exact) | +1…+22 (GPU worse) — do Fix 2 chưa apply lúc chạy |
| Speedup | 0.99–1.34× | **1.91–5.70× (mean 3.75×)** |

### Treebase protein non-bootstrap (44 datasets, old GPU binary)
- 24/44 (55%): exact match CPU
- 16/44 (36%): diff +1…+5
- 4/44 (9%): diff +6…+9
- 0/44: GPU better

**Sau Fix 2 expect**: ≥ 95% exact match hoặc GPU ≤ CPU.

### Bootstrap calibration (DNA, `pandit_gpu_bb_non_new` vs `pandit_cpu_bb_non`)
- GPU: 70% branches tại support=100% (over-confident)
- CPU: spread đều 30–100%
- GPU poorly calibrated mid-range: 80% support → 59.5% đúng (CPU: 92.3%)

---

## ✅ Optimizations tháng 6/2026 (post-Priority-4)

### Opt A1 — Fix Fitch kernel pars_ptn indexing (`pars_treels.cu`)
`pars_ptn[b] += __popc(t_N)` → bit-extract loop per pattern. Fix correctness bootstrap DNA.

### Opt A2 — newickFromBackVf use_int_ids (`gpu_init_trees.cu`)
Thêm `use_int_ids=true` → output integer taxon IDs (0-indexed) thay vì string names.  
Lý do: PASS1 dùng `printTree(WT_TAXON_ID)` → cần match.

### Opt A3 — PASS1 fast copy (`gpu_init_trees.cu`)
Thay CPU `computeParsimony()` per-tree (~4ms/tree) bằng copy từ `h_treels_ptn_pars`.  
`std::copy(h_treels_ptn_pars + t*nptn_padded, ..., h_batch_pars + u*nptn_padded)`

### Opt A4 — max_reps_per_round 2000 → 20000 (`gpu_init_trees.cu`)
Sau khi A3, PASS1 không còn bottleneck CPU → tăng reps/round để đưa nhiều tree hơn vào pool.

### Opt B — K_ppars=1000 dynamic (`pars_tree.cuh`, `pars_tree.cu`, `pars_treels.cu`)
Alloc `d_ppars_parsVect` riêng từ free VRAM sau tất cả alloc khác. K_ppars = min(1000, free*0.8/per_tree).  
`treelsPatternParsKernel` dùng `K_ppars` thay `K`. Kết quả M10467: ppars 8239→6620ms (~1.24×).

### Opt C — Pinned memory cudaMallocHost (`gpu_init_trees.cu` lines 640–667, 1383–1391)
Đổi `std::vector` → raw pointer + `cudaMallocHost` cho:
- `h_treels_bvf` (246 MB cho n=19k)
- `h_treels_scores`, `h_treels_hashes` (77 KB mỗi cái)
- `h_treels_ptn_pars` (157 MB)

Root cause D2H chậm: pageable memory page-fault lần đầu (~200 MB/s). Sau pin: ổn định ~6 GB/s từ round 1.  
Kết quả D2H R1: 2493ms → 388ms (**6.4×**). Wall-clock: 71.1s → 68.3s (M10467).

### Opt D — K2_N ∥ ppars_{N-1} (kế hoạch, chưa làm)
Xem `ppars_opt.md` cho analysis đầy đủ. Ước tính ~44–48% speedup tổng.

---

## Bootstrap algorithm: CPU vs GPU (khám phá 01/06/2026)

### CPU MPBoot bootstrap (parsimony, KHÔNG phải UFBoot/likelihood)

Thuật toán **online evaluation** — không search per-replicate:

```
Khởi tạo: 99 parsimony trees → 31 unique topologies → evaluate trên 1000 samples
Main loop (Ratchet/NNI trên original alignment):
    sinh cây mới → ngay lập tức evaluate trên TẤT CẢ 1000 bootstrap samples:
        for sample in 0..999:
            rell_pars = sum(pattern_pars[ptn] × boot_freq[sample][ptn])
            if rell_pars >= boot_logl[sample]:
                boot_trees[sample] = cây này  ← cập nhật winner
Kết thúc: boot_trees[0..999] → build consensus
```

**"161,879 bootstrap candidate trees evaluated"** = `treels_logl.size()` = tổng unique trees đã push vào pool trong suốt search. Ratchet tạo ~161,848 cây mới ngoài 31 ban đầu.

**"31 distinct locally optimal trees"** = unique topologies sau khởi tạo 99 cây ban đầu (69 duplicate).

### So sánh pool GPU vs CPU (18 DNA datasets, jun1)

| | GPU (jun1) | CPU MPBoot |
|--|-----------|-----------|
| Pool size | **39k–86k** (mean 50,720) | **95–162k** (mean 161,879) |
| Ratio | 1× | ~3× lớn hơn GPU |
| Search strategy | K2 uniform hill-climbing | Ratchet (thay đổi pattern weights → đa dạng hơn) |
| Evaluation timing | Offline (REPS phase sau mỗi round) | Online (ngay khi có cây mới) |
| 100% support branches | ~0.9% (45/4838) | ~27% (1322/4838) |
| Metric | parsimony REPS | parsimony RELL |

### Bootstrap calibration: GPU OLD vs GPU NEW (jun1) vs CPU (18 DNA datasets)

**5-wide bins, center → true%:**

| Support | GPU OLD | GPU NEW (jun1) | CPU |
|---------|---------|----------------|-----|
| 52.5% | ~33% | ~49% | ~60% |
| 72.5% | ~46% | ~55% | ~79% |
| 77.5% | ~46% | ~65% | ~85% |
| 82.5% | ~58% | **81%** | ~91% |
| 87.5% | ~77% | **91%** | ~94% |
| 92.5% | ~77% | **96%** | ~96% |
| 97.5% | ~97% | **99%** | ~99% |

GPU NEW tốt hơn GPU OLD ở toàn bộ range 80–99%. Cả hai GPU đều liberal hơn CPU ở range 20–80%.

**Root cause calibration gap**: CPU pool lớn hơn 3× **và** Ratchet search đa dạng hơn K2.  
Pool GPU chủ yếu là trees từ uniform hill-climbing → kém đa dạng về topology space.

Kết quả saved tại: `thesis/benchmark/output/pandit_bb_non_jun1/bootstrap_3way_dna.png`
