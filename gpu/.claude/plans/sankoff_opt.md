# Sankoff GPU: Bug Fixes & Performance Optimizations

## Status Overview

| Item | Status | Effect |
|------|--------|--------|
| Pattern set + encoding fix | ✅ DONE (gpu_init_trees.cu) | GPU reaches 6662 = CPU ✅ |
| Ratchet for Sankoff | ✅ DONE (2026-05-20) | K=200 đạt 6662 = CPU (was 6664) ✅ |
| Opt 1: Memory layout [ptn][state] → [state][ptn] | ✅ DONE (2026-05-20) | Sankoff 2.1×, Fitch 2.5× ✅ |
| Opt 3: Remove weighted newview in Sankoff | ✅ DONE (2026-05-20, combined w/ Opt 1) | Cleaned up dead code ✅ |
| Opt 4: Register preload lv_[]/rv_[] | ✅ DONE (2026-06-07) | Protein 1.4–2.3× ms/tree ✅ |
| Opt 5: Cost matrix → CUDA constant memory | ✅ DONE (2026-06-07) | Protein 1.39–1.83× vs Opt4; DRAM=0% (NCU) ✅ |
| ~~Identity matrix O(S)~~ | DROPPED | Not used in practice |

### Benchmark timeline (tree1.phy / dna.cost, K=200, sprdist=6, gpu_worker=200)

| Checkpoint | ms/tree | BEST SCORE |
|-----------|---------|-----------|
| Baseline (pre-ratchet) | ~168 | unstable |
| Post-ratchet | 346 | 6662 ✅ |
| Post-Opt1 | 167 | 6662 ✅ |
| Post-Opt4 (register preload, S=4) | ~130 | 6662 ✅ |
| Post-Opt5 (constant memory, S=4) | ~111 | 6662 ✅ |

Fitch post-Opt1 (K=400): ~6.4 ms/tree (was ~16 ms/tree)

---

## Bug Fix (DONE): Pattern Set Mismatch + PLL Encoding

**Root cause (primary)**: `uploadSankoffTipParsVect` used `width = iqtree.aln->size() = 1400`
(all unique patterns), but CPU `compressSankoffDNA` only processes `parsimonyLength = 1072`
*informative* patterns via PLL's `isInformative()`. This width mismatch caused wrong scores.

**Root cause (secondary)**: Tip data was read from `iqtree.aln->at(ptn)` using IQTree's
character encoding, but should be read from PLL's `tr->yVector` using PLL bitmask encoding.
Also, `extern Params *globalParam` was declared inside `namespace mpbootgpu` → linker error.

**Verification** (2026-05-20):
- K=200, sprdist=6: BEST SCORE 6664 (not yet converged, stochastic)
- K=1000, sprdist=6, gpu_worker_stop=3: 6663
- K=5000, sprdist=6, gpu_worker_stop=3, gpu_pool_size=20: **6662 = CPU** ✅
- Fitch unchanged: 6662 ✅

**Fix**: Rewrote `uploadSankoffTipParsVect` in `gpu/src/gpu_init_trees.cu`:
- `width = parsimonyLength` (not `iqtree.aln->size()`) — both Fitch and Sankoff now identical
- Read from `tr->yVector[tipNum][i]` for `i` in `[lower, upper)` (PLL bitmask encoding)
- Apply same `isInformative()` logic as CPU: count distinct bitmask values `< undetermined`; skip if ≤ 1
- Moved `extern Params *globalParam` to **before** `namespace mpbootgpu {` (was causing linker error)

**PLL bitmask encoding** (DNA): A=1 (bit0), C=2 (bit1), G=4 (bit2), T=8 (bit3).
State s compatible iff `(bitmask >> s) & 1`. Undetermined (gap/N) = `(1<<states)-1 = 15`.

Current `uploadSankoffTipParsVect` (in `gpu/src/gpu_init_trees.cu`):
```cpp
// Before namespace mpbootgpu:
extern Params *globalParam;

static void uploadSankoffTipParsVect(
    GpuParsimonyMem* mem, const pllInstance* tr, const partitionList* pr, cudaStream_t stream)
{
    const int N = mem->mxtips;
    const int P = mem->width;  // = parsimonyLength (informative patterns)
    const int states = mem->states;
    const int lower = (int)pr->partitionData[0]->lower;
    const int upper = (int)pr->partitionData[0]->upper;
    const unsigned int undetermined = (1u << states) - 1u;
    std::vector<parsimonyNumber> h_buf(mem->parsVectPerTree, (parsimonyNumber)kSankoffInf);

    const bool all_informative = globalParam && !globalParam->sort_alignment;
    int ptn_gpu = 0;
    for (int i = lower; i < upper && ptn_gpu < P; ++i) {
        if (!all_informative) {
            bool seen[256] = {};
            for (int j = 1; j <= N; ++j)
                seen[(unsigned char)tr->yVector[j][i]] = true;
            int counter = 0;
            for (unsigned int v = 0; v < undetermined; ++v)
                if (seen[v]) counter++;
            if (counter <= 1) continue;  // uninformative: skip
        }
        for (int tipNum = 1; tipNum <= N; ++tipNum) {
            unsigned int bitmask = (unsigned int)(unsigned char)tr->yVector[tipNum][i];
            parsimonyNumber* out = &h_buf[(size_t)tipNum * P * states + (size_t)ptn_gpu * states];
            for (int s = 0; s < states; ++s)
                out[s] = ((bitmask >> s) & 1u) ? 0u : (parsimonyNumber)kSankoffInf;
        }
        ptn_gpu++;
    }
    // upload to all K GPU trees ...
}
```

**Why kSankoffInf works** (not `highest_cost`): For any cost matrix with bounded costs << kSankoffInf,
`min(valid_cost, kSankoffInf + c) = valid_cost` always holds — so incompatible states are
correctly excluded by the min-over-j operation in Sankoff newview/evaluate.

**Note on "informative-only" approach**: The earlier idea of using IQTree's
`n_informative_patterns=1055` was a different criterion (not used in the PLL parsimony path).
What we DO use is PLL's `isInformative()` with `parsimonyLength=1072` — which is exactly what
CPU `compressSankoffDNA` uses.

---

## Ratchet for GPU Sankoff (DONE — 2026-05-20)

### Problem

`iter_is_nni = sh.use_sankoff || (blockIdx.x % 2 == 0)` — tất cả Sankoff workers đều làm NNI,
không có ratchet. Root cause:
- Ratchet Fitch ghi `sw_k[b] = {1,2}` trực tiếp vào `d_siteWeights` rồi restore về `nullptr`
- Với Sankoff: `d_siteWeights` lưu **pattern frequencies gốc** (non-uniform, từ `informativePtnWgt`)
  → overwrite = xóa frequencies; restore về `nullptr` = uniform weight = sai

### Fix

Thêm `d_ratchetScratch[K][width]` làm scratch buffer riêng cho Sankoff ratchet.
Khi ratchet: `ratchet_k[b] = sw_k[b] * {1 hoặc 2}` — `sw_k` (frequencies gốc) không bị đụng.
Sau ratchet: `sh.site_weights = sw_k` (restore về frequencies gốc).

### Files modified

| File | Thay đổi |
|------|---------|
| `gpu/include/pars_tree.cuh` | +1 field `unsigned int* d_ratchetScratch` trong `GpuParsimonyMem` |
| `gpu/src/pars_tree.cu` | alloc `d_ratchetScratch` (Sankoff only, same size as `siteWeightsBytes`); free trong `gpuParsimonyMemFree` |
| `gpu/src/pars_build.cu` | (a) `runPhase3` sig: +`ratchet_k`; (b) `iter_is_nni`: bỏ `sh.use_sankoff \|\|`; (c) ratchet branch: Sankoff dùng `ratchet_k`, Fitch dùng `sw_k`; (d) `buildPhase3Kernel` sig: +`d_ratchetScratch`; (e) kernel body: tính `ratchet_k`; (f) `runPhase3` call: +`ratchet_k` |
| `gpu/src/gpu_init_trees.cu` | K2 launch: +`mem->d_ratchetScratch` |

### Ratchet else-branch (pars_build.cu)

```cuda
if (sh.use_sankoff)
{
    // Multiply original pattern frequencies by {1,2} — sw_k stays untouched
    for (int b = 0; b < width; b++)
        ratchet_k[b] = sw_k[b] * ((gpuRandum(&sh.seed) < 0.5) ? 2u : 1u);
    sh.site_weights = ratchet_k;
}
else
{
    // Fitch: write {1,2} into sw_k (uniform baseline, no real freqs to preserve)
    for (int b = 0; b < width; b++)
        sw_k[b] = (gpuRandum(&sh.seed) < 0.5) ? 2u : 1u;
    sh.site_weights = sw_k;
}
// ... SPR1 ...
sh.site_weights = sh.use_sankoff ? sw_k : nullptr;  // restore
// ... SPR2 ...
```

### Verification (2026-05-20)

```bash
# Solo Sankoff (K=200, gpu_device=2):
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 -cost e 2>&1 | grep "BEST SCORE\|ms/tree"
# → 346.40 ms/tree,  BEST SCORE FOUND : 6662  ✅ (= CPU, ngay ở K=200!)
# Pre-ratchet (K=200): 6664, ~168 ms/tree → cần K=5000 mới đạt 6662

# Fitch regression (K=200):
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 2>&1 | grep "BEST SCORE\|ms/tree"
# → 16.28 ms/tree,  BEST SCORE FOUND : 6662  ✅ (Fitch không đổi)
```

**ms/tree tăng ~2× (168→346)** — expected: ratchet odd workers chạy thêm 1 re-evaluate +
1 SPR pass so với NNI. Trade-off hợp lý vì convergence nhanh hơn rất nhiều.

### Dev build note

`CMakeLists.txt` line 41: `sm_86` + `sm_89` tạm comment để build nhanh hơn ~3×:
```cmake
# set(CMAKE_CUDA_ARCHITECTURES "80;86;89" ...)  # full (tắt khi dev)
set(CMAKE_CUDA_ARCHITECTURES "80" ...)           # A100 only
```
Nhớ bật lại khi release.

---

## Opt 1: Memory Layout [ptn][state] → [state][ptn]

### Problem

Current GPU layout: `base[b * STATES + s]` where `b = lane + k*32`
- Lane 0 reads addr: `base + 0*4 + s`, Lane 1: `base + 1*4 + s` → stride = 4 words between lanes
- 128-byte cache line covers lanes 0..7 → need 4 cache lines for 32 lanes = **4× bandwidth waste**

New layout: `base[s * width + b]`
- Lane 0 reads `base + s*width + 0`, Lane 1: `base + s*width + 1` → stride = 1 word
- **Perfectly coalesced** — 1 cache line per 32 lanes

### Changes

#### A. `gpu/include/pars_tree.cuh` — kernel accesses (~line 68, 354–460)

Update struct comment:
```cpp
// OLD: // parsVect GPU layout: [tree][node][block][state]  i.e. [tree][node][block][state]
// NEW: // parsVect GPU layout: [tree][node][state][block]  i.e. [tree][node][state][block]
```

In `newviewParsimony` — Fitch branch (lines ~354–368):
```cpp
// OLD: q_base[b * STATES + s], r_base[b * STATES + s], p_base[b * STATES + s]
// NEW: q_base[(size_t)s * width + b], r_base[(size_t)s * width + b], p_base[(size_t)s * width + b]
```

In `newviewParsimony` — Sankoff newview branch (lines ~383–389):
```cpp
// OLD: q_base[b * STATES + jj], r_base[b * STATES + jj], p_base[b * STATES + ii]
// NEW: q_base[(size_t)jj * width + b], r_base[(size_t)jj * width + b], p_base[(size_t)ii * width + b]
```

In `newviewParsimony` — Sankoff evaluate branch (lines ~447–452):
```cpp
// OLD: q_base[b * STATES + ii], r_base[b * STATES + jj]
// NEW: q_base[(size_t)ii * width + b], r_base[(size_t)jj * width + b]
```

#### B. `gpu/src/pars_tree.cu` — `uploadTipParsVect` (lines ~254–336)

Fitch branch: after layout change, GPU layout `[state][block]` == CPU layout `[state][block]`
→ **no reorder needed anymore**. Replace triple-loop with simple memcpy:
```cpp
for (int tipNum = 1; tipNum <= N; ++tipNum) {
    memcpy(
        &h_buf[(size_t)tipNum * parsimonyLength * states],
        cpu_pars + (size_t)tipNum * parsimonyLength * states,
        (size_t)parsimonyLength * states * sizeof(parsimonyNumber)
    );
}
```
Sankoff branch in this function: dead code (never called — `uploadSankoffTipParsVect` in
`gpu_init_trees.cu` handles Sankoff). Remove the else block entirely.

#### C. `gpu/src/gpu_init_trees.cu` — `uploadSankoffTipParsVect`

Change h_buf indexing from `[ptn][state]` to `[state][ptn]` (already uses `tr->yVector` + isInformative after the bug fix).
Full new loop (Opt 1 layout applied on top of the already-fixed function):
```cpp
// Outer loops unchanged (tr->yVector + isInformative filter)
// Change only the h_buf write index:
// OLD: out = &h_buf[tipNum * P * states + ptn_gpu * states]; out[s] = ...
// NEW:
for (int s = 0; s < states; ++s)
    if ((bitmask >> s) & 1u)
        h_buf[(size_t)tipNum * P * states + (size_t)s * P + ptn_gpu] = 0u;
    // else stays kSankoffInf (already initialized)
```

### Verification (2026-05-20)

```bash
# Fitch K=400 (shows stable 6662 at more trees):
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 400 -sprdist 6 -gpu_device 2 -gpu_worker 400 \
    -gpu_pool_size 20 -gpu_worker_stop 3 2>&1 | grep "BEST SCORE\|ms/tree"
# → ~6.4 ms/tree, BEST SCORE 6662 ✅  (was ~16 ms/tree pre-Opt1, speedup ~2.5×)

# Sankoff K=200:
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 -cost e 2>&1 | grep "BEST SCORE\|ms/tree"
# → 167 ms/tree, BEST SCORE 6662 ✅  (was 346 ms/tree pre-Opt1, speedup ~2.1×)
```

### Expected speedup
- Theoretical: 4× (perfectly coalesced vs 4× waste)
- Achieved: Fitch ~2.5×, Sankoff ~2.1× (limited by other ops, L2 cache)
- Applies to both Fitch and Sankoff kernels

---

## Opt 3: Remove Weighted Score Accumulation from Sankoff Newview

### Problem

In Sankoff newview (`pars_tree.cuh:~374–393`):
```cuda
unsigned int min_site = kSankoffInf;
for (ii): ... min_site = min(min_site, cost_ii);
score += (sw ? sw[b] : 1u) * min_site;   // dead code
```
`score_tree[p_num] = score;` is set but **never read** in Sankoff evaluate.
Sankoff evaluate computes `min_edge` directly from parsVect values — not score_tree.

### Fix

In Sankoff newview branch: remove `min_site` tracking and `score +=`:
```cuda
// Sankoff newview: only compute and store cost values (no score accumulation)
#pragma unroll
for (int ii = 0; ii < STATES; ++ii) {
    unsigned int best_left = kSankoffInf, best_right = kSankoffInf;
    #pragma unroll
    for (int jj = 0; jj < STATES; ++jj) {
        unsigned int c = cm[ii * STATES + jj];
        unsigned int lv = (unsigned int)q_base[(size_t)jj * width + b];
        unsigned int rv = (unsigned int)r_base[(size_t)jj * width + b];
        best_left  = min(best_left,  lv + c);
        best_right = min(best_right, rv + c);
    }
    p_base[(size_t)ii * width + b] = (parsimonyNumber)(best_left + best_right);
}
// score_tree[p_num] left at 0 (never read in Sankoff evaluate)
```

### Implementation note (2026-05-20)

Opt 3 was implemented **combined with Opt 1** — the Sankoff newview branch was rewritten
in place in `pars_tree.cuh` to use new `[state][block]` indexing AND removed `min_site`/`score +=`
in the same edit.

### Effect
- Eliminates dead computation (1 min-reduce + 1 multiply-add per pattern per newview)
- Absorbed into Opt 1 rewrite — no separate measurable delta

---

## Files to Modify

| File | Change |
|------|--------|
| `gpu/include/pars_tree.cuh` | Opt 1: 7 access sites (Fitch + Sankoff newview + evaluate); Opt 3: remove min_site+score in Sankoff newview; struct comment |
| `gpu/src/pars_tree.cu` | Opt 1: Fitch uploadTipParsVect → memcpy, remove dead Sankoff else block |
| `gpu/src/gpu_init_trees.cu` | Opt 1+fix combined: new h_buf indexing in uploadSankoffTipParsVect |

---

## Verification

Build:
```bash
cd build && make -j8 2>&1 | tail -5
```

Sankoff correctness check:
```bash
# Quick check (K=200) — confirmed 2026-05-20 post-Opt1:
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 -cost e 2>&1 | grep "BEST SCORE\|ms/tree"
# BEST SCORE FOUND : 6662  ✅ (= CPU)
# 167 ms/tree (post-Opt1, vs 346 ms/tree post-ratchet-pre-Opt1 = 2.1× speedup)
```

CPU reference (confirmed 2026-05-20):
```bash
./mpboot-avx -s ../data_debug/tree1.phy -seed 1 -cost e 2>&1 | grep "BEST SCORE"
# BEST SCORE FOUND : 6662  ✅
```

Fitch mode regression check (confirmed 2026-05-20):
```bash
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 2>&1 | grep "BEST SCORE\|ms/tree"
# BEST SCORE FOUND : 6662  ✅  (Fitch unchanged)
```

General Sankoff with dna.cost (post-Opt1):
```bash
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 -cost ../output/dna.cost 2>&1 | grep "BEST SCORE\|ms/tree"
# Expected: no regression, ms/tree improves after Opt 1
```

---

## Opt 4: Preload Child parsVect Columns vào Registers (DONE — 2026-06-07)

### Problem

Trong Sankoff newview (post-Opt1 layout `[state][block]`), vòng lặp lồng nhau:

```cuda
for (int ii = 0; ii < STATES; ++ii) {
    for (int jj = 0; jj < STATES; ++jj) {
        unsigned int c  = cm[ii * STATES + jj];      // global
        unsigned int lv = q_base[jj * width + b];    // GLOBAL — repeated S times!
        unsigned int rv = r_base[jj * width + b];    // GLOBAL — repeated S times!
        best_left  = min(best_left,  lv + c);
        best_right = min(best_right, rv + c);
    }
    p_base[ii * width + b] = best_left + best_right;
}
```

`q_base[jj*width+b]` và `r_base[jj*width+b]` với **cùng jj, cùng b** được load **S lần** (mỗi vòng ii một lần) → tổng S² loads cho q và r, trong khi chỉ cần S giá trị phân biệt.

Tương tự ở evaluate: `r_base[jj*width+b]` bị load S lần trong vòng ii.

**Số loads per pattern:**

| Path | S=4 current | S=4 → optimized | S=20 current | S=20 → optimized |
|------|-------------|-----------------|--------------|------------------|
| newview q | 16 | **4** | 400 | **20** |
| newview r | 16 | **4** | 400 | **20** |
| evaluate r | 16 | **4** | 400 | **20** |

### Fix — newview Sankoff branch (`pars_tree.cuh` ~line 387)

```cuda
// Preload jj columns once — eliminates S-fold redundant global loads
unsigned int lv[STATES], rv_[STATES];
#pragma unroll
for (int jj = 0; jj < STATES; ++jj) {
    lv[jj]  = (unsigned int)q_base[(size_t)jj * width + b];
    rv_[jj] = (unsigned int)r_base[(size_t)jj * width + b];
}
unsigned int min_cost_b = kSankoffInf;
#pragma unroll
for (int ii = 0; ii < STATES; ++ii) {
    unsigned int best_left = kSankoffInf, best_right = kSankoffInf;
    #pragma unroll
    for (int jj = 0; jj < STATES; ++jj) {
        unsigned int c = cm[ii * STATES + jj];
        best_left  = min(best_left,  lv[jj]  + c);
        best_right = min(best_right, rv_[jj] + c);
    }
    unsigned int val = best_left + best_right;
    p_base[(size_t)ii * width + b] = (parsimonyNumber)val;
    min_cost_b = min(min_cost_b, val);
}
score += (min_cost_b < kSankoffInf) ? min_cost_b : 0u;
```

### Fix — evaluate Sankoff branch (`pars_tree.cuh` ~line 454)

```cuda
// Preload r columns once
unsigned int rv_[STATES];
#pragma unroll
for (int jj = 0; jj < STATES; ++jj)
    rv_[jj] = (unsigned int)r_base[(size_t)jj * width + b];

unsigned int min_edge = kSankoffInf;
#pragma unroll
for (int ii = 0; ii < STATES; ++ii) {
    unsigned int qi = (unsigned int)q_base[(size_t)ii * width + b];
    #pragma unroll
    for (int jj = 0; jj < STATES; ++jj) {
        unsigned int c = cm[ii * STATES + jj];
        min_edge = min(min_edge, qi + c + rv_[jj]);
    }
}
score += (sw ? sw[b] : 1u) * min_edge;
```

**Chi phí extra registers**: `lv[S] + rv_[S]` = 2×STATES per lane per b-iteration.
- S=4: 8 extra — không đáng kể
- S=20: 40 extra — theo dõi occupancy sau khi build

**Files**: `gpu/include/pars_tree.cuh` (newview + evaluate Sankoff branches)

### Kết quả thực tế (2026-06-07)

| Dataset | N | width | Opt1 ms/tree | Opt4 ms/tree | Speedup |
|---------|---|-------|-------------|-------------|---------|
| prot_M10236 | 59 | 164 | 71.62 | 40.53 | 1.77× |
| prot_M10866 | 88 | 3329 | 1905.14 | 1051.96 | 1.81× |
| prot_M11595 | 66 | 463 | 213.62 | 87.15 | 2.45× |
| prot_M11740 | 138 | 4427 | 8220.64 | 3569.13 | 2.30× |
| prot_M3807 | 82 | 591 | 530.14 | 229.36 | 2.31× |

**NCU sau Opt-4 (buildPhase3Kernel<20,800>, prot_M10236):**
- Registers/thread: **255** (hardware max A100) — full unroll của 20×20 loop + 40 live regs lv_[]+rv_[]
- Stack size: **4096 bytes** (register spill — NVCC refuses to inline, bumped stack)
- Theoretical occupancy: **12.5%** (register-limited: 8 blocks/SM)
- DRAM throughput: **43%** (cost matrix still in global memory)

---

## Opt 5: Cost Matrix vào CUDA Constant Memory (DONE — 2026-06-07)

### Problem

`d_cost_matrix` được alloc bằng `cudaMalloc` (`pars_tree.cu:144`):
```cpp
CUDA_CHECK(cudaMalloc(&mem->d_cost_matrix, costBytes));
CUDA_CHECK(cudaMemcpy(mem->d_cost_matrix, cost_matrix, ...));
```

Kernel truy cập `cm[ii*STATES+jj]` qua global memory → L2 cache. Với S=20: 400 entries = 1600 bytes, gây cache pressure. Với S=4: 64 bytes nhỏ nhưng vẫn là L2 truy cập.

**CUDA constant memory**: 64 KB dedicated, hardware broadcast khi all 32 lanes đọc same address → zero additional latency.

### Fix

**Bước A** — `gpu/include/pars_tree.cuh` (trước namespace):
```cuda
static constexpr int kMaxSankoffStates = 20;
extern __constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
```

**Bước B** — `gpu/src/pars_tree.cu`:
```cpp
__constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
```

Trong `gpuParsimonyMemAlloc()`, thay cudaMalloc:
```cpp
// OLD:
CUDA_CHECK(cudaMalloc(&mem->d_cost_matrix, costBytes));
CUDA_CHECK(cudaMemcpy(mem->d_cost_matrix, cost_matrix, costBytes, cudaMemcpyHostToDevice));
// NEW:
CUDA_CHECK(cudaMemcpyToSymbol(g_sankoff_cm, cost_matrix, costBytes));
mem->d_cost_matrix = nullptr;
```

Bỏ `cudaFree(mem->d_cost_matrix)` — constant memory là static.

**Bước C** — `pars_tree.cuh` kernel: thay `sh.cost_matrix` bằng `g_sankoff_cm`:
```cuda
const unsigned int* cm = sh.use_sankoff ? g_sankoff_cm : nullptr;
```

**Lưu ý**: `cudaMemcpyToSymbol` upload vào device hiện tại → gọi sau `cudaSetDevice(params.gpu_device)`.

**Files**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_tree.cu`, `gpu/src/pars_build.cu`

### Bugs gặp khi implement Opt-5

1. **NVCC redefinition**: `pars_build.cu` include header → 2 definitions của `g_sankoff_cm`. Fix: `#ifdef PARS_BUILD_DEFINE_CM` conditional — chỉ `pars_build.cu` define.
2. **`sh.use_sankoff = false`**: Sau khi set `mem->d_cost_matrix = nullptr`, kernel check `d_cost_matrix != nullptr` trả về false → Fitch mode → score sai. Fix: alloc 4-byte sentinel `cudaMalloc(&mem->d_cost_matrix, 4)` chỉ để giữ `!= nullptr`.
3. **`cuda_runtime_api.h` not found**: `iqtree.cpp` → `pars_bootstrap.cuh` → header CUDA không tìm thấy bởi clang++. Fix: `#ifdef __CUDACC__` guard, forward-declare `cudaStream_t` cho non-CUDA TUs.

### Kết quả thực tế (Opt4 → Opt5)

| Dataset | N | width | Opt4 ms/tree | Opt5 ms/tree | Speedup | vs GPU cũ |
|---------|---|-------|-------------|-------------|---------|-----------|
| prot_M10236 | 59 | 164 | 40.53 | **25.66** | 1.58× | **2.79×** |
| prot_M10866 | 88 | 3329 | 1051.96 | **574.74** | 1.83× | **3.32×** |
| prot_M11595 | 66 | 463 | 87.15 | **62.45** | 1.39× | **3.42×** |
| prot_M11740 | 138 | 4427 | 3569.13 | **2074.64** | 1.72× | **3.96×** |
| prot_M3807 | 82 | 591 | 229.36 | **137.40** | 1.67× | **3.86×** |

**DNA Sankoff (Opt4+5 vs GPU cũ):**

| Dataset | N | width | GPU cũ ms/tree | Opt5 ms/tree | Speedup |
|---------|---|-------|---------------|-------------|---------|
| dna_M10243 | 203 | 1771 | 93.49 | 54.71 | 1.71× |
| dna_M10467 | 202 | 4074 | 386.75 | 263.45 | 1.47× |
| dna_M1838 | 228 | 1131 | 162.40 | 111.51 | 1.46× |
| dna_M214 | 295 | 1836 | 280.90 | 168.82 | 1.66× |
| dna_M12051 | 699 | 6914 | 4751.77 | 4142.43 | 1.15× |

**NCU sau Opt-5 (buildPhase3Kernel<20,800>, prot_M10236):**
- Registers/thread: **255** (hardware max)
- Stack size: **4096 bytes** (spill từ Opt-4)
- Theoretical occupancy: **12.5%** (8 blocks/SM, register-limited)
- **DRAM throughput: 0.00%** ← constant memory hoạt động, không còn global DRAM traffic cho cost matrix
- Warp Cycles/Issued Instruction: **2.93** (low stall)
- SM Busy: **15.19%**, Issue Slots Busy: **8.51%** ← bottleneck chính: register-limited occupancy

**Bootstrap Sankoff GPU Opt-5 vs CPU (wall time):**

| Dataset | CPU wall | GPU Opt-5 wall | vs CPU |
|---------|---------|---------------|--------|
| prot_M10236 | 2:37 | 0:11 | **13.7×** |
| prot_M11595 | 8:20 | 0:31 | **16.1×** |
| prot_M3807 | 15:07 | 0:59 | **15.3×** |
| prot_M10866 | 1:07:01 | 6:09 | **10.9×** |
| prot_M11740 | 4:08:45 | 14:27 | **17.2×** |

---

## Phân Tích Bottleneck Hiện Tại (Post-Opt4+5)

### NCU insights (prot_M10236, buildPhase3Kernel<20,800>)

```
Registers/thread  = 255  (hardware max → compiler cannot add more)
Stack size        = 4096 bytes  (local mem spill từ S=20 unroll)
Blocks/SM         = 8   (65536 regs / (255×32) ≈ 8)
Theoretical occ.  = 12.5% (8 warps / 64 max warps/SM on A100)
SM Busy           = 15.19%
Issue Slots Busy  = 8.51%  ← rất thấp, nhiều stall
DRAM throughput   = 0.00%  ← constant memory hoàn toàn loại bỏ cost matrix traffic
Warp Stall        = ~66% cycles wasted
```

**Root cause**: Chỉ có 8 warps/SM. Để hide 40-cycle memory latency cần ≥ 16-32 warps. Với 8 warps, GPU không có warp khác để chạy trong lúc chờ → idle stall.

### Hạn chế của Opt-4: register ceiling

Với S=20 và `#pragma unroll` full:
- `lv_[20]` + `rv_[20]` = **40 extra registers** per thread
- NVCC không thể inline `newviewParsimony` vào kernel vì tổng vượt 255 → compiler bump stack=4096
- Không thể tăng STATES → không thể unroll tốt hơn

---

## Các Optimization Còn Lại

### Opt-6: Giảm register pressure → tăng occupancy (HIGH PRIORITY)

**Vấn đề**: 255 regs, 12.5% occupancy. Cần < 128 regs để đạt 25% occupancy (2× blocks/SM).

**Hướng A — Partial unroll (lv_/rv_ tile)**:
Thay vì preload toàn bộ STATES=20 values vào registers, xử lý theo tiles jj=[0..T):

```cuda
// Tile T = 4 (chỉ 8 extra regs thay vì 40)
for (int jt = 0; jt < STATES; jt += 4) {
    unsigned int lv_t[4], rv_t[4];
    #pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
        lv_t[jj] = (unsigned int)q_base[(size_t)(jt+jj)*width + b];
        rv_t[jj] = (unsigned int)r_base[(size_t)(jt+jj)*width + b];
    }
    #pragma unroll
    for (int ii = 0; ii < STATES; ++ii) {
        #pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            unsigned int c = cm[ii*STATES + jt+jj];
            best_left[ii]  = min(best_left[ii],  lv_t[jj] + c);
            best_right[ii] = min(best_right[ii], rv_t[jj] + c);
        }
    }
}
```

Trade-off: 5× nhiều global reads (5 tile passes × 4 values) vs full preload (1 pass × 20 values). Nhưng reads đã coalesced → L1 cache hit → cheap.

**Hướng B — `__launch_bounds__(32, 16)` trên kernel**:
Báo với compiler target 16 blocks/SM → compiler cố gắng giữ regs ≤ 128. Có thể khiến compiler dùng local memory thêm cho thứ khác, hoặc không inline `newviewParsimony`.

**Hướng C — Shared memory cho lv_/rv_**:
`__shared__ uint s_lv[32][20]` + `s_rv[32][20]` = 5120 bytes extra smem/block. Với NTAXA=128 (smem=3984 + 5120 = 9104 bytes), A100 có thể chạy 163840/9104 ≈ 18 blocks/SM. Ít regs hơn → 16 blocks/SM → 25% occupancy.
Trade-off: smem access ~4 cycles vs register 1 cycle → có thể chậm hơn.

**Ước tính speedup**: nếu occupancy tăng 12.5% → 25%: **1.5–2× throughput** cho compute-bound phase.

---

### Opt-7: SPR Incremental Score Update (MEDIUM PRIORITY)

**Vấn đề**: Sau mỗi `applyMove`, `recomputeAllNodes` cập nhật toàn bộ (2N-2) nodes. Nhưng 1 SPR move chỉ ảnh hưởng đến các tổ tiên của 3 nodes (remove_node, insert_edge endpoints). Chỉ cần recompute O(depth) ≈ O(log N) nodes thay vì O(N).

**Điều kiện**: Cần tracking dirty bits per node. Khi `applyMove(p, q_ins)`:
- Dirty: p, tất cả tổ tiên của p dọc theo đường tới root
- Clean: các subtree không bị touch

**Expected speedup**: Với N=138 (prot_M11740), trung bình recompute từ 272 nodes → ~14 nodes (log2(138)≈7, nhân 2). Nếu SPR phase chiếm 60% runtime: speedup lên tới **5–10×** cho SPR phase, **3–5×** total.

**File**: `gpu/src/pars_build.cu` — hàm `recomputeAllNodes` và vòng SPR chính.

---

### Opt-8: Batch testInsert — Distribute Edges Across Lanes (LOW PRIORITY, COMPLEX)

**Vấn đề**: `doAddTraverse` xử lý candidate insertion edges tuần tự (1 edge per warp per SPR test). Với sprDist=6 và N=138: ~300 candidate edges per SPR move.

**Ý tưởng**: Mỗi lane trong warp xử lý 1 candidate edge song song → 32× speedup cho testInsert phase. Nhưng cần xử lý conflicts khi nhiều lanes tìm cùng best move.

**Complexity**: Rất cao — phải restructure toàn bộ SPR traversal. Không phù hợp thesis hiện tại.

---

### Opt-9: STATES=20 với f16 computation (EXPERIMENTAL)

**Ý tưởng**: Dùng FP16 (half precision) cho cost matrix lookups và parsVect. FP16 GEMM throughput của A100 là 312 TFLOPS vs 19.5 TFLOPS FP32 (16×). Sankoff là min-plus, không phải multiply-add, nhưng có thể exploit FP16 bandwidth.

**Rủi ro**: Độ chính xác, overflow, incompatibility với current logic. Nghiên cứu thêm cần thiết.

---

## Verification Hiện Tại

```bash
# Build
cd /raid/home/loinguyen/workspace/mpboot-gpu/build && make -j8 2>&1 | tail -5

# DNA Sankoff correctness (BEST SCORE phải = 6662)
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -gpu_pool_size 20 -gpu_worker_stop 3 -cost ../output/dna.cost \
    2>&1 | grep "BEST SCORE\|ms/tree"

# Protein S=20
./mpboot-avx -s ../data_treebase/prot_M10236_59_164.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    -cost ../output/prot.cost \
    2>&1 | grep "BEST SCORE\|ms/tree"

# Fitch regression (phải giữ nguyên)
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 6 -gpu_device 2 -gpu_worker 200 \
    2>&1 | grep "BEST SCORE\|ms/tree"
```
