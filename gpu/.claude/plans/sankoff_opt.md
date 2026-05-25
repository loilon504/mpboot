# Sankoff GPU: Bug Fixes & Performance Optimizations

## Status Overview

| Item | Status | Effect |
|------|--------|--------|
| Pattern set + encoding fix | ✅ DONE (gpu_init_trees.cu) | GPU reaches 6662 = CPU ✅ |
| Ratchet for Sankoff | ✅ DONE (2026-05-20) | K=200 đạt 6662 = CPU (was 6664) ✅ |
| Opt 1: Memory layout [ptn][state] → [state][ptn] | ✅ DONE (2026-05-20) | Sankoff 2.1×, Fitch 2.5× ✅ |
| Opt 3: Remove weighted newview in Sankoff | ✅ DONE (2026-05-20, combined w/ Opt 1) | Cleaned up dead code ✅ |
| ~~Identity matrix O(S)~~ | DROPPED | Not used in practice |

Baseline (K=200, -cost e, -sprdist 6, gpu_worker=200): ~168 ms/tree (pre-ratchet)
Post-ratchet (K=200): **346 ms/tree** (pre-Opt1)
Post-Opt1 (K=200): **167 ms/tree**, BEST SCORE **6662** = CPU ✅
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
