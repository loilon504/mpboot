# GPU Parsimony – Developer Context

## 0. Quick Reference — Common Commands

Working directory: `mpboot-gpu/build/`  (created by `mkdir build && cd build`)

### Build
```bash
# Configure (run once from build/)
cmake ../mpboot -DIQTREE_FLAGS=avx -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=14 -DUSE_GPU=ON

# Build (default: GPU_NTAXA_TEMPLATE=OFF, NTAXA=800 only, fast ~5 min)
make -j4

# Build with all NTAXA buckets (slow ~16 min, needed for full Opt-P L3 speedup)
cmake ../mpboot ... -DGPU_NTAXA_TEMPLATE=ON
make -j4
```

### Run CPU
```bash
make -j4 && /usr/bin/time -v ./mpboot-avx \
    -s ../data_debug/tree1.phy -seed 1 \
    > tree2.txt 2>&1
```

### Run GPU
```bash
make -j4 && /usr/bin/time -v ./mpboot-avx \
    -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 3 \
    > tree2.txt 2>&1
```

### Benchmark scripts (run from `build/`)
```bash
bash bench_cpu.sh                          # all datasets → output/cpu/*.log
bash bench_gpu.sh                          # all datasets → output/gpu/*.log
bash bench_cpu.sh path/to/file.phy         # single file
python3 ../output/summarize.py             # → output/results.xlsx
```

### Key CLI flags

#### User-facing (CLI)
| Flag | Default | Meaning |
|------|---------|---------|
| `-use_gpu` | off | Enable GPU mode — calls `mpbootGpu()` then `gpuHillClimbing()` |
| `-gpu_device N` | **0** | CUDA device ID. `cudaSetDevice(N)` at start of `mpbootGpu` |
| `-numpars K` | 100 | Number of initial parsimony trees. K1 uses K blocks (tree index 1..K) |
| `-sprdist N` | 6 | SPR radius for all phases (Phase 2 initial SPR + Phase 3 NNI+SPR ratchet) |
| `-gpu_nni_strength X` | **0.5** | NNI perturbation strength: `numNNI = max(1, X×(N−3))`. Even-worker iterations |
| `-gpu_pool_size N` | **10** | Number of population pool slots for K2 topology restarts |
| `-gpu_worker N` | **100** | Number of K2 blocks (workers); if ≤0 or >K uses K. K2 runs independently of K1 count |
| `-gpu_worker_stop N` | **1** | Stop threshold multiplier: `unsuccess_thresh = unsuccess_iteration + k2_workers×N` |
| `-seed N` | random | RNG seed for all K trees |

Note: `-gpu_stop`, `-gpu_k1_ratio`, `-gpu_pool_stop`, `-gpu_hc_iter` do **not** exist in the current codebase.

#### Derived / internal (not CLI)
| Parameter | Source | Meaning |
|-----------|--------|---------|
| `numNNI` | `max(1, gpu_nni_strength×(N−3))` | NNI perturbation per even iteration; computed in `mpbootGpu` and `gpuHillClimbing` |
| `K_alloc` | `max(K, k2_workers)` | Actual GPU memory allocated for max(numpars, gpu_worker) trees |
| `unsuccess_thresh` | `unsuccess_iteration + k2_workers × gpu_worker_stop` | Hill-climbing stop condition (see below) |

#### Recommended benchmark command
```bash
./mpboot-avx -s <dataset> -use_gpu -seed 1 \
    -numpars 200 -sprdist 3 -gpu_pool_size 20 -gpu_worker 400
# Defaults: gpu_device=0, gpu_nni_strength=0.5, gpu_worker=100, gpu_worker_stop=1
```

#### Kernel info printed at runtime — format (2026-05-28)
```
[GPU] ═══════════════════════════════════════════════════════
[GPU]   K=200  N=295  device=0
[GPU]
[GPU]   [1]      CPU parsimony alloc         :    0.006 s  (width=40 states=4)
[GPU]   [2]      GPU memory alloc            :    0.001 s  (0.05 GB)
[GPU]   [3]      Upload tip parsVect (H->D)  :    0.009 s
[GPU]   [4]      Upload topologies (H->D)    :    0.001 s  (200 trees)
[GPU]
[GPU] --------------------------------------------
[GPU]   buildTreesKernel<STATES=4,NTAXA=800>
[GPU]         K1=200  k2=100  sprDist=3  shared=11.3 KB
[GPU]         time: 1.234 s
[GPU] --------------------------------------------
[GPU]   hillClimbingKernel<STATES=4,NTAXA=800>
[GPU]         time: 4.567 s
[GPU]
[GPU]   [5]  Kernels                     :    5.801 s  (200 trees, 29.00 ms/tree)
[GPU]            sprDist=3  NNI=29(0.50)  pool_size=10
[GPU] ═══════════════════════════════════════════════════════

[GPU Bootstrap]  or  [GPU HillClimb]  K2-hillclimb: K=100 pool=10 unsuccess=101
[GPU HillClimb] Round 1  done=100   last_impr=100   best=6664     t=0.45s
...
[GPU HillClimb] Done: N rounds
[GPU HillClimb] Pool → candidateTrees: 10/10 slots added, best_pool=6664
```

---

## 1. Pipeline Overview: Porting MPBoot to GPU

MPBoot's candidate-tree generation (`makeParsimonyTreeFast` in `sprparsimony.cpp`) builds
`numInitTrees` trees from scratch using stepwise-addition + SPR hill-climbing.
The CPU runs them **serially**; the GPU runs **K trees in parallel** (one CUDA block per tree,
one warp of 32 lanes per block).

### Entry points
- `mpbootGpu()` in `gpu/src/gpu_init_trees.cu` — K1 build + K2 one-shot pass; returns `GpuParsimonyMem*`.
- `gpuHillClimbing()` in `gpu/src/gpu_init_trees.cu` — iterative K2 outer loop for both bootstrap and non-bootstrap.
- `runBasicMpbootGpu()` in `phyloanalysis.cpp` — caller that orchestrates both. Sets `need_hc_loop = params.maximum_parsimony` for both bootstrap and non-bootstrap modes.

### Pipeline steps (as printed at runtime)

| Step | Code | What happens |
|------|------|-------------|
| [1] | `_allocateParsimonyDataStructures` | CPU allocs `parsVect` buffers, compresses alignment into Fitch bit-vectors (or Sankoff cost vectors) per partition |
| [2] | `gpuParsimonyMemAlloc` | GPU allocs `d_parsVect[K][2N+1][width][states]`, `d_parsScore[K][2N+1]`, `d_topos[K]`, pool, treels buffers |
| [3] | `uploadTipParsVect` (Fitch) or `uploadSankoffTipParsVect` + `uploadSankoffSiteWeights` | Upload tip parsVect; both Fitch and Sankoff use same layout `[node][state][block]` (no reorder needed) |
| [4] | `cpuToGpuTopology` + `uploadTopology` | Converts PLL pointer-ring to flat integer arrays (`GpuTopology`), uploads same initial topology to all K trees |
| [5] | `gpuStepwiseBuildTrees` (K1=`buildParsimonyTreesKernel` + hybrid_cb + K2=`buildPhase3Kernel`) | K1: each block builds one tree via stepwise addition + initial SPR; hybrid_cb: CPU builds trees concurrently; K2: skip (max_outer_iters=0) |
| [HC] | `gpuHillClimbing` outer loop | Iterative rounds: reset treels/pool, K2 from pool, download treels → candidateTrees/saveCurrentTree, convergence check, stopping criterion |

After `mpbootGpu` returns, `pool → candidateTrees` registration happens at **end of `gpuHillClimbing`** (not in mpbootGpu itself).

### NTAXA template dispatch (Opt-P Layer 3)

Both kernels are templated `<int STATES, int NTAXA>`. Dispatch at runtime by `mxtips`:
- `GPU_NTAXA_TEMPLATE=OFF` (default): always NTAXA=800 regardless of actual N.
- `GPU_NTAXA_TEMPLATE=ON`: 5 buckets — N≤128→128, N≤256→256, N≤384→384, N≤512→512, else→800.

**Observed NTAXA at runtime** (NCU profiling): N=202→NTAXA=256, N=413→NTAXA=512 (only when ON).

STATES dispatch: states=20 → S=20, else → S=4. Binary (2) and 32-state removed.

### Data layout

**GpuTopology** (`pars_tree.cuh`): flat integer arrays mirroring PLL's pointer ring.
- Tips 1..N: one vface each, `vf = num - 1`.
- Inner nodes N+1..2N-1: three vfaces each, `vf = N + 3*(num-N-1) + face_idx`.
- `face_idx`: GPU face[0], face[1], face[2]. Ring direction: **face[2]→face[1]→face[0]→face[2]**.
- `nodepVf(num, N)` = the formula for face[2] of node num (default canonical in `cpuToGpuTopology`).
- `topo->nodep[num]` = DFS-canonical vface for node num, set by `gpuNodeRectifierPars`.
  - Tips: `nodep[num] = num-1` (fixed).
  - Inner: DFS-encountered face; analogous to CPU `tr->nodep[num]`.
- `back_vf[vf]` = the back-neighbor's vface (analogous to PLL's `p->back`).

**GpuTopology current fields** (after Opt-D + dead code removal 2026-05-17):
```
back_vf[kMaxVFaces]      // int[3200]   12.5 KB  HOT
xpars[kMaxVFaces]        // int[3200]   12.5 KB  HOT
mxtips, ntips, nextnode  // int          12 B
bestParsimony            // unsigned int  4 B
preSprParsimony          // unsigned int  4 B   (after stepwise, before SPR)
postSprParsimony         // unsigned int  4 B   (after initial SPR; K2 entry point)
savedSeed                // long          8 B   (RNG state saved after Phase 2)
start_vface              // int           4 B
num_vfaces               // int           4 B
nodep[kMaxNodes]         // int[1600]     6.2 KB  MEDIUM
n_improved_even          // int           4 B
n_improved_odd           // int           4 B
n_total_even             // int           4 B
n_total_odd              // int           4 B
```
Total ≈ 31 KB. `best_back_vf[]` was removed 2026-05-17. No `number[]`, `next_vf[]`, `nnxt_vf[]` — pure arithmetic replaces them.

**Inner node boundary rules** (easy to confuse):
- By **vface**: inner if `vf >= N` (tips have `vf = 0..N-1`).
- By **node number**: inner if `num > N` (tips have `num = 1..N`).
- Never use `vf > N` (misses face[0] of first inner node) or `num >= N` (includes tip N).

**PLL vs GPU face mapping**:
| PLL pointer | GPU vface |
|-------------|-----------|
| `p` = `tr->nodep[i]` = PLL face[0] | GPU `nodepVf(i, N)` = **face[2]** (canonical in cpuToGpuTopology); after gpuNodeRectifierPars, `topo->nodep[i]` = any DFS-encountered face |
| `p->next` = PLL face[1] | GPU face[1] = `vfNextFace(nodepVf(i))` |
| `p->next->next` = PLL face[2] | GPU face[0] = `vfNnxtFace(nodepVf(i))` |
| `p->back` | `back_vf[nodepVf(i)]` |

PLL ring: face[0]→face[1]→face[2]→face[0].
GPU ring: face[2]→face[1]→face[0]→face[2].
face[1] is shared; face[0] and face[2] are swapped between PLL and GPU.

**parsVect layout**: `pars_tree[node * width * states + state * width + block]` (i.e. `[node][state][block]`).
`score_tree[node]` = accumulated Fitch parsimony for the entire subtree of `node` (from its
DFS-children's side), equivalent to PLL's `parsimonyScore[node]`.

---

## 2. CPU PLL Parsimony Implementation

### Fitch parsimony fundamentals

Each inner node stores a **parsVect** (bit-vector per state per site-block).
For a node with children L and R:
```
intersection[s] = L[s] & R[s]          // sites where L and R agree on state s
union[s]        = L[s] | R[s]
t_N             = ~(OR of all intersection[s])  // sites with NO common state
node[s]         = intersection[s] | (t_N & union[s])  // Fitch rule
partial_score   = popcount(t_N)          // substitutions at this node
parsimonyScore[node] = partial_score + parsimonyScore[L] + parsimonyScore[R]
```
`parsimonyScore[n]` = total parsimony for **all branches in n's subtree** (not counting the
edge to n's parent). For tips, `parsimonyScore[tip] = 0`.

Full-tree parsimony at edge (p, q):
```
evaluateParsimony(p) = parsimonyScore[p->number] + parsimonyScore[q->number] + cross(p, q)
```
where `cross(p, q) = popcount(~(OR of (parsVect_p[s] & parsVect_q[s]))`.
This equals GPU's `newviewParsimony(..., evaluate=true, ...)` at edge (p, q).

### The `ti[]` array and xPars lazy evaluation

PLL avoids redundant recomputation via the **xPars flag** (per nodeptr, not per node number):
- `p->xPars = 1` means `parsVect[p->number]` was computed from `p`'s perspective
  (children = `p->next->back`, `p->next->next->back`).
- `computeTraversalInfoParsimony(p, ti, ...)` traverses from `p` and fills `ti[]` with
  `[p->number, q->number, r->number]` tuples for all stale descendants (xPars=0).
- `newviewParsimonyIterativeFast(tr)` processes `ti[]` bottom-up, recomputes parsVect and
  parsimonyScore for each listed node.
- `getxnodeLocal(p)` moves the xPars flag from `p->next` or `p->next->next` to `p` itself.

This means **parsimonyScore[n] is direction-dependent**: the value depends on which face
last called newview on node n. In PLL's SPR, `evaluateParsimony(p, FALSE)` at the start of
`rearrangeParsimony` (line 2291, marked "VERY IMPORTANT") lazily refreshes parsimonyScore
for p and any stale ancestors before the search begins.

### Stepwise addition (`makeParsimonyTreeFast`)

1. Shuffle tip order (Fisher-Yates).
2. Start 3-tip tree from tips `perm[1..3]`.
3. For each new tip `perm[4..N]`:
   - Allocate a new inner node q from `tr->nodep[N + ntips - 1]`.
   - `buildNewTip(p, q)`: hookup PLL face[0] of q to tip p; face[1], face[2] = NULL.
   - DFS over candidate edges; call `stepwiseAddition` → `testInsert`:
     - `insertParsimony(q, q_cand)`: hookup face[1]↔q_cand, face[2]↔r=q_cand->back;
       `newviewParsimony(q)` from face[0]'s children (face[1]→q_cand, face[2]→r).
     - Evaluate: `evaluateParsimony(q->next->next, FALSE)` at edge (face[2] of q, r).
       Internally: recomputes parsVect[q->number] from face[2]'s perspective
       (children = face[0]→q_back, face[1]→q_cand), then sums.
     - Undo: `hookupDefault(q_cand, r)`, set face[1]->back = face[2]->back = NULL.
   - Insert at best edge.

### SPR hill-climbing (`rearrangeParsimony` + main loop in `makeParsimonyTreeFast`)

Outer do-while repeats until `randomMP` stops decreasing.
Inner loop: for each node i = 1..2N-2:

```
evaluateParsimony(tr->nodep[i], FALSE)   // lazy refresh of parsimonyScore[i] — line 2291 CRITICAL
```

**P-branch** (p = `tr->nodep[i]`, must be inner):
```
p1 = p->next->back,  p2 = p->next->next->back
if (p1 or p2 is inner):
    removeNodeParsimony(p)    // hookup(p1,p2); p->next->back = p->next->next->back = NULL
    tip_p = p->back           // unchanged (the "other side" of p)
    addTraverseParsimony(p, p1->next->back, mintrav=1, maxtrav=sprDist)
    addTraverseParsimony(p, p1->next->next->back, ...)
    addTraverseParsimony(p, p2->next->back, ...)  (if p2 inner)
    addTraverseParsimony(p, p2->next->next->back, ...)
    hookupDefault(p->next, p1); hookupDefault(p->next->next, p2)  // restore
    newviewParsimony(p)        // refresh parsimonyScore[i]
```

**Q-branch** (q = `p->back`, must be inner with at least one inner grandchild):
```
q1 = q->next->back,  q2 = q->next->next->back
if (condition on grandchildren):
    removeNodeParsimony(q)
    tip_q = p  (q's back, unchanged = node i)
    addTraverseParsimony(q, ..., mintrav=2, maxtrav=sprDist)
    restore q; newviewParsimony(q)
```

**testInsertParsimony(p, q_cand)**:
```
r = q_cand->back
insertParsimony(p, q_cand)        // face[1]↔q_cand, face[2]↔r; newviewParsimony(p) from face[0]
mp = evaluateParsimony(p->next->next, FALSE)
     // face[2] of p: recomputes parsVect[i] from face[2]'s perspective:
     //   children = face[0]->back = tip_p, face[1]->back = q_cand
     // parsimonyScore[i] = cross(tip_p, q_cand) + score[tip_p] + score[q_cand]
     // result = parsimonyScore[i] + parsimonyScore[r] + cross(i, r)
undo insertParsimony
if mp < tr->bestParsimony: record (removeNode=p, insertNode=q_cand)
```

After the full for-loop, apply best move (if better than `randomMP`):
```
restoreTreeRearrangeParsimony:
    removeNodeParsimony(tr->removeNode)
    restoreTreeParsimony(tr->removeNode, tr->insertNode)
       // face[1]↔q_cand, face[2]↔r; newviewParsimony from face[0]
    randomMP = tr->bestParsimony
```

---

## 3. New GPU Functions (pars_tree.cuh) — xPars-Aware Traversal

The following device functions mirror the CPU's `computeTraversalInfoParsimony` /
`newviewParsimonyIterativeFast` / `evaluateParsimonyIterativeFast` pipeline.
All traversal functions template on `SharedT` (works for `BuildSharedT<NTAXA>`).

### `computeTraversalInfoParsimony(topo, sh, node, N, full)`
Builds `sh.ti[]` from `node`'s perspective (iterative DFS via `sh.tiStack`).
- Mirrors CPU's xPars flag transfer: if `!xpars[p]`, moves xpars from `pNext` or `pNnxt`.
- `full=false`: only recurses into children with `xpars=0` (lazy mode).
- `full=true`: recurses into all inner children unconditionally.
- Appends `(p_num, q_num, r_num)` tuples to `sh.ti[]` (starting at `sh.tiSize`).
- **Must be called from lane 0 only.**

### `newviewParsimony<SharedT, STATES>(pars_tree, score_tree, sh, evaluate, width)`
Processes `sh.ti[]` bottom-up (`i = tiSize-3 … 3`), updating `parsVect` and `score_tree`.
Supports both Fitch (`sh.use_sankoff=false`) and Sankoff (`sh.use_sankoff=true`) modes.
- If `evaluate=false`: just updates, returns `score_tree[sh.ti[3]]` (top-of-ti node).
- If `evaluate=true`: after newview, also evaluates at edge `(sh.ti[1], sh.ti[2])` and
  returns `score_tree[ti[1]] + score_tree[ti[2]] + cross(parsVect[ti[1]], parsVect[ti[2]])`.
- All 32 lanes participate in parsVect computation; lane 0 handles score accumulation.

### `createTiAndNewviewParsimony<SharedT, STATES>(pars_tree, score_tree, topo, sh, p, N, width, lane)`
Wrapper: sets `sh.tiSize = 3`, calls `computeTraversalInfoParsimony(p, lazy)`, then
`newviewParsimony(evaluate=false)`. Updates `parsVect[vfToNum(p)]` and all stale descendants.

### `createTiAndEvaluateParsimony<SharedT, STATES>(pars_tree, score_tree, topo, sh, p, N, full, width)`
Wrapper: sets `sh.tiSize = 3`, sets `sh.ti[1]=vfToNum(p)`, `sh.ti[2]=vfToNum(back_vf[p])`,
then calls `computeTraversalInfoParsimony` for both p and `back_vf[p]`.
Returns full-tree parsimony at edge `(p_num, back_vf[p]_num)`.
- `full=true`: equivalent to CPU's `evaluateParsimony(p, PLL_TRUE)` — unconditional refresh.
- `full=false`: equivalent to CPU's `evaluateParsimony(p, PLL_FALSE)` — lazy refresh.

**GPU equivalent of CPU line-2291 call:**
```cpp
createTiAndEvaluateParsimony<SharedT, STATES>(pars_tree, score_tree, topo, sh, p, N, /*full=*/false, width)
```

---

## 4. Current GPU Implementation Status

### Kernel architecture: two-kernel design (K1 + K2)

**K1: `buildParsimonyTreesKernel<STATES, NTAXA>`** (`pars_build.cu`)
- Stepwise addition (Phase 0-1) + initial SPR (Phase 2).
- Saves `topo->savedSeed` and `topo->postSprParsimony` for K2 entry.
- Launch grid: `dim3(k1_count)` blocks × `dim3(32)` threads.

**K2: `buildPhase3Kernel<STATES, NTAXA>`** (`pars_build.cu`)
- Restores inter-kernel state from `GpuTopology`: `sh.seed = topo->savedSeed`, `sh.randomMP = topo->postSprParsimony`.
- Calls `runPhase3()` once per launch: pool restart → NNI or ratchet → SPR → pool insert → treels write.
- Launch grid: `dim3(k2_workers)` blocks × `dim3(32)` threads.
- Called repeatedly from `gpuHillClimbing` outer loop (one kernel launch per round).

**Host wrapper: `gpuStepwiseBuildTrees(mem, seeds, k1_count, sprDist, numNNI, poolSize, stream, after_k1, after_k2, k2_workers, max_outer_iters)`**
- `k1_count=0, max_outer_iters=1`: skip K1, run K2 once (gpuHillClimbing rounds).
- `k1_count=K, max_outer_iters=0`: run K1, skip K2 (mpbootGpu initial call).
- `after_k1` = `hybrid_cb`: CPU builds parsimony trees while K1 runs, then fills pool.

### `runPhase3` device function (shared by both kernels in `pars_build.cu`)

Runs exactly one iteration per K2 launch:
1. **Pool restart**: random linear probe for non-empty pool slot; load `back_vf` from pool, reset `xpars=0`, call `gpuNodeRectifierPars`.
2. **Perturbation** (fixed per worker by `blockIdx.x % 2`):
   - Even workers (NNI): `gpuRandomNNIs` → re-evaluate → `gpuSPRHillClimb`.
   - Odd workers (Ratchet): random site weights {1,2} → SPR → restore weights → second SPR.
3. **Pool update**: compute topology hash (Knuth multiplicative), dedup check, find worst slot, `atomicCAS` claim, copy `back_vf` with `__threadfence()` for cross-SM visibility, release per-slot spinlock.
4. **Treels write**: if `randomMP ≤ d_treelsCutoff`, `atomicAdd(treels_filled)` → copy `back_vf` to treels slot.
5. **Global best update**: `atomicMin(global_best, randomMP)`.

### `gpuSPRHillClimb` device function

Implements the SPR do-while loop (mirrors CPU `makeParsimonyTreeFast` inner loop):
- `do { nodeRectifierPars; for i=1..2N-2: eval+SPR; apply best move } while (randomMP < startMP)`.
- Treels-per-testInsert saves: inside the for-i loop, after finding best insert for each node i, optionally saves a topology snapshot to `treels_back_vf` if `bestParsimony ≤ cutoff`. Temporarily applies move if improvement was not already applied, copies `back_vf`, then undoes.

### `gpuHillClimbing` outer loop

```
for (;;):
    resetTreelsRound(cutoff); resetPoolRound()
    gpuStepwiseBuildTrees(k1_count=0, max_outer_iters=1)  // K2 one round
    download treels → for each: gpuTopoToCpu → pllTreeToNewick → score → saveCurrentTree or candidateTrees.update
    track improvement (bootstrap: max treels_logl; non-bootstrap: min pool score)
    update logl_cutoff if bootstrap
    check convergence (bootstrap: correlation; non-bootstrap: no check)
    if total_done - last_impr_at > unsuccess_thresh [&& correlation OK]: break
end pool → candidateTrees registration
```

**Stopping threshold**: `unsuccess_thresh = unsuccess_iteration + k2_workers × gpu_worker_stop`
- `unsuccess_iteration`: CPU parameter (default -1, set by `-nstep`); when -1 it defaults to 0 in the formula.
- `k2_workers × gpu_worker_stop`: compensates for K workers per round not being independent.

**Bootstrap mode**: `max_treels = K_alloc * 1000`; treels written per round → `saveCurrentTree(-(double)pars)`.
**Non-bootstrap mode**: `max_treels = 0` (disabled); pool scores tracked per round → `candidateTrees.update`.

### BuildSharedT<NTAXA> fields (current, in `pars_tree.cuh`)

```cpp
template<int NTAXA>
struct alignas(16) BuildSharedT {
    int16_t ti[NTAXA * 3];       // traversal info tuples
    int16_t tiStack[NTAXA];      // DFS stack
    int16_t stack[NTAXA * 2];    // Phase 1: node-pair DFS; Phase 3: addTraverse stackVf + NNI edge list
    union {
        int16_t perm[NTAXA + 2]; // Phase 0-1: Fisher-Yates permutation
        int16_t stackMint[NTAXA]; // Phase 2-3: mintrav per doAddTraverse stack entry
    };
    int stackMaxt[kMaxSprStack]; // SPR maxtrav + NNI bitset (kMaxSprStack=64)
    int stackTop, tiSize;
    int bcast[11];               // warp broadcast slots
    unsigned int bestParsimony, bestHits, randomMP, randomMPHits;
    int bestRemoveVf, bestInsertVf;
    long seed;
    const unsigned int* site_weights;  // nullptr=uniform; ratchet weights or Sankoff freqs
    bool use_sankoff;
    const unsigned int* cost_matrix;   // nullptr=Fitch
    // BUILD-ONLY (Phase 0-1):
    int insertVf, startVf, qnum, tipnum, qf0, qf1, qf2;
};
using BuildShared = BuildSharedT<kMaxTaxa>;  // = BuildSharedT<800>
```

**Note**: `SprShared` was removed. `BuildSharedT<NTAXA>` is the only shared memory struct.
Timing fields (`t_build`, `t_phase2`, etc.) and dead Opt-C/sym fields were removed 2026-05-17.
Size for NTAXA=800: ≈11.3 KB (print in kernel output as `shared=11.3 KB`).

### GpuParsimonyMem key fields (`pars_tree.cuh`)

```
d_parsVect          [K][2N+1][width][states]
d_parsScore         [K][2N+1]
d_topos             [K] GpuTopology
d_siteWeights       [K][width]  Fitch=ratchet weights; Sankoff=pattern frequencies
d_ratchetScratch    [K][width]  Sankoff ratchet scratch (nullptr=Fitch)
d_postSprScores     [K]         postSprParsimony per K1 tree (used by hybrid_cb)
pool_size           runtime value from -gpu_pool_size
d_poolScores        [pool_size]
d_poolBackVf        [pool_size × kMaxVFaces]
d_poolFilled        atomic fill counter
d_poolSlotLocks     [pool_size]  per-slot spinlocks (Opt-S)
d_poolHashes        [pool_size]  topology hash for dedup
d_globalBest        global best parsimony across all warps
max_treels          capacity (0=disabled)
d_treelsScores      [max_treels]
d_treelsBackVf      [max_treels × kMaxVFaces]
d_treelsFilled      atomic fill counter
d_treelsCutoff      score ≤ cutoff → write to treels
```

### Register counts (NCU profiling 2026-05-28)

NCU measured on A100 with default `GPU_NTAXA_TEMPLATE=OFF` (NTAXA=800):

| Kernel | Regs/thread | Smem/block | Blocks/SM (smem limit) |
|--------|------------|-----------|----------------------|
| K1 `buildParsimonyTreesKernel<4,800>` | ~160 | ~11.3 KB | ~13/SM |
| K2 `buildPhase3Kernel<4,800>` | ~253 | ~11.3 KB | ~13/SM |

Note: Earlier measurements (2026-05-18) showed K1=96, K2=128 after Opt-S (pool simplification). The higher 2026-05-28 values likely reflect additional code added since then (Sankoff mode, treels-per-testInsert, hash dedup, etc.). Shared memory (≈11.3 KB/block on A100) limits occupancy to ~13 blocks/SM = 20.31% theor. occupancy regardless of register count.

---

### Optimization history (cumulative)

| Opt | Description | Key result |
|-----|-------------|-----------|
| testInsert pre-refresh | Refresh q, r, tip_p subtrees before eval; skip redundant Fitch step | eval_size 1.72→1.00; t_search −34% |
| Opt-D | Remove `number[]`, `next_vf[]`, `nnxt_vf[]` from GpuTopology; use pure arithmetic | struct −37.5 KB; −8.7% ms/tree |
| Refactor #3 (2026-05-17) | Remove timing fields, dead Opt-C/sym fields from BuildSharedT; remove `best_back_vf[]` from GpuTopology; remove `-gpu_hc_iter` | K1 regs 155→96 (historical) |
| Pool restart simplification | Replace warp-parallel k-th min scan with lane-0 O(pool_size²) selection-sort | K2 regs 151→128 (historical) |
| Opt-S (2026-05-18) | Per-slot spinlocks `d_poolSlotLocks[pool_size]` replace single `pool_lock` | Contention K=1000→50 blocks/lock |
| Opt-LessK1 | Hybrid K1: only build K'=max(pool_size, K×ratio) trees; CPU builds rest concurrently | +22–27% total speedup |
| Sankoff mode | `sh.use_sankoff`, `sh.cost_matrix`; Sankoff tip upload; `newviewParsimony` dual-mode | Full Sankoff support |
| Treels-per-testInsert | Save topology snapshot per node-i inside `gpuSPRHillClimb` for-i loop | More diverse treels for bootstrap |
| Pool hash dedup | Knuth multiplicative hash; skip insert if hash already in pool | Reduces redundant pool entries |

**Current K1/K2 architecture note**: `gpu_k1_ratio` was planned in an earlier optimization (`Opt-LessK1`) but is **not in the current tools.h/tools.cpp**. The current code always passes `k1_count=K` (all K trees to K1). K2 is always `k2_workers = gpu_worker` (default 100).

### Fixed bugs (cumulative)

| Bug | Status |
|-----|--------|
| #A `testInsert`: `vfNnxtFace(q,N)` → `vfNnxtFace(p,N)` (crash) | Fixed |
| #B `sh.randomMP` uninitialized (wrong SPR threshold) | Fixed (use bestParsimony) |
| #C Missing `__syncwarp()` in `createTiAndEvaluateParsimony` | Fixed |
| #6 `__syncwarp;` missing `()` in SPR loop (race on nodep[]) | Fixed 2026-05-11 |
| #7 `q_num >= N` in `doAddTraverse` (should be `> N`) | Fixed 2026-05-11 |
| #8 `node_num >= N` in stepwise DFS (should be `> N`) | Fixed 2026-05-11 |

### Open issues

- **Build DFS off-by-one**: last inner node (2N-1) may have `back_vf[face[0]] = -1` after build.
  No observable impact because `gpuNodeRectifierPars` DFS stops at leaves (vf < N check).

---

## 5. Key File Map

| File | Purpose |
|------|---------|
| `gpu/src/gpu_init_trees.cu` | Entry points: `mpbootGpu` (K1+hybrid_cb), `gpuHillClimbing` (K2 outer loop), Sankoff tip upload |
| `gpu/src/pars_build.cu` | **Main kernels**: `buildParsimonyTreesKernel` (K1), `buildPhase3Kernel` (K2); device functions: `testInsert`, `doAddTraverse`, `applyMove`, `gpuSPRHillClimb`, `gpuRandomNNIs`, `runPhase3`, `gpuNodeRectifierPars`, `gpuStepwiseBuildTrees` |
| `gpu/src/pars_tree.cu` | Memory alloc/free, topology conversion (cpu↔gpu), upload/download, pool helpers |
| `gpu/include/pars_tree.cuh` | `GpuTopology`, `GpuParsimonyMem`, `BuildSharedT<NTAXA>`, template traversal functions (`computeTraversalInfoParsimony`, `newviewParsimony`, `createTiAndNewviewParsimony`, `createTiAndEvaluateParsimony`) |
| `gpu/include/pars_build.cuh` | `CpuTreeData`, `AfterK1Callback`, `AfterK2Callback`, `gpuStepwiseBuildTrees` declaration |
| `gpu/include/topo_helpers.cuh` | `vfToNum`, `nodepVf`, `vfNextFace`, `vfNnxtFace`, `gpuRandum`, `gpuHookup` |
| `gpu/include/pars_bootstrap.cuh` | Bootstrap memory: `gpuBootstrapMemAlloc`, `gpuUploadBootSamples` |
| `gpu/src/gpu_spr.cu` | **Does not exist** — SPR is entirely in `pars_build.cu` |
| `mpboot/sprparsimony.cpp` | CPU reference: `_pllSprOnCurrentTree`, `rearrangeParsimony`, `testInsertParsimony` |
| `mpboot/phyloanalysis.cpp` | `runBasicMpbootGpu`: calls `mpbootGpu` then `gpuHillClimbing` for all modes |
| `mpboot/tools.h` / `tools.cpp` | GPU params: `use_gpu`, `gpu_device`, `gpu_nni_strength`, `gpu_pool_size`, `gpu_worker`, `gpu_worker_stop` |
