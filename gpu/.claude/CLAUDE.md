# GPU Parsimony – Developer Context

## 1. Pipeline Overview: Porting MPBoot to GPU

MPBoot's candidate-tree generation (`makeParsimonyTreeFast` in `sprparsimony.cpp`) builds
`numInitTrees` trees from scratch using stepwise-addition + SPR hill-climbing.
The CPU runs them **serially**; the GPU runs **K trees in parallel** (one CUDA block per tree,
one warp of 32 lanes per block).

### Entry point
`gpuInitCandidateTrees()` in `gpu/src/gpu_init_trees.cu`.
Called from `IQTree::initCandidateTreesParsimony()` when `--use_gpu` is passed.

### Pipeline steps (numbered as printed at runtime)

| Step | Code | What happens |
|------|------|-------------|
| [1] | `_allocateParsimonyDataStructures` | CPU allocs `parsVect` buffers and compresses alignment into Fitch bit-vectors per partition |
| [2] | `gpuParsimonyMemAlloc` | GPU allocs `d_parsVect[K][2N+1][width][states]`, `d_parsScore[K][2N+1]`, `d_topos[K]` |
| [3] | `uploadTipParsVect` | Reorders tip parsVect from CPU layout `[node][state][block]` → GPU layout `[node][block][state]` and uploads (tips are **read-only**, shared across all K trees) |
| [4] | `cpuToGpuTopology` + `uploadTopology` | Converts PLL pointer-ring to flat integer arrays (`GpuTopology`), uploads **one initial topology** to all K trees |
| [5] | `buildParsimonyTreesKernel` (`pars_build.cu`) | Each block builds one tree via **stepwise addition** on GPU |
| [6] | `gpuSprKernel` (`gpu_spr.cu`) | Each block runs **SPR hill-climbing** on its tree |
| [7] | `downloadTopology` + `gpuTopoToCpu` + `pllTreeToNewick` | Download K topologies, convert back to PLL pointer-rings, emit Newick strings |

### Data layout

**GpuTopology** (`pars_tree.cuh`): flat integer arrays mirroring PLL's pointer ring.
- Tips 1..N: one vface each, `vf = num - 1`.
- Inner nodes N+1..2N-1: three vfaces each, `vf = N + 3*(num-N-1) + face_idx`.
- `face_idx`: GPU face[0], face[1], face[2]. Ring direction: **face[2]→face[1]→face[0]→face[2]**.
- `nodepVf(num, N)` = canonical face for `tr->nodep[num]` = **GPU face[2]**.
- `back_vf[vf]` = the back-neighbor's vface (analogous to PLL's `p->back`).

**PLL vs GPU face mapping**:
| PLL name | GPU vface |
|----------|-----------|
| `p` = `tr->nodep[i]` = face[0] | GPU face[2] (nodepVf) |
| `p->next` = face[1] | GPU face[1] |
| `p->next->next` = face[2] | GPU face[0] |

PLL ring direction: face[0]→face[1]→face[2]→face[0].
GPU ring direction: face[2]→face[1]→face[0]→face[2].
They agree on face[1]; face[0] and face[2] are swapped.

**parsVect layout**: `pars_tree[node * width * states + block * states + state]`.
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
This equals GPU's `warpEvaluateScore(p_num, q_num)`.

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
All are template on `SharedT` (works for both `BuildShared` and `SprShared`).

### `computeTraversalInfoParsimony(topo, sh, node, N, full)`
Builds `sh.ti[]` from `node`'s perspective (iterative DFS via `sh.tiStack`).
- Mirrors CPU's xPars flag transfer: if `!xpars[p]`, moves xpars from `pNext` or `pNnxt`.
- `full=false`: only recurses into children with `xpars=0` (lazy mode).
- `full=true`: recurses into all inner children unconditionally.
- Appends `(p_num, q_num, r_num)` tuples to `sh.ti[]` (starting at `sh.tiSize`).
- **Must be called from lane 0 only.**

### `newviewParsimony(pars_tree, score_tree, sh, evaluate, width, states)`
Processes `sh.ti[]` bottom-up (`i = tiSize-3 … 3`), updating `parsVect` and `score_tree`.
- If `evaluate=false`: just updates, returns `score_tree[sh.ti[3]]` (top-of-ti node).
- If `evaluate=true`: after newview, also evaluates at edge `(sh.ti[1], sh.ti[2])` and
  returns `score_tree[ti[1]] + score_tree[ti[2]] + cross(parsVect[ti[1]], parsVect[ti[2]])`.
- All 32 lanes participate in parsVect computation; lane 0 handles score accumulation.

### `createTiAndNewviewParsimony(pars_tree, score_tree, topo, sh, p, N, width, states, lane)`
Wrapper: sets `sh.tiSize = 3`, calls `computeTraversalInfoParsimony(p, lazy)`, then
`newviewParsimony(evaluate=false)`. Updates `parsVect[vfToNum(p)]` and all stale descendants.

### `createTiAndEvaluateParsimony(pars_tree, score_tree, topo, sh, p, N, full, width, states)`
Wrapper: sets `sh.tiSize = 3`, sets `sh.ti[1]=vfToNum(p)`, `sh.ti[2]=vfToNum(back_vf[p])`,
then calls `computeTraversalInfoParsimony` for both p and `back_vf[p]`.
Returns full-tree parsimony at edge `(p_num, back_vf[p]_num)`.
- `full=true`: equivalent to CPU's `evaluateParsimony(p, PLL_TRUE)` — unconditional refresh.
- `full=false`: equivalent to CPU's `evaluateParsimony(p, PLL_FALSE)` — lazy refresh.

**GPU equivalent of CPU line-2291 call:**
```cpp
createTiAndEvaluateParsimony(pars_tree, score_tree, topo, sh, p, N, /*full=*/true, width, states)
```

---

## 4. Current GPU Implementation Status (2026-05-07)

### ✅ Fully working

**Joined kernel `buildParsimonyTreesKernel`** (`pars_build.cu`) — stepwise-addition + SPR in
one `__global__` function sharing `BuildShared` shared memory (≈24.8 KB on A100).

**Verified results** for N=295, K=99, sprDist=6:
| | Best | Worst |
|--|------|-------|
| Pre-SPR parsimony (after build) | 6734 | 6808 |
| Post-SPR parsimony | **6676** | 6764 |
| CPU reference (_pllSprOnCurrentTree) | ~6668 | ~6760 |
| Total kernel time | 7851 ms | — |
| ms/tree (build+SPR) | 79.3 ms | — |

GPU post-SPR best (6676) is within **0.1%** of CPU best (6668). ✅

### Joined kernel structure (pars_build.cu)

`buildParsimonyTreesKernel(... sprDist ...)`:
1. **Phase 0** (lane 0): Fisher-Yates shuffle → `sh.seed`, build 3-tip tree.
2. **Phase 1**: Stepwise addition for tips 4..N. Uses `sh.ti[]` + `computeTraversalInfoParsimony`
   + `newviewParsimony` (xPars-aware). After this, `sh.bestParsimony` = tree parsimony.
3. **Phase 2**: Set `topo->start_vface = nodepVf(1, N)`.
4. **Phase 3** (if `sprDist > 0`): SPR hill-climbing.
   - Init: `sh.randomMP = sh.bestParsimony` (from build phase — xPars already consistent).
   - **No `recomputeAllNodes` needed**: xPars flags from build are valid.
   - `do-while randomMP < startMP`:
     - For i = 1..2N-2: `createTiAndEvaluateParsimony(p, full=true)` (line-2291 equivalent)
     - P-branch: removeNode → doAddTraverse → restore → createTiAndNewview
     - Q-branch: same with q
     - Apply best move if improving

### Key design decisions

- **`SprShared` removed** — `BuildShared` now holds all fields for both phases.
- **`sh.seed` shared** — build sets it via `sh.seed = d_seeds[k]`; SPR inherits it directly.
- **`sh.randomMP` init** — set from `sh.bestParsimony` (last build insertion score) instead of
  calling `recomputeAllNodes`. Valid because xPars flags are already consistent after build.
- **`sh.stack[]` reused** — build DFS stack reused as SPR addTraverse stackVf (sequential phases).
- **`gpuSprKernel` removed** — `gpu_spr.cu` now contains only a no-op `gpuSprBuildTrees` stub.
- **Pipeline steps [5]+[6] merged** → single call `gpuStepwiseBuildTrees(mem, seeds, sprDist, stream)`.

### Fixed bugs

| Bug | Status |
|-----|--------|
| #A `testInsert`: `vfNnxtFace(q,N)` → `vfNnxtFace(p,N)` (crash) | ✅ Fixed |
| #B `sh.randomMP` uninitialized (wrong SPR threshold) | ✅ Fixed (use bestParsimony) |
| #C Missing `__syncwarp()` in `createTiAndEvaluateParsimony` | ✅ Fixed |

### Open issues

- **Bug #C DFS off-by-one**: `recomputeAllNodes` (now unused) processed 293/294 nodes due to
  last inner node having `back_vf = -1`. No longer blocks correctness because we don't use
  `recomputeAllNodes` for SPR init. Root cause (last node of build not fully hookup'd) may
  still exist but has no observable impact.

---

## 5. Key File Map

| File | Purpose |
|------|---------|
| `gpu/src/gpu_init_trees.cu` | Entry point — calls `gpuStepwiseBuildTrees(sprDist)`, shows [5+6] timing |
| `gpu/src/pars_build.cu` | **Main kernel**: build phase + SPR phase; all SPR device functions |
| `gpu/src/gpu_spr.cu` | No-op stub for `gpuSprBuildTrees` (SPR is now in pars_build.cu) |
| `gpu/src/pars_tree.cu` | Memory alloc, topology conversion, upload/download |
| `gpu/include/pars_tree.cuh` | `BuildShared`, `GpuTopology`, template traversal functions |
| `gpu/include/pars_build.cuh` | `gpuStepwiseBuildTrees(mem, seeds, sprDist, stream)` declaration |
| `gpu/include/topo_helpers.cuh` | `vfToNum`, `nodepVf`, `vfNextFace`, `vfNnxtFace`, `gpuRandum` |
| `mpboot/sprparsimony.cpp` | CPU reference: `_pllSprOnCurrentTree`, `rearrangeParsimony`, `testInsertParsimony` |
