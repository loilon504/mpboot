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

## 3. Current GPU Implementation Status

### What is working
- **Step [5] Stepwise build**: kernel builds 99 trees in ~440 ms (4.4 ms/tree). Trees are
  topologically valid and parsimony is plausible (pre-SPR best=15368 for N=295).
- **Topology conversion** (`cpuToGpuTopology`, `gpuTopoToCpu`): correct.
- **`recomputeAllNodes`**: correctly traverses N-2 inner nodes in post-order from start_vface
  and returns the full-tree parsimony. Verified: processes 293 nodes for N=295 (see bug #2).
- **`warpNewviewStep`, `warpEvaluateScore`**: Fitch logic matches CPU.
- **`applyMove`** (= `removeNodeParsimony` + `restoreTreeParsimony` + newview):
  hookup logic matches CPU.

### Bug #1 — SPR makes trees WORSE (main open bug)

**Symptom**: Pre-SPR best = 15368, Post-SPR best = **16369** (should be ~6668 matching CPU).
SPR is consistently worsening trees. For k=0, `sh.randomMP` goes from 18175 → ~18800
across iterations.

**Root cause hypothesis**: `testInsert`'s `mp` estimate is **too low** (underestimate),
causing moves to be accepted that actually increase true tree parsimony. After `applyMove`,
`recomputeAllNodes` reveals the true (higher) parsimony, and the do-while loop exits
thinking it improved (randomMP < startMP from the wrong baseline).

**Suspected mechanism**: After `removeNodeParsimony(p)`, some `score_tree[]` values used
in `testInsert` are **stale** from the last `recomputeAllNodes`. Specifically,
`score_tree[sh.tip_p_num]` (= score for p's back-neighbor q) may include p's old
contribution if q is an ancestor of p in the rooted DFS from start_vface.
The CPU avoids this via the lazy `evaluateParsimony(p, FALSE)` call at line 2291 (before
removeNodeParsimony), which refreshes parsimonyScore[i] and q's parsimonyScore from their
current directions.

**Attempted fixes** (all applied in current code):
1. Added `q_num > N` guard in `doAddTraverse` to skip tip edges in testInsert. ✓ (necessary)
2. Added `recomputeAllNodes` after each `applyMove` to keep score_tree fresh. ✓ (necessary)
3. Corrected `testInsert` formula: parsVect[p_num] = Fitch(sh.tip_p_num, q_edge_num);
   evaluate at (p_num, r_num). ✓ (analytically matches CPU)

**Result after all fixes**: still 16369. Fixes 1 and 2 were necessary but not sufficient.

**Next angle to investigate**:
The GPU's `testInsert` does NOT replicate the CPU's critical `evaluateParsimony(p, FALSE)`
at line 2291, which refreshes stale scores before removeNodeParsimony. In the GPU, when
testing moves for node i, `score_tree[q_num]` (= sh.tip_p_num) and `score_tree[ins_num]`
were computed by the last `recomputeAllNodes` from the **start_vface direction**. If p (node i)
lies along the DFS path from start to some other node, removing p may not immediately
affect score_tree values in p's subtree — but score_tree for p's ancestors (including q)
will be stale after the topology changes from prior moves.

**Debug to add**: Print `sh.bestParsimony` (testInsert's mp estimate) vs
`fullMP` (recomputeAllNodes after applyMove). If they consistently differ, this confirms
the underestimation. Target: make these equal.

### Bug #2 — DFS processes 293 nodes, expects N-1=294

`recomputeAllNodes` prints `inner_nodes_processed=293 (expected N-1=294)` for N=295.
Expected: N-1 = 294 inner nodes. Off by one.
The code initializes counter to 0 and increments in the `top_idx==2` branch for each inner
node. Root node (back_vf[start_vface] = node 462) is pushed first; it should be processed.
Likely cause: one inner node in the tree has a back_vf pointing to NULL/−1 (disconnected
after the build step), or a cycle is causing the DFS to skip a node. The debug print
`[TOPO k=0] back_vf[1174]=-1` for node 589 (= 2N-1 = 589 for N=295) suggests that the
last inner node built in stepwise addition has a stale NULL back_vf.

### Performance note

SPR kernel takes 11384 ms for 99 trees (115 ms/tree). This is dominated by:
- O(2N × sprDist-depth DFS × testInsert evaluations) per iteration
- `recomputeAllNodes` after every applied move (O(N) work)

Once correctness is achieved, optimization should:
- Limit recomputeAllNodes to only nodes on the path affected by the move
- Consider warp-level parallelism for the DFS (currently lane 0 drives stack, all lanes do parsVect)

---

## 4. Key File Map

| File | Purpose |
|------|---------|
| `gpu/src/gpu_init_trees.cu` | Entry point: orchestrates the full pipeline |
| `gpu/src/pars_build.cu` | Stepwise-addition kernel (`buildParsimonyTreesKernel`) |
| `gpu/src/gpu_spr.cu` | SPR hill-climbing kernel (`gpuSprKernel`) |
| `gpu/src/pars_tree.cu` | Memory alloc, topology conversion, upload/download |
| `gpu/include/pars_tree.cuh` | `GpuTopology`, `GpuParsimonyMem`, `warpNewviewStep`, `warpEvaluateScore` |
| `gpu/include/topo_helpers.cuh` | `vfToNum`, `nodepVf`, `vfNextFace`, `vfNnxtFace`, `gpuRandum` |
| `mpboot/sprparsimony.cpp` | CPU reference: `makeParsimonyTreeFast`, `rearrangeParsimony`, `testInsertParsimony`, `newviewParsimonyIterativeFast`, `evaluateParsimonyIterativeFast` |
