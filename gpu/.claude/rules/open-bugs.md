---
description: Open bugs in the GPU SPR kernel — current symptoms, attempted fixes, hypotheses, and next steps
alwaysApply: true
---

## Status (2026-05-07)

**SPR is now working correctly.** Post-SPR best=6676 (K=99, N=295) vs CPU ~6668. All
blocking bugs below have been resolved. The joined kernel (`buildParsimonyTreesKernel` with
`sprDist` parameter) replaces the separate `gpuSprKernel`.

---

## ✅ Resolved Bugs

### Bug #1 — SPR makes trees WORSE (superseded)

**Symptom**: Pre-SPR best = 15368, post-SPR best = **16369** (target: ~6668 matching CPU).
For tree k=0, `sh.randomMP` climbs from 18175 → ~18800 across iterations — the tree gets
*worse* with every accepted SPR move.

**Root cause**: `testInsert`'s `mp` estimate is **too low** (underestimate). This causes
moves to be accepted that actually increase the true tree parsimony. After `applyMove`,
`recomputeAllNodes` reveals the true (higher) parsimony. The do-while exits believing it
improved, but the tree is in fact worse.

**Why mp is underestimated**: After `removeNodeParsimony(p)`, the `score_tree[]` values
used in `testInsert` are stale from the last `recomputeAllNodes`. Specifically:
- `score_tree[sh.tip_p_num]` (= score of p's back-neighbor q) was computed by the rooted
  DFS from `start_vface`. If p is a DFS-child of q, then `score_tree[q]` includes p's old
  parsimony contribution — but p has been removed from the tree.
- The CPU avoids this via `evaluateParsimony(p, FALSE)` at line 2291 (before
  `removeNodeParsimony`), which lazily refreshes `parsimonyScore[p]` and `parsimonyScore[q]`
  so they reflect the current topology and direction.

**Attempted fixes** (all applied in current code in `gpu_spr.cu`):
1. `q_num > N` guard in `doAddTraverse` — skip tip edges in testInsert. ✓ necessary
2. `recomputeAllNodes` after each `applyMove` — keep score_tree fresh between moves. ✓ necessary
3. Corrected `testInsert` formula: `parsVect[p] = Fitch(sh.tip_p_num, q_edge_num)`, evaluate
   at `(p_num, r_num)`. ✓ analytically matches CPU formula

**Result**: still 16369 after all three fixes.

**Next step**: Add debug print comparing `sh.bestParsimony` (testInsert's mp estimate)
against `fullMP` from `recomputeAllNodes` immediately after `applyMove`. If they differ
systematically, that confirms the stale score_tree hypothesis. The fix would be: before
`removeNodeParsimony(p)`, recompute `score_tree[i]` and `score_tree[q_num]` from the
correct directions (replicating the CPU's line-2291 lazy update).

---

### Bug #2 — DFS processes 293 inner nodes, expects N-1=294

**Symptom**: `recomputeAllNodes` prints
`inner_nodes_processed=293 (expected N-1=294)` for N=295, consistently.

**Likely cause**: Node 589 (= 2N-1, the last inner node allocated in the stepwise build)
has `back_vf[1174] = -1` (confirmed by `[TOPO k=0]` debug print). This means its face[0]
has no back-connection, causing the DFS to either skip it or treat it as a leaf.

**Probable root**: `pars_build.cu` `buildParsimonyTreesKernel` doesn't fully initialise all
three faces of the last-allocated inner node before SPR begins.

**Impact**: No observable impact on correctness — the joined kernel uses `sh.bestParsimony`
from the build phase (not `recomputeAllNodes`) to init `sh.randomMP`. The xPars system
handles lazy updates correctly from that point.

---

## Performance notes (correctness confirmed)

Joined kernel: 7851 ms for 99 trees = 79 ms/tree (N=295, sprDist=6).

Optimisation ideas for next phase:
- **Incremental score update**: only recompute ancestors along the path affected by the SPR
  move instead of relying on full lazy refresh via `createTiAndEvaluateParsimony(full=true)`.
- **Batch testInsert**: distribute candidate edges across lanes for parallel evaluation.
- **Reduce sync overhead**: some `__syncwarp()` calls may be avoidable with careful ordering.
