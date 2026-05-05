---
description: CPU PLL Fitch parsimony — scoring formula, xPars flag, ti[] traversal array, lazy newview mechanism
globs: ["**/sprparsimony.cpp", "**/*.cu", "**/*.cuh"]
alwaysApply: false
---

## PLL Fitch Parsimony Algorithm

### Scoring formula

For an inner node with children L and R (over `width` site-blocks, `states` states):
```
intersection[s] = L[s] & R[s]             // agree on state s
union[s]        = L[s] | R[s]
t_N             = ~(OR of intersection[s]) // no common state at this site
node[s]         = intersection[s] | (t_N & union[s])  // Fitch set
partial_score   = popcount(t_N)
parsimonyScore[node] = partial_score + parsimonyScore[L] + parsimonyScore[R]
```

`parsimonyScore[n]` = total parsimony for **all branches in n's subtree**, not counting the
edge to n's parent. Tips have `parsimonyScore = 0`.

Full-tree parsimony at edge (p, p->back):
```
result = parsimonyScore[p->number] + parsimonyScore[p->back->number] + cross(p, p->back)
```
where `cross(p, q) = popcount(~(OR_s(parsVect_p[s] & parsVect_q[s])))`.

GPU equivalents: `warpNewviewStep` (Fitch rule), `warpEvaluateScore` (evaluate at edge).

### `ti[]` traversal array

`ti[]` is an int array encoding a post-order traversal plan:
- `ti[0]` = total length (entries start at index 4, each tuple = 4 ints: `[p, q, r, _]`).
- `ti[1]`, `ti[2]` = edge endpoints for `evaluateParsimonyIterativeFast`.
- Entries `ti[4], ti[8], ...` = `[p_num, q_num, r_num, _]` for each newview step.

`newviewParsimonyIterativeFast` processes `ti[]` bottom-up, setting:
```c
parsimonyScore[p] = cross(q, r) + parsimonyScore[q] + parsimonyScore[r]
```

`evaluateParsimonyIterativeFast` calls `newviewParsimonyIterativeFast` first (if `ti[0] > 4`),
then returns `parsimonyScore[p] + parsimonyScore[q] + cross(p, q)`.

### xPars lazy evaluation

**xPars flag** is per-nodeptr (not per node number). Each inner node has 3 nodeptrs (faces).
Only one face holds `xPars = 1` at a time — the face from which parsVect was last computed.

- `p->xPars = 1` → `parsVect[p->number]` valid when computed from `p`'s children
  (`p->next->back`, `p->next->next->back`).
- `getxnodeLocal(p)` moves xPars from `p->next` or `p->next->next` onto `p`.
- `computeTraversalInfoParsimony(p, ti, full=FALSE)` fills `ti[]` only for descendants
  with `xPars = 0` (stale). With `full=TRUE`, re-traverses unconditionally.
- `newviewParsimony(p)` calls `computeTraversalInfoParsimony(p)` then `newviewParsimonyIterativeFast`.

**Critical**: `parsimonyScore[n]` is **direction-dependent**. The value reflects whichever
face of node n last called newview. Different faces of the same node can give different
`parsimonyScore` values (they should all equal the correct subtree parsimony, but only when
viewed from the right direction for the current evaluation).

### The line-2291 "VERY IMPORTANT" call

In `rearrangeParsimony` (`sprparsimony.cpp:2291`), before any SPR move is attempted:
```c
evaluateParsimony(tr, pr, p, PLL_FALSE, perSiteScores);
```
This lazily refreshes `parsimonyScore[p->number]` and `parsimonyScore[q->number]` (where
`q = p->back`) so they reflect the **current topology** before removeNodeParsimony is called.
Without this, stale `parsimonyScore` values from a previous iteration would corrupt testInsert
evaluations. The GPU must replicate this or use an equivalent eager refresh.
