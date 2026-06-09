---
description: GPU topology encoding — GpuTopology vface indexing, PLL↔GPU face mapping, parsVect and score_tree layout
globs: ["**/*.cu", "**/*.cuh"]
alwaysApply: false
---

## GPU Data Layout

### GpuTopology (`pars_tree.cuh`)

Flat integer arrays that mirror PLL's pointer ring. Each position is a **vface** (virtual face) index.

- **Tips** 1..N: one vface each. `vf = num - 1`.
- **Inner nodes** N+1..2N-1: three vfaces each. `vf = N + 3*(num-N-1) + face_idx`.
- `face_idx` ∈ {0, 1, 2}. GPU ring direction: **face[2] → face[1] → face[0] → face[2]**.
- `nodepVf(num, N)` = canonical vface for `tr->nodep[num]` = **GPU face[2]**.
- `back_vf[vf]` = back-neighbor's vface (≡ PLL's `p->back`).
- `vfNextFace(vf, N)` and `vfNnxtFace(vf, N)` traverse the ring.

### PLL ↔ GPU face mapping

| PLL pointer | GPU vface |
|-------------|-----------|
| `p` = `tr->nodep[i]` (face[0] in PLL) | GPU **face[2]** (= `nodepVf`) |
| `p->next` (face[1] in PLL) | GPU **face[1]** |
| `p->next->next` (face[2] in PLL) | GPU **face[0]** |

PLL ring: face[0]→face[1]→face[2]→face[0].
GPU ring: face[2]→face[1]→face[0]→face[2].
**face[1] is shared; face[0] and face[2] are swapped.**

Concrete operations:
- `vfNextFace(nodepVf(i))` = GPU face[1] = PLL `p->next`
- `vfNnxtFace(nodepVf(i))` = GPU face[0] = PLL `p->next->next`
- `back_vf[nodepVf(i)]` = PLL `p->back`
- `back_vf[vfNextFace(nodepVf(i))]` = PLL `p->next->back`
- `back_vf[vfNnxtFace(nodepVf(i))]` = PLL `p->next->next->back`

### parsVect layout

Both CPU and GPU use `[node][state][block]` layout:
```
index = k * parsVectPerTree + node * width * states + state * width + block
```
Direct `memcpy` from CPU to GPU — no reordering needed. `uploadTipParsVect` copies tip
parsVect directly without transposition.

### score_tree

`score_tree[node]` = accumulated Fitch parsimony for node's entire subtree from its
DFS-children's side. Equivalent to PLL's `parsimonyScore[node]`.
- Tips: always 0.
- Inner: `partial_cross(c1, c2) + score_tree[c1] + score_tree[c2]`.
- **Direction-dependent**: value reflects the direction the last newview was computed from.
- Set by `warpNewviewStep` + `warpReduceU32`; read by `warpEvaluateScore`.

Full-tree parsimony at edge (A, B):
```
warpEvaluateScore(A, B) = cross(parsVect[A], parsVect[B]) + score_tree[A] + score_tree[B]
```
