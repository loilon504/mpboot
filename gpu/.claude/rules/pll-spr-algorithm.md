---
description: CPU SPR and stepwise-addition algorithms — rearrangeParsimony, testInsertParsimony, removeNodeParsimony, restoreTreeParsimony, and GPU equivalents
globs: ["**/sprparsimony.cpp", "**/gpu_spr.cu", "**/pars_build.cu"]
alwaysApply: false
---

## CPU SPR and Stepwise-Addition Algorithms

### Stepwise addition (`makeParsimonyTreeFast`, `sprparsimony.cpp`)

1. Fisher-Yates shuffle of tip indices `perm[1..N]`.
2. Build 3-tip star: `hookupDefault(tip_ip, tip_iq)`, `buildNewTip(tip_ir, inner1)`.
3. For each new tip `perm[4..N]`:
   - Allocate inner node q via `buildNewTip(tip, q)`:
     `p->face[0]->back = tip; face[1]->back = face[2]->back = NULL`.
   - DFS: `stepwiseAddition` → `testInsert` at every candidate edge.
   - **`insertParsimony(q, q_cand)`**: `face[1]↔q_cand`, `face[2]↔r=q_cand->back`;
     `newviewParsimony(q)` from face[0] (children = face[1]→q_cand, face[2]→r).
   - **Evaluate**: `evaluateParsimony(q->next->next, FALSE)` at edge (face[2] of q, r).
     Recomputes parsVect[q->number] from face[2]'s perspective:
     children = face[0]→q_back, face[1]→q_cand.
     Result = `parsimonyScore[q] + parsimonyScore[r] + cross(q, r)`.
   - **Undo**: `hookupDefault(q_cand, r)`, null face[1] and face[2] backs.
   - Insert at best edge.

### SPR hill-climbing (main loop in `makeParsimonyTreeFast`)

```
do {
    startMP = randomMP
    nodeRectifierPars(tr)           // normalise tr->start
    for i = 1..2N-2:
        evaluateParsimony(nodep[i], FALSE)   // line 2291 — CRITICAL lazy refresh
        rearrangeParsimony(nodep[i], mintrav=1, maxtrav=sprDist)
        if tr->bestParsimony improved: restoreTreeRearrangeParsimony(); randomMP = tr->bestParsimony
} while (randomMP < startMP)
```

### `rearrangeParsimony(p, mintrav, maxtrav)`

`q = p->back`. Two independent branches searched:

**P-branch** (only if p = inner node, and at least one of p1/p2 is inner):
```
p1 = p->next->back,  p2 = p->next->next->back
removeNodeParsimony(p)       // hookup(p1,p2); face[1]->back = face[2]->back = NULL
// tip_p = p->back (unchanged — used in testInsert evaluation)
addTraverseParsimony(p, p1->next->back,       mintrav=1, maxtrav)
addTraverseParsimony(p, p1->next->next->back, mintrav=1, maxtrav)  // if p1 inner
addTraverseParsimony(p, p2->next->back,       mintrav=1, maxtrav)  // if p2 inner
addTraverseParsimony(p, p2->next->next->back, mintrav=1, maxtrav)
hookupDefault(p->next, p1); hookupDefault(p->next->next, p2)       // restore
newviewParsimony(p)
```

**Q-branch** (only if q = inner AND at least one of q1/q2 is inner with inner grandchild):
```
q1 = q->next->back,  q2 = q->next->next->back
condition: (q1 inner && q1 has inner grandchild) || (q2 inner && q2 has inner grandchild)
removeNodeParsimony(q)       // hookup(q1,q2); face[1]->back = face[2]->back = NULL
// tip_q = q->back = p (node i, unchanged)
addTraverseParsimony(q, ..., mintrav=2, maxtrav)   // note mintrav2 = max(mintrav,2)
restore q; newviewParsimony(q)
```

### `testInsertParsimony(p, q_cand)`

```
r = q_cand->back
insertParsimony(p, q_cand)    // face[1]↔q_cand, face[2]↔r; newviewParsimony(p) from face[0]
mp = evaluateParsimony(p->next->next, FALSE)
     // face[2] of p: recomputes parsVect[p->number] from face[2]'s children:
     //   child1 = face[0]->back = p->back = tip_p
     //   child2 = face[1]->back = q_cand
     // parsimonyScore[p] = cross(tip_p, q_cand) + score[tip_p] + score[q_cand]
     // result = parsimonyScore[p] + parsimonyScore[r] + cross(p, r)
undo: hookupDefault(q_cand, r); face[1]->back = face[2]->back = NULL
if mp < tr->bestParsimony: record (removeNode=p, insertNode=q_cand)
```

**GPU `testInsert` equivalent** (in `gpu_spr.cu`):
```cpp
// tip_p_num = sh.tip_p_num = vfToNum(back_vf[p_vf], N)  (= p->back number)
// q_edge_vf = the candidate edge endpoint
// r_vf = back_vf[q_edge_vf]
parsVect[p_num] = warpNewviewStep(p_num, sh.tip_p_num, q_edge_num)
score_tree[p_num] = partial + score_tree[sh.tip_p_num] + score_tree[q_edge_num]
mp = warpEvaluateScore(p_num, r_num)
```

### `removeNodeParsimony(p)` (`sprparsimony.cpp:2251`)

```c
q = p->next->back;   r = p->next->next->back;
hookupDefault(q, r);
p->next->back = p->next->next->back = NULL;
// p->back remains unchanged
```

GPU equivalent (p_vf = nodepVf = face[2]):
```cpp
gpuHookup(back_vf, back_vf[vfNextFace(p_vf)], back_vf[vfNnxtFace(p_vf)]);
back_vf[vfNextFace(p_vf)] = -1;   // face[1]->back = NULL
back_vf[vfNnxtFace(p_vf)] = -1;   // face[0]->back = NULL
// back_vf[p_vf] unchanged
```

### `restoreTreeParsimony(p, q_ins)` (`sprparsimony.cpp:2197`)

```c
r = q_ins->back;
hookupDefault(p->next,       q_ins);  // face[1]↔q_ins
hookupDefault(p->next->next, r);      // face[2]↔r
computeTraversalInfoParsimony(p, ...); newviewParsimonyIterativeFast(tr)
// parsimonyScore[p] = cross(q_ins, r) + score[q_ins] + score[r]
```

GPU `applyMove` equivalent:
```cpp
gpuHookup(back_vf, vfNextFace(rm_vf), ins_vf);   // face[1]↔ins
gpuHookup(back_vf, vfNnxtFace(rm_vf), r_vf);     // face[0]↔r
warpNewviewStep(rm_num, ins_num, r_num);
score_tree[rm_num] = partial + score_tree[ins_num] + score_tree[r_num];
```

### `addTraverseParsimony(p, q, mintrav, maxtrav)`

Recursive DFS over candidate insertion edges:
```
if --mintrav <= 0: testInsertParsimony(p, q)   // only at inner nodes (q->number > mxtips)
if q is inner && --maxtrav > 0:
    addTraverseParsimony(p, q->next->back,      mintrav, maxtrav)
    addTraverseParsimony(p, q->next->next->back, mintrav, maxtrav)
```
GPU iterative equivalent: `doAddTraverse` in `gpu_spr.cu` uses `sh.stkVf/stkMt/stkMa`.
