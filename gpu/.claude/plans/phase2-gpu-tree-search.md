# Kế hoạch Phase 2: GPU Tree Search (Iterative SPR Hill-Climbing)

## Context

Phase 1 hoàn thành: GPU build K cây parsimony, kết quả lưu trong GPU global memory:
- `d_parsVect[K][2N+1][width][states]` — parsVect mỗi node mỗi cây
- `d_parsScore[K][2N+1]` — score mỗi node
- `d_topos[K]` — topology mỗi cây

**Phase 2 = tiếp tục tìm kiếm trên K trees đã có sẵn trên GPU.** Không cần upload/download
giữa các iterations — chỉ download best tree 1 lần cuối để in output.

---

## 1. Flow chính xác của `doTreeSearch` (parsimony mode)

**Tham số:** `ratchet_iter=1` → xen kẽ NNI perturb và ratchet.

### Iteration lẻ (NNI perturb):
```
1. Chọn source tree từ K pool (random)
2. doRandomNNIs(numNNI)        numNNI = curPerStrength × (N-3)
3. pllOptimizeSprParsimony()  = SPR hill-climbing đến plateau
4. Cập nhật slot nếu tốt hơn
```

### Iteration chẵn (ratchet):
```
1. Chọn source tree từ K pool
2. Perturb alignment: tăng weight ~50% compressed blocks × 2
3. SPR hclimb1 với weighted parsimony
4. Restore weights về 1
5. SPR hclimb2 với parsimony gốc
6. Cập nhật slot nếu tốt hơn
```

**Insight ratchet:** `createPerturbAlignment` chỉ thay đổi **frequency (weight)** của sites.
Ở mức compressed block (width=40): nhân `__popc(t_N)` của block b với `site_weight[b]`.
→ Có thể implement 100% on-GPU với array `d_siteWeights[K][width]`.

---

## 2. Thiết kế GPU Phase 2 — Không cần CPU↔GPU round-trip

### Key insight (do user chỉ ra)
K trees đã có sẵn trên GPU sau Phase 1. Mọi thao tác (pick, perturb, SPR, update) đều
làm trong GPU global memory. CPU chỉ cần:
- Truyền tham số kernel (numIter, numNNI, sprDist)
- Download `d_topos[best_k]` 1 lần cuối để output Newick

### Flow trên GPU (K blocks song song):

```
KERNEL gpuDoTreeSearchKernel(K iters = T):

  [Phase 1 output đã có: d_topos[K], d_parsVect[K], d_parsScore[K]]

  FOR iter = 1..T:
    // Mỗi block k độc lập:
    
    IF iter is ODD:
      // Pick random source tree từ K pool (D2D copy nếu khác slot mình)
      src = rand_int(0, K-1) using sh.seed
      if (src != k): copy_topo(d_topos[src] → working buffer)
      
      // Perturb topology
      gpuRandomNNIs(sh, topo, numNNI, width, states, lane)
      
      // Optimize
      gpuSPRHillClimb(pars_tree, score_tree, topo, sh, sprDist, N, width, states, lane)
    
    ELSE (iter is EVEN, ratchet):
      // Generate site weights (50% blocks doubled)
      if (lane == 0): gpuGenerateRatchetWeights(d_siteWeights[k], width, sh.seed)
      __syncwarp()
      
      // SPR hclimb1 với weighted parsimony
      gpuSPRHillClimbWeighted(d_siteWeights[k], ...)
      
      // Reset weights
      if (lane == 0): resetSiteWeights(d_siteWeights[k], width)
      __syncwarp()
      
      // SPR hclimb2 với parsimony gốc
      gpuSPRHillClimb(...)
    
    // Update d_topos[k] với best topology tìm được trong iteration này
    // (dùng sh.randomMP để track best score)
    
    topo->bestParsimony = sh.randomMP
  
  // Sau T iters: mỗi d_topos[k] chứa best tree block k tìm được
```

### CandidateSet thay thế
Thay vì CandidateSet phức tạp, dùng cơ chế đơn giản:
- Mỗi block k giữ 1 "current best tree" trong `d_topos[k]`
- Khi iteration tìm được cây tốt hơn: update `d_topos[k]` với topology mới
- "Pick random source": block k chọn 1 trong K slots ngẫu nhiên để copy topology từ đó

Điều này tự nhiên tạo ra **population diversity** (K trees khác nhau) và convergence pressure
(các blocks tốt nhất được chọn nhiều hơn theo xác suất).

---

## 3. Hai hướng tiếp cận

---

### Hướng 1: Port y hệt — GPU thay SPR, CPU quản lý outer loop

**Ý tưởng:** Giữ doTreeSearch logic trên CPU, mỗi iteration chỉ thay `pllOptimizeSprParsimony`
bằng GPU với K blocks. CPU vẫn pick/perturb/update CandidateSet, upload 1 tree per iter.

```
CPU outer loop per iteration:
  Pick 1 tree từ candidateTrees → upload topology + parsVect
  GPU: 1 block SPR
  CPU: download → update candidateTrees
```

**Điểm mạnh:**
- Thuật toán IDENTICAL với CPU gốc (1:1 validation dễ)
- Ít code nhất: chỉ upload/download 1 tree
- CandidateSet dedup + quality management giữ nguyên

**Điểm yếu:**
- Chỉ 1 GPU block per iteration → lãng phí 107 SMs
- Upload/download overhead > speedup SPR → **có thể chậm hơn CPU**
- Không khai thác parallelism thực sự của GPU

---

### Hướng 2: K search instances + GPU-native ratchet (ĐỀ XUẤT)

**Ý tưởng:** K blocks chạy T outer iterations độc lập trên GPU. Mọi state (topologies,
parsVect, site weights) đều trong GPU global memory. CPU sync mỗi T iters để check convergence.

**Điểm mạnh:**
- **Zero CPU↔GPU round-trip** trong quá trình tìm kiếm
- K blocks luôn bận → GPU utilization tối đa
- Ratchet on-GPU: site weight array, không cần re-upload parsVect
- Kết thúc: chỉ download 1 topology tốt nhất

**Điểm yếu:**
- Cần implement mới: `gpuRandomNNI`, `warpNewviewStepWeighted`
- Algorithm diverges từ CPU: không dùng CandidateSet chính xác như CPU
- Convergence criterion: cần global reduction giữa K blocks (dùng cooperative groups
  hoặc atomic trong global memory)

---

## 4. Thứ tự triển khai Hướng 2

### Phase 2a — doRandomNNIs + outer loop (không ratchet)

**Bước 1:** `gpuRandomNNI` trong `pars_build.cu`
```cuda
// Lane 0: chọn random inner edge, apply NNI flip
// NNI: với edge (p, q): swap p.nnxt.back ↔ q.next.back (NNI type 1)
//                    OR swap p.nnxt.back ↔ q.nnxt.back (NNI type 2)
// Sau NNI: createTiAndNewviewParsimony(p) + createTiAndNewviewParsimony(q)
```

**Bước 2:** Thêm outer loop T vào Phase 3 của kernel:
```cuda
for (int iter = 1; iter <= numSearchIter; iter++) {
    // Pick source tree (optional: D2D copy)
    gpuRandomNNI(topo, sh, numNNI, N, width, states, lane)
    gpuSPRLoop(...)  // đã có
}
```

**Bước 3:** Host wrapper `gpuDoTreeSearch` — launch, sync, download best.

### Phase 2b — GPU-native ratchet

**Bước 1:** Thêm `d_siteWeights[K][width]` vào `GpuParsimonyMem`

**Bước 2:** `warpNewviewStepWeighted(... weights ...)`:
```cuda
score += weights[b] * __popc(t_N);  // thay vì score += __popc(t_N)
```

**Bước 3:** Xen kẽ trong outer loop: ODD = NNI, EVEN = ratchet (generate weights → SPR ×2).

---

## 5. Các file cần sửa

| File | Thay đổi |
|------|---------|
| `gpu/include/pars_tree.cuh` | Thêm `d_siteWeights` vào `GpuParsimonyMem`; `warpNewviewStepWeighted` |
| `gpu/src/pars_build.cu` | `gpuRandomNNI`; outer search loop với alternating NNI/ratchet |
| `gpu/include/pars_build.cuh` | Thêm `gpuDoTreeSearch(mem, numIter, numNNI, sprDist, stream)` |
| `gpu/src/gpu_init_trees.cu` | `gpuDoTreeSearch` host wrapper; download best tree sau khi xong |
| `mpboot/iqtree.cpp` | Gọi `gpuDoTreeSearch` trong `doNNISearch` khi `use_gpu && spr_parsimony` |
