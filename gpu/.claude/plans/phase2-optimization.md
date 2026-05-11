# Kế hoạch tối ưu thời gian Phase 2

## Context

Phase 2b đang hoạt động đúng (best=6664, CPU ref=6668), nhưng chậm:
- Phase 1 (build + initial SPR): **79 ms/tree**
- Phase 2b (100 outer NNI+ratchet iters): **4672 ms/tree** — chậm hơn **59×**

File chính: `gpu/src/pars_build.cu` (kernel `buildParsimonyTreesKernel`, hàm `gpuSPRHillClimb`).

---

## Phân tích bottleneck

### Bottleneck 1 — `createTiAndEvaluateParsimony(full=true)` trong vòng lặp SPR (nặng nhất)

Trong `gpuSPRHillClimb`, mỗi iteration của inner for-loop (2N-2 nodes) gọi:
```cuda
createTiAndEvaluateParsimony(p, N, true, ...)  // line-2291 equivalent
```
Với `full=true`, hàm này traverse TẤT CẢ descendants của p VÀ p->back để recompute parsVect.
→ O(N) newviewStep calls × 2N-2 nodes = **O(N²) mỗi do-while pass**.

Với N=295: 588 nodes × 294 inner nodes/call ≈ 173,000 newview calls mỗi pass.

**Nguyên nhân dùng `full=true`:** xPars flags không được invalidate sau NNI/applyMove, nên `full=false` (lazy) không thể phát hiện stale ancestors. Dùng `full=true` để đảm bảo correctness.

### Bottleneck 2 — SPR do-while chạy đến plateau sau mỗi NNI/ratchet

Mỗi Phase 4 outer iteration gọi `gpuSPRHillClimb` (do-while đến plateau). Với cây đã gần tối ưu (sau Phase 3), do-while vẫn chạy ít nhất 1 full pass (2N-2 nodes) trước khi phát hiện không cải thiện.

### Bottleneck 3 — sprDist=6 cho Phase 4 quá lớn

`doAddTraverse` với sprDist=6 duyệt tối đa 2^6=64 candidate edges mỗi lần testInsert.
Trong Phase 4 (trees đã gần tối ưu), sprDist lớn tìm kiếm sâu hơn nhưng cây ít thay đổi.
sprDist nhỏ hơn (2-3) sẽ đủ để recover từ NNI perturbation.

### Bottleneck 4 — Ratchet chạy 2 SPR do-while thay vì 1

Mỗi even iteration: `gpuSPRHillClimb` (hclimb1) + `gpuSPRHillClimb` (hclimb2) = 2× time.

---

## Ba tối ưu đề xuất (theo thứ tự độ khó)

---

### Tối ưu 1 — Dùng `full=false` trong SPR loop (khai thác xPars lazy evaluation)

**Ý tưởng:** Sau mỗi NNI move và sau mỗi `applyMove`, chỉ một số nodes có parsVect stale (ancestors của node bị di chuyển). Nếu ta explicitly set `xpars=0` cho các nodes đó, `createTiAndEvaluateParsimony(full=false)` sẽ chỉ recompute đúng các nodes stale.

**Implemention:**
1. Sau `gpuOneRandomNNI(p_vf, q_vf)`: walk path từ p's parent và q's parent lên root, set `topo->xpars[...] = 0` cho các ancestors.
2. Sau `applyMove(rm, ins)`: walk path từ rm's new position lên root, set xpars=0.
3. Thay `full=true` → `full=false` trong `createTiAndEvaluateParsimony` bên trong `gpuSPRHillClimb`.

**Giả sử depth trung bình O(log N) ≈ 8 cho N=295:**
- Cũ: O(N) = 294 newview calls mỗi lần line-2291
- Mới: O(log N) ≈ 8 newview calls mỗi lần → ~**36× speedup** cho bước này

**Speedup ước tính:** Vì line-2291 chiếm phần lớn SPR time, tổng Phase 4 có thể giảm từ 4672 → ~130 ms/tree.

**Files cần sửa:**
- `pars_build.cu`: thêm `invalidateAncestors(topo, start_vf, N)` device function, gọi sau NNI và applyMove.
- `gpuSPRHillClimb`: đổi `full=true` → `full=false` ở line-2291 call.
- Sau `createTiAndEvaluateParsimony(start_vface, true)` (pre-SPR init): NOT thay đổi — vẫn dùng full=true để init sh.randomMP.

**Điểm cần verify:** `full=false` sau xPars reset cho kết quả parsimony bằng `full=true`. Test với K=1, log output của sh.randomMP theo từng iteration.

---

### Tối ưu 2 — Dùng sprDist nhỏ hơn cho Phase 4

**Ý tưởng:** Thêm tham số `sprDist4` (SPR radius cho Phase 4), riêng với `sprDist` (dùng cho Phase 3 initial SPR).

```cuda
// Phase 3: initial SPR — dùng sprDist gốc (thường 6)
gpuSPRHillClimb(..., sprDist, ...)

// Phase 4: recovery sau NNI/ratchet — dùng sprDist4 nhỏ hơn (thường 2-3)
gpuSPRHillClimb(..., sprDist4, ...)
```

**Ảnh hưởng lên doAddTraverse:** sprDist=6 → 64 candidates/call; sprDist=2 → 4 candidates/call. Giảm **16×** số testInsert calls.

**Speedup ước tính:** Nếu testInsert chiếm 50% SPR time, dùng sprDist4=2 giảm Phase 4 thêm 4-8×.

**Files cần sửa:**
- `pars_build.cu`: kernel nhận thêm `int sprDist4` param, dùng trong Phase 4 loop.
- `pars_build.cuh`: update signature `gpuStepwiseBuildTrees(..., sprDist4, ...)`.
- `gpu_init_trees.cu`: tính `sprDist4 = min(3, params.sprDist)`.

---

### Tối ưu 3 — Giới hạn số vòng do-while per outer Phase 4 iteration

**Ý tưởng:** Sau NNI/ratchet, cây đã gần tối ưu. Giới hạn SPR do-while ở max K vòng (thay vì đến plateau) để đảm bảo thời gian predictable.

```cuda
int maxDoWhile = 2;
do { ... } while (sh.randomMP < startMP && --maxDoWhile > 0);
```

**Trade-off:** Có thể bỏ sót một số cải thiện, nhưng thêm diversity thông qua nhiều outer iterations hơn.

**Speedup ước tính:** Nếu do-while thường chạy 3-4 pass, giới hạn 2 pass → 30-50% speedup.

**Files cần sửa:**
- `pars_build.cu`: `gpuSPRHillClimb` nhận thêm `int maxIter = INT_MAX` param.

---

## Thứ tự triển khai đề xuất

1. **Tối ưu 2** (dễ nhất, không ảnh hưởng correctness): thêm `sprDist4` param → test kết quả
2. **Tối ưu 3** (dễ, tune `maxDoWhile`): thêm cap → test
3. **Tối ưu 1** (phức tạp nhất, tiềm năng nhất): invalidate xPars + `full=false`

Mục tiêu: giảm Phase 4 từ 4672 ms/tree xuống dưới 500 ms/tree mà vẫn giữ best ≤ 6665.

---

## Verification

```bash
cd /raid/home/loinguyen/workspace/mpboot-gpu/build
make -j2 && ./mpboot-avx -s ../data_debug/tree1.phy -seed 1 -use_gpu > tree2.txt 2>&1
grep "\[GPU\]" tree2.txt
```

Mục tiêu sau tối ưu:
- `[5+6+7] GPU kernel`: < 1000 ms/tree (hiện tại 4672 ms)
- `Post-SPR parsimony best`: ≤ 6665 (hiện tại 6664)
