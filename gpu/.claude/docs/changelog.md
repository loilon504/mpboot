# Bug Changelog — GPU Parsimony Port

Mỗi entry ghi lại một bug đã được tìm ra và fix trong quá trình porting MPBoot sang GPU.
Dùng làm tài liệu tham khảo khi viết khóa luận tốt nghiệp.

---

## Bug #3 — `testInsert` dùng sai vface khi hookup: `vfNnxtFace(q)` thay vì `vfNnxtFace(p)`

**Ngày**: 2026-05-06
**Task**: Step [6] SPR hill-climbing — `testInsert` (`gpu_spr.cu:77`)
**File liên quan**: `gpu/src/gpu_spr.cu:77`

### Triệu chứng

SPR kernel crash với `CUDA error: an illegal memory access was encountered` tại
`gpu_spr.cu:736` (`cudaStreamSynchronize` sau kernel). Xảy ra ngay khi SPR kernel bắt đầu.

### Root cause

Tại `gpu_spr.cu:77`, hookup cho `insertParsimony` bị sai:

```cuda
// WRONG (line 77):
gpuHookup(topo->back_vf, vfNnxtFace(q, N), r);

// CORRECT:
gpuHookup(topo->back_vf, vfNnxtFace(p, N), r);
```

CPU `insertParsimony(p, q_cand)` hookup:
- `p->next` (GPU face[1] = `vfNextFace(p)`) ↔ q_cand ✓ (dòng 76 đúng)
- `p->next->next` (GPU face[0] = `vfNnxtFace(p)`) ↔ r ✗ (dòng 77 sai — dùng q thay vì p)

Hệ quả của lỗi:
1. `back_vf[vfNnxtFace(p)] = -1` (từ `removeNodeParsimony`) không được hookup → khi
   `createTiAndNewviewParsimony(p)` chạy, nó đọc `r_child = -1` → `vfToNum(-1, N) = 0` →
   truy cập `parsVect[node_0]` — **illegal memory access** trên GPU.
2. `back_vf[vfNnxtFace(q)]` của candidate node bị ghi sai và **không được khôi phục**
   bởi undo code. Topology bị corrupt vĩnh viễn qua mỗi lần `testInsert`.

Undo code (lines 122-125) chỉ restore `q↔r` và nullify `vfNextFace(p)`, `vfNnxtFace(p)`,
nhưng không restore `back_vf[vfNnxtFace(q)]` về giá trị ban đầu.

### Fix

Thay `vfNnxtFace(q, N)` bằng `vfNnxtFace(p, N)` tại `gpu_spr.cu:77`.

*(Đây là bug chặn chính cần fix trước. Sau khi fix bug này, cần xác nhận lại Bug #4.)*

### Bài học / Ghi chú cho khóa luận

Đây là loại lỗi "typo trong tên biến" (q thay vì p) trong GPU kernel — rất khó debug vì:
- GPU không có segfault ngay lập tức khi đọc out-of-bounds (không có MMU per-thread)
- Lỗi chỉ xuất hiện sau `cudaStreamSynchronize`, không phải tại điểm xảy ra
- Stack trace GPU không có; chỉ biết lỗi xảy ra trong quá trình sync

Bài học: trong kernel, tên biến `p` (pruned node) và `q` (candidate edge) xuất hiện cùng nhau
rất dễ nhầm. Cần review kỹ mọi `vfNnxt*` / `vfNext*` call trong hookup để đảm bảo đúng node.

---

## Bug #5 — Thiếu `__syncwarp()` trong `createTiAndEvaluateParsimony`

**Ngày**: 2026-05-07
**Task**: Hàm helper trong `pars_tree.cuh`
**File liên quan**: `gpu/include/pars_tree.cuh` (cuối hàm `createTiAndEvaluateParsimony`)

### Triệu chứng

Kết quả SPR không ổn định hoặc sai do race condition giữa lane 0 (viết `sh.ti[]`) và các
lane khác (đọc `sh.ti[]` trong `newviewParsimony`).

### Root cause

Hàm `createTiAndEvaluateParsimony` chạy `computeTraversalInfoParsimony` chỉ từ lane 0 (ghi
vào `sh.ti[]`, `sh.tiSize`), sau đó gọi `newviewParsimony` mà không có `__syncwarp()` trước.
Các lane 1-31 có thể đọc `sh.ti[]` trước khi lane 0 ghi xong → undefined behavior.

Tương tự, `createTiAndNewviewParsimony` đã có `__syncwarp()` ở dòng 407 ✓, nhưng
`createTiAndEvaluateParsimony` thiếu.

### Fix

Thêm `__syncwarp()` ngay sau block `if (lane == 0) { ... computeTraversalInfoParsimony ... }`
và trước `return newviewParsimony(...)` trong `createTiAndEvaluateParsimony`.

### Bài học / Ghi chú cho khóa luận

Mọi nơi lane 0 ghi vào shared memory và các lane khác sẽ đọc, cần có `__syncwarp()` giữa.
Trong CUDA warp, không có đảm bảo về thứ tự memory visibility giữa các thread trong warp trừ
khi có explicit sync. `__syncwarp()` là barrier nhẹ (chỉ cho warp, không cho block) nhưng đủ
khi tất cả 32 lanes thuộc cùng một warp.

---

## Refactor #1 — Joined kernel: SPR vào buildParsimonyTreesKernel, chung BuildShared

**Ngày**: 2026-05-07
**Task**: Tối ưu hóa kiến trúc — join SPR vào sau build để tận dụng shared memory chung
**File liên quan**: `gpu/src/pars_build.cu`, `gpu/include/pars_tree.cuh`, `gpu/src/gpu_spr.cu`

### Thay đổi

Trước: hai kernel riêng biệt — `buildParsimonyTreesKernel` và `gpuSprKernel`, mỗi cái có
struct shared memory riêng (`BuildShared` và `SprShared`). Giữa hai kernel phải đồng bộ CPU.

Sau: một kernel duy nhất `buildParsimonyTreesKernel(sprDist)` gồm 4 phase:
- Phase 0: init (Fisher-Yates shuffle, 3-tip tree)
- Phase 1: stepwise addition (tips 4..N)
- Phase 2: set `start_vface`
- Phase 3: SPR hill-climbing (if `sprDist > 0`)

### Thiết kế BuildShared mới (≈24.8 KB trên A100)

`SprShared` được xóa bỏ, các field của nó được merge vào `BuildShared`:
- `stack[]` (2×kMaxTaxa) — build DFS stack, tái dụng làm SPR `stackVf`
- `stackMint[]`, `stackMaxt[]` (kMaxTaxa mỗi) — SPR addTraverse / recompute child_index
- `seed` (long) — từ biến local thành shared, kế thừa từ build phase sang SPR
- `randomMP`, `randomMPHits`, `bestRemoveVf`, `bestInsertVf`, `tip_p_num`, `bcast[16]` — SPR fields
- `bestPars` đổi tên thành `bestParsimony` — dùng cho cả hai phase

### Init `sh.randomMP` không cần `recomputeAllNodes`

Sau Phase 1, `sh.bestParsimony` là điểm parsimony hợp lệ của cây (từ lần evaluate cuối
trong stepwise addition), và xPars flags đã nhất quán với parsVect. Do đó:
```cuda
sh.randomMP = sh.bestParsimony;  // valid: xPars consistent after build
```

Đây là insight quan trọng: `recomputeAllNodes` (O(N) work) KHÔNG cần thiết nếu xPars system
đã nhất quán từ phase trước. Tiết kiệm N−1 newview calls per tree.

### Kết quả

| Metric | Trước (2 kernel) | Sau (joined) |
|--------|-----------------|--------------|
| Shared memory | BuildShared + SprShared | BuildShared duy nhất (24.8 KB) |
| CPU sync giữa build-SPR | 1 lần | Không cần |
| Post-SPR best (N=295, K=99) | 16369 (sai) | **6676** (≈CPU 6682) |
| ms/tree | 79 ms/tree (build) + fail | 79.3 ms/tree (build+SPR) |

### Bài học / Ghi chú cho khóa luận

**Ưu điểm join kernel**:
1. Loại bỏ CPU synchronization overhead giữa hai kernel (cudaStreamSynchronize)
2. Shared memory được tái dụng — không cần tải lại parsVect/score_tree
3. xPars state được kế thừa — không cần recomputeAllNodes
4. Đơn giản hóa API: một hàm `gpuStepwiseBuildTrees(mem, seeds, sprDist, stream)`

**Lưu ý cho khóa luận**: Việc join kernel không chỉ là optimization mà còn là architectural
change — nó khai thác tính sequential của build→SPR để chia sẻ state (xPars, seed, parsVect)
mà không cần round-trip qua global memory. Đây là pattern quan trọng trong GPU kernel design.

---

## Kết quả thực nghiệm — Scaling theo số cây K (2026-05-07)

**Ngày**: 2026-05-07
**Dataset**: `data_debug/tree1.phy` — N=295 taxa, 1836 columns, 1400 patterns (DNA)
**Hardware**: NVIDIA A100-SXM4-80GB
**Tham số**: sprDist=6, seed=1
**CPU reference** (`_pllSprOnCurrentTree`, 1 cây): best parsimony ≈ **6668**

### Bảng kết quả

| K (số cây GPU) | Post-SPR best | ms/tree | Tổng thời gian |
|----------------|--------------|---------|----------------|
| 99             | 6676         | 79.3 ms | ~7.9 s         |
| 199            | 6671         | 44.9 ms | ~9.0 s         |
| 499            | 6671         | 25.0 ms | 12.5 s         |
| 999            | 6670         | 21.2 ms | 21.2 s         |
| **9999**       | **6664**     | **16.3 ms** | **163 s**   |
| CPU ref (1 cây) | ~6668       | —       | —              |

### Kết quả quan trọng

**Với K=9999: GPU đạt best=6664 — vượt CPU reference 6668.**
Đây là lần đầu tiên GPU tìm được cây tốt hơn CPU sau khi tăng số lượng cây.

### Phân tích scaling

**ms/tree giảm theo K:**
- 99→999 cây: 79→21 ms/tree (giảm 3.7×) — tốt hơn tuyến tính
- 999→9999 cây: 21→16 ms/tree (giảm 1.3×) — gần bão hòa

GPU A100 có 108 SM × nhiều warps/SM. Khi K nhỏ (99 blocks), nhiều SM idle.
Khi K lớn (9999 blocks), tất cả SM đều bận → throughput tối đa đạt ~16 ms/tree.

**Post-SPR best cải thiện theo K:**
Càng nhiều cây khởi đầu từ random seeds khác nhau → tìm kiếm bao phủ landscape
parsimony rộng hơn → tìm được optimum tốt hơn. Đây là lợi thế cốt lõi của GPU
parallelism so với CPU serial: GPU có thể chạy 9999 independent SPR hill-climbs
trong ~163 giây.

**So sánh tổng thời gian:**
- CPU chạy 9999 cây serial: 9999 × (thời gian 1 cây CPU) ≈ 9999 × ~0.2s ≈ **33 phút**
- GPU chạy 9999 cây parallel: **163 giây (2.7 phút)** — speedup ~12×

### Bài học / Ghi chú cho khóa luận

**Ý nghĩa chính của kết quả này:**

1. **Correctness**: GPU SPR đúng thuật toán và cho kết quả không tệ hơn CPU (với đủ cây,
   GPU tìm được cây TỐT HƠN CPU vì khám phá nhiều starting points hơn).

2. **Scalability**: ms/tree giảm từ 79 ms (K=99) xuống 16 ms (K=9999), gần tuyến tính
   theo số SM. GPU amortizes overhead (upload, alloc) tốt hơn khi K lớn.

3. **Practical impact**: Thay vì chạy serial 9999 cây (33 phút), GPU chạy 9999 cây trong
   2.7 phút — **speedup 12×**. Với mục tiêu MPBoot (tìm cây parsimony tốt nhất nhanh nhất),
   đây là kết quả có giá trị thực tiễn cao.

4. **Limitation**: Tổng thời gian tăng tuyến tính theo K (kernel time), nên không thể tăng K
   vô hạn. Điểm "sweet spot" phụ thuộc vào yêu cầu chất lượng vs. thời gian: K=999 cho
   kết quả gần CPU (6670) trong 21 giây là cân bằng tốt.

---

## Refactor #2 — Thêm `nodep[kMaxNodes]` vào `GpuTopology`; rewrite `gpuNodeRectifierPars`

**Ngày**: 2026-05-11
**Task**: Cấu trúc lại cách GPU lưu hướng DFS-canonical của mỗi inner node — mirror CPU `tr->nodep[]`
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`

### Thay đổi

Trước: GPU cố định canonical face = `nodepVf(num, N)` = face[2] và `gpuNodeRectifierPars` cũ
PERMUTE `back_vf[base+0,1,2]` để face[2] luôn hướng về DFS-parent. Cách này gây ra bug vì
testInsert dùng `vfNnxtFace(p)` (= face[0]) để evaluate, nhưng sau permutation face[0]
không còn là child1 hợp lệ nữa.

Sau: Thêm field `int nodep[kMaxNodes]` vào `GpuTopology` (giống CPU `tr->nodep[]`). Field này
lưu DFS-encountered vface của mỗi node:
- Tips: `nodep[num] = num - 1` (vface tương ứng tip, cố định)
- Inner nodes: `nodep[num]` = vface DFS vào node `num`, set bởi `gpuNodeRectifierPars`

`gpuNodeRectifierPars` được rewrite hoàn toàn:
- DFS từ `nodep[1]->back` (back của node 1)
- Gán `nodep[count + N + 1] = m_vf` (DFS-encountered face)
- KHÔNG thay đổi `back_vf` hay `xpars`
- Match CPU `reorderNodes(sprparsimony.cpp:2052)` theo logic count-based

`gpuSPRHillClimb` updated:
- `topo->start_vface = topo->nodep[1]` (thay vì `nodepVf(1, N)`)
- `sh.bcast[4] = topo->nodep[i]` (thay vì `nodepVf(i, N)`)

Phase 0 khởi tạo `nodep[]` cho tips và inner nodes.

### Kết quả

Post-SPR best = 6665 (K=100, N=295) — đúng, match kết quả trước refactor.

### Bài học / Ghi chú cho khóa luận

CPU `tr->nodep[i]` không phải là con trỏ cố định theo số node — nó là face mà DFS vào node đó.
Sau `nodeRectifierPars`, `nodep[i]` trỏ đến face cụ thể của node i từ đó `p->back` là DFS-parent.
GPU phải replicate cùng semantic này; cố định canonical = face[2] không đủ vì ring arithmetic
phụ thuộc vào face nào là "entry face" của DFS. Giải pháp đúng: lưu face trong `nodep[]` thay
vì hard-code, đúng như CPU làm.

---

## Bug #7 — `q_num >= N` trong `doAddTraverse`: dùng node-number comparison thay vì vface comparison

**Ngày**: 2026-05-11
**Task**: `doAddTraverse` trong `pars_build.cu`
**File liên quan**: `gpu/src/pars_build.cu` (line ~111)

### Triệu chứng

SPR traverse sai vào tip node (node number N = tip N): tip N có node number = N, nên
`q_num >= N` = true, nhưng `q_num > N` = false. Dùng `>=` khiến code cố gắng push children
của tip N lên stack — với tip vface = N-1, `vfNextFace(N-1, N) = N-1` (self), nên push
`back_vf[N-1]` (inner neighbor của tip N) lên stack 2 lần, gây duplicate evaluations.

### Root cause

Nhầm lẫn giữa hai loại check:
- **Vface check** (inner vface): `vf >= N` (đúng, inner vfaces bắt đầu từ N)
- **Node number check** (inner node): `num > N` (đúng, inner nodes có number N+1..2N-1)

`q_num = vfToNum(cur_q, N)` là **node number**, không phải vface. Phải dùng `q_num > N`,
không phải `q_num >= N`. Tip N (cuối cùng trong danh sách tips 1..N) có `num = N`, nên:
- `q_num >= N` → true ← sai, tip N được coi là inner
- `q_num > N` → false ← đúng, tip N bị skip

### Fix

Thay `if (q_num >= N && maxt > 0)` thành `if (q_num > N && maxt > 0)`.

### Bài học / Ghi chú cho khóa luận

Đây là nguồn gốc của một lớp bug tinh vi trong GPU parsimony code: tip vfaces có index 0..N-1
(tức vf < N), inner vfaces có index N..N+3*(N-2)+2 (tức vf >= N). Nhưng node numbers: tips
có num 1..N (tức num <= N), inner nodes có num N+1..2N-1 (tức num > N). Hai boundary condition
khác nhau một đơn vị và dễ nhầm. Quy tắc:
- Nếu biến là **vface**: dùng `>= N` cho inner
- Nếu biến là **node number**: dùng `> N` cho inner

Luôn kiểm tra: biến đang so sánh là vface hay node number?

---

## Kết quả thực nghiệm — Scaling sau Refactor #2 + Bug #6-8 fixes (2026-05-11)

**Ngày**: 2026-05-11
**Dataset**: `data_debug/tree1.phy` — N=295 taxa, 1836 columns, 1400 patterns (DNA)
**Hardware**: NVIDIA A100-SXM4-80GB
**Tham số**: sprDist=6, seed=1, iters=100, NNI=29, sprDist4=3, maxDW=5
**CPU reference** (`_pllSprOnCurrentTree`, 1 cây): best parsimony ≈ **6668**

### Bảng kết quả

| K (số cây GPU) | Thực chạy | Post-SPR best | Pre-SPR best | ms/tree | Tổng thời gian |
|----------------|-----------|--------------|-------------|---------|----------------|
| 100            | 99        | **6665**     | 6734        | 83.21 ms | 8.24 s        |
| 200            | 199       | **6665**     | 6723        | 39.55 ms | 7.87 s        |
| 500            | 499       | **6665**     | 6713        | 24.15 ms | 12.05 s       |
| 1000           | 999       | **6665**     | 6713        | 23.09 ms | 23.07 s       |
| **10000**      | **9999**  | **6662**     | 6713        | **20.76 ms** | **207.6 s** |
| CPU ref (1 cây) | —        | ~6668        | —           | —        | —              |

### Kết quả quan trọng

**K=10000: GPU best=6662 — vượt CPU reference 6668.**
Post-SPR best ổn định ở 6665 với K=100..1000, cải thiện thêm khi K=10000 (6662).

### So sánh với kết quả trước (2026-05-07, trước Refactor #2)

| K | Post-SPR best cũ | ms/tree cũ | Post-SPR best mới | ms/tree mới | Nhận xét |
|---|-----------------|-----------|------------------|------------|---------|
| ~100 | 6676 | 79.3 ms | 6665 | 83.21 ms | Best cải thiện; ms/tree tăng nhẹ |
| ~1000 | 6670 | 21.2 ms | 6665 | 23.09 ms | Best cải thiện; overhead nhỏ |
| ~10000 | 6664 | 16.3 ms | 6662 | 20.76 ms | Best cải thiện; ms/tree cao hơn |

ms/tree tăng nhẹ (~4 ms) so với trước refactor — chi phí từ `gpuNodeRectifierPars` mới
(DFS gán `nodep[]` sau mỗi do-while iteration) và mảng `nodep[kMaxNodes]` bổ sung vào
`GpuTopology` (tăng shared memory bandwidth).

### Phân tích scaling

**ms/tree giảm theo K:**
- 100→500 cây: 83→24 ms/tree (giảm 3.5×) — GPU SM occupancy tăng rõ rệt
- 500→1000 cây: 24→23 ms/tree (gần bão hòa ở ~108 SM × warps/SM)
- 1000→10000 cây: 23→21 ms/tree (bão hòa — SM overhead amortized)

**Pre-SPR best hội tụ từ K=500**: Pre-SPR best ổn định ở 6713 khi K ≥ 500 —
stepwise addition bão hòa về chất lượng; cải thiện thêm đến từ SPR.

**Post-SPR best cải thiện chậm hơn**: Từ 6665 (K=100) xuống 6662 (K=10000) —
landscape parsimony có nhiều local optima tốt ở 6665; cần rất nhiều starting points
để lọt qua vào optima tốt hơn. Đây là tính chất NP-hard của bài toán.

**So sánh với CPU serial:**
- CPU 10000 cây serial: ~10000 × 0.2s ≈ **33 phút**
- GPU 10000 cây: **207 giây (3.5 phút)** — speedup ~**9.6×**

### Bài học / Ghi chú cho khóa luận

1. **Correctness**: Sau Refactor #2 và Bug #6-8, GPU SPR cho kết quả đúng thuật toán
   và tìm được cây tốt hơn CPU với đủ starting points (6662 < 6668).

2. **Overhead của nodep[]**: Thêm mảng `nodep[]` và DFS của `gpuNodeRectifierPars` tăng
   ms/tree ~4 ms (~20% overhead với K=10000). Đây là trade-off chấp nhận được để đảm bảo
   correctness (ring arithmetic đúng với mọi face).

3. **Scaling behavior**: ms/tree bão hòa ở ~20-21 ms sau K≈500. Điểm "sweet spot":
   - K=500: 24 ms/tree, best=6665, tổng 12s — nhanh, chất lượng tốt
   - K=1000: 23 ms/tree, best=6665, tổng 23s — balanced
   - K=10000: 21 ms/tree, best=6662, tổng 207s — best quality, tốn thời gian

4. **GPU speedup ~9.6×** so với CPU serial (thực tế có thể cao hơn vì CPU reference
   chỉ chạy 1 cây với search parameters đầy đủ — 10000 cây CPU sẽ còn chậm hơn).

---

## Kết quả thực nghiệm — CPU serial baseline (branch mpboot gốc, 2026-05-11)

**Ngày**: 2026-05-11
**Dataset**: `data_debug/tree1.phy` — N=295 taxa, 1836 columns, 1400 patterns (DNA)
**Hardware**: CPU serial (1 core), không dùng GPU
**Tham số**: sprDist=6, seed=1 (`_pllComputeRandomizedStepwiseAdditionParsimonyTree`)
**Code**: branch mpboot gốc — `initCandidateTreeSet` single-loop gốc

### Bảng kết quả

| K (numpars) | Thực chạy | ms/tree | Tổng thời gian |
|-------------|-----------|---------|----------------|
| 100         | 99        | 44.2 ms | 4.38 s         |
| 200         | 199       | 45.0 ms | 8.95 s         |
| 500         | 499       | 43.7 ms | 21.80 s        |
| 1000        | 999       | 43.8 ms | 43.73 s        |
| **10000**   | **9999**  | **45.3 ms** | **453.41 s** |

### Kết quả quan trọng

**ms/tree bằng phẳng ~44ms ở mọi K** — CPU serial là thuần tuyến tính O(K).
Post-initCandidateSet best parsimony = **6668** (nhất quán mọi K với seed=1).
Không có lợi thế scaling: chạy 10000 cây = chạy 10 lần × 1000 cây.

### So sánh với GPU (A100-SXM4-80GB, 2026-05-11)

| K | CPU serial (ms/tree) | GPU (ms/tree) | GPU total | CPU total | Speedup tổng |
|---|---------------------|---------------|-----------|-----------|-------------|
| 100 | 44.2 ms | 83.2 ms | 8.24 s | 4.38 s | 0.53× |
| 200 | 45.0 ms | 39.6 ms | 7.87 s | 8.95 s | 1.14× |
| 500 | 43.7 ms | 24.2 ms | 12.05 s | 21.80 s | 1.81× |
| 1000 | 43.8 ms | 23.1 ms | 23.07 s | 43.73 s | **1.90×** |
| **10000** | **45.3 ms** | **20.8 ms** | **207.6 s** | **453.4 s** | **2.18×** |

**GPU breakeven điểm ~K=150**: dưới đó GPU chậm hơn CPU do overhead khởi tạo (upload parsVect, alloc, kernel launch). Trên K≈500 GPU vượt CPU rõ rệt.

**K=10000: GPU tổng thời gian 207s vs CPU 453s → GPU speedup tổng 2.18×.**

### Phân tích

**Tại sao GPU speedup chỉ ~2× thay vì 10×+?**
- CPU single-loop `_pllComputeRandomizedStepwiseAdditionParsimonyTree` chạy fast (~44ms/tree)
  nhờ: xPars lazy eval bỏ qua newview cho subtrees đã fresh; CPU cache locality tốt.
- GPU ~21ms/tree là bottleneck tính toán thực sự (A100 SM saturation với K≥500).
- Overhead GPU: upload 295 tips × parsVect + alloc + kernel launch ≈ vài giây cố định.
- Lợi thế GPU không phải tốc độ per-tree mà là **chất lượng**: K=10000 GPU best=6662
  (sau SPR) vs CPU K=10000 best=6668 (chỉ stepwise+SPR, chưa NNI).

**Tại sao hai-phase code (branch GPU cũ trước fix) chậm ~900ms/tree?**

Branch GPU trung gian dùng cấu trúc hai phase:
- Phase 1: build tất cả K cây, lưu Newick strings vào `vector<string>`
- Phase 2: với mỗi cây, gọi `readTreeString(candidateTrees[i])`

Khi Phase 2 xử lý cây i, `pllInst` đang giữ topology của **cây N** (cây cuối build).
`readTreeString` phải:
1. `freeNode()` — giải phóng topology hiện tại
2. `readTree()` — parse Newick và rebuild topology từ đầu
3. `pllTreeInitTopologyNewick()` — sync PLL topology (O(N))

Ngược lại trong code gốc single-loop, `readTreeString` được gọi ngay sau build — `pllInst`
đang giữ đúng topology → bước rebuild rất nhẹ, chỉ sync metadata pointer mapping.

Fix: khôi phục single-loop inline registration (branch GPU branch hiện tại đã fix).

### Bài học / Ghi chú cho khóa luận

1. **CPU baseline thực sự ~44ms/tree** — con số "~0.2s/tree" từ trước đó ước tính quá cao.
   Speedup GPU so với CPU là ~2.2× về tổng thời gian (K=10000), không phải 12×.

2. **Ưu thế GPU chủ yếu là quality, không phải speed**: GPU với K=10000 tìm best=6662
   (sau SPR) — bằng CPU+NNI-refined. CPU K=10000 serial chỉ cho best=6668.
   GPU khám phá nhiều starting points hơn trong cùng thời gian → landscape coverage tốt hơn.

3. **readTreeString cost phụ thuộc ngữ cảnh**: Gọi khi pllInst khớp topology → O(N) nhẹ.
   Gọi khi pllInst có topology khác → O(N) nặng + PLL rebuild. Không thể nhìn code và
   biết cost mà không biết state của pllInst tại thời điểm gọi.

---

## Bug #8 — `node_num >= N` trong stepwise addition: cùng lỗi vf vs. num

**Ngày**: 2026-05-11
**Task**: Phase 1 (stepwise addition) trong `buildParsimonyTreesKernel`
**File liên quan**: `gpu/src/pars_build.cu` (Phase 1 DFS stack push, line ~673)

### Triệu chứng

Stepwise addition có thể push vào stack children của tip N hai lần (vì tip N có `node_num = N`,
thỏa `node_num >= N`). Tuy nhiên điều kiện `score_tree[node_num] > 0` che khuất bug này
(tips luôn có `score_tree = 0` → điều kiện false), nên không gây lỗi observable.

### Root cause

Cùng nhầm lẫn như Bug #7: `node_num >= N` thay vì `node_num > N`.

### Fix

Thay `if (node_num >= N && score_tree[node_num] > 0)` thành
`if (node_num > N && score_tree[node_num] > 0)`.

### Bài học

Dù bug bị che khuất bởi guard condition khác, cần sửa để code đúng về mặt semantic và tránh
phụ thuộc vào điều kiện vô tình "che" lỗi. Code đúng phải đúng từ nguyên tắc, không chỉ đúng
vì side effect của điều kiện khác.
---

## Refactor #3 — Thống nhất `sprDist` + thêm `postSprParsimony` + timing breakdown (2026-05-11)

**Ngày**: 2026-05-11
**Task**: Clean up API, thêm metric, thêm timing instrumentation vào kernel
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/include/pars_build.cuh`, `gpu/src/pars_build.cu`, `gpu/src/gpu_init_trees.cu`

### Thay đổi

**1. Thống nhất `sprDist`** — xóa `sprDist4` / `-gpu_hc_spr_dist`: tất cả 3 phase (build SPR, NNI+SPR, Ratchet) dùng chung `params.sprDist` (CLI: `-sprdist`). Hàm `gpuStepwiseBuildTrees` không còn nhận `int sprDist4`.

**2. Thêm `GpuTopology::postSprParsimony`** — lưu điểm parsimony sau Phase 2 (initial SPR) và trước Phase 3 (hill-climbing). Ba metric riêng biệt:
- `preSprParsimony` — sau stepwise addition (build)
- `postSprParsimony` — sau Phase 2 SPR
- `bestParsimony` — sau Phase 3 NNI+ratchet

**3. Thêm timing breakdown per-phase vào `BuildShared`** (block 0 only, clock cycles):

| Field | Ý nghĩa |
|-------|---------|
| `t_build` | Phase 1: stepwise addition |
| `t_phase2` | Phase 2: initial SPR |
| `t_p3_nni` | Phase 3 even iters: NNI+setup |
| `t_p3_nni_spr` | Phase 3 even iters: SPR |
| `t_p3_ratchet` | Phase 3 odd iters: tổng ratchet |
| `n_p3_even` / `n_p3_odd` | số lần lặp mỗi loại |
| `t_line2291`, `t_search`, `t_apply` | nội bộ SPR (đã có) |

Output in ra từ kernel (block 0, lane 0):
```
[TIMING k=0] Phase1 build=XXX cyc
[TIMING k=0] Phase2 initial-SPR=XXX cyc
[TIMING k=0] Phase3 NNI+SPR (5 iters): nni_setup=XXX cyc  spr=XXX cyc
[TIMING k=0] Phase3 Ratchet  (5 iters): total=XXX cyc
[TIMING k=0] Phase3 SPR breakdown: line2291=XXX (11.2%)  search=XXX (88.7%)  apply=XXX (0.0%)  moves=336  dowhile=3
```

### Kết quả timing điển hình (N=295, K=199, sprDist=3, gpu_hc_iter=10)

| Phase | Clock cycles | Ghi chú |
|-------|-------------|---------|
| Phase 1 build | 2.376B | stepwise addition N=295 |
| Phase 2 initial-SPR | 1.871B | 1 lần SPR hill-climb |
| Phase 3 NNI+SPR (5 iters) — setup | 22.5M | NNI nhỏ so với SPR |
| Phase 3 NNI+SPR (5 iters) — SPR | 9.539B | mỗi iter ~1.9B |
| Phase 3 Ratchet (5 iters) | 13.237B | 2 SPR calls → ~2× so với 1 |
| SPR breakdown: search | 88.7% | `doAddTraverse` = bottleneck chính |
| SPR breakdown: line2291 | 11.2% | lazy refresh overhead |
| SPR breakdown: apply | 0.05% | `applyMove` không đáng kể |

### Bài học / Ghi chú cho khóa luận

`doAddTraverse` (candidate edge search) chiếm gần 90% thời gian SPR. Đây là phần cần tối ưu trước tiên nếu muốn tăng tốc SPR. Phần `applyMove` (chỉ 0.05%) gần như free — bottleneck không phải là số lần apply move mà là số lần evaluate candidate.

---

## Experiment Log — Phase 3 Hill-Climbing Benchmark (2026-05-11)

**Dataset**: `data_debug/tree1.phy` (N=295 taxa, 1140 sites)
**Settings**: `-seed 1 -numpars 100` (K=99 trees), Phase 2 `sprDist=6`
**GPU**: A100-SXM4-80GB

### Baseline

| Config | best | worst | ms/tree | Total |
|--------|------|-------|---------|-------|
| Phase 2 only (sprDist=6, no Phase 3) | 6665 | — | 79 ms | 7.9s |

### Phase 3 experiments (`-gpu_hc_iter` × `-gpu_hc_spr_dist`)

| gpu_hc_iter | gpu_hc_spr_dist | best | worst | ms/tree | Total | moves |
|-------------|-----------------|------|-------|---------|-------|-------|
| 10 | 4 | 6662 | 6684 | 881 ms | 90s | 384 |
| 10 | 6 | 6662 | 6685 | 1426 ms | 145s | 356 |
| 20 | 4 | 6662 | 6684 | 1385 ms | 140s | 704 |
| 20 | 6 | 6662 | 6683 | 2144 ms | 215s | 687 |
| 50 | 4 | 6662 | 6680 | 2370 ms | 238s | 1590 |
| **50** | **6** | **6662** | **6666** | **3282 ms** | **330s** | **1654** |
| 100 | 3 | 6662 | 6681 | 1254 ms | 124s | 2827 |

CPU reference (1 tree, serial): best ≈ 6668

### Quan sát

1. **`best` hội tụ sớm**: mọi cấu hình đều cho `best=6662`, từ 10 đến 100 iterations.
   Phase 3 cải thiện 3 điểm so với Phase 2 (6665→6662), nhưng tăng thêm iter không cải thiện `best`.

2. **`worst` giảm dần theo iter và sprDist**: tăng iter từ 10→50 và sprDist từ 4→6
   cải thiện worst-case đáng kể (6685 → 6666). Với `iter=50, spr=6`, worst=6666 gần bằng best=6662
   → phân phối chất lượng cây rất đồng đều.

3. **Trade-off thời gian**: sprDist=6 tốn ~50-60% thêm thời gian so với sprDist=4 cùng iter.
   Iter tỉ lệ tuyến tính với thời gian.

4. **Khuyến nghị**: `iter=10, spr=4` (880 ms/tree) nếu ưu tiên tốc độ;
   `iter=50, spr=6` (3282 ms/tree) nếu muốn worst-case tốt nhất.
   `best` không thay đổi theo cấu hình nào trong số này.

5. **So với CPU**: GPU K=99 trees best=6662 tốt hơn CPU single-tree best≈6668 (seed=1).

---

## Optimization #1 — `testInsert` pre-refresh: loại bỏ Fitch step dư thừa cho remove node

**Ngày**: 2026-05-12
**Task**: Tối ưu hóa `testInsert` trong `gpu/src/pars_build.cu` (Phase 3 SPR hill-climbing)
**File liên quan**: `gpu/src/pars_build.cu` (hàm `testInsert`, ~line 45-104)

### Phân tích root cause

Timing instrumentation (N=295, sprDist=3, K=99, 5 even iters) cho thấy:

```
[TIMING k=0] testInsert avg traversal: newview=2.51 nodes  eval=1.00 nodes
```

`testInsert` gọi hai bước liên tiếp:

| Bước | Hàm | Mục đích | Cost |
|------|-----|---------|------|
| 1 | `createTiAndNewviewParsimony(p, face[2])` | (A) refresh q/r nếu stale; (B) tính parsVect[p] từ face[2] | 2.51 node/call |
| 2 | `createTiAndEvaluateParsimony(face[0], lazy)` | tính parsVect[p] từ face[0] + evaluate tại edge | 1.00 node/call |

**Lãng phí**: Bước 1 tính `parsVect[p_num]` từ face[2], nhưng bước 2 lập tức ghi đè bằng tính từ face[0].
Một Fitch step (1.00 node) là hoàn toàn dư thừa — không được dùng đến.

Trong 2.51 node trung bình của bước 1:
- ~1.51 node: traverse subtree của cur_q và r (cần thiết — refresh xPars)
- ~1.00 node: tính parsVect[p] từ face[2] (lãng phí)

### Fix: Pre-refresh block thay thế bước 1

Thay bước 1 bằng **pre-refresh block** chỉ refresh q, r_vf, và `tip_p = back_vf[p]` (3 children
mà step 2 cần fresh), mà KHÔNG tính parsVect[p] từ face[2]:

```cpp
if (lane == 0) {
    const int r_vf     = sh.bcast[0];
    const int tip_p_vf = topo->back_vf[p];  // back của remove node, không đổi khi hookup
    sh.tiSize = 3;
    if (q >= N && !topo->xpars[q])          computeTraversalInfoParsimony(q);
    if (r_vf >= N && !topo->xpars[r_vf])    computeTraversalInfoParsimony(r_vf);
    if (tip_p_vf >= N && !topo->xpars[tip_p_vf]) computeTraversalInfoParsimony(tip_p_vf);
}
__syncwarp();
if (sh.tiSize > 3) newviewParsimony(..., evaluate=false);
if (lane == 0) {
    topo->xpars[q] = 1;
    topo->xpars[r_vf] = 1;
    topo->xpars[tip_p_vf] = 1;
    topo->xpars[vfNnxtFace(p, N)] = 0;  // buộc step 2 tính lại p từ face[0]
}
```

### Bug tìm được trong quá trình fix: `tip_p` degradation

**Triệu chứng**: Sau khi áp dụng pre-refresh (chỉ refresh q và r_vf, chưa có tip_p), eval_size tăng lên 1.72 thay vì 1.00 mong đợi.

**Root cause**: Trong `createTiAndEvaluateParsimony(face0, lazy)` (step 2), face0 cần tính:
```
parsVect[p_num] = Fitch(parsVect[tip_p_num], parsVect[q_num])
```
`face0`'s children (theo vface ring) là:
- `back_vf[vfNextFace(face0)]` = `back_vf[p_vf]` = **tip_p**
- `back_vf[vfNnxtFace(face0)]` = **q** (sau hookup)

Nếu `xpars[tip_p_vf] = 0` (stale), `computeTraversalInfoParsimony(face0)` sẽ push tip_p vào stack → traverse thêm một node → eval_size = 2.

**Tại sao tip_p bị stale?** Code cũ (bước 1 = `createTiAndNewviewParsimony(p, face[2])`) vô tình duy trì xpars[tip_p] = 1 thông qua cơ chế xPars move: mỗi lần bước 1 chạy, nó traverse từ face[2] với children là cur_q và r_vf (sau hookup), và xPars move logic di chuyển xpars vào face[2], gián tiếp giữ xpars[tip_p] không bị degrade. Bỏ bước 1 mà không thêm refresh tip_p → tip_p dần degraded sau hàng nghìn calls.

**Fix**: Thêm `tip_p_vf = back_vf[p]` vào pre-refresh block (refresh nếu stale, set xpars=1 sau đó).

### Pitfall: `if (log)` gây regression 48%

Khi cleanup debug code, cấu trúc step 2 được đổi thành:
```cpp
if (log) { /* inlined step 2 */ }
else     { mp = createTiAndEvaluateParsimony(...); }
```

**Kết quả**: search time tăng từ 5.5B → 9.4B cyc (+48%). Dù `log` là warp-uniform (block 0 = true, các block khác = false), CUDA compiler vẫn sinh code cho CẢ HAI nhánh vì nó không biết giá trị runtime của `log`. Điều này gây:
1. Tăng I-cache pressure do code lớn hơn
2. Register allocation kém hơn do compiler cần giữ biến qua 2 nhánh

**Fix**: Luôn dùng inline cho step 2 (không dùng `if/else` trên `log`); chỉ guard phần timing accumulation bằng `if (log)`.

**Bài học**: Trong CUDA, `if (runtime_flag) { path_A } else { path_B }` — ngay cả khi `runtime_flag` uniform trong warp — vẫn có thể gây regression do compiler không biết tại compile time. Nếu cả hai nhánh làm cùng tính toán cơ bản (chỉ khác phần debug), luôn inline phần chung và guard phần debug.

### Kết quả

**Timing** (N=295, sprDist=3, K=99, 5 even iters Phase 3):

| Metric | Trước (bước 1 cũ) | Sau (pre-refresh) | Δ |
|--------|-------------------|-------------------|---|
| newview nodes/call | 2.51 | **1.72** | −31% |
| eval nodes/call | 1.00 | **1.00** ✓ | 0% |
| Phase3 SPR search | 8.296B cyc | **~5.5B cyc** | **−34%** |

**Correctness**: post-hc best parsimony = 6662 ✓ (không thay đổi, đúng với nhiều seeds).

### Bài học / Ghi chú cho khóa luận

1. **Xác định lãng phí bằng timing instrumentation**: Đo `n_ti_newview_size / n_testInsert` (avg newview traversal) và `n_ti_eval_size / n_testInsert` (avg eval traversal) cho thấy rõ 1.00 node dư thừa trong newview. Timing chi tiết per-step (t_ti_newview, t_ti_eval) xác nhận phân bổ chi phí.

2. **xPars invariant phải maintained cho tất cả children của face cần evaluate**: Khi tối ưu bằng cách bỏ một bước, phải kiểm tra kỹ xem bước đó có "side-effect" nào duy trì invariant. Ở đây bước 1 vô tình duy trì xpars[tip_p]=1; bỏ nó mà không thêm refresh tip_p phá vỡ invariant.

3. **CUDA compiler divergence với uniform boolean**: `if (uniform_flag)` vẫn có thể gây performance regression. Cách an toàn: inlining code chung, guard debug code bằng `if (flag)`.

4. **Children của face[0] trong testInsert**: face0 = `vfNnxtFace(p, N)`. Children = `back_vf[vfNextFace(face0)]` = tip_p và `back_vf[vfNnxtFace(face0)]` = q. Cả hai phải fresh (xpars=1) trước khi step 2 chạy. Đây là điều kiện cần thiết để eval_size = 1.00.

---

## Bug #9 — `bestParsimony` báo score tốt hơn `Current best score` do topology drift

**Ngày**: 2026-05-12
**Task**: Benchmark 115 dataset → so sánh CPU/GPU → phát hiện discrepancy
**File liên quan**: `gpu/src/pars_build.cu`, `gpu/src/gpu_init_trees.cu`, `gpu/include/pars_tree.cuh`

### Triệu chứng

Trong 14/115 dataset benchmark, log GPU in ra:
```
[GPU]   [6b] Post-Hillclimbing parsimony: best=2938  worst=3007
Current best score: 2941 / CPU time: 5
```
`Post-Hillclimbing best` (2938) nhỏ hơn (tốt hơn) `Current best score` (2941). Delta: −1 đến −46 units.
Hướng lệch luôn nhất quán: post-HC luôn tốt hơn hoặc bằng final score, không bao giờ ngược lại.

### Root cause — Topology Drift

Hai luồng dữ liệu hoàn toàn tách biệt:

**1. `topo->bestParsimony` — historical minimum** (`pars_build.cu:899-901`):
```cpp
// Cuối mỗi Phase 3 iteration:
if (lane == 0 && sh.randomMP < topo->bestParsimony)
    topo->bestParsimony = sh.randomMP;  // lưu score tốt nhất từ trước đến nay
```

**2. `downloadTopology` — tải end-state** (`gpu_init_trees.cu:161`):
```cpp
downloadTopology(mem, i, &h_topo, stream);  // tải back_vf[] cuối iteration cuối cùng
```

**3. Bước `[6b]`** đọc `h_topo_tmp.bestParsimony` cho tất cả K trees → in "Post-Hillclimbing best=2938"
(đây là minimum thực sự tìm được trong toàn bộ Phase 3).

**4. CPU re-score end-state topology từ đầu** (`phyloanalysis.cpp:1295-1297`):
```cpp
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -iqtree.computeParsimony();  // → 2941 (topology end-state)
```

**Tại sao topology bị drift?** Phase 3 loop có cấu trúc chẵn/lẻ:
- **Iter chẵn**: NNI perturbation ngẫu nhiên → SPR hill-climb. NNI phá vỡ topology hiện tại trước.
- **Iter lẻ**: Ratchet (trọng số SPR) → SPR thông thường.

Topology đạt `bestParsimony` ở iteration k bị ghi đè khi iteration k+1 bắt đầu (NNI perturbation
hoặc ratchet moves không lưu lại topology cũ). Sau nhiều iterations, cây cuối cùng (end-state)
có thể khác — và tệ hơn — topology đã đạt `bestParsimony`.

Đây là **quality loss thực sự**: GPU tìm được cây tốt nhưng không lưu lại, cuối cùng trả về
cây kém hơn cho CPU.

### Phân tích để tìm bug: dùng Agent debugger

Bug được phát hiện qua pipeline:
1. Chạy benchmark 115 dataset → so sánh `post_hc_best` vs `final_score` trong Python
2. Phát hiện 14/115 dataset có discrepancy (luôn theo chiều post-HC tốt hơn)
3. Spawn Agent debugger tìm code path: trace từ `bestParsimony` print → download → CPU re-score
4. Agent xác định 3 write site của `bestParsimony` và vị trí download topology

### Fix: Lưu `best_back_vf[]` mỗi khi `bestParsimony` cập nhật

**Nguyên tắc**: khi tìm được score tốt hơn, lưu snapshot của `back_vf[]` (topology tại thời điểm đó).
Khi download, dùng snapshot này thay vì end-state.

**4 thay đổi trong 4 file:**

**1. `pars_tree.cuh`** — thêm field vào `GpuTopology`:
```cpp
int back_vf[kMaxVFaces];       // topology hiện tại (end-state sau mỗi iteration)
int best_back_vf[kMaxVFaces];  // snapshot back_vf[] khi bestParsimony được cập nhật
```
Memory overhead: +12.8 KB/tree. Với K=199: +2.5 MB GPU memory.

**2. `pars_tree.cu`** — `cpuToGpuTopology`: khởi tạo `best_back_vf = back_vf`.

**3. `pars_build.cu`** — copy `back_vf → best_back_vf` tại 3 điểm update `bestParsimony`:
```cpp
// End of Phase 1 (build):
topo->bestParsimony = sh.bestParsimony;
for (int vf = 0; vf < topo->num_vfaces; vf++)
    topo->best_back_vf[vf] = topo->back_vf[vf];

// End of Phase 2 (initial SPR):
topo->bestParsimony = sh.randomMP;
for (int vf = 0; vf < topo->num_vfaces; vf++)
    topo->best_back_vf[vf] = topo->back_vf[vf];

// Phase 3 per-iteration:
if (lane == 0 && sh.randomMP < topo->bestParsimony) {
    topo->bestParsimony = sh.randomMP;
    for (int vf = 0; vf < topo->num_vfaces; vf++)
        topo->best_back_vf[vf] = topo->back_vf[vf];
}
```

**4. `gpu_init_trees.cu`** — bước [7] sau download, trước `gpuTopoToCpu`:
```cpp
// Restore best-seen topology (not end-state after last iteration)
for (int vf = 0; vf < h_topo.num_vfaces; vf++)
    h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
```
Không cần thay đổi `gpuTopoToCpu` — chỉ swap `back_vf` ← `best_back_vf` trên CPU trước khi convert.

### Kết quả

**Test nhanh 5 dataset có discrepancy lớn nhất:**

| Dataset | Post-HC best | Final (trước fix) | Final (sau fix) |
|---------|-------------|-----------------|----------------|
| dna_M8692_395_3583 | 2938 | 2941 (+3) | **2938** ✓ |
| dna_M11113_344_9778 | 113713 | 113718 (+5) | **113713** ✓ |
| dna_M3198_216_2578 | 35866 | 35870 (+4) | **35866** ✓ |
| dna_M10933_229_2696 | 21854 | 21857 (+3) | **21854** ✓ |
| dna_M7024_767_5814 | 95109 | 95155 (+46) | **95109** ✓ |

Post-HC best = Final score ở tất cả 5 dataset. Full benchmark 115 dataset đang chạy.

### Bài học / Ghi chú cho khóa luận

1. **Score ≠ Topology**: Lưu score tốt nhất là không đủ — phải lưu kèm topology tương ứng.
   Đây là lỗi kiến trúc: thiết kế ban đầu coi `bestParsimony` là điểm thống kê, không phải
   pointer tới trạng thái tốt nhất. Khi thêm Phase 3 với NNI perturbation (có thể làm cây tệ
   tạm thời), sự tách biệt này trở thành bug.

2. **Phát hiện qua benchmark cross-validation**: Bug không xuất hiện ở test nhỏ (1 dataset),
   chỉ lộ ra khi chạy 115 dataset và so sánh hai metric `post_hc_best` vs `final_score` bằng
   script Python. Bài học: tổng hợp kết quả nhiều dataset giúp phát hiện bugs systematic.

3. **Trade-off của Option A vs B vs C**: Fix đúng nhất (A) là lưu topology — tốn thêm bộ nhớ
   nhưng không tốn thêm thời gian đáng kể (copy `num_vfaces` ints ≈ 4.7 KB, rất nhanh).
   Option C (sửa log) chỉ ẩn bug không fix. Option B (thêm 1 pass HC cuối) tốn thêm 1 iteration
   thời gian (~10% overhead) và không đảm bảo recover đúng topology tốt nhất.

4. **Subtlety của copy loop**: Copy chỉ cần `num_vfaces` (= 4N-3) elements, không phải `kMaxVFaces`
   (= 4×800 = 3200). Với N=295: num_vfaces=1177 ints = 4.7 KB — rất nhỏ, overhead không đáng kể.

---

## Optimization #2 — Opt-M: Template specialization `newviewParsimony<STATES>` (2026-05-13)

**Ngày**: 2026-05-13  
**Task**: Giảm register pressure và enable loop unrolling cho Fitch parsimony  
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`

### Phân tích

`newviewParsimony` (tính Fitch parsimony) dùng mảng `t_A[kMaxStates=32]` và `o_A[kMaxStates=32]`
với runtime loop `for (int s = 0; s < states; s++)`. Với DNA (states=4): 28/32 slots lãng phí,
loop không được unroll → register pressure cao, không tận dụng ILP.

### Fix

Toàn bộ call chain được template hóa theo `STATES`:
- `newviewParsimony<SharedT, STATES>`: `t_A[STATES]`, `o_A[STATES]`, `#pragma unroll`
- `testInsert<STATES, SharedT>`, `doAddTraverse<STATES, SharedT>`, v.v.
- Kernels: `buildParsimonyTreesKernel<STATES, NTAXA>`, `buildPhase3Kernel<STATES, NTAXA>`
- Host dispatch: `states==4 → launch<4>`, `states==20 → launch<20>` (chỉ 2 STATES)

**Quyết định bỏ states=2 (binary) và states=32 (fallback)**: Không xuất hiện trong dataset thực tế (DNA=4, protein=20). Bỏ giúp giảm số kernels từ 8 → 4 → build time ~5 phút thay vì ~9 phút.

### Kết quả

**Benchmark** (50 datasets, numpars=200, gpu_stop=4, seed=1):
- Average speedup: **2.16×** vs baseline trước Opt-M
- Protein datasets benefit đặc biệt: prot_M4860 (62 taxa) 3.71×, prot_M9973 (60 taxa) 3.52×
- Speedup đến từ **ILP/loop unrolling**, KHÔNG phải occupancy (shared mem vẫn là bottleneck)

### Bài học / Ghi chú cho khóa luận

Template specialization là kỹ thuật quan trọng trong GPU kernel optimization: biến runtime constant
thành compile-time constant cho phép compiler unroll loop, allocate registers chính xác, và loại
bỏ dead code. Với Fitch parsimony có states=4 cố định (DNA), compiler tạo code tối ưu hoàn toàn.

---

## Optimization #3 — Opt-P: BuildSharedT shrink để tăng blocks/SM (2026-05-13)

**Ngày**: 2026-05-13  
**Task**: Tăng GPU occupancy bằng cách giảm kích thước shared memory  
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`, `gpu/CMakeLists.txt`

### Phân tích bottleneck

ptxas profiling xác nhận: `BuildShared = 28.4 KB` là binding constraint → **5 blocks/SM**
(164 KB/SM ÷ 28.4 KB), trong khi register limit cho phép 13 blocks/SM. Occupancy = 7.8%.

### Layer 1: Union perm/stackMint + giảm stackMaxt

**Quan sát phase usage**:
- `perm[]` (Phase 0-1 ONLY) và `stackMint[]` (Phase 2-3 ONLY) không bao giờ dùng đồng thời
- `stackMaxt[]` được dùng cho 2 mục đích: SPR stack (depth ≤ 12) và NNI bitset (≤ 51 words)
  → kMaxTaxa=800 entries hoàn toàn thừa, chỉ cần 64

```cpp
union {
    int perm[NTAXA + 2];   // Phase 0-1 only
    int stackMint[NTAXA];  // Phase 2-3 only
};
int stackMaxt[kMaxSprStack=64];  // was [kMaxTaxa=800]
```

**Kết quả**: 28.4 KB → **22.4 KB**, blocks/SM: 5 → 7 (+40% occupancy)  
**Speedup**: avg **+1.197×** vs Opt-M (50 datasets, 0 regression)

### Layer 2b: int16_t cho các mảng + reorder fields

**int16_t**: `ti[]`, `tiStack[]`, `stack[]`, `perm/stackMint[]` đều lưu giá trị ≤ 3197 (vface IDs)
hoặc ≤ 1599 (node numbers) → fit int16_t. `stackMaxt[]` giữ `int` vì NNI bitset cần 32-bit.

**Kết quả int16_t alone**: shared 22.4 → 11.4 KB, nhưng chỉ **−1.6% performance** (slower!).

**Lý do**: với numpars=200 (K=199 blocks), tất cả blocks đã fit trong GPU ở capacity 7 blocks/SM
(= 756 total). Tăng lên 14 blocks/SM (register limit) không giúp ích vì K < capacity. Overhead
của `ld.shared.s16` instruction bù trừ mất lợi ích occupancy.

**Reorder fields** (HOT int16_t arrays trước): +2.0% vs L1 — offset nhỏ trong `ld.shared` giúp
compiler generate code hiệu quả hơn, giảm I-cache pressure.

**Kết hợp int16_t + reorder**: **+2.0% vs L1** với shared mem giảm 50%.

### Layer 3: BuildSharedT<NTAXA> template

Template hóa BuildShared theo NTAXA bucket (128/256/512/800) để array sizes thu nhỏ theo N thực tế.

**Phân tích giới hạn**: lợi ích chỉ xuất hiện khi K > capacity. Với numpars=200 (K=199):
- Capacity ở NTAXA=800 = 7 × 108 SMs = 756 > 199 → 1-wave, không queue → NTAXA không giúp
- Lợi ích thực khi numpars > 757 (K > 756)

**CMake flag** `GPU_NTAXA_TEMPLATE=ON/OFF` để control build time:
- OFF (default): 4 kernels × 2 STATES = 8 kernels → ~5 phút
- ON: 4 NTAXA × 2 STATES × 2 kernels = 16 kernels → ~16 phút

### Bài học / Ghi chú cho khóa luận

1. **Occupancy không phải lúc nào cũng là bottleneck**: Khi K nhỏ (199 blocks, 108 SMs), GPU không
   thiếu chỗ để schedule blocks — tăng blocks/SM không cải thiện throughput.
2. **16-bit shared memory có overhead**: `ld.shared.s16` vs `ld.shared.s32` — dù tất cả accesses
   là lane-0 (không có bank conflict), instruction overhead vẫn tồn tại trên A100.
3. **Field ordering trong struct quan trọng**: Hot fields ở low offset → smaller immediates
   trong `ld.shared` → compiler generates denser code.
4. **GPU_NTAXA_TEMPLATE là framework cho tương lai**: Khi dataset lớn hơn hoặc numpars cao hơn,
   NTAXA template sẽ có lợi hơn.

---

## Experiment Log — Phase 3 Effectiveness Analysis (2026-05-13)

**Ngày**: 2026-05-13  
**Dataset**: 10 datasets N=100–767  
**Settings**: `-numpars 200 -sprdist 3 -gpu_stop 4 -seed 1`

### Kết quả

| Dataset | N | NNI+SPR (even) | Ratchet (odd) | Winner |
|---------|---|---------------|--------------|--------|
| prot N=100 | 100 | 24.7% | 13.6% | NNI+SPR |
| prot N=137 | 137 | 30.6% | 23.0% | NNI+SPR |
| prot N=169 | 169 | 0.0% | 0.0% | tie (converged) |
| dna N=295 | 295 | 20.5% | 35.8% | Ratchet |
| dna N=330 | 330 | 2.6% | 14.3% | Ratchet |
| dna N=350 | 350 | 39.1% | 32.6% | NNI+SPR |
| dna N=405 | 405 | 6.7% | 9.1% | Ratchet |
| dna N=544 | 544 | 4.2% | 9.9% | Ratchet |
| dna N=699 | 699 | 27.9% | 51.5% | Ratchet |
| dna N=767 | 767 | 35.5% | 55.6% | Ratchet |
| **Average** | | **19.2%** | **24.5%** | **Ratchet 6/10** |

### Quan sát

1. **Ratchet hiệu quả hơn tổng thể** (avg 24.5% vs 19.2%): Ratchet tạo "escape" khỏi local optima
   tốt hơn NNI perturbation vì thay đổi objective function (weighted parsimony), không chỉ swap edges.

2. **NNI+SPR hiệu quả hơn với protein nhỏ** (N≤137): Protein datasets có state space phức tạp hơn
   (states=20), NNI perturbation phù hợp hơn vì ít phá vỡ cấu trúc tốt.

3. **N=169: cả hai 0%** — cây hội tụ hoàn toàn sau Phase 2 (initial SPR); Phase 3 không tìm thêm
   cải thiện. Đây là trường hợp `gpu_stop` hoạt động hiệu quả — dừng sớm không lãng phí.

4. **Dataset lớn N≥699: Ratchet vượt trội rõ** (51–56% vs 28–36%): Landscape parsimony lớn có nhiều
   local optima, Ratchet tạo perturbation mạnh hơn (thay đổi trọng số sites) giúp thoát sâu hơn.

### Bài học / Ghi chú cho khóa luận

**Ratchet trong GPU context**: Ratchet gốc (Nixon, 1999) dùng cho maximum parsimony: tăng gấp đôi
trọng số một subset ngẫu nhiên các sites, chạy SPR, rồi trở về trọng số đều. GPU implementation
thực hiện cả 2 SPR calls (weighted + unweighted) trong 1 odd iteration. Kết quả: Ratchet tạo
perturbation 2-phase mạnh hơn NNI 1-phase, đặc biệt hiệu quả với dataset lớn.

**Implication cho thiết kế**: Nếu tài nguyên hạn chế, ưu tiên Ratchet (odd iters) hơn NNI+SPR
(even iters) — đặc biệt với dataset DNA lớn. Với protein nhỏ, NNI+SPR cần thiết hơn.

---

## Kết quả thực nghiệm — GPU vs CPU Full Benchmark (2026-05-13)

**Ngày**: 2026-05-13  
**Dataset**: 115 datasets, N=50–767 (50 protein, 65 DNA)  
**Config GPU**: numpars=400, sprdist=3, gpu_stop=4, seed=1  
**Config CPU**: numpars=100 (serial, gốc)  
**GPU**: A100-SXM4-80GB; **CPU**: Intel Xeon (single core, serial)

### Kết quả tổng hợp

| Metric | Value |
|--------|-------|
| Average speedup | **2.28×** |
| N ≤ 60 | 0.68–1.3× (overhead > compute) |
| N = 200–400 | 1.4–3.7× |
| N ≥ 400 | **3.5–6.7×** |
| GPU chất lượng tốt hơn | 33/115 (29%) |
| Chất lượng bằng nhau | 57/115 (50%) |
| CPU tốt hơn | 25/115 (22%) |
| Avg Δparsimony (GPU−CPU) | **−0.13** |

Full results: [benchmark/gpu_vs_cpu_np400_115datasets.md](../benchmark/gpu_vs_cpu_np400_115datasets.md)

### Phân tích

**Tại sao N nhỏ (≤60) GPU chậm hơn?**  
Overhead cố định (cudaMalloc, upload parsVect, kernel launch) chiếm tỉ lệ lớn khi kernel time ngắn.
Với N=55, kernel chạy chỉ 1–2 giây — overhead ~1-2 giây → tổng thời gian gần gấp đôi kernel.

**Tại sao GPU tốt hơn chất lượng (29% cases)?**  
GPU chạy 399 trees độc lập với seeds khác nhau → khám phá nhiều vùng của landscape parsimony.
CPU serial chỉ chạy 99 trees → ít diverse starting points hơn.

**Avg Δ = −0.13**: GPU nhỉnh hơn CPU về parsimony score trung bình, xác nhận GPU tìm được cây
tốt hơn hoặc tương đương trong 79% trường hợp.

### Bài học / Ghi chú cho khóa luận

1. **GPU phù hợp nhất với N≥200**: Breakeven point khoảng N=80–100 tùy dataset.
   Với N<80, CPU serial vẫn cạnh tranh hoặc nhanh hơn.

2. **Chất lượng GPU không kém hơn CPU**: Dù GPU dùng khác thuật toán (NNI+Ratchet thay vì pure SPR),
   kết quả parsimony tương đương hoặc tốt hơn trong đa số trường hợp.

3. **Speedup lớn nhất ở dataset "nặng" về width**: prot_M10273 (N=169, 11009 sites) đạt 6.69× —
   vì parsVect lớn làm CPU memory-bound hơn, trong khi GPU parallel Fitch computation mạnh hơn.

---

## Optimization #4 — Phase 3 Micro-optimizations (2026-05-14)

**Ngày**: 2026-05-14  
**Task**: Tối ưu Phase 3 hill-climbing — loại bỏ redundant work  
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`

### Opt-Q1: Xóa score_tree zero-init loop

**Vấn đề**: Trong `newviewParsimony` (pars_tree.cuh), có vòng lặp:
```cpp
if (lane == 0) {
    for (int i = 3; i < sh.tiSize; i += 3)
        score_tree[sh.ti[i]] = 0;
}
```
Loop này là **dead code**: main loop bên dưới ghi `score_tree[p_num] = score + score_tree[q_num] + score_tree[r_num]` — là assignment hoàn toàn, không dùng giá trị cũ của `score_tree[p_num]`.

**Xác nhận**: Loop phục vụ như implicit prefetch (đưa cache lines vào L1). Tuy nhiên qua benchmark công bằng (cùng env), xóa loop cho −3.4% avg ms/tree.

**Fix**: Xóa 8 dòng init loop.  
**Kết quả** (10 datasets, numpars=200, gpu_stop=4): avg **−3.4%** ms/tree, 0 quality regression.

### Opt-Q2: Lazy gpuNodeRectifierPars — skip first do-while iteration

**Vấn đề**: `gpuSPRHillClimb` gọi `gpuNodeRectifierPars` ở đầu **mỗi** do-while iteration.
Nhưng iteration đầu tiên luôn là redundant vì:
- Even phase: `runPhase3` gọi `gpuNodeRectifierPars` ngay TRƯỚC khi gọi `gpuSPRHillClimb`
- Odd phase (Ratchet SPR#1): topology chưa thay đổi từ lần SPR trước
- Odd phase (Ratchet SPR#2): SPR#1 kết thúc với do-while cuối không có move → nodep[] fresh

Với do-while iter 2+: topology đã thay đổi sau move → rectify IS needed.

**Fix**:
```cpp
bool first_iter = true;
do {
    startMP = sh.randomMP;
    if (lane == 0) sh.n_dowhile++;
    if (!first_iter) {
        if (lane == 0) gpuNodeRectifierPars(topo, sh, N);
        __syncwarp();
    }
    first_iter = false;
    // ...
} while (sh.randomMP < startMP);
```

**Tại sao không dùng "rectify after applyMove"**: Khi nhiều move được apply trong một do-while,
calling rectify after each move = M calls vs original 1 call per do-while. Với N=699 (nhiều moves),
cách này chậm hơn 11.9%. Skip-first approach: chỉ tiết kiệm 1 call/iteration của gpuSPRHillClimb
→ consistent speedup không có regression.

**Kết quả** (10 datasets, numpars=200, -gpu_top_pct -1, seed=1):

| N | Baseline | Lazy Rectify | Change |
|---|----------|-------------|--------|
| 100 | 21.82 | 17.83 | −18.3% |
| 137 | 28.87 | 23.81 | −17.5% |
| 169 | 118.51 | 113.78 | −4.0% |
| 295 | 104.44 | 83.76 | −19.8% |
| 330 | 71.74 | 68.30 | −4.8% |
| 350 | 208.05 | 168.29 | −19.1% |
| 405 | 157.65 | 111.48 | −29.3% |
| 544 | 244.29 | 204.74 | −16.2% |
| 699 | 628.61 | 628.54 | −0.0% |
| 767 | 650.14 | 642.70 | −1.1% |
| **Avg** | | | **−13.0%** |

**Quality**: 10/10 datasets không có regression. 3/10 tìm được cây tốt hơn.

### Bài học / Ghi chú cho khóa luận

1. **Dead code trong GPU kernel**: Zero-init loop tưởng như cần thiết nhưng thực ra bị overwrite hoàn toàn. Phân tích kỹ read/write pattern của từng dòng để phát hiện dead code.

2. **Redundant computation pattern**: `gpuNodeRectifierPars` là O(N) DFS — chi phí đáng kể. Khi một function được gọi với invariant "caller đã maintain property này", skip có thể tiết kiệm đáng kể.

3. **Trade-off giữa "rectify after each move" vs "skip first iteration"**: Cả hai đều đúng về logic, nhưng performance khác nhau. After-each-move tốt khi ít moves/iteration; skip-first tốt khi nhiều moves/iteration (amortizes better). Cần benchmark để chọn.

---

## Optimization #5 — Restore best_back_vf trước mỗi Phase 3 perturbation (2026-05-14)

**Ngày**: 2026-05-14  
**Task**: Cải tiến Phase 3 hill-climbing — loại bỏ "topology drift"  
**File liên quan**: `gpu/src/pars_build.cu` — `runPhase3`

### Phát hiện vấn đề

Phân tích `runPhase3` cho thấy mỗi iteration bắt đầu từ **end-state** của iteration trước, không phải từ `best_back_vf` (cây tốt nhất đã tìm được):

```
topo->back_vf[]      = working topology (bị modify mỗi iter)
topo->best_back_vf[] = snapshot khi bestParsimony được update
```

Flow cũ:
```
Iter 0 (even): NNI(T_init) → SPR → T0;  if T0<best: best=T0
Iter 1 (odd):  Ratchet(T0)  → SPR → T1;  if T1<best: best=T1
Iter 2 (even): NNI(T1)      → SPR → T2   ← T1 có thể tệ hơn T0!
```

Hậu quả: GPU "drift" xa vùng parsimony tốt sau nhiều iterations, làm SPR phải tốn thêm do-while để leo đồi trở lại từ điểm tệ.

### Fix: Restore warp-parallel trước mỗi iteration

Thêm warp-parallel copy `best_back_vf → back_vf` đầu mỗi even (NNI) VÀ odd (Ratchet) iteration:

```cpp
// Warp-parallel restore (32 lanes, tất cả tham gia):
for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
    topo->back_vf[vf] = topo->best_back_vf[vf];
__syncwarp();
// Sau đó: NNI / Ratchet weights
// createTiAndEvaluateParsimony(full=true) đã có sẵn → sync lại parsVect tự động
```

**Tại sao parsVect không cần reset thêm**: `createTiAndEvaluateParsimony(full=true)` được gọi ngay sau mỗi perturbation (đã có sẵn trong code), `full=true` bỏ qua xpars flags và recompute toàn bộ cây.

### Thực nghiệm 3 options

| Option | Restore khi nào | Avg Δms/tree (10 datasets) |
|--------|----------------|--------------------------|
| Baseline | Không restore | 0% |
| A | Chỉ trước even (NNI) | **−24.5%** |
| B+C | Trước cả even + odd | **−38.0%** |

Option C (restore cả hai) thắng rõ ràng — improvement rate Ratchet tăng từ 35% → 52% (N=295) và 52% → 70% (N=699).

### Giải thích speedup

1. **SPR do-while ít iterations hơn**: Restore về best → SPR bắt đầu từ điểm gần-optimal → hội tụ nhanh hơn.
2. **Ratchet improvement rate cao hơn**: Ratchet từ best tree tìm được cải thiện thực sự thay vì từ end-state tệ của NNI.

### Kết quả benchmark đầy đủ (115 datasets, numpars=400, sprdist=3, gpu_stop=4)

So sánh gpu_final_800 (Option C) vs gpu_np400 (baseline):

| Metric | gpu_np400 | gpu_final_800 | Cải thiện |
|--------|-----------|---------------|-----------|
| Avg speedup vs CPU | 2.28× | **3.66×** | +60% |
| Max speedup | 6.69× | **12.88×** | |
| GPU better quality | 33/115 (29%) | **40/115 (35%)** | |
| GPU worse quality | 25/115 (22%) | **20/115 (17%)** | |
| Avg Δparsimony (GPU−CPU) | −0.13 | **−0.22** | |

Full results: `output/gpu_final_800/`

### Bài học / Ghi chú cho khóa luận

1. **State machine analysis quan trọng**: Phân tích luồng state `back_vf` (working) vs `best_back_vf` (saved) cho thấy "drift" problem — không phải bug về correctness mà là sub-optimal exploration strategy.

2. **Restore từ best = independent exploration**: Mỗi iteration trở thành một independent perturbation + hill-climb từ best known, thay vì sequential chain. Tương tự concept "iterated local search" trong combinatorial optimization.

3. **Tại sao nhanh hơn CHỨ KHÔNG phải chậm hơn**: Restore thêm O(N) work mỗi iteration, nhưng tiết kiệm O(N×k) SPR do-while work (k = số do-while extra cần thiết để "leo đồi" từ tệ về tốt). Với k>>1, net speedup lớn.

4. **Quality cải thiện cùng với speed**: Đây là dấu hiệu của exploitation-exploration balance tốt hơn. Khi mỗi perturbation bắt đầu từ best, exploration có ý nghĩa hơn → tìm được cây tốt hơn trong ít thời gian hơn.

---

## Bug #10 — ODR violation: `sizeof(SearchInfo)` khác nhau giữa CXX TU và CUDA TU

**Ngày**: 2026-05-16
**Task**: GPU mode crash ngay sau `mpbootGpu()` — `candidateTrees.aln` đọc từ địa chỉ sai
**File liên quan**: `mpboot/tools.h`, `mpboot/nnisearch.h`, `mpboot/gpu/src/gpu_init_trees.cu`

### Triệu chứng

GPU mode (`-use_gpu`) crash ngay khi bắt đầu `gpuInitCandidateTrees()`. Debug prints in ra
layout `IQTree` với `candidateTrees.aln` lệch nhiều byte so với kỳ vọng. Crash xảy ra khi
`gpu_init_trees.cu` đọc `iqtree.candidateTrees.aln` — giá trị là garbage pointer.

### Root cause — ODR (One Definition Rule) violation

`SearchInfo` struct (`nnisearch.h`) chứa field:
```cpp
unordered_set<string> aBranches;
```

`tools.h` chọn implementation của `unordered_set` theo `GCC_VERSION` macro:
- **Nhánh `< 40300`**: `__gnu_cxx::hash_set<string>` → sizeof = **40 bytes**
- **Nhánh `>= 40300`**: `std::tr1::unordered_set<string>` → sizeof = **48 bytes**

Hai translation units dùng hai compiler khác nhau:
- **CXX TU** (`phyloanalysis.cpp`, compiled by `clang++`): `__GNUC__=4.2.1` → `GCC_VERSION=40201 < 40300` → `hash_set` (40 bytes) → `sizeof(SearchInfo)=160`
- **CUDA TU** (`gpu_init_trees.cu`, compiled by nvcc với host `gcc 9.4`): `__GNUC__=9.4.0` → `GCC_VERSION=90400 >= 40300` → `tr1::unordered_set` (48 bytes) → `sizeof(SearchInfo)=168`

Hệ quả: mọi field của `IQTree` sau `searchinfo` (field đầu tiên dùng `SearchInfo`) đều có offset sai trong CUDA TU, bao gồm `candidateTrees` và `candidateTrees.aln` — đọc từ địa chỉ lệch 8 bytes → garbage pointer → crash.

**Root cause thứ hai**: `CMakeLists.txt` có `set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})` tại line 43 nhưng AFTER `enable_language(CUDA)` tại line 42 → vô hiệu. CUDA host compiler thực sự là system gcc 9.4.0, không phải `clang++`.

### Fix: `tools.h` — thêm nhánh `__cplusplus >= 201103L`

Thêm nhánh C++11 trước các nhánh GCC_VERSION:
```cpp
#if defined(USE_HASH_MAP) && !defined(_MSC_VER)
    #if __cplusplus >= 201103L
        // C++11+: dùng std::unordered_map/set — tránh ODR violation
        #include <unordered_map>
        #include <unordered_set>
    #elif !defined(__GNUC__)
        // ...
    #elif GCC_VERSION < 40300
        // ...
    #else
        // ...
    #endif
#endif
```

Cả hai TU đều build với `-std=c++14` → cả hai đều thấy `__cplusplus >= 201103L` → cùng dùng `std::unordered_set` (56 bytes) → `sizeof(SearchInfo)=176` nhất quán.

Cũng guard `__gnu_cxx::hash<string>` specialization:
```cpp
#if defined(USE_HASH_MAP) && GCC_VERSION < 40300 && !defined(_MSC_VER) && __cplusplus < 201103L
```

### Xác nhận fix

Sau fix: `offsetof(IQTree, candidateTrees.aln)` giống nhau trong cả hai TU. GPU mode chạy được qua `gpuInitCandidateTrees()` không crash.

### Bài học / Ghi chú cho khóa luận

1. **ODR violation không compile-error**: Hai TU định nghĩa `SearchInfo` khác nhau mà không có cảnh báo. Linker kết hợp hai object files mà không biết conflict. Chỉ lộ ra khi runtime dùng wrong offsets.

2. **nvcc host compiler ≠ cmake CXX compiler**: Dù CMakeLists.txt ghi `CMAKE_CUDA_HOST_COMPILER = clang++`, việc set sau `enable_language(CUDA)` không có effect. Dùng `cmake --trace` để xác nhận compiler thực sự.

3. **`tools.h` type selection fragile**: Dùng `GCC_VERSION` để chọn container type là anti-pattern khi codebase dùng nhiều compilers. Fix đúng: dùng `__cplusplus` hoặc `__has_include` — consistent trên mọi compiler.

4. **Chẩn đoán bằng `offsetof`**: Khi crash xảy ra tại boundary TU (data từ TU A, đọc từ TU B), print `offsetof(Struct, field)` trong cả hai TU để xác nhận ODR violation.

---

## Bug #11 — `hybrid_cb2`: ba crash độc lập khi GPU hill-climbing chạy song song

**Ngày**: 2026-05-16
**Task**: `AfterK2Callback` (`hybrid_cb2`) trong `gpu_init_trees.cu` — CPU perturbation trong khi GPU hillClimbingKernel chạy
**File liên quan**: `gpu/src/gpu_init_trees.cu`

### Bối cảnh

`hybrid_cb2` là callback chạy trên CPU trong khi GPU `hillClimbingKernel` chạy async:
- Even iterations: NNI perturbation với `iqtree.aln`, rồi score
- Odd iterations: Ratchet perturbation (weighted alignment), rồi score
- Sau mỗi iter: lấy tree string hiện tại, update `candidateTrees`

### Crash #1 — SIGFPE: `topologies.count()` sau `pllOptimizeSprParsimonyTree`

**Root cause**: Code gọi `topologies.count(key)` trên một `unordered_map` sau khi `pllOptimizeSprParsimonyTree` đã corrupt `bucket_count` của hash map về 0. Modulo 0 → SIGFPE.

**Fix**: Bỏ toàn bộ `topologies` access trong `hybrid_cb2` — không cần kiểm tra duplicate trong callback này vì `candidateTrees.treeExist()` đã handle.

### Crash #2 — SIGSEGV: `parsVect=NULL` trong PLL parsimony functions

**Root cause**: `ratchet_iter=1` (default) → `_pllFreeParsimonyDataStructures` được gọi sau mỗi `doNNISearch` → `parsVect=NULL`. Lần gọi tiếp theo với `first_call=false` bỏ qua alloc → NULL parsVect → crash khi truy cập.

Quan sát thêm: **tất cả** PLL parsimony calls (`_pllSprOnCurrentTree`, `_pllComputeRandomizedStepwiseAdditionParsimonyTree`) corrupt heap gần `aln->seq_names` ngay cả trên clones. Phase 2 dùng PLL parsimony gây crash sau ~300 iterations.

**Fix**: Bỏ hoàn toàn mọi PLL parsimony function call trong `hybrid_cb2`. Dùng `PhyloTree::computeParsimony()` thay thế — safe, không PLL heap ops.

### Crash #3 — SIGSEGV/ABORT: `candidateTrees.aln = iqtree.aln` sau post-processing

**Root cause**: `iqtree.aln` là sorted alignment (sau `sort_taxa()`). `candidateTrees.aln` từ `setParams` là pre-sort alignment — object khác, được populate trước post-processing. PLL Newick dùng `nameList` tied với original aln object — set `candidateTrees.aln = iqtree.aln` (sorted) → `getTopology()` lookup taxa không match → SIGSEGV/ABORT.

**Fix**: Không bao giờ gán `candidateTrees.aln = iqtree.aln`. `candidateTrees.aln` được setup đúng bởi `setParams` trước khi vào `mpbootGpu`.

### Crash #4 — `doNNISearch()` reinitializes PLL cho alignment hiện tại

**Root cause**: `doNNISearch()` trong parsimony mode reinitializes PLL cho current alignment. Khi gọi với perturbed alignment set, PLL state mismatch → crash. Ngoài ra, `pllTreeInitTopologyNewick` trên clone trả về ORIGINAL `tr` pointers → `child->back=parent` set original tip back-ptrs sang freed clone memory → next `pllInstanceClone(tr)` tạo broken clones.

**Fix**: Xóa `doNNISearch()` call. Thay bằng `iqtree.getTreeString()` — chỉ lấy current tree topology string, không search.

### Crash #5 — `optimizeAllBranches()` trong parsimony mode

**Root cause**: `optimizeAllBranches()` là likelihood function — cần model parameters không tồn tại trong parsimony mode. Crash ngay khi gọi.

**Fix**: Thay bằng:
```cpp
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -(double)iqtree.computeParsimony();
```

### Final `hybrid_cb2` structure sau fix

```cpp
// Even: NNI perturbation + parsimony score
iqtree.doNNI(numNNI);  // safe — in-place NNI, không đụng PLL
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -(double)iqtree.computeParsimony();

// Odd: Ratchet perturbation + parsimony score
Alignment* perturb_aln = new Alignment;
perturb_aln->createPerturbAlignment(iqtree.aln, ...);
iqtree.setAlignment(perturb_aln);
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -(double)iqtree.computeParsimony();

// Sau odd iter: restore original alignment
delete iqtree.aln;  // frees perturb_aln
iqtree.setAlignment(saved_aln);
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -(double)iqtree.computeParsimony();

// Get tree string — không search
std::string imd_tree = iqtree.getTreeString();
candidateTrees.update(imd_tree, iqtree.curScore);
```

### Kết quả sau fix

GPU mode chạy hoàn chỉnh:
```
[GPU]   [hybrid2] Phase2: 569 iters (285 NNI, 284 ratchet)
[GPU]   [5+6+7]  Kernel (build+SPR+search)   :    7.723 s  (200 trees, 38.61 ms/tree)
[GPU]   built = 200 / 200 trees
(0 duplicated parsimony trees)
```
Exit 0, parsimony score 6727.

### Bài học / Ghi chú cho khóa luận

1. **PLL là thư viện riêng với internal state**: Gọi PLL function khi state không khớp (alignment khác, parsVect freed) → crash không dự đoán được. Quy tắc: chỉ dùng PLL functions khi `pllInst` đang giữ đúng alignment + parsVect đã allocated.

2. **`PhyloTree::computeParsimony()` là "safe zone"**: Không dùng PLL internal, chỉ dùng PhyloTree's own parsVect. Trong parsimony mode, đây là cách score cây an toàn nhất.

3. **Multiple independent crashes cùng code path**: Bug không phải một bug duy nhất mà là chuỗi 5 crash độc lập cùng xuất hiện khi code path được kích hoạt. Debug từng crash tuần tự bằng cách fix và rerun.

4. **`candidateTrees.aln` vs `iqtree.aln`**: Hai object khác nhau — `iqtree.aln` là sorted; `candidateTrees.aln` là pre-sort. Không được gán cross (dù trỏ về cùng dataset). Đây là invariant ngầm trong MPBoot không được document.

---

## Bug #12 — `hybrid_cb2`: `searchinfo.curPerStrength` chưa init → `doRandomNNIs(677M)` → 570s overhead

**Ngày**: 2026-05-16  
**Task**: GPU early-stop — dừng CPU hill-climbing khi GPU K2 xong  
**File liên quan**: `gpu/src/gpu_init_trees.cu`

### Bối cảnh

Mục tiêu: khi `hillClimbingKernel` (K2) xong (~5–9s), CPU SPR loop trong `doNNISearch`
phải thoát sớm thay vì chạy đến hội tụ (~191–627s trên 3 outlier datasets N=219–242).

Cơ chế đã implement (session trước):
1. `volatile int stop_search` trong `pllInstance` struct (`pll.h`)
2. `if(tr->stop_search) break` tại đầu for-loop + `&& !tr->stop_search` trong do-while condition
   trong cả `pllOptimizeSprParsimony` và `_pllSprOnCurrentTree` (`sprparsimony.cpp`)
3. Monitor thread trong `hybrid_cb2`: `cudaStreamSynchronize(cb_stream)` → set flag
4. `index += 4` → `index += 2` fix trong vòng replace `:nan` của `doNNISearch` (`iqtree.cpp`)

Sau khi implement, test vẫn cho kết quả **572s** — không cải thiện.

### Triệu chứng

Dataset `dna_M14678_225_2673` (N=225, sites=2673):
```
[GPU]   hillClimbingKernel: time: 2.834 s
[CPU]   CPU Hill-Climbing: 1 iters (1 NNI, 0 ratchet), bestScore = 9021
[GPU]   [5]  Kernels  :  570.794 s  (200 trees, 2853.97 ms/tree)
```

K2 chỉ mất 2.834s nhưng tổng Kernels là 570.794s. `hybrid_cb2` chạy 1 outer iteration
và không thoát được. Debug prints bằng stderr cho thấy code stuck tại `doRandomNNIs`.

### Root cause

Trong `hybrid_cb2`, NNI perturbation block:
```cpp
// TRƯỚC (sai):
int numNNI = floor(iqtree.searchinfo.curPerStrength * (iqtree.aln->getNSeq() - 3));
iqtree.doRandomNNIs(numNNI);
```

`iqtree.searchinfo.curPerStrength` **không bao giờ được initialize** trong context của
`hybrid_cb2`. Giá trị là garbage float (~3,051,130). Kết quả:
```
numNNI = floor(3051130.0 * (225 - 3)) = 677,352,759
```

`doRandomNNIs(677352759)` cố gắng thực hiện **677 triệu NNI moves** trên một cây N=225.
Mỗi NNI là một tree operation, nên đây thực chất là vòng lặp vô tận ~570s.

Cơ chế early-stop (monitor thread + `stop_search` flag) đã hoạt động **đúng** từ đầu —
nhưng flag chỉ được check bên trong `pllOptimizeSprParsimony`. Code không bao giờ reach được
đến `pllOptimizeSprParsimony` vì bị stuck trước đó tại `doRandomNNIs`.

### Debug methodology

Thêm `fprintf(stderr, ...)` tại các checkpoint:
1. `[CPU-DBG] entering while loop check` — while loop được entered ✓
2. `[CPU-DBG] readTreeString done` — readTreeString nhanh ✓
3. `[CPU-DBG] doRandomNNIs(677352759)` — **stuck here** ✗

Thiếu `fprintf` ngay sau `doRandomNNIs` confirm đây là bottleneck.

### Fix

```cpp
// SAU (đúng):
int numNNI = std::max(1, (int)floor(params.gpu_nni_strength * (iqtree.aln->getNSeq() - 3)));
iqtree.doRandomNNIs(numNNI);
```

Dùng `params.gpu_nni_strength` (CLI flag `-gpu_nni_strength`, default=0.1) — cùng formula
với GPU K2 kernel. Cho N=225: `numNNI = max(1, floor(0.1 * 222)) = 22`.

**File**: `gpu/src/gpu_init_trees.cu`, NNI perturb block bên trong while loop của `hybrid_cb2`.

### Kết quả sau fix

| Dataset | Trước | Sau | Speedup |
|---------|-------|-----|---------|
| dna_M14678_225_2673 (N=225, sites=2673) | 9:32 (572s) | **0:05 (5s)** | **115×** |
| dna_M5731_242_9626 (N=242, sites=9626) | ~300–600s (est.) | **0:08 (8s)** | **~50×** |
| dna_M6134_219_5158 (N=219, sites=5158) | ~200–400s (est.) | **0:05 (5s)** | **~50×** |

Dataset bình thường (N=295): 8.83s, không đổi so với trước.  
CPU Hill-Climbing: 218–248 iters/run, bestScore tốt hơn hoặc bằng trước.

### Các thay đổi trong session này (tổng hợp)

1. **`pll.h`**: thêm `volatile int stop_search` vào struct `pllInstance` (auto-zero bởi `rax_calloc`)
2. **`sprparsimony.cpp`**: thêm `if(tr->stop_search) break` đầu for-loop + `&& !tr->stop_search`
   vào do-while condition trong `pllOptimizeSprParsimony` và `_pllSprOnCurrentTree`
3. **`gpu_init_trees.cu`**: thêm `#include <thread>`; monitor thread wrap `doNNISearch` calls;
   **fix `curPerStrength` → `gpu_nni_strength`**
4. **`iqtree.cpp`**: fix `index += 4` → `index += 2` trong `:nan` replace loop của `doNNISearch`
   (bug cũ: `replace(pos, 4, ":0")` xong advance 4 thay vì 2, bỏ sót adjacent `:nan`)

### Bài học / Ghi chú cho khóa luận

1. **Uninitialized float → huge integer**: Garbage float × integer = số nguyên khổng lồ. Không
   có warning từ compiler vì cast từ float sang int là valid. Cần thêm assert hoặc clamp:
   `assert(numNNI >= 1 && numNNI <= getNSeq()); // hoặc max(1, min(numNNI, N))`

2. **Early-stop mechanism ẩn sau bug khác**: Cơ chế `stop_search` flag hoạt động đúng ngay
   từ đầu — 572s không phải do flag không work mà do code không reach được đến `pllOptimizeSprParsimony`.
   Root cause ẩn sâu dưới symptom (572s overhead tương tự trước khi implement early-stop).

3. **Debug by elimination với stderr**: stdout bị buffered/interleaved với GPU output → dùng
   `fprintf(stderr, ...)` để trace execution path. Thấy code pass qua `readTreeString` nhưng
   không qua `doRandomNNIs` → stuck tại đó.

4. **Tham số nào để dùng**: `searchinfo.curPerStrength` là state của IQ-TREE search heuristic —
   chỉ valid trong `runTreeSearch()` flow, không phải trong GPU callback context. `params.gpu_nni_strength`
   là CLI parameter luôn có giá trị hợp lệ — đây là lựa chọn đúng cho hybrid_cb2.

---

## Thay đổi thiết kế #1 — [7] loop: bỏ pllClone, chuyển candidateTrees.update vào mpbootGpu

**Ngày**: 2026-05-16  
**Task**: Refactor bước [7] trong `mpbootGpu()` (`gpu_init_trees.cu`)  
**Files**: `gpu/src/gpu_init_trees.cu`, `mpboot/phyloanalysis.cpp`, `gpu/include/gpu_init_trees.cuh`

### Trước (hybrid3 design)

Vòng [7] làm:
1. `pllInstanceClone(tr)` + `pllPartitionsClone(pr)` — tạo bản sao PLL instance cho mỗi cây
2. `gpuTopoToCpu(&h_topo, tr_clone)` — convert topology vào bản sao
3. `pllTreeToNewick(tr_clone, ...)` — emit Newick string
4. Lưu Newick vào `candidateTrees[i]` (vector<string>)

Sau khi `mpbootGpu()` trả về, caller trong `phyloanalysis.cpp` (lines 1429–1450) loop lại:
- `readTreeString()` + `computeParsimony()` + `candidateTrees.update()` cho mỗi cây

**Vấn đề**:
- `pllInstanceClone` + `pllPartitionsClone` = O(K) clone overhead (mỗi cây ~50–200 ms cho dataset lớn)
- Caller phải re-parse và re-score K cây → O(K) redundant work sau khi mpbootGpu đã xong

### Sau (hybrid4 design)

Vòng [7] dùng `tr`/`pr` trực tiếp (loop tuần tự → safe, không cần clone):
```cpp
gpuTopoToCpu(&h_topo, tr);           // reuse tr, không clone
pllTreeToNewick(tr, pr, ...);        // emit Newick
iqtree.readTreeString(tree_str);
iqtree.initializeAllPartialPars();
iqtree.clearAllPartialLH();
iqtree.curScore = -(double)iqtree.computeParsimony();
bool isNew = iqtree.candidateTrees.update(tree_str, iqtree.curScore);
if (isNew && iqtree.curScore > iqtree.bestScore)
    iqtree.setBestTree(tree_str, iqtree.curScore);
```

Caller post-loop trong `phyloanalysis.cpp` bị xóa hoàn toàn.

**Print format đổi**:
- Cũ: `"built = %d / %d trees"`
- Mới: `"best CPU tree: %-7u  best GPU tree: %u"` — in best parsimony score của CPU hybrid vs GPU post-HC

**Alignment fix**: `%d` → `%3d` cho iteration number trong `hybrid_cb2` print.

### Kết quả (hybrid4 benchmark — 20 datasets, 5 configs)

- **Correctness**: Quality y hệt hybrid3 (≤±2 điểm stochastic) — refactor không tạo regression ✅
- **Timing overhead nhỏ**: [7] mới thêm K lần `computeParsimony` — overhead ~20% cho N≤100, không đáng kể cho N≥200
- **GPU vs CPU quality**: 12/20 datasets GPU tốt hơn CPU, 7/20 bằng, 1/20 thua (N=640)

### Bài học / Ghi chú cho khóa luận

1. **Sequential loop = no clone needed**: Loop [7] chạy tuần tự trên CPU — `tr` bị overwrite mỗi iteration nhưng không ai đọc topology của `tr` sau khi loop kết thúc. Không cần clone instance chỉ để "an toàn".

2. **`CandidateSet::update()` return value**: Trả về `bool` (true = cây mới, false = duplicate/rejected). Dùng làm guard cho `setBestTree` — không cần `treeExist()` check riêng.

3. **`best_cpu_pars` capture trước [7] loop**: `(unsigned int)(-iqtree.bestScore)` phải được capture TRƯỚC khi GPU trees được thêm vào candidateTrees — vì `setBestTree` trong [7] có thể cập nhật `bestScore`. Print so sánh CPU-only vs GPU-only scores.

4. **initializeAllPartialPars + clearAllPartialLH**: Cần gọi trước `computeParsimony()` sau `readTreeString()` để reset parsVect state. Đây là overhead chính của [7] mới — tương đương với `pllInstanceClone` trước đó.

---

## Experiment — Reseed bad GPU slots from threshold pool (2026-05-17)

**Ngày**: 2026-05-17  
**Task**: Cải thiện chất lượng K2 bằng cách tái sử dụng topology tốt cho các bad GPU slots  
**File**: `gpu/src/gpu_init_trees.cu` — `hybrid_cb` lambda (Step 3.5)

### Vấn đề

Trong `hybrid_cb` (sau K1), Step 3 upload CPU threshold trees vào các bad GPU slots (score > threshold). Thường số CPU trees < số bad slots → nhiều bad slots không được điền, tiếp tục vào K2 với topology kém từ K1 SPR — lãng phí compute budget.

### Giải pháp — Step 3.5

Sau Step 3, với các bad slots còn lại chưa được điền:
1. **Thu thập good GPU topos**: Download các slots có `gpu_scores[k] <= threshold`, restore `best_back_vf → back_vf` (topology tốt nhất của K1 Phase 3).
2. **Build pool**: Kết hợp good GPU topos + good CPU trees (score ≤ threshold) vào một pool.
3. **Reseed**: Mỗi remaining bad slot nhận một topology ngẫu nhiên từ pool, với `savedSeed` khác nhau (`params.ran_seed + slot * 99991L`) để diversity, và `needs_recompute=1` để K2 recompute parsVect từ actual topology.

```cpp
GpuTopology fill_topo = *pool[random_int((int)pool.size())];
fill_topo.savedSeed = params.ran_seed + (long)slot * 99991L;
fill_topo.needs_recompute = 1;
uploadTopology(cb_mem, slot, &fill_topo, cb_stream);
```

### Cơ chế K2 compatibility

- `buildPhase3Kernel` filter: `if (topo->postSprParsimony > phase3Threshold) return;`
  → Donor's `postSprParsimony ≤ threshold` → reseeded slots được vào K2.
- `needs_recompute=1` trigger (lines 1107–1133): Recompute parsVect từ actual topology (`back_vf`), override `sh.randomMP`, `topo->bestParsimony`, `topo->postSprParsimony` → Phase 3 bắt đầu từ trạng thái đúng với seed mới.

### Kết quả benchmark

**N≈200 (37 datasets, numpars=200, sprdist=4, gpu_stop=6):**

| Metric | Value |
|--------|-------|
| Better (↑) | 14/37 |
| Same (=) | 23/37 |
| Worse (↓) | **0/37** |
| Avg Δscore | −1.2 (reseed tốt hơn trung bình) |
| Avg time | 8.2s → 9.4s (+14.5%) |

**N=767 (dna_M7024, numpars=400):**
- Score cải thiện: 95079 vs baseline 95088 (−9)
- K2 time tăng ~2× (143s vs 66.5s) — tất cả 400 slots start từ good topology → chạy nhiều Phase 3 iterations hơn

### Trade-offs

- ✅ Quality không bao giờ tệ hơn (0 regressions trên 37 datasets N≈200)
- ✅ 38% datasets cải thiện score
- ⚠️ Time overhead: +14.5% (N≈200), +100% (N=767) — do K2 có nhiều improving iterations hơn
- ⚠️ Overhead tăng theo N: slots lớn hơn → K2 iterations dài hơn khi start từ good topology

### Bài học / Ghi chú cho khóa luận

1. **Slot diversity vs quality**: Mỗi slot nhận cùng donor topology nhưng seed khác nhau → trajectories diverge trong K2, tạo diversity mà không cần extra memory.
2. **`needs_recompute` flag**: Cho phép upload topology shell (chỉ `back_vf`) mà không cần upload toàn bộ parsVect (2.44 MB/slot). GPU recompute trong kernel → tiết kiệm bandwidth đáng kể.
3. **Overhead asymmetry**: Reseed có giá trị nhất khi K2 budget lớn tương đối so với K1. Với N nhỏ (≈200), +14.5% là chấp nhận được. Với N lớn (≥700), cần cân nhắc giới hạn `gpu_stop` để kiểm soát overhead.

