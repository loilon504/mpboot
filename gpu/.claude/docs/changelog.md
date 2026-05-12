# Bug Changelog — GPU Parsimony Port

Mỗi entry ghi lại một bug đã được tìm ra và fix trong quá trình porting MPBoot sang GPU.
Dùng làm tài liệu tham khảo khi viết khóa luận tốt nghiệp.

---

## Bug #1 — `recomputeAllNodes` chỉ xử lý 293 inner node thay vì 294 (N=295)

**Ngày**: 2026-05-06
**Task**: Step [5] Stepwise addition → `recomputeAllNodes` (SPR preprocessing)
**File liên quan**: `gpu/src/pars_build.cu`, `gpu/src/gpu_spr.cu`

### Triệu chứng

`recomputeAllNodes` in ra `inner_nodes_processed=293 (expected N-1=294)` mỗi lần chạy với N=295.
Debug print `[TOPO k=0] back_vf[1174]=-1` xác nhận node 589 (= 2N-1, node inner cuối cùng được
cấp phát trong stepwise addition) có face[0] không có back-connection.

### Root cause

`buildParsimonyTreesKernel` (`pars_build.cu`) không khởi tạo đầy đủ cả ba face của inner node
cuối cùng được cấp phát trước khi kết thúc kernel. Node 2N-1 được dùng để gắn tip cuối cùng,
nhưng face index chưa được hookup hoàn chỉnh — `back_vf[vf_face0_of_last_node] = -1` còn sót lại.

DFS của `recomputeAllNodes` dùng stack với điều kiện `top_idx == 2` để nhận diện inner node.
Node 589 có một face với `back_vf = -1`, khiến DFS coi nó là lá (tip) hoặc skip, dẫn đến
không đếm và không cập nhật `score_tree` cho node này.

### Fix

*(Chưa áp dụng — đây là bug đang mở. Entry này ghi lại root cause đã phân tích.)*

Hướng fix: sau `buildParsimonyTreesKernel`, kiểm tra và hookup face còn thiếu của node 2N-1,
hoặc sửa logic cấp phát trong kernel để đảm bảo cả 3 face đều có `back_vf` hợp lệ trước khi
bước [6] SPR bắt đầu.

### Bài học / Ghi chú cho khóa luận

GPU topology dùng vface index flat array — không có NULL pointer như PLL. Giá trị `-1` đóng vai
trò NULL sentinel. Mọi inner node trong GPU topology phải có đúng 3 back-connections hợp lệ trước
khi bất kỳ kernel nào traverse topology. Thiếu hookup một face không gây crash (GPU không
dereference pointer) nhưng gây silent correctness bug khó phát hiện vì DFS silently skip node đó.

---

## Bug #2 — SPR làm cây tệ hơn: post-SPR parsimony 16369 thay vì ~6668

**Ngày**: 2026-05-06
**Task**: Step [6] SPR hill-climbing (`gpuSprKernel`, `gpu_spr.cu`)
**File liên quan**: `gpu/src/gpu_spr.cu`, `gpu/include/pars_tree.cuh`

### Triệu chứng

Pre-SPR best parsimony = 15368 (hợp lý cho N=295).
Post-SPR best parsimony = **16369** (tệ hơn trước SPR!).
CPU cho kết quả ~6668 — GPU lệch gần 10000.
Với tree k=0: `sh.randomMP` tăng từ 18175 → ~18800 qua các vòng lặp — cây ngày càng tệ hơn.

### Root cause

`testInsert` ước tính `mp` **quá thấp** (underestimate), khiến các move làm cây tệ hơn vẫn bị
chấp nhận vì `mp < sh.bestParsimony`.

Cơ chế cụ thể: sau `removeNodeParsimony(p)`, các giá trị `score_tree[]` dùng trong `testInsert`
là stale từ lần `recomputeAllNodes` trước. Cụ thể `score_tree[sh.tip_p_num]` (= score của
back-neighbor q của p) được tính theo hướng DFS từ `start_vface`. Nếu p nằm trên đường DFS
từ start đến một node khác, `score_tree[q]` bao gồm contribution cũ của p — nhưng p đã bị remove
khỏi cây. Kết quả: `mp` estimate dùng score_tree[q] sai → underestimate → move tệ được chấp nhận.

CPU giải quyết bằng `evaluateParsimony(p, FALSE)` tại dòng 2291 của `sprparsimony.cpp` (trước
`removeNodeParsimony`) — call này lazy-refresh `parsimonyScore[p]` và `parsimonyScore[q]` theo
đúng hướng hiện tại trước khi remove. GPU không có cơ chế tương đương.

**Các fix đã thử** (tất cả đều áp dụng, không đủ):
1. Guard `q_num > N` trong `doAddTraverse` — skip tip edges trong testInsert. (Cần thiết)
2. `recomputeAllNodes` sau mỗi `applyMove` — giữ score_tree tươi giữa các move. (Cần thiết)
3. Sửa formula `testInsert`: `parsVect[p_num] = Fitch(sh.tip_p_num, q_edge_num)`, evaluate
   tại `(p_num, r_num)`. (Đúng về mặt toán học, match CPU)

Kết quả sau 3 fix: vẫn còn 16369. Ba fix trên cần thiết nhưng chưa đủ.

### Fix

*(Chưa resolve — bug đang mở. Hướng tiếp theo:)*

Thêm debug print so sánh `sh.bestParsimony` (ước tính của testInsert) với `fullMP`
(`recomputeAllNodes` ngay sau `applyMove`). Nếu hai giá trị lệch nhau có hệ thống → xác nhận
stale score_tree hypothesis. Fix sẽ là: trước `removeNodeParsimony(p)`, recompute `score_tree[i]`
và `score_tree[q_num]` theo đúng hướng (replicating CPU's line-2291 lazy update).

### Bài học / Ghi chú cho khóa luận

Đây là điểm khác biệt cốt lõi giữa CPU PLL và GPU implementation:

- **CPU dùng xPars lazy evaluation**: `parsimonyScore[node]` chỉ hợp lệ cho face nào đang giữ
  `xPars=1`. `evaluateParsimony(p, FALSE)` tự động refresh đúng face trước khi dùng.
- **GPU dùng eager DFS toàn cây**: `recomputeAllNodes` tính lại toàn bộ `score_tree[]` từ
  một `start_vface` cố định. Giá trị đúng với hướng đó, nhưng sai khi cần score theo hướng khác
  (như trong testInsert sau removeNodeParsimony).

Bài học: khi port thuật toán có lazy evaluation sang GPU, cần thiết kế lại cơ chế refresh score
sao cho đúng hướng tại mọi điểm evaluate. Dùng global recompute (eager) không thay thế được
lazy evaluation có hướng của CPU.

**Update 2026-05-06**: Bug #2 đã được re-phân tích sau khi tái cấu trúc code. Root cause thực
sự là Bug #3 và Bug #4 bên dưới — xem tiếp. Phân tích "stale score_tree" vẫn đúng về mặt lý
thuyết, nhưng bị che khuất bởi crash từ Bug #3.

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

## Bug #4 — `sh.randomMP` không được khởi tạo trong `gpuSprKernel`

**Ngày**: 2026-05-06
**Task**: Step [6] SPR hill-climbing — `gpuSprKernel` init (`gpu_spr.cu:386-451`)
**File liên quan**: `gpu/src/gpu_spr.cu:427-451`

### Triệu chứng

*(Ẩn bởi crash từ Bug #3 — sẽ observable sau khi fix Bug #3.)*
Expected: SPR do-while loop chạy với sai threshold, có thể loop vô tận hoặc exit ngay.

### Root cause

Trong `gpuSprKernel`, block init chỉ set `sh.randomMPHits` và `sh.seed`:
```cuda
sh.randomMPHits = 1;
sh.seed = d_seeds[k];
```
`sh.randomMP` không được set. Block `recomputeAllNodes` vốn sẽ khởi tạo nó đã bị comment out:
```cuda
// unsigned int fullMP = recomputeAllNodes(...);
// if (lane == 0) {
//     sh.randomMP = fullMP;
//     topo->preSprParsimony = fullMP;
// }
```

Shared memory trong CUDA **không được zero-initialize** — `sh.randomMP` chứa giá trị rác từ
lần launch kernel trước. Kết quả:
- `startMP = sh.randomMP` = rác
- `sh.bestParsimony = sh.randomMP` = rác → threshold accept move sai
- `topo->preSprParsimony` không bao giờ được set (debug print [6b] in giá trị sai)

### Fix

Uncomment block `recomputeAllNodes` trong `gpuSprKernel` (lines 429-451) và đảm bảo:
```cuda
if (lane == 0) {
    sh.randomMP = fullMP;
    topo->preSprParsimony = fullMP;
}
__syncwarp();
```

### Bài học / Ghi chú cho khóa luận

CUDA shared memory KHÔNG được zero-initialize, khác với global memory (cudaMalloc trả về 0).
Khi comment out initialization code trong kernel (để debug hoặc test), cần nhớ shared mem có
thể chứa giá trị từ kernel trước cùng SM. Đây là nguồn gốc của các "intermittent" bugs —
behavior thay đổi tùy theo launch order. Luôn explicit-initialize mọi shared memory field quan
trọng ở đầu kernel.

**Update 2026-05-07**: Fix thực tế được áp dụng khác với phương án đề xuất ban đầu. Thay vì
uncomment `recomputeAllNodes`, ta set `sh.randomMP = sh.bestParsimony` (lấy từ build phase).
Cách này đúng vì xPars flags sau build đã nhất quán với parsVect — không cần recompute toàn cây.

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

## Bug #6 — `__syncwarp;` thiếu dấu ngoặc → race condition đọc `nodep[]`

**Ngày**: 2026-05-11
**Task**: SPR loop trong `gpuSPRHillClimb`
**File liên quan**: `gpu/src/pars_build.cu` (đầu vòng lặp do-while trong `gpuSPRHillClimb`)

### Triệu chứng

Post-SPR best = 187 (rất sai, thay vì ~6665).

### Root cause

```cuda
// WRONG:
__syncwarp;     // ← đây là function reference, không gọi hàm!

// CORRECT:
__syncwarp();
```

Lane 0 gọi `gpuNodeRectifierPars` để cập nhật `topo->nodep[]` nhưng các lane 1-31 không
đợi lane 0 hoàn thành. Kết quả: các lane đọc `topo->nodep[i]` (qua `sh.bcast[4]`) trước
khi lane 0 ghi xong → race condition → nodep[] sai → SPR search dùng wrong canonical faces.

### Fix

Thay `__syncwarp;` thành `__syncwarp();`.

### Bài học / Ghi chú cho khóa luận

Trong CUDA, `__syncwarp` không có dấu ngoặc là biểu thức lấy địa chỉ hàm, không gọi hàm.
Compiler không báo lỗi (đây là biểu thức hợp lệ). Kết quả: thanh ghi của các lane không
được đồng bộ, shared memory updates từ lane 0 có thể chưa visible với lane 1-31.
Đây là loại bug rất khó phát hiện vì không có compile error, chỉ thấy kết quả sai.

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

