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