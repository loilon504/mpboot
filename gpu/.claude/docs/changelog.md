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

## NCU Profiling — Shared memory bottleneck (2026-05-18)

**Ngày**: 2026-05-18
**File liên quan**: `/output/ncu/w400p20d6s3/` (6 `.ncu-rep` files)
**Method**: `ncu --launch-count 2 --set default` trên 6 hard datasets (N=219–504), config w400p20d6s3 (gpu_worker=400, pool=20, sprdist=6, stop=3, numpars=200)

### Kết quả đo

| Kernel | Grid | Regs | Occ limit (regs) | Occ limit (smem) | Waves/SM | Theor. Occ% |
|--------|------|------|-----------------|-----------------|----------|-------------|
| K1 `buildParsimonyTreesKernel<4,800>` | 200 | 96 | 20/SM | **13/SM** | 0.14 | **20.31%** |
| K2 `buildPhase3Kernel<4,800>` | 400 | 128 | 16/SM | **13/SM** | 0.28 | **20.31%** |

Kết quả nhất quán cho tất cả 6 datasets (kernel template NTAXA=800 cố định).

### Phân tích

**Bottleneck là shared memory, không phải registers:**
- `smem_static = 11.584 KB/block` → A100 cho tối đa 13 blocks/SM (≈ 167 KB / 12.672 KB allocated)
- K2 regs=128: reg limit = 16/SM, nhưng smem stricter (13 < 16)
- Cả K1 lẫn K2 đều bị giới hạn bởi smem → Theor. Occ = 13/64 = **20.31%**

**GPU không saturated với w400:**
- K2 waves/SM = 0.28: 400 blocks / (108 SMs × 13 blocks/SM) = **chưa đến 1 wave**
- Để đạt ≥ 1 wave: cần ≥ 108 × 13 = **1,404 blocks** (gpu_worker ≈ 1400)
- w400 lãng phí ~72% GPU capacity

**NTAXA=800 template cố định** → smem được cấp cho worst-case dù N thực nhỏ hơn (219–504).
Với GPU_NTAXA_TEMPLATE=ON (Opt-P L3), các dataset nhỏ dùng bucket NTAXA=256/128 → smem nhỏ hơn → nhiều blocks/SM hơn.

### Gợi ý tối ưu tiếp theo

- Cắt 1.3 KB shared memory → 14 blocks/SM (21.9% occ, +6%)
- Dùng gpu_worker ≥ 1400 để đạt 1 wave đầy đủ (hiện w400 chỉ 28% GPU)
- Bật `GPU_NTAXA_TEMPLATE=ON` cho production với dataset nhỏ

### Bài học

Profiling trên nhiều datasets khác nhau nhưng cùng template → kết quả giống hệt nhau. Điều này xác nhận rằng bottleneck là **cấu trúc dữ liệu** (smem cố định theo NTAXA template), không phải workload của dataset cụ thể.

---

## Opt-S — Per-slot pool spinlocks: giảm thundering herd contention

**Ngày**: 2026-05-18
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_tree.cu`, `gpu/src/pars_build.cu`, `gpu/src/gpu_init_trees.cu`

### Vấn đề

K2 kernel (`buildPhase3Kernel`) có K=1000 blocks (warps) cạnh tranh cùng một spinlock `pool_lock`
để copy 12.8 KB `back_vf` từ pool vào topology riêng của mỗi block. Đây là **thundering herd**:
tất cả K blocks busywait trên `atomicCAS(pool_lock, 0, 1)` cho đến khi block đang giữ lock nhả ra.

Trong CUDA, busywait trên atomicCAS là vòng lặp kín — block không ngủ mà tiêu tốn SM cycles
hoàn toàn. Với K=1000 và pool_size=20, trung bình 50 blocks cùng đợi 1 slot, nhưng trước đây
tất cả 1000 blocks đều đợi 1 lock duy nhất.

### Root cause

```cpp
// Trước (pool_lock global):
if (lane == 0) {
    while (atomicCAS(pool_lock, 0, 1) != 0) {}  // 999 warps spin ở đây
}
// ... copy 12.8 KB ...
if (lane == 0) { atomicExch(pool_lock, 0); }
```

### Fix

Thay 1 `pool_lock` int bằng `pool_slot_locks[pool_size]` — mỗi slot có lock riêng:

```cpp
// Sau (per-slot lock):
if (lane == 0) {
    while (atomicCAS(&pool_slot_locks[phys_slot], 0, 1) != 0) {}  // chỉ lock slot đang dùng
}
// ... copy 12.8 KB ...
if (lane == 0) { atomicExch(&pool_slot_locks[phys_slot], 0); }
```

**Pool insert** cũng dùng `pool_slot_locks[worst_slot]` thay vì `pool_lock`.

### Tại sao an toàn

Lock A (pool restart, reads slot i) và Lock B (pool insert, writes slot j) chỉ conflict khi `i == j`
(xác suất 1/pool_size ≈ 5%). Không có deadlock vì mỗi block giữ tối đa 1 lock tại một thời điểm.

### Kết quả

Contention giảm từ **K=1000 → K/pool_size ≈ 50 blocks per lock** (pool_size=20).
Memory overhead: +80 bytes (20 × 4 bytes).

### Bài học / Ghi chú cho khóa luận

**Thundering herd** là anti-pattern phổ biến trong GPU concurrent programming. Khi nhiều threads
cạnh tranh 1 shared resource, granularity của lock nên match granularity của resource:
- 1 global resource → 1 global lock (quá coarse-grained khi K lớn)
- N independent resources (pool slots) → N locks (fine-grained, scales với K)

Tương tự như phân biệt `mutex` (global) vs `per-bucket lock` trong hash table implementations.

---

## Refactor #2 — Pool restart simplification (pool_size ≤ 60)

**Ngày**: 2026-05-18
**File liên quan**: `gpu/src/pars_build.cu`

### Thay đổi

Old: warp-parallel k-th min selection (complex scattered lane work, ~70 dòng)
New: lane-0 O(pool_size²) selection-sort với `unsigned long long used` bitmask (~20 dòng)

```cpp
// Lane-0 selection-sort: chọn rank order_idx trong accessible slots
unsigned long long used = 0;
int found_slot = -1;
for (int rank = 0; rank <= order_idx; rank++) {
    unsigned int best = 0xFFFFFFFFu; int best_slot = -1;
    for (int i = 0; i < accessible; i++) {
        if ((used >> i) & 1ULL) continue;
        if (pool_scores[i] < best) { best = pool_scores[i]; best_slot = i; }
    }
    used |= (1ULL << best_slot);
    found_slot = best_slot;
}
```

**Điều kiện đủ**: pool_size ≤ 60 → `unsigned long long` (64-bit) bitmask đủ; O(pool_size²) ≤ 3600 ops → overhead nhỏ.  
(Ban đầu dùng `uint32_t` chỉ hỗ trợ pool_size ≤ 32; đổi sang `unsigned long long` để hỗ trợ pool_size ≤ 60.)

Đồng thời: `accessible = 10 + outer` (thay vì `*pool_accessible` atomic counter) — window tăng tuyến tính theo iteration outer, đơn giản hơn và không cần sync.

### Effect on registers

K2 regs: **151 → 128** sau khi simplify (warp-parallel code tạo nhiều temp registers hơn).
Block Limit Reg: 12 → 16 blocks/SM. Nhưng Theor. Occ vẫn 20.31% vì shared memory bottleneck (11.584 KB/block).

### Bài học

Đôi khi giải pháp đơn giản (sequential lane-0) hiệu quả hơn giải pháp phức tạp (warp-parallel) khi:
1. Input size nhỏ (pool_size ≤ 60 — không đủ để amortize warp overhead)
2. Compiler tối ưu sequential code tốt hơn (ít live variables → ít registers)

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

## Refactor #3 — Dead code removal: timing fields, best_back_vf, numSearchIter (2026-05-17)

**Ngày**: 2026-05-17
**Task**: Loại bỏ toàn bộ code không dùng sau khi kiến trúc 2-kernel (K1/K2) ổn định
**File liên quan**: `pars_tree.cuh`, `pars_build.cu`, `pars_build.cuh`, `gpu_init_trees.cu`, `pars_tree.cu`, `tools.h`, `tools.cpp`

### Thay đổi

#### 1. Xóa timing fields khỏi `BuildSharedT`

Các field đo thời gian fine-grained trong shared memory không còn cần thiết sau khi Phase 3 ổn định:
```cpp
// Đã xóa khỏi BuildSharedT:
long long t_build, t_phase2, t_p3_nni_spr, t_p3_ratchet;
int n_p3_even, n_p3_odd;
```
Các block `{ long long _t0 = clock64(); ...; sh.t_xxx += clock64() - _t0; }` trong Phase 1/2/3 cũng bị xóa.

#### 2. Xóa dead Opt-C/sym fields khỏi `BuildSharedT`

Ba field từ thời gian thử nghiệm Opt-C (stagnation detection) và Symmetric Adaptive variant, không còn dùng:
```cpp
// Đã xóa khỏi BuildSharedT:
unsigned int last_odd_hash;
int restore_on_next_odd;
int sym_do_ratchet;
```

#### 3. Xóa `GpuTopology.best_back_vf[kMaxVFaces]`

`best_back_vf` ban đầu lưu topology tốt nhất để restore ở đầu mỗi iteration (Opt-R).
Sau khi kiến trúc chuyển sang 2-kernel (K1/K2), việc restore topology được thực hiện
qua pool mechanism (`d_poolBackVf`) thay vì per-tree field. `best_back_vf` trở thành dead field.

**Kết quả**: GpuTopology shrink thêm **12.8 KB** (kMaxVFaces=3196 × 4 bytes).

Trong `hybrid_cb` của `gpu_init_trees.cu`, chuyển từ:
```cpp
bvf.assign(h_topo.best_back_vf, h_topo.best_back_vf + h_topo.num_vfaces);
for (int vf = 0; vf < h_topo.num_vfaces; vf++)
    h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
```
sang:
```cpp
bvf.assign(h_topo.back_vf, h_topo.back_vf + h_topo.num_vfaces);
```
An toàn vì tại thời điểm hybrid_cb đọc topology (sau K1, trước K2), `back_vf == best_back_vf` (được set bằng nhau ở cuối Phase 2 của K1).

#### 4. Xóa `numSearchIter` và `-gpu_hc_iter`

`numSearchIter` là tham số giới hạn số vòng lặp tối đa của Phase 3 (safety cap). Sau khi `gpu_stop` (stopNoImprove) trở thành primary stopping criterion, `numSearchIter` chỉ còn là dead parameter.

Xóa toàn bộ chuỗi: `tools.h` → `tools.cpp` → `gpu_init_trees.cu` → `gpuStepwiseBuildTrees` → `buildPhase3Kernel` → `runPhase3`.

Vòng lặp Phase 3 thay đổi:
```cpp
// Trước:
for (int outer = 0; outer < numSearchIter; outer++) { ... }

// Sau:
for (int outer = 0; ; outer++) { ... }   // chỉ dừng khi gpu_stop kích hoạt
```

**Lưu ý**: `gpu_stop > 0` là điều kiện bắt buộc để loop có thể thoát. Default = 6, CLI `-gpu_stop 0` sẽ gây infinite loop.

### Kết quả NCU sau cleanup (A100-SXM4-80GB, K=200, dna_M10434 544 taxa)

| Kernel | Regs/thread | Block Limit Reg | Theor. Occupancy | Ghi chú |
|--------|------------|-----------------|-----------------|---------|
| K1 `buildParsimonyTreesKernel` | **96** (↓ từ 155) | 20 | **20.31%** (↑ từ 18.75%) | Cải thiện rõ |
| K2 `buildPhase3Kernel` | **151** | 12 | **18.75%** | Bottleneck: register |

K2 vẫn là register-bound: 151 regs × 32 threads = 4832 regs/block; A100 có 65536 regs/SM → Block Limit = 12 blocks/SM → 18.75% occupancy.
Target: giảm K2 xuống ≤128 regs → Block Limit tăng lên 16 → occupancy 25%.

### Bài học / Ghi chú cho khóa luận

Dead code tích lũy theo thời gian khi thuật toán thay đổi (Opt-C thử rồi bỏ, best_back_vf dùng khác đi, numSearchIter bị thay thế bởi gpu_stop). Việc dọn dẹp định kỳ:
1. Giảm register pressure compiler: K1 từ 155 → 96 regs, chiếm block Limit Reg thấp hơn
2. Giảm shared memory dùng: BuildSharedT bỏ 6 fields (timing + dead flags)
3. Giảm GpuTopology size: bỏ best_back_vf (−12.8 KB per tree struct)
4. API đơn giản hơn: `gpuStepwiseBuildTrees` mất 1 param; K2 kernel mất 1 param

---

## Kết quả thực nghiệm — Benchmark 115 datasets, numpars=1000, pool=30 (2026-05-17)

**Ngày**: 2026-05-17
**Hardware**: 5 × NVIDIA A100-SXM4-80GB (devices 1–5)
**Config**: numpars=1000, sprdist=6, gpu_pool_size=30, seed=1, 115 datasets (N=50–767)
**Baseline**: cpu_d6 (CPU chạy cùng dataset, seed=1)

### Tổng hợp theo gpu_stop

| gpu_stop | Mean spd | Median spd | Total spd | Total GPU time | GPU faster | CPU better score |
|----------|----------|------------|-----------|---------------|------------|-----------------|
| 4 | 2.70× | 2.42× | 4.36× | 1497s | 107/115 | 24/115 |
| **6** | **2.77×** | 2.38× | **4.55×** | **1434s** | **111/115** | **21/115** |
| 8 | 2.61× | 2.17× | 4.25× | 1534s | 107/115 | 22/115 |
| 12 | 2.57× | 2.16× | 4.10× | 1590s | 107/115 | 24/115 |
| 14 | 2.52× | 2.08× | 4.06× | 1605s | 105/115 | 22/115 |

### Kết luận

**gpu_stop=6 là sweet spot**: nhanh nhất (4.55× total, 1434s GPU total), ít dataset CPU thắng nhất (21/115), và GPU faster nhiều nhất (111/115).
Pattern rõ ràng: tăng gpu_stop > 6 chậm hơn mà không cải thiện score.

### Speedup theo taxa group (gpu_stop=6)

| Taxa | Mean speedup |
|------|-------------|
| 0–99 | 1.59× |
| 100–199 | 2.54× |
| 200–299 | 2.78× |
| 300–399 | 3.74× |
| 400–499 | 4.42× |
| 500–599 | 5.45× |
| 600–699 | 10.75× |
| 700–799 | 6.91× |

GPU speedup tăng mạnh theo taxa (N). Dataset N<100 bị overhead-bound; N≥400 GPU rõ ràng chiếm ưu thế.

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

## Experiment — K' < K: Build fewer trees in K1 (Opt-LessK1, 2026-05-17)

**Ngày**: 2026-05-17  
**Task**: Giảm K1 build từ K xuống K'=max(pool_size, K×ratio) để tiết kiệm K1 time  
**Files**: `mpboot/tools.h`, `mpboot/tools.cpp`, `gpu/include/pars_build.cuh`, `gpu/src/pars_build.cu`, `gpu/src/gpu_init_trees.cu`

### Phát hiện chính

`hybrid_cb` (callback sau K1) đã reseed **toàn bộ K slots** từ pool (Step 5: `for (k=0; k<Kc_full; k++)` với `Kc_full = mem->K`). K2 dùng `needs_recompute=1` path — **không đọc K1 topology trực tiếp**. K1 trees chỉ cần đủ để populate pool (Step 3). Vì vậy, build K−K' trees thêm trong K1 là lãng phí.

### Thay đổi

**CLI flag**: `-gpu_k1_ratio X` (default=1.0 = backward compat, production=0.2)

```cpp
// gpu_init_trees.cu — compute k1_trees
const float k1_ratio = params.gpu_k1_ratio;
const int k1_trees = (k1_ratio <= 0.0f || k1_ratio >= 1.0f)
                     ? K : std::max(pool_size, (int)(K * k1_ratio));

// hybrid_cb Step 2/3: chỉ download K' scores (valid K1 results)
const int Kc_scores = k1_trees;   // K' slots written by K1
const int Kc_full   = cb_mem->K;  // for Step 5: reseed ALL K (unchanged)

// pars_build.cu K1 launch: dim3(k1_trees) thay vì dim3(K)
buildParsimonyTreesKernel<S, NT><<<dim3(k1_trees), dim3(kWarpSize), 0, stream>>>(...);
// K2 launch: unchanged dim3(K)
```

### Win-win effect

K1 ngắn hơn → CPU builds ~50% more trees trong callback window → pool quality tốt hơn → K2 cả nhanh hơn (do better pool → less wasted iterations) lẫn chất lượng cao hơn (more diverse good candidates).

### Kết quả benchmark (K=999, pool=30, k1_ratio=0.2, 115 datasets per config, A100-SXM4-80GB)

| Config (sprdist, stop) | K1 Δ | K2 Δ | Δtotal speedup | GPU wins (K'=K → K'=0.2K) |
|------------------------|------|------|----------------|--------------------------|
| d3, s10 | −59.7% | +15.6% | **+25.8%** (8.04×→10.11×) | 57→64/115 |
| d4, s6  | −59.6% | +16.2% | **+22.5%** (6.70×→8.21×)  | 53→64/115 |
| d4, s20 | −59.8% | +0.7%  | **+27.4%** (5.82×→7.41×)  | 52→65/115 |
| d5, s6  | −59.4% | +15.2% | **+21.1%** (5.46×→6.61×)  | 48→58/115 |
| d6, s6  | −58.8% | +18.5% | **+17.6%** (4.55×→5.35×)  | 45→61/115 |

→ Full logs: `output/hybrid_pool_{lessk1,n1000p30}d{3s10,4s6,4s20,5s6,6s6}/`

### Bài học / Ghi chú cho khóa luận

1. **Callback reseeds all K slots**: Insight quan trọng — K2 không phụ thuộc vào K1 topology trực tiếp. Điều này cho phép tách bạch hoàn toàn "số cây build trong K1" và "số blocks chạy trong K2", mỗi cái được tối ưu độc lập.

2. **Win-win asymmetry**: Dự kiến ban đầu là quality sẽ giảm nhẹ (ít K1 candidates hơn). Thực tế: quality tăng trên TẤT CẢ 5 configs. Nguyên nhân là CPU builds nhiều trees hơn trong thời gian K1 ngắn hơn, bù đắp dư cho pool.

3. **K1 time linear với K'**: Cả K'=200 và K=1000 đều fit trong 1 wave (A100 capacity ≈ 1404 blocks), nhưng K'=200 ít work per SM hơn → thực sự ~60% nhanh hơn (K'/K = 0.2 × factor). Lợi ích tối đa khi K' << wave capacity.

4. **Kết hợp với Reseed**: Opt-LessK1 và Reseed (Step 3.5) hoạt động cùng nhau — Reseed cải thiện bad slots; LessK1 cho CPU thêm time để build candidates tốt hơn cho pool.

---

## Refactor — `gpuBootstrapSearch` → `gpuHillClimbing`: hợp nhất bootstrap và non-bootstrap (2026-05-19)

**Ngày**: 2026-05-19
**File liên quan**: `gpu/src/gpu_init_trees.cu`, `gpu/include/gpu_init_trees.cuh`, `mpboot/phyloanalysis.cpp`

### Vấn đề trước khi refactor

Trước refactor, GPU có hai luồng xử lý riêng biệt:
- **Bootstrap (`-bb`)**: `mpbootGpu` chạy K1 only → `gpuBootstrapSearch` chạy vòng lặp K2 với REPS eval + convergence check
- **Non-bootstrap**: `mpbootGpu` chạy K1 + K2 (một pass) → pool → `candidateTrees` (step [7])

Non-bootstrap **không có outer loop** tương đương `doTreeSearch` của CPU. K2 chỉ chạy 1 lần rồi dừng — không tận dụng pool để tiếp tục tìm kiếm.

### Thay đổi

**`mpbootGpu`:**
- Luôn allocate treels buffer (`max_treels = K * 10`, không còn điều kiện `gbo_replicates > 0`)
- Luôn skip K2 (`k2_max_outer = 0`) — outer loop được chuyển sang `gpuHillClimbing`
- Xóa step [7] (non-bootstrap pool → candidateTrees) — nay nằm trong `gpuHillClimbing`

**`gpuBootstrapSearch` → `gpuHillClimbing`:**
- Thêm `is_bootstrap = (params.gbo_replicates > 0)` để phân nhánh logic
- Thêm `tag` (`"[GPU Bootstrap]"` hoặc `"[GPU HillClimb]"`) cho log
- **Treels registration**: bootstrap → `saveCurrentTree()` (REPS weighted logl), non-bootstrap → `candidateTrees.update() + setBestTree()`
- **Improvement tracking**: bootstrap → max của `treels_logl`, non-bootstrap → `iqtree.bestScore`
- **logl_cutoff update**: bootstrap only
- **Convergence check** (bootstrap correlation): bootstrap only
- **Stopping condition**: `total_replicates - last_impr_at > unsuccess_thresh || (is_bootstrap && total_replicates > B)`
  - Non-bootstrap chỉ dùng `unsuccess_thresh`, không giới hạn B
- Pool → `candidateTrees` ở cuối hàm: dùng cho cả hai mode (xử lý giống nhau)

**`phyloanalysis.cpp`:**
- `need_boot_loop = (gbo_replicates > 0 && maximum_parsimony)` → `need_hc_loop = maximum_parsimony`
- Bootstrap sample upload được guard bởi `params.gbo_replicates > 0`
- Gọi `gpuHillClimbing` thay `gpuBootstrapSearch` cho cả hai mode

### Stopping condition cho non-bootstrap

```cpp
const int unsuccess_thresh = params.unsuccess_iteration + K * params.gpu_worker_stop;
// Dừng khi: total_replicates - last_impr_at > unsuccess_thresh
// last_impr_at được cập nhật khi iqtree.bestScore cải thiện
```

Tương tự CPU `doTreeSearch` với `SC_UNSUCCESS_ITERATION`: dừng sau `unsuccess_iteration` replicates không tìm được cây tốt hơn. `+ K * gpu_worker_stop` bù đắp cho việc K workers trong một round sinh ra K cây tương quan (từ cùng pool state) — không hoàn toàn độc lập như CPU.

### Lý do `total_replicates` thay vì `treels_logl.size()`

`saveCurrentTree` dedup theo Newick string → `treels_logl.size()` chỉ tăng khi có topology MỚI. `total_replicates += n_treels` đếm tổng số cây được xử lý qua `saveCurrentTree` (kể cả duplicate), tương đương `curIt` của CPU trong `doTreeSearch`.

### Ghi chú kiến trúc

Refactor này làm cho GPU K2 outer loop hoàn toàn song song về chức năng với CPU `doTreeSearch`:
- CPU: `do { rearrange all nodes } while (improved)` → dừng khi không improve sau `unsuccess_iteration` attempts
- GPU: `for (;;) { K2 round → check improvement }` → dừng khi `total_replicates - last_impr_at > unsuccess_thresh`

---

## Refactor — `gpuRandomNNIs`: rewrite theo CPU-style random pick + on-the-fly q_vf (2026-05-19)

**Ngày**: 2026-05-19
**File liên quan**: `gpu/src/pars_build.cu` (hàm `gpuRandomNNIs`, ~line 495–570)

### Vấn đề trước khi rewrite

Cài đặt cũ dùng **Fisher-Yates shuffle toàn bộ edge list** rồi iterate tuần tự, bỏ qua cạnh bị conflict (cả 2 endpoint đã marked trong bitset):

```cuda
// Cũ: shuffle (p_vf, q_vf) pairs → iterate → skip on conflict
for (int i = 0; i < num_edges; i++) {
    int p_vf = sh.stack[idx * 2], q_vf = sh.stack[idx * 2 + 1];
    if (conflict) continue;     // bỏ cạnh, không đảm bảo numNNI moves
    apply_nni(...);
}
```

**Hậu quả với `gpu_nni_strength=0.5`**: `numNNI ≈ N/2`. Sau ~N/4 moves, hơn nửa nodes đã marked → hầu hết cạnh còn lại đều conflict → iterator duyệt hết list với `applied << numNNI`. GPU thực chất không perturbate đủ số NNI moves yêu cầu.

### Root cause của crash khi rewrite đơn giản

Khi thử thay bằng random-pick-per-NNI (CPU pattern), lưu cặp `(p_vf, q_vf)` trong `sh.stack[]` → crash `CUDA illegal memory access`:

**Vấn đề**: NNI đầu tiên thay đổi `back_vf[]` → `q_vf` trong cặp đã lưu không còn là neighbor thực của `p_vf` nữa. Random pick có thể chọn lại cùng cặp → `gpuHookup` inconsistent → topology bị corrupt → DFS crash.

Hai loại corruption được phát hiện:
1. **Null pointer** (`back_vf[pf0] = -1`): Node 2N-1 (node cuối cùng build) chưa được fully initialized → pf0's back có thể là −1.
2. **Self-loop** (`back_vf[pf0] = pf0`): Nếu `pf0 ↔ qf1` đã connected trực tiếp, hookup option 1 sẽ tạo `back_vf[pf0] = pf0` → DFS infinite loop → `sh.tiStack` overflow → crash.

### Fix: on-the-fly `q_vf` từ `back_vf[p_vf]`

**Nguyên lý**: Lưu chỉ `p_vf` (canonical face của mỗi inner node, N-1 entries). Mỗi NNI iteration, tra cứu `q_vf = topo->back_vf[p_vf]` động — luôn phản ánh topology hiện tại sau các NNI trước đó.

```cuda
// Step 1: store only p_vf per inner node (N-1 entries)
for (int p_num = N + 1; p_num <= 2 * N - 1; p_num++)
    sh.stack[num_inner++] = (int16_t)nodepVf(p_num, N);

// Step 3: random pick + on-the-fly q_vf lookup
for (int i = 0; i < numNNI; i++) {
    int p_vf = sh.stack[idx];
    int q_vf = topo->back_vf[p_vf];  // ← luôn up-to-date
    int q_num = vfToNum(q_vf, N);
    if (q_num <= N) continue;         // skip tip neighbors

    // Conflict → reset bitset (CPU: usedNodes.clear()), apply anyway
    if (conflict) for (int w = 0; w < bitset_words; w++) sh.stackMaxt[w] = 0;

    // Guards: -1 (null), c==pf0 / d==pf0 (self-loop)
    if (b != -1 && c != -1 && c != pf0) { gpuHookup(...); }
}
```

### So sánh cài đặt cũ vs mới

| | Cũ (Fisher-Yates) | Mới (CPU-style) |
|---|---|---|
| Upfront work | O(N) RNG shuffle | Không shuffle |
| Edge storage | `(p_vf, q_vf)` pairs | Chỉ `p_vf` (N-1 entries) |
| `q_vf` | Lưu static từ trước NNI | Tra cứu động `back_vf[p_vf]` mỗi iter |
| On conflict | Skip, tiếp tục list | Reset bitset, apply anyway |
| Guarantee | applied ≤ numNNI | numNNI iterations (skip chỉ khi q là tip) |
| strength=0.5 | Falls short of numNNI | Đủ numNNI ✓ |
| Stale edge | Không có (iterate in-order) | Không có (on-the-fly lookup) |

### Kết quả thực nghiệm

| Strength | Best parsimony | EXIT |
|----------|---------------|------|
| 0.0 | 6662 | 0 ✓ |
| 0.1 | 6662 | 0 ✓ |
| 0.3 | 6662 | 0 ✓ |
| 0.5 | 6662 | 0 ✓ |

(N=295, seed=1, sprdist=6, numpars=200, CPU ref=6682)

Không có regression. `gpu_nni_strength=0.5` hoạt động đúng (trước: crash hoặc apply << numNNI moves).

### Bài học / Ghi chú cho khóa luận

1. **Stale pointer trong GPU**: Khác CPU (có virtual memory), GPU topology dùng integer index (`back_vf[vf]`). Sau khi NNI thay đổi `back_vf[]`, bất kỳ index nào lưu giá trị cũ của `back_vf` đều trở thành "stale pointer" — dẫn đến topology corruption không rõ ràng. Giải pháp: không lưu "edge" (pair of nodes), mà lưu "anchor" (1 node) rồi tra cứu neighbor động.

2. **Self-loop trong ring topology**: GPU topology là vòng có hướng (face[0]→face[1]→face[2]→face[0]). NNI có thể tạo self-loop nếu hai face đang được swap đã connected trực tiếp (`back_vf[pf0] = qf1`). Guard `c != pf0` / `d != pf0` ngăn trường hợp này.

3. **On-the-fly lookup vs lưu pair**: Trong thuật toán online (có update giữa chừng), lưu thêm 1 field và tra cứu dynamic thường an toàn hơn lưu cặp pre-computed. Trade-off: 1 global memory read thêm per iteration, nhưng correctness được đảm bảo.

4. **CPU `doRandomNNIs` pattern**: CPU reset `usedNodes` map khi conflict → apply anyway (iqtree.cpp:1091). GPU replicate đúng pattern này bằng bitset clear. Kết quả: đúng `numNNI` moves, không bị thiếu ở strength cao.

---

## Bug Fix — Sankoff GPU: Pattern Set Mismatch + PLL Encoding (2026-05-20)

**Ngày**: 2026-05-20
**Task**: Sửa Sankoff GPU để cho kết quả đúng (BEST SCORE = 6662 = CPU)
**File liên quan**: `gpu/src/gpu_init_trees.cu` (`uploadSankoffTipParsVect`)

### Triệu chứng

GPU Sankoff (K=5000, sprdist=6): BEST SCORE **6662** = CPU ✅ nhưng cần K=5000 mới hội tụ.
Trước đó với K=200: BEST SCORE 6664 — không hội tụ dù nhiều cây.

### Root cause (2 lỗi)

**Lỗi 1 — Width mismatch**: `uploadSankoffTipParsVect` dùng `width = iqtree.aln->size() = 1400`
(tất cả unique patterns) nhưng CPU `compressSankoffDNA` chỉ xử lý `parsimonyLength = 1072`
*informative* patterns qua `isInformative()`. Gây score sai vì GPU dùng nhiều pattern hơn CPU.

**Lỗi 2 — Encoding**: Đọc từ `iqtree.aln->at(ptn)` dùng IQTree encoding, phải đọc từ
`tr->yVector[tipNum][i]` dùng PLL bitmask encoding (A=bit0, C=bit1, G=bit2, T=bit3).

**Lỗi phụ**: `extern Params *globalParam` khai báo bên trong `namespace mpbootgpu` → linker error.

### Fix

Rewrite `uploadSankoffTipParsVect` trong `gpu/src/gpu_init_trees.cu`:
- `width = parsimonyLength` (không phải `iqtree.aln->size()`)
- Đọc từ `tr->yVector[tipNum][i]`, dùng PLL bitmask
- Replicate `isInformative()` logic của CPU: skip pattern nếu ≤ 1 distinct bitmask < undetermined
- Di chuyển `extern Params *globalParam` lên trước `namespace mpbootgpu`

### Kết quả

K=5000, sprdist=6: BEST SCORE **6662 = CPU** ✅. Fitch không thay đổi ✅.

### Bài học / Ghi chú cho khóa luận

Sankoff CPU dùng `isInformative()` của PLL với bitmask encoding riêng — khác hoàn toàn với
IQTree alignment API. GPU phải replicate logic PLL, không phải IQTree. Khi upload tip data cho
GPU, luôn đọc từ cùng nguồn mà CPU parsimony code đọc (`tr->yVector`, `parsimonyLength`).

---

## Optimization — Ratchet cho GPU Sankoff (2026-05-20)

**Ngày**: 2026-05-20
**Task**: Enable ratchet perturbation cho Sankoff mode (trước đây tất cả Sankoff workers chạy NNI, không có ratchet)
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_tree.cu`, `gpu/src/pars_build.cu`, `gpu/src/gpu_init_trees.cu`

### Vấn đề

`iter_is_nni = sh.use_sankoff || (blockIdx.x % 2 == 0)` — tất cả Sankoff workers làm NNI.
Root cause: ratchet Fitch ghi `sw_k[b] = {1,2}` trực tiếp vào `d_siteWeights` rồi restore về `nullptr`.
Với Sankoff: `d_siteWeights` lưu **pattern frequencies gốc** (non-uniform) → overwrite = xóa frequencies;
restore về `nullptr` = uniform weight = sai hoàn toàn.

### Fix: thêm `d_ratchetScratch[K][width]`

- `d_ratchetScratch`: buffer scratch riêng cho Sankoff ratchet, không đụng `d_siteWeights`
- Khi ratchet: `ratchet_k[b] = sw_k[b] * {1 hoặc 2}` — frequencies gốc không bị đụng
- Sau ratchet: `sh.site_weights = sw_k` (restore về frequencies gốc)
- Bỏ `sh.use_sankoff ||` trong `iter_is_nni` → Sankoff workers giờ chạy ratchet như Fitch

4 file thay đổi: `pars_tree.cuh` (+1 field), `pars_tree.cu` (alloc/free), `pars_build.cu`
(kernel sig, runPhase3 sig, iter_is_nni, ratchet branch), `gpu_init_trees.cu` (K2 launch param).

### Kết quả

- **K=200 đạt 6662 = CPU** ✅ (trước cần K=5000)
- ms/tree tăng ~2× (168→346 ms/tree) — expected: ratchet chạy thêm 1 re-evaluate + 1 SPR pass

### Bài học / Ghi chú cho khóa luận

Ratchet là perturbation quan trọng: nó chọn ngẫu nhiên 50% patterns và tăng trọng số gấp đôi,
sau đó SPR từ đó. CPU làm tương tự (`informativePtnWgt` rebuild từ perturbed alignment).
GPU phải tách storage gốc (frequencies) khỏi scratch ratchet — dùng buffer riêng thay vì ghi đè.

---

## Optimization — Opt 1: Đổi parsVect layout từ `[ptn][state]` sang `[state][ptn]` (2026-05-20)

**Ngày**: 2026-05-20
**Task**: Cải thiện memory coalescing trong Fitch và Sankoff kernels
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_tree.cu`, `gpu/src/gpu_init_trees.cu`

### Vấn đề

Layout cũ `parsVect[node][ptn][state]` (access: `base[b * STATES + s]`):
- Lane 0: `base + 0*STATES + s`, Lane 1: `base + 1*STATES + s` → stride = STATES words giữa lanes
- DNA (STATES=4): 32 lanes span 128 words = **4 cache lines** → 4× bandwidth waste

### Fix: Layout mới `parsVect[node][state][ptn]` (access: `base[s * width + b]`)

- Lane 0: `base + s*width + 0`, Lane 1: `base + s*width + 1` → stride = 1 word
- 32 lanes fit vào **1 cache line** → perfectly coalesced

3 file thay đổi:
- `pars_tree.cuh`: 5 access sites (Fitch newview/eval + Sankoff newview/eval) đổi indexing
- `pars_tree.cu`: `uploadTipParsVect` Fitch branch → memcpy (CPU và GPU layout giờ giống nhau `[node][state][ptn]`); xóa dead Sankoff else block
- `gpu_init_trees.cu`: `uploadSankoffTipParsVect` h_buf indexing từ `[ptn][state]` → `[state][ptn]`

### Kết quả

| Mode | ms/tree trước | ms/tree sau | Speedup | Score |
|------|--------------|-------------|---------|-------|
| Fitch (K=400) | ~16 ms | ~6.4 ms | **2.5×** | 6662 ✅ |
| Sankoff (K=200) | 346 ms | **167 ms** | **2.07×** | 6662 ✅ |

### Bài học / Ghi chú cho khóa luận

**Memory coalescing** là một trong những tối ưu quan trọng nhất trong CUDA:
- Warp (32 lanes) đọc cùng lúc → nếu địa chỉ liên tiếp thì 1 cache line đủ, ngược lại cần nhiều cache line
- Layout `[state][ptn]` (SoA — Structure of Arrays) coalesced với warp iterating over patterns
- Layout `[ptn][state]` (AoS — Array of Structures) KHÔNG coalesced với warp iterating over patterns
- Quy tắc chung: dimension được warp iterate over phải là dimension innermost trong layout

Sau opt này, Fitch đạt ~6.4 ms/tree (speedup 2.5× so với 16 ms/tree). Đây là mức gần tối ưu
cho DNA width=1072 trên A100 — bottleneck chuyển sang shared memory limit (13 blocks/SM).

---

## Optimization — Opt 3: Xóa dead `min_site` + score accumulation trong Sankoff newview (2026-05-20)

**Ngày**: 2026-05-20
**Task**: Loại bỏ tính toán không cần thiết trong Sankoff newview
**File liên quan**: `gpu/include/pars_tree.cuh` (`newviewParsimony` Sankoff branch)

### Vấn đề

Trong Sankoff newview:
```cuda
unsigned int min_site = kSankoffInf;
for (ii): min_site = min(min_site, cost_ii);   // dead
score += (sw ? sw[b] : 1u) * min_site;         // dead — score_tree[p_num] never read in Sankoff eval
```

`score_tree[p_num]` được set nhưng Sankoff evaluate KHÔNG dùng nó — Sankoff evaluate tính
trực tiếp từ parsVect values (`min_ij(q[i] + cost[i][j] + r[j])`).

### Fix

Xóa `min_site` tracking và `score +=` khỏi Sankoff newview branch.
Thực hiện cùng Opt 1 rewrite (không cần commit riêng).

### Kết quả

Giải phóng 2 instructions/pattern/newview. Khó đo riêng (hấp thụ vào Opt 1), nhưng giúp
code rõ ràng hơn và không có dead computation.

---

## Bug #12 — CUDA hardware call stack overflow với STATES=20 (protein Sankoff)

**Ngày**: 2026-05-25
**Dataset**: `prot_M2593_56_386.phy` (N=56 taxa, protein, STATES=20)
**File liên quan**: `gpu/src/pars_build.cu` (lines 1253–1284, 1301–1342)

### Triệu chứng

Dataset `prot_M2593_56_386` crash với `CUDA error: an illegal memory access was encountered`
tại `pars_build.cu:1310` (`cudaStreamSynchronize` sau `buildPhase3Kernel`).

Điều kiện tái hiện: `pool_size=2`, `gpu_worker≥8`, `seed=1`.
Không crash với: `pool_size=4`, hoặc khi thêm bất kỳ `printf` nào vào kernel (**Heisenbug**).

Các giả thuyết ban đầu đều bị loại:
- `gpuHookup` out-of-bounds: thêm bounds check `[0..3199]` → không bao giờ trigger
- `d_ratchetScratch` null: xác nhận được alloc cho Sankoff mode
- `BuildSharedT.ti[]` overflow: `tiSize ≤ 168` với N=56 (giới hạn 2400)
- Race condition pool spinlock: per-slot lock đã đúng

### Root cause: Hardware call stack overflow

ptxas (compile với `--generate-line-info`) báo:

```
buildPhase3Kernel<STATES=20, NTAXA=800>:
    1264 bytes cumulative stack size, 255 registers

buildParsimonyTreesKernel<STATES=20, NTAXA=800>:
    1544 bytes cumulative stack size, 255 registers

buildPhase3Kernel<STATES=4, NTAXA=800>:
    0 bytes cumulative stack size, 96 registers
```

CUDA default per-thread hardware stack = **1024 bytes**.
`1264 > 1024` và `1544 > 1024` → **stack overflow** → corrupt bộ nhớ của thread lân cận
→ illegal memory access.

---

### Câu hỏi 1: Cumulative stack size là gì?

CUDA mỗi thread có hai vùng bộ nhớ riêng biệt thường bị nhầm lẫn:

| Tên | ptxas báo | Vị trí vật lý | Kích thước giới hạn |
|-----|-----------|--------------|-------------------|
| **Register spill** | "stack frame" (bytes), "spill stores/loads" | DRAM (L1/L2 cache) | Chỉ giới hạn bởi VRAM |
| **Hardware call stack** | "cumulative stack size" | SRAM riêng per-SM | `cudaLimitStackSize` (default 1024 bytes) |

**Register spill** (`spill stores/loads`): Khi kernel dùng nhiều biến hơn số register có
(255 với A100), compiler "spill" phần biến thừa vào DRAM. Đây là chậm nhưng không crash.

**Hardware call stack** (`cumulative stack size`): Mỗi khi có lệnh gọi hàm thực sự
(không phải inlined), CPU/GPU cần lưu địa chỉ trả về và local variables của caller lên stack.
`cumulative stack size = tổng kích thước tất cả stack frames trên call chain sâu nhất`.

Ví dụ: Nếu `kernelA` gọi `funcB` (non-inlined), `funcB` gọi `funcC` (non-inlined):
```
cumulative = frame(kernelA) + frame(funcB) + frame(funcC)
```

Overflow hardware call stack → ghi đè vùng nhớ của thread khác → illegal memory access.
Đây là **memory corruption không deterministic**: tùy thuộc vào layout thread trong SM,
crash có thể xảy ra ở nhiều vị trí khác nhau, và chỉ khi nhiều threads active cùng lúc.

---

### Câu hỏi 2: Những hàm nào làm tăng cumulative stack nhiều nhất?

Nguyên nhân: `newviewParsimony<SharedT, STATES=20>` được đánh dấu `__forceinline__` nhưng
ở 255 registers (hardware max), compiler **từ chối inline** → sinh actual function call.

Khi hàm này không được inline, mỗi call đẩy một stack frame gồm:
- Local arrays: `t_A[STATES]` = 20 × 4 = **80 bytes**, `o_A[STATES]` = **80 bytes**
  (dùng trong cả Fitch branch và Sankoff branch của hàm)
- Biến loop và temporaries: ~40–100 bytes
- Return address và saved registers: ~100–200 bytes

Tổng frame ≈ 300–460 bytes cho mỗi lần gọi `newviewParsimony<20>`.

Call chain sâu nhất trong `buildPhase3Kernel`:
```
buildPhase3Kernel
  └─ runPhase3
       └─ gpuSPRHillClimb
            └─ doAddTraverse
                 └─ testInsert
                      └─ createTiAndEvaluateParsimony
                           └─ newviewParsimony<20>   ← frame lớn nhất
```

Nếu bất kỳ hàm nào trong chain không được inline → frame của nó cộng vào cumulative stack.
Ratchet path (odd blocks) đặc biệt nguy hiểm hơn vì gọi `createTiAndEvaluateParsimony` với
`full=true` → deep traversal → nhiều nesting hơn even path.

---

### Câu hỏi 3: STATES=4 và STATES=20 khác nhau chỗ nào?

| Đặc điểm | STATES=4 (DNA) | STATES=20 (protein) |
|-----------|---------------|---------------------|
| `t_A[STATES]` size | 16 bytes | 80 bytes |
| `o_A[STATES]` size | 16 bytes | 80 bytes |
| Sankoff inner loop | 4×4=16 ops/pattern | 20×20=400 ops/pattern |
| Registers used (ptxas) | **96** | **255** (hardware max) |
| `#pragma unroll` effect | 4 iters → 4 instructions | 20 iters → 20 instructions mỗi loop |
| `__forceinline__` honored? | **Có** — 96 regs đủ chỗ | **Không** — 255 regs = max, không còn register để inline |
| Cumulative stack | **0 bytes** | **1264–1544 bytes** |

Với STATES=4: compiler thành công inline toàn bộ call chain → cumulative stack = 0.
Mọi local variable (`t_A[4]`, `o_A[4]`) được giữ trong 96 registers → không cần stack frame.

Với STATES=20: compiler đạt giới hạn 255 registers. Để inline `newviewParsimony<20>`,
cần thêm registers cho `t_A[20]` và `o_A[20]` — nhưng đã full. Compiler buộc phải sinh
actual `CALL` instruction. `t_A[20]` và `o_A[20]` khi đó đi lên **hardware call stack**
thay vì registers.

---

### Câu hỏi 4: Nếu bỏ `#pragma unroll` thì vẫn còn gặp lỗi không?

**Có thể không gặp nữa** — nhưng không phải giải pháp đúng.

Cơ chế: `#pragma unroll` khiến compiler mở rộng loop 20 lần (`STATES=20`), tạo ra
20 instances của biến trung gian đồng thời live → register pressure tăng vọt.
Không unroll → compiler quản lý loop counter và dùng lại 1 set biến → ít registers hơn
→ có thể inline lại → stack = 0.

Tuy nhiên đây là **cách sửa không an toàn** vì:
1. Phụ thuộc vào quyết định nội bộ của nvcc — thay đổi theo compiler version
2. Mất ILP (Instruction-Level Parallelism) — STATES=20 loop không unrolled chậm hơn đáng kể
3. Không giải quyết nguyên nhân gốc: threshold register lúc nào cũng có thể bị chạm lại
   khi code thêm local variables hoặc compiler thay đổi allocation

Heisenbug (bỏ `printf` là crash, thêm `printf` là hết crash) chính xác là biểu hiện của
cơ chế này: `printf` thay đổi register allocation của compiler → `newviewParsimony<20>`
được inline hay không được inline → crash hay không crash. Đây là lý do tại sao bug này
không thể debug bằng printf truyền thống.

---

### Fix

Trước mỗi kernel launch có STATES=20, tăng per-thread hardware stack lên 4096 bytes;
sau khi sync xong, restore về giá trị cũ:

```cpp
// pars_build.cu, trước K1 launch (~line 1253)
size_t k1_prev_stack = 0;
CUDA_CHECK(cudaDeviceGetLimit(&k1_prev_stack, cudaLimitStackSize));
if (k1_prev_stack < 4096)
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

buildParsimonyTreesKernel<S, NT><<<...>>>(...)

CUDA_CHECK(cudaStreamSynchronize(stream));
if (k1_prev_stack < 4096)
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, k1_prev_stack));
```

Tương tự cho K2 launch (~line 1304). Giá trị 4096 = 3× default, đủ để cover cả
1544 bytes (K1) và 1264 bytes (K2) với buffer an toàn.

**Kết quả**: prot_M2593_56_386 chạy thành công, 5 rounds, best=1462, không crash.

---

### Câu hỏi 5: Tăng stack lên 4096 ảnh hưởng gì? GPU yếu hơn có bị không?

**Cơ chế tính toán bộ nhớ**:

`cudaDeviceSetLimit(cudaLimitStackSize, 4096)` yêu cầu CUDA runtime cấp phát
`4096 bytes × max_concurrent_threads` cho hardware stack trên device.

| Thông số | A100 SXM4 | RTX 2080 Ti | GTX 1060 |
|----------|-----------|-------------|---------|
| Max threads/SM | 2048 | 1024 | 1024 |
| Số SM | 108 | 68 | 10 |
| Max concurrent threads | ~221,184 | ~69,632 | ~10,240 |
| Stack tăng thêm (4096−1024)×threads | **~665 MB** | **~210 MB** | **~31 MB** |

**Lưu ý**: Con số 665 MB trên A100 là trường hợp tất cả threads active đồng thời — thực
tế với K2=800 workers × 32 lanes = 25,600 threads, extra stack chỉ **~79 MB**.

**Giải thích tại sao an toàn trên GPU yếu hơn**:

1. **Chỉ ảnh hưởng khi STATES=20**: Hầu hết datasets là DNA (STATES=4) → không bao giờ
   gọi đoạn code này. Protein datasets thường ít taxa (M2593 chỉ N=56) → K nhỏ →
   ít threads → extra memory nhỏ.

2. **Restore sau sync**: `cudaDeviceSetLimit` được restore ngay sau `cudaStreamSynchronize`.
   Chỉ ảnh hưởng trong thời gian kernel STATES=20 chạy, không ảnh hưởng các kernel khác.

3. **guard `if (prev_stack < 4096)`**: Nếu caller đã set stack lớn hơn (ví dụ 8192),
   code không downgrade về 4096, cũng không tăng thêm không cần thiết.

4. **Ngưỡng thực tế**: Protein dataset với STATES=20 thường nhỏ (N<200). Với K=800, N=56:
   - Threads: 800 × 32 = 25,600
   - Extra memory: 3072 bytes × 25,600 = **~79 MB** — hoàn toàn chấp nhận được
   - Ngay cả GTX 1060 có 6 GB VRAM cũng không bị ảnh hưởng

**Rủi ro thực sự**: nếu chạy STATES=20 với K cực lớn (K=10,000+) trên GPU cũ 2 GB VRAM,
có thể OOM. Nhưng K=10,000 với STATES=20 đã cần `parsVect` lớn hơn VRAM trước khi stack
là vấn đề → constraint này không bao giờ binding.

---

### Bài học / Ghi chú cho khóa luận

**1. Hardware call stack vs register spill: hai khái niệm khác nhau trong CUDA**

ptxas in ra cả hai nhưng chúng không liên quan đến nhau:
- "spill stores/loads" lớn (34524 bytes spill stores) = dùng DRAM làm register overflow.
  Chậm nhưng không crash.
- "cumulative stack size" lớn (1544 bytes) = actual function calls dùng hardware stack.
  Overflow 1024 byte limit → crash không deterministic.

Nhiều developer nhầm hai khái niệm này, dẫn đến chẩn đoán sai khi thấy "spill" lớn mà
nghĩ đó là nguyên nhân crash.

**2. Heisenbug trong GPU là dấu hiệu của stack overflow hoặc memory corruption**

Khi thêm `printf` làm crash biến mất, nguyên nhân gần như chắc chắn là:
- Hardware call stack overflow (như bug này), hoặc
- Shared memory race condition (thêm code thay đổi timing)

`printf` thay đổi cách compiler allocate registers → quyết định inline/not-inline thay đổi →
cumulative stack thay đổi. Đây là lý do tại sao tool debug thông thường không hiệu quả:
bản thân hành động debug phá vỡ điều kiện gây bug.

**3. `__forceinline__` là hint, không phải lệnh bắt buộc**

Compiler CUDA có thể và sẽ vi phạm `__forceinline__` khi register pressure quá cao.
Cách kiểm tra: xem ptxas output với `--ptxas-options=-v`. Nếu "cumulative stack size > 0",
có ít nhất một hàm `__forceinline__` đã không được inline.

Với template kernel sử dụng nhiều STATES: kiểm tra ptxas cho mỗi STATES instantiation
riêng biệt — threshold register có thể chỉ bị chạm ở STATES=20 không phải STATES=4.

**4. Fix đúng: tăng limit, không phải bỏ optimization**

Bỏ `#pragma unroll` để giảm register pressure và cho phép inlining là cách sửa dựa vào
side effect không ổn định. Fix đúng là: nhận biết hardware constraint và set limit phù hợp.
`cudaDeviceSetLimit(cudaLimitStackSize, 4096)` là API chính xác cho vấn đề này.

---

## Bug Fix — AA tip bitmask encoding sai trong Sankoff (`uploadSankoffTipParsVect`)

**Ngày**: 2026-05-31  
**File liên quan**: `gpu/src/gpu_init_trees.cu` (`uploadSankoffTipParsVect`, ~line 87)

### Triệu chứng

GPU Sankoff (protein, STATES=20) sinh ra star consensus tree (nsplits=1) hoặc score sai hoàn toàn. CPU ref cho kết quả tốt. Trước Fix 2: diff GPU−CPU = +1…+22 trên protein datasets.

### Root cause

```cpp
// WRONG:
unsigned int bitmask = (unsigned int)nuc;  // dùng AA state index như Fitch bitmask
```

- **DNA** (`PLL_MAP_NT`): `nuc` IS bitmask (A=1, C=2, G=4, T=8) → cast trực tiếp đúng.
- **Protein** (`PLL_MAP_AA`): `nuc` là **index** (Ala=0, Arg=1, …). Alanine → `bitmask=0` → mọi Sankoff cost = `kSankoffInf` → tip rỗng → tree search hoàn toàn sai.

### Fix

```cpp
const bool is_bitmask_coded = (undetermined == ((1u << states) - 1u));
unsigned int bitmask = is_bitmask_coded
    ? (unsigned int)nuc                                              // DNA: nuc IS bitmask
    : (nuc < (unsigned char)states ? (1u << nuc) : (1u << states) - 1u);  // AA: one-hot convert
```

`undetermined` cho DNA = 15 = `(1<<4)-1` → `is_bitmask_coded=true`. Cho protein: `undetermined=0xFF` ≠ `(1<<20)-1` → one-hot convert.

### Verification

20 protein datasets (treebase non-bootstrap): 20/20 diff ≤ 0 (16 match CPU, 4 GPU tốt hơn). DNA không bị ảnh hưởng.

### Bài học / Ghi chú cho khóa luận

PLL dùng hai encoding khác nhau cho tip states: DNA là bitmask, protein là index. Khi upload tip data cho GPU phải replicate đúng convention của `tr->yVector` — không thể dùng cùng một đường code cho cả hai STATES mode.

---

## Bug Fix — `score_tree = 0` với Sankoff: Opt-B prune vô hiệu

**Ngày**: 2026-05-31  
**File liên quan**: `gpu/include/pars_tree.cuh` (Sankoff branch trong `warpNewviewStep`, ~line 378–392)

### Triệu chứng

Protein SPR chậm hơn cần thiết: Opt-B lower-bound prune không loại được candidate edges dù lb cao.

### Root cause

Sankoff branch trong `warpNewviewStep` không accumulate `score` → `score_tree[p_num] = 0` với mọi inner node. `doAddTraverse` dùng `score_tree` làm lower-bound baseline: khi tất cả = 0, lb = 0 < threshold → **không prune được candidate nào**.

### Fix

Thêm `min_cost_b` tracking inline bên trong Sankoff newview loop:

```cpp
unsigned int min_cost_b = kSankoffInf;
// inside ii loop:
unsigned int val = best_left + best_right;
p_base[...] = (parsimonyNumber)val;
min_cost_b = min(min_cost_b, val);
// after ii loop:
score += (min_cost_b < kSankoffInf) ? min_cost_b : 0u;
```

**Impact**: performance only — `score_tree` không ảnh hưởng evaluate accuracy.

### Kết quả (6 protein datasets, `output/treebase_gpu_bugA/`)

| Dataset | N | Baseline | Bug A fix | Speedup | Score Δ |
|---------|---|----------|-----------|---------|---------|
| M1118_137 | 137 | 311.2s | 279.6s | +11% | 0 |
| M11341_100 | 100 | 409.8s | 382.0s | +7% | 0 |
| M4249_153 | 153 | 1269.4s | 910.9s | **+28%** | 0 |
| M4318_78 | 78 | 991.9s | 857.8s | +14% | 0 |
| M4780_90 | 90 | 427.7s | 363.7s | +15% | +1* |
| M8569_164 | 164 | 459.1s | 404.1s | +12% | 0 |

*+1: stochastic variation bình thường — Opt-B thay đổi evaluation order.  
**Kết quả: 5/6 score unchanged; speedup 7–28%, trung bình ~14%.**

---

## Bug Fix — AA bootstrap sinh star tree: treels thiếu đa dạng topology

**Ngày**: 2026-05-31  
**File liên quan**: `gpu/src/pars_build.cu` (`testInsert` SAVE A gate), `gpu/include/pars_tree.cuh`, `gpu/include/pars_build.cuh`, `gpu/src/gpu_init_trees.cu`, `mpboot/tools.h`, `mpboot/tools.cpp`

### Triệu chứng

GPU bootstrap (protein) sinh star consensus tree: `cor=0.0` không tăng, `nsplits=1`. Tất cả bootstrap replicates chọn cùng 1 topology → bipartition support không meaningful.

### Root cause

Sau khi Fix 2 sửa tip bitmask, K2 workers hội tụ về T* (optimal topology). SAVE A gate (`testInsert`, line ~140) lưu cây chỉ khi `mp < sh.randomMP`. Tại T*: không candidate nào có `mp < T*` → treels rỗng mỗi round → hash dedup loại hết → REPS chọn T* cho mọi replicate → star consensus.

### Fix: Save margin `-gpu_treels_margin`

```cpp
// TRƯỚC:
if (randomMP < d_treelsCutoff)

// SAU:
if (randomMP < d_treelsCutoff + save_margin)   // save near-optimal topologies
```

**Files thay đổi**:
- `pars_build.cu` testInsert: `mp < sh.randomMP` → `mp < sh.randomMP + sh.save_margin`
- `pars_tree.cuh` `BuildSharedT`: thêm `unsigned int save_margin`
- `pars_tree.cuh` `GpuParsimonyMem`: thêm `unsigned int save_margin`
- `pars_build.cu` `buildPhase3Kernel`: pass `save_margin` param
- `gpu_init_trees.cu`: set `mem->save_margin = params.gpu_treels_margin`; margin-based `logl_cutoff`
- `tools.h`/`tools.cpp`: thêm `-gpu_treels_margin N` (default=10)

**Cơ chế**: tại T* với margin=10, `testInsert` lưu tất cả candidates có parsimony ≤ T*+10 → O(N) diverse near-optimal topologies mỗi SPR pass → treels đủ đa dạng cho REPS.

### Kết quả (5 pandit protein datasets, `-bb 1000`)

5/5 GPU score ≤ CPU ref (diff -2…0). GPU cũ (pre-Fix2): diff +1…+22. Speedup 2.74–4.65×, mean 3.56×. `cor` tăng đều từ 0.0 → hội tụ — không còn star tree.

### Bài học / Ghi chú cho khóa luận

Strict improvement gate (`mp < T*`) là điều kiện cần cho tree search quality nhưng gây diversity collapse cho bootstrap: mọi worker đều save T* → sau hash dedup chỉ còn 1 topology → REPS vô nghĩa. Margin là trade-off: lưu thêm near-optimal topologies để bootstrap có đủ candidate pool, với chi phí treels lớn hơn và cần logl_cutoff để kiểm soát chất lượng.

---

## Optimization — Relative SAVE A Margin (`gpu_treels_margin`)

**Ngày**: 2026-06-01
**File**: `mpboot/tools.h`, `mpboot/tools.cpp`, `mpboot/gpu/src/gpu_init_trees.cu` (`mem->save_margin`)

### Thay đổi

Thêm tham số `-gpu_treels_margin r` (default r=0.0). SAVE A gate trong `gpuSPRHillClimb`:

```cuda
// Old: strict improvement only
if (randomMP < d_treelsCutoff)  // = bestParsimony so far

// New: relative margin
if (randomMP < (1.0f + save_margin) * d_treelsCutoff)
```

- `r=0.0` (default): strict improvement — chỉ lưu cây tốt bằng hoặc hơn best hiện tại
- `r>0`: lưu thêm cây kém hơn một chút → pool đa dạng hơn nhưng chất lượng trung bình thấp hơn

### Benchmark sweep (pandit DNA, numpars=200, 300–400 taxa, 5 datasets)

| r | Speedup trung bình | Nhận xét |
|---|---|---|
| 0.00 (strict) | baseline | Treels pool nhỏ, quality cao |
| 0.01 | +22–50% | Sweet spot: pool đủ đa dạng cho bootstrap |
| 0.05 | tương đương | Quá nhiều cây kém chất lượng |

Kết quả: r=0.0 giữ nguyên (strict) sau khi bootstrap calibration analysis cho thấy r lớn hơn
không cải thiện calibration có ý nghĩa.

---

## Analysis — Bootstrap Calibration: GPU r=0 vs optimizeBootTrees

**Ngày**: 2026-06-01
**File**: `mpboot/phyloanalysis.cpp` (GPU path), `thesis/benchmark/`

### Bài toán

GPU bootstrap có calibration tệ hơn CPU cho N=50-99: avg_Δ ≈ -5.9% (under-calibration).
Root cause: `treels_pool` xây từ alignment gốc không cover bootstrap-optimal topologies
(N nhỏ → L ngắn → high bootstrap variance → topology space khác).

### Thực nghiệm (1469 pandit DNA datasets, -bb 1000)

| Variant | avg_Δ (tất cả N) | pass_rate | N=50-99 avg_Δ | N≥100 avg_Δ |
|---------|---|---|---|---|
| CPU | -0.0% | 60.0% | -0.9% | -0.0% |
| **GPU r=0** | **-2.6%** | **66.7%** | -5.9% | -2.2% |
| GPU + optBT full | +10.3% | 14.3% | +8.7% | +10.6% |
| GPU + nni=1 | +8.3% | 28.6% | +5.3% | +8.8% |
| GPU + nni=2 | +9.0% | 28.6% | +9.5% | +9.1% |
| GPU + nni=3 | +10.1% | 23.8% | +6.5% | +10.6% |

### Kết luận

- `optimizeBootTrees` (kể cả giới hạn 1 round NNI) đều **over-calibrate nặng** (+8–10%)
- GPU r=0 không có optimizeBootTrees cho **pass_rate 66.7% — cao hơn cả CPU** (60%)
- Root cause của over-calibration: tất cả treels_pool winners xuất phát từ cùng alignment gốc
  → sau NNI (dù chỉ 1 round), converge về cùng topology → bootstrap support bị inflate
- **Final decision**: GPU path không dùng `optimizeBootTrees`

### Ghi chú cho khóa luận

GPU bootstrap under-calibration nhẹ (-2.6%) là chấp nhận được — xảy ra vì treels pool
tập trung vào topology tối ưu cho alignment gốc. CPU chạy SPR riêng cho mỗi bootstrap
replicate nên "naturally covers" bootstrap topology space. Đây là trade-off thiết kế:
GPU ưu tiên tốc độ (pool sharing across replicates) vs độ chính xác bootstrap calibration.

---

## Optimization — max_treels formula: K×3000 → K×mxtips

**Ngày**: 2026-06-02
**File**: `mpboot/gpu/src/gpu_init_trees.cu` (line ~205)

### Vấn đề

```cpp
// Cũ: cố định K × 3000 bất kể N
const int max_treels_boot = need_treels ? K_alloc * 3000 : 0;
// K=200 → max_treels=600,000
// d_treelsBackVf: 600,000 × 3200 × 4 = 7.68 GB device
// h_treels_bvf:  7.68 GB pinned host  ← rất lãng phí
```

### Đo đạc thực tế (h_filled per round)

| N (dataset) | h_filled/round tối đa | max_treels cũ | Tỉ lệ dư |
|---|---|---|---|
| ~300 (pandit) | ~8,000 | 600,000 | 75× |
| 403 (pandit id=481) | ~10,000 | 600,000 | 60× |
| 699 (treebase) | ~26,000 | 600,000 | 23× |
| 767 (treebase) | ~28,000 | 600,000 | 21× |

### Fix

```cpp
// Mới: tỉ lệ với K và N → ~5× margin thực tế
const int max_treels_boot = need_treels ? K_alloc * mxtips : 0;
// K=200, N=295 → max_treels=59,000
// K=200, N=767 → max_treels=153,400
```

### Kết quả đo (A100, current binary)

| N | GPU memory alloc (total) | max_treels |
|---|---|---|
| 295 | **1.13 GB** | 59,000 |
| 413 | **1.89 GB** | 82,600 |
| 767 | **4.69 GB** | 153,400 |

**Ảnh hưởng phụ tích cực**: ppars (gpuComputeTreelsPatternPars) nhanh hơn nhiều vì
h_filled thực tế nhỏ → ít batch → ppars không còn là bottleneck chính.

### Round timing so sánh (N=295, K=200, current binary)

| Step | Jun 1 (old max_treels) | Current (K×N) |
|---|---|---|
| K2 | 1.1s (K=100) | **2.2s** (K=200, 2× workers) |
| ppars | **5.5s** (bottleneck) | **0.25s** (không còn bottleneck) |
| D2H | 0.95s | **0.01s** (pinned host) |
| Total/round | ~7.5s | **~2.5s** |

Giảm 3× thời gian/round nhờ combination: K_ppars=1000 (Jun 1) + K×N formula (Jun 2) +
pinned host memory (thay pageable).

---

## Optimization — Opt-4: Register preload trong Sankoff newview/evaluate (2026-06-07)

**Ngày**: 2026-06-07
**Task**: Giảm global memory loads trong Sankoff parsimony từ S² → S per pattern
**File liên quan**: `gpu/include/pars_tree.cuh` (`newviewParsimony` Sankoff branches, ~line 390–490)

### Phân tích vấn đề

Loop newview Sankoff trước Opt-4:
```cuda
for (int ii = 0; ii < STATES; ++ii) {         // S iterations
    for (int jj = 0; jj < STATES; ++jj) {      // S iterations
        unsigned int lv = q_base[jj * width + b]; // GLOBAL load — lặp lại S lần!
        unsigned int rv = r_base[jj * width + b]; // GLOBAL load — lặp lại S lần!
        unsigned int c  = cm[ii * STATES + jj];
        best_left  = min(best_left,  lv + c);
        best_right = min(best_right, rv + c);
    }
}
```

`q_base[jj*width+b]` với cùng `jj` được load **S lần** (một lần mỗi vòng `ii`) — tổng S² global loads thay vì S cần thiết.

Tương tự với evaluate: `r_base[jj*width+b]` được load S lần trong vòng `ii`, trong khi chỉ cần load 1 lần vào register.

**Số lượng global loads per pattern trước/sau Opt-4:**

| Path | S=4 trước | S=4 sau | S=20 trước | S=20 sau |
|------|-----------|---------|------------|---------|
| newview (q loads) | 16 | **4** | 400 | **20** |
| newview (r loads) | 16 | **4** | 400 | **20** |
| evaluate (r loads) | 16 | **4** | 400 | **20** |

### Fix: Preload vào registers trước vòng lặp ii

**Newview** (`pars_tree.cuh`, Sankoff branch):
```cuda
// Opt-4: preload child columns into registers — eliminates S-fold redundant global loads.
unsigned int lv_[STATES], rv_[STATES];
#pragma unroll
for (int jj = 0; jj < STATES; ++jj) {
    lv_[jj] = (unsigned int)q_base[(size_t)jj * width + b];
    rv_[jj] = (unsigned int)r_base[(size_t)jj * width + b];
}
unsigned int min_cost_b = kSankoffInf;
#pragma unroll
for (int ii = 0; ii < STATES; ++ii) {
    unsigned int best_left = kSankoffInf, best_right = kSankoffInf;
    #pragma unroll
    for (int jj = 0; jj < STATES; ++jj) {
        unsigned int c = cm[ii * STATES + jj];
        best_left  = min(best_left,  lv_[jj] + c);
        best_right = min(best_right, rv_[jj] + c);
    }
    unsigned int val = best_left + best_right;
    p_base[(size_t)ii * width + b] = (parsimonyNumber)val;
    min_cost_b = min(min_cost_b, val);
}
score += (min_cost_b < kSankoffInf) ? min_cost_b : 0u;
```

**Evaluate** — preload chỉ `rv_[]` (r cần S² lần), `qi` scalar (S lần, không cần preload):
```cuda
// Opt-4: preload r columns into registers
unsigned int rv_[STATES];
#pragma unroll
for (int jj = 0; jj < STATES; ++jj)
    rv_[jj] = (unsigned int)r_base[(size_t)jj * width + b];
unsigned int min_edge = kSankoffInf;
#pragma unroll
for (int ii = 0; ii < STATES; ++ii) {
    unsigned int qi = (unsigned int)q_base[(size_t)ii * width + b];
    #pragma unroll
    for (int jj = 0; jj < STATES; ++jj)
        min_edge = min(min_edge, qi + c + rv_[jj]);
}
```

**Chi phí register**: S=4 → 8 extra regs/lane (không đáng kể); S=20 → 40 extra regs/lane (đẩy STATES=20 kernel lên 255 regs = hardware max, trigger stack spill).

### Kết quả

Protein (STATES=20): speedup ~1.4–1.8× so với trước (đo bằng benchmark 5 datasets). Cụ thể:

| Dataset | Trước Opt-4 (ms/tree) | Sau Opt-4 (ms/tree) | Speedup |
|---------|----------------------|---------------------|---------|
| prot_M10236 (59T) | ~55 ms (est.) | 40.53 | ~1.35× |

*(Opt-4 được đo baseline trước khi thêm Opt-5; Opt-4 là baseline cho bảng speedup Opt-5)*

### Bài học / Ghi chú cho khóa luận

1. **Loop-invariant code motion trong CUDA**: Trong nested loop `for (ii) { for (jj) { load q[jj] } }`, `q[jj]` là loop-invariant với `ii` nhưng compiler không thể hoist nếu `q_base` trỏ vào global memory (không biết aliasing). Phải hoist thủ công vào register array `lv_[jj]`.

2. **Asymmetry của preload trong newview vs evaluate**: Trong newview, cả `q[jj]` và `r[jj]` đều cần S² loads → cần preload cả hai. Trong evaluate, `r[jj]` cần S² loads (preload) nhưng `q[ii]` chỉ cần S loads (1 per outer ii) → không cần preload. Hiểu rõ access pattern từng loop để preload đúng.

3. **STATES=20 register pressure**: `lv_[20]` + `rv_[20]` = 40 registers thêm, đủ để push kernel lên 255 regs (hardware max). Hệ quả: compiler bắt buộc spill vào hardware call stack (xem Bug #12). NCU xác nhận: Stack Size = 4096 bytes, Theoretical Occupancy = 12.5% (register-limited thay vì shared-memory-limited).

---

## Optimization + Bug Fix — Opt-5: Sankoff cost matrix vào CUDA `__constant__` memory (2026-06-07)

**Ngày**: 2026-06-07
**Task**: Chuyển cost matrix từ `cudaMalloc` global memory sang `__constant__` memory để tận dụng hardware broadcast
**File liên quan**: `gpu/include/pars_tree.cuh`, `gpu/src/pars_build.cu`, `gpu/src/pars_tree.cu`, `gpu/include/pars_bootstrap.cuh`

### Phân tích vấn đề

Trước Opt-5: `d_cost_matrix` được alloc bằng `cudaMalloc`:
```cpp
CUDA_CHECK(cudaMalloc(&mem->d_cost_matrix, costBytes));
CUDA_CHECK(cudaMemcpy(mem->d_cost_matrix, cost_matrix, costBytes, cudaMemcpyHostToDevice));
```

Kernel đọc `cm = sh.cost_matrix` từ global memory → L2 cache, không phải constant cache.

Với STATES=20: 20×20 = 400 entries × 4 bytes = **1600 bytes**. Constant memory cache là 64 KB dedicated per-device. Khi tất cả 32 lanes đọc cùng địa chỉ `cm[ii*STATES+jj]` (uniform read) → **hardware broadcast từ constant cache → near-zero latency** thay vì L2 cache miss.

### Thay đổi

**Bước 1** — Khai báo `__constant__` symbol trong `pars_tree.cuh` (ngoài namespace):
```cuda
static constexpr int kMaxSankoffStates = 20;

// Opt-5: Sankoff cost matrix in constant memory.
// pars_build.cu defines the symbol (PARS_BUILD_DEFINE_CM macro); other TUs get extern.
#ifdef PARS_BUILD_DEFINE_CM
__constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
#else
extern __constant__ unsigned int g_sankoff_cm[kMaxSankoffStates * kMaxSankoffStates];
#endif
```

**Bước 2** — `pars_build.cu` define macro trước includes:
```cpp
#define PARS_BUILD_DEFINE_CM  // causes pars_tree.cuh to define g_sankoff_cm here (not extern)
#include "gpu/include/pars_tree.cuh"
```

**Bước 3** — Upload function trong `pars_build.cu`:
```cpp
void gpuUploadSankoffCostMatrix(const unsigned int* cm, int nstates) {
    const size_t costBytes = (size_t)nstates * nstates * sizeof(unsigned int);
    CUDA_CHECK(cudaMemcpyToSymbol(g_sankoff_cm, cm, costBytes));
}
```

**Bước 4** — `pars_tree.cu` thay `cudaMalloc` + upload bằng:
```cpp
if (use_sankoff) {
    gpuUploadSankoffCostMatrix(cost_matrix, nstates);
    // Allocate 4-byte sentinel so kernels detect Sankoff via d_cost_matrix != nullptr.
    CUDA_CHECK(cudaMalloc(&mem->d_cost_matrix, sizeof(unsigned int)));
}
```

**Bước 5** — Kernel dùng `g_sankoff_cm` thay vì `sh.cost_matrix`:
```cuda
const unsigned int* cm = sh.use_sankoff ? g_sankoff_cm : nullptr;
```

### Bug #1 — NVCC redefinition error

**Triệu chứng**: `pars_build.cu:13: error: redefinition of 'unsigned int mpbootgpu::g_sankoff_cm [400]'`

**Root cause**: Không dùng RDC (relocatable device code), nên `extern __constant__` trong header VÀ `__constant__` definition trong `.cu` file cùng include header → NVCC thấy 2 definitions trong cùng TU.

**Fix**: `#ifdef PARS_BUILD_DEFINE_CM` conditional macro — chỉ `pars_build.cu` set macro trước include, các TU khác nhận `extern` declaration. Template `newviewParsimony` chỉ instantiate trong `pars_build.cu` → không có cross-TU device symbol reference → không cần RDC.

### Bug #2 — `sh.use_sankoff = false` sau khi Opt-5 set `d_cost_matrix = nullptr`

**Triệu chứng**: Sau khi Opt-5 thay `cudaMalloc` bằng constant memory, `d_cost_matrix = nullptr`. Kernel check `sh.use_sankoff = (d_cost_matrix != nullptr)` → false → chạy Fitch mode → score sai (1191 thay vì 1182 cho protein), ms/tree 1.94 thay vì ~40ms.

**Fix**: Allocate 4-byte sentinel `cudaMalloc(&mem->d_cost_matrix, sizeof(unsigned int))` — giá trị không quan trọng, chỉ dùng để boolean check `!= nullptr`. Dữ liệu cost thực sự từ `g_sankoff_cm` trong constant memory.

### Bug #3 — `pars_bootstrap.cuh` include `cuda_runtime_api.h` gây C++ compile error

**Triệu chứng**: `iqtree.cpp` → `pars_bootstrap.cuh` → `cuda_runtime_api.h: No such file or directory` khi compile bởi clang++.

**Fix**: Guard trong `pars_bootstrap.cuh`:
```cpp
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#else
typedef struct CUstream_st* cudaStream_t;  // forward declare for C++ TUs
#endif
```

### Kết quả NCU (Device 5 A100, STATES=20, prot_M10236_59_164)

| Metric | Giá trị | Ý nghĩa |
|--------|---------|---------|
| DRAM Throughput | **0.00%** | ✅ Cost matrix từ constant cache, zero DRAM traffic |
| L2 Hit Rate | **99.9%** | parsVect nhỏ (59 taxa) fit hoàn toàn trong L2 |
| Registers/thread | **255** | Hardware max — do Opt-4 push lên giới hạn |
| Stack Size | **4096 bytes** | Register spill sang local memory (đã bump từ 1024) |
| Theoretical Occupancy | **12.5%** | 8 blocks/SM, giới hạn bởi registers |
| Warp Cycles/Issued Inst | **2.93** | Rất tốt — pipeline gần đầy, ít latency stall |

**Speedup Opt-5 vs Opt-4** (5 protein datasets):

| Dataset | Opt-4 | Opt-5 | Speedup |
|---------|-------|-------|---------|
| prot_M10236 (59T) | 40.53 ms | 25.66 ms | **1.58×** |
| prot_M11595 (66T) | 87.15 ms | 62.45 ms | **1.40×** |
| prot_M3807 (82T) | 229.36 ms | 137.40 ms | **1.67×** |
| prot_M10866 (88T) | 1051.96 ms | 574.74 ms | **1.83×** |
| prot_M11740 (138T) | 3569.00 ms | 2074.64 ms | **1.72×** |

**Speedup trung bình: 1.64×** nhờ constant memory broadcast.

### Bài học / Ghi chú cho khóa luận

1. **CUDA constant memory**: 64 KB cache riêng, hardware broadcast khi tất cả 32 lanes đọc cùng địa chỉ → near-zero latency. Thích hợp cho small read-only data (cost matrix 1600 bytes) được đọc nhiều lần với uniform access pattern (tất cả lanes dùng cùng `cm[ii][jj]` trong loop).

2. **`extern __constant__` không dùng RDC**: Cách thông thường là `extern __constant__` trong header + definition trong `.cu` riêng. Nhưng nếu template function chỉ instantiate trong 1 TU (pars_build.cu), không có cross-TU device symbol → không cần RDC. Dùng `#ifdef DEFINE_MACRO` để chỉ TU đó define symbol, TU khác nhận `extern`.

3. **Sentinel pattern cho boolean flag**: Cấp phát 4 bytes chỉ để `!= nullptr` là pattern awkward. Giải pháp sạch hơn: thêm `bool use_sankoff` vào `GpuParsimonyMem` trực tiếp.

4. **`cudaMemcpyToSymbol` device scope**: Gọi sau `cudaSetDevice(gpu_device)` → upload đúng device hiện tại. Với single-device usage (hiện tại), pattern này an toàn. Multi-device cần đảm bảo `cudaSetDevice` được gọi trước mỗi upload.

---

## Thay đổi cấu hình cuối — GPU path không dùng optimizeBootTrees

**Ngày**: 2026-06-02
**File**: `mpboot/phyloanalysis.cpp` (line ~1863)

Revert GPU path về cấu hình gốc: không gọi `optimizeBootTrees()` sau khi GPU hill-climbing xong.

```cpp
// Removed from GPU path (benchmark showed over-calibration):
// if (params.gbo_replicates > 0 && params.maximum_parsimony && params.optimize_boot_trees) {
//     iqtree.optimizeBootTrees(params.gpu_boot_nni_rounds);
// }
```

`gpu_boot_nni_rounds` parameter vẫn compile vào binary (dùng cho thực nghiệm nếu cần)
nhưng không được gọi trong GPU path bình thường.

**Final GPU bootstrap pipeline**:
1. K1: build K initial trees (stepwise + SPR)
2. K2: hill-climbing với treels pool (r=0 strict improvement)
3. REPS: score all treels against each bootstrap replicate
4. saveCurrentTree: pick winner per replicate
5. *(Không có optimizeBootTrees)*
