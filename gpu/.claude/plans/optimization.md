# GPU Parsimony Optimizations — Tổng quan (2026-05-13)
<!-- Lưu tại: /raid/home/loinguyen/workspace/mpboot-gpu/mpboot/gpu/.claude/plans/optimization.md -->

## Đã hoàn thành

| Opt | Tên | Kết quả |
|-----|-----|---------|
| testInsert pre-refresh | Loại bỏ Fitch step thừa | −24% kernel time |
| Opt-D | GpuTopology struct shrink | −8.7% ms/tree |
| Opt-H | Early stopping Phase 3 (gpu_stop) | −32.4% ms/tree |
| Opt-G | Selective Phase 3 atomicMin margin | −22–50% tùy dataset |
| Opt-G2 | Selective Phase 3 two-kernel exact top-X% | −44% trên N=295 |
| Opt-I | NNI strength configurable (gpu_nni_strength) | Baseline alignment |

---

## Còn lại — theo độ khó và rủi ro

### 🟢 Dễ, rủi ro thấp

#### Opt-B: Subtree prune trong SPR DFS
**Ý tưởng**: Trong `doAddTraverse` (`pars_build.cu`), trước khi DFS vào subtree q: nếu `score_tree[q_num] + score_tree[back_of_q]` ≥ `sh.randomMP` → không thể improve → skip toàn bộ subtree.

**Lý thuyết**: Lower bound cho parsimony của bất kỳ insertion nào vào subtree q.  
**File**: `pars_build.cu` — `doAddTraverse`, ~5 dòng.  
**Rủi ro thấp**: Nếu prune rate thấp (như Opt-A với 0.2%), không có hại.  
**Effort**: 1-2 giờ.

---

### 🟡 Trung bình, rủi ro vừa

#### Opt-M: Template specialization + loop unrolling cho `newviewParsimony`

**Vấn đề hiện tại** (`pars_tree.cuh:304`):
```cpp
parsimonyNumber t_A[kMaxStates], o_A[kMaxStates];  // kMaxStates = 32 HARDCODED
for (int s = 0; s < states; ++s) { ... }            // states là runtime variable
```

- `t_A[32]` và `o_A[32]` luôn chiếm **64 register slots** per lane, bất kể actual states
- Với DNA (states=4): **28/32 slots bị waste**
- Với Protein (states=20): **12/32 slots bị waste**  
- Inner loop `for s in [0, states)` với `states` là runtime → compiler không unroll được

**`newviewParsimony` được gọi rất thường xuyên**: mỗi `testInsert` (step 2), mỗi `createTiAndEvaluateParsimony`, mỗi `createTiAndNewviewParsimony` → đây là **hot path** quan trọng nhất.

**Fix**: Template specialization với `states` là compile-time constant:

```cpp
template<int STATES>
__device__ __forceinline__ void warpFitchStep(
    parsimonyNumber* p_base, const parsimonyNumber* q_base,
    const parsimonyNumber* r_base, int b, unsigned int& score
) {
    parsimonyNumber t_N = 0;
    parsimonyNumber t_A[STATES], o_A[STATES];  // STATES known at compile time → exact registers

    #pragma unroll
    for (int s = 0; s < STATES; ++s) { ... }  // compiler fully unrolls

    #pragma unroll
    for (int s = 0; s < STATES; ++s) { ... }
    score += __popc(~t_N);
}
```

Gọi từ kernel launch site: detect states at host, instantiate template:
```cpp
if (states == 4)       buildParsimonyTreesKernel<4><<<...>>>(...)
else if (states == 20) buildParsimonyTreesKernel<20><<<...>>>(...)
```

**Với DNA (states=4)**:
- t_A[4], o_A[4] → chỉ 8 register slots (vs 64 hiện tại) → **−56 registers per lane**
- Loop unrolled → ILP tốt hơn, compiler có thể pipeline load/compute
- Ước tính: **10–20% speedup** trên DNA

**Với Protein (states=20)**:
- t_A[20], o_A[20] → 40 register slots (vs 64 hiện tại) → **−24 registers per lane**
- `warpNewviewStep` nặng hơn DNA (5× nhiều operations) → register pressure là bottleneck chính
- Ít lanes hơn per SM nếu registers bị spill → với kMaxStates=32, protein đã bị ảnh hưởng
- Với STATES=20: ít registers hơn → **có thể giữ occupancy cao hơn** → speedup protein

**File**: `pars_tree.cuh` (template thêm vào `newviewParsimony`), `pars_build.cu` (kernel template), `pars_build.cuh`, `gpu_init_trees.cu` (dispatch dựa theo states).

**Effort**: 1-2 ngày.

---

#### Opt-N: Thread coarsening — xử lý nhiều parsimony blocks per lane

**Ý tưởng**: Mỗi lane hiện xử lý `b = lane, lane+32, lane+64, ...` (stride kWarpSize). Tăng lên 2 blocks/lane giúp giảm `__syncwarp()` overhead và tăng arithmetic intensity.

**Rủi ro**: Tăng register pressure, cần benchmark cẩn thận.

---

#### Opt-O: Incremental parsimony update sau SPR move

**Ý tưởng**: Sau `applyMove`, chỉ update path từ insert point lên đến LCA của old/new subtrees thay vì full lazy refresh.

**File**: `pars_build.cu` — `applyMove`.  
**Rủi ro**: Cần chứng minh correctness.

---

### 🔴 Phức tạp, rủi ro cao

#### Opt-J: parsVect caching vào shared memory *(bị bác)*
Giảm occupancy, phức tạp, rủi ro chưa rõ benefit vượt chi phí.

#### Opt-E: Sub-warp parallel candidate evaluation *(future)*
4× speedup lý thuyết nhưng redesign hoàn toàn kernel. Effort 1-2 tuần.

---

## Thứ tự ưu tiên đề xuất

1. **Opt-K** (gpu_stop 2→4): 5 phút, confirmed
2. **Opt-B** (subtree prune): 2 giờ, safe
3. **Opt-M** (template specialization): 1-2 ngày, **highest expected benefit** (~10–20%)
4. Profile Nsight sau Opt-M để xác nhận bottleneck tiếp
5. **Opt-E** khi tất cả opt nhỏ xong
