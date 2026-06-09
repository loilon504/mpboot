# Plan: Sorted Dynamic Pool on GPU

## Context

Pool hiện tại (`buildPhase3Kernel`) có 2 vấn đề:
1. **Không sorted**: Warp ghi vào `my_slot = k % pool_size` (fixed) — sau khi K2 chạy,
   pool mất thứ tự. Step 1 random draw từ bất kỳ slot nào → không ưu tiên tree tốt hơn.
2. **Worst-scan O(pool_size)**: Mỗi outer iter, lane 0 scan toàn bộ pool để tìm worst
   (lines 718–721 trong `pars_build.cu`).

Mục tiêu: pool luôn sorted ascending (best first), worst-scan → O(1), dynamic accessible range.

---

## Review notes (2026-05-17)

1. `d_poolOrder` dùng `uint8_t` vì pool_size ≤ 200 (fits in 0..255)
2. Pre-check `sh.randomMP < worst` trước khi `atomicCAS` lock để tránh contention
3. `base_pool` default = **10** (không phải `pool_size / 2`)
4. Shift order array: chọn hướng ngắn hơn (left hoặc right)

---

## Giải pháp: Index Array (Indirection Layer)

Tách biệt **thứ tự logic** khỏi **vị trí vật lý**:

```
d_poolOrder[pool_size]             ← uint8_t[], SORTED indices into pool storage
d_poolScores[pool_size]            ← unsigned int[], score tại physical slot (unsorted)
d_poolBackVf[pool_size*kMaxVFaces] ← int[], topology tại physical slot (layout unchanged)
d_poolFilled                       ← int, số slot đã dùng (0..pool_size)
d_poolLock                         ← int, global spinlock (0=free, 1=held)
```

**Invariant**: `d_poolScores[d_poolOrder[0]] ≤ d_poolScores[d_poolOrder[1]] ≤ ...`

Shift chỉ thao tác trên `d_poolOrder` (**≤ 200 bytes** với uint8_t), KHÔNG shift topology.

---

## Thuật toán

### Step 1 — Slot selection với dynamic accessible range

```cuda
int filled     = *pool_filled;
int accessible = min(base_pool + outer / expand_step, filled);
accessible     = max(accessible, 1);

int order_idx = (int)(((unsigned)k * 2654435761u + (unsigned)outer * 1013904223u)
                >> 8) % (unsigned)accessible;
int phys_slot = (int)(uint8_t)pool_order[order_idx];   // indirection

const int* src = pool_back_vf + (size_t)phys_slot * kMaxVFaces;
for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize) {
    topo->back_vf[vf] = src[vf];
    topo->xpars[vf]   = 0;
}
__syncwarp();
sh.randomMP = pool_scores[phys_slot];
```

### Step 3 — Pool update: pre-check → spinlock → sorted insert

```cuda
if (lane == 0)
{
    // ── Pre-check without lock (speculative, may be stale) ──────────────
    // Avoids lock contention when clearly can't improve pool.
    int filled_spec        = *pool_filled;
    unsigned int worst_spec = (filled_spec < pool_size) ? 0xFFFFFFFFu
                             : pool_scores[(int)(uint8_t)pool_order[filled_spec - 1]];
    sh.bcast[4] = (sh.randomMP < worst_spec) ? 0 : -1;
}
__syncwarp();

if (sh.bcast[4] >= 0 && lane == 0)
{
    // ── Acquire spinlock ──────────────────────────────────────────────
    while (atomicCAS(pool_lock, 0, 1) != 0) {}

    int filled           = *pool_filled;
    unsigned int worst   = (filled < pool_size) ? 0xFFFFFFFFu
                         : pool_scores[(int)(uint8_t)pool_order[filled - 1]]; // O(1)

    if (sh.randomMP < worst)
    {
        // Binary search insert position
        int lo = 0, hi = filled;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (pool_scores[(int)(uint8_t)pool_order[mid]] <= sh.randomMP) lo = mid + 1;
            else hi = mid;
        }
        int pos = lo;

        int phys, actual_pos;

        if (filled < pool_size)
        {
            // Pool not full — always shift right (no eviction needed)
            phys       = filled++;
            actual_pos = pos;
            *pool_filled = filled;
            for (int i = filled - 1; i > pos; i--)
                pool_order[i] = pool_order[i - 1];
        }
        else
        {
            // Pool full — evict from shorter side
            int cost_right = pool_size - 1 - pos;   // evict worst (pool_order[pool_size-1])
            int cost_left  = pos;                    // evict best  (pool_order[0])

            if (pos == 0 || cost_right <= cost_left)
            {
                // Shift right: evict worst
                phys       = (int)(uint8_t)pool_order[pool_size - 1];
                actual_pos = pos;
                for (int i = pool_size - 1; i > pos; i--)
                    pool_order[i] = pool_order[i - 1];
            }
            else
            {
                // Shift left: evict best (cheaper when pos > pool_size/2)
                // Acceptable trade-off: new tree is near-best anyway
                phys       = (int)(uint8_t)pool_order[0];
                actual_pos = pos - 1;
                for (int i = 0; i < pos - 1; i++)
                    pool_order[i] = pool_order[i + 1];
            }
        }

        pool_order[actual_pos] = (uint8_t)phys;
        pool_scores[phys]      = sh.randomMP;
        sh.bcast[4]            = phys;   // signal lanes: copy topology here
    }
    else
    {
        sh.bcast[4] = -1;
    }

    atomicExch(pool_lock, 0);   // release
}
__syncwarp();

// All 32 lanes copy topology OUTSIDE lock (slot already claimed inside lock)
if (sh.bcast[4] >= 0)
{
    int* dst = pool_back_vf + (size_t)sh.bcast[4] * kMaxVFaces;
    for (int vf = lane; vf < topo->num_vfaces; vf += kWarpSize)
        dst[vf] = topo->best_back_vf[vf];
    __syncwarp();
}
```

**Lock safety**: topology write xảy ra NGOÀI lock; `phys` slot đã "claimed" bên trong lock
→ không warp nào khác được assign cùng slot đó.

**Shift left trade-off**: khi `pos > pool_size/2`, evict `pool_order[0]` (best, score
marginally better than new tree). Chấp nhận được vì new tree near-best và giảm 50% shift ops.

---

## Tham số mới

| Param | Ý nghĩa | Default | CLI |
|-------|---------|---------|-----|
| `base_pool` | Accessible slots lúc outer=0 | **10** | `-gpu_base_pool N` |
| `expand_step` | Cứ mỗi N outer iter thì +1 | 10 | `-gpu_pool_expand N` |

`expand_step=0` → `accessible = filled` mọi lúc (disable dynamic range).

---

## Bonus: Worst-scan O(pool_size) → O(1)

Code hiện tại tại `pars_build.cu:718–721`:
```cuda
for (int i = 0; i < pool_size; i++)
    if (pool_scores[i] > worst_s) { worst_s = pool_scores[i]; worst_i = i; }
```
Với sorted pool: `worst = pool_scores[(int)(uint8_t)pool_order[filled-1]]` → **O(1)**. Xóa loop.

---

## Files cần thay đổi

| File | Thay đổi |
|------|---------|
| `gpu/include/pars_tree.cuh` | Thêm `d_poolOrder` (`uint8_t*`), `d_poolFilled` (`int*`), `d_poolLock` (`int*`) vào `GpuParsimonyMem` |
| `gpu/src/pars_tree.cu` | `gpuParsimonyMemAlloc`: alloc + init 3 arrays mới |
| `gpu/src/pars_build.cu` | `runPhase3`: Step 1 dynamic range, Step 3 sorted insert (thay whole block lines 694–756), xóa worst-scan loop 718–721 |
| `gpu/src/gpu_init_trees.cu` | Sau upload pool trong `hybrid_cb`: init `d_poolOrder=[0,1,...,actual-1]`, `d_poolFilled=actual_pool`, `d_poolLock=0` |
| `gpu/include/pars_build.cuh` | Cập nhật `buildPhase3Kernel` nếu signature thay đổi |
| `tools.h` / `tools.cpp` | Thêm `gpu_base_pool` (int, default=10), `gpu_pool_expand` (int, default=10) |

### `buildPhase3Kernel` — tham số thêm

```cuda
uint8_t* pool_order,   // [pool_size] sorted uint8_t physical-slot indices
int*     pool_filled,  // device ptr, filled count
int*     pool_lock,    // device ptr, spinlock
int      base_pool,    // accessible range at outer=0
int      expand_step   // +1 per expand_step outer iters (0 = no expansion)
```

### Init trong `gpu_init_trees.cu` (hybrid_cb, sau upload pool)

```cpp
// Pool đã sorted từ diversity selection → identity order [0, 1, ..., actual_pool-1]
std::vector<uint8_t> h_order(pool_size);
std::iota(h_order.begin(), h_order.end(), (uint8_t)0);
CUDA_CHECK(cudaMemcpy(cb_mem->d_poolOrder, h_order.data(),
                      pool_size * sizeof(uint8_t), cudaMemcpyHostToDevice));
int h_filled = actual_pool, h_lock = 0;
CUDA_CHECK(cudaMemcpy(cb_mem->d_poolFilled, &h_filled, sizeof(int), cudaMemcpyHostToDevice));
CUDA_CHECK(cudaMemcpy(cb_mem->d_poolLock,   &h_lock,   sizeof(int), cudaMemcpyHostToDevice));
```

---

## Verification

```bash
cd /raid/home/loinguyen/workspace/mpboot-gpu/build && make -j8

# Sanity: no crash, pool sorted ascending
./mpboot-avx -s ../data_debug/tree1.phy -use_gpu -seed 1 \
    -numpars 200 -sprdist 4 -gpu_stop 6 -gpu_pool_size 30 \
    -gpu_base_pool 10 -gpu_pool_expand 10 -gpu_k1_ratio 0.2 -gpu_device 6

# Benchmark vs baseline
./mpboot-avx -s ../data_treebase/dna_M3605_260_5315.phy \
    -use_gpu -seed 1 -numpars 1000 -sprdist 6 -gpu_stop 6 \
    -gpu_pool_size 30 -gpu_k1_ratio 0.2 \
    -gpu_base_pool 10 -gpu_pool_expand 10 -gpu_device 2
```
