# GPU Kernel Profile — Post Opt-M (2026-05-13)

**Method**: `ptxas --ptxas-options=-v` compile-time register analysis  
**Hardware**: A100-SXM4-80GB (compute 8.0, 108 SMs, 164 KB shared/SM, 65536 regs/SM)  
**Note**: `perf_event_paranoid=3` blocks hardware counters → ptxas compile-time only

---

## Register Count (ptxas info)

| Kernel | STATES | Registers/thread | Stack | Spill stores | Spill loads |
|--------|--------|-----------------|-------|-------------|-------------|
| buildParsimonyTreesKernel | 32 (fallback) | **173** | 200 B | 0 | 0 |
| buildParsimonyTreesKernel | 20 (protein) | **158** | 200 B | 0 | 0 |
| buildParsimonyTreesKernel | 4 (DNA) | TBD | 200 B | 0 | 0 |
| buildPhase3Kernel | 32 | **166** | 0 | 0 | 0 |
| buildPhase3Kernel | 20 (protein) | **134** | 0 | 0 | 0 |
| buildPhase3Kernel | 4 (DNA) | **148** | 0 | 0 | 0 |

**Key**: 0 register spills across ALL specializations → Opt-M không cải thiện memory traffic từ spill, mà speedup đến từ **ILP + instruction count reduction** từ loop unrolling.

---

## Occupancy Analysis

### Kernel parameters
- Shared memory/block: **28,400 bytes (28.4 KB)** = `sizeof(BuildShared)`
- Threads/block: **32** (1 warp)

### A100 SM resources
- Shared memory/SM: **164 KB** (max configurable)
- Registers/SM: 65,536
- Max warps/SM: 64 (= 2048 threads / 32)
- Max blocks/SM: 32

### Occupancy calculation per STATES

| Constraint | STATES=4 | STATES=20 | STATES=32 |
|-----------|----------|-----------|-----------|
| Shared mem limit: `164KB / 28.4KB` | **5 blocks** | **5 blocks** | **5 blocks** |
| Register limit (Phase3 kernel) | 65536/(148×32)=13 | 65536/(134×32)=15 | 65536/(166×32)=12 |
| Hardware max | 32 | 32 | 32 |
| **Binding constraint** | **Shared mem** | **Shared mem** | **Shared mem** |
| **Max blocks/SM** | **5** | **5** | **5** |
| **Max warps/SM** | **5** | **5** | **5** |
| **Theoretical occupancy** | **7.8%** | **7.8%** | **7.8%** |

**→ SharedMemory là binding constraint cho tất cả STATES. Opt-M KHÔNG thay đổi occupancy.**

---

## Phân tích Speedup của Opt-M

Opt-M cho 2.16× speedup (np=200) mặc dù occupancy không đổi. Nguyên nhân:

1. **Loop unrolling**: `#pragma unroll` với STATES compile-time → compiler unroll 4/20/32 lần → loại bỏ loop counter overhead, pipeline instructions tốt hơn
2. **ILP (Instruction-Level Parallelism)**: Compiler biết STATES tại compile time → sắp xếp instructions để pipeline không stall
3. **Instruction count giảm**: Unrolled loop có ít branch instructions hơn
4. **Register file pressure thấp hơn**: Ít registers hơn → ít competition trên register file → higher instruction throughput

**KHÔNG phải do**: Occupancy (unchanged), register spill reduction (0 spills đã có từ trước)

---

## Key Finding: Next Bottleneck

**`BuildShared = 28.4 KB` là bottleneck duy nhất cho occupancy.**

Nếu giảm `BuildShared` xuống:
| Target size | blocks/SM | warps/SM | Occupancy |
|-------------|-----------|----------|-----------|
| 28.4 KB (current) | 5 | 5 | **7.8%** |
| 20 KB | 8 | 8 | **12.5%** |
| 16 KB | 10 | 10 | **15.6%** |
| 14 KB | 11 | 11 | **17.2%** |
| ~8 KB | 20 | 20 | **31.3%** |

**Potential**: Nếu tăng occupancy từ 7.8% lên 15.6% (2× warps/SM) → latency hiding tốt hơn → thêm speedup đáng kể.

---

## BuildShared Breakdown (current 28.4 KB)

| Field | Size | Ghi chú |
|-------|------|---------|
| `perm[802]` | 3.2 KB | Stepwise addition only (Phase 1) |
| `stack[1600]` | 6.4 KB | Build DFS + SPR addTraverse |
| `stackMint[800]` | 3.2 KB | doAddTraverse mintrav |
| `stackMaxt[800]` | 3.2 KB | doAddTraverse maxtrav (cũng dùng làm NNI bitset) |
| `ti[2400]` | 9.6 KB | Traversal info array |
| `tiStack[800]` | 3.2 KB | DFS stack cho computeTraversalInfo |
| Scalars + timing | ~0.4 KB | seed, randomMP, etc. |
| **Total** | **~29 KB** | |

**Lớn nhất**: `ti[2400]` = 9.6 KB (3 × kMaxTaxa × 4 bytes) và `stack[1600]` = 6.4 KB.

---

## Ghi chú

- `kMaxTaxa = 800` là compile-time constant → arrays sized for worst case
- Thực tế với N=295: ti cần max ~(2N-2) × 3 = 1764 entries → 7.1 KB (đủ)
- Với N=100: ti cần max ~600 entries → 2.4 KB (có 7.2 KB dư)
- Arrays được sized cho kMaxTaxa=800 nhưng phần lớn datasets có N<400
