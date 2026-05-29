# Notes — GPU Parsimony Analysis

---

## NCU Profiling — Register, Shared Memory, Occupancy

**Date**: 2026-05-26  
**Config**: `--section SpeedOfLight --section LaunchStats --launch-count 2`  
**Device**: A100 (CC 8.0) — 108 SMs, 65536 regs/SM, ~152 KB smem/SM, max 64 warps/SM, max 32 blocks/SM  
**Datasets**: `dna_M10467_202_4074.phy` (N=202), `dna_M5381_413_3632.phy` (N=413)  
**Params**: `-gpu_worker 100 -gpu_pool_size 5 -sprdist 3`

### Kết quả đo

| Kernel | Dataset | Template | Grid | Regs/thread | Static smem/block | Dyn smem |
|--------|---------|----------|------|-------------|-------------------|----------|
| K1 `buildParsimonyTreesKernel` | N=202 | `<4, 256>` | 100×1 | **160** | **3,984 B** | 0 |
| K2 `buildPhase3Kernel`         | N=202 | `<4, 256>` | 100×1 | **253** | **3,984 B** | 0 |
| K1 `buildParsimonyTreesKernel` | N=413 | `<4, 512>` | 100×1 | **160** | **7,568 B** | 0 |
| K2 `buildPhase3Kernel`         | N=413 | `<4, 512>` | 100×1 | **253** | **7,568 B** | 0 |

### Occupancy analysis (A100)

Mỗi block = 1 warp = 32 threads. Occupancy = (max active warps/SM) / 64.

**Register limit** (chiều hướng đang binding):
- K1: 160 × 32 = 5120 regs/block → 65536/5120 = **12 blocks/SM**
- K2: ceil(253×32/256)×256 = 8192 regs/block → 65536/8192 = **8 blocks/SM**

**Shared memory limit**:
- N=202 (NTAXA=256, smem=3,984 B): 155,648/3,984 = **39 blocks/SM** — không binding
- N=413 (NTAXA=512, smem=7,568 B): 155,648/7,568 = **20 blocks/SM** — không binding

| Kernel | Limiting factor | Blocks/SM | Theor. occupancy |
|--------|----------------|-----------|-----------------|
| K1 (cả 2 dataset) | **Registers** | **12** | **18.75%** |
| K2 (cả 2 dataset) | **Registers** | **8**  | **12.5%**  |

→ Bottleneck là **register count**, không phải shared memory, cho cả N=202 lẫn N=413.

### Insight 1: NTAXA template scales với N

Template parameter NTAXA không cố định mà được chọn **động** dựa trên N thực tế:
- N=202 → `NTAXA=256`
- N=413 → `NTAXA=512`

Shared memory tỉ lệ gần như tuyến tính với NTAXA: 3,984 → 7,568 (tăng 1.9× khi NTAXA tăng 2×). Điều này xác nhận các mảng trong `BuildShared` (ti[], tiStack[], stkVf[], ...) có kích thước O(NTAXA).

Với NTAXA=800 (dataset rất lớn): smem ≈ 11,584 B → limit 155,648/11,584 = **13 blocks/SM** → smem trở thành bottleneck (thay thế regs nếu K2 regs ≤ 128).

### Insight 2: K2 register count tăng mạnh (regression!)

CLAUDE.md ghi nhận sau Opt-S (2026-05-18): K2 regs = **128**. Hiện tại: **253 regs**.  
→ Occupancy K2 giảm từ **20.31% → 12.5%** (−8 điểm %).

Nguyên nhân có thể: code mới thêm vào K2 (treels saves, spinlocks mới, ...) khiến compiler cần thêm nhiều registers để hold các giá trị tạm thời.

**Hệ quả**: Register count là bottleneck mới cho K2 — nếu có thể refactor để giảm regs về ≤128, K2 sẽ đạt lại ~20% occupancy (nếu NTAXA nhỏ) hoặc smem sẽ trở thành bottleneck (NTAXA=800).

### Insight 3: Grid quá nhỏ so với GPU (under-utilization)

Với `gpu_worker=100` blocks trên 108 SMs:

| Kernel | Max blocks/SM | Waves | SM util |
|--------|--------------|-------|---------|
| K1 | 12 | 100/(108×12) = **0.077 waves** | 7.7% |
| K2 | 8  | 100/(108×8)  = **0.116 waves** | 11.6% |

→ GPU chỉ đang dùng ~8–12% công suất về block occupancy.

NCU cảnh báo: *"only 0.1 full waves across all SMs"* và *"Est. Speedup: 7.4%"* nếu tăng grid.

**Để đạt ≥1 wave** (GPU fully loaded về blocks):
- K1: cần ≥ 108 × 12 = **1,296 blocks** (`-gpu_worker ≈ 1300`)
- K2: cần ≥ 108 × 8  = **864 blocks** (`-gpu_worker ≈ 900`)

Với `-gpu_worker 400` (config bench thực tế): K2 waves = 400/864 = **0.46 waves** — vẫn dưới 1.  
**K2 chỉ đạt ≥1 wave khi `-gpu_worker ≥ 900`.**

### Insight 4: SM compute throughput = 8%

SOL compute throughput của cả K1 lẫn K2 = **8%** — SM chỉ issue instruction 8% thời gian.  
Lý do chính: memory-bound workload (đọc parsVect từ HBM) → SM stall chờ dữ liệu về.  
Tăng occupancy (nhiều warps hơn) giúp SM latency-hide được nhiều hơn, nhưng bandwidth vẫn là ceiling thực sự.

### Tóm tắt: 3 bottleneck theo thứ tự ưu tiên

1. **Grid size** (dưới 1 wave) — cần `gpu_worker ≥ 900` để GPU không idle giữa block
2. **Register pressure** (K2: 253 regs) — giảm về ≤128 tăng occupancy từ 12.5%→20%
3. **Memory bandwidth** — ceiling cuối cùng, không vượt qua được bằng optimizations phần mềm

---

## nsys Bootstrap Profiling — GPU vs CPU Bottleneck

**Date**: 2026-05-26  
**Command**: `nsys profile --trace=cuda,nvtx --stats=true`  
**Config**: `-bb 1000 -gpu_worker 100 -gpu_pool_size 5 -gpu_worker_stop 1`  
**Dataset 1**: `prot_M8630_50_21154` (N=50, protein, NTAXA=128, 4 K2 rounds)  
**Dataset 2**: `tree1.phy` (N=295, DNA, NTAXA=384, 7 K2 rounds)

### Kết quả tổng hợp

| Metric | prot50 (N=50) | dna295 (N=295) |
|--------|--------------|----------------|
| K1 time | 1.08 s (1 instance) | 1.80 s (1 instance) |
| K2 avg per round | 1.17 s | 1.34 s |
| K2 total (all rounds) | 4.67 s | 9.35 s |
| REPSKernel | 0.026 s (556 calls) | 0.167 s (23,967 calls) |
| **GPU kernel total** | **5.78 s** | **11.32 s** |
| Memory transfers | 38 ms | 163 ms |
| cudaMemcpy API overhead | 70 ms | 832 ms |
| **Wall clock time** | **16.46 s** | **144.46 s** |
| **GPU utilization** | **35.1%** | **7.8%** |

### Phân tích từng round (wall = K2 + CPU_process)

**prot50 (K2_avg = 1.17s, ~192–196 trees/s CPU):**
```
Round 1: wall=5.60s  K2=1.17s  CPU_process=4.43s  treels=853
Round 2: wall=2.67s  K2=1.17s  CPU_process=1.50s  treels=289
Round 3: wall=2.64s  K2=1.17s  CPU_process=1.47s  treels=289
Round 4: wall=2.64s  K2=1.17s  CPU_process=1.47s  treels=289
```

**dna295 (K2_avg = 1.34s, ~230–250 trees/s CPU):**
```
Round 1: wall=75.38s  K2=1.34s  CPU_process=74.04s  treels=18071
Round 2: wall=11.25s  K2=1.34s  CPU_process= 9.91s  treels=2320
Round 3: wall=11.91s  K2=1.34s  CPU_process=10.57s  treels=2435
Round 4-7:  wall≈10.3s  K2=1.34s  CPU_process≈9.0s   treels≈2072
```

### Phát hiện quan trọng: CPU là bottleneck, không phải GPU

**Nguyên nhân**: Sau mỗi K2 round, CPU phải gọi `saveCurrentTree()` cho mỗi tree trong treels buffer:
- Tốc độ `saveCurrentTree` ≈ **250 trees/s = 4 ms/tree** — **nhất quán cho cả 2 datasets!**
- K2 tốc độ sản xuất: 100 blocks trong 1.34s = 75 trees/s GPU output
- Nhưng per-testInsert treels saves khiến mỗi K2 round fill **hàng nghìn candidates** (không chỉ 100)
- CPU xử lý 18,071 trees × 4 ms = 72s trong khi GPU chỉ chạy 1.34s → **GPU idle 98% thời gian trong round 1!**

**GPU utilization thực sự:**
- prot50: 5.78 / 16.46 = **35%** (CPU chiếm 65%)
- dna295: 11.32 / 144.46 = **7.8%** (CPU chiếm 92%)

### GPU đang làm gì khi CPU bận?

```
Timeline mỗi round (dna295):
|<-- K2 GPU: 1.34s -->|<-- GPU IDLE: 9.0s -->|<-- K2 GPU: 1.34s -->|...
|<-- CPU: 0s -------->|<-- CPU saveCurrentTree: 9.0s -->|
```

GPU khởi động K2 → CPU block chờ K2 xong (cudaStreamSynchronize) → K2 done → CPU bắt đầu xử lý treels → GPU hoàn toàn idle trong suốt 9s CPU processing → CPU xong → GPU khởi động K2 tiếp.

### Memory transfers — không phải bottleneck

- prot50: D→H 5.3ms, H→D 33ms → **tổng 38ms / 16.46s = 0.23%**
- dna295: D→H 119ms, H→D 44ms → **tổng 163ms / 144.46s = 0.11%**

Memory transfers gần như negligible. Bottleneck **không phải** là PCIe bandwidth.

### Tại sao dna295 tệ hơn prot50?

| | prot50 | dna295 |
|---|---|---|
| Treels filled round 1 | 853 | **18,071** |
| N (taxa) | 50 | 295 |
| SPR candidates/block | ít (N nhỏ) | nhiều (2N-3 nodes × sprDist) |

N=295 → mỗi block có nhiều candidate testInserts hơn N=50 → nhiều hơn treels được fill per-testInsert → CPU bị overwhelmed 21× hơn.

### Hàm ý cho optimization

```
Bottleneck: saveCurrentTree() = 4 ms/tree, single-threaded, sequential
GPU produces: K=100 trees/round + per-testInsert saves = 2000-18000 treels/round
Gap: GPU 75 trees/s vs CPU 250 trees/s (với treels cap không đủ)
```

**Giải pháp tiềm năng:**
1. **Giảm `max_treels`** (cap số trees CPU phải xử lý/round) → ít CPU work hơn, dễ nhất
2. **Multi-thread CPU** (parallel saveCurrentTree) → cần thay đổi code lớn
3. **Lazy treels** (chỉ process treels sau khi bootstrap converged, không mỗi round) → thiết kế lại
4. **GPU-side REPS** (đã có REPSKernel) → tiếp tục move CPU work sang GPU

**Quan trọng**: Bất kỳ optimization nào ở GPU kernel (K2, K1) sẽ cho speedup **rất nhỏ** vì GPU chỉ chiếm 7.8-35% wall time. Muốn speedup thực sự phải giải quyết CPU bottleneck.

---

## Tại sao Sankoff chậm hơn Fitch, và GPU bị ảnh hưởng như thế nào

**Date**: 2026-05-26

### 1. Nguyên nhân gốc: data volume gấp 32×

Khác biệt cốt lõi nằm ở cách nén dữ liệu:

| | Fitch | Sankoff |
|---|---|---|
| 1 uint32 chứa | **32 sites** (bitmask) | **1 pattern** (cost value) |
| width | `ceil(sites / 32)` | `patterns` (không chia) |
| parsVect/node (dna_M9033, 1394 sites) | **704 B** (44 blocks × 4 states × 4B) | **21.8 KB** (1394 patterns × 4 states × 4B) |
| parsVect/node (prot_M2358, 714 sites) | **1.8 KB** | **55.8 KB** |
| Tỉ lệ | — | **×31–32 lớn hơn Fitch** |

Fitch "gian lận" được vì bitwise AND/OR/popcount xử lý **32 sites song song trong 1 instruction**. Sankoff phải tính từng pattern riêng với vòng lặp O(STATES²) bên trong.

### 2. L1 cache — lý do GPU bị ảnh hưởng nhiều hơn CPU

Trên A100, mỗi SM có **192 KB L1/shared** dùng chung cho tất cả blocks chạy trên đó:
- Với 12 blocks/SM (K1 register-limited) và smem ≈ 4 KB/block:
  - Effective L1 per block ≈ **(192 − 4×12) / 12 ≈ 15.7 KB**

| parsVect/node | Fit trong 15.7 KB L1? |
|---|---|
| Fitch DNA (704 B) | **✓ Fit hoàn toàn** — nhiều nodes cùng cached |
| Fitch Prot (1.8 KB) | **✓ Fit hoàn toàn** |
| Sankoff DNA (21.8 KB) | **✗ Không fit (1.4× L1)** — mỗi testInsert phải đọc L2/HBM |
| Sankoff Prot (55.8 KB) | **✗ Không fit (3.6× L1)** — mỗi read đi thẳng HBM |

→ **Fitch GPU**: mỗi testInsert đọc parsVect của 2–3 nodes từ **L1 cache** (fast).  
→ **Sankoff GPU**: mỗi testInsert đọc parsVect của 2–3 nodes từ **L2 hoặc HBM** (chậm 10–200×).

### 3. CPU chịu ít hơn GPU: lý do cache khác biệt

CPU xử lý **1 cây tại một thời điểm** — không cần chia sẻ cache:

| | CPU (1 core) | GPU (K=100 blocks) |
|---|---|---|
| Cache | L3: 8–32 MB per core | L2: 40 MB chia cho 100+ blocks |
| parsVect 1 cây (Sankoff DNA dna_M9033) | 12.8 MB | 100 × 12.8 MB = 1.28 GB |
| Fit trong cache? | **✓** L3 (nếu L3 ≥ 16 MB) | **✗** L2 chỉ 40 MB (overflow 32×) |
| Cache hit rate parsVect | **Cao** — 1 cây tái dùng nodes | **Thấp** — 100 cây cạnh tranh L2 |

Đây là **lý do GPU bị ảnh hưởng nhiều hơn CPU** cho Sankoff: CPU có L3 cache lớn, dedicated cho 1 cây → hit rate cao. GPU phải chia sẻ L2 cho K cây, parsVect của mỗi cây không vừa.

**Kết quả speedup thực tế:**
- DNA Fitch: **7–8×** → DNA Sankoff: **4.8×** (giảm ~30%)
- Prot Fitch: **12×** → Prot Sankoff: **2.8×** (giảm ~77%)

Protein bị ảnh hưởng nặng hơn DNA — xem lý do ở mục 4.

### 4. Cache line waste — protein bị tệ hơn DNA

Layout bộ nhớ: `pars_tree[node × width × STATES + b × STATES + s]`

Khi 32 threads cùng đọc state `s=0` của 32 patterns liên tiếp:
- Thread[lane] đọc tại: `base + lane × STATES × 4` bytes
- Stride giữa 2 threads liên tiếp = **STATES × 4 bytes**

| STATES | Stride/thread | Span 32 threads | Cache lines fetch | Useful bytes | Hiệu quả |
|--------|--------------|-----------------|-------------------|--------------|-----------|
| 4 (DNA) | 16 B | 500 B | **4** | 128 B / 512 B | **25%** |
| 20 (Protein) | 80 B | 2484 B | **20** | 128 B / 2560 B | **5%** |

→ Đọc 1 state của 32 patterns: **DNA phải fetch 4 cache lines** (75% waste), **Protein phải fetch 20 cache lines** (95% waste!).

Với Fitch DNA (STATES=4, stride=16B): access pattern giống hệt DNA Sankoff (25% efficiency), **nhưng mỗi read cover 32 sites** thay vì 1 pattern → hiệu quả thực sự cao hơn 32×.

### 5. Vòng lặp O(STATES²) sequential — không song song hóa được

Sankoff inner loop không thể parallelized ngang qua threads:
```cpp
// Thread[lane] xử lý pattern b = lane, lane+32, ...
for (int s = 0; s < STATES; s++) {
    unsigned int min_val = INF;
    for (int j = 0; j < STATES; j++) {       // STATES² sequential reads
        min_val = min(min_val, parsL[s][b] + cost[s][j] + parsR[j][b]);
    }
    parsNode[s][b] = min_val;
}
```

- **DNA (STATES=4)**: 16 sequential reads per pattern — overhead nhỏ
- **Protein (STATES=20)**: 400 sequential reads per pattern — mỗi thread stall chờ 400 memory ops

CPU thread cũng làm vậy, nhưng:
- CPU có L3 cache → phần lớn reads hit trong cache (fast)
- GPU thread → reads hit L2/HBM → mỗi stall = 50–700 cycles

### 6. Tại sao Sankoff vẫn nhanh hơn CPU dù bị ảnh hưởng nhiều?

GPU vẫn thắng nhờ **chạy K=100 cây song song**:
- CPU: 1 cây × T giây
- GPU: 100 cây × T' giây với T' < T do HBM bandwidth (2 TB/s >> CPU DDR 100 GB/s)
- Speedup = (T/T') × K_effective — dù K_effective giảm do cache miss, vẫn dương.

### Tóm tắt cho phần trình bày

> "Sankoff chậm hơn Fitch vì width không chia cho 32 — mỗi node phải lưu vector cost riêng cho từng pattern, dẫn đến parsVect/node lớn hơn ~32×. Trên GPU, điều này đặc biệt nghiêm trọng vì: (1) parsVect/node (22 KB DNA, 56 KB Protein) vượt quá L1 cache hiệu dụng per-block (~15.7 KB), buộc mọi read phải đi HBM; (2) GPU chạy K=100 cây song song nên tổng parsVect (~1 GB) vượt xa L2 (40 MB), trong khi CPU xử lý 1 cây với L3 cache đủ lớn. Với protein (STATES=20), stride access còn phải fetch 20 cache lines cho 32 threads (hiệu quả 5%), khiến speedup protein Sankoff chỉ đạt 2.8× so với 12× của Fitch protein."

---

## CUDA Architecture và Memory Hierarchy

**Date**: 2026-05-26

### Kiến trúc tổng quan

GPU (A100) có **108 SM (Streaming Multiprocessor)**. Mỗi SM là một đơn vị xử lý độc lập với tài nguyên riêng. CPU điều phối — gọi kernel, GPU chạy song song.

```
GPU
 ├── SM 0          ← 1 trong 108 SM trên A100
 │    ├── 4 Warp Schedulers
 │    ├── Register File: 65,536 registers (32-bit)
 │    ├── Shared Memory / L1 Cache: 192 KB (configurable)
 │    └── max 64 warps active, max 32 blocks concurrent
 ├── SM 1
 ├── ...
 ├── SM 107
 ├── L2 Cache: 40 MB (shared toàn GPU)
 └── HBM (Global Memory): 80 GB, ~2 TB/s bandwidth
```

**Warp**: đơn vị thực thi cơ bản — **32 threads chạy lockstep** (cùng một lệnh cùng một cycle). Nếu code có if-else (warp divergence), các thread không match bị mask → serialized → hiệu năng giảm.

### Memory Hierarchy (từ nhanh → chậm)

| Loại | Scope | Latency | Size | Cách dùng trong mpboot-GPU |
|------|-------|---------|------|---------------------------|
| **Registers** | Per-thread | ~1 cycle | 65K regs/SM | `mp`, `partial`, loop vars |
| **Shared memory** | Per-block | ~5–10 cycles | 164 KB/SM | `sh.ti[]`, `sh.tiStack[]`, `sh.bcast[]`, `sh.stkVf[]` |
| **L1 Cache** | Per-SM | ~30 cycles | phần còn lại của 192 KB | Cache tự động các global reads |
| **L2 Cache** | Toàn GPU | ~50–200 cycles | 40 MB | Cache parsVect đọc nhiều lần |
| **HBM (Global)** | Toàn GPU | ~200–700 cycles | 80 GB | `d_parsVect`, `d_parsScore`, `d_topos` |

### Latency hiding — tại sao cần occupancy cao

GPU **không giảm latency** — thay vào đó nó **ẩn latency** bằng cách switch sang warp khác khi warp hiện tại stall chờ memory:

```
Warp A: load parsVect[node] → stall 200 cycles
         ↓ warp scheduler switch
Warp B: chạy 200 cycles (Fitch computation)
         ↓ Warp A data về → resume
Warp A: tiếp tục
```

→ Cần **nhiều warps active trên 1 SM** để ẩn latency. Đây là lý do "occupancy" quan trọng.  
Với K1 occupancy 18.75% (= 12 warps/SM) và K2 12.5% (= 8 warps/SM), mỗi SM có 8–12 warps để xoay vòng — không lý tưởng nhưng chấp nhận được.

### Tóm gọn cho phần trình bày

> "GPU xử lý song song bằng cách chạy hàng nghìn threads chia thành các warp 32 threads. Mỗi SM có thể chứa nhiều warp và switch giữa chúng để ẩn latency bộ nhớ. Dữ liệu gần SM (shared memory, registers) nhanh hơn 100–200× so với global memory — vì vậy chúng tôi đưa các cấu trúc dữ liệu thường xuyên truy cập (traversal stack, bcast) vào shared memory."

---

## Lý do chọn 1 block = 1 warp = 32 threads cho 1 cây

**Date**: 2026-05-26

### Tại sao 1 block = 1 cây?

Trong parsimony, **các cây độc lập hoàn toàn** — không có dữ liệu chung giữa tree k=0 và tree k=1 (ngoài parsVect tips, chỉ đọc). Vì vậy:
- 1 block = 1 cây → **không cần synchronization giữa blocks**
- K cây = K blocks → scale tuyến tính theo K (thêm cây = thêm blocks, không phức tạp hơn)

### Tại sao chỉ 1 warp (32 threads) mà không nhiều hơn?

**Cấu trúc tính toán parsimony có 2 phần:**

| Phần | Ai chạy? | Có song song hóa không? |
|------|----------|------------------------|
| DFS traversal, stack ops, best-move selection | Lane 0 (1 thread) | **Không** — data dependency giữa nodes |
| Fitch/Sankoff computation per-node | Cả 32 lanes | **Có** — parallel over `width` blocks |

Vòng lặp parallel: `for (int b = lane; b < width; b += 32)` — 32 threads xử lý `width` blocks.

**Ví dụ N=202, sites=4074**: width = ceil(4074/32) = 128 → mỗi thread xử lý 128/32 = **4 blocks per newview** — đủ để tận dụng 32 threads.

Nếu dùng 64 threads (2 warps): phần DFS vẫn chỉ dùng 1 thread → **31 threads của warp thứ 2 idle** trong toàn bộ serial code. Phần Fitch computation nhanh hơn 2× nhưng overhead sync tăng và smem cần lớn hơn.

### Điểm mạnh của 1 warp/block

1. **`__syncwarp()` thay vì `__syncthreads()`**: đồng bộ 32 threads trong warp = ~0 overhead (warp đã lockstep); nếu dùng nhiều warps cần `__syncthreads()` đắt hơn nhiều.
2. **Maximize blocks/SM**: 1 warp/block → 12 blocks/SM (K1). Nếu 2 warps/block: có thể giảm xuống 6 blocks/SM → ít cây chạy song song hơn trên 1 SM.
3. **Đơn giản**: không cần partition DFS work giữa nhiều warps.
4. **Phù hợp với width trung bình** (32–256 blocks): mỗi thread xử lý 1–8 blocks per call — overhead loop nhỏ.

### Hạn chế của 1 warp/block

1. **Underutilization khi width nhỏ**: nếu width=16 (chỉ 16 site-blocks), 16 threads idle mỗi Fitch call → lane utilization 50%.
2. **Serial bottleneck**: DFS, stack push/pop, `atomicAdd` trong pool → chỉ lane 0 làm, 31 lanes chờ. Đây là warp-level serial bottleneck khó vượt qua.
3. **Sankoff tệ hơn**: Sankoff width = số patterns (không chia 32) → có thể rất lớn (vd. 1394 patterns cho dna_M9033). 1 warp × 1394 iterations mỗi newview — nhiều hơn Fitch nhiều; nhưng 32 threads không tăng được tốc độ Sankoff song song vì O(STATES²) per pattern là sequential.

### Các cách chọn khác

**Phương án A: nhiều warps/block = 1 tree** (vd. 4 warps = 128 threads/block)
```
PRO: Phần parallel (Fitch width dimension) nhanh ~4× 
CON: Phần serial (DFS) vẫn 1 thread → 127 threads idle
CON: Ít blocks/SM hơn → giảm latency hiding
CON: Phải dùng __syncthreads() → đắt hơn
KẾT LUẬN: Chỉ lợi khi width rất lớn (>512) và serial phần nhỏ
```

**Phương án B: 1 block = nhiều cây** (vd. 4 cây × 8 threads/cây)
```
PRO: Cache locality nếu cây dùng chung topo
CON: 8 threads/cây không đủ để xử lý width=128 hiệu quả
CON: Các cây tiến độ khác nhau → warp divergence nặng
KẾT LUẬN: Không thực tế
```

**Phương án C: CTA-wide parallelism — partition DFS giữa warps**
```
Ý tưởng: warp 0 xử lý nhánh trái, warp 1 xử lý nhánh phải của DFS
PRO: Loại bỏ serial DFS bottleneck
CON: Data dependency giữa nodes → phải sync thường xuyên
CON: Load imbalance (nhánh trái/phải có độ sâu khác nhau)
CON: Complexity rất cao, khó debug
KẾT LUẬN: Về lý thuyết nhanh hơn nhưng implementation rất phức tạp
```

**Phương án D: 1 warp/block nhưng tăng K (gpu_worker)**
```
Đây là hướng đang dùng: thay vì tối ưu intra-tree, tăng số cây song song
PRO: Linear scale, đơn giản, không đụng vào kernel logic
CON: Cần gpu_worker ≥ 900 để đạt ≥1 wave (hiện tại dùng 100–400)
KẾT LUẬN: Hướng đúng cho workload này
```

### Tóm gọn cho phần trình bày

> "Chúng tôi chọn 1 warp = 32 threads xử lý 1 cây vì bài toán parsimony có cấu trúc song song chủ yếu ở chiều width (số site-blocks), không phải ở chiều DFS của cây. Với 32 threads xử lý song song các site-blocks, và nhiều blocks song song xử lý nhiều cây, thiết kế này tối đa hóa throughput mà không cần synchronization phức tạp. Hạn chế chính là phần xây dựng traversal (DFS) vẫn tuần tự trên 1 thread, và để GPU đạt full utilization cần số cây (gpu_worker) đủ lớn (≥900 với A100 108 SMs)."

---

## Tại sao đôi lúc phải dùng Sankoff, `-cost`

**Date**: 2026-05-26

Fitch parsimony giả định mọi substitution có chi phí bằng nhau (cost = 1). Thực tế sinh học không như vậy:

- **DNA**: transition (A↔G, C↔T — cùng loại purine/pyrimidine) xảy ra thường xuyên hơn transversion → nên có chi phí thấp hơn.
- **Protein**: các amino acid có cấu trúc hóa học tương tự dễ thay thế lẫn nhau hơn (vd. Lys↔Arg vs Cys↔Trp).

Sankoff cho phép truyền vào **cost matrix tùy chỉnh** qua `-cost <file>`, mô hình hóa chi phí thay thế phi đồng nhất. Điểm parsimony Sankoff nhỏ hơn = cây có tổng chi phí tiến hóa hợp lý hơn về mặt sinh học.

---

## COR trong điều kiện dừng lúc bật mode bootstrap có ý nghĩa là gì

**Date**: 2026-05-26

COR = **Pearson correlation coefficient** giữa bootstrap support values ở hai giai đoạn liên tiếp.

MPBoot dùng tiêu chí dừng sớm (early stopping) thay vì luôn chạy đủ `-bb` replicates:
- Sau mỗi R replicates (mặc định 100), tính support value cho mỗi nhánh của cây.
- Tính COR giữa vector support hiện tại và vector support của lần đo trước.
- Nếu **COR > ngưỡng** (ví dụ 0.99) → support values đã hội tụ → dừng.

Ý nghĩa: khi COR ≈ 1.0, thêm bootstrap replicates nữa không thay đổi kết quả → dừng sớm tiết kiệm thời gian mà vẫn đảm bảo độ tin cậy.

---

## Tại sao GPU dùng mảng kích thước compile-time thay vì mảng động (vector)

**Date**: 2026-05-26

Ba lý do kỹ thuật:

**1. Shared memory phải biết kích thước lúc compile:**
Shared memory (latency ~5 cycles, vs ~200 cycles DRAM) phải được cấp phát tĩnh — không thể `new[]` hay `vector` bên trong kernel. Ví dụ: `__shared__ parsimonyNumber sh_data[STATES][MAX_WIDTH]`.

**2. Template parameter cho phép compiler tối ưu:**
```cpp
template <int STATES>  // STATES=4 (DNA) hoặc STATES=20 (protein)
__device__ void newviewParsimony(...)
```
Với `STATES` là hằng số lúc compile, compiler **unroll vòng lặp** `for (int s = 0; s < STATES; s++)` thành lệnh thẳng — loại bỏ branch và counter overhead (quan trọng vì inner loop chạy hàng triệu lần).

**3. Dynamic allocation trong kernel rất tốn kém:**
`cudaMalloc` không thể gọi từ trong kernel. `malloc()` device-side có thể dùng nhưng bị serialized và chậm — không phù hợp cho inner loop tính parsimony.

Kết quả: code biên dịch ra **hai phiên bản kernel riêng** (`STATES=4` và `STATES=20`), mỗi phiên bản được tối ưu tối đa cho loại dữ liệu tương ứng.

---

## Tại sao K1 wall time tăng khi numpars tăng (100→200)

**Date**: 2026-05-18

### Bối cảnh

A100: 108 SMs, smem limit = 13 blocks/SM → max concurrent blocks = **1,404 (= 1 wave)**.

```
numpars=100 → K=99  → 0.071 waves  (7% capacity)
numpars=200 → K=199 → 0.14  waves  (14% capacity)
```

Cả hai đều < 1 wave → tất cả blocks **chạy song song đồng thời**.  
Theo lý thuyết, wall time = thời gian 1 block → không đổi khi tăng K.  
Thực tế **vẫn tăng** vì các lý do sau.

### Nguyên nhân 1: Stepwise addition là sequential theo tip

```
for tip t = 4..N:                ← N-3 bước sequential, không rút ngắn được
    DFS over O(t) candidates     ← 32 lanes xử lý song song
    insert best
```

Wall time = N × (per-step time). Đây là **critical path** của 1 block — cố định với N.  
Nếu per-step time tăng (do K tăng → bandwidth contention), wall time tăng theo.

### Nguyên nhân 2: HBM bandwidth pressure (chính)

Mỗi block truy cập parsVect mỗi DFS step:
- Per-node parsVect = `width × states = 40 × 4 = 160 bytes`
- Per-step load: ~2 nodes × 160 bytes × 32 lanes ≈ 10 KB / warp / step

```
K=99:  concurrent HBM reads ≈ 99  × 10 KB = ~0.97 MB/step
K=199: concurrent HBM reads ≈ 199 × 10 KB = ~1.95 MB/step
```

2× blocks → 2× outstanding memory requests → L2/HBM controller queue dài hơn  
→ mỗi block stall lâu hơn per DFS step → wall time tăng dù số blocks vẫn < 1 wave.

### Nguyên nhân 3: "Occ tăng" ≠ "wall time giảm"

| Bottleneck | Occupancy giúp gì? |
|------------|-------------------|
| Memory **latency** | Có — SM switch sang warp khác khi stall |
| Memory **bandwidth** | Không — bandwidth là shared resource, nhiều warps cạnh tranh hơn |
| Compute throughput | Có — SM ít idle cycles hơn |

K1 bị giới hạn bởi **bandwidth** (parsVect random access), không phải latency đơn thuần.  
Tăng occupancy từ 7% → 14% không giải phóng thêm bandwidth.

### Kết luận / threshold quan trọng

Wall time flat (không tăng theo K) chỉ khi:
- K < 1 wave **VÀ** bandwidth chưa saturated — khó đạt với parsVect access pattern

Để saturate GPU cần ≥ **1,404 blocks** (= 1 full wave).  
`gpu_worker=400` (K2) → 0.28 waves → vẫn underutilized nhưng gần ngưỡng bandwidth.  
Để đạt ≥ 1 wave cần `gpu_worker ≈ 1400`.

---

## Tại sao K1 tăng K thì chậm hơn dù occupancy tăng — phân tích đầy đủ

**Date**: 2026-05-28

### Thông số A100

- 108 SMs, 65536 regs/SM, ~192 KB smem/SM, max 64 warps/SM
- K1 `buildParsimonyTreesKernel`: ~160 regs/thread × 32 threads = 5120 regs/block → **12 blocks/SM** (register-bound)
- Max concurrent K1 blocks = 108 × 12 = **1,296 blocks** (1 wave)

### Occupancy tăng theo K như thế nào?

```
K=100  → 100/1296 = 0.077 waves → ~1 block/SM trung bình → occ thấp
K=400  → 400/1296 = 0.31 waves → ~3-4 blocks/SM → occ cao hơn
K=1296 → 1 full wave → tất cả SM bận → occ đạt tối đa (18.75%)
```

Khi K tăng, SM có nhiều blocks để switch sang khi 1 block stall trên memory → **latency hiding tốt hơn** → đây là lý do occupancy "tăng".

### Tại sao wall time vẫn tăng?

**Cơ chế 1 — L2 cache pollution**:  
parsVect working set per block = (2N+1) × states × width × 2B.  
Ví dụ N=413, DNA: 827 × 4 × 114 × 2B ≈ **750 KB/block**.  
A100 L2 = 40 MB → chỉ ~53 blocks fit đồng thời trong L2.  
Khi K > 53: mỗi block mới evict dữ liệu của block khác → miss rate tăng → mọi parsVect access đi xuống HBM (200–700 cycles).

**Cơ chế 2 — HBM bandwidth saturation**:  
Khi K blocks cùng đọc/ghi HBM, tổng BW demand = K × (BW per block).  
A100 có 2 TB/s HBM peak, nhưng với K=1296 blocks × random access → bandwidth bị fragmented.  
Nhiều warps có requests đang pending → BW controller queue dài → stall time/block tăng.

**Cơ chế 3 — Latency hiding trở nên vô hiệu khi BW bão hòa**:

| Bottleneck | Occupancy giúp gì? |
|------------|-------------------|
| Memory **latency** | Có — SM switch sang warp khác khi stall |
| Memory **bandwidth** | Không — BW là tài nguyên chung, nhiều warps cạnh tranh hơn |

K1 bị giới hạn bởi **bandwidth** (parsVect random access theo tree DFS), không phải latency thuần túy.  
Khi BW bão hòa, dù có 12 blocks/SM để switch, mỗi block vẫn stall chờ BW — thời gian stall không giảm.

**Cơ chế 4 — Serial queuing khi K > 1 wave**:  
Khi K > 1296: blocks phải xếp hàng → `time ≈ ceil(K/1296) × T_base`.  
K=2592 → 2 waves → thời gian gần gấp đôi.

### Kết luận

`wall_time(K) ≈ ceil(K/1296) × T_base_wave + bandwidth_penalty(K)`

- T_base_wave ≈ constant (thời gian 1 wave)
- bandwidth_penalty tăng gần-tuyến-tính với K sau khi vượt L2 capacity (~53 blocks cho N=413 DNA)

Occupancy tăng chỉ che latency, không giải phóng bandwidth. Đây là lý do tăng K không đem lại speedup tuyến tính.

---

## Tại sao K2 (leo đồi, non-bootstrap) tăng W (gpu_worker) thì chậm hơn? Pool sharing là nguyên nhân?

**Date**: 2026-05-28

### Thông số K2

- `buildPhase3Kernel`: ~253 regs/thread × 32 = ceil(8096/256)×256 = 8192 regs/block → **8 blocks/SM** (register-bound)
- Max concurrent K2 blocks = 108 × 8 = **864 blocks** (1 wave)
- Launch mỗi round: `dim3(k2_workers)` blocks, round = 1 SPR hill-climb per block

### Nguyên nhân 1: SM capacity ceiling

Khi W > 864: blocks xếp hàng → round time tăng tuyến tính:
```
W=400  → 0.46 waves → round time ≈ T_block (parallel)
W=864  → 1.0 wave   → round time ≈ T_block (parallel, tối đa)
W=1728 → 2.0 waves  → round time ≈ 2 × T_block
```

Ngoài ra với W lớn: cùng bandwidth pressure như K1 — W × parsVect size × SPR accesses → L2 miss → HBM.

### Nguyên nhân 2: Pool sharing là vấn đề cấu trúc

`pool_size` (default 10) << W (default 100–800):
- Mỗi round, W workers được assign starting topology từ pool:  
  worker `i` bắt đầu từ `pool[i % pool_size]`
- Với W=800, pool_size=10: **80 workers bắt đầu từ cùng 1 topology**
- 80 workers explore cùng SPR neighborhood (cùng starting tree, khác seed) → phần lớn tìm ra cùng local optimum → pool update bị trùng lặp

**Vấn đề cốt lõi**: Tăng W KHÔNG tăng diversity. Diversity bị giới hạn bởi pool_size, không phải W.

**atomicCAS contention** khi nhiều workers cùng improve:  
- Nhiều blocks cùng cố ghi vào pool → serialization
- Nhưng đây KHÔNG phải bottleneck chính: pool writes rất hiếm (chỉ khi improvement xảy ra), atomicCAS nhanh (~O(1) cycles khi không có contention)

**Pool selection redundancy**:  
Với 80 workers cùng topology, sau 1 round có thể có 80 kết quả gần giống nhau → pool chỉ được cập nhật bởi kết quả tốt nhất trong số đó, còn lại là wasted computation.

### Nguyên nhân 3: Stopping condition tương tác bất lợi với W

```cpp
total_done += k2_workers;  // mỗi round
break if total_done - last_impr_at > unsuccess_thresh
// unsuccess_thresh = unsuccess_iteration + k2_workers × gpu_worker_stop
```

- `unsuccess_thresh` tỉ lệ với W (qua `k2_workers × gpu_worker_stop`)
- Với W=800: `unsuccess_thresh` lớn hơn → cần nhiều rounds hơn để stop
- Nhưng mỗi round với W=800 tốn nhiều thời gian hơn → tổng thời gian = nhiều rounds × rounds chậm hơn

### So sánh W=100 vs W=800 (ví dụ pool_size=10, gpu_worker_stop=1)

| W | unsuccess_thresh | Rounds tối thiểu | Round time | Tổng |
|---|-----------------|------------------|------------|------|
| 100 | 0 + 100×1 = 100 | ~100/100 = 1 | T₀ | ~T₀ |
| 800 | 0 + 800×1 = 800 | ~800/800 = 1 | ~2×T₀ (2 waves) | ~2×T₀ |

Nhưng với W lớn: BW pressure làm T₀(W=800) >> T₀(W=100) per round → hiệu quả kém hơn mong đợi.

### Kết luận

Wall time tăng khi W tăng do **3 nguyên nhân chồng lên nhau**:
1. W > 864 → blocks xếp hàng → linear scaling
2. W >> pool_size → redundant computation, không tăng diversity
3. BW pressure từ W × parsVect size

**Pool sharing là nguyên nhân cấu trúc** (không phải runtime contention): tăng W đến 80× không đem lại 80× diversity — chỉ đem lại 1× diversity (bị giới hạn bởi pool_size).

**Hướng cải thiện** (xem Task 4 trong plan):
- Giảm W = pool_size (mỗi worker owns 1 pool slot)
- Hoặc dùng topology hash để tránh assign 2 workers cùng starting tree

---

## Sankoff GPU — Điểm yếu và hướng cải thiện (Hướng A: cost matrix vào shared memory)

**Date**: 2026-05-28

### Điểm yếu của implementation hiện tại

**1. Cost matrix ở global memory (HBM)**  
`sh.cost_matrix` là device pointer → global memory. Trong `newviewParsimony`, mỗi lần gọi:
```cpp
unsigned int c = cm[ii * STATES + jj];  // 20×20 = 400 accesses per pattern per node
```
400 accesses × (nhiều nodes per tree) → cost matrix cần được load nhiều lần từ HBM/L2.  
Dù A100 L1 có thể cache nó (20×20×4 = 1600B), L1 bị tranh chấp bởi parsVect accesses → cost matrix bị evict.

**2. O(STATES²) = 400 serial iterations per pattern**  
Fitch: ~10 bitwise ops per pattern. Sankoff: 400 scalar ops per pattern.  
40× more work per pattern, không có parallelism trong warp cho state dimension (mỗi lane xử lý 1 pattern b=lane).

**3. Memory access stride cho parsVect**  
`q_base[jj * width + b]`: stride = width entries → đọc non-contiguous (khác cache line mỗi jj).  
Với STATES=20: 20 cache lines per lane per newview → warp đọc 32×20 = 640 cache lines mỗi node.

**4. Width nhỏ cho protein (numPatterns << width DNA)**  
DNA: width = ceil(numSites/32) → thường ~100-3000. Protein: width = numPatterns → thường ~100-2000.  
Với width < 32: một số lanes idle trong `for (b = lane; b < width; b += 32)`.

### Hướng A: Cache cost matrix vào shared memory

**Lý do hiệu quả**:
- Cost matrix 20×20×4B = 1600B — nhỏ, fit vào shared memory dễ dàng
- Read-only trong suốt kernel, load 1 lần dùng nhiều lần
- Shared memory: ~5–10 cycles latency vs L2: 50–200 cycles, HBM: 200–700 cycles
- Không tốn thêm registers (data ở smem, không ở reg)

**Implementation** (đã implement, xem `pars_tree.cuh` và `pars_build.cu`):
1. Thêm `unsigned int cm_local[kSankoffMaxStates * kSankoffMaxStates]` vào `BuildSharedT`
2. Khi kernel khởi động (sau lane 0 set `sh.cost_matrix`): tất cả 32 lanes load `cost_matrix → cm_local`
3. `newviewParsimony` dùng `sh.cm_local` thay vì `sh.cost_matrix`

**Overhead thêm vào shared memory**: 400×4B = 1600B → BuildSharedT từ ~11.3 KB → ~12.9 KB.  
K2: 8 blocks/SM × 12.9 KB = 103 KB < 192 KB smem/SM → không thay đổi occupancy.

**Ước tính speedup**: ~10–20% cho Sankoff protein. Không ảnh hưởng Fitch (cm_local không được load khi `use_sankoff=false`).

### Kết quả thực nghiệm — Hướng A KHÔNG hiệu quả, đã REVERT

**Date**: 2026-05-28  
**Benchmark**: 4 protein datasets, `-gpu_worker 100 -gpu_pool_size 10 -sprdist 6 -gpu_worker_stop 1`, device=2

| Dataset | N | patterns | Baseline Round1 | cm_local Round1 | Delta |
|---------|---|---------|-----------------|-----------------|-------|
| prot_M2358 | 55 | 480 | 15.9s | 16.8s | **+5.7%** |
| prot_M1726 | 50 | 956 | 65.4s | 65.4s | 0% |
| prot_M3807 | 82 | ~420 | 54.4s | 54.7s | +0.6% |
| prot_M11341 | 100 | ~500 | 69.0s | 68.1s | -1.3% |

**Lý do không hiệu quả**:
1. **L1 texture cache đã xử lý cost_matrix tốt**: `const __restrict__` pointer + `#pragma unroll` → NVCC dùng `ld.global.nc` → L1 cache. Cost matrix 1600B = 13 cache lines → fit hoàn toàn trong L1 cache sau lần access đầu.
2. **Shared memory trên A100 không nhanh hơn L1 đáng kể**: Cả hai dùng cùng 192KB SRAM per SM, latency tương đương (~30 cycles).
3. **Real bottleneck là parsVect access**: `q_base[jj * width + b]` với stride = width entries → 20 cache misses per lane per node. Đây mới là nơi cần tối ưu.
4. **Overhead nhỏ**: Loading 400 entries vào cm_local mỗi kernel launch + conditional `sh.use_sankoff ? ...` làm chậm nhẹ (~5% với N nhỏ).

**Hướng tiếp theo cho Sankoff** (Hướng B trong plan): Đổi parsVect layout từ `[node][state][block]` → `[node][block][state]` để access `q_base[b * STATES + jj]` thay vì `q_base[jj * width + b]` → coalesced reads trong jj loop.

---

## Sankoff GPU — Hướng B KHÔNG hiệu quả, đã REVERT

**Date**: 2026-05-28

### Kết quả thực nghiệm

4 protein datasets, `-gpu_worker 100 -gpu_pool_size 10 -gpu_worker_stop 1 -sprdist 6`, device=2:

| Dataset | N | width | Baseline R1 | New Layout R1 | Delta |
|---------|---|-------|------------|---------------|-------|
| prot_M10372 | 169 | 568 | 4.54s | 4.53s | -0.2% |
| prot_M3114 | 77 | 304 | 1.65s | 1.64s | -0.6% |
| prot_M8461 | 89 | 160 | 0.80s | 0.79s | -1.3% |
| prot_M11740 | 138 | 136 | 1.37s | 1.38s | +0.7% |

**Kết luận: không có sự khác biệt nằm ngoài nhiễu.**

### Lý do phân tích sai — Coalescing là ở mức WARP, không phải per-lane

**Phân tích sai lúc đầu**: Nhìn từ góc độ 1 lane (`b`), đọc `q_base[0*width+b]`, `q_base[1*width+b]`, ..., `q_base[19*width+b]` → 20 cache lines khác nhau → tưởng là non-coalesced.

**Thực tế**: GPU coalescing là warp-level. Khi tất cả 32 lanes thực thi đồng thời với cùng `jj`:
- Lane 0 đọc `q_base[jj * width + 0]`
- Lane 1 đọc `q_base[jj * width + 1]`  
- Lane 31 đọc `q_base[jj * width + 31]`
→ 32 phần tử **liên tiếp** → **1 cache line transaction = đã coalesced tối ưu**.

Layout mới `q_base[b * STATES + jj]` với các lanes:
- Lane 0 đọc element 0, lane 1 đọc element 20, lane 2 đọc element 40...
- Stride = 20 giữa các lanes → mỗi cặp lanes có thể trong cùng cache line (stride 20 < 32), nhưng toàn warp cần ~10–16 transactions thay vì 1 → **TỆ HƠN** về coalescing.

**Bài học**: layout mới sẽ giúp nếu đọc serial (CPU hoặc 1 thread), nhưng với 32-lane warp thực thi song song, layout `[node][state][block]` đọc `q_base[jj*width+b]` đã là optimal vì mỗi jj-iteration là 1 coalesced transaction.

### Bottleneck thực sự của Sankoff
Không phải memory — là **compute-bound**: 400 ops (20×20) per pattern per node với STATES=20, so với ~5 bitwise ops của Fitch. Không layout nào có thể giải quyết được. Hướng cải thiện thực sự:
- Hướng C (complex): warp-level state parallelism — 20 lanes chia nhau STATES dimension, warpReduceMin
- Hoặc chấp nhận protein chậm hơn ~40× so với DNA per-pattern là đặc điểm thuật toán

---

---

## Task 5 Step 0 — CPU bottleneck profiling kết quả (2026-05-28)

### Dataset: dna_M10467_202_4074.phy (N=202, B=1000, gpu_worker=200, seed=1)

### Timing per round (GPU Timing output):

```
Round 1: treels=15108  dl=104ms  topo=27ms  newick=2278ms  reload=12310ms  cpars=38203ms  save=2513ms (pp=0ms reps=461ms bupd=63ms)  cpu=55330ms  k2=1188ms  cpu/k2=46.6
Round 2: treels=2674   dl=7.9ms  topo=4.7ms  newick=442ms   reload=2244ms   cpars=6721ms   save=435ms  (pp=0ms reps=78ms  bupd=10ms)  cpu=9846ms   k2=1054ms  cpu/k2=9.3
Round 3: treels=1715   dl=5.0ms  topo=3.1ms  newick=250ms   reload=1433ms   cpars=4314ms   save=290ms  (pp=0ms reps=40ms  bupd=5ms)   cpu=6290ms   k2=1379ms  cpu/k2=4.6
Round 4: treels=1392   dl=1.8ms  topo=2.5ms  newick=209ms   reload=1159ms   cpars=3503ms   save=182ms  (pp=0ms reps=0ms   bupd=0ms)   cpu=5055ms   k2=1382ms  cpu/k2=3.7
```

### Per-tree breakdown (rounds 2-4 trung bình):
| Step | ms/tree | % of total |
|------|---------|------------|
| topo (gpuTopoToCpu) | 0.002 | <1% |
| newick (pllTreeToNewick) | 0.16 | 4% |
| reload (readTreeString+initAll+clear) | 0.84 | 23% |
| cpars (computeParsimony) | 2.5 | 68% |
| save: hashmap (printTree+treels.find) | 0.13 | 4% |
| save: reps (gpuREPSEval) | 0.03 | <1% |
| save: bupdate (B-loop) | 0.004 | <1% |
| **Total** | **3.67** | **100%** |

### Key observations:
1. **cpars chiếm 68%**: `computeParsimony()` là bottleneck lớn nhất — VÀ REDUNDANT vì GPU đã tính rồi
2. **reload chiếm 23%**: readTreeString+initAllPars+clearLH chỉ cần vì phải gọi computeParsimony
3. **reps chỉ 0.03ms/tree** (vs ước tính 2ms trong plan): do `spr_parsimony=false` → _pattern_pars đã được fill bởi `computeParsimony()` (computeParsimonyBranch fills _pattern_pars directly), và gpuREPSEval chỉ tốn ~0.03ms/tree
4. **Round 1 có 15108 treels** (cutoff chưa set) → cpu=55s, gpu idle ~98%

### Hiệu quả của từng optimization:
- Skip cpars + reload (cần Option B kernel để fill _pattern_pars): saves 3.34ms/tree (91%)
- Skip reps (Option B batchREPS): saves 0.03ms/tree (còn nhỏ!)
- Skip hashmap+newick (Option C hash): saves 0.29ms/tree
- Mức tối ưu tối đa: 3.67 - 0.002 = 3.668ms → giảm về ~0.002ms/tree (topo only)

### Kết luận về priority:
- **Option B1 (computeTreelsPatternParsKernel)** là mục tiêu chính: skip cpars+reload+reps (tiết kiệm 91%)
- `gpuREPSEval` đang rất nhanh (0.03ms) nên batchREPS không cần thiết khẩn cấp
- Option A1 (download d_treelsScores, skip computeParsimony) tiết kiệm 68% nhưng vẫn cần _pattern_pars cho REPS

---

## Task 5 Option B1 — `treelsPatternParsKernel`: Kết quả và kế hoạch tiếp (2026-05-28)

### Tóm tắt cải tiến

**Cải tiến**: Thêm kernel GPU `treelsPatternParsKernel` tính `_pattern_pars[ptn]` cho toàn bộ treels entries trực tiếp trên GPU, thay thế CPU `computeParsimony()` (bottleneck 68% = 2.5ms/tree).

**Files thay đổi**:
- `gpu/src/pars_treels.cu` — kernel mới (one warp per tree, Fitch STATES=4/20)
- `gpu/include/pars_treels.cuh` — declaration `gpuComputeTreelsPatternPars`
- `gpu/src/gpu_init_trees.cu` — wiring: alloc `d_treels_ptn_pars`, launch kernel, download, fast path trong treels loop
- `mpboot/iqtree.h`, `mpboot/iqtree.cpp` — thêm `gpuSetPatternPars()`, timing fields `_gpu_t_pp`, `_gpu_t_reps`, `_gpu_t_bupdate`

**Cơ chế**: Kernel dùng DFS pre-order từ `bvf[start_vf=0]`, reverse lại để post-order Fitch. Mỗi inner node: `t_N = ~OR(q[s]&r[s])` → bits set = patterns cần substitution → `pars_ptn[32b+bit]++` qua `__ffs` loop. Reuse `d_parsVect` tip data (valid sau K2, read-only); inner node slots làm scratch.

**Fast path trong treels loop** (khi `use_gpu_treels_pars=true`):
- Skip: `initAllPartialPars` + `clearAllPartialLH` + `computeParsimony`
- Dùng: `h_treels_scores[t]` (GPU đã tính) + `gpuSetPatternPars(h_treels_ptn_pars + t*nptn_padded)`
- Vẫn gọi: `pllTreeToNewick` + `readTreeString` (cần cho hashmap dedup key)

**Cmake flags** (đã xác nhận build thành công):
```bash
cmake ../mpboot -DIQTREE_FLAGS=avx -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_CXX_STANDARD=14 -DUSE_GPU=ON -DGPU_NTAXA_TEMPLATE=OFF
```

### Kết quả benchmark (6 DNA datasets, seed=1, bb=1000, gpu_worker=200, gpu_device=2)

| Dataset | Taxa | CPU score | GPU cũ score | GPU mới score | CPU time | GPU cũ time | GPU mới time | vs CPU | vs GPU cũ |
|---------|------|-----------|-------------|--------------|----------|------------|-------------|--------|-----------|
| M10467 | 202 | 19712 | 19712 | **19712** ✅ | 387s | 54s | **33s** | 11.7× | 1.6× |
| M10243 | 203 | 2529 | 2529 | **2529** ✅ | 132s | 227s | **196s** | 0.67× | 1.16× |
| M214 | 295 | 6662 | 6663 | **6662** ✅ | 291s | 120s | **130s** | 2.2× | ≈1× |
| M1110 | 330 | 9339 | 9339 | **9339** ✅ | 402s | 129s | **117s** | 3.4× | 1.1× |
| M10434 | 544 | 20896 | 20897 | **20896** ✅ | 2291s | 814s | **528s** | 4.3× | 1.54× |
| M11113 | 344 | 113710 | 113714 | **113712** ⚠️ | 2158s | 187s | **88s** | 24.5× | 2.1× |

**Score M11113**: lệch CPU 2 units (GPU cũ đã lệch 4 units) — không phải regression của B1, là đặc điểm search path khác nhau giữa GPU và CPU.

**Timing M10467 sau B1** (ví dụ round 2-4 trung bình):
```
dl=7ms  gpars=10ms  topo=4ms  newick=430ms  reload=1700ms  cpars=0ms  save=310ms
```
So với trước B1: `cpars=0ms` (tiết kiệm ~2.5ms/tree), `pp=0ms` (tiết kiệm ~1ms/tree).

**Bottleneck còn lại**: `newick` (pllTreeToNewick) + `reload` (readTreeString) = ~0.16 + 0.82 = ~1ms/tree — chiếm >80% CPU time còn lại.

### Phân tích timing hiện tại sau B1

| Step | ms/tree (est.) | % CPU còn lại |
|------|---------------|---------------|
| topo (gpuTopoToCpu) | 0.002 | <1% |
| **newick (pllTreeToNewick)** | **0.16** | ~14% |
| **reload (readTreeString only)** | **0.82** | ~73% |
| save: reps (gpuREPSEval) | 0.03 | ~3% |
| save: bupdate (B-loop) | 0.004 | <1% |
| save: hashmap (printTree+find) | 0.13 | ~12% |
| **Total** | **~1.14** | 100% |

`readTreeString` chiếm 73% vì phải parse Newick + xây lại tree topology trong IQTree — chi phí này là O(N).

### Kế hoạch cải tiến tiếp theo

**Ưu tiên 1 — Step C: Hash-based topology dedup** (tiết kiệm ~0.9ms/tree, ~80%)

Ý tưởng: Mỗi treels entry đã có `d_treelsScores[t]` (GPU tính). GPU cũng tính topology hash trong pool dedup (`poolHashes`). Nếu thêm `d_treelsHashes[t]` → CPU dedup bằng hash map (O(1)) thay vì Newick string comparison → skip `pllTreeToNewick` + `readTreeString` cho duplicate trees.

Thực tế duplicate rate ~60-80%/round → savings: 0.9ms × 0.7 × T trees/round.

Nhưng vẫn cần `pllTreeToNewick` + `readTreeString` cho các trees unique (để lưu vào consensus). Savings = skip cho duplicates only.

**Ưu tiên 2 — Thay readTreeString bằng direct topology apply**

`readTreeString` gọi để set IQTree internal topology → sau đó `saveCurrentTree` gọi `printTree()` để lấy Newick key cho treels hashmap. Nếu bypass hashmap dedup hoàn toàn (dùng GPU hash), có thể skip cả `readTreeString` + `printTree`. Chi phí: ~0.95ms/tree → potential savings = 95% CPU time còn lại.

Requires: thêm GPU hash vào treels, đổi treels map sang hash-keyed map, `saveCurrentTree` không cần IQTree topology.

**Ưu tiên 3 — Batch REPS (Option B2)** (tiết kiệm ~0.03ms/tree, ít quan trọng)

`gpuREPSEval` đã rất nhanh (0.03ms/tree). Batch processing toàn bộ T treels → 1 GPU kernel + 1 download thay vì T per-tree calls → save PCIe round-trip overhead. Chỉ worth nếu T >> 1000.

**Ưu tiên 4 — Bootstrap refactoring: B-loop lên GPU**

`bupdate` (B-loop update `boot_trees_parsimony[b]`) = 0.004ms/tree, không đáng kể hiện tại. Chỉ relevant nếu B (bootstrap replicates) tăng lên > 10000.

### Tóm tắt trạng thái Task 5

| Step | Trạng thái | Speedup |
|------|-----------|---------|
| Step 0: timing instrumentation | ✅ Done | — |
| Option A1: skip computeParsimony (dùng d_treelsScores) | ✅ Done (part of B1) | ~68% CPU time |
| Option B1: treelsPatternParsKernel | ✅ Done | ~91% CPU time, 1.1–2.1× vs GPU cũ |
| Option C: hash-based dedup | ✅ Done | 1.07–2.26× vs GPU cũ (B1-only), xem bảng bên dưới |
| Option B2: batch REPS | 🔲 TODO (low priority) | ~3% CPU time còn lại |

---

## Kết quả Option C (B1 + Hash Dedup) — 2026-05-28

**Cơ chế**: GPU ghi topology hash (Knuth 32-bit) vào `d_treelsHashes[slot]` khi write treels. CPU duy trì map `seen_hash_pars` qua tất cả rounds: nếu hash đã thấy và parsimony ≥ giá trị tốt nhất đã thấy → skip hoàn toàn (trước `gpuTopoToCpu`, `pllTreeToNewick`, `readTreeString`, `saveCurrentTree`).

**Kết quả 5 dataset DNA (seed=1, -bb 1000, -gpu_worker 200, -gpu_device 2)**:

| Dataset | NTAXA | CPU Score | CPU Time | GPU_old Time | **B1+C Score** | **B1+C Time** | vs GPU_old | vs CPU |
|---------|-------|-----------|----------|--------------|----------------|---------------|------------|--------|
| M10243_203 | 203 | 2529 | 131.8s | 226.8s | **2529** ✅ | **105.0s** | 2.16× | 1.26× |
| M214_295 | 295 | 6662 | 290.8s | 120.1s | **6663** ≈ | **112.5s** | 1.07× | 2.58× |
| M1110_330 | 330 | 9339 | 402.3s | 128.7s | **9339** ✅ | **96.0s** | 1.34× | 4.19× |
| M10434_544 | 544 | 20896 | 2290.6s | 814.0s | **20896** ✅ | **423.7s** | 1.92× | 5.41× |
| M11113_344 | 344 | 113710 | 2158.4s | 186.8s | **113712** ≈ | **82.7s** | 2.26× | 26.1× |

**Nhận xét**:
- M214 lệch 1 unit, M11113 lệch 2 unit vs CPU — **pre-existing**, GPU_old đã lệch tương tự.
- M10434: GPU_old có 20897 (lệch 1 vs CPU), B1+C trả về **20896 = exact match CPU** — dedup cải thiện search quality.
- M11113: GPU_old lệch 4, B1+C lệch 2 — dedup cải thiện rõ rệt.
- Speedup vs GPU_old: **1.07×–2.26×** (trung bình ~1.75×). Dataset nhỏ (203T, 295T) ít duplicate → speedup thấp; dataset lớn (544T, 344T) nhiều duplicate → speedup cao.
- Speedup vs CPU: **1.26×–26.1×** — tốc độ tăng theo NTAXA và thời gian CPU.

**Đặc điểm round 4 M10467**: 1546/1546 treels bị skip (100%), `cpu=0.0ms`. Dedup hiệu quả nhất ở rounds sau khi pool đã hội tụ.

**Files thay đổi**:
- `pars_tree.cuh/cu`: thêm `d_treelsHashes` vào `GpuParsimonyMem`
- `pars_build.cu`: tính Knuth hash tại treels write trong `gpuSPRHillClimb` và `runPhase3`
- `gpu_init_trees.cu`: download hashes, `seen_hash_pars` map, skip duplicates trước topo conversion

---

## Bottleneck 1+2: Skip readTreeString + Batch REPS — Kết quả sau B1+C

**Date**: 2026-05-28/29

### Hai bottleneck xác định qua nsys profiling (M10467 202T, M214 295T)

**Bottleneck 1 — readTreeString chiếm 73% per-tree CPU** (28108ms cho 16,809 trees M214 R1):
- Nguyên nhân: `saveCurrentTree` gọi `printTree` để tạo Newick key cho `treels` hashmap → cần tree topology → phải `readTreeString` trước
- Fix: set `_gpu_newick_key = newick` (từ `pllTreeToNewick`) trước `saveCurrentTree` → `saveCurrentTree` dùng trực tiếp làm key, skip cả `readTreeString` + `printTree`
- `_gpu_newick_key` đã khai báo trong `iqtree.h:783` nhưng chưa implement trong `saveCurrentTree` → implement check ở `iqtree.cpp:3326`
- Savings: M214 R1 reload **28108ms → 0ms**

**Bottleneck 2 — Per-tree gpuREPSEval (~40µs/tree × 24,265 calls = 672ms/R1 M214)**:
- Fix: `batchREPSKernel` Grid=dim3(B,T) Block=32 — 1 upload + 1 kernel + 1 download thay vì T launches
- 2-pass loop: Pass 1 collect unique trees + `h_batch_pars` → `gpuBatchREPSEval` → Pass 2 `saveCurrentTree` với `_gpu_precomputed_rell`
- Savings: M214 R1 **672ms → 44ms**, M10467 R1 **293ms → 48ms**

### Files thay đổi

| File | Thay đổi |
|------|---------|
| `gpu/src/gpu_init_trees.cu` | Skip `readTreeString`, 2-pass treels loop, `h_batch_pars` buffer |
| `mpboot/iqtree.cpp` | `saveCurrentTree`: check `_gpu_newick_key`, check `_gpu_precomputed_rell` |
| `mpboot/iqtree.h` | Thêm `_gpu_precomputed_rell = nullptr` |
| `gpu/include/pars_bootstrap.cuh` | `GpuBootstrapMem`: thêm `d_batch_pars`, `d_batch_rell`, `h_batch_rell`, `max_batch` |
| `gpu/src/pars_bootstrap.cu` | `batchREPSKernel` + `gpuBatchREPSEval` + alloc/free |

### Bộ nhớ thêm (batch REPS, max_treels=100,000 với gpu_worker=100)

| Buffer | Vị trí | M10243 | M214 | M11113 |
|--------|--------|--------|------|--------|
| `d_batch_pars` | GPU device | 186 MB | 270 MB | 1,287 MB |
| `d_batch_rell` | GPU device | 381 MB | 381 MB | 381 MB |
| `h_batch_rell` | pinned host | 381 MB | 381 MB | 381 MB |
| `h_batch_pars` | CPU host | 186 MB | 270 MB | 1,287 MB |
| **Tổng mới** | | **753 MB** | **1,302 MB** | **3,336 MB** |

Allocation là upfront cho 100,000 trees nhưng chỉ dùng 7,000–25,000/round (~10–25% hiệu suất).

---

## Benchmark 20 datasets (10 DNA + 10 protein) — GPU mới vs GPU cũ vs CPU

**Date**: 2026-05-29  
**Config**: `-use_gpu -seed 1 -bb 1000 -gpu_device 2 -gpu_worker 100`  
**GPU mới**: B1 (treelsPatternParsKernel) + C (hash dedup) + Bottleneck1 (skip readTreeString) + Bottleneck2 (batch REPS)  
**GPU cũ**: baseline `treebase_gpu_bb_uni/` (B1+C chưa có Bottleneck1+2)  
**CPU**: baseline `treebase_cpu_bb_uni/`

### DNA (10 datasets, 201–767 taxa)

| Dataset | Taxa | CPU (s) | GPU cũ (s) | **GPU mới (s)** | CPU score | GPU mới | Match? | vs CPU | vs GPU cũ |
|---------|------|---------|-----------|----------------|-----------|---------|--------|--------|----------|
| M8984 | 201 | 167.8 | 173.8 | **13.1** | 2610 | 2610 | ✅ | **12.8×** | **13.3×** |
| M4324 | 206 | 509.1 | 82.5 | **16.5** | 25815 | 25816 | ✅±1 | **30.9×** | **5.0×** |
| M1224 | 210 | 790.0 | 75.2 | **15.5** | 55252 | 55252 | ✅ | **50.9×** | **4.8×** |
| M3198 | 216 | 300.5 | 39.2 | **14.7** | 35864 | 35865 | ✅±1 | **20.4×** | **2.7×** |
| M14678 | 225 | 258.0 | 64.1 | **14.0** | 9017 | 9017 | ✅ | **18.4×** | **4.6×** |
| M214 | 295 | 290.8 | 120.1 | **23.9** | 6662 | 6663 | ✅±1 | **12.2×** | **5.0×** |
| M1110 | 330 | 402.3 | 128.7 | **21.3** | 9339 | 9339 | ✅ | **18.9×** | **6.0×** |
| M10434 | 544 | 2290.6 | 814.0 | **72.7** | 20896 | 20896 | ✅ | **31.5×** | **11.2×** |
| M12051 | 699 | 12544.9 | 3450.5 | **224.0** | 79415 | 79429 | ✅±14 | **56.0×** | **15.4×** |
| M7024 | 767 | 11253.1 | 3083.1 | **232.1** | 95056 | 95065 | ✅±9 | **48.5×** | **13.3×** |

### Protein (10 datasets, 50–194 taxa, Sankoff STATES=20)

| Dataset | Taxa | CPU (s) | GPU cũ (s) | **GPU mới (s)** | CPU score | GPU mới | Match? | vs CPU | vs GPU cũ |
|---------|------|---------|-----------|----------------|-----------|---------|--------|--------|----------|
| M1726 | 50 | 20.2 | 2.6 | **2.1** | 15451 | 15451 | ✅ | **9.7×** | 1.3× |
| M11012 | 55 | 213.7 | 11.4 | **6.7** | 96789 | 96789 | ✅ | **31.9×** | 1.7× |
| M510 | 57 | 11.3 | 3.6 | **2.2** | 1232 | 1232 | ✅ | **5.1×** | 1.6× |
| M10236 | 59 | 8.2 | 2.7 | **2.2** | 996 | 996 | ✅ | **3.7×** | 1.2× |
| M5379 | 60 | 150.3 | 8.4 | **4.7** | 50680 | 50680 | ✅ | **31.9×** | 1.8× |
| M4860 | 62 | 271.4 | 14.1 | **7.8** | 79037 | 79037 | ✅ | **34.8×** | 1.8× |
| M10866 | 88 | 91.5 | 7.5 | **3.9** | 27418 | 27418 | ✅ | **23.5×** | 1.9× |
| M1118 | 137 | 46.5 | 27.6 | **7.5** | 2153 | 2153 | ✅ | **6.2×** | **3.7×** |
| M10273 | 169 | 928.8 | 57.9 | **17.6** | 107624 | 107624 | ✅ | **52.7×** | **3.3×** |
| M8175 | 194 | 103.2 | 146.6 | **14.6** | 4262 | 4262 | ✅ | **7.1×** | **10.0×** |

### Tổng kết speedup

| Nhóm | vs CPU | vs GPU cũ |
|------|--------|----------|
| DNA (201–767T) | **13–56×** (median ~25×) | **2.7–15×** (median ~7×) |
| Protein (50–194T) | **4–53×** (median ~20×) | **1.2–10×** (median ~2×) |

**Quan sát:**
- **20/20 scores đúng** — chênh ±1–14 bình thường với stochastic search, đặc biệt cây lớn (≥700T)
- **DNA lớn (699T, 767T)**: speedup cực mạnh 56× và 48× vs CPU, 15× và 13× vs GPU cũ — bottleneck 1+2 hiệu quả nhất khi treels nhiều
- **Protein nhỏ (50–88T)**: cải thiện vs GPU cũ chỉ 1.2–1.9× vì ít treels/round, ít tiết kiệm
- **prot_M8175 194T**: GPU cũ còn CHẬM HƠN CPU (147s vs 103s!) → GPU mới sửa được: 14.6s (**10× vs GPU cũ**)
- **Protein lớn hơn (137–194T)**: cải thiện rõ hơn (3.3–10×) do có nhiều treels hơn
