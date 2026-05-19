# Notes — GPU Parsimony Analysis

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
