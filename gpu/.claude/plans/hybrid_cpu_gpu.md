# Hybrid CPU-GPU Parsimony Tree Building

## Context

**Vấn đề**: CPU hoàn toàn rảnh trong khi GPU chạy kernel 1 (stepwise + SPR, ~10-40s). Tận dụng CPU để build cây song song → tăng diversity của candidate set → cải thiện quality.

**Mục tiêu**: CPU build thêm cây trong khi GPU chạy kernel 1, kết hợp kết quả để chọn top trees cho Phase 3, CPU và GPU chạy Phase 3 song song.

**Code hiện tại** (two-kernel Opt-G2):
1. Kernel 1: `buildParsimonyTreesKernel` (Phase 0-2, stepwise + SPR) — K trees, đồng bộ
2. Download `d_postSprScores[K]` → tính threshold
3. Kernel 2: `buildPhase3Kernel` (Phase 3) — chỉ top X% trees

---

## Architecture mới — 5 bước

### Bước 1: CPU + GPU Kernel 1 song song
- Launch Kernel 1 **asynchronously** (không sync ngay)
- CPU main thread: clone `pllInst` thành N_cpu bản, chạy `_pllMakeParsimonyTreeFast` trong `std::thread`
- Dùng `std::atomic<bool> cpu_stop` để CPU dừng khi GPU xong
- CPU lưu kết quả: `vector<pair<string,unsigned int>> cpu_trees` (Newick + postSprScore)

```
Launch Kernel 1 (async) ─────────────────────────────┐ GPU
CPU build loop: while(!cpu_stop) { makeParsTree → cpu_trees.push_back } │
cudaStreamSynchronize(stream) ← GPU done, set cpu_stop=true ────────────┘
```

### Bước 2: Tổng hợp và tính threshold
- Download `d_postSprScores[K]` từ GPU
- Gộp `K + M` scores (GPU + CPU), sort → `top_k = ceil((K+M) * top_pct)`
- `threshold = sorted_scores[top_k - 1]`

```cpp
// K GPU scores + M CPU scores → combined sort
vector<unsigned int> all_scores(gpu_scores.begin(), gpu_scores.end());
for (auto& [nwk, score] : cpu_trees) all_scores.push_back(score);
sort(all_scores.begin(), all_scores.end());
unsigned int threshold = all_scores[min(top_k, all_scores.size()) - 1];
```

### Bước 3a: Upload CPU trees → thay thế GPU slots dưới threshold

CPU trees có score ≤ threshold sẽ thay thế GPU trees có score > threshold:
- Tìm GPU slots: `replace_slots` = indices k với `gpu_scores[k] > threshold`, sort descending (worst first)
- Với mỗi CPU tree đủ điều kiện (up to `replace_slots.size()`):
  1. `cpuToGpuTopology(cpu_pllInst, &h_topo)` — convert topology sang GPU format
  2. Set `h_topo.postSprParsimony = cpu_score`
  3. Set `h_topo.savedSeed = seed_for_this_tree`
  4. Set `h_topo.needs_recompute = 1` (flag mới trong GpuTopology)
  5. `uploadTopology(mem, replace_slot_k, &h_topo, stream)` — upload lên GPU
- **KHÔNG cần kernel riêng**: modify `buildPhase3Kernel` để xử lý `needs_recompute`:

```cpp
// Trong buildPhase3Kernel, sau threshold check:
if (topo->needs_recompute) {
    // Recompute toàn bộ parsVect từ topology mới (CPU tree vừa upload)
    createTiAndEvaluateParsimony<SharedT,STATES>(
        pars_tree, score_tree, topo, sh, topo->start_vface, N, /*full=*/true, width
    );
    if (lane == 0) { topo->needs_recompute = 0; sh.randomMP = ...; }
    __syncwarp();
}
```

**Lưu ý**: `needs_recompute` (int, 1 field × 4 bytes) thêm vào GpuTopology trong `pars_tree.cuh`.

### Bước 3b: Xây dựng candidateSet
- Tất cả trees (CPU + GPU) có score ≤ threshold → push vào `vector<string> candidateSet`
- GPU trees: download topology → Newick (có thể defer đến sau Phase 3)
- CPU trees: đã có Newick string

### Bước 4: GPU Kernel 2 + CPU hill-climbing song song

```
buildPhase3Kernel<<<K,32,0,stream>>>() ──────────────────────────────── GPU async
CPU main thread (while GPU runs):
  for each tree in candidateSet (top cpu_trees):
    iqtree.readTreeString(tree_str)      ← load từ treeString
    _pllComputeRandomizedStepwiseAdditionParsimonyTree() hoặc SPR loop
    ghi lại best result
cudaStreamSynchronize(stream) ─────────────────────────────────────────  GPU done
```

**CPU hill-climbing** (phyloanalysis.cpp:1867, iqtree.cpp:1585):
- Input: treeString của top cpu_trees
- Dùng: `iqtree.readTreeString(tree_str)` + parsimony SPR (`rearrangeParsimony` loop)
- Output: updated bestParsimony + bestTreeString
- Chạy trên CPU main thread trong khi GPU Phase 3 chạy async → CPU không bị lãng phí

**Lưu ý**: `iqtree.readTreeString` + SPR không cần GPU → hoàn toàn song song với GPU Phase 3.

### Bước 5: So sánh và trả về kết quả
- Download GPU best tree (topology + bestParsimony)
- So sánh vs CPU best tree
- Return winner vào `gpuTrees[k]` vector

---

## Implementation Details

### Files cần sửa

| File | Thay đổi |
|------|---------|
| `gpu/include/pars_tree.cuh` | Thêm `int needs_recompute` vào GpuTopology |
| `gpu/src/pars_build.cu` | Thêm `recomputeParsKernel`; sửa `gpuStepwiseBuildTrees` |
| `gpu/src/gpu_init_trees.cu` | Pass thêm data cho CPU tree building |

### Kernel mới: recomputeParsKernel

```cpp
template<int STATES, int NTAXA>
__global__ void recomputeParsKernel(
    parsimonyNumber* d_parsVect,
    unsigned int* d_parsScore,
    GpuTopology* d_topos,
    size_t parsVectPerTree,
    size_t parsScorePerTree
) {
    int k = blockIdx.x;
    GpuTopology* topo = d_topos + k;
    if (!topo->needs_recompute) return;  // skip if not replaced
    
    // Full recompute of parsVect from scratch for this tree
    createTiAndEvaluateParsimony<SharedT, STATES>(
        pars_tree, score_tree, topo, sh, topo->start_vface, N, /*full=*/true, width
    );
    topo->postSprParsimony = sh.randomMP;  // update score
    topo->needs_recompute = 0;
}
```

### Sửa gpuStepwiseBuildTrees (pars_build.cu)

```cpp
void gpuStepwiseBuildTrees(..., pllInstance* cpu_tr, partitionList* cpu_pr, ...) {
    // 1. Launch Kernel 1 async
    buildParsimonyTreesKernel<<<K, 32, 0, stream>>>(...);
    // (NO sync here yet)

    // 2. CPU builds trees while GPU runs kernel 1
    std::atomic<bool> cpu_stop{false};
    vector<pair<string,unsigned int>> cpu_trees;
    std::mutex cpu_mutex;
    
    std::thread cpu_thread([&]() {
        while (!cpu_stop.load()) {
            // clone tr, run makeParsimonyTreeFast, get score + Newick
            // push to cpu_trees (under lock)
        }
    });
    
    // 3. GPU kernel 1 sync
    cudaStreamSynchronize(stream);
    cpu_stop = true;
    cpu_thread.join();
    
    // 4. Download GPU scores + combine threshold
    cudaMemcpy(gpu_scores.data(), mem->d_postSprScores, K * sizeof(uint32_t), ...);
    // ... combine, sort, threshold ...
    
    // 5. Upload CPU trees to replace slots (if needed)
    // ... cpuToGpuTopology, uploadTopology ...
    
    // 6. Recompute parsVect for replaced slots
    if (any_replaced) recomputeParsKernel<<<K,32,0,stream>>>(...);
    cudaStreamSynchronize(stream);
    
    // 7. GPU Kernel 2 + CPU hill-climbing in parallel
    buildPhase3Kernel<<<K,32,0,stream>>>(... threshold ...);
    // CPU hill-climbing on top cpu_trees (in thread)
    
    cudaStreamSynchronize(stream);
    cpu_hc_thread.join();
    
    // 8. Return best result
}
```

---

## Quyết định thiết kế đã xác nhận

1. **N_cpu = tự động**: CPU build liên tục đến khi GPU kernel 1 xong (dùng `std::atomic<bool> cpu_stop`)
2. **CPU hill-climbing**: dùng `iqtree.readTreeString(tree_str)` + parsimony SPR loop → chỉ cần treeString
3. **Thread model**: `std::thread` cho CPU building (step 1); CPU hill-climbing (step 4) chạy trên main thread (GPU là async → không cần thread riêng)
4. **Replacement strategy**: thay bao nhiêu có thể — `min(cpu_qualifying, gpu_replace_slots)` slots

---

## Verification

```bash
# Build
make -j4 2>&1 | grep -E "error:|Built"

# Test smoke: numpars=50, sprdist=3 (kernel 1 đủ nhanh để test)
./mpboot-avx -s ../data_treebase/dna_M214_295_1836.phy \
    -use_gpu -seed 1 -numpars 50 -sprdist 3 -gpu_stop 4 \
    2>&1 | grep -E "CPU|GPU|hybrid|trees|ms/tree|Current best"

# Expect: log line showing CPU built X trees, GPU built K trees, combined threshold
# Expect: final quality >= current GPU-only quality
```
