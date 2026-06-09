# Bootstrap trong MPBoot

---

## Flow tổng thể — Từ alignment đến `.contree`

### Điểm khác biệt cốt lõi vs Standard Bootstrap

**Standard bootstrap** (IQ-TREE, RAxML): với mỗi trong B replicates → build 1 cây mới từ đầu (O(B × tree_search)).

**MPBoot**: build K candidate trees **một lần** từ alignment gốc, rồi **tái dùng** để vote cho từng replicate bằng RELL approximation (O(tree_search) + O(B × P) dot products).

### Phase 1 — Build candidate trees (K1 kernel)

```
GPU: buildParsimonyTreesKernel
  Grid = (K, 1, 1), mỗi block = 1 cây
  Stepwise addition + SPR hill-climbing
  → K cây tốt, lưu vào iqtree.treels (StringIntMap: newick → tree_id)
```

Sau K1 + outer loop (`gpuHillClimbing`), `treels` chứa toàn bộ unique candidate trees tìm được trong suốt quá trình.

### Phase 2 — REPS scoring: gắn mỗi replicate với cây tốt nhất

Với mỗi candidate tree `t`, GPU tính dot product:

```
rell[sample] = Σ_p  pattern_pars[t][p] × boot_weight[sample][p]
```

**RELL approximation** (Kishino et al. 1990): thay vì re-evaluate cây trên resampled alignment, dùng dot-product site-parsimony × bootstrap site weights.

Kernel là `REPSKernel` ([pars_bootstrap.cu](../src/pars_bootstrap.cu)):

```
Grid  = (B=1000, 1, 1) — 1 block per replicate
Block = (32,     1, 1) — 1 warp
Mỗi block: rell[i] = Σ_p d_pattern_pars[p] × d_boot_samples[i×nunit + p]
```

`d_boot_samples[B × nunit]` upload **1 lần** lúc đầu (`gpuUploadBootSamples`).
`d_pattern_pars` upload **mỗi candidate tree** (pattern parsimony của tree đó).

**Online update** (mỗi candidate tree `t`, sau khi download `h_rell[0..B-1]`):

```cpp
for each sample s:
    if h_rell[s] <= boot_logl[s]:          // tree t tốt hơn hoặc bằng
        if h_rell[s] < boot_logl[s]:
            boot_trees_parsimony[s].clear() // reset — chỉ giữ best
            boot_logl[s] = h_rell[s]
        boot_trees_parsimony[s].insert(t)   // multiple hits OK
```

→ `boot_trees_parsimony[s]` = set tree_id best cho replicate `s`.

### Phase 3 — Tổng hợp weights

`summarizeBootstrapParsimonyWeight()` ([iqtree.cpp:4124](../../iqtree.cpp#L4124)):

```cpp
for (int s = 0; s < B; s++) {
    scale = B / boot_trees_parsimony[s].size();  // chia đều nếu multiple hits
    for (tree_id in boot_trees_parsimony[s]):
        btree_weights[tree_id] += scale;
}
```

`btree_weights[t]` = tổng replicate-votes cho tree `t`. Sum = B (mỗi replicate đóng góp đúng 1 phần).

### Phase 4 — Consensus từ splits

`summarizeBootstrap(params, btrees)` ([iqtree.cpp:3899](../../iqtree.cpp#L3899)):

**4a. Decompose tất cả cây thành bipartitions:**
```cpp
trees.convertSplits(taxname, sg, hash_ss, SW_COUNT, -1, false);
```
Với mỗi internal branch của tree `t` có `btree_weights[t]` → cộng weight vào split tương ứng trong `SplitGraph sg`. Mỗi **split** = bipartition: chia N taxa thành 2 tập (bitmask).

**4b. Normalize:**
```cpp
sg.scaleWeight(1.0 / sum_weights);  // → [0, 1]
sg.scaleWeight(100.0);              // → [0, 100] %
```

**4c. `.suptree` — best tree + support values:**
```cpp
mytree.createBootstrapSupport(taxname, trees, sg, hash_ss);
```
Lấy cây parsimony tốt nhất, tra cứu từng internal branch trong `sg` → ghi support % lên branch.

**4d. `.contree` — majority-rule consensus:**
`computeConsensusTree()` chỉ giữ splits có support > 50%, xây cây mới thoả tất cả splits đó (có thể multifurcating nếu splits xung đột).

### Sơ đồ tổng thể

```
Alignment gốc
    │
    ├─ K1 kernel ──► K candidate trees (treels)
    │                    │
    │        ┌───────────┘
    │        │  REPSKernel (mỗi candidate tree × B replicates)
    │        │  rell[s] = Σ parsimony[p] × boot_weight[s][p]
    │        │
    │        ▼
    │  boot_trees_parsimony[s] = best tree(s) per replicate
    │        │
    │        ▼
    │  btree_weights[t] = Σ_s (replicate votes)
    │        │
    │        ▼
    │  convertSplits → SplitGraph (bipartitions + frequencies %)
    │        │
    │        ├──► .suptree  (best tree + support annotated)
    │        └──► .contree  (majority-rule consensus)
    │
    └─ .splits.nex (tất cả splits)
```

### Key code locations

| Function | File | Vai trò |
|----------|------|---------|
| `REPSKernel` | `gpu/src/pars_bootstrap.cu` | GPU dot-product B replicates × 1 candidate tree |
| `gpuREPSEval()` | `gpu/src/pars_bootstrap.cu` | Upload pattern_pars + launch REPSKernel + download |
| `gpuHillClimbing()` | `gpu/src/gpu_init_trees.cu` | Outer loop: K2 → saveCurrentTree → REPS update |
| `saveCurrentTree()` | `iqtree.cpp:3288` | Thêm cây vào `treels` + `treels_logl` |
| `summarizeBootstrap(Params&)` | `iqtree.cpp:4047` | Entry: dispatch to ParsimonyWeight or Weight |
| `summarizeBootstrapParsimonyWeight()` | `iqtree.cpp:4124` | Tổng hợp btree_weights từ boot_trees_parsimony |
| `summarizeBootstrap(Params&, MTreeSet&)` | `iqtree.cpp:3899` | convertSplits → createBootstrapSupport → .suptree/.contree |

### nsys profiling (bb=1000, gpu_worker=200, N=295)

| Kernel | Instances | Total | % GPU | Avg/call |
|--------|-----------|-------|-------|---------|
| K2 `buildPhase3Kernel` | 10 | 12.19 s | 89.1% | 1.22 s |
| K1 `buildParsimonyTreesKernel` | 1 | 1.48 s | 10.8% | 1.48 s |
| `REPSKernel` | **581** | ~3 ms | **0.02%** | 5.18 µs |

581 REPS calls = 581 unique candidate trees được evaluated. **REPSKernel không phải bottleneck.**

---

## Ví dụ minh hoạ — 4 taxa, 10 sites

Dùng ví dụ **4 taxa, 10 sites** để trace qua toàn bộ algorithm.

---

## Setup: Alignment nhỏ

```
Taxon   Site: 1  2  3  4  5  6  7  8  9  10
A             G  G  A  A  C  C  G  A  G  C
B             G  G  T  T  C  C  A  A  G  T
C             A  A  A  A  T  T  G  G  A  C
D             A  A  T  T  T  T  A  G  A  T
```

**Bước 1: Compress thành patterns** (sites giống hệt nhau gộp lại)

| Pattern | Sites | A B C D | Count (original) |
|---------|-------|---------|-----------------|
| p0 | 1,2 | G G A A | 2 |
| p1 | 3,4 | A T A T | 2 |
| p2 | 5,6 | C C T T | 2 |
| p3 | 7 | G A G A | 1 |
| p4 | 8 | A A G G | 1 |
| p5 | 9 | G G A A | (= p0, merged) |
| p6 | 10 | C T C T | 1 |

Sau compression: **P = 5 distinct patterns** (p0 count=3, p1=2, p2=2, p3=1, p4=1, p6=1), tổng = 10 sites.

Đơn giản hoá: gọi P=5, counts = [3, 2, 2, 1, 1].

---

## Bước 2: Tạo bootstrap replicates

Bootstrap replicate = **resample 10 sites with replacement** từ 10 sites gốc.
Kết quả lưu dưới dạng **pattern frequency vector** (không lưu lại sequences).

**Ví dụ 2 replicates:**

```
Replicate 0:
  Resample 10 sites: [1,1,3,5,5,7,7,9,9,10]
  → p0: sites {1,1} → count=2
    p1: sites {3}   → count=1
    p2: sites {5,5} → count=2
    p3: sites {7,7} → count=2
    p4: sites {}    → count=0
    p6: sites {9,9,10} → p0 count+=2, p6 count=1
  → boot_samples_pars[0] = [4, 1, 2, 2, 0, 1]

Replicate 1:
  Resample 10 sites: [2,3,3,4,6,6,8,8,9,10]
  → p0: {2,9}   → count=2
    p1: {3,3,4} → count=3
    p2: {6,6}   → count=2
    p3: {}      → count=0
    p4: {8,8}   → count=2
    p6: {10}    → count=1
  → boot_samples_pars[1] = [2, 3, 2, 0, 2, 1]
```

**Lưu ý**: Với 4 taxa chỉ có 3 possible unrooted trees:
```
Tree T0: ((A,B),(C,D))   ← AB clade + CD clade
Tree T1: ((A,C),(B,D))   ← AC clade + BD clade
Tree T2: ((A,D),(B,C))   ← AD clade + BC clade
```

---

## Bước 3: Compute `_pattern_pars` cho từng candidate tree

**Fitch parsimony per pattern** = minimum cost (substitutions) cần thiết để explain pattern p trên tree T.

### Tree T0: ((A,B),(C,D))

```
       root
      /    \
  inner1   inner2
   /  \     /  \
  A    B   C    D
```

Tính Fitch score cho mỗi pattern (Fitch algorithm: bottom-up intersection/union):

**Pattern p0** (A=G, B=G, C=A, D=A):
- inner1: {G}∩{G} = {G} ≠ ∅ → cost=0, set={G}
- inner2: {A}∩{A} = {A} ≠ ∅ → cost=0, set={A}
- root: {G}∩{A} = ∅ → cost=1, set={G,A}
- **parsimony(p0, T0) = 1**

**Pattern p1** (A=A, B=T, C=A, D=T):
- inner1: {A}∩{T} = ∅ → cost=1, set={A,T}
- inner2: {A}∩{T} = ∅ → cost=1, set={A,T}
- root: {A,T}∩{A,T} = {A,T} ≠ ∅ → cost=0
- **parsimony(p1, T0) = 2**

**Pattern p2** (A=C, B=C, C=T, D=T):
- inner1: {C}∩{C} = {C} → cost=0
- inner2: {T}∩{T} = {T} → cost=0
- root: {C}∩{T} = ∅ → cost=1
- **parsimony(p2, T0) = 1**

**Pattern p3** (A=G, B=A, C=G, D=A):
- inner1: {G}∩{A} = ∅ → cost=1, set={G,A}
- inner2: {G}∩{A} = ∅ → cost=1, set={G,A}
- root: {G,A}∩{G,A} = {G,A} → cost=0
- **parsimony(p3, T0) = 2**

**Pattern p4** (A=A, B=A, C=G, D=G):
- inner1: {A}∩{A} = {A} → cost=0
- inner2: {G}∩{G} = {G} → cost=0
- root: {A}∩{G} = ∅ → cost=1
- **parsimony(p4, T0) = 1**

**Kết quả** `_pattern_pars[T0]` = [1, 2, 1, 2, 1]

**Total parsimony của T0 trên original alignment:**
```
score(T0) = sum_p( _pattern_pars[p] × count[p] )
          = 1×3 + 2×2 + 1×2 + 2×1 + 1×1
          = 3 + 4 + 2 + 2 + 1 = 12
```

### Tree T1: ((A,C),(B,D))

```
       root
      /    \
  inner1   inner2
   /  \     /  \
  A    C   B    D
```

**Pattern p0** (A=G, C=A, B=G, D=A):
- inner1: {G}∩{A} = ∅ → cost=1, set={G,A}
- inner2: {G}∩{A} = ∅ → cost=1, set={G,A}
- root: cost=0
- **parsimony(p0, T1) = 2**

**Pattern p1** (A=A, C=A, B=T, D=T):
- inner1: {A}∩{A} = {A} → cost=0
- inner2: {T}∩{T} = {T} → cost=0
- root: {A}∩{T} = ∅ → cost=1
- **parsimony(p1, T1) = 1**

**Pattern p2** (A=C, C=T, B=C, D=T):
- inner1: {C}∩{T} = ∅ → cost=1, set={C,T}
- inner2: {C}∩{T} = ∅ → cost=1, set={C,T}
- root: cost=0
- **parsimony(p2, T1) = 2**

**Pattern p3** (A=G, C=G, B=A, D=A):
- inner1: {G}∩{G} = {G} → cost=0
- inner2: {A}∩{A} = {A} → cost=0
- root: {G}∩{A} = ∅ → cost=1
- **parsimony(p3, T1) = 1**

**Pattern p4** (A=A, C=G, B=A, D=G):
- inner1: {A}∩{G} = ∅ → cost=1
- inner2: {A}∩{G} = ∅ → cost=1
- root: cost=0
- **parsimony(p4, T1) = 2**

**Kết quả** `_pattern_pars[T1]` = [2, 1, 2, 1, 2]

```
score(T1) = 2×3 + 1×2 + 2×2 + 1×1 + 2×1 = 6+2+4+1+2 = 15
```

T0 tốt hơn T1 trên original alignment (12 < 15).

---

## Bước 4: REPS evaluation

Sau khi compute `_pattern_pars`, đánh giá tree trên **từng bootstrap replicate**:

```
REPS(T, replicate_i) = sum_p( _pattern_pars[p, T] × boot_samples_pars[i][p] )
```

**Lưu ý**: Trong MPBoot, parsimony score = COST (thấp = tốt hơn).
REPS = tổng cost trên replicate alignment. Tree nào có REPS **thấp nhất** là best per replicate.

### Tính REPS cho replicate 0: boot_samples = [4, 1, 2, 2, 0, 1]

```
(Chỉ có 5 patterns p0..p4, bỏ p5 vì merged vào p0 trong ví dụ đơn giản hoá)

REPS(T0, rep0) = 1×4 + 2×1 + 1×2 + 2×2 + 1×0
               = 4 + 2 + 2 + 4 + 0 = 12

REPS(T1, rep0) = 2×4 + 1×1 + 2×2 + 1×2 + 2×0
               = 8 + 1 + 4 + 2 + 0 = 15
```

→ T0 tốt hơn trên replicate 0 (REPS thấp hơn).
```
best_tree[0] = T0,  best_pars[0] = 12
```

### Tính REPS cho replicate 1: boot_samples = [2, 3, 2, 0, 2, 1]

```
REPS(T0, rep1) = 1×2 + 2×3 + 1×2 + 2×0 + 1×2
               = 2 + 6 + 2 + 0 + 2 = 12

REPS(T1, rep1) = 2×2 + 1×3 + 2×2 + 1×0 + 2×2
               = 4 + 3 + 4 + 0 + 4 = 15
```

→ T0 vẫn tốt hơn.
```
best_tree[1] = T0,  best_pars[1] = 12
```

---

## Bước 5: Online update (candidate trees)

Sau nhiều iterations, giả sử collect thêm tree T2:

**Tree T2: ((A,D),(B,C))**

`_pattern_pars[T2]` = [2, 2, 1, 1, 2]  (tính tương tự)

```
REPS(T2, rep0) = 2×4 + 2×1 + 1×2 + 1×2 + 2×0 = 8+2+2+2+0 = 14
REPS(T2, rep1) = 2×2 + 2×3 + 1×2 + 1×0 + 2×2 = 4+6+2+0+4 = 16
```

→ T2 tệ hơn T0 trên cả 2 replicates → `best_tree` không thay đổi.

**Giả sử iteration 500 tìm được T3: ((A,B),(C,D)) với SPR cải tiến**

`_pattern_pars[T3]` = [1, 2, 1, 2, 1] (giống T0 vì cùng topology, nhưng sau SPR có thể cùng)

Điều thực tế: với 4 taxa chỉ có 3 topologies, nhưng với N lớn: hàng nghìn topologies, REPS giúp chọn nhanh cái nào fit replicate alignment nhất.

---

## Bước 6: Compute bootstrap support

Sau khi main loop dừng (B=1000 replicates đã collect best_tree[0..999]):

```
Giả sử sau 1000 replicates:
  best_tree[i] = T0  cho i = 0..612  (613 replicates)
  best_tree[i] = T1  cho i = 613..749 (137 replicates)
  best_tree[i] = T2  cho i = 750..999 (250 replicates)

Split "AB|CD" (clade {A,B} vs {C,D}) có trong T0:
  support("AB|CD") = 613/1000 = 61.3%

Split "AC|BD" (clade {A,C} vs {B,D}) có trong T1:
  support("AC|BD") = 137/1000 = 13.7%

Split "AD|BC" có trong T2:
  support("AD|BC") = 250/1000 = 25.0%
```

Output newick: `((A,B)61,(C,D)):0.1;`  (support value trên branch)

---

## Bước 7: Tại sao REPS = dot product?

Parsimony score của tree T trên alignment là:

```
total_parsimony(T, alignment) = sum_{all sites s} parsimony_cost(T, site_s)
                               = sum_p ( count[p] × parsimony_cost(T, p) )
                               = count_vector · pattern_pars_vector
```

Với bootstrap replicate i, `count[p]` thay bằng `boot_samples_pars[i][p]`:

```
parsimony(T, replicate_i) = sum_p( boot_samples_pars[i][p] × _pattern_pars[p, T] )
                           = boot_samples_pars[i] · _pattern_pars[T]
```

**Đây chính là dot product** — không cần tree traversal! `_pattern_pars` đã tính sẵn 1 lần từ Fitch traversal, rồi tái dụng cho tất cả B replicates.

---

## So sánh: Standard Bootstrap vs MPBoot

```
Standard bootstrap (naive):
  for i = 1..B:
    alignment_i = resample(original_alignment)         # O(sites)
    T_i = full_tree_search(alignment_i)                # O(N² × iterations) ← SLOW
    add T_i to set

MPBoot:
  [1-time] compute _pattern_pars[P] for each candidate T  # O(N × P) Fitch
  [1-time] for i = 1..B:
    REPS_i = _pattern_pars · boot_samples[i]             # O(P) dot product ← FAST
    update best_tree[i] if better
```

**Với B=1000, P=5000, N=300:**
- Standard: 1000 × (tree search) ≈ 1000 × 10s = **~3 giờ**
- MPBoot REPS per tree: 1000 × 5000 ops = 5M ops ≈ **< 1ms per candidate**
- MPBoot tổng: dominated by tree search (1 lần) ≈ **vài phút**

---

## GPU Opportunity — Visualized

```
CPU hiện tại (per candidate tree):
  _pattern_pars[P]  computed (Fitch)
  for i = 0..999:                   # sequential
      rell = dot(_pattern_pars, boot_samples[i])
      if rell < best_pars[i]:
          best_pars[i] = rell
          best_tree[i] = cur_tree_idx

GPU (proposed REPSKernel):
  blockDim = (32,)   # 1 warp per replicate
  gridDim  = (1000,) # 1 block per replicate = fully parallel

  blockIdx.x = replicate i
  for p = lane, lane+32, lane+64, ...:   # parallel over patterns
      local_sum += _pattern_pars[p] × boot_samples[i][p]
  warpReduceAdd(local_sum) → lane 0
  if local_sum < best_pars[i]: update

All 1000 replicates evaluated SIMULTANEOUSLY in 1 kernel launch
vs. 1000 sequential iterations on CPU
```

**Memory layout cho GPU:**
```
d_boot_samples[B × P]:  layout [replicate][pattern]
  → block i reads boot_samples[i*P .. (i+1)*P-1]
  → 32 lanes read strided (coalesced within each warp)

d_pattern_pars[P]:  broadcast (constant per kernel launch)
  → fits in L1 cache (~20KB cho P=5000 × uint16_t)
```

---

## Bước 8: Cây output khi `-bb` — từ `treels`, không phải `candidateTrees`

### Nguồn dữ liệu

Khi `-bb` được dùng, **`candidateTrees` không được populate**. Thay vào đó:

| Cấu trúc | Nội dung |
|---|---|
| `treels` (StringIntMap) | topology Newick string → index |
| `treels_logl` (DoubleVector) | parsimony score của từng topology |
| `candidateTrees` | **trống** (bị clear trước khi search) |

Mỗi iteration, `saveCurrentTree(score)` (`iqtree.cpp:3288`) thêm cây vào `treels` nếu topology chưa có (dedup theo Newick string).

### Flow từ `treels` đến `.treefile`

```
[tree search]
  saveCurrentTree(-(double)pars)       ← mỗi candidate tree
      treels[newick_str] = idx
      treels_logl[idx]  = score

[sau search — phyloanalysis.cpp:1963]
  iqtree.summarizeBootstrap(params)    ← iqtree.cpp:4047
      đọc tất cả cây trong treels
      tính bootstrap weight cho từng topology
          (mỗi replicate i vote cho cây có REPS thấp nhất)
      createBootstrapSupport()         ← tạo consensus tree với support values
      readTree(consensus_stream)       ← load consensus vào current tree object

[phyloanalysis.cpp:1969]
  iqtree.printResultTree()             ← iqtree.cpp:4402
      in current tree (= consensus) ra .treefile
```

### Điểm quan trọng

- `.treefile` là **consensus tree** có support values — KHÔNG phải cây có parsimony tốt nhất đơn thuần
- Cây được chọn theo **bootstrap weight** (số replicate support), không phải max `treels_logl`
- GPU `gpuHillClimbing` gọi `saveCurrentTree()` → cây GPU vào `treels` → `summarizeBootstrap()` xử lý giống hệt CPU
- Chất lượng output cuối phụ thuộc vào **độ đa dạng + chất lượng tổng thể của cây trong `treels`**, không chỉ best score

### Key lines (iqtree.cpp)

| Dòng | Function | Vai trò |
|------|----------|---------|
| 3288 | `saveCurrentTree()` | Thêm cây vào `treels` + `treels_logl` |
| 4047 | `summarizeBootstrap(Params&)` | Tính bootstrap weight, tạo consensus |
| 3907 | `max_element(tree_weights)` | Chọn topology thắng nhiều replicates nhất |
| 3968 | `readTree(tree_stream)` | Load consensus vào current tree object |
| 4402 | `printResultTree()` | Ghi ra `.treefile` |
