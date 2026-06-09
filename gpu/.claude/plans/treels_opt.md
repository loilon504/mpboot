# Plan: Relative Margin — gpu_treels_margin thành double

## Context

Fixed margin (`-gpu_treels_margin 10`, int) không scale theo dataset size:
- N=50 (randomMP≈5,000): ngưỡng 10 = 0.2% relative → rộng vừa
- N=400 (randomMP≈200,000): ngưỡng 10 = 0.005% → gần như chỉ lưu cây tốt hơn hiện tại

Giải pháp: đổi `gpu_treels_margin` từ `int` sang `double`, dùng làm **relative fraction**:
- Giá trị `-1` → save all (unlimited)
- Giá trị `r ≥ 0` → `pass_save_a = mp < randomMP × (1 + r)` (relative margin)

Một flag duy nhất, không cần flag thứ hai.

## Kết quả benchmark hiện tại (18 DNA datasets, 2026-06-01)

| Variant | avg Δ | pass_rate ±5% | speedup |
|---------|-------|---------------|---------|
| CPU | +1.9% | 52.4% | — |
| GPU unlimited (margin=-1, cũ) | −8.3% | 28.6% | 4.58× |
| GPU pool10w (margin=10 abs) | −7.1% | 42.9% | 4.63× |
| GPU saveall (margin=-1, no cutoff) | −15.0% | 27.8% | 2.22× |
| GPU saveall_fix (margin=-1 + cutoff) | đang chạy | — | — |

margin=10 tốt hơn unlimited về calibration nhưng lý do là vô tình: trên dataset lớn nó gần như không lưu gì qua SAVE A.

---

## Semantics mới

| Giá trị | SAVE A behavior |
|---------|----------------|
| `-1` (default) | save all testInsert hits (unlimited) |
| `0.0` | chỉ lưu cây tốt hơn randomMP (strict improvement) |
| `0.001` | `mp < randomMP × 1.001` — trong vòng 0.1% |
| `0.01` | `mp < randomMP × 1.01` — trong vòng 1% |

---

## Các thay đổi

### File 1: `mpboot/tools.h`

```cpp
// CŨ:
int gpu_treels_margin;

// MỚI:
double gpu_treels_margin;  // relative SAVE A margin: -1=save all, r≥0 → mp < randomMP*(1+r)
```

### File 2: `mpboot/tools.cpp`

```cpp
// initParams():
params.gpu_treels_margin = -1.0;  // default: save all (unlimited)

// parseArg():
if (strcmp(argv[cnt], "-gpu_treels_margin") == 0) {
    params.gpu_treels_margin = convert_double(argv[++cnt]);
    continue;
}
```

`convert_double` đã có sẵn trong codebase (cùng pattern với `convert_float`).

### File 3: `mpboot/gpu/include/pars_tree.cuh`

Đổi `save_margin` từ `unsigned int` → `float` trong cả hai struct:

```cpp
// GpuParsimonyMem:
float save_margin;   // -1=save all; r≥0 → mp < randomMP*(1+r)

// BuildSharedT<NTAXA>:
float save_margin;   // same semantics
```

### File 4: `mpboot/gpu/src/gpu_init_trees.cu`

**a) Set từ params:**
```cpp
mem->save_margin = (float)params.gpu_treels_margin;
```

**b) Update `logl_cutoff` sau mỗi round** (thay thế toàn bộ khối if/else hiện tại):
```cpp
if (!iqtree.treels_logl.empty()) {
    if (params.gpu_treels_margin < 0.0) {
        // save-all: dùng percentile để lọc SAVE B/C
        DoubleVector logl = iqtree.treels_logl;
        nth_element(logl.begin(),
                    logl.begin() + logl.size() * params.cutoff_percent / 100,
                    logl.end(), std::greater<double>());
        iqtree.logl_cutoff = logl[logl.size() * params.cutoff_percent / 100];
    } else {
        // relative margin: cutoff nhất quán với SAVE A
        // logl = -parsimony → best_logl = -min_parsimony < 0
        // boot_cutoff = min_parsimony*(1+r) ↔ logl_cutoff = best_logl*(1+r)
        double best_logl = *std::max_element(
            iqtree.treels_logl.begin(), iqtree.treels_logl.end());
        iqtree.logl_cutoff = best_logl * (1.0 + params.gpu_treels_margin);
    }
}
```

Lưu ý: `best_logl < 0` và `r ≥ 0` → `logl_cutoff = best_logl*(1+r) ≤ best_logl` → `boot_cutoff ≥ min_parsimony` ✓

**c) Các chỗ dùng `params.gpu_treels_margin` với so sánh `< 0`, `> 0`**: vẫn đúng vì:
- `-1.0 < 0` ✓ (save-all)
- `0.001 > 0` ✓ (relative margin)
- `0.0 == 0` ✓ (strict improvement)

### File 5: `mpboot/gpu/src/pars_build.cu`

**Kernel param**: đổi kiểu từ `unsigned int save_margin` → `float save_margin`:
```cpp
__global__ void buildPhase3Kernel(..., float save_margin, int treels_writers)
```

**Gán vào shared** (giữ nguyên, chỉ đổi kiểu):
```cpp
sh.save_margin = save_margin;
```

**SAVE A gate** (thay thế đoạn `0xFFFFFFFFu` check):
```cpp
bool pass_save_a = (sh.save_margin < 0.0f)
                   ? true   // save all
                   : (mp < (unsigned int)((float)sh.randomMP * (1.0f + sh.save_margin)));
```

**Launch site**: đổi sang `mem->save_margin` (float):
```cpp
buildPhase3Kernel<S, NT><<<...>>>(
    ..., mem->save_margin,  // float
    treels_writers);
```

---

## Files thay đổi

| File | Thay đổi |
|------|---------|
| `mpboot/tools.h` | `int` → `double gpu_treels_margin` |
| `mpboot/tools.cpp` | default `-1.0`, parse `convert_double` |
| `mpboot/gpu/include/pars_tree.cuh` | `unsigned int` → `float save_margin` trong GpuParsimonyMem + BuildSharedT |
| `mpboot/gpu/src/gpu_init_trees.cu` | Set `(float)params.gpu_treels_margin`; thay toàn bộ logl_cutoff update logic |
| `mpboot/gpu/src/pars_build.cu` | Kernel param float; SAVE A gate mới; launch site |

---

## Benchmark sau implement

Chạy 18 DNA datasets với 3 giá trị r (output vào thư mục riêng mỗi variant):

```bash
-gpu_treels_margin 0.001   # 0.1%
-gpu_treels_margin 0.005   # 0.5%
-gpu_treels_margin 0.01    # 1%
```

So sánh calibration (avg Δ, pass_rate) và speedup vs CPU bằng:
`thesis/benchmark/compare_calibration_4variants.py` (thêm các config mới)

---

## Verification

```bash
cd build && make -j4 2>&1 | tail -5

# Smoke test
CUDA_VISIBLE_DEVICES=2 ./mpboot-avx -s ../data_pandit/dna/8/data.8 \
  -seed 1 -bb 1000 -cost ../output/dna.cost \
  -use_gpu -gpu_device 0 -gpu_worker 200 \
  -gpu_treels_margin 0.001 \
  -pre /tmp/smoke_rel 2>&1 | grep -E "Round|BEST SCORE"
```

Kiểm tra:
- `+trees=X` mỗi round không tràn 600k trên dataset vừa (N≈100)
- Dataset lớn (N=400) có treels lớn hơn dataset nhỏ — scale đúng theo N²×r
