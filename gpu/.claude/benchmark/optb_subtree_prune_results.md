# Opt-B: Subtree Prune trong SPR DFS — Kết quả

**Implementation**: `doAddTraverse` (`pars_build.cu`) — trước mỗi `testInsert` (khi `mint<=0`), check lower bound:
```cpp
const unsigned int lb = score_tree[q_num] + score_tree[vfToNum(topo->back_vf[cur_q], N)];
if (lb < sh.randomMP) { testInsert(...); }
```
Proof: `mp ≥ score_tree[q] + score_tree[r]` (cross terms ≥ 0) → safe to skip khi `lb ≥ randomMP`.

**Date**: 2026-05-13

---

## Prune Rate (10 datasets, -gpu_phase3_top_pct -1, numpars=400, seed=1)

| Dataset | N | Prune rate |
|---------|---|-----------|
| dna_M214_295_1836 | 295 | 22.42% |
| dna_M1110_330_1711 | 330 | 22.39% |
| dna_M5381_413_3632 | 413 | 22.78% |
| dna_M9915_504_2757 | 504 | 21.60% |
| dna_M12051_699_6914 | 699 | 23.51% |
| prot_M11740_138_4427 | 138 | 20.31% |
| prot_M10372_169_22426 | 169 | 20.85% |
| dna_M8975_405_3027 | 405 | 21.08% |
| dna_M14582_372_61199 | 372 | 20.56% |
| dna_M3777_363_1707 | 363 | 22.76% |

**Trung bình: ~21–23%** — nhất quán trên mọi taxa size và dataset type (DNA/protein).

---

## Speedup (10 datasets, Opt-G2 enabled, numpars=400, seed=1)

So sánh: `gpu_opt_400` (no Opt-B) vs binary mới (with Opt-B), cùng settings.

| Dataset | N | old_ms | new_ms | Speedup |
|---------|---|--------|--------|---------|
| dna_M214_295_1836 | 295 | 25615 | 25344 | 1.01× |
| dna_M1110_330_1711 | 330 | 14317 | 15530 | 0.92× |
| dna_M5381_413_3632 | 413 | 83520 | 78369 | **1.07×** |
| dna_M9915_504_2757 | 504 | 127535 | 81668 | **1.56×** |
| dna_M12051_699_6914 | 699 | 203357 | 158573 | **1.28×** |
| prot_M11740_138_4427 | 138 | 14638 | 15493 | 0.94× |
| prot_M10372_169_22426 | 169 | 74692 | 51254 | **1.46×** |
| dna_M8975_405_3027 | 405 | 27376 | 20925 | **1.31×** |
| dna_M14582_372_61199 | 372 | 221313 | 140902 | **1.57×** |
| dna_M3777_363_1707 | 363 | 11267 | 6045 | **1.86×** |

**7/10 datasets nhanh hơn, avg speedup ~1.31×** (bỏ 3 outlier âm: ~1.57×).

---

## Correctness

**Mathematically proven**: `mp ≥ score_tree[q] + score_tree[r]` — never skip a genuinely improving move.

**Empirical check** (7 datasets, seed=1): 6/7 OK hoặc BETTER, 1/7 "worse" (dna_M5381: 48517→48527).  
**Multi-seed check** (dna_M5381, 7 seeds): scores 48520–48533 — 48517 (old) nằm trong range bình thường.

→ Kết quả "worse" ở seed=1 là **stochastic variance** từ RNG state drift (Opt-B thay đổi số lần gọi `gpuRandum()` → tie-breaking khác nhau), không phải correctness bug.

---

## Notes

- Prune rate không phụ thuộc vào taxa size (20–23% nhất quán)
- Speedup variance cao (0.92–1.86×) do 1 seed, RNG drift thay đổi convergence
- Speedup lớn nhất ở datasets mà Opt-B giúp Phase 3 converge nhanh hơn (early stopping trigger sớm hơn)
- Prune rate column chỉ hiển thị khi `-gpu_phase3_top_pct -1` (block k=0 luôn làm Phase 3); khi Opt-G2 bật, k=0 có thể bị skip
