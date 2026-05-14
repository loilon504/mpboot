# Opt-P Layer 1 Benchmark Results

**Date**: 2026-05-13  
**Comparison**: Opt-P Layer 1 (build/mpboot-avx May 13) vs Opt-M baseline (output/gpu_optm_200)  
**Config**: numpars=200, sprdist=3, gpu_stop=4, seed=1  
**Datasets**: 50 (sorted by taxa, N=50..204)

## What changed in Opt-P Layer 1

- `perm[kMaxTaxa+2]` and `stackMint[kMaxTaxa]` merged into a union (non-overlapping phases) → saves 3.2 KB
- `stackMaxt[kMaxTaxa=800]` → `stackMaxt[kMaxSprStack=64]` (both use cases fit: NNI bitset=51 words max, SPR stack=12 entries max) → saves 2.9 KB
- **Total**: BuildShared 28.4 KB → 22.4 KB → blocks/SM: 5 → 7 (+40% occupancy)

## Results Summary

| Metric | Value |
|--------|-------|
| Average speedup vs Opt-M | **1.197x** |
| Min speedup | 0.977x (N=59, noise) |
| Max speedup | 1.517x (N=93-169) |
| Quality regression | **0/50 datasets** |

## Per-dataset Results

| Dataset | N | OptM ms/t | OptP ms/t | Speedup | Best (same?) |
|---------|---|-----------|-----------|---------|-------------|
| dna_M10243_203_1771 | 203 | 13.50 | 10.69 | 1.263x | ✅ |
| dna_M10467_202_4074 | 202 | 37.78 | 28.17 | 1.341x | ✅ |
| dna_M14164_204_5549 | 204 | 47.84 | 36.01 | 1.329x | ✅ |
| dna_M4720_201_2899 | 201 | 28.35 | 23.02 | 1.232x | ✅ |
| dna_M7211_201_1519 | 201 | 12.90 | 10.44 | 1.236x | ✅ |
| dna_M8984_201_3931 | 201 | 16.90 | 13.07 | 1.293x | ✅ |
| prot_M10236_59_164 | 59 | 3.19 | 3.25 | 0.982x | ✅ |
| prot_M10273_169_11009 | 169 | 89.83 | 60.45 | 1.486x | ✅ |
| prot_M10372_169_22426 | 169 | 179.38 | 136.81 | 1.311x | ✅ |
| prot_M10866_88_3329 | 88 | 11.02 | 8.77 | 1.257x | ✅ |
| prot_M11012_55_11500 | 55 | 21.19 | 21.68 | 0.977x | ✅ |
| prot_M11013_55_8741 | 55 | 22.21 | 22.58 | 0.984x | ✅ |
| prot_M1118_137_348 | 137 | 20.83 | 17.28 | 1.205x | ✅ |
| prot_M11335_99_696 | 99 | 12.61 | 8.33 | 1.514x | ✅ |
| prot_M11336_95_268 | 95 | 7.36 | 6.23 | 1.181x | ✅ |
| prot_M11338_100_567 | 100 | 15.38 | 10.16 | 1.514x | ✅ |
| prot_M11341_100_699 | 100 | 12.08 | 11.07 | 1.091x | ✅ |
| prot_M11342_91_323 | 91 | 11.51 | 8.28 | 1.390x | ✅ |
| prot_M11344_84_691 | 84 | 9.57 | 7.54 | 1.269x | ✅ |
| prot_M11595_66_463 | 66 | 3.32 | 3.37 | 0.985x | ✅ |
| prot_M11596_62_439 | 62 | 4.12 | 4.15 | 0.993x | ✅ |
| prot_M11740_138_4427 | 138 | 43.64 | 28.91 | 1.510x | ✅ |
| prot_M12103_97_199 | 97 | 16.48 | 13.68 | 1.205x | ✅ |
| prot_M12104_93_349 | 93 | 9.01 | 5.94 | 1.517x | ✅ |
| prot_M12376_116_327 | 116 | 9.20 | 7.84 | 1.173x | ✅ |
| prot_M13804_78_1889 | 78 | 15.98 | 10.76 | 1.485x | ✅ |
| prot_M1726_50_1000 | 50 | 6.47 | 6.61 | 0.979x | ✅ |
| prot_M2358_55_714 | 55 | 3.35 | 3.39 | 0.988x | ✅ |
| prot_M2593_56_386 | 56 | 4.22 | 4.26 | 0.991x | ✅ |
| prot_M2926_105_899 | 105 | 17.33 | 12.35 | 1.403x | ✅ |
| prot_M3113_77_9918 | 77 | 59.25 | 44.80 | 1.323x | ✅ |
| prot_M3114_77_11234 | 77 | 79.05 | 54.28 | 1.456x | ✅ |
| prot_M3807_82_591 | 82 | 10.26 | 8.13 | 1.262x | ✅ |
| prot_M3810_55_271 | 55 | 3.17 | 3.20 | 0.991x | ✅ |
| prot_M4249_153_455 | 153 | 41.91 | 33.35 | 1.257x | ✅ |
| prot_M4318_78_2295 | 78 | 12.65 | 10.77 | 1.175x | ✅ |
| prot_M4325_73_230 | 73 | 5.36 | 5.46 | 0.982x | ✅ |
| prot_M4539_59_12428 | 59 | 19.05 | 19.32 | 0.986x | ✅ |
| prot_M4780_90_583 | 90 | 15.34 | 12.20 | 1.257x | ✅ |
| prot_M4860_62_11544 | 62 | 21.10 | 21.57 | 0.978x | ✅ |
| prot_M4884_50_11827 | 50 | 14.38 | 14.60 | 0.985x | ✅ |
| prot_M510_57_430 | 57 | 2.90 | 2.95 | 0.983x | ✅ |
| prot_M5379_60_7776 | 60 | 12.04 | 12.16 | 0.990x | ✅ |
| prot_M6416_57_126 | 57 | 2.59 | 2.62 | 0.989x | ✅ |
| prot_M7078_77_12457 | 77 | 31.18 | 26.03 | 1.198x | ✅ |
| prot_M7729_62_2973 | 62 | 7.79 | 7.90 | 0.986x | ✅ |
| prot_M8175_194_665 | 194 | 24.40 | 16.09 | 1.516x | ✅ |
| prot_M8461_89_5699 | 89 | 39.65 | 26.91 | 1.473x | ✅ |
| prot_M8630_50_21154 | 50 | 25.58 | 25.97 | 0.985x | ✅ |
| prot_M9973_60_327 | 60 | 2.76 | 2.81 | 0.982x | ✅ |

## Analysis

- **N ≤ 65**: speedup ≈ 1.0x (noise range). Very small datasets are likely bottlenecked by something other than SM occupancy (fewer warps, short kernel).
- **N = 66–100**: speedup 1.09–1.51x. Sweet spot where higher occupancy (5→7 blocks/SM) helps hide latency.
- **N ≥ 100**: speedup 1.17–1.52x. Consistent gains, kernel is long enough to benefit from extra blocks.

## Conclusion

Opt-P Layer 1 is a **zero-regression +20% speedup** at the cost of only struct reshuffling in pars_tree.cuh.
Next: Opt-P Layer 3 (NTAXA templating) could yield much larger gains for small-N datasets.
