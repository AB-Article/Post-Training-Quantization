# robustness_extra — cifar10_root

## Pool B_constrained: 3539 pairs, 1185 architectures

| Device dependence (%) | rate [95% CI] | n arch |
|---|---|---|
| all three devices, eps=0% | 83.1 [80.9, 85.2] | 1169 |
| all three devices, eps=2% | 62.0 [59.4, 64.8] | 1169 |
| all three devices, eps=5% | 47.0 [44.2, 49.8] | 1169 |
| all three devices, eps=10% | 36.0 [33.3, 38.9] | 1169 |
| all three, FP32 TFLite acc >= 0.5 | 85.7 [82.6, 88.5] | 524 |
| pairwise Redmi 13 vs Pixel 4 (exact) | 73.4 [70.8, 75.9] | 1169 |
| pairwise Redmi 13 vs Tab S7+ (exact) | 73.6 [70.9, 76.0] | 1169 |
| pairwise Pixel 4 vs Tab S7+ (exact) | 38.1 [35.4, 40.9] | 1185 |
| pairwise Redmi 13 vs Pixel 4 (eps=5%) | 33.9 [31.2, 36.5] | 1169 |
| pairwise Redmi 13 vs Tab S7+ (eps=5%) | 37.0 [34.2, 39.8] | 1169 |
| pairwise Pixel 4 vs Tab S7+ (eps=5%) | 13.5 [11.6, 15.4] | 1185 |

Per-device majority modes: {'Redmi 13': 'mixed', 'Pixel 4': 'dynamic', 'Tab S7+': 'fp16'}

| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |
|---|---|---|---|
| global_fp32 | 8.6 [7.5, 9.6] | 70.0 [65.6, 74.2] | 338.2 [295.8, 373.1] |
| global_fp16 | 32.1 [30.2, 33.8] | 31.4 [29.1, 33.9] | 114.0 [106.6, 127.8] |
| global_mixed | 13.0 [12.0, 14.0] | 43.3 [40.4, 46.4] | 170.9 [160.5, 187.7] |
| global_dynamic | 35.4 [33.6, 37.4] | 36.8 [33.5, 40.1] | 238.4 [210.0, 287.2] |
| global_int8 | 10.9 [9.7, 12.2] | 37.6 [35.0, 40.4] | 152.6 [145.1, 161.0] |
| per_device_majority | 39.0 [37.6, 40.4] | 25.2 [23.6, 26.8] | 119.3 [111.7, 147.7] |
| per_arch_majority | 66.1 [65.0, 67.2] | 7.4 [6.8, 8.1] | 32.7 [22.2, 42.8] |

## Pool A_unconstrained: 3995 pairs, 1338 architectures

| Device dependence (%) | rate [95% CI] | n arch |
|---|---|---|
| all three devices, eps=0% | 89.2 [87.5, 90.9] | 1320 |
| all three devices, eps=2% | 67.7 [65.2, 70.2] | 1320 |
| all three devices, eps=5% | 52.3 [49.5, 55.2] | 1320 |
| all three devices, eps=10% | 41.0 [38.3, 43.5] | 1320 |
| all three, FP32 TFLite acc >= 0.5 | 88.1 [85.2, 90.7] | 528 |
| pairwise Redmi 13 vs Pixel 4 (exact) | 82.0 [80.0, 84.0] | 1320 |
| pairwise Redmi 13 vs Tab S7+ (exact) | 79.2 [77.0, 81.4] | 1320 |
| pairwise Pixel 4 vs Tab S7+ (exact) | 39.9 [37.3, 42.5] | 1337 |
| pairwise Redmi 13 vs Pixel 4 (eps=5%) | 39.5 [36.9, 42.3] | 1320 |
| pairwise Redmi 13 vs Tab S7+ (eps=5%) | 41.6 [38.9, 44.3] | 1320 |
| pairwise Pixel 4 vs Tab S7+ (eps=5%) | 14.9 [12.9, 16.8] | 1337 |

Per-device majority modes: {'Redmi 13': 'mixed', 'Pixel 4': 'dynamic', 'Tab S7+': 'fp16'}

| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |
|---|---|---|---|
| global_fp32 | 6.0 [5.2, 6.9] | 74.3 [70.1, 78.7] | 335.6 [297.9, 371.8] |
| global_fp16 | 27.7 [26.2, 29.3] | 38.6 [36.1, 41.1] | 142.1 [124.8, 159.3] |
| global_mixed | 18.3 [17.3, 19.4] | 37.3 [34.9, 39.9] | 145.3 [137.0, 152.0] |
| global_dynamic | 33.7 [32.0, 35.4] | 37.3 [34.0, 40.5] | 230.7 [197.0, 276.0] |
| global_int8 | 14.3 [13.0, 15.6] | 29.6 [27.8, 31.4] | 124.2 [121.8, 129.2] |
| per_device_majority | 43.8 [42.4, 45.2] | 26.1 [24.5, 27.9] | 142.5 [114.8, 156.8] |
| per_arch_majority | 62.9 [61.9, 63.9] | 9.0 [8.3, 9.6] | 58.7 [49.0, 65.1] |

