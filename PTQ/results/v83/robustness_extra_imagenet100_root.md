# robustness_extra — imagenet100_root

## Pool B_constrained: 5047 pairs, 1692 architectures

| Device dependence (%) | rate [95% CI] | n arch |
|---|---|---|
| all three devices, eps=0% | 76.4 [74.3, 78.4] | 1663 |
| all three devices, eps=2% | 27.1 [25.0, 29.3] | 1663 |
| all three devices, eps=5% | 18.3 [16.3, 20.2] | 1663 |
| all three devices, eps=10% | 13.2 [11.6, 14.9] | 1663 |
| all three, FP32 TFLite acc >= 0.5 | 70.0 [67.1, 72.8] | 1024 |
| pairwise Redmi 13 vs Pixel 4 (exact) | 59.9 [57.5, 62.2] | 1663 |
| pairwise Redmi 13 vs Tab S7+ (exact) | 61.3 [58.9, 63.6] | 1663 |
| pairwise Pixel 4 vs Tab S7+ (exact) | 53.1 [50.8, 55.4] | 1692 |
| pairwise Redmi 13 vs Pixel 4 (eps=5%) | 9.3 [8.0, 10.9] | 1663 |
| pairwise Redmi 13 vs Tab S7+ (eps=5%) | 14.3 [12.7, 16.0] | 1663 |
| pairwise Pixel 4 vs Tab S7+ (eps=5%) | 5.8 [4.7, 6.9] | 1692 |

Per-device majority modes: {'Redmi 13': 'dynamic', 'Pixel 4': 'dynamic', 'Tab S7+': 'dynamic'}

| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |
|---|---|---|---|
| global_fp32 | 23.2 [21.9, 24.5] | 12.6 [11.4, 13.7] | 51.6 [45.8, 56.3] |
| global_fp16 | 26.1 [24.8, 27.3] | 9.3 [8.5, 10.1] | 43.8 [39.5, 47.2] |
| global_mixed | 3.4 [2.8, 4.1] | 10.7 [9.8, 11.8] | 35.7 [32.0, 39.8] |
| global_dynamic | 41.6 [40.0, 43.3] | 4.2 [3.7, 4.8] | 10.3 [8.5, 12.9] |
| global_int8 | 5.7 [4.9, 6.4] | 11.3 [10.3, 12.3] | 39.1 [36.0, 41.4] |
| per_device_majority | 41.6 [40.0, 43.3] | 4.2 [3.7, 4.8] | 10.3 [8.5, 13.1] |
| per_arch_majority | 67.4 [66.3, 68.6] | 2.4 [2.2, 2.7] | 3.5 [2.9, 4.2] |

## Pool A_unconstrained: 5017 pairs, 1693 architectures

| Device dependence (%) | rate [95% CI] | n arch |
|---|---|---|
| all three devices, eps=0% | 90.7 [89.3, 92.2] | 1631 |
| all three devices, eps=2% | 39.5 [37.1, 41.9] | 1631 |
| all three devices, eps=5% | 28.0 [25.8, 30.2] | 1631 |
| all three devices, eps=10% | 19.7 [17.7, 21.6] | 1631 |
| all three, FP32 TFLite acc >= 0.5 | 88.1 [86.1, 90.1] | 990 |
| pairwise Redmi 13 vs Pixel 4 (exact) | 74.0 [71.7, 76.1] | 1631 |
| pairwise Redmi 13 vs Tab S7+ (exact) | 75.2 [73.0, 77.2] | 1631 |
| pairwise Pixel 4 vs Tab S7+ (exact) | 64.3 [62.1, 66.7] | 1693 |
| pairwise Redmi 13 vs Pixel 4 (eps=5%) | 14.7 [13.1, 16.4] | 1631 |
| pairwise Redmi 13 vs Tab S7+ (eps=5%) | 21.5 [19.4, 23.5] | 1631 |
| pairwise Pixel 4 vs Tab S7+ (eps=5%) | 10.5 [9.2, 12.0] | 1693 |

Per-device majority modes: {'Redmi 13': 'int8', 'Pixel 4': 'dynamic', 'Tab S7+': 'dynamic'}

| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |
|---|---|---|---|
| global_fp32 | 19.2 [18.1, 20.4] | 15.4 [14.1, 16.6] | 61.1 [58.8, 64.3] |
| global_fp16 | 22.7 [21.3, 23.9] | 11.8 [10.9, 12.8] | 56.7 [52.3, 58.7] |
| global_mixed | 11.2 [10.1, 12.2] | 9.6 [8.7, 10.5] | 27.1 [26.0, 28.2] |
| global_dynamic | 29.7 [28.3, 31.0] | 5.8 [5.3, 6.3] | 22.6 [20.1, 24.1] |
| global_int8 | 17.3 [16.2, 18.4] | 10.7 [9.9, 11.6] | 34.0 [30.0, 38.0] |
| per_device_majority | 30.5 [29.3, 31.9] | 4.6 [4.2, 5.0] | 13.0 [11.9, 15.0] |
| per_arch_majority | 59.3 [58.3, 60.2] | 3.9 [3.6, 4.2] | 7.8 [6.4, 9.2] |

