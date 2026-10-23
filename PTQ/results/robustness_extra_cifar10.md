# robustness_extra — cifar10

## Pool B_constrained: 660 pairs, 223 architectures

| Device dependence (%) | rate [95% CI] | n arch |
|---|---|---|
| all three devices, eps=0% | 81.3 [75.7, 86.4] | 214 |
| all three devices, eps=2% | 52.8 [46.3, 58.9] | 214 |
| all three devices, eps=5% | 32.7 [26.6, 38.8] | 214 |
| all three devices, eps=10% | 15.9 [11.2, 21.0] | 214 |
| all three, FP32 TFLite acc >= 0.5 | 82.6 [75.8, 88.6] | 132 |
| pairwise Redmi 13 vs Pixel 4 (exact) | 65.9 [59.8, 72.4] | 214 |
| pairwise Redmi 13 vs Tab S7+ (exact) | 72.4 [66.4, 78.0] | 214 |
| pairwise Pixel 4 vs Tab S7+ (exact) | 44.4 [37.7, 50.7] | 223 |
| pairwise Redmi 13 vs Pixel 4 (eps=5%) | 17.3 [12.1, 22.4] | 214 |
| pairwise Redmi 13 vs Tab S7+ (eps=5%) | 25.7 [20.1, 31.8] | 214 |
| pairwise Pixel 4 vs Tab S7+ (eps=5%) | 13.5 [9.0, 18.4] | 223 |

Per-device majority modes: {'Redmi 13': 'mixed', 'Pixel 4': 'dynamic', 'Tab S7+': 'dynamic'}

| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |
|---|---|---|---|
| global_fp32 | 8.5 [6.1, 10.9] | 33.7 [28.6, 39.1] | 134.0 [121.4, 142.1] |
| global_fp16 | 18.9 [16.0, 21.9] | 27.2 [23.3, 31.8] | 100.0 [91.7, 110.8] |
| global_mixed | 13.3 [10.9, 15.7] | 23.4 [20.4, 26.9] | 101.8 [84.7, 124.5] |
| global_dynamic | 55.3 [51.1, 59.2] | 4.5 [3.2, 6.1] | 9.2 [7.9, 10.6] |
| global_int8 | 3.9 [2.4, 5.5] | 24.1 [21.0, 27.6] | 106.0 [82.6, 127.0] |
| per_device_majority | 56.8 [53.0, 60.5] | 5.2 [4.0, 6.5] | 10.2 [8.7, 12.9] |
| per_arch_majority | 66.5 [63.8, 69.3] | 4.5 [3.2, 5.9] | 10.2 [7.9, 12.6] |

(Unconstrained pool A needs --data-root; the shipped CSV holds the constrained pool only.)
