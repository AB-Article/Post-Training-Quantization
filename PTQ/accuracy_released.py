#!/usr/bin/env python3
"""PTQ accuracy loss (matched FP32 TFLite minus quantized) on the released
accuracy files, and the ImageNet-100 checkpoint-to-graph gap from the public
epoch-50 checkpoint accuracies (huggingface.co/NN-Dataset/checkpoints-imagenet100-epoch-50).
    python accuracy_released.py NND_ROOT IM100_ACC_DIR CHECKPOINT_JSON"""
import json, sys
import numpy as np
M = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
root, imdir, ckpt = sys.argv[1:4]
for c, D in (('CIFAR-10', f'{root}/_work/stats'), ('ImageNet-100', imdir)):
    A = {m: json.load(open(f'{D}/all_models_accuracy_{m}.json')) for m in M}
    ref = {k: v['accuracy'] for k, v in A['fp32'].items() if v['accuracy'] > 0}
    print(f'{c}: FP32 graphs {len(ref)}')
    for m in M[1:]:
        l = np.array([100 * (ref[k] - v['accuracy']) for k, v in A[m].items() if k in ref])
        print(f'  {m:8s} n={len(l):5d} mean loss {l.mean():5.2f} pp  median {np.median(l):5.2f}  '
              f'>1pp {100*np.mean(l>1):4.1f}%  feasible@tau.99 {100*np.mean([A[m][k]["accuracy"]>=0.99*ref[k] for k in A[m] if k in ref]):4.1f}%')
ck = json.load(open(ckpt))
A = json.load(open(f'{imdir}/all_models_accuracy_fp32.json'))
g = np.array([100 * (ck[k]['accuracy'] - v['accuracy']) for k, v in A.items() if k in ck])
print(f'ImageNet-100 checkpoint - FP32 graph: n={len(g)} mean {g.mean():.2f} median {np.median(g):.2f} '
      f'IQR [{np.percentile(g,25):.2f}, {np.percentile(g,75):.2f}]  |gap|<1pp {100*np.mean(abs(g)<1):.1f}%')
r = np.corrcoef([ck[k]['accuracy'] for k in A if k in ck], [A[k]['accuracy'] for k in A if k in ck])[0, 1]
print(f'  Pearson r(checkpoint, graph) = {r:.3f}')
