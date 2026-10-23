#!/usr/bin/env python3
"""LaTeX fragments for v85 from the result JSONs: results/v85/tab_named.tex, tab_named_full_<c>.tex, tab_robust.tex
    python make_tables_v85.py
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
nm = json.load(open('results/v85/named_models.json'))
rb = json.load(open('results/v85/robustness_v85.json'))
MAIN = ['ShuffleNet', 'SqueezeNet-1', 'MobileNetV2', 'EfficientNet', 'RegNet', 'DenseNet', 'ResNet', 'AlexNet', 'VGG']
def row(r, esc=lambda s: s):
    cells = [esc(r['model'].replace('InceptionV3-1', 'InceptionV3')), f'{r["mb"]:.1f}' if r['mb'] == r['mb'] else '--', f'{r["acc"]:.1f}']
    for d in ('Redmi 13', 'Pixel 4', 'Tab S7+'):
        v = r[d]; cells.append(v.replace(' ', '\\,') if v != '--' else '--')
    cells.append(f'{r["r1"]:.1f}')
    return ' & '.join(cells) + r' \\'
T = {r['model']: r for r in nm['imagenet100']['table']}
open('results/v85/tab_named.tex', 'w').write('\n'.join(row(T[m]) for m in MAIN) + '\n')
for c in ('cifar10', 'imagenet100'):
    open(f'results/v85/tab_named_full_{c}.tex', 'w').write('\n'.join(row(r) for r in nm[c]['table']) + '\n')
SC = [('base', 'All pairs'), ('F1 FP32 accuracy >= 50%', 'FP32 accuracy $\\geq50\\%$'), ('F2 mixed + static INT8 merged', 'Mixed and static INT8 merged'),
      ('F3 1 pp absolute accuracy budget', '$1$\\,pp absolute budget'), ('F4 min-latency (best of 20) labels', 'Best-of-20 latency labels'),
      ('F6 family unq', 'Family: LLM-generated (unq)'), ('F6 family rag', 'Family: retrieval (rag)'), ('F6 named', 'Family: named')]
L = []
for k, lab in SC:
    cells = [lab]
    for c in ('cifar10', 'imagenet100'):
        o = rb[c].get(k)
        cells += [f'{o["n"]}', f'{o["K1"]:.1f}', f'{o["K2"]:.1f}', f'{o["removed"]:.0f}\\%'] if o else ['--'] * 4
    L.append(' & '.join(cells) + r' \\')
open('results/v85/tab_robust.tex', 'w').write('\n'.join(L) + '\n')
print('ok')
