#!/usr/bin/env python3
"""Checkpoint-to-graph export loss on CIFAR-10, split by the preprocessing of
the training transform (Table II, lower panel, of the main paper; caption of
supplementary Fig. S25).

Every converted graph is evaluated under one canonical resize-and-normalize
protocol, whereas each checkpoint was trained under its own transform from
ab/nn/transform/. Models are grouped by how far the two differ:

  Matched preprocessing  training transform normalizes and applies no
                         crop/pad, i.e. it agrees with the evaluation protocol
  + crop or pad          normalizes, but CenterCrop/RandomCrop/
                         RandomResizedCrop/Pad changes the input geometry
  No normalization       no Normalize in the training transform, so the graph
                         receives normalized input it never saw in training

The gap is the best recorded training accuracy of a model minus the accuracy of
its converted FP32 graph, in percentage points. It precedes quantization and is
common to all five PTQ modes, so it does not enter any latency or regret
result; those use the matched converted FP32 graph as the reference.

    python export_gap.py NND_ROOT
"""
import glob
import json
import os
import re
import sys

import numpy as np

GEO = re.compile(r'transforms\.(CenterCrop|RandomCrop|RandomResizedCrop|Pad)\(')
GROUPS = ('Matched preprocessing', '+ crop or pad', 'No normalization')


def group(root, name):
    """Classify a training transform by its distance from the eval protocol."""
    try:
        src = open(f'{root}/ab/nn/transform/{name}.py').read()
    except OSError:
        return None
    if 'Normalize' not in src:
        return 'No normalization'
    return '+ crop or pad' if GEO.search(src) else 'Matched preprocessing'


def main(root):
    acc = json.load(open(f'{root}/_work/stats/all_models_accuracy_fp32.json'))
    rows, no_record = [], 0
    for model, rec in acc.items():
        trials = glob.glob(
            f'{root}/ab/nn/stat/train/'
            f'img-classification_cifar-10_acc_{model}/*.json')
        grp = group(root, rec.get('transform', ''))
        if not trials or not grp:
            no_record += 1
            continue
        best = max(t['accuracy'] for f in trials for t in json.load(open(f)))
        rows.append((grp, 100 * (best - rec['accuracy'])))

    print(f'CIFAR-10 FP32 graphs {len(acc)}; with public training records '
          f'and a resolvable transform {len(rows)} '
          f'({no_record} without)\n')
    head = f'{"group":24s} {"n":>5s} {"mean":>6s} {"med.":>6s} {">20pp":>7s}'
    print(head)
    print('-' * len(head))
    for grp in GROUPS + ('pooled',):
        gaps = np.array([g for k, g in rows if grp == 'pooled' or k == grp])
        print(f'{grp:24s} {len(gaps):5d} {gaps.mean():6.1f} '
              f'{np.median(gaps):6.1f} {100 * np.mean(gaps > 20):6.1f}%')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'NND')
