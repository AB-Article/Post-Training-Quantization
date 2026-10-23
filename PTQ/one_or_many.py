#!/usr/bin/env python3
"""'One model or many?' analyses (v84).

For every model the developer can ship either ONE quantized artifact to all
three devices or ONE PER DEVICE CLASS. Per-device shipping attains the
constrained optimum m*(model, device) by definition (regret 0). For a single
artifact we report:

  * best single artifact (oracle): per model, the feasible mode that minimizes
    the geometric-mean latency ratio to the per-device optimum across the
    devices on which it is labeled -- the best any one-artifact strategy can do;
  * always dynamic INT8 (fallback as in predictor.py);
  * flagship-tuned: per model, the optimum measured on one reference device
    (Tab S7+ or Pixel 4), shipped to every device ("tune on the flagship").

Outputs, by corpus and by latency band -- of the pair (best feasible latency
<2, 2-10, >=10 ms) and of the model (its best feasible latency on its slowest
device, the quantity a developer knows when deciding what to ship) -- and by device: geometric-mean regret, mean absolute regret
(ms), and the share of pairs paying more than 10 % / 25 %.
Also writes the benchmarking-budget table (runs needed per policy).

    python one_or_many.py   ->  results/v83/one_or_many.{json,txt}
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
import numpy as np, pandas as pd
import v83_analysis as V
import guideline as G

M = V.MODES
DEVS = V.DEVS
NAME = V.NAME
BANDS = [('<2 ms', 0, 2), ('2-10 ms', 2, 10), ('>=10 ms', 10, 1e9)]


def lat(r, m):
    v = r[f'lat_{m}']
    return v if pd.notna(v) else None


def regret_of(d, pick):
    """pick: array of modes per row; fallback as in predictor.py."""
    rat, ms = [], []
    for i, r in enumerate(d.itertuples()):
        c = {m: getattr(r, f'lat_{m}') for m in M if pd.notna(getattr(r, f'lat_{m}'))}
        p = pick[i]
        if p not in c:
            p = next(x for x in V.FALLBACK if x in c)
        b = min(c.values())
        rat.append(c[p] / b); ms.append((c[p] - b) / 1e6)
    return np.array(rat), np.array(ms)


def single_artifact_oracle(d):
    pick = np.empty(len(d), dtype=object)
    for mo, g in d.groupby('model'):
        best, bestv = None, np.inf
        for m in M:
            # the artifact must be feasible (accuracy) -- same on every device
            if m not in g.feas.iloc[0]:
                continue
            ls = []
            for r in g.itertuples():
                c = {k: getattr(r, f'lat_{k}') for k in M if pd.notna(getattr(r, f'lat_{k}'))}
                if not c:
                    continue
                v = c.get(m, c.get(next(x for x in V.FALLBACK if x in c)))
                ls.append(np.log(v / min(c.values())))
            if ls and np.mean(ls) < bestv:
                best, bestv = m, np.mean(ls)
        pick[g.index.values] = best or 'fp32'
    return pick


def flagship_tuned(d, ref):
    lab = d[d.device == ref].set_index('model').label.to_dict()
    return np.array([lab.get(mo, 'dynamic') for mo in d.model], dtype=object), d.model.isin(lab).values


def summarize(rat, ms, mask=None):
    if mask is not None:
        rat, ms = rat[mask], ms[mask]
    return dict(n=int(len(rat)), geo=100 * (np.exp(np.log(rat).mean()) - 1), ms=float(ms.mean()),
                gt10=100 * float(np.mean(rat > 1.10)), gt25=100 * float(np.mean(rat > 1.25)))


def run(c):
    d = G.load(c).reset_index(drop=True)
    d['best_ms'] = d[[f'lat_{m}' for m in M]].min(axis=1) / 1e6
    # model-level latency class: best feasible latency on the model's slowest device
    d['model_ms'] = d.groupby('model').best_ms.transform('max')
    pol = {'single artifact (oracle)': single_artifact_oracle(d),
           'always dynamic INT8': np.array(['dynamic'] * len(d), dtype=object)}
    masks = {}
    for ref in ('SM-T970', 'Pixel 4'):
        p, mk = flagship_tuned(d, ref)
        pol[f'tuned on {NAME[ref]}'] = p; masks[f'tuned on {NAME[ref]}'] = mk
    out = {}
    for k, p in pol.items():
        rat, ms = regret_of(d, p)
        mk = masks.get(k, np.ones(len(d), bool))
        res = {'all': summarize(rat, ms, mk)}
        for b, lo, hi in BANDS:
            res[b] = summarize(rat, ms, mk & (d.best_ms >= lo).values & (d.best_ms < hi).values)
            res['model ' + b] = summarize(rat, ms, mk & (d.model_ms >= lo).values & (d.model_ms < hi).values)
        res['model <10 ms'] = summarize(rat, ms, mk & (d.model_ms < 10).values)
        res['n_models_by_band'] = {b: int(d[(d.model_ms >= lo) & (d.model_ms < hi)].model.nunique()) for b, lo, hi in BANDS}
        for dv in DEVS:
            res[NAME[dv]] = summarize(rat, ms, mk & (d.device == dv).values)
        out[k] = res
    # share of architectures whose best single artifact differs from the per-device
    # optimum on at least one device by more than 10 %
    rat, _ = regret_of(d, pol['single artifact (oracle)'])
    d['r1'] = rat
    worst = d.groupby('model').r1.max()
    out['archs_single_costs_gt10pct'] = 100 * float((worst > 1.10).mean())
    out['archs_single_costs_gt25pct'] = 100 * float((worst > 1.25).mean())
    out['n_arch'] = int(d.model.nunique())
    return out


def main():
    res = {c: run(c) for c in ('cifar10', 'imagenet100')}
    lines = []
    for c, r in res.items():
        lines.append(f'== {c}  ({r["n_arch"]} architectures; single artifact costs >10% on some device for '
                     f'{r["archs_single_costs_gt10pct"]:.1f}% of archs, >25% for {r["archs_single_costs_gt25pct"]:.1f}%)')
        for k, v in r.items():
            if not isinstance(v, dict):
                continue
            lines.append(f'  {k}')
            for kk, s in v.items():
                if kk == 'n_models_by_band':
                    lines.append(f'    models per model-level band: {s}'); continue
                lines.append(f'    {kk:10s} n={s["n"]:5d} geo {s["geo"]:6.1f}%  ms {s["ms"]:5.2f}  '
                             f'>10% {s["gt10"]:5.1f}%  >25% {s["gt25"]:5.1f}%')
    # benchmarking budget (runs on the target device per model)
    lines.append('\nBenchmarking budget per new model on a known device class: exhaustive = 5 runs '
                 '(one per mode); selector / rules = 0; new device class: ~100 models x up to 5 modes once.')
    txt = '\n'.join(lines)
    print(txt)
    open('results/v83/one_or_many.txt', 'w').write(txt + '\n')
    json.dump(res, open('results/v83/one_or_many.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
