#!/usr/bin/env python3
"""All new numbers of revision v83.

Inputs
  results/v83/dataset_full_cifar10.csv   predictor.py --data-root <nn-dataset> --out-csv ...
                                         (released constrained pool, tau = 0.99)
  results/v83/runs_long.csv              load_runs.py --data-root <nn-dataset>
  <nn-dataset>/_work/stats/all_models_accuracy_<mode>.json

Sections (printed and written to results/v83/v83_results.json)
  A  selection policies on the released constrained CIFAR-10 pool, with
     relative and absolute (ms) regret, feasibility-aware baselines, a
     feasibility-masked selector, clustered bootstrap CIs and paired tests
  B  leave-one-device-out and few-shot transfer to an unseen device
  C  fixed-backend (CPU-only / GPU-only) device dependence
  D  ImageNet-100 latency-only (accuracy-agnostic) selection
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import argparse, collections, json, os
import numpy as np, pandas as pd
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)

import predictor as P

MODES = P.MODES
DEVS = ['24040RN64Y', 'Pixel 4', 'SM-T970']
NAME = {'24040RN64Y': 'Redmi 13', 'Pixel 4': 'Pixel 4', 'SM-T970': 'Tab S7+'}
FALLBACK = P.FALLBACK
REPS = 2000
OUT = {}
TAG = ''
PICKS = {}


# ------------------------------------------------------------------ helpers
def feasible_acc(r):
    if 'feas' in r.index and isinstance(r['feas'], (list, tuple)):
        return r['feas']
    return _feasible_acc(r)


def _feasible_acc(r):
    """Modes satisfying the accuracy constraint -- known offline, before any
    benchmarking on the target device."""
    ref = r['acc_fp32']
    s = ['fp32']
    for m in MODES[1:]:
        a = r[f'acc_{m}']
        if pd.notna(a) and a >= ref * (1 - P.TAU):
            s.append(m)
    return s


def evaluate(d, pick):
    """Per-row selected-mode metrics (fallback as in predictor.py)."""
    hit, rat, ms = [], [], []
    for i, r in d.iterrows():
        c = {m: r[f'lat_{m}'] for m in MODES if pd.notna(r[f'lat_{m}'])}
        p = pick[i]
        hit.append(p == r['label'])          # exact match is scored before fallback
        if p not in c:
            p = next(x for x in FALLBACK if x in c)
        best = min(c.values())
        rat.append(c[p] / best)
        ms.append((c[p] - best) / 1e6)
    return np.array(hit, float), np.array(rat), np.array(ms)


def summary(hit, rat, ms):
    return dict(exact=100 * hit.mean(), geo=100 * (np.exp(np.log(rat).mean()) - 1),
                p90=100 * (np.percentile(rat, 90) - 1), ms_mean=ms.mean(),
                ms_p90=np.percentile(ms, 90))


def cluster_boot(groups, arrays, fn, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    g = pd.Series(range(len(groups))).groupby(np.asarray(groups)).apply(list)
    idx = [np.array(v) for v in g.values]
    out = []
    for _ in range(reps):
        take = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx[t] for t in take])
        out.append(fn(*[a[ii] for a in arrays]))
    return np.percentile(np.array(out), [2.5, 97.5], axis=0)


def policy_row(d, pick, name, store):
    h, r, m = evaluate(d, pick)
    s = summary(h, r, m)
    lo, hi = cluster_boot(d.model.values, [h, r, m],
                          lambda h, r, m: np.array(list(summary(h, r, m).values())))
    s['ci'] = {k: [lo[j], hi[j]] for j, k in enumerate(['exact', 'geo', 'p90', 'ms_mean', 'ms_p90'])}
    store[name] = dict(s=s, h=h, r=r, m=m)
    PICKS[name] = pick
    print(f'{name:44s} exact {s["exact"]:5.1f} [{lo[0]:4.1f},{hi[0]:4.1f}]  '
          f'geo {s["geo"]:5.1f} [{lo[1]:4.1f},{hi[1]:4.1f}]  p90 {s["p90"]:6.1f}  '
          f'ms mean {s["ms_mean"]:5.2f} [{lo[3]:4.2f},{hi[3]:4.2f}]  ms p90 {s["ms_p90"]:5.2f}')
    return s


def paired(d, a, b, store, label):
    A, B = store[a], store[b]
    diff = lambda h1, r1, m1, h2, r2, m2: np.array([
        100 * (h1.mean() - h2.mean()),
        100 * (np.exp(np.log(r1).mean()) - np.exp(np.log(r2).mean())),
        m1.mean() - m2.mean()])
    pt = diff(A['h'], A['r'], A['m'], B['h'], B['r'], B['m'])
    lo, hi = cluster_boot(d.model.values, [A['h'], A['r'], A['m'], B['h'], B['r'], B['m']], diff)
    print(f'  paired {label}: exact {pt[0]:+.1f} [{lo[0]:+.1f},{hi[0]:+.1f}] pp, '
          f'geo {pt[1]:+.1f} [{lo[1]:+.1f},{hi[1]:+.1f}] pp, ms {pt[2]:+.2f} [{lo[2]:+.2f},{hi[2]:+.2f}]')
    return dict(exact=[pt[0], lo[0], hi[0]], geo=[pt[1], lo[1], hi[1]], ms=[pt[2], lo[2], hi[2]])


def oof(d, num, cat, kind='hgb', masked=False, k=5):
    """Out-of-fold predictions under architecture-grouped CV (seed-0 order as
    in predictor.py). masked=True restricts the arg-max to modes that satisfy
    the accuracy constraint (known offline)."""
    yhat = np.empty(len(d), dtype=object)
    for tr, te in GroupKFold(k).split(d, d.label, groups=d.model):
        m = P.model(num, cat, kind).fit(d.iloc[tr][num + cat], d.iloc[tr].label)
        if not masked:
            yhat[te] = m.predict(d.iloc[te][num + cat])
            continue
        pr = m.predict_proba(d.iloc[te][num + cat])
        cls = list(m.classes_)
        for j, i in enumerate(te):
            fs = feasible_acc(d.iloc[i])
            order = np.argsort(-pr[j])
            yhat[i] = next((cls[o] for o in order if cls[o] in fs), cls[order[0]])
    return yhat


def ordered_pick(d, order_by_dev):
    return np.array([next((m for m in order_by_dev[r.device] if m in feasible_acc(r)), 'fp32')
                     for _, r in d.iterrows()], dtype=object)


def win_order(sub):
    vc = sub.label.value_counts()
    return list(vc.index) + [m for m in MODES if m not in vc.index]


# ------------------------------------------------------------------ section A
def section_a(d):
    print('\n=== A. Policies on released constrained CIFAR-10 pool:', len(d), 'pairs,',
          d.model.nunique(), 'architectures ===')
    st = {}
    print('median best latency (ms):', round(np.nanmedian(d[[f'lat_{m}' for m in MODES]].min(1)) / 1e6, 2))
    for m in MODES:
        policy_row(d, np.array([m] * len(d), dtype=object), f'global {m}', st)
    devmaj = d.groupby('device').label.agg(lambda s: s.value_counts().index[0])
    policy_row(d, d.device.map(devmaj).values.astype(object), 'per-device majority (naive)', st)
    glob_order = {dv: win_order(d) for dv in DEVS}
    policy_row(d, ordered_pick(d, glob_order), 'global order, feasibility-aware', st)
    dev_order = {dv: win_order(d[d.device == dv]) for dv in DEVS}
    policy_row(d, ordered_pick(d, dev_order), 'per-device order, feasibility-aware', st)
    print('  per-device orders:', {NAME[k]: v for k, v in dev_order.items()})
    maj = d.groupby('model').label.agg(lambda s: s.value_counts().index[0])
    policy_row(d, d.model.map(maj).values.astype(object), 'per-arch oracle majority (in-sample)', st)
    NA, CA = P.NUM_ARCH, P.CAT_ARCH
    ND, CD = P.NUM_ARCH + P.NUM_DEV, P.CAT_ARCH + P.CAT_DEV
    policy_row(d, oof(d, NA, CA), 'learned offline (GBT)', st)
    policy_row(d, oof(d, NA, CA, masked=True), 'learned offline (GBT, masked)', st)
    policy_row(d, oof(d, ND, CD, 'rf'), 'learned +dev (RF)', st)
    policy_row(d, oof(d, ND, CD), 'learned +dev (GBT)', st)
    policy_row(d, oof(d, ND, CD, masked=True), 'learned +dev (GBT, masked)', st)
    pr = {}
    for a, b in [('learned +dev (GBT, masked)', 'per-device order, feasibility-aware'),
                 ('learned +dev (GBT, masked)', 'per-arch oracle majority (in-sample)'),
                 ('learned +dev (GBT, masked)', 'global dynamic'),
                 ('learned +dev (GBT, masked)', 'learned offline (GBT, masked)'),
                 ('learned +dev (GBT, masked)', 'learned +dev (GBT)')]:
        pr[f'{a} vs {b}'] = paired(d, a, b, st, f'{a} - {b}')
    pk = d[['model', 'device', 'label']].copy()
    pk['best_ms'] = d[[f'lat_{m}' for m in MODES]].min(axis=1) / 1e6
    for k, v in st.items():
        pk[k + '|ratio'] = v['r']; pk[k + '|ms'] = v['m']; pk[k + '|hit'] = v['h']
    pk.to_csv(f'results/v83/picks_A{TAG}.csv', index=False)
    # per-device exact match of the masked selector
    h = st['learned +dev (GBT, masked)']['h']
    perdev = {NAME[dv]: 100 * h[d.device.values == dv].mean() for dv in DEVS}
    print('  masked GBT per device exact:', {k: round(v, 1) for k, v in perdev.items()})
    # label distribution
    lab = (100 * d.label.value_counts(normalize=True)).round(1).to_dict()
    labdev = {NAME[dv]: (100 * d[d.device == dv].label.value_counts(normalize=True)).round(1).to_dict() for dv in DEVS}
    print('  label share:', lab, '\n  per device:', labdev)
    OUT['A'] = dict(n_pairs=len(d), n_arch=int(d.model.nunique()),
                    policies={k: v['s'] for k, v in st.items()}, paired=pr,
                    masked_per_device=perdev, label_share=lab, label_share_dev=labdev,
                    dev_order={NAME[k]: v for k, v in dev_order.items()})


# ------------------------------------------------------------------ section B
def section_b(d):
    print('\n=== B. Unseen-device transfer (leave one device out) ===')
    ND, CD = P.NUM_ARCH + P.NUM_DEV, P.CAT_ARCH + P.CAT_DEV
    NA, CA = P.NUM_ARCH, P.CAT_ARCH
    res = collections.defaultdict(dict)
    for dv in DEVS:
        tr_all, te = d[d.device != dv], d[d.device == dv].reset_index(drop=True)
        rows = {}
        # (i) order learned on the two other devices
        rows['global order from other devices'] = ordered_pick(te, {dv: win_order(tr_all)})
        # (ii) offline features only, trained on other devices
        m = P.model(NA, CA, 'hgb').fit(tr_all[NA + CA], tr_all.label)
        rows['offline GBT, other devices'] = mask_pred(m, te, NA + CA)
        # (iii) with catalog descriptor (cores, RAM, SoC unseen)
        m = P.model(ND, CD, 'hgb').fit(tr_all[ND + CD], tr_all.label)
        rows['+descriptor GBT, other devices'] = mask_pred(m, te, ND + CD)
        # (iv) few-shot: k architectures benchmarked on the new device
        for k in (25, 50, 100, 200):
            accs = []
            for seed in range(5):
                rng = np.random.default_rng(seed)
                cal = set(rng.choice(te.model.unique(), k, replace=False))
                trk = pd.concat([tr_all, te[te.model.isin(cal)]])
                tek = te[~te.model.isin(cal)].reset_index(drop=True)
                m = P.model(ND, CD, 'hgb').fit(trk[ND + CD], trk.label)
                h, r, ms = evaluate(tek, mask_pred(m, tek, ND + CD))
                accs.append(list(summary(h, r, ms).values()))
            res[NAME[dv]][f'few-shot k={k}'] = dict(zip(['exact', 'geo', 'p90', 'ms_mean', 'ms_p90'],
                                                        np.mean(accs, 0)))
        # in-distribution reference: per-device rows of the grouped-CV selector
        for name, pick in rows.items():
            h, r, ms = evaluate(te, pick)
            res[NAME[dv]][name] = summary(h, r, ms)
        for name, s in res[NAME[dv]].items():
            print(f'  {NAME[dv]:9s} {name:34s} exact {s["exact"]:5.1f}  geo {s["geo"]:6.1f}  '
                  f'p90 {s["p90"]:6.1f}  ms {s["ms_mean"]:5.2f}')
    OUT['B'] = res


def mask_pred(m, te, cols):
    pr = m.predict_proba(te[cols])
    cls = list(m.classes_)
    out = np.empty(len(te), dtype=object)
    for i in range(len(te)):
        fs = feasible_acc(te.iloc[i])
        o = np.argsort(-pr[i])
        out[i] = next((cls[j] for j in o if cls[j] in fs), cls[o[0]])
    return out


# ------------------------------------------------------------------ section C
def dependence(pool, eps=0.0):
    """pool: (model, device) -> {mode: latency}. Share of architectures labeled
    on all three devices whose eps-optimal sets are not all intersecting
    (eps=0: exact winner differs)."""
    by = collections.defaultdict(dict)
    for (mo, dv), c in pool.items():
        by[mo][dv] = c
    flags = []
    for mo, dd in by.items():
        if len(dd) < 3:
            continue
        sets = []
        for c in dd.values():
            b = min(c.values())
            sets.append({m for m, v in c.items() if v <= b * (1 + eps)} if eps > 0 else {min(c, key=c.get)})
        flags.append(len(set.intersection(*sets)) == 0)
    return 100 * np.mean(flags), len(flags), np.array(flags)


def build_pool(runs, acc, col, constrained=True, corpus='cifar10'):
    r = runs[(runs.corpus == corpus) & runs.valid & runs[col].notna() & (runs[col] >= 0.1)]
    pool = collections.defaultdict(dict)
    for t in r.itertuples():
        pool[(t.model, t.device)][t.mode] = getattr(t, col)
    if not constrained:
        return {k: v for k, v in pool.items() if len(v) >= 2}
    out = {}
    for (mo, dv), c in pool.items():
        ref = acc['fp32'].get(mo)
        if not ref or ref['accuracy'] <= 0:
            continue
        cc = {m: v for m, v in c.items()
              if acc[m].get(mo) and acc[m][mo]['accuracy'] >= ref['accuracy'] * (1 - P.TAU)}
        if cc:
            out[(mo, dv)] = cc
    return out


def section_c(runs, acc, constrained_corpus='cifar10'):
    print('\n=== C. Fixed-backend device dependence ===')
    runs = runs.copy()
    runs['best_ms'] = runs['duration'] / 1e6
    res = {}
    combos = (('cifar10', True), ('cifar10', False), ('imagenet100', False)) if constrained_corpus == 'cifar10' \
        else (('imagenet100', True),)
    for corpus, constrained in combos:
        for col, lab in (('best_ms', 'fastest backend'), ('cpu_ms', 'CPU only'), ('gpu_ms', 'GPU only')):
            pool = build_pool(runs, acc, col, constrained, corpus)
            row = {}
            for eps in (0.0, 0.05):
                rate, n, fl = dependence(pool, eps)
                lo, hi = np.percentile([np.random.default_rng(s).choice(fl, len(fl)).mean() * 100
                                        for s in range(REPS)], [2.5, 97.5])
                row[f'eps{int(eps*100)}'] = [rate, lo, hi]
                row['n'] = n
            key = f'{corpus}/{"constrained" if constrained else "unconstrained"}/{lab}'
            res[key] = row
            print(f'  {key:42s} n={row["n"]:5d}  exact {row["eps0"][0]:5.1f} '
                  f'[{row["eps0"][1]:4.1f},{row["eps0"][2]:4.1f}]  eps5 {row["eps5"][0]:5.1f} '
                  f'[{row["eps5"][1]:4.1f},{row["eps5"][2]:4.1f}]')
    # how often the winning backend differs between the winning mode and FP32
    r = runs[(runs.corpus == 'cifar10') & runs.valid]
    share = r.groupby('device').unit.value_counts(normalize=True).unstack().round(3) * 100
    print('  fastest-backend share (CIFAR-10, %):\n', share)
    OUT['C'] = res
    OUT['C_backend_share'] = share.to_dict()


# ------------------------------------------------------------------ section D
def latency_only_table(runs, corpus):
    r = runs[(runs.corpus == corpus) & runs.valid & (runs.duration >= P.FLOOR)]
    rows = []
    for (mo, dv), g in r.groupby(['model', 'device']):
        c = dict(zip(g['mode'], g['duration']))
        if len(c) < 2:
            continue
        f0 = g.iloc[0]
        row = {'model': mo, 'device': dv, 'label': min(c, key=c.get), 'family': P.family(mo),
               'in_h': f0.in_h, 'in_w': f0.in_w, 'in_c': f0.in_c, 'in_px': f0.in_h * f0.in_w,
               'size_fp32_kib': f0.size_fp32_kib, 'dev_cores': f0.cores,
               'dev_ram_gb': round(f0.total_ram_kb / 1048576, 2), 'dev_soc': f0.soc}
        sz = dict(zip(g['mode'], g['size_kib']))
        for m in MODES[1:]:
            row[f'size_{m}_kib'] = sz.get(m, np.nan)
        for m in MODES:
            row[f'lat_{m}'] = c.get(m, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def section_d(runs):
    print('\n=== D. Latency-only (accuracy-agnostic) selection ===')
    NA = ['in_h', 'in_w', 'in_c', 'in_px', 'size_fp32_kib'] + [f'size_{m}_kib' for m in MODES[1:]]
    CA = ['family']
    ND, CD = NA + ['dev_cores', 'dev_ram_gb'], CA + ['device', 'dev_soc']
    res = {}
    for corpus in ('imagenet100', 'cifar10'):
        d = latency_only_table(runs, corpus)
        d = d.sample(frac=1, random_state=0).reset_index(drop=True)
        print(f'-- {corpus}: {len(d)} pairs, {d.model.nunique()} architectures; '
              f'median best latency {np.nanmedian(d[[f"lat_{m}" for m in MODES]].min(1))/1e6:.1f} ms')
        st = {}
        for m in ('dynamic', 'fp16', 'mixed', 'int8'):
            policy_row(d, np.array([m] * len(d), dtype=object), f'global {m}', st)
        devmaj = d.groupby('device').label.agg(lambda s: s.value_counts().index[0])
        policy_row(d, d.device.map(devmaj).values.astype(object), 'per-device majority', st)
        maj = d.groupby('model').label.agg(lambda s: s.value_counts().index[0])
        policy_row(d, d.model.map(maj).values.astype(object), 'per-arch oracle majority (in-sample)', st)
        policy_row(d, oof(d, NA, CA), 'learned offline (GBT)', st)
        policy_row(d, oof(d, ND, CD), 'learned +dev (GBT)', st)
        pr = paired(d, 'learned +dev (GBT)', 'per-device majority', st, 'GBT - device majority')
        pr2 = paired(d, 'learned +dev (GBT)', 'learned offline (GBT)', st, 'GBT+dev - GBT offline')
        res[corpus] = dict(n_pairs=len(d), n_arch=int(d.model.nunique()),
                           policies={k: v['s'] for k, v in st.items()},
                           paired_vs_devmaj=pr, paired_vs_offline=pr2,
                           devmaj={NAME[k]: v for k, v in devmaj.to_dict().items()})
    OUT['D'] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--csv', default='results/v83/dataset_full_cifar10.csv')
    ap.add_argument('--runs', default='results/v83/runs_long.csv')
    ap.add_argument('--sections', default='ABCD')
    ap.add_argument('--acc-dir', default=None, help='all_models_accuracy_<mode>.json (default: <data-root>/_work/stats)')
    ap.add_argument('--corpus', default='cifar10', help="corpus for section C constrained rows: cifar10 | imagenet100")
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    # same seed-0 architecture order as predictor.py
    rng = np.random.RandomState(0)
    order = {g: i for i, g in enumerate(rng.permutation(pd.unique(d.model.values)))}
    d = d.iloc[np.argsort([order[g] for g in d.model.values], kind='stable')].reset_index(drop=True)
    d['feas'] = [_feasible_acc(r) for _, r in d.iterrows()]
    runs = pd.read_csv(a.runs)
    accdir = a.acc_dir or f'{a.data_root}/_work/stats'
    acc = {m: json.load(open(f'{accdir}/all_models_accuracy_{m}.json')) for m in MODES}
    global TAG
    TAG = a.tag
    if 'A' in a.sections: section_a(d)
    if 'B' in a.sections: section_b(d)
    if 'C' in a.sections: section_c(runs, acc, a.corpus)
    if 'D' in a.sections: section_d(runs)
    path = f'results/v83/v83_results_{a.sections}{TAG}.json'
    json.dump(OUT, open(path, 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('written', path)


if __name__ == '__main__':
    main()
