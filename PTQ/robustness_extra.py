#!/usr/bin/env python3
"""Additional robustness checks for the device-dependence result and a
device-only baseline for the selector tables.

Reproduces the rows added to Table VIII (winner robustness) and Table IX
(selection policies) of the TCE manuscript, revision v82:

  * device dependence restricted to "healthy" architectures, i.e. those whose
    converted FP32 TFLite accuracy is at least --healthy-acc (default 0.5);
  * pairwise device dependence for every pair of devices;
  * equivalence-band device dependence (eps = 0, 2, 5, 10 %);
  * the per-device-majority policy (device identity only, no model features),
    next to the global and per-architecture-majority policies.

All intervals are 95 % percentile bootstrap intervals clustered by
architecture (2000 resamples, seed 0).

Usage
-----
    # constrained pool from the shipped table (what the manuscript reports)
    python robustness_extra.py --csv dataset_final.csv --tag cifar10

    # both pools from a checkout of the released per-run records
    python robustness_extra.py --data-root /path/to/nn-dataset --tag cifar10
    python robustness_extra.py --data-root /path/to/nn-dataset \
        --run-subdir imagenet100 --acc-dir /path/to/imagenet100/stats --tag imagenet100

Outputs results/robustness_extra_<tag>.json, .md and _rows.tex.
"""
import argparse, collections, itertools, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robustness as R                                   # noqa: E402

MODES = R.MODES
DEVICES = R.DEVICES
NAMES = R.NAMES
FALLBACK = R.FALLBACK
EPS = (0.0, 0.02, 0.05, 0.10)
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- loading
def pool_from_csv(path, tau):
    """Constrained pool B from dataset_final.csv: (model, device) -> mode -> (lat, 0)."""
    df = pd.read_csv(path)
    pool, acc = {}, {}
    for r in df.itertuples():
        ref = r.acc_fp32
        acc[r.model] = ref
        c = {}
        for m in MODES:
            a, lat = getattr(r, f'acc_{m}'), getattr(r, f'lat_{m}')
            if pd.notna(a) and pd.notna(lat) and lat >= R.FLOOR and a >= (1 - tau) * ref:
                c[m] = (float(lat), 0.0)
        if c:
            pool[(r.model, r.device)] = c
    return pool, acc


def pools_from_root(root, run_subdir, acc_dir):
    """Both pools from the released per-run JSON records (same loader as
    robustness.py, with optional sub-directory / accuracy directory)."""
    import glob
    S = acc_dir or os.path.join(root, '_work', 'stats')
    Rdir = os.path.join(root, 'ab', 'nn', 'stat', 'run', 'tflite', run_subdir)
    acc = {m: json.load(open(f'{S}/all_models_accuracy_{m}.json')) for m in MODES}
    runs = collections.defaultdict(dict)
    for m in MODES:
        for f in glob.glob(f'{Rdir}/{m}/*/android_*.json'):
            r = json.load(open(f))
            if not (r.get('valid') and r.get('device_type') in DEVICES):
                continue
            d = r.get('duration')
            if not d or d < R.FLOOR:
                continue
            k = R.unit_key(r.get('unit'))
            runs[(r['model_name'], r['device_type'])][m] = (float(d), float(r.get(f'{k}_std_dev') or 0.0))
    A, B = R.pools(acc, runs)
    fp32 = {mo: v['accuracy'] for mo, v in acc['fp32'].items() if v and v.get('accuracy') is not None}
    return A, B, fp32


# ---------------------------------------------------------------- statistics
def by_model(pool):
    by = collections.defaultdict(dict)
    for (mo, dv), c in pool.items():
        by[mo][dv] = c
    return by


def winner(c):
    return min(c, key=lambda m: c[m][0])


def eps_sets(c, eps):
    b = min(v[0] for v in c.values())
    return {m for m, v in c.items() if v[0] <= b * (1 + eps)}


def dependence_rows(pool, devs, eps=0.0, keep=None):
    """Per-architecture indicator (1 = device-dependent) for the given device
    subset; eps > 0 uses disjoint eps-optimal sets instead of exact winners."""
    by = by_model(pool)
    out = {}
    for mo, d in by.items():
        if keep is not None and not keep(mo):
            continue
        if not all(dv in d for dv in devs):
            continue
        sets = [eps_sets(d[dv], eps) for dv in devs]
        out[mo] = 0.0 if set.intersection(*sets) else 1.0
    return out


def cluster_ci(values_by_arch, reps=2000):
    """Percentile bootstrap over architectures of the mean of a per-architecture
    indicator."""
    v = np.array(list(values_by_arch.values()), float)
    if len(v) == 0:
        return float('nan'), (float('nan'), float('nan')), 0
    idx = RNG.choice(len(v), (reps, len(v)), replace=True)
    s = v[idx].mean(1)
    return float(v.mean()), (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))), len(v)


def policy_rows(pool, policy):
    """Per-pair exact-match hit and latency ratio for a policy k -> mode."""
    keys = list(pool)
    hit = np.array([policy(k) == winner(pool[k]) for k in keys], float)
    rat = np.array([R.regret(pool[k], policy(k)) for k in keys], float)
    archs = np.array([k[0] for k in keys])
    return hit, rat, archs


def cluster_ci_rows(hit, rat, archs, reps=2000):
    groups = collections.defaultdict(list)
    for i, a in enumerate(archs):
        groups[a].append(i)
    G = [np.array(g) for g in groups.values()]
    geo = lambda r: float(np.exp(np.log(r).mean()) - 1)
    p90 = lambda r: float(np.percentile(r, 90) - 1)
    ex, gm, pn = [], [], []
    for _ in range(reps):
        b = np.concatenate([G[j] for j in RNG.choice(len(G), len(G), replace=True)])
        ex.append(hit[b].mean()); gm.append(geo(rat[b])); pn.append(p90(rat[b]))
    pct = lambda s: (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    return {'exact': float(hit.mean()), 'exact_ci': pct(ex),
            'geo_regret': geo(rat), 'geo_ci': pct(gm),
            'p90_regret': p90(rat), 'p90_ci': pct(pn), 'n': int(len(hit))}


def analyse(pool, fp32acc, healthy, tag):
    res = {'pairs': len(pool), 'architectures': len({k[0] for k in pool})}
    dev3 = list(DEVICES)
    # --- device dependence, exact and eps bands, all devices and pairwise
    dep = {}
    for eps in EPS:
        m, ci, n = cluster_ci(dependence_rows(pool, dev3, eps))
        dep[f'all3_eps{int(100*eps)}'] = {'rate': m, 'ci': ci, 'n': n}
        for a, b in itertools.combinations(dev3, 2):
            m, ci, n = cluster_ci(dependence_rows(pool, [a, b], eps))
            dep[f'{NAMES[a]}|{NAMES[b]}_eps{int(100*eps)}'] = {'rate': m, 'ci': ci, 'n': n}
    # --- healthy-model filter
    for thr in sorted({healthy, 0.3, 0.7}):
        keep = lambda mo, t=thr: fp32acc.get(mo, 0.0) >= t
        m, ci, n = cluster_ci(dependence_rows(pool, dev3, 0.0, keep))
        dep[f'all3_healthy{thr:g}'] = {'rate': m, 'ci': ci, 'n': n}
        for a, b in itertools.combinations(dev3, 2):
            m, ci, n = cluster_ci(dependence_rows(pool, [a, b], 0.0, keep))
            dep[f'{NAMES[a]}|{NAMES[b]}_healthy{thr:g}'] = {'rate': m, 'ci': ci, 'n': n}
    res['device_dependence'] = dep
    # --- policies
    pol = {}
    for m in MODES:
        pol[f'global_{m}'] = cluster_ci_rows(*policy_rows(pool, lambda k, m=m: m))
    devmaj = {}
    for dv in dev3:
        labs = [winner(c) for (mo, d), c in pool.items() if d == dv]
        if labs:
            devmaj[dv] = collections.Counter(labs).most_common(1)[0][0]
    res['per_device_majority_mode'] = {NAMES[d]: m for d, m in devmaj.items()}
    pol['per_device_majority'] = cluster_ci_rows(*policy_rows(pool, lambda k: devmaj[k[1]]))
    archmaj = R.arch_majority(pool)
    pol['per_arch_majority'] = cluster_ci_rows(*policy_rows(pool, lambda k: archmaj[k[0]]))
    res['policies'] = pol
    res['healthy_threshold'] = healthy
    res['tag'] = tag
    return res


# ---------------------------------------------------------------- reporting
def fmt(x, ci=None, pct=True):
    s = f'{100*x:.1f}' if pct else f'{x:.3f}'
    if ci is not None:
        s += f' [{100*ci[0]:.1f}, {100*ci[1]:.1f}]'
    return s


def report(res, poolname):
    L = [f'## Pool {poolname}: {res["pairs"]} pairs, {res["architectures"]} architectures', '']
    d = res['device_dependence']
    L.append('| Device dependence (%) | rate [95% CI] | n arch |')
    L.append('|---|---|---|')
    for eps in EPS:
        k = f'all3_eps{int(100*eps)}'
        L.append(f'| all three devices, eps={int(100*eps)}% | {fmt(d[k]["rate"], d[k]["ci"])} | {d[k]["n"]} |')
    thr = res['healthy_threshold']
    k = f'all3_healthy{thr:g}'
    L.append(f'| all three, FP32 TFLite acc >= {thr:g} | {fmt(d[k]["rate"], d[k]["ci"])} | {d[k]["n"]} |')
    for a, b in itertools.combinations(DEVICES, 2):
        k = f'{NAMES[a]}|{NAMES[b]}_eps0'
        L.append(f'| pairwise {NAMES[a]} vs {NAMES[b]} (exact) | {fmt(d[k]["rate"], d[k]["ci"])} | {d[k]["n"]} |')
    for a, b in itertools.combinations(DEVICES, 2):
        k = f'{NAMES[a]}|{NAMES[b]}_eps5'
        L.append(f'| pairwise {NAMES[a]} vs {NAMES[b]} (eps=5%) | {fmt(d[k]["rate"], d[k]["ci"])} | {d[k]["n"]} |')
    L += ['', f'Per-device majority modes: {res["per_device_majority_mode"]}', '',
          '| Policy | exact (%) | geo-mean regret (%) | p90 regret (%) |', '|---|---|---|---|']
    for name, p in res['policies'].items():
        L.append(f'| {name} | {fmt(p["exact"], p["exact_ci"])} | {fmt(p["geo_regret"], p["geo_ci"])} | '
                 f'{fmt(p["p90_regret"], p["p90_ci"])} |')
    return '\n'.join(L)


def latex_rows(res):
    d = res['device_dependence']; thr = res['healthy_threshold']
    r = lambda k: f'{100*d[k]["rate"]:.1f} [{100*d[k]["ci"][0]:.1f},\\,{100*d[k]["ci"][1]:.1f}]'
    rows = [
        f'FP32 TFLite accuracy $\\geq{thr:g}$ ($n={d[f"all3_healthy{thr:g}"]["n"]}$) & {r(f"all3_healthy{thr:g}")} & --- \\\\',
        f'Pairwise, Pixel~4 vs Tab~S7+ & {r("Pixel 4|Tab S7+_eps0")} & --- \\\\',
        f'Pairwise, Redmi~13 vs Pixel~4 & {r("Redmi 13|Pixel 4_eps0")} & --- \\\\',
        f'Pairwise, Redmi~13 vs Tab~S7+ & {r("Redmi 13|Tab S7+_eps0")} & --- \\\\',
    ] + [f'$\\epsilon={int(100*e)}\\%$ equivalence band & {r(f"all3_eps{int(100*e)}")} & --- \\\\' for e in EPS[1:]]
    p = res['policies']['per_device_majority']
    rows.append(f'% Table IX row: Per-device majority & {100*p["exact"]:.1f} [{100*p["exact_ci"][0]:.1f}, {100*p["exact_ci"][1]:.1f}] '
                f'& {100*p["geo_regret"]:.1f} [{100*p["geo_ci"][0]:.1f}, {100*p["geo_ci"][1]:.1f}] & {100*p["p90_regret"]:.1f}\\\\')
    return '\n'.join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='dataset_final.csv (constrained pool only)')
    ap.add_argument('--data-root', default=None, help='nn-dataset checkout (both pools)')
    ap.add_argument('--run-subdir', default='', help="e.g. 'imagenet100'")
    ap.add_argument('--acc-dir', default=None, help='directory with all_models_accuracy_<mode>.json')
    ap.add_argument('--tau', type=float, default=0.01)
    ap.add_argument('--healthy-acc', type=float, default=0.5)
    ap.add_argument('--tag', default='cifar10')
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    results, md = {}, [f'# robustness_extra — {a.tag}', '']
    if a.data_root:
        R.TAU = a.tau
        A, B, fp32 = pools_from_root(a.data_root, a.run_subdir, a.acc_dir)
        for name, P in (('B_constrained', B), ('A_unconstrained', A)):
            results[name] = analyse(P, fp32, a.healthy_acc, a.tag)
            md.append(report(results[name], name)); md.append('')
    else:
        B, fp32 = pool_from_csv(a.csv or 'dataset_final.csv', a.tau)
        results['B_constrained'] = analyse(B, fp32, a.healthy_acc, a.tag)
        md.append(report(results['B_constrained'], 'B_constrained')); md.append('')
        md.append('(Unconstrained pool A needs --data-root; the shipped CSV holds the constrained pool only.)')

    base = os.path.join(a.out_dir, f'robustness_extra_{a.tag}')
    json.dump(results, open(base + '.json', 'w'), indent=1)
    open(base + '.md', 'w').write('\n'.join(md) + '\n')
    open(base + '_rows.tex', 'w').write(latex_rows(results['B_constrained']) + '\n')
    print('\n'.join(md))
    print(f'\nwritten: {base}.json / .md / _rows.tex')


if __name__ == '__main__':
    main()
