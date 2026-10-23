#!/usr/bin/env python3
"""
Reproduces Table IX (mode-selection policies, Section VI) of the TCE manuscript.

Usage:
    python predictor.py --data-root /path/to/nn-dataset
    python predictor.py --csv dataset_final.csv      # skip rebuild, use shipped table

    # ImageNet-100 records (same corpus layout, different sub-directory and
    # accuracy files); writes dataset_final_imagenet100.csv next to this script
    python predictor.py --data-root /path/to/nn-dataset --run-subdir imagenet100 \
        --acc-dir /path/to/imagenet100/stats --out-csv dataset_final_imagenet100.csv

    # architecture folds re-drawn at random for every seed (GroupKFold itself is
    # deterministic, so without this flag all seeds give identical folds)
    python predictor.py --csv dataset_final.csv --shuffle-groups

--data-root must point at a checkout of
https://github.com/ABrain-One/nn-dataset  (needs ab/nn/stat/run/tflite/ and
_work/stats/all_models_accuracy_*.json).

To rerun on the authors' full-corpus accuracy snapshot, replace the five
all_models_accuracy_<mode>.json files with the internal ones; nothing else
changes.
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import argparse, collections, glob, json, os
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODES = ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']
DEVICES = ['24040RN64Y', 'Pixel 4', 'SM-T970']
FLOOR = 1e5      # ns; runs below 0.1 ms are implausible for a CNN inference
TAU = 0.01       # relative accuracy budget
FALLBACK = ['fp32', 'fp16', 'dynamic', 'mixed', 'int8']


def family(name):
    n = name.lower()
    for p in ('unq-', 'rag-', 'alt-', 'moe', 'gen'):
        if n.startswith(p):
            return p.strip('-')
    return 'named'


def build(root, run_subdir='', acc_dir=None):
    S = acc_dir or os.path.join(root, '_work', 'stats')
    R = os.path.join(root, 'ab', 'nn', 'stat', 'run', 'tflite', run_subdir)
    acc = {m: json.load(open(f'{S}/all_models_accuracy_{m}.json')) for m in MODES}
    recs = collections.defaultdict(dict)
    for m in MODES:
        for f in glob.glob(f'{R}/{m}/*/android_*.json'):
            r = json.load(open(f))
            if r.get('valid') and r.get('device_type') in DEVICES and r.get('duration'):
                recs[(r['model_name'], r['device_type'])][m] = r
    rows, feas = [], {}
    for (mo, d), by in recs.items():
        ref = acc['fp32'].get(mo)
        if not ref or ref['accuracy'] <= 0:
            continue
        ref = ref['accuracy']
        c = {}
        for m in MODES:
            a = acc[m].get(mo)
            if a and m in by and by[m]['duration'] >= FLOOR and a['accuracy'] >= ref * (1 - TAU):
                c[m] = by[m]['duration']
        if not c:
            continue
        feas[(mo, d)] = c
        r0 = by.get('fp32') or next(iter(by.values()))
        ci = r0.get('device_analytics', {}).get('cpu_info', {})
        f = {'model': mo, 'device': d, 'label': min(c, key=c.get), 'family': family(mo),
             'in_h': r0.get('in_dim_1', 0), 'in_w': r0.get('in_dim_2', 0),
             'in_c': r0.get('in_dim_3', 0),
             'size_fp32_kib': r0.get('original_model_size_kib', np.nan),
             'acc_fp32': ref, 'dev_cores': ci.get('cpu_cores', np.nan),
             'dev_ram_gb': round(r0.get('total_ram_kb', 0) / 1048576, 2),
             'dev_soc': ci.get('arm_architecture', {}).get('hardware', '?')}
        f['in_px'] = f['in_h'] * f['in_w']
        for m in MODES[1:]:
            a = acc[m].get(mo)
            f[f'acc_{m}'] = a['accuracy'] if a else np.nan
            f[f'accdrop_{m}'] = (ref - a['accuracy']) if a else np.nan
            f[f'size_{m}_kib'] = by[m].get('quantized_model_size_kib', np.nan) if m in by else np.nan
        for m in MODES:
            f[f'lat_{m}'] = c.get(m, np.nan)   # kept for regret only, never a feature
        rows.append(f)
    return pd.DataFrame(rows).reset_index(drop=True)


NUM_ARCH = (['in_h', 'in_w', 'in_c', 'in_px', 'size_fp32_kib', 'acc_fp32']
            + [f'acc_{m}' for m in MODES[1:]]
            + [f'accdrop_{m}' for m in MODES[1:]]
            + [f'size_{m}_kib' for m in MODES[1:]])
CAT_ARCH = ['family']
NUM_DEV = ['dev_cores', 'dev_ram_gb']
CAT_DEV = ['device', 'dev_soc']


def model(num, cat, kind):
    pre = ColumnTransformer([
        ('n', Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler())]), num),
        ('c', OneHotEncoder(handle_unknown='ignore'), cat)])
    clf = {'hgb': HistGradientBoostingClassifier(max_iter=300, learning_rate=.08,
                                                 max_leaf_nodes=15, l2_regularization=1.,
                                                 min_samples_leaf=10, random_state=0),
           'rf': RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                        random_state=0, n_jobs=-1),
           'lr': LogisticRegression(max_iter=3000)}[kind]
    return Pipeline([('p', pre), ('m', clf)])


def stats(d, yhat):
    """exact-match, median / geometric-mean / p90 latency regret, infeasible rate."""
    exact = (yhat == d['label'].values).mean()
    ratios, infeasible = [], 0
    for i in range(len(d)):
        c = {m: d[f'lat_{m}'][i] for m in MODES if not pd.isna(d[f'lat_{m}'][i])}
        p = yhat[i]
        if p not in c:
            infeasible += 1
            p = next(x for x in FALLBACK if x in c)
        ratios.append(c[p] / min(c.values()))
    r = np.array(ratios)
    return (exact, np.median(r) - 1, np.exp(np.log(r).mean()) - 1,
            np.percentile(r, 90) - 1, infeasible / len(d))


def show(name, t):
    print(f'{name:36s} exact {100*t[0]:5.1f}   regret med {100*t[1]:5.1f}%  '
          f'geo-mean {100*t[2]:5.1f}%  p90 {100*t[3]:6.1f}%   infeasible {100*t[4]:4.1f}%')


SHUFFLE_GROUPS = False   # set by --shuffle-groups


def folds(d, seed, k=5):
    """Architecture-grouped folds. Default: the frozen grouped split in
    grouped_cv.py (deterministic across numpy/scikit-learn versions,
    balanced by pair count). With --shuffle-groups the architectures are
    assigned to the k folds at random for every seed, so repeated seeds give
    genuinely different splits."""
    if not SHUFFLE_GROUPS:
        yield from GroupKFold(k).split(d, d.label, groups=d.model)
        return
    rng = np.random.RandomState(seed)
    archs = rng.permutation(pd.unique(d.model.values))
    fold_of = {a: i % k for i, a in enumerate(archs)}
    f = d.model.map(fold_of).values
    for i in range(k):
        yield np.where(f != i)[0], np.where(f == i)[0]


def cv(df, name, num, cat, kind, seeds=range(10)):
    out = []
    for s in seeds:
        rng = np.random.RandomState(s)
        order = {g: i for i, g in enumerate(rng.permutation(pd.unique(df.model.values)))}
        d = df.iloc[np.argsort([order[g] for g in df.model.values], kind='stable')].reset_index(drop=True)
        yhat = np.empty(len(d), dtype=object)
        for tr, te in folds(d, s):
            m = model(num, cat, kind)
            m.fit(d.iloc[tr][num + cat], d.iloc[tr].label)
            yhat[te] = m.predict(d.iloc[te][num + cat])
        out.append(stats(d, yhat))
    out = np.array(out)
    show(name, out.mean(0))
    if len(out) > 1:
        print(f'{"":36s} exact per seed: {np.round(100*out[:, 0], 1).tolist()}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--run-subdir', default='', help="sub-directory under ab/nn/stat/run/tflite, e.g. 'imagenet100'")
    ap.add_argument('--acc-dir', default=None, help='directory holding all_models_accuracy_<mode>.json')
    ap.add_argument('--out-csv', default='dataset_final.csv', help='where the rebuilt table is written')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--shuffle-groups', action='store_true', help='random architecture folds per seed')
    a = ap.parse_args()
    global SHUFFLE_GROUPS
    SHUFFLE_GROUPS = a.shuffle_groups
    if a.csv:
        df = pd.read_csv(a.csv)
    else:
        df = build(a.data_root, a.run_subdir, a.acc_dir)
        df.to_csv(a.out_csv, index=False)
    print(f'{len(df)} labeled model-device pairs over {df.model.nunique()} architectures, '
          f'tau={TAU}, floor={FLOOR/1e6:g} ms\n')

    show('global single choice (dynamic)', stats(df, np.array(['dynamic'] * len(df), dtype=object)))
    show('global single choice (fp16)', stats(df, np.array(['fp16'] * len(df), dtype=object)))
    devmaj = df.groupby('device').label.agg(lambda s: s.value_counts().index[0])
    print('per-device majority modes:', devmaj.to_dict())
    show('per-device majority (device only)', stats(df, df.device.map(devmaj).values.astype(object)))
    maj = df.groupby('model').label.agg(lambda s: s.value_counts().index[0])
    show('per-architecture oracle majority', stats(df, df.model.map(maj).values.astype(object)))
    seeds = range(a.seeds)
    A = cv(df, 'learned: offline features only', NUM_ARCH, CAT_ARCH, 'hgb', seeds)
    B = cv(df, 'learned: + device descriptor (GBT)', NUM_ARCH + NUM_DEV, CAT_ARCH + CAT_DEV, 'hgb', seeds)
    cv(df, 'learned: + device descriptor (RF)', NUM_ARCH + NUM_DEV, CAT_ARCH + CAT_DEV, 'rf', seeds)
    cv(df, 'learned: + device descriptor (LogReg)', NUM_ARCH + NUM_DEV, CAT_ARCH + CAT_DEV, 'lr', seeds)
    print('\ndevice-descriptor gain per seed (pp):', np.round(100 * (B[:, 0] - A[:, 0]), 1))

    # per-device / per-family breakdown and permutation importance, seed 0
    rng = np.random.RandomState(0)
    order = {g: i for i, g in enumerate(rng.permutation(pd.unique(df.model.values)))}
    d = df.iloc[np.argsort([order[g] for g in df.model.values], kind='stable')].reset_index(drop=True)
    yhat = np.empty(len(d), dtype=object)
    for tr, te in folds(d, 0):
        m = model(NUM_ARCH + NUM_DEV, CAT_ARCH + CAT_DEV, 'hgb')
        m.fit(d.iloc[tr][NUM_ARCH + NUM_DEV + CAT_ARCH + CAT_DEV], d.iloc[tr].label)
        yhat[te] = m.predict(d.iloc[te][NUM_ARCH + NUM_DEV + CAT_ARCH + CAT_DEV])
    d['pred'] = yhat
    print('\nper device: ', {k: round(100 * (g.pred == g.label).mean(), 1)
                             for k, g in d.groupby('device')})
    print('per family: ', {k: (len(g), round(100 * (g.pred == g.label).mean(), 1))
                           for k, g in d.groupby('family')})
    print('\nconfusion (rows=true, cols=predicted):')
    print(pd.crosstab(d.label, d.pred).reindex(index=MODES, columns=MODES, fill_value=0))

    cols = NUM_ARCH + NUM_DEV + CAT_ARCH + CAT_DEV
    m = model(NUM_ARCH + NUM_DEV, CAT_ARCH + CAT_DEV, 'rf').fit(df[cols], df.label)
    pi = permutation_importance(m, df[cols], df.label, n_repeats=10, random_state=0, n_jobs=-1)
    top = sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])[:8]
    print('\ntop features:', [(k, round(v, 3)) for k, v in top])


if __name__ == '__main__':
    main()
