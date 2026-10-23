#!/usr/bin/env python3
"""Cluster-bootstrap CIs for the learned selector, from out-of-fold predictions.
Usage: python predictor_ci.py [dataset_final.csv]   (Table IX brackets)."""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import numpy as np, pandas as pd, collections
from grouped_cv import GroupKFold   # deterministic drop-in; see grouped_cv.py (sklearn >= 1.6 changed GroupKFold's fold assignment)
import predictor as P
MODES=P.MODES
import sys
df=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else 'dataset_final.csv')
rng=np.random.default_rng(0)
LAT=df[[f'lat_{m}' for m in MODES]].to_numpy(float)
BEST=np.nanmin(LAT,axis=1)

def oof(num,cat,kind,seed=0):
    r=np.random.RandomState(seed)
    order={g:i for i,g in enumerate(r.permutation(pd.unique(df.model.values)))}
    idx=np.argsort([order[g] for g in df.model.values],kind='stable')
    d=df.iloc[idx].reset_index(drop=True)
    yh=np.empty(len(d),dtype=object)
    for tr,te in GroupKFold(5).split(d,d.label,groups=d.model):
        m=P.model(num,cat,kind); m.fit(d.iloc[tr][num+cat],d.iloc[tr].label)
        yh[te]=m.predict(d.iloc[te][num+cat])
    inv=np.argsort(idx)                      # back to original row order
    return yh[inv]

def per_row(pred):
    hit=(pred==df.label.values).astype(float); rat=np.empty(len(df))
    for i,p in enumerate(pred):
        c={m:LAT[i,j] for j,m in enumerate(MODES) if not np.isnan(LAT[i,j])}
        q=p if p in c else next(x for x in P.FALLBACK if x in c)
        rat[i]=c[q]/BEST[i]
    return hit,rat

by=collections.defaultdict(list)
for i,mo in enumerate(df.model.values): by[mo].append(i)
ARCHS=list(by); IDX=[np.array(by[a]) for a in ARCHS]
def boot_idx(reps=2000):
    return [np.concatenate([IDX[j] for j in rng.choice(len(ARCHS),len(ARCHS),replace=True)])
            for _ in range(reps)]
BI=boot_idx()

def ci(v,f=np.mean):
    s=np.array([f(v[b]) for b in BI]); return np.percentile(s,2.5),np.percentile(s,97.5)
geo=lambda r:np.exp(np.log(r).mean())-1
p90=lambda r:np.percentile(r,90)-1

runs={}
for name,num,cat,kind in [
    ('learned, offline features only',P.NUM_ARCH,P.CAT_ARCH,'hgb'),
    ('learned, + device descr. (GBT)',P.NUM_ARCH+P.NUM_DEV,P.CAT_ARCH+P.CAT_DEV,'hgb'),
    ('learned, + device descr. (RF)',P.NUM_ARCH+P.NUM_DEV,P.CAT_ARCH+P.CAT_DEV,'rf')]:
    pred=oof(num,cat,kind); hit,rat=per_row(pred); runs[name]=(hit,rat)
    print(f'{name:32s} exact {100*hit.mean():5.1f} [{100*ci(hit)[0]:.1f}, {100*ci(hit)[1]:.1f}]  '
          f'geo {100*geo(rat):4.1f} [{100*ci(rat,geo)[0]:.1f}, {100*ci(rat,geo)[1]:.1f}]  '
          f'p90 {100*p90(rat):5.1f} [{100*ci(rat,p90)[0]:.1f}, {100*ci(rat,p90)[1]:.1f}]')

# baselines on the same rows
maj=df.groupby('model').label.agg(lambda s:s.value_counts().index[0])
devmaj=df.groupby('device').label.agg(lambda s:s.value_counts().index[0])
for name,pred in [('per-architecture oracle major.',df.model.map(maj).values.astype(object)),
                  ('per-device majority (dev. only)',df.device.map(devmaj).values.astype(object)),
                  ('global: dynamic INT8',np.array(['dynamic']*len(df),dtype=object)),
                  ('global: FP16',np.array(['fp16']*len(df),dtype=object))]:
    hit,rat=per_row(pred)
    print(f'{name:32s} exact {100*hit.mean():5.1f} [{100*ci(hit)[0]:.1f}, {100*ci(hit)[1]:.1f}]  '
          f'geo {100*geo(rat):4.1f} [{100*ci(rat,geo)[0]:.1f}, {100*ci(rat,geo)[1]:.1f}]  '
          f'p90 {100*p90(rat):5.1f} [{100*ci(rat,p90)[0]:.1f}, {100*ci(rat,p90)[1]:.1f}]')

# paired gain: device descriptor vs offline-only
ha=runs['learned, offline features only'][0]; hb=runs['learned, + device descr. (GBT)'][0]
d=hb-ha; s=np.array([d[b].mean() for b in BI])
print(f'\ndevice-descriptor gain {100*d.mean():.1f} pp, 95% CI [{100*np.percentile(s,2.5):.1f}, '
      f'{100*np.percentile(s,97.5):.1f}], P(gain>0)={100*(s>0).mean():.1f}%')
ra=runs['learned, offline features only'][1]; rb=runs['learned, + device descr. (GBT)'][1]
sr=np.array([geo(rb[b])-geo(ra[b]) for b in BI])
print(f'geo-regret change {100*(geo(rb)-geo(ra)):.1f} pp, 95% CI [{100*np.percentile(sr,2.5):.1f}, {100*np.percentile(sr,97.5):.1f}]')

# paired: selector vs per-architecture oracle majority
hm,rm=per_row(df.model.map(maj).values.astype(object))
se=np.array([(hb[b]-hm[b]).mean() for b in BI])
sg=np.array([geo(rb[b])-geo(rm[b]) for b in BI])
print(f'\nselector vs oracle majority: exact {100*(hb-hm).mean():+.1f} pp CI [{100*np.percentile(se,2.5):+.1f}, {100*np.percentile(se,97.5):+.1f}], P(>0)={100*(se>0).mean():.1f}%')
print(f'selector vs oracle majority: geo-regret {100*(geo(rb)-geo(rm)):+.1f} pp CI [{100*np.percentile(sg,2.5):+.1f}, {100*np.percentile(sg,97.5):+.1f}], P(<0)={100*(sg<0).mean():.1f}%')

# paired: selector vs per-device majority (device identity only)
hd,rd=per_row(df.device.map(devmaj).values.astype(object))
se=np.array([(hb[b]-hd[b]).mean() for b in BI])
sg=np.array([geo(rb[b])-geo(rd[b]) for b in BI])
print(f'selector vs per-device majority: exact {100*(hb-hd).mean():+.1f} pp CI [{100*np.percentile(se,2.5):+.1f}, {100*np.percentile(se,97.5):+.1f}], P(>0)={100*(se>0).mean():.1f}%')
print(f'selector vs per-device majority: geo-regret {100*(geo(rb)-geo(rd)):+.1f} pp CI [{100*np.percentile(sg,2.5):+.1f}, {100*np.percentile(sg,97.5):+.1f}], P(<0)={100*(sg<0).mean():.1f}%')
