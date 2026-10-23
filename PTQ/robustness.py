#!/usr/bin/env python3
"""Policy latency regret, tie/uncertainty robustness and bootstrap CIs,
computed entirely from the released per-run records."""
import argparse, collections, glob, json, os
import numpy as np, pandas as pd
from scipy import stats

MODES=['fp32','fp16','mixed','dynamic','int8']
DEVICES=['24040RN64Y','Pixel 4','SM-T970']
NAMES={'24040RN64Y':'Redmi 13','Pixel 4':'Pixel 4','SM-T970':'Tab S7+'}
FLOOR=1e5; TAU=0.01; NIT=20
FALLBACK=['fp32','fp16','dynamic','mixed','int8']
RNG=np.random.default_rng(0)

def unit_key(u):
    u=(u or '').lower()
    return 'npu' if 'npu' in u else ('cpu' if 'cpu' in u else 'gpu')

def load(root):
    S=os.path.join(root,'_work','stats'); R=os.path.join(root,'ab','nn','stat','run','tflite')
    acc={m:json.load(open(f'{S}/all_models_accuracy_{m}.json')) for m in MODES}
    runs=collections.defaultdict(dict)          # (model,device) -> mode -> (mean, sd)
    for m in MODES:
        for f in glob.glob(f'{R}/{m}/*/android_*.json'):
            r=json.load(open(f))
            if not(r.get('valid') and r.get('device_type') in DEVICES): continue
            d=r.get('duration')
            if not d or d<FLOOR: continue
            k=unit_key(r.get('unit'))
            runs[(r['model_name'],r['device_type'])][m]=(float(d),float(r.get(f'{k}_std_dev') or 0.0))
    return acc,runs

def pools(acc,runs):
    """Pool A: unconstrained (fastest valid mode).  Pool B: constrained at tau."""
    A,B={},{}
    for (mo,dv),by in runs.items():
        if len(by)>=2: A[(mo,dv)]={m:v for m,v in by.items()}
        ref=acc['fp32'].get(mo)
        if not ref or ref['accuracy']<=0: continue
        ref=ref['accuracy']
        c={m:v for m,v in by.items() if acc[m].get(mo) and acc[m][mo]['accuracy']>=ref*(1-TAU)}
        if c: B[(mo,dv)]=c
    return A,B

def winner(c): return min(c,key=lambda m:c[m][0])

def regret(c,pred):
    p=pred if pred in c else next(x for x in FALLBACK if x in c)
    return c[p][0]/min(v[0] for v in c.values())

def summarize_regret(pool,policy):
    r=np.array([regret(c,policy(k)) for k,c in pool.items()])
    ex=np.mean([policy(k)==winner(c) for k,c in pool.items()])
    return ex,np.median(r)-1,np.exp(np.log(r).mean())-1,np.percentile(r,90)-1

def arch_majority(pool):
    by=collections.defaultdict(list)
    for (mo,dv),c in pool.items(): by[mo].append(winner(c))
    return {mo:collections.Counter(v).most_common(1)[0][0] for mo,v in by.items()}

# ---------- tie robustness -------------------------------------------------
def margins(pool):
    out=[]
    for c in pool.values():
        s=sorted(((v[0],m) for m,v in c.items()))
        if len(s)<2: continue
        out.append(s[1][0]/s[0][0]-1)
    return np.array(out)

def welch_sig(pool,alpha=0.05):
    """fraction of pairs where the winner is significantly faster than runner-up"""
    sig=0;n=0
    for c in pool.values():
        s=sorted(((v[0],v[1],m) for m,v in c.items()))
        if len(s)<2: continue
        (m1,s1,_),(m2,s2,_)=s[0],s[1]
        se1,se2=s1**2/NIT,s2**2/NIT
        if se1+se2==0: sig+=1;n+=1;continue
        t=(m2-m1)/np.sqrt(se1+se2)
        df=(se1+se2)**2/max(se1**2/(NIT-1)+se2**2/(NIT-1),1e-300)
        if stats.t.sf(t,df)<alpha: sig+=1
        n+=1
    return sig/n,n

def eps_device_dependence(pool,eps):
    """device-dependent iff the eps-optimal winner SETS are disjoint across devices"""
    by=collections.defaultdict(dict)
    for (mo,dv),c in pool.items(): by[mo][dv]=c
    full=[mo for mo,d in by.items() if len(d)==3]
    dep=0
    for mo in full:
        sets=[]
        for dv in DEVICES:
            c=by[mo][dv]; b=min(v[0] for v in c.values())
            sets.append({m for m,v in c.items() if v[0]<=b*(1+eps)})
        inter=set.intersection(*sets)
        if not inter: dep+=1
    return dep/len(full),len(full)

def mc_perturb(pool,reps=1000):
    """resample every run mean from N(mean, sd/sqrt(20)); recompute labels"""
    by=collections.defaultdict(dict)
    for (mo,dv),c in pool.items(): by[mo][dv]=c
    full=[mo for mo,d in by.items() if len(d)==3]
    base={(mo,dv):winner(by[mo][dv]) for mo in full for dv in DEVICES}
    dep=[];stab=[]
    for _ in range(reps):
        lab={}
        for mo in full:
            for dv in DEVICES:
                c=by[mo][dv]
                draw={m:RNG.normal(v[0],v[1]/np.sqrt(NIT)) for m,v in c.items()}
                lab[(mo,dv)]=min(draw,key=draw.get)
        dep.append(np.mean([len({lab[(mo,dv)] for dv in DEVICES})>1 for mo in full]))
        stab.append(np.mean([lab[k]==base[k] for k in base]))
    return np.array(dep),np.array(stab),len(full)

# ---------- cluster bootstrap ---------------------------------------------
def cluster_bootstrap(pool,stat,reps=2000):
    by=collections.defaultdict(list)
    for k in pool: by[k[0]].append(k)
    archs=list(by); out=[]
    for _ in range(reps):
        pick=RNG.choice(len(archs),len(archs),replace=True)
        sub={}
        for i,j in enumerate(pick):
            for k in by[archs[j]]: sub[(f'{k[0]}#{i}',k[1])]=pool[k]
        v=stat(sub)
        if v is not None: out.append(v)
    out=np.array(out)
    return np.percentile(out,2.5),np.percentile(out,97.5)

def dep_rate(pool):
    by=collections.defaultdict(dict)
    for (mo,dv),c in pool.items(): by[mo][dv]=c
    full=[mo for mo,d in by.items() if len(d)==3]
    if not full: return None
    return np.mean([len({winner(by[mo][dv]) for dv in DEVICES})>1 for mo in full])

def arch_rate(pool):
    maj=arch_majority(pool)
    return np.mean([maj[k[0]]==winner(c) for k,c in pool.items()])

def global_rate(pool,m):
    return np.mean([m==winner(c) for c in pool.values()])

def geo_regret(pool,policy):
    r=np.array([regret(c,policy(k)) for k,c in pool.items()])
    return np.exp(np.log(r).mean())-1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True)
    a=ap.parse_args()
    acc,runs=load(a.data_root)
    A,B=pools(acc,runs)
    print(f'pool A (unconstrained): {len(A)} model-device pairs, {len({k[0] for k in A})} architectures')
    print(f'pool B (constrained tau={TAU}): {len(B)} pairs, {len({k[0] for k in B})} architectures\n')

    for nm,P in (('A unconstrained',A),('B constrained',B)):
        print(f'=== POOL {nm} ===')
        d,n=dep_rate(P),len({k[0] for k in P})
        lo,hi=cluster_bootstrap(P,dep_rate)
        print(f'  device-dependence  {100*d:5.1f}%  95% CI [{100*lo:.1f}, {100*hi:.1f}]')
        maj=arch_majority(P)
        for label,pol in [('arch-only oracle majority',lambda k:maj[k[0]])]+[
                          (f'global {m}',(lambda m: (lambda k: m))(m)) for m in MODES]:
            ex,med,geo,p90=summarize_regret(P,pol)
            print(f'  {label:26s} exact {100*ex:5.1f}%  regret med {100*med:5.1f}%  geo {100*geo:5.1f}%  p90 {100*p90:6.1f}%')
        lo,hi=cluster_bootstrap(P,arch_rate); print(f'    arch-only exact 95% CI [{100*lo:.1f}, {100*hi:.1f}]')
        lo,hi=cluster_bootstrap(P,lambda s:geo_regret(s,lambda k:maj.get(k[0].split("#")[0],'fp32')))
        print(f'    arch-only geo-regret 95% CI [{100*lo:.1f}, {100*hi:.1f}]')
        # ties
        mg=margins(P)
        print(f'  winner margin to runner-up: median {100*np.median(mg):.1f}%, '
              f'{100*np.mean(mg<0.02):.1f}% below 2%, {100*np.mean(mg<0.05):.1f}% below 5%')
        sg,ns=welch_sig(P)
        print(f'  winner significantly faster than runner-up (Welch, alpha=.05): {100*sg:.1f}% of {ns} pairs')
        for eps in (0.0,0.02,0.05,0.10):
            r,nn=eps_device_dependence(P,eps)
            print(f'  eps={100*eps:4.0f}%  device-dependence {100*r:5.1f}%  (n={nn} architectures)')
        dep,stab,nf=mc_perturb(P,reps=500)
        print(f'  Monte Carlo (500 draws, n={nf}): device-dependence {100*dep.mean():.1f}% '
              f'[{100*np.percentile(dep,2.5):.1f}, {100*np.percentile(dep,97.5):.1f}], '
              f'label stability {100*stab.mean():.1f}%\n')

if __name__=='__main__':
    main()
