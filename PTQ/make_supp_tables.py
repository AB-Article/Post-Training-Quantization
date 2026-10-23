#!/usr/bin/env python3
"""Regenerate the LaTeX fragments pasted into Supplementary.tex (v83/v84):
  results/v83/supp_s6.tex      Table S6  (device-dependence grid, CIFAR-10)
  results/v83/supp_new.tex     Tables S12-S17 (policies, paired tests, strata,
                               fixed backend, latency-only, CIFAR-10 transfer)
  results/v83/supp_im100.tex   Tables S18-S20 (ImageNet-100 policies, transfer, trees)
Inputs: results/v83/*.json, picks_A*.csv.     python make_supp_tables.py"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
import numpy as np, pandas as pd
R = 'results/v83/'


def fci(x, fmt='%.1f'):
    return (fmt + ' [' + fmt + ',\\,' + fmt + ']') % tuple(x)


def s6():
    d = json.load(open(R + 'robustness_extra_cifar10_root.json'))['B_constrained']['device_dependence']
    f = lambda k: '%.1f [%.1f,\\,%.1f]' % (100 * d[k]['rate'], 100 * d[k]['ci'][0], 100 * d[k]['ci'][1])
    L = [r"""\begin{table}[!htbp]
\centering
\caption{Robustness of the device-dependence rate on the released constrained
pool ($\tau=0.99$; $3{,}539$ labeled pairs, $1{,}185$ architectures). An architecture
is device-dependent when its $\epsilon$-optimal mode sets are disjoint across
the listed devices ($\epsilon=0$: exact winners differ). The last column
restricts the pool to architectures whose converted FP32 TFLite graph reaches
at least $50\%$ CIFAR-10 accuracy (number of architectures in parentheses).
Brackets are 95\% bootstrap intervals clustered by architecture; $n$ is the
number of architectures labeled on all listed devices.}
\label{tab:robust_extra}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrlllll@{}}
\toprule
\textbf{Devices} & $n$ & \textbf{Exact} & $\epsilon=2\%$ & $\epsilon=5\%$ & $\epsilon=10\%$ & \textbf{FP32 acc.\ $\geq50\%$} \\
\midrule"""]
    for n, k in [('All three devices', 'all3'), ('Redmi~13 vs Pixel~4', 'Redmi 13|Pixel 4'),
                 ('Redmi~13 vs Tab~S7+', 'Redmi 13|Tab S7+'), ('Pixel~4 vs Tab~S7+', 'Pixel 4|Tab S7+')]:
        L.append(f"{n} & {d[k+'_eps0']['n']} & {f(k+'_eps0')} & {f(k+'_eps2')} & {f(k+'_eps5')} & "
                 f"{f(k+'_eps10')} & {f(k+'_healthy0.5')} ({d[k+'_healthy0.5']['n']}) \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(R + 'supp_s6.tex', 'w').write('\n'.join(L))


def lodo(js, label, caption):
    B = json.load(open(R + js))['B']
    L = [r"""\begin{table}[!htbp]
\centering
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}l rrrr rrrr rrrr@{}}
\toprule
& \multicolumn{4}{c}{\textbf{Redmi 13}} & \multicolumn{4}{c}{\textbf{Pixel 4}} & \multicolumn{4}{c}{\textbf{Tab S7+}}\\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
\textbf{Policy} & exact & geo & p90 & ms & exact & geo & p90 & ms & exact & geo & p90 & ms\\
\midrule"""]
    order = [('global order from other devices', 'Win-rate order of other devices'),
             ('offline GBT, other devices', 'GBT, offline features, zero-shot'),
             ('+descriptor GBT, other devices', 'GBT, + descriptor, zero-shot')] + \
            [(f'few-shot k={k}', f'GBT, + device, few-shot $k={k}$') for k in (25, 50, 100, 200)]
    for key, lab in order:
        cells = []
        for dv in ('Redmi 13', 'Pixel 4', 'Tab S7+'):
            s = B[dv][key]; cells += ['%.1f' % s['exact'], '%.1f' % s['geo'], '%.0f' % s['p90'], '%.2f' % s['ms_mean']]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
        if key.startswith('+desc'): L.append(r'\midrule')
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return '\n'.join(L)


NAMES = [('global fp32', 'Global: FP32'), ('global fp16', 'Global: FP16'), ('global mixed', 'Global: mixed INT8'),
         ('global dynamic', 'Global: dynamic INT8'), ('global int8', 'Global: static INT8'),
         ('global order, feasibility-aware', 'Global win-rate order, feasible'),
         ('per-device majority (naive)', 'Device-only: majority'),
         ('per-device order, feasibility-aware', 'Device-only: feasible order'),
         ('per-arch oracle majority (in-sample)', 'Per-arch.\\ majority (in-sample oracle)'),
         ('learned offline (GBT)', 'GBT, offline features'), ('learned offline (GBT, masked)', 'GBT, offline features, masked'),
         ('learned +dev (RF)', 'RF, + device'), ('learned +dev (GBT)', 'GBT, + device'),
         ('learned +dev (GBT, masked)', '\\textbf{GBT, + device, masked}')]


def policy_rows(P):
    L = []
    for k, n in NAMES:
        s = P[k]
        L.append(f"{n} & {fci([s['exact']]+s['ci']['exact'])} & {fci([s['geo']]+s['ci']['geo'])} & {s['p90']:.1f} & "
                 f"{fci([s['ms_mean']]+s['ci']['ms_mean'],'%.2f')} & {s['ms_p90']:.2f} \\\\")
        if k in ('global int8', 'per-arch oracle majority (in-sample)'): L.append(r'\midrule')
    return L


HEAD = r"""\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lllrlr@{}}
\toprule
\textbf{Policy} & \textbf{Exact (\%)} & \textbf{Geo.\ regret (\%)} & \textbf{p90 (\%)} & \textbf{Mean (ms)} & \textbf{p90 (ms)}\\
\midrule"""


def strata_rows(path):
    p = pd.read_csv(path); bl = ['<2', '2-10', '>=10']
    p['band'] = pd.cut(p.best_ms, [0, 2, 10, 1e9], labels=bl, right=False)
    pols = [('global dynamic', 'Always dynamic INT8'), ('per-device order, feasibility-aware', 'Device-only feasible order'),
            ('per-arch oracle majority (in-sample)', 'Per-arch.\\ majority (oracle)'), ('learned +dev (GBT, masked)', 'GBT, + device, masked')]
    L = []
    for k, n in pols:
        L.append(n + ' & ' + ' & '.join('%.1f / %.2f' % (100 * (np.exp(np.log(p[p.band == b][k + '|ratio']).mean()) - 1),
                                                            p[p.band == b][k + '|ms'].mean()) for b in bl) + r' \\')
    return p.band.value_counts(), L


def supp_new():
    A = json.load(open(R + 'v83_results_A.json'))['A']; C = json.load(open(R + 'v83_results_C.json'))['C']
    D = json.load(open(R + 'v83_results_D.json'))['D']; C2 = json.load(open(R + 'v83_results_C_im100.json'))['C']
    L = [r"""\section{Additional Analyses}
These tables and figures support Sections~V-B to V-C and~VI of the main paper. All
numbers are computed on the released constrained pools; learned rows use
architecture-grouped five-fold cross-validation; brackets are 95\% bootstrap
intervals clustered by architecture (2{,}000 resamples). ``Masked'' selectors
ship the most probable mode among those satisfying the accuracy constraint,
which is known before any on-device measurement.

\begin{table}[!htbp]
\centering
\caption{All CIFAR-10 selection policies ($3{,}539$ pairs, $1{,}185$
architectures, $\tau=0.99$). Exact match is scored before the fallback to a
valid mode; regret after it. ms: mean and 90th-percentile absolute regret.}
\label{tab:policies_full}""", HEAD] + policy_rows(A['policies']) + ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]
    lab = {'per-device order, feasibility-aware': 'vs device-only feasible order', 'per-arch oracle majority (in-sample)': 'vs per-arch.\\ majority (oracle)',
           'global dynamic': 'vs always dynamic INT8', 'learned offline (GBT, masked)': 'vs same model without device',
           'learned +dev (GBT)': 'vs unmasked GBT'}
    L += [r"""\begin{table}[!htbp]
\centering
\caption{Paired differences of the masked GBT selector (with device
descriptor) against other policies on the same pairs: exact match and
geometric-mean regret in percentage points, absolute regret in ms. Brackets:
95\% intervals clustered by architecture.}
\label{tab:paired}
\footnotesize
\begin{tabular}{@{}llll@{}}
\toprule
\textbf{Comparison} & \textbf{$\Delta$ exact (pp)} & \textbf{$\Delta$ geo.\ regret (pp)} & \textbf{$\Delta$ ms}\\
\midrule"""]
    for k, v in A['paired'].items():
        L.append(f"{lab[k.split(' vs ')[1]]} & {fci(v['exact'],'%+.1f')} & {fci(v['geo'],'%+.1f')} & {fci(v['ms'],'%+.2f')} \\\\")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    cnt, rows = strata_rows(R + 'picks_A.csv'); cnt2, rows2 = strata_rows(R + 'picks_A_im100.csv')
    L += [r"""\begin{table}[!htbp]
\centering
\caption{Regret by the best feasible latency of the pair (CIFAR-10 and
ImageNet-100 released constrained pools): geometric-mean relative regret (\%) / mean absolute regret
(ms), by the latency of the model--device pair. Fig.~2(d) of the main paper
uses model-level latency classes instead (\texttt{fig2b\_model\_bands.txt}).}
\label{tab:strata}
\footnotesize
\begin{tabular}{@{}lccc@{}}
\toprule""", f"\\textbf{{Policy}} & $<2$\\,ms ($n={cnt['<2']}$) & $2$--$10$\\,ms ($n={cnt['2-10']}$) & $\\geq10$\\,ms ($n={cnt['>=10']}$)\\\\\n\\midrule"] + rows + \
         [f"\\midrule\n\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{ImageNet-100}} ($n={cnt2['<2']}$, ${cnt2['2-10']}$, ${cnt2['>=10']}$)}}\\\\"] + rows2 + \
         ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]
    L += [r"""\begin{table}[!htbp]
\centering
\caption{Device dependence (\% of architectures labeled on all three devices)
when latency is taken from the fastest successful backend or from a fixed
backend. ImageNet-100 unconstrained rows use the fastest valid mode;
constrained rows use the public per-mode converted accuracies. Intervals
come from an independent bootstrap over architectures and may differ from
Table~V of the main paper in the last digit.}
\label{tab:backend}
\footnotesize
\begin{tabular}{@{}llrll@{}}
\toprule
\textbf{Pool} & \textbf{Latency source} & $n$ & \textbf{Exact} & \textbf{$\epsilon=5\%$}\\
\midrule"""]
    pn = {'cifar10/constrained': 'CIFAR-10, $\\tau=0.99$', 'cifar10/unconstrained': 'CIFAR-10, unconstrained',
          'imagenet100/unconstrained': 'ImageNet-100, unconstrained', 'imagenet100/constrained': 'ImageNet-100, $\\tau=0.99$'}
    for CC in (C, C2):
        for k, v in CC.items():
            pool, src = k.rsplit('/', 1)
            L.append(f"{pn[pool]} & {src} & {v['n']} & {fci(v['eps0'])} & {fci(v['eps5'])} \\\\")
            if src == 'GPU only' and pool != 'imagenet100/constrained': L.append(r'\midrule')
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    L += [r"""\begin{table}[!htbp]
\centering
\caption{Latency-only (accuracy-agnostic) selection on the released records:
label is the fastest valid mode; features exclude accuracy. ImageNet-100:
$5{,}017$ pairs, $1{,}693$ architectures, median best latency $18.1$\,ms;
CIFAR-10: $3{,}995$ pairs, $1{,}338$ architectures, $3.5$\,ms.}
\label{tab:latonly}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llllrl@{}}
\toprule
\textbf{Corpus} & \textbf{Policy} & \textbf{Exact (\%)} & \textbf{Geo.\ regret (\%)} & \textbf{p90 (\%)} & \textbf{Mean (ms)}\\
\midrule"""]
    nm = [('global dynamic', 'Global: dynamic INT8'), ('global fp16', 'Global: FP16'), ('per-device majority', 'Device-only: majority'),
          ('per-arch oracle majority (in-sample)', 'Per-arch.\\ majority (oracle)'), ('learned offline (GBT)', 'GBT, offline features'),
          ('learned +dev (GBT)', 'GBT, + device')]
    for c, cn in (('imagenet100', 'ImageNet-100'), ('cifar10', 'CIFAR-10')):
        for k, n in nm:
            s = D[c]['policies'][k]
            L.append(f"{cn} & {n} & {fci([s['exact']]+s['ci']['exact'])} & {fci([s['geo']]+s['ci']['geo'])} & {s['p90']:.1f} & "
                     f"{fci([s['ms_mean']]+s['ci']['ms_mean'],'%.2f')} \\\\"); cn = ''
        if c == 'imagenet100': L.append(r'\midrule')
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    L.append(lodo('v83_results_B.json', 'tab:lodo', r"""Transfer to an unseen device (CIFAR-10 released constrained pool,
$\tau=0.99$). Each device is held out in turn. Zero-shot rows train on the
other two devices only; few-shot rows add $k$ architectures benchmarked on the
held-out device to the training set and evaluate on its remaining
architectures (mean over five random calibration sets). All learned rows use
feasibility masking. Regret: geometric mean (geo, \%), 90th percentile (p90,
\%), mean absolute (ms)."""))
    open(R + 'supp_new.tex', 'w').write('\n'.join(L))


def supp_im100():
    A = json.load(open(R + 'v83_results_A_im100.json'))['A']; P = A['paired']
    k1 = 'learned +dev (GBT, masked) vs per-device order, feasibility-aware'; k2 = 'learned +dev (GBT, masked) vs global dynamic'
    f = lambda x: '%+.1f [%+.1f,\\,%+.1f]' % tuple(x)
    L = [r"""\begin{table}[!htbp]
\centering
\caption{All ImageNet-100 constrained selection policies on the released pool
($5{,}047$ pairs, $1{,}692$ architectures, $\tau=0.99$; per-mode converted
accuracies from \url{huggingface.co/NN-Dataset/tflite}). Protocol as in
Table~\ref{tab:policies_full}. Paired differences of the masked GBT: versus
the device-only feasible order """ + f(P[k1]['exact']) + " pp exact,\n" + f(P[k1]['geo']) +
         " pp regret; versus always dynamic INT8\n" + f(P[k2]['exact']) + " and " + f(P[k2]['geo']) + r""".}
\label{tab:im100constr}""", HEAD] + policy_rows(A['policies']) + ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]
    L.append(lodo('v83_results_B_im100.json', 'tab:lodo_im100', r"""Transfer to an unseen device on ImageNet-100 (constrained pool,
$\tau=0.99$). Protocol as in Table~\ref{tab:lodo}."""))
    G = json.load(open(R + 'guideline.json'))
    tr = lambda c: ', '.join('%s: %.1f\\%%/%.1f\\%%' % (k, v['exact'], v['geo']) for k, v in G[c]['tree'].items())
    lk = lambda c: '%.1f\\%%/%.1f\\%%' % (G[c]['lookup_cv']['exact'], G[c]['lookup_cv']['geo'])
    L += [r"""\begin{table}[!htbp]
\centering
\caption{Depth-3 decision trees (fitted on all pairs; feature names as in
\texttt{predictor.py}; sizes in KiB, accuracy drops as fractions). Cross-validated
performance with feasibility masking, by depth: CIFAR-10 """ + tr('cifar10') + "; ImageNet-100 " + tr('imagenet100') +
          r""" (exact / geo.\ regret). Size-band look-up rule (device $\times$ FP32 size band, CV): CIFAR-10 """ + lk('cifar10') +
          ", ImageNet-100 " + lk('imagenet100') + r""".}
\label{tab:trees}
\scriptsize
\begin{minipage}[t]{0.48\linewidth}
\textbf{CIFAR-10}
\begin{verbatim}""", G['cifar10']['tree_rules_depth3'].rstrip(), r"""\end{verbatim}
\end{minipage}\hfill
\begin{minipage}[t]{0.48\linewidth}
\textbf{ImageNet-100}
\begin{verbatim}""", G['imagenet100']['tree_rules_depth3'].rstrip(), r"""\end{verbatim}
\end{minipage}
\end{table}"""]
    open(R + 'supp_im100.tex', 'w').write('\n'.join(L))


if __name__ == '__main__':
    s6(); supp_new(); supp_im100()
    print('written supp_s6.tex, supp_new.tex, supp_im100.tex')
