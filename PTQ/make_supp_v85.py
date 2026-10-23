#!/usr/bin/env python3
"""Supplementary fragment of v85: results/v85/supp_v85.tex (Tables S21-S32, Fig. S-last), from the result JSONs.
    python make_supp_v85.py
"""
import _runctx  # noqa: F401  -- run from analysis/ regardless of the caller's cwd
import json
R = 'results/v85/'
prof = json.load(open(R + 'accel_profile.json')); vd = json.load(open(R + 'virtual_device.json'))
ka = json.load(open(R + 'k_artifacts.json')); om = json.load(open(R + 'overhead_model.json'))
rb = json.load(open(R + 'robustness_v85.json')); nm = json.load(open(R + 'named_models.json'))
ft = json.load(open(R + 'family_transfer.json')); ct = json.load(open(R + 'cross_task.json'))
try: sv = json.load(open(R + 'selector_variants.json'))
except Exception: sv = None
CN = {'cifar10': 'CIFAR-10', 'imagenet100': 'ImageNet-100'}
DV = ['Redmi 13', 'Pixel 4', 'Tab S7+']; MODES = ['fp16', 'mixed', 'dynamic', 'int8']
SH = {'fp32': 'FP32', 'fp16': 'FP16', 'mixed': 'Mixed', 'dynamic': 'Dyn.', 'int8': 'INT8'}
O = []
def tab(cap, label, cols, rows, fs='\\footnotesize', tc=3):
    O.append('\\begin{table}[!t]\n\\centering\n\\caption{' + cap + '}\n\\label{' + label + '}\n' + fs + '\n\\setlength{\\tabcolsep}{' + str(tc) + 'pt}\n\\begin{tabular}{' + cols + '}\n\\toprule')
    for r in rows: O.append(r)
    O.append('\\bottomrule\n\\end{tabular}\n\\end{table}\n')
O.append('\\section{Additional Analyses of the Accelerator and Shipping-Unit Study}\n\\label{sec:supp_v85}\n')
O.append('This section collects the analyses behind the mechanism, shipping-unit, transfer, and robustness results of the main paper. All numbers are produced by the scripts of Section~\\ref{sec:supp_repro_v85} from the released records.\n')
# S21 slowdown
rows = ['\\textbf{Mode} & \\multicolumn{3}{c}{\\textbf{CIFAR-10}} & \\multicolumn{3}{c}{\\textbf{ImageNet-100}} \\\\', ' & $n$ & slower & $>1.5\\times$ & $n$ & slower & $>1.5\\times$ \\\\', '\\midrule']
for m in ['fp16', 'mixed', 'dynamic', 'int8']:
    a, b = rb['cifar10']['F7'][m], rb['imagenet100']['F7'][m]
    rows.append(f'{SH[m]} & {a["n"]} & {a["slower1"]:.1f}\\% & {a["slower15"]:.1f}\\% & {b["n"]} & {b["slower1"]:.1f}\\% & {b["slower15"]:.1f}\\% \\\\')
tab('Share of model--device pairs in which a quantized mode is slower than the FP32 graph on the same device (best backend, mean latency): at all, and by more than $1.5\\times$.', 'tab:slower', '@{}lrrrrrr@{}', rows)
# S22 availability + profile
rows = ['\\multicolumn{6}{@{}l}{\\textbf{(a) Median latency / FP32 latency on the same backend [Q1, Q3] (paired by model; both corpora)}}\\\\', '\\textbf{Device, backend} & $n$ (FP16) & FP16 & Mixed & Dynamic & Static INT8 \\\\', '\\midrule']
for d in DV:
    for b in ('cpu', 'gpu', 'npu'):
        cells = [f'{d}, {b.upper()}']; ns = []
        for m in MODES:
            v = prof['profile'].get(f'{d}|{b}|{m}')
            if v: cells.append(f'{v["med"]:.2f} [{v["q1"]:.2f}, {v["q3"]:.2f}]' + ('$^{*}$' if v['n'] < 20 else '')); ns.append(v['n'])
            else: cells.append('--'); ns.append(0)
        rows.append(' & '.join([cells[0], str(ns[0])] + cells[1:]) + ' \\\\')
    rows.append('\\midrule' if d != DV[-1] else '')
rows.append('\\multicolumn{6}{@{}l}{\\textbf{(b) Share of valid runs with a measured backend latency ($>0.1$\\,ms)}}\\\\')
rows.append('\\textbf{Device, mode} & $n$ & CPU & GPU & NPU & \\\\\n\\midrule')
for d in DV:
    for m in ['fp32', 'fp16', 'mixed', 'dynamic', 'int8']:
        v = prof['availability'][f'{d}|{m}']
        rows.append(f'{d}, {SH[m]} & {v["n"]} & {100*v["cpu"]:.1f}\\% & {100*v["gpu"]:.1f}\\% & {100*v["npu"]:.1f}\\% & \\\\')
tab('Accelerator profiles. $^{*}$: fewer than $20$ models (the delegate rarely executed the mode). FP32 GPU/CPU and NPU/GPU latency ratios by latency class are in \\texttt{accel\\_profile.txt}.', 'tab:profile', '@{}lrcccc@{}', [r for r in rows if r != ''], tc=2.5)
# S23 K artifacts
rows = ['\\textbf{Corpus, artifacts} & all & $<2$\\,ms & $2$--$10$\\,ms & $\\geq10$\\,ms & total time \\\\', '\\midrule']
names = {'K=1 single artifact': 'One (best single)', 'K=2 vendor split (Redmi | Pixel+Tab)': 'Two: Redmi $|$ Pixel+Tab', 'K=2 (Pixel | Redmi+Tab)': 'Two: Pixel $|$ Redmi+Tab', 'K=2 (Tab | Redmi+Pixel)': 'Two: Tab $|$ Redmi+Pixel', 'K=2 best split per model': 'Two: best split per model', 'K=3 per device': 'Three (per device)'}
for c in ka:
    for k, nme in names.items():
        v = ka[c][k]; ci = v['all']['ci']
        rows.append(f'{CN[c] if k.startswith("K=1") else ""} {nme} & {v["all"]["geo"]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] & {v["model <2 ms"]["geo"]:.1f} & {v["model 2-10 ms"]["geo"]:.1f} & {v["model >=10 ms"]["geo"]:.1f} & {v["all"]["tot"]:.2f} \\\\')
    rows.append('\\midrule')
tab('Number of artifacts: geometric-mean regret (\\%) overall (95\\% clustered bootstrap interval) and by model latency class, and total-time regret (sum of lost milliseconds over sum of best milliseconds, \\%). Group choice minimizes the mean log regret within each group (Eq.~(3) of the main paper).', 'tab:kart', '@{}lccccc@{}', rows[:-1], tc=3)
# S24 overhead
rows = ['\\textbf{Device, path} & $a$ (ms) & $s$ & $R^2$ & med.\\ rel.\\ err. & $n$ \\\\', '\\midrule']
for k, v in om['fit'].items():
    d, b, m = k.split('|')
    if b == 'cpu' and m == 'fp32': continue
    rows.append(f'{d}, {b.upper()} {SH[m]} & {v["a"]:.2f} & {v["s"]:.2f} & {v["r2"]:.2f} & {100*v["medrelerr"]:.0f}\\% & {v["n"]} \\\\')
rows.append('\\midrule\n\\multicolumn{6}{@{}l}{\\textbf{Crossovers} (path 1 faster than path 2 below $L^{*}$; $L^{*}=(a_1-a_2)/(s_2-s_1)$ versus the empirical 50\\% crossing)}\\\\')
rows.append('\\textbf{Device, pair} & predicted (ms) & empirical (ms) & \\multicolumn{2}{c}{$P$(path 1 faster), $x<2$ / $>20$\\,ms} & $n$ \\\\\n\\midrule')
for k, v in om['cross'].items():
    d, p1, p2 = k.split('|')
    p1 = eval(p1); p2 = eval(p2)
    rows.append(f'{d}, {p1[0].upper()} {SH[p1[1]]} vs {p2[0].upper()} {SH[p2[1]]} & {v["L_pred"]:.1f} & {v["L_emp"]:.1f} & \\multicolumn{{2}}{{c}}{{{v["p1_faster_small"]:.2f} / {v["p1_faster_large"]:.2f}}} & {v["n"]} \\\\')
tab('Overhead-plus-throughput model $L_{\\text{path}}=a+s\\,L_{\\text{CPU,FP32}}$ (Theil--Sen, both corpora, $L_{\\text{CPU,FP32}}<400$\\,ms). Negative $a$ on NPU paths is an artifact of the linear form.', 'tab:overhead', '@{}lccccc@{}', rows, tc=3)
# S25-26 named
for c, lab in (('cifar10', 'tab:named_c10'), ('imagenet100', 'tab:named_im')):
    body = open(R + f'tab_named_full_{c}.tex').read().strip().split('\n')
    tab(f'All named architectures, {CN[c]}: FP32 size (MB), FP32 accuracy (\\%), fastest feasible mode and latency (ms) per device, and regret of the best single artifact (\\%).', lab, '@{}lrrcccr@{}',
        ['\\textbf{Model} & \\textbf{MB} & \\textbf{Acc.} & \\textbf{Redmi~13} & \\textbf{Pixel~4} & \\textbf{Tab~S7+} & \\textbf{1-art.}\\\\', '\\midrule'] + body, tc=2.5)
# S27 drift
rows = ['\\textbf{Corpus, device} & span (days) & $n$ & Spearman $\\rho$ & shift last vs first third \\\\', '\\midrule']
for c in ('cifar10', 'imagenet100'):
    for d in DV:
        v = rb[c]['F5'][d]
        rows.append(f'{CN[c] if d == DV[0] else ""} {d} & {v["span_days"]:.1f} & {v["n"]} & {v["rho"]:+.2f} & {v["shift_pct"]:+.1f}\\% \\\\')
    rows.append('\\midrule')
tab('Measurement drift: run timestamp versus the log GPU/CPU latency ratio of the FP32 runs (detrended by a cubic in log CPU latency, since run order is not independent of model size).', 'tab:drift', '@{}lcccc@{}', rows[:-1])
# S29 selector variants
if sv:
    rows = ['\\textbf{Policy} & \\multicolumn{3}{c}{\\textbf{CIFAR-10}} & \\multicolumn{3}{c}{\\textbf{ImageNet-100}} \\\\', ' & exact & regret & viol. & exact & regret & viol. \\\\', '\\midrule']
    keys = ['direct GBT, per-device (masked)', 'direct GBT, per-device (unmasked)', 'direct GBT, per-family (masked)', 'direct GBT, offline only (masked)', 'latency regressors, per-device (masked)', 'latency regressors, per-device (unmasked)', 'latency regressors, per-family (masked)']
    for k in keys:
        cells = [k]
        for c in ('cifar10', 'imagenet100'):
            v = sv.get(c, {}).get(k)
            cells += [f'{v["exact"]:.1f}', f'{v["geo"]:.1f}', f'{v["viol"]:.1f}\\%'] if v else ['--'] * 3
        rows.append(' & '.join(cells) + ' \\\\')
    tab('Selector variants under identical grouped five-fold cross-validation: exact match (\\%), geometric-mean regret (\\%), and share of picks violating the accuracy budget. ``Per-family\'\': the device features are replaced by the SoC vendor; with cores and RAM added the device is identified again.', 'tab:selvar', '@{}lcccccc@{}', rows)
# S29 transfer (renders as Table S29)
rows = ['\\textbf{Corpus, train $\\to$ test} & relation & GBT exact & GBT regret & majority exact & majority regret \\\\', '\\midrule']
for c in ('cifar10', 'imagenet100'):
    for x in DV:
        for y in DV:
            g = ft[c][f'GBT|{x}->{y}']; m = ft[c][f'majority|{x}->{y}']
            rows.append(f'{CN[c] if (x, y) == (DV[0], DV[0]) else ""} {x} $\\to$ {y} & {g["rel"]} & {g["exact"]:.1f} & {g["geo"]:.1f} & {m["exact"]:.1f} & {m["geo"]:.1f} \\\\')
    rows.append('\\midrule')
tab('Zero-shot transfer between devices (offline features only, architecture-grouped folds): exact match and geometric-mean regret (\\%) of the selector and of the majority-mode order of the training device.', 'tab:transfer_full', '@{}llcccc@{}', rows[:-1], tc=2.5)
# S30 virtual device (renders as Table S30)
rows = ['\\textbf{Corpus, backends allowed} & \\multicolumn{3}{c}{integer clearly fastest (\\%)} & gap [95\\% CI] & same winner \\\\', ' & Redmi & Pixel & Tab & (pp) & on all 3 (\\%) \\\\', '\\midrule']
for c in ('cifar10', 'imagenet100'):
    for s, v in vd[c].items():
        g = v['gap_dec']
        rows.append(f'{CN[c] if s.startswith("all") else ""} {s} & {v["Redmi 13"]["dec_int"]:.0f} & {v["Pixel 4"]["dec_int"]:.0f} & {v["Tab S7+"]["dec_int"]:.0f} & {g["mean"]:+.1f} [{g["lo"]:+.1f}, {g["hi"]:+.1f}] & {v["same_winner_all3"]:.1f} \\\\')
    rows.append('\\midrule')
tab('Within-device control: an integer mode is clearly fastest if it is more than $5\\%$ faster than the best floating-point mode; gap $=$ Redmi~13 minus the mean of the two Qualcomm devices.', 'tab:virtual', '@{}lccccc@{}', rows[:-1])
# S31 cross-task (renders as Table S31)
rows = ['\\textbf{Quantity} & \\textbf{Value} \\\\', '\\midrule',
        f'Shared model--device pairs / architectures & {ct["n"]} / 1167 \\\\',
        f'Constrained winner agrees (CIFAR-10 vs ImageNet-100) & {100*ct["agree"]:.1f}\\% \\\\',
        f'CIFAR-10 choice reused on ImageNet-100: exact / regret & {ct["reuse"]["exact"]:.1f}\\% / {ct["reuse"]["geo"]:.1f}\\% \\\\',
        f'Depth-3 tree trained on CIFAR-10, applied to ImageNet-100: exact / regret & {ct["tree_c2i"]["exact"]:.1f}\\% / {ct["tree_c2i"]["geo"]:.1f}\\% \\\\',
        f'Depth-3 tree trained on ImageNet-100 (in sample): exact / regret & {ct["tree_i2i"]["exact"]:.1f}\\% / {ct["tree_i2i"]["geo"]:.1f}\\% \\\\']
tab('Cross-task reuse.', 'tab:crosstask', '@{}lc@{}', rows)
# fig_v83_device_and_regret.png is Fig. 2 of the main paper; not repeated here.
# NOTE: Table S32 (device dependence, tab:ties) is NOT emitted here. It is
# maintained by hand in Supplementary.tex, immediately after the \\input of
# this file so that it keeps the number S32. It used to be appended to the end
# of this generated file, where re-running this script silently deleted it.
open(R + 'supp_v85.tex', 'w').write('\n'.join(O))
print('ok', len(O))
