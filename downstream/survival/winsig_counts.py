"""Compute DeepDXA-vs-tabular (and vs DINO) win + FDR-significant counts on the
displayed Fig-3 endpoint set, from a per-seed Cox CSV. Writes a summary text file.

Usage: python cox_winsig_counts.py <perseed_csv> [out_txt]
"""
import sys, importlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

pf = importlib.import_module('dexa_fm.ukbb.plot_fig3_cox')
SKIP = {'Obesity', 'Death', 'Angina Pectoris', 'Heart Failure', 'Acute Renal Failure', 'Myocardial Infarction'}
LE = 'DXA SSL (LeJEPA) + Covariates'; TAB = 'DXA Tabular + Covariates'; DINO = 'DXA SSL (DINO) + Covariates'


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else path.replace('_perseed.csv', '_winsig.txt')
    ps = pd.read_csv(path)
    arm_names = [a[0] for a in pf.ARMS]
    sb = {}
    for (ev, arm), g in ps.groupby(['Event', 'Arm']):
        sb[(ev, arm)] = g.sort_values('Seed')['C-Index'].dropna().to_numpy()

    disp = []
    for ev in sorted(ps['Event'].unique()):
        lab = pf._prettify_event(ev)
        if lab is None or lab in SKIP:
            continue
        cov = sb.get((ev, 'Covariates'), np.array([]))
        ok = False
        for a in arm_names:
            if a == 'Covariates':
                continue
            sv = sb.get((ev, a), np.array([])); n = min(len(sv), len(cov), 10)
            if n > 1 and not np.allclose(sv[:n], cov[:n]):
                try:
                    _, p = wilcoxon(sv[:n], cov[:n], alternative='two-sided')
                    if p < 0.05 and sv[:n].mean() > cov[:n].mean():
                        ok = True; break
                except ValueError:
                    pass
        if ok:
            disp.append((ev, lab))

    rows = []
    for ev, lab in disp:
        le = sb.get((ev, LE), np.array([])); tb = sb.get((ev, TAB), np.array([])); dn = sb.get((ev, DINO), np.array([]))
        n = min(len(le), len(tb), 10)
        try:
            p = wilcoxon(le[:n], tb[:n], alternative='two-sided')[1]
        except Exception:
            p = 1.0
        rows.append(dict(label=lab, lejepa=le[:n].mean(), tabular=tb[:n].mean(),
                         dino=dn[:n].mean() if len(dn) else np.nan,
                         dC=le[:n].mean() - tb[:n].mean(), p_tab=p))
    r = pd.DataFrame(rows)
    r['p_adj'] = multipletests(r['p_tab'], method='fdr_bh')[1]
    r = r.sort_values('dC', ascending=False)

    lines = []
    lines.append(f"source: {path}")
    lines.append(f"displayed endpoints: {len(r)}")
    lines.append(f"DeepDXA C > tabular (directional): {int((r['dC'] > 0).sum())}/{len(r)}")
    lines.append(f"DeepDXA > DINO (directional): {int((r['lejepa'] > r['dino']).sum())}/{len(r)}")
    lines.append(f"SIGNIFICANT vs tabular (FDR<0.05 & DeepDXA>tab): {int(((r['p_adj'] < 0.05) & (r['dC'] > 0)).sum())}")
    lines.append("")
    lines.append(r[['label', 'lejepa', 'tabular', 'dino', 'dC', 'p_adj']].round(4).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    with open(out, 'w') as f:
        f.write(txt + "\n")
    print(f"\n[written] {out}")


if __name__ == '__main__':
    main()
