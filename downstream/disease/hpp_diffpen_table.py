#!/usr/bin/env python
"""Build a supp_tableA-format HPP disease-AUC table from a given raw seed file.

Replicates the Table-A reshape in export_supplementary_tables_4arm.py (same
_dis_label mapping, same arm-label columns) but reads an arbitrary raw file and
writes to an arbitrary output — used to render Figure 2 with the differential-
penalisation HPP results without touching the published supp_tableA.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.plot_style import _dis_label, DISEASE_TARGETS_CSV, _DISEASE_NAMES_JSON  # noqa: E402

MODELS = ['covariates', 'lejepa_cov', 'dino_cov', 'tab_cov']
LBL = {'covariates': 'Covariates (age/sex/BMI)', 'lejepa_cov': 'DXA-FM + Covariates',
       'dino_cov': 'DINOv3 + Covariates', 'tab_cov': 'DXA Tabular + Covariates'}


def _sex_flag(d, dt, thresh=0.90):
    if d not in dt.columns or 'gender' not in dt.columns:
        return ''
    pos = dt[dt[d] == 1.0]['gender'].dropna()
    if len(pos) == 0:
        return ''
    pf = (pos == 0.0).mean()
    return "Women's Disease" if pf > thresh else ("Men's Disease" if pf < 1 - thresh else '')


def _n_pos(d, dt):
    return int(dt[d].sum()) if d in dt.columns else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    display = json.load(open(_DISEASE_NAMES_JSON)) if os.path.exists(_DISEASE_NAMES_JSON) else {}
    dtcov = os.path.join(os.path.dirname(DISEASE_TARGETS_CSV), 'disease_targets_with_covs.csv')
    dt = pd.read_csv(dtcov) if os.path.exists(dtcov) else pd.read_csv(DISEASE_TARGETS_CSV)

    raw = pd.read_csv(a.raw, encoding='latin-1')
    raw = raw[raw['target'].astype(str).str.startswith('dis__')]
    if 'metric' in raw.columns:
        raw = raw[raw['metric'] == 'auc']
    agg = raw.groupby(['target', 'model'])['score'].agg(['mean', 'sem', 'count'])
    targets = sorted({t for t in raw['target'].unique() if _n_pos(t, dt) and _n_pos(t, dt) >= 100})

    rows = []
    for t in targets:
        r = {'Disease': _dis_label(t, display), 'N_positives': _n_pos(t, dt), 'Sex_flag': _sex_flag(t, dt)}
        for m in MODELS:
            if (t, m) in agg.index:
                r[f'{LBL[m]}_mean'] = round(float(agg.loc[(t, m), 'mean']), 4)
                r[f'{LBL[m]}_SE']   = round(float(agg.loc[(t, m), 'sem']), 4)
                r[f'{LBL[m]}_N']    = int(agg.loc[(t, m), 'count'])
            else:
                r[f'{LBL[m]}_mean'] = np.nan
                r[f'{LBL[m]}_SE']   = np.nan
                r[f'{LBL[m]}_N']    = 0
        rows.append(r)
    df = pd.DataFrame(rows).sort_values('Disease')
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} diseases × {len(MODELS)} arms")


if __name__ == '__main__':
    main()
