"""Sex-split paired before/after medication analysis on the LeDXA bio-age gap.

Replicates run_paired_medication_analysis() exactly, but splits EVERY analyzable
drug class into Female / Male paired Wilcoxon tests on age_residual (the gap),
then applies a single Benjamini-Hochberg FDR correction across all drug x sex
tests. This tests whether the antidepressant male-specificity is a real,
multiplicity-aware sex difference or a post-hoc subgroup artifact.

Uses the cached post-BMD-exclusion predictions (age_prediction_rate_partitions_bmd_excl.csv)
so no embedding re-extraction is needed; medication start events are loaded via
the same Medications10KLoader path the pipeline uses.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import downstream.bioage.generate_age_partitions as gap  # reuse loaders/config

OUTPUT_DIR = gap.OUTPUT_DIR
MIN_PAIRED = gap.MIN_PAIRED_SUBJECTS  # 10
PARTITIONS_BMD_EXCL = os.path.join(OUTPUT_DIR, "age_prediction_rate_partitions_bmd_excl.csv")

def main():
    # 1. df_final (post-BMD-exclusion), cached from the canonical run
    df_final = pd.read_csv(PARTITIONS_BMD_EXCL)
    df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce')
    print(f"df_final: {len(df_final)} visit-rows, "
          f"{df_final['RegistrationCode'].nunique()} subjects")

    # 2. gender map
    df_targets = gap.clean_format(pd.read_csv(gap.TARGETS_PATH))
    gmap = (df_targets[['RegistrationCode', 'gender']].drop_duplicates()
            .set_index('RegistrationCode')['gender'])  # 0=Female, 1=Male

    # 3. medications
    start_events, _ = gap.load_medications()
    if start_events is None:
        print("No medication start events; aborting.")
        return
    analyzable = gap._build_analyzable_drugs(start_events)
    print(f"{len(analyzable)} analyzable ATC-3 classes (>= {gap.MIN_POPULAR_DRUG_USERS} users)")

    visits = df_final[['RegistrationCode', 'Date', 'age_residual']].dropna().copy()

    rows = []
    for group in analyzable:
        drug = group['drug']; first_start = group['first_start']
        dv_all = visits[visits['RegistrationCode'].isin(first_start.index)].copy()
        for sex_label, sex_code in [('Female', 0), ('Male', 1)]:
            sex_ids = gmap.index[gmap == sex_code]
            dv = dv_all[dv_all['RegistrationCode'].isin(sex_ids)].copy()
            if dv.empty:
                continue
            dv['start_date'] = dv['RegistrationCode'].map(first_start)
            dv = dv.dropna(subset=['start_date'])
            dv['period'] = np.where(dv['Date'] < dv['start_date'], 'Before', 'After')
            sm = dv.groupby(['RegistrationCode', 'period'])['age_residual'].mean().unstack()
            if 'Before' not in sm.columns or 'After' not in sm.columns:
                continue
            sm = sm.dropna(subset=['Before', 'After'])
            n = len(sm)
            if n < MIN_PAIRED:
                continue
            delta = sm['After'] - sm['Before']
            try:
                _, p = wilcoxon(sm['Before'], sm['After'])
            except ValueError:
                continue
            rows.append({'drug': drug, 'label': gap.atc3_label(drug),
                         'sex': sex_label, 'n': n,
                         'mean_before': sm['Before'].mean(),
                         'mean_after': sm['After'].mean(),
                         'delta_gap': delta.mean(), 'p_raw': p})

    res = pd.DataFrame(rows)
    if res.empty:
        print("No drug x sex cell met the minimum-n threshold.")
        return
    _, res['p_fdr'], _, _ = multipletests(res['p_raw'], method='fdr_bh')
    res = res.sort_values('p_raw').reset_index(drop=True)

    pd.set_option('display.width', 160); pd.set_option('display.max_rows', 200)
    print(f"\n{len(res)} drug x sex tests (n >= {MIN_PAIRED}); BH-FDR across all:\n")
    print(res[['drug', 'label', 'sex', 'n', 'mean_before', 'mean_after',
               'delta_gap', 'p_raw', 'p_fdr']].to_string(index=False,
               float_format=lambda x: f"{x:.4f}"))
    out = os.path.join(OUTPUT_DIR, "medication_sexsplit_paired_gap.csv")
    res.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("\nSignificant at FDR<0.05:")
    print(res[res['p_fdr'] < 0.05][['drug','label','sex','n','delta_gap','p_raw','p_fdr']].to_string(index=False) or "  none")

if __name__ == '__main__':
    main()
