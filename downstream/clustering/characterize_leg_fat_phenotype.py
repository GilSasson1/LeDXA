"""characterize_leg_fat_phenotype.py

Characterises the low-leg-fat body composition phenotype identified in the DXA
foundation model embedding (female subjects only).

Pipeline
--------
1. Female subjects, first DXA visit  (n ≈ 4 200)
2. OLS within women: legs_fat_mass ~ total_fat_mass + age + height
   → residuals = "leg fat relative to body size"
3. Q1 (low leg fat residual) = cases; Q4 (high/normal) = control pool
4. 1:2 Mahalanobis matching on age + height + total_fat_mass  (caliper 1.0)
5. SMD validation; system-level volcano (Mann–Whitney U, Cohen's d, BH-FDR)
6. Export RegistrationCode / Cluster_ID CSV  (same format as cluster_matched.csv)

Validation
----------
Saves per-subject embedding PC scores + group label so the figure script can
confirm the DXA model separates Q1 from Q4 without having been given leg-fat
labels.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '$HOME/.cache/tmp/matplotlib')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.extend([
    '/path/to/LabData',
    '/path/to/LabUtils',
    '/path/to/LabQueue',
])
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import characterize_female_clusters as cfc
from . import paths
from .characterize_pc_axes import (
    N_PCS, load_full_cohort_embeddings, _mahalanobis_match,
)

warnings.filterwarnings('ignore')

CALIPER       = 1.0
K_MATCH       = 2     # 1:2 matching
MATCH_COV     = ['age', 'height', 'total_fat_mass']
LEG_FAT_COL   = 'body_comp_legs_fat_mass'
TOTAL_FAT_COL = 'body_comp_total_fat_mass'
_MATCH_CSV_DIR = cfc.MATCH_CSV_DIR


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_female_body_comp(index: pd.Index) -> pd.DataFrame:
    """Load age, height, leg fat, total fat for female subjects only.

    Returns a DataFrame indexed to `index` (embedding MultiIndex), female rows only.
    """
    targets = pd.read_csv(cfc.TARGETS_PATH, index_col=[0, 1])
    cov     = pd.read_csv(os.path.join(cfc.BODY_SYSTEMS_DIR, 'Age_Gender_BMI.csv'),
                          index_col=[0, 1])
    bc      = pd.read_csv(os.path.join(cfc.BODY_SYSTEMS_DIR, 'body_composition.csv'),
                          index_col=[0, 1])

    sex       = targets[['gender']].reindex(index)
    age       = cov[['age']].reindex(index)
    height    = bc[['height']].reindex(index)
    leg_fat   = bc[[LEG_FAT_COL]].reindex(index)
    total_fat = bc[[TOTAL_FAT_COL]].reindex(index)

    df = pd.concat([sex, age, height, leg_fat, total_fat], axis=1)
    df.columns = ['sex', 'age', 'height', 'leg_fat_mass', 'total_fat_mass']

    # Zero values = invalid DXA scan region
    for c in ['height', 'leg_fat_mass', 'total_fat_mass']:
        df.loc[df[c] == 0, c] = np.nan

    females = df[df['sex'] == 0].drop(columns='sex')
    return females.dropna(subset=['age', 'height', 'leg_fat_mass', 'total_fat_mass'])


# ── Residual computation ──────────────────────────────────────────────────────

def _compute_leg_fat_residual(df: pd.DataFrame) -> pd.Series:
    """OLS: leg_fat ~ total_fat + age + height (all standardised).

    Returns standardised residuals indexed like df.
    """
    X = df[['total_fat_mass', 'age', 'height']].values
    y = df['leg_fat_mass'].values

    sx = StandardScaler().fit(X)
    sy = StandardScaler().fit(y.reshape(-1, 1))
    X_s = sx.transform(X)
    y_s = sy.transform(y.reshape(-1, 1)).ravel()

    coef = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_s)), X_s]), y_s, rcond=None
    )[0]
    y_pred = np.column_stack([np.ones(len(X_s)), X_s]) @ coef
    resid  = y_s - y_pred
    return pd.Series(resid, index=df.index, name='leg_fat_resid')


# ── Matching ──────────────────────────────────────────────────────────────────

def _smd(a: pd.Series, b: pd.Series, pooled_std: float) -> float:
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float('nan')


def _validate_smd(cases: pd.DataFrame, controls: pd.DataFrame,
                  match_cols: list[str], pre_std: pd.Series) -> None:
    for col in match_cols:
        smd    = _smd(cases[col], controls[col], pre_std[col])
        status = 'OK' if abs(smd) < 0.1 else 'NOTE'
        print(f"    SMD {col}: {smd:.4f} ({status})")


# ── Export ────────────────────────────────────────────────────────────────────

def _export_matched_ids(cases: pd.DataFrame, controls: pd.DataFrame) -> None:
    cases_regs = (cases.index.get_level_values('RegistrationCode')
                  .unique().tolist())
    ctrl_regs  = (controls.index.get_level_values('RegistrationCode')
                  .unique().tolist())
    rows = (
        [{'RegistrationCode': r, 'Cluster_ID': 1} for r in cases_regs] +
        [{'RegistrationCode': r, 'Cluster_ID': 0} for r in ctrl_regs]
    )
    df = pd.DataFrame(rows).drop_duplicates('RegistrationCode')

    local_path = paths.out_table('pc_leg_fat_matched_ids.csv')
    df.to_csv(local_path, index=False)
    print(f"  Exported {len(df)} subjects → {local_path}")

    try:
        net_path = os.path.join(_MATCH_CSV_DIR, 'leg_fat_phenotype_matched.csv')
        df.to_csv(net_path, index=False)
        print(f"  Network copy → {net_path}")
    except OSError:
        print("  (network path not mounted — local copy only)")


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_leg_fat_analysis() -> dict:
    print("Loading embeddings (full cohort, first visit)...")
    X_pca, var_ratio, index, _ = load_full_cohort_embeddings()

    print("Loading female body composition data...")
    fem_df = _load_female_body_comp(index)
    print(f"  {len(fem_df)} female subjects with complete data")

    print("Computing leg-fat residuals (OLS: leg_fat ~ total_fat + age + height)...")
    resid = _compute_leg_fat_residual(fem_df)
    fem_df = fem_df.copy()
    fem_df['leg_fat_resid'] = resid

    q25 = resid.quantile(0.25)
    q75 = resid.quantile(0.75)
    cases    = fem_df[resid <= q25].copy()   # low leg fat = cases  (Cluster 1)
    controls = fem_df[resid >= q75].copy()   # high/normal leg fat  (Cluster 0)
    print(f"  Q1 (low leg fat): n={len(cases)}")
    print(f"  Q4 (high leg fat): n={len(controls)}")

    cases['Cluster_ID']    = 1
    controls['Cluster_ID'] = 0

    print(f"\nMatching on {MATCH_COV}  (1:{K_MATCH}, caliper={CALIPER})...")
    pre_std = fem_df[MATCH_COV].std()
    matched_cases, matched_controls = _mahalanobis_match(
        cases, controls,
        match_cols=MATCH_COV,
        caliper=CALIPER,
        k=K_MATCH,
    )
    print(f"  Matched: {len(matched_cases)} cases, "
          f"{len(matched_controls)} controls")
    _validate_smd(matched_cases, matched_controls, MATCH_COV, pre_std)

    # ── Embedding validation ──────────────────────────────────────────────────
    # Save PC scores for matched subjects so the figure can confirm the embedding
    # separates Q1 from Q4 without having been given leg-fat labels.
    pc_df = pd.DataFrame(
        X_pca, index=index,
        columns=[f'PC{i+1}' for i in range(N_PCS)],
    )
    case_pcs = pc_df.reindex(matched_cases.index).assign(group='low_leg_fat')
    ctrl_pcs = pc_df.reindex(matched_controls.index).assign(group='high_leg_fat')
    embedding_val = pd.concat([case_pcs, ctrl_pcs])
    embedding_val.to_csv(paths.out_table('leg_fat_matched_pcs.csv'))
    print(f"  Saved embedding PCs for {len(embedding_val)} matched subjects")

    # Also save residuals for ALL female subjects (for scatter plot)
    all_pcs = pc_df.reindex(fem_df.index).copy()
    all_pcs['leg_fat_resid'] = fem_df['leg_fat_resid']
    all_pcs['leg_fat_mass']  = fem_df['leg_fat_mass']
    all_pcs['total_fat_mass'] = fem_df['total_fat_mass']
    all_pcs.to_csv(paths.out_table('leg_fat_female_pcs.csv'))

    # ── Export matched IDs ────────────────────────────────────────────────────
    print("\nExporting matched subject IDs...")
    _export_matched_ids(matched_cases, matched_controls)

    # ── System volcano ────────────────────────────────────────────────────────
    print("\n--- Running system volcanos ---")
    output_dir = paths.out_table('')   # per-system CSVs go alongside other tables
    systems    = cfc.get_body_system_files()
    microbiome_map = cfc.get_microbiome_annotation()

    all_results: dict[str, pd.DataFrame] = {}
    for system_name, system_path in systems:
        res = cfc.run_system_volcano(
            matched_cases, matched_controls,
            system_name, system_path,
            output_dir, gender_label='female',
            microbiome_map=microbiome_map,
        )
        if res is not None:
            all_results[system_name] = res

    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        n_sig = combined.get('significant', pd.Series(dtype=bool)).sum()
        out   = paths.out_table('tableS_leg_fat_phenotype.csv')
        combined.to_csv(out, index=False)
        print(f"\nWrote {out} ({len(combined)} rows, {n_sig} significant)")

    return all_results


if __name__ == '__main__':
    run_leg_fat_analysis()
