"""characterize_pc_axes.py

Continuous PC-axis body composition analysis — full HPP cohort.

Replaces the discrete UMAP-cluster story with:
  Phase 1: Per-PC partial OLS regressions across all phenotype systems (heatmap data)
  Phase 2: KDE-mode unsupervised healthy centroid + Mahalanobis-matched group volcano

Run from the DEXA root:
    python -m dexa_fm.hpp.clustering.characterize_pc_axes
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '$HOME/.cache/tmp/matplotlib')

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, normalize
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import characterize_female_clusters as cfc

from . import paths

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────

N_PCS = 20       # PCs to characterize in heatmap
KDE_PCS = 6      # PCs used for KDE density-mode search (avoids CoD)
Q_NEAR = 0.25    # Distance percentile: "core / healthy-like" group
Q_FAR  = 0.75    # Distance percentile: "divergent" group

_COVARIATE_COLS = {x.lower() for x in
                   {'age', 'bmi', 'gender', 'participant_id', 'total_scan_vat_mass'}}

# Systems to exclude from per-PC regressions (binary-only or redundant)
_EXTRA_EXCLUDE = {
    'medical_conditions.csv', 'medications.csv',
    'family_history.csv', 'family_medical_conditions.csv',
    'medical_conditions_grouped.csv',
}


# ── OLS helpers (local copies — no dep on story_sensitivity imports) ──────────

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors='coerce')
        mu, sd = s.mean(), s.std(ddof=1)
        out[col] = (s - mu) / sd if (sd and np.isfinite(sd)) else np.nan
    return out


def _ols_standardized_beta(y: pd.Series, x: pd.Series,
                            covars: pd.DataFrame) -> dict | None:
    """OLS: y ~ x + covars, all standardised. Returns std_beta, CI, p."""
    data = pd.concat([y.rename('y'), x.rename('x'), covars], axis=1)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    if len(data) < 30 or data['y'].nunique() < 3:
        return None
    z = _standardize_columns(data).dropna()
    if len(z) < 30:
        return None
    X = np.column_stack([np.ones(len(z)), z.drop(columns='y').to_numpy(dtype=float)])
    Y = z['y'].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = Y - X @ beta
    dof = len(Y) - X.shape[1]
    if dof <= 0:
        return None
    sigma2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    if not np.isfinite(se[1]) or se[1] == 0:
        return None
    tval = beta[1] / se[1]
    pval = 2 * student_t.sf(abs(tval), dof)
    crit = student_t.ppf(0.975, dof)
    return {
        'n': int(len(Y)),
        'std_beta': float(beta[1]),
        'ci_low':  float(beta[1] - crit * se[1]),
        'ci_high': float(beta[1] + crit * se[1]),
        'p_value': float(pval),
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_full_cohort_embeddings() -> tuple[np.ndarray, np.ndarray, pd.Index, PCA]:
    """Load all subjects (first visit), L2-norm → StandardScaler → PCA.

    Returns (X_pca[:, :N_PCS], explained_variance_ratio[:N_PCS], index, pca_model).
    """
    print("Loading embeddings (full cohort, first visit per subject)...")
    emb = pd.read_pickle(cfc.EMB_PATH)

    l0 = emb.index.get_level_values(0).astype(str).map(
        lambda x: x if x.startswith('10K_') else f'10K_{x}'
    )
    l1 = emb.index.get_level_values(1).astype(str).map(
        lambda x: 'baseline' if x == '00_00_visit' else x
    )
    emb.index = pd.MultiIndex.from_arrays(
        [l0, l1], names=['RegistrationCode', 'research_stage']
    )

    regs   = emb.index.get_level_values('RegistrationCode')
    stages = emb.index.get_level_values('research_stage')
    order = pd.DataFrame(
        {'reg': regs, 'ord': [cfc.STAGE_ORDINAL.get(s, 99) for s in stages]},
        index=emb.index,
    ).sort_values('ord')
    keep_idx = order[~order['reg'].duplicated(keep='first')].index
    emb = emb.loc[keep_idx]
    print(f"  {len(emb)} unique subjects")

    X = normalize(emb.to_numpy(dtype=float), norm='l2', axis=1)
    X = StandardScaler().fit_transform(X)
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=cfc.PCA_SEED)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    print(f"  PCA top {N_PCS}/{n_comp} PCs, "
          f"cumul. var = {var[:N_PCS].sum():.3f}")
    return X_pca[:, :N_PCS], var[:N_PCS], emb.index, pca


def load_pc_covariates(index: pd.Index) -> pd.DataFrame:
    """Return (sex, age, height, total_fat_mass, total_lean_mass) aligned to index."""
    targets = pd.read_csv(cfc.TARGETS_PATH, index_col=[0, 1])
    cov = pd.read_csv(
        os.path.join(cfc.BODY_SYSTEMS_DIR, 'Age_Gender_BMI.csv'), index_col=[0, 1]
    )
    bc = pd.read_csv(
        os.path.join(cfc.BODY_SYSTEMS_DIR, 'body_composition.csv'), index_col=[0, 1]
    )
    sex    = targets[['gender']].reindex(index).rename(columns={'gender': 'sex'})
    age    = cov[['age']].reindex(index)
    height = bc[['height']].reindex(index)
    fat    = bc[['body_comp_total_fat_mass']].reindex(index)
    lean   = bc[['body_comp_total_lean_mass']].reindex(index)
    for col, df in [('body_comp_total_fat_mass', fat), ('body_comp_total_lean_mass', lean)]:
        df.loc[df[col] == 0, col] = np.nan
    fat  = fat.rename(columns={'body_comp_total_fat_mass': 'total_fat_mass'})
    lean = lean.rename(columns={'body_comp_total_lean_mass': 'total_lean_mass'})
    return pd.concat([sex, age, height, fat, lean], axis=1)


# ── Per-PC regression helpers ─────────────────────────────────────────────────

def _list_system_csvs() -> list[tuple[str, str]]:
    out = []
    for fname in sorted(os.listdir(cfc.BODY_SYSTEMS_DIR)):
        if fname.startswith('.'):          # skip macOS resource-fork files (._*)
            continue
        if not fname.endswith('.csv'):
            continue
        if fname in cfc.EXCLUDE_SYSTEMS or fname in _EXTRA_EXCLUDE:
            continue
        if cfc.TEMPORAL_SUFFIXES.search(fname):
            continue
        out.append((fname.replace('.csv', ''), os.path.join(cfc.BODY_SYSTEMS_DIR, fname)))
    return out


def _load_phenotype_frame(system: str, path: str, index: pd.Index) -> pd.DataFrame | None:
    """Load numeric phenotypes from a system CSV, aligned to MultiIndex."""
    try:
        if system == 'high_level_diet':
            raw = pd.read_csv(path, index_col=0)
            regs = index.get_level_values('RegistrationCode')
            df = raw.reindex(regs)
            df.index = index
        elif system in cfc.OMICS_SYSTEMS:
            raw = pd.read_csv(path, index_col=[0, 1]).reset_index()
            raw['_ord'] = raw['research_stage'].map(cfc.STAGE_ORDINAL).fillna(99)
            raw = (raw.sort_values(['RegistrationCode', '_ord'])
                      .drop_duplicates('RegistrationCode', keep='first')
                      .set_index('RegistrationCode')
                      .drop(columns=['research_stage', '_ord'], errors='ignore'))
            regs = index.get_level_values('RegistrationCode')
            df = raw.reindex(regs)
            df.index = index
        else:
            df = pd.read_csv(path, index_col=[0, 1]).reindex(index)
    except Exception as e:
        print(f"  Skip {system}: {e}")
        return None

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [c for c in num_cols if c.lower() not in _COVARIATE_COLS]
    if not keep:
        return None
    return df[keep]


# ── Phase 1: Per-PC regressions ───────────────────────────────────────────────

def run_per_pc_regressions(X_pca: np.ndarray, index: pd.Index,
                            covariates: pd.DataFrame) -> pd.DataFrame:
    """Regress each phenotype on each of the top N_PCS PC scores.

    Covariates: sex, age, height, total_fat_mass (all standardised in OLS).
    Returns tidy DataFrame with one row per (pc, system, feature).
    """
    print(f"\nPhase 1: per-PC regressions ({N_PCS} PCs)...")
    pc_df = pd.DataFrame(
        X_pca,
        index=index,
        columns=[f'PC{i+1}' for i in range(N_PCS)],
    )
    covars = covariates.reindex(index)

    rows = []
    for system, path in _list_system_csvs():
        pheno = _load_phenotype_frame(system, path, index)
        if pheno is None:
            continue
        n_feats = len(pheno.columns)
        for pc_col in pc_df.columns:
            for feat in pheno.columns:
                res = _ols_standardized_beta(pheno[feat], pc_df[pc_col], covars)
                if res is None:
                    continue
                res.update({'pc': pc_col, 'system': system, 'feature': feat})
                rows.append(res)
        print(f"  {system}: {n_feats} phenotypes")

    if not rows:
        raise RuntimeError("No regression results — check data paths.")

    df = pd.DataFrame(rows)

    # BH-FDR within each PC
    df['fdr_q_value'] = np.nan
    for pc in df['pc'].unique():
        mask = df['pc'] == pc
        _, q, _, _ = multipletests(df.loc[mask, 'p_value'], method='fdr_bh')
        df.loc[mask, 'fdr_q_value'] = q

    df = df.sort_values(['pc', 'fdr_q_value'])
    print(f"  Total: {len(df)} associations, "
          f"{(df['fdr_q_value'] < 0.05).sum()} FDR-significant (q<0.05)")
    return df


# ── Phase 2: KDE mode + group assignment ─────────────────────────────────────

def find_kde_mode(X_pca: np.ndarray) -> np.ndarray:
    """Find the density mode of the embedding in the top KDE_PCS subspace.

    Uses Scott's bandwidth rule + Nelder-Mead gradient ascent on log-density.
    The returned mode is padded with zeros for dimensions > KDE_PCS.
    """
    print(f"\nFitting KDE on top {KDE_PCS} PCs to find population-mode centroid...")
    X_sub = X_pca[:, :KDE_PCS]
    n, d = X_sub.shape
    bw = n ** (-1.0 / (d + 4))   # Scott's rule
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(X_sub)

    x0 = X_sub.mean(axis=0)
    res = minimize(
        lambda x: -kde.score_samples(x.reshape(1, -1))[0],
        x0,
        method='Nelder-Mead',
        options={'maxiter': 100_000, 'xatol': 1e-8, 'fatol': 1e-10},
    )
    mode_sub = res.x
    log_density_at_mode = kde.score_samples(mode_sub.reshape(1, -1))[0]
    print(f"  Converged={res.success}, log-density at mode={log_density_at_mode:.3f}")
    print(f"  PC1={mode_sub[0]:.3f}, PC2={mode_sub[1]:.3f}, PC3={mode_sub[2]:.3f}")

    mode = np.zeros(N_PCS)
    mode[:KDE_PCS] = mode_sub
    return mode


def assign_distance_groups(X_pca: np.ndarray, mode: np.ndarray,
                            index: pd.Index) -> pd.DataFrame:
    """Assign subjects to 'core' (near mode) vs 'divergent' (far from mode).

    Returns DataFrame with Cluster_ID (1=core, 0=divergent) and dist_to_mode.
    Only subjects in Q≤Q_NEAR (core) or Q≥Q_FAR (divergent) are included.
    """
    dists = np.linalg.norm(X_pca - mode, axis=1)
    q_lo  = np.quantile(dists, Q_NEAR)
    q_hi  = np.quantile(dists, Q_FAR)
    near_mask = dists <= q_lo
    far_mask  = dists >= q_hi
    include   = near_mask | far_mask

    cluster_id = np.where(near_mask[include], 1, 0)
    print(f"  Core (≤Q{int(Q_NEAR*100)}, dist≤{q_lo:.3f}): {near_mask.sum()}")
    print(f"  Divergent (≥Q{int(Q_FAR*100)}, dist≥{q_hi:.3f}): {far_mask.sum()}")

    return pd.DataFrame(
        {'Cluster_ID': cluster_id, 'dist_to_mode': dists[include]},
        index=index[include],
    )


def run_centroid_volcano(group_df: pd.DataFrame) -> pd.DataFrame:
    """Match core vs divergent groups and run per-system phenotype volcano.

    Reuses cfc.build_matched_cohort (Mahalanobis 1:1, caliper=0.5) and
    cfc.run_system_volcano for consistency with the existing pipeline.
    """
    print("\nPhase 2: Mahalanobis matching (core vs divergent)...")
    valid_cases, valid_controls = cfc.build_matched_cohort(group_df)

    output_dir = os.path.join(cfc.OUTPUT_DIR, 'pc_axes_centroid')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for system, path in _list_system_csvs():
        try:
            res = cfc.run_system_volcano(
                valid_cases, valid_controls,
                system, path, output_dir,
                gender_label='full_cohort',
            )
        except Exception as e:
            print(f"  Skip {system} (volcano): {e}")
            res = None
        if res is not None:
            all_results.append(res)

    if not all_results:
        print("  No volcano results.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    out_path = paths.out_table('tableS_pc_centroid_matched.csv')
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows)")
    return combined


# ── Phase 3: Directional PC comparison (Option A) ────────────────────────────

def _mahalanobis_match(cases: pd.DataFrame, pool: pd.DataFrame,
                       match_cols: list[str], caliper: float = 1.0,
                       k: int = 1,
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:k Mahalanobis NN matching without replacement.

    Each case is matched to up to k controls within `caliper` standardised
    Mahalanobis distance. Controls are never reused. Cases are processed
    easiest-first (smallest nearest-neighbour distance) so the hardest cases
    get the remaining pool.

    Returns (matched_cases, matched_controls).  matched_cases contains one
    row per case that found ≥1 match; matched_controls may have up to k rows
    per case (each control row is unique).  For a paired pipeline pass the
    returned DataFrames directly; for an unpaired / regression pipeline pass
    only the unique subject indices.
    """
    from scipy.spatial.distance import cdist

    cases_clean = cases.dropna(subset=match_cols).copy()
    pool_clean  = pool.dropna(subset=match_cols).copy()

    if len(cases_clean) == 0 or len(pool_clean) == 0:
        raise ValueError("No valid subjects after NaN drop for matching.")

    all_X  = np.vstack([cases_clean[match_cols].values, pool_clean[match_cols].values])
    scaler = StandardScaler().fit(all_X)
    Xc = scaler.transform(cases_clean[match_cols].values)
    Xp = scaler.transform(pool_clean[match_cols].values)

    cov_mat = np.cov(np.vstack([Xc, Xp]), rowvar=False)
    VI = np.linalg.pinv(cov_mat)

    D = cdist(Xc, Xp, metric='mahalanobis', VI=VI)
    case_order = np.argsort(D.min(axis=1))   # easiest-match first
    used = np.zeros(D.shape[1], dtype=bool)
    matched_ci, matched_pi = [], []

    for ci in case_order:
        row = np.where(used, np.inf, D[ci])
        # collect up to k candidates within caliper, closest first
        candidates = np.argsort(row)
        found = 0
        for pi in candidates:
            if row[pi] > caliper:
                break
            matched_ci.append(ci)
            matched_pi.append(pi)
            used[pi] = True
            found += 1
            if found == k:
                break

    mc = cases_clean.iloc[matched_ci]
    mp = pool_clean.iloc[matched_pi]

    n_matched_cases = len(set(matched_ci))
    n_unmatched = len(cases_clean) - n_matched_cases
    print(f"    Matched: {n_matched_cases} cases × {k} controls "
          f"= {len(mc)} rows ({n_unmatched} cases excluded — beyond caliper)")

    # SMD on unique subjects (not inflated by repeated case rows)
    mc_uniq = cases_clean.iloc[sorted(set(matched_ci))]
    mp_uniq = pool_clean.iloc[sorted(set(matched_pi))]
    all_std = pd.concat([cases_clean[match_cols], pool_clean[match_cols]]).std()
    for col in match_cols:
        sd = all_std[col]
        smd = (mc_uniq[col].mean() - mp_uniq[col].mean()) / sd if sd > 0 else 0.0
        flag = 'OK' if abs(smd) < 0.1 else 'NOTE'
        print(f"    SMD {col}: {smd:.4f} ({flag})")

    return mc, mp


_MATCH_COLS_BY_LABEL = {
    # PC4 (fat distribution): age+sex+fat are the essential confounders.
    # Lean mass and height add little beyond sex for this axis.
    'fat_distribution': ['age', 'sex', 'total_fat_mass'],
    # PC3 (bone density): lean mass (muscle loading) is a primary BMD determinant,
    # must be controlled; height is dispensable given fat+lean captures body size.
    'bone_density':     ['age', 'sex', 'total_fat_mass', 'total_lean_mass'],
}
_MATCH_CSV_DIR = '/data/hpp_labdata/Analyses/gilsa/csv_files'


def _export_matched_subjects(valid_cases: pd.DataFrame, valid_controls: pd.DataFrame,
                              label: str) -> None:
    """Write RegistrationCode,Cluster_ID CSV for the external DE pipeline."""
    cases_regs = (valid_cases.index.get_level_values('RegistrationCode')
                  .unique().tolist())
    ctrl_regs  = (valid_controls.index.get_level_values('RegistrationCode')
                  .unique().tolist())
    rows = (
        [{'RegistrationCode': r, 'Cluster_ID': 1} for r in cases_regs] +
        [{'RegistrationCode': r, 'Cluster_ID': 0} for r in ctrl_regs]
    )
    df = pd.DataFrame(rows).drop_duplicates('RegistrationCode')

    local_path = paths.out_table(f'pc_{label}_matched_ids.csv')
    df.to_csv(local_path, index=False)
    print(f"    Exported {len(df)} subject IDs → {local_path}")

    try:
        os.makedirs(_MATCH_CSV_DIR, exist_ok=True)
        net_path = os.path.join(_MATCH_CSV_DIR, f'pc_{label}_matched.csv')
        df.to_csv(net_path, index=False)
        print(f"    Also wrote → {net_path}")
    except OSError:
        print(f"    (network path not mounted — local copy only)")


def run_directional_comparison(X_pca: np.ndarray, index: pd.Index,
                                covariates: pd.DataFrame,
                                pc_idx: int, label: str,
                                quantile_cut: float = 0.25,
                                caliper: float = 1.0,
                                k: int = 2,
                                ) -> pd.DataFrame:
    """Compare top vs bottom quartile on PC `pc_idx+1`, matched on body composition.

    Matching covariates are label-specific (see _MATCH_COLS_BY_LABEL).
    1:k greedy Mahalanobis matching without replacement.

    Args:
        pc_idx:        0-based PC index
        label:         key into _MATCH_COLS_BY_LABEL  (e.g. 'fat_distribution')
        quantile_cut:  tail fraction (default 0.25 = Q1/Q4)
        caliper:       Mahalanobis caliper
        k:             controls per case (default 2 → 1:2 matching)
    """
    match_cols = _MATCH_COLS_BY_LABEL.get(label, ['age', 'sex', 'total_fat_mass'])
    pc_name = f'PC{pc_idx + 1}'
    print(f"\nDirectional comparison: {pc_name} ({label}) ──")
    print(f"  Match covariates: {match_cols}  caliper={caliper}  k={k}")

    scores = X_pca[:, pc_idx]
    q_lo = np.quantile(scores, quantile_cut)
    q_hi = np.quantile(scores, 1 - quantile_cut)

    top_mask = scores >= q_hi
    bot_mask = scores <= q_lo
    print(f"  {pc_name} top (≥Q{int((1-quantile_cut)*100)}): {top_mask.sum()} subjects")
    print(f"  {pc_name} bot (≤Q{int(quantile_cut*100)}):  {bot_mask.sum()} subjects")

    combined_mask = top_mask | bot_mask
    cluster_id = np.where(top_mask[combined_mask], 1, 0)  # 1=top, 0=bottom

    group_df = pd.DataFrame(
        {'Cluster_ID': cluster_id, f'{pc_name}_score': scores[combined_mask]},
        index=index[combined_mask],
    )

    covs = covariates[match_cols].reindex(index[combined_mask]).copy()
    for c in ['total_fat_mass', 'total_lean_mass']:
        if c in covs:
            covs.loc[covs[c] == 0, c] = np.nan

    df = group_df.join(covs, how='inner').dropna(subset=match_cols + ['Cluster_ID'])
    cases    = df[df['Cluster_ID'] == 1]
    controls = df[df['Cluster_ID'] == 0]
    print(f"  After NaN drop: cases={len(cases)}, pool={len(controls)}")

    valid_cases, valid_controls = _mahalanobis_match(
        cases, controls, match_cols, caliper=caliper, k=k
    )

    if len(valid_cases) == 0:
        print(f"  No matched subjects for {pc_name} ({label}) — skipping.")
        return pd.DataFrame()

    _export_matched_subjects(valid_cases, valid_controls, label)

    output_dir = os.path.join(cfc.OUTPUT_DIR, f'pc_axes_{label}')
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for system, path in _list_system_csvs():
        try:
            res = cfc.run_system_volcano(
                valid_cases, valid_controls,
                system, path, output_dir,
                gender_label='full_cohort',
            )
        except Exception as e:
            print(f"  Skip {system} (volcano): {e}")
            res = None
        if res is not None:
            all_results.append(res)

    if not all_results:
        print(f"  No volcano results for {pc_name} ({label}).")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    out_name = f'tableS_{pc_name.lower()}_{label}.csv'
    out_path = paths.out_table(out_name)
    combined.to_csv(out_path, index=False)
    n_sig = combined.get('significant', pd.Series(dtype=bool)).sum()
    print(f"  Wrote {out_path} ({len(combined)} rows, {n_sig} significant)")
    return combined


# ── Entry point ───────────────────────────────────────────────────────────────

_DIRECTIONAL_PCS = [
    # (pc_idx, label, description, caliper)
    # Covariates per label → _MATCH_COLS_BY_LABEL; k=2 (1:2) everywhere.
    # PC4: age+sex+fat, caliper=1.5, 1:2
    # PC3: age+sex+fat+lean, caliper=1.0, 1:2  (tighter — lean critical for BMD)
    (3, 'fat_distribution', 'PC4 — visceral vs peripheral fat distribution', 1.5),
    (2, 'bone_density',     'PC3 — bone mineralisation density',             1.0),
]


def main(phase2_only: bool = False, directional_only: bool = False):
    X_pca, var_ratio, index, _ = load_full_cohort_embeddings()
    covariates = load_pc_covariates(index)

    if not phase2_only and not directional_only:
        # ── Phase 1 ──────────────────────────────────────────────────────────
        assoc_df = run_per_pc_regressions(X_pca, index, covariates)
        assoc_path = paths.out_table('tableS_pc_associations.csv')
        assoc_df.to_csv(assoc_path, index=False)
        print(f"\nWrote {assoc_path}  ({len(assoc_df):,} rows)")

        var_df = pd.DataFrame({
            'pc':                      [f'PC{i+1}' for i in range(N_PCS)],
            'explained_variance_ratio': var_ratio,
        })
        var_path = paths.out_table('tableS_pc_variance.csv')
        var_df.to_csv(var_path, index=False)
        print(f"Wrote {var_path}")

    if not directional_only:
        # ── Phase 2: KDE centroid (kept for reference) ────────────────────────
        mode     = find_kde_mode(X_pca)
        group_df = assign_distance_groups(X_pca, mode, index)

        group_path = paths.out_table('tableS_pc_group_assignments.csv')
        group_df.reset_index().to_csv(group_path, index=False)
        print(f"Wrote {group_path}")

        run_centroid_volcano(group_df)

    # ── Phase 3: Directional PC comparisons (Option A) ───────────────────────
    print("\n=== Phase 3: Directional PC comparisons (matched on fat + lean) ===")
    for pc_idx, label, desc, caliper in _DIRECTIONAL_PCS:
        print(f"\n{desc}")
        run_directional_comparison(X_pca, index, covariates, pc_idx, label,
                                   caliper=caliper, k=2)

    print("\nDone.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase2-only', action='store_true',
                    help='Skip Phase 1 regressions (use existing associations CSV)')
    ap.add_argument('--directional-only', action='store_true',
                    help='Skip Phase 1 and KDE centroid — run only directional comparisons')
    args = ap.parse_args()
    main(phase2_only=args.phase2_only, directional_only=args.directional_only)
