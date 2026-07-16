# LIVE — generates the Figure-2 HPP 4-arm raw scores. Run with
#   --targets-csv csvs/disease_targets_with_covs.csv  -> lp_cov_disease_4arm_raw.csv
# which is consumed by export_supplementary_tables_4arm.py (supp_tableA) and
# compute_disease_topset.py (disease_pairwise.csv → heatmap significance).
# NOTE: an earlier header called this "deprecated, superseded by compare_lp.py" —
# that is INACCURATE: compare_lp.py feeds a different figure (plot_combined_figure.py)
# and does NOT produce the 4-arm raw file this script does.
"""
compare_lp_cov.py

Covariate-adjusted LP comparison for LP regression and disease classification.

Covariates are incorporated by EARLY FUSION: concatenated directly into the
design matrix alongside the embedding / tabular feature block, then a single
regularized model (RidgeCV / LogisticRegressionCV) is fit on the combined
features. (This replaces an earlier meta-learner stacking implementation —
early fusion lets the fit weight individual covariate and feature dimensions
jointly, rather than only blending two frozen per-block prediction scores.)

Three covariate-adjusted arms (+ a covariates-only baseline):
  lejepa_cov : [LeJEPA FM bone+tissue] + covariates
  dino_cov   : [DINO FM bone+tissue]   + covariates
  tab_cov    : [DXA tabular]           + covariates

Usage:
  python compare_lp_cov.py --targets age ahi --num-seeds 20
  python compare_lp_cov.py --targets-csv csvs/disease_targets.csv --num-seeds 20
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import common.utils as U
from downstream.disease.linear_probe import (
    _COV_COLS, _SEX_FILTER_MAP, _SSL_MODELS,
    _TABULAR_LEAKAGE_EXCLUSIONS, _impute, load_embeddings, _fit_predict_blocks,
)

_RESULTS_DIR = "/data/hpp_labdata/Analyses/gilsa/results/comparison"


DIFF_PEN    = False   # --diff-pen: per-block penalisation (covariate block loosened vs feature block)
SINGLE_PEN  = False   # --single-pen: include regional block but fit ONE shared penalty (no per-block)
TWO_PEN     = False   # --two-pen: 2 blocks only — [bone+tissue+regional] | [cov], one penalty each
GROUP_LASSO = False   # --group-lasso: block-level L1 (whole-block selection) + within-block L2
BONE_POOL   = False   # --bone-pool: pool regional INTO the bone view (no extra block), then [fm|cov]
REGION_POOL = False   # --region-pool: split SSL arm into bone | tissue | regional | cov blocks,
                      #                each separately penalised (coordinate-descent scale sweep)
COV_FREE_SCALE = None # --cov-free-scale: sensitivity mode; fix last/covariate block scale
                      #                   after standardization instead of CV-tuning it.


def _fit_base(X_tr, X_val, y_tr, is_cls: bool, n_cov: int = 0):
    """Fit Ridge/Logistic on train block, return (train_pred_in_sample, val_pred).

    Under DIFF_PEN: tuned per-block penalisation (mirrors the Cox per-block penalizer).
    CV-selects a covariate-block looseness s (covariates penalised 1/s² as hard as the
    feature block) jointly with the feature penalty. s=1 ≡ shared penalty; large s ≈
    covariates free. Avoids both over-shrinking covariates (the drops) and over-shrinking
    the feature block (which hurt imaging-dominant diseases under full unpenalisation).
    """
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    diff = DIFF_PEN and 0 < n_cov < X_tr.shape[1]
    s_grid = (1.0, 4.0, 16.0, 64.0) if diff else (1.0,)
    best = None
    for s in s_grid:
        Xtr_s, Xval_s = X_tr, X_val
        if diff and s != 1.0:
            Xtr_s = X_tr.copy();  Xtr_s[:, -n_cov:]  *= s
            Xval_s = X_val.copy(); Xval_s[:, -n_cov:] *= s
        if is_cls:
            m = LogisticRegressionCV(Cs=5, cv=U.RIDGE_CV_FOLDS, solver="lbfgs",
                                     max_iter=2000, n_jobs=1,
                                     scoring=("roc_auc" if diff else None))
            m.fit(Xtr_s, y_tr)
            sc = float(np.mean(next(iter(m.scores_.values())), axis=0).max()) if diff else 0.0
            pred = (m.decision_function(Xtr_s), m.decision_function(Xval_s))
        else:
            m = RidgeCV(alphas=U.RIDGE_ALPHAS, cv=U.RIDGE_CV_FOLDS)
            m.fit(Xtr_s, y_tr)
            sc = float(getattr(m, "best_score_", 0.0)) if diff else 0.0
            pred = (m.predict(Xtr_s), m.predict(Xval_s))
        if best is None or sc > best[0]:
            best = (sc, pred)
        if not diff:
            break
    return best[1]


def _cov_adjusted_score(base_specs, y_tr, y_val, is_cls):
    """Early-fusion covariate adjustment.

    Concatenate every feature block (e.g. the embedding/tabular block and the
    covariate block) into a single design matrix, then fit ONE regularized model
    (RidgeCV / LogisticRegressionCV) on the combined features. The covariates
    enter as ordinary features, so the fit weights individual covariate and
    feature dimensions jointly — unlike meta-learner stacking, which can only
    blend two frozen per-block prediction scores.

    base_specs: list of (name, X_train_block, X_val_block) — e.g. [bone, tissue,
    regional, cov]. Under --diff-pen or --region-pool each block is separately
    penalised via coordinate-descent scale selection; otherwise a single shared fit.
    Returns (score, None, None) — the trailing Nones keep the call-site
    signature unchanged from the previous stacking implementation.
    """
    if GROUP_LASSO:
        block_sizes = [X.shape[1] for _, X, _ in base_specs]
        val_pred = _fit_group_lasso(base_specs, y_tr, y_val, is_cls, block_sizes)
        return float(U.metric(y_val, val_pred, is_cls)), None, None
    X_tr  = np.concatenate([X for _, X, _ in base_specs], axis=1)
    X_val = np.concatenate([X for _, _, X in base_specs], axis=1)
    sizes = ([X.shape[1] for _, X, _ in base_specs]
             if ((DIFF_PEN or REGION_POOL or TWO_PEN) and not SINGLE_PEN) else None)
    fixed_scales = None
    if COV_FREE_SCALE is not None and sizes is not None and len(sizes) >= 2:
        fixed_scales = [1.0] * len(sizes)
        fixed_scales[-1] = float(COV_FREE_SCALE)
    val_pred, _ = _fit_predict_blocks(
        X_tr, X_val, y_tr, y_val, is_cls,
        block_sizes=sizes,
        fixed_block_scales=fixed_scales,
    )
    return float(U.metric(y_val, val_pred, is_cls)), None, None


def _fit_group_lasso(base_specs, y_tr, y_val, is_cls, block_sizes):
    """Group lasso: block-level L1 (zero out whole blocks) + within-block L2.
    Groups = the feature blocks (bone|tissue|regional|cov). scale_reg='group_size'
    applies the canonical sqrt(block size) weighting so a big embedding block is not
    cheaper to switch on than the 3-dim covariate block. group_reg is inner-CV-tuned."""
    from group_lasso import LogisticGroupLasso, GroupLasso
    from sklearn.model_selection import StratifiedKFold, KFold
    X_tr = np.concatenate([X for _, X, _ in base_specs], axis=1)
    X_val = np.concatenate([X for _, _, X in base_specs], axis=1)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_val = sc.transform(X_tr), sc.transform(X_val)
    groups = np.concatenate([[gi] * b for gi, b in enumerate(block_sizes)])
    grid = [0.003, 0.01, 0.03]

    def _mk(gr):
        kw = dict(groups=groups, group_reg=gr, l1_reg=0.0, scale_reg="group_size",
                  n_iter=600, tol=3e-3, supress_warning=True)
        return LogisticGroupLasso(**kw) if is_cls else GroupLasso(**kw)

    def _pred(m, X):
        if is_cls:
            p = np.asarray(m.predict_proba(X)); return p[:, 1] if p.ndim == 2 else p
        return np.asarray(m.predict(X)).ravel()

    # inner CV to pick group_reg
    splitter = (StratifiedKFold(3, shuffle=True, random_state=0) if is_cls
                else KFold(3, shuffle=True, random_state=0))
    best_gr, best_cv = grid[0], -np.inf
    for gr in grid:
        sc_folds = []
        for itr, ite in splitter.split(X_tr, y_tr if is_cls else None):
            try:
                m = _mk(gr).fit(X_tr[itr], y_tr[itr])
                sc_folds.append(U.metric(y_tr[ite], _pred(m, X_tr[ite]), is_cls))
            except Exception:
                sc_folds.append(np.nan)
        mcv = np.nanmean(sc_folds)
        if np.isfinite(mcv) and mcv > best_cv:
            best_cv, best_gr = mcv, gr
    m = _mk(best_gr).fit(X_tr, y_tr)
    return _pred(m, X_val)


def _ssl_cov_specs(embs, model, tr_idx, val_idx, cov_tr, cov_val):
    """Build the (name, X_tr, X_val) blocks for an SSL+covariate arm.
    Region-pool → [bone | tissue | regional | cov] (each separately penalised);
    otherwise → [fused bone+tissue | cov] (the published 2-block layout)."""
    bone_tr  = embs[model]["bone"].loc[tr_idx].values
    bone_val = embs[model]["bone"].loc[val_idx].values
    tis_tr   = embs[model]["tissue"].loc[tr_idx].values
    tis_val  = embs[model]["tissue"].loc[val_idx].values
    has_region = REGION_POOL and "regionpool" in embs[model]
    if has_region:
        rp_tr, rp_val = _impute(embs[model]["regionpool"].reindex(tr_idx).values.astype(float),
                                embs[model]["regionpool"].reindex(val_idx).values.astype(float))
    if TWO_PEN:
        # 2 blocks only: [bone+tissue(+regional)] | [cov]. One penalty per block.
        parts_tr = [bone_tr, tis_tr] + ([rp_tr] if has_region else [])
        parts_val = [bone_val, tis_val] + ([rp_val] if has_region else [])
        emb_tr = np.concatenate(parts_tr, axis=1)
        emb_val = np.concatenate(parts_val, axis=1)
        return [("emb", emb_tr, emb_val), ("cov", cov_tr, cov_val)]
    if has_region:
        return [("bone", bone_tr, bone_val), ("tissue", tis_tr, tis_val),
                ("regional", rp_tr, rp_val), ("cov", cov_tr, cov_val)]
    fm_tr  = np.concatenate([bone_tr, tis_tr], axis=1)
    fm_val = np.concatenate([bone_val, tis_val], axis=1)
    return [("fm", fm_tr, fm_val), ("cov", cov_tr, cov_val)]


def run_target_cov(target_col, target_df_full, embs, tabular_df, seeds,
                        do_lejepa_cov=True, do_dino_cov=True, do_tab_cov=True,
                        do_cov_only=True, min_prevalence: float = 0.0):
    is_cls = target_col.lower() in U.CLASSIFICATION_TARGETS
    metric_name = "auc" if is_cls else "pearson"

    target_df = target_df_full[[target_col]].dropna().copy()
    if target_col in _SEX_FILTER_MAP and "gender" in target_df_full.columns:
        required_sex = _SEX_FILTER_MAP[target_col]
        sex_idx = target_df_full.index[target_df_full["gender"] == required_sex]
        target_df = target_df[target_df.index.isin(sex_idx)]

    if len(target_df) < U.MIN_SAMPLES_PER_TARGET:
        print(f"[Skip] {target_col}: only {len(target_df)} labeled samples")
        return [], []

    if is_cls:
        uniq = np.sort(target_df[target_col].unique())
        if len(uniq) != 2:
            print(f"[Skip] {target_col}: not binary ({len(uniq)} classes)")
            return [], []
        target_df[target_col] = target_df[target_col].map({uniq[0]: 0.0, uniq[1]: 1.0}).astype(float)
        n_pos = int((target_df[target_col] == 1.0).sum())
        if n_pos < U.MIN_POSITIVE_CASES:
            print(f"[Skip] {target_col}: only {n_pos} positive cases")
            return [], []
    else:
        std = target_df[target_col].std()
        if std == 0 or pd.isna(std):
            print(f"[Skip] {target_col}: zero variance")
            return [], []
        target_df[target_col] = (target_df[target_col] - target_df[target_col].mean()) / std

    avail_cov = [c for c in _COV_COLS if c in target_df_full.columns and c != target_col]
    if not avail_cov:
        print(f"[Skip] {target_col}: no covariates available")
        return [], []

    # Subject filtering: must have lejepa embedding AND (if tab_cov requested) valid tabular
    bone_idx = set(embs["lejepa"]["bone"].index)
    labeled_subjects = {}
    for idx in target_df.index:
        reg, _ = idx
        labeled_subjects.setdefault(reg, []).append(idx)
    labeled_subjects = {
        s: rows for s, rows in labeled_subjects.items()
        if any(r in bone_idx for r in rows)
    }

    if do_tab_cov and tabular_df is not None:
        feat_cols_tab = [c for c in tabular_df.columns
                         if pd.api.types.is_numeric_dtype(tabular_df[c])]
        if target_col in _TABULAR_LEAKAGE_EXCLUSIONS:
            excl = set(_TABULAR_LEAKAGE_EXCLUSIONS[target_col](feat_cols_tab))
            feat_cols_tab = [c for c in feat_cols_tab if c not in excl]
        thresh = int(0.5 * len(feat_cols_tab))
        tab_pass = tabular_df[feat_cols_tab].isnull().sum(axis=1) <= thresh
        valid_tab_idx = set(tab_pass[tab_pass].index)
        valid_tab_rcs = {rc for rc, _ in valid_tab_idx}
        labeled_subjects = {
            s: [r for r in rows if r in valid_tab_idx]
            for s, rows in labeled_subjects.items() if s in valid_tab_rcs
        }
        labeled_subjects = {s: rows for s, rows in labeled_subjects.items() if rows}

    valid_subjects = sorted(labeled_subjects)
    if len(valid_subjects) < 2:
        print(f"[Skip] {target_col}: insufficient subjects ({len(valid_subjects)})")
        return [], []

    # Prevalence filter on intersected cohort (embedding ∩ labels)
    if is_cls and min_prevalence > 0:
        all_valid_idx = [r for s in valid_subjects for r in labeled_subjects[s]]
        n_pos_valid = int((target_df.loc[all_valid_idx, target_col] == 1.0).sum())
        n_valid = len(all_valid_idx)
        if n_valid > 0 and n_pos_valid / n_valid < min_prevalence:
            print(f"[Skip] {target_col}: prevalence {n_pos_valid/n_valid:.1%} < {min_prevalence:.0%} "
                  f"in intersected cohort (n_pos={n_pos_valid}, n={n_valid})")
            return [], []

    print(f"\n>>> {target_col} | {len(valid_subjects)} subjects | {metric_name}")

    summary_rows, raw_rows = [], []
    scores = {"lejepa_cov": [], "dino_cov": [], "tab_cov": [], "covariates": []}
    have_dino = "dino" in embs

    for seed in seeds:
        train_subs, val_subs = train_test_split(valid_subjects, test_size=0.2, random_state=seed)
        train_idx = [r for s in train_subs for r in labeled_subjects[s] if r in target_df.index]
        val_idx   = [r for s in val_subs   for r in labeled_subjects[s] if r in target_df.index]

        bone_idx_set = set(embs["lejepa"]["bone"].index)
        dino_idx_set = set(embs["dino"]["bone"].index) if have_dino else None
        tab_idx_set = set(tabular_df.index) if tabular_df is not None else None

        # Arm-specific valid sets (each arm needs all required blocks present)
        tr_lc  = [r for r in train_idx if r in bone_idx_set]
        val_lc = [r for r in val_idx   if r in bone_idx_set]
        if dino_idx_set is not None:
            tr_dc  = [r for r in train_idx if r in dino_idx_set]
            val_dc = [r for r in val_idx   if r in dino_idx_set]
        else:
            tr_dc, val_dc = [], []
        if tab_idx_set is not None:
            tr_tc  = [r for r in train_idx if r in bone_idx_set and r in tab_idx_set]
            val_tc = [r for r in val_idx   if r in bone_idx_set and r in tab_idx_set]
        else:
            tr_tc, val_tc = [], []

        if len(val_lc) < 10:
            print(f"  seed={seed}: too few lejepa val rows, skipping")
            continue

        if do_cov_only:
            # Cov-only baseline scored on the SAME cohort as lejepa_cov (anchored on
            # tr_lc / val_lc) so paired t-tests are valid both in seed AND in val rows.
            try:
                y_tr_c  = target_df.loc[tr_lc,  target_col].values.astype(float)
                y_val_c = target_df.loc[val_lc, target_col].values.astype(float)
                cov_tr_raw  = target_df_full.loc[tr_lc,  avail_cov].values.astype(float)
                cov_val_raw = target_df_full.loc[val_lc, avail_cov].values.astype(float)
                cov_tr, cov_val = _impute(cov_tr_raw, cov_val_raw)
                _, p_val = _fit_base(cov_tr, cov_val, y_tr_c, is_cls)
                score_cv = float(U.metric(y_val_c, p_val, is_cls))
                print(f"  seed={seed}  covariates={score_cv:.4f}  (n_val={len(val_lc)})")
                scores["covariates"].append(score_cv)
                raw_rows.append({"target": target_col, "metric": metric_name,
                                 "model": "covariates", "mode": "ridge",
                                 "fusion": "early", "view": "fusion", "seed": seed,
                                 "score": score_cv})
            except Exception as e:
                print(f"  seed={seed}  covariates ERROR: {e}")

        if do_lejepa_cov:
            y_tr_l  = target_df.loc[tr_lc,  target_col].values.astype(float)
            y_val_l = target_df.loc[val_lc, target_col].values.astype(float)
            cov_tr_raw  = target_df_full.loc[tr_lc,  avail_cov].values.astype(float)
            cov_val_raw = target_df_full.loc[val_lc, avail_cov].values.astype(float)
            cov_tr, cov_val = _impute(cov_tr_raw, cov_val_raw)
            specs = _ssl_cov_specs(embs, "lejepa", tr_lc, val_lc, cov_tr, cov_val)
            score_lc, _, _ = _cov_adjusted_score(specs, y_tr_l, y_val_l, is_cls)
            print(f"  seed={seed}  lejepa_cov={score_lc:.4f}  (n_val={len(val_lc)})")
            scores["lejepa_cov"].append(score_lc)
            raw_rows.append({"target": target_col, "metric": metric_name,
                             "model": "lejepa_cov", "mode": "ridge",
                             "fusion": "early", "view": "fusion", "seed": seed,
                             "score": score_lc})

        if do_dino_cov and have_dino and len(val_dc) >= 10:
            try:
                y_tr_d  = target_df.loc[tr_dc,  target_col].values.astype(float)
                y_val_d = target_df.loc[val_dc, target_col].values.astype(float)
                cov_tr_raw  = target_df_full.loc[tr_dc,  avail_cov].values.astype(float)
                cov_val_raw = target_df_full.loc[val_dc, avail_cov].values.astype(float)
                cov_tr, cov_val = _impute(cov_tr_raw, cov_val_raw)
                specs = _ssl_cov_specs(embs, "dino", tr_dc, val_dc, cov_tr, cov_val)
                score_dc, _, _ = _cov_adjusted_score(specs, y_tr_d, y_val_d, is_cls)
                print(f"  seed={seed}  dino_cov={score_dc:.4f}  (n_val={len(val_dc)})")
                scores["dino_cov"].append(score_dc)
                raw_rows.append({"target": target_col, "metric": metric_name,
                                 "model": "dino_cov", "mode": "ridge",
                                 "fusion": "early", "view": "fusion", "seed": seed,
                                 "score": score_dc})
            except Exception as e:
                print(f"  seed={seed}  dino_cov ERROR: {e}")

        if do_tab_cov and tabular_df is not None and len(val_tc) >= 10:
            try:
                y_tr_t  = target_df.loc[tr_tc,  target_col].values.astype(float)
                y_val_t = target_df.loc[val_tc, target_col].values.astype(float)
                cov_tr_raw  = target_df_full.loc[tr_tc,  avail_cov].values.astype(float)
                cov_val_raw = target_df_full.loc[val_tc, avail_cov].values.astype(float)
                cov_tr, cov_val = _impute(cov_tr_raw, cov_val_raw)
                feat_cols = [c for c in tabular_df.columns
                             if pd.api.types.is_numeric_dtype(tabular_df[c])]
                if target_col in _TABULAR_LEAKAGE_EXCLUSIONS:
                    excl = set(_TABULAR_LEAKAGE_EXCLUSIONS[target_col](feat_cols))
                    feat_cols = [c for c in feat_cols if c not in excl]
                tab_tr_raw  = tabular_df.loc[tr_tc,  feat_cols].values.astype(float)
                tab_val_raw = tabular_df.loc[val_tc, feat_cols].values.astype(float)
                tab_tr, tab_val = _impute(tab_tr_raw, tab_val_raw)
                score_tc, _, _ = _cov_adjusted_score(
                    [("tab", tab_tr, tab_val), ("cov", cov_tr, cov_val)],
                    y_tr_t, y_val_t, is_cls,
                )
                print(f"  seed={seed}  tab_cov={score_tc:.4f}  (n_val={len(val_tc)})")
                scores["tab_cov"].append(score_tc)
                raw_rows.append({"target": target_col, "metric": metric_name,
                                 "model": "tab_cov", "mode": "ridge",
                                 "fusion": "early", "view": "fusion", "seed": seed,
                                 "score": score_tc})
            except Exception as e:
                print(f"  seed={seed}  tab_cov ERROR: {e}")

    print(f"\n{'═'*60}")
    print(f"  {target_col} — covariate-adjusted summary (val {metric_name})")
    print(f"{'═'*60}")
    for k, lst in scores.items():
        if not lst:
            continue
        arr = np.array(lst, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se  = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        print(f"  {k:<22} {arr.mean():.4f} ± {arr.std():.4f} (n={len(arr)})")
        summary_rows.append({"target": target_col, "metric": metric_name,
                             "model": k, "mode": "ridge", "fusion": "early",
                             "view": "fusion",
                             "n": len(lst), "mean": float(arr.mean()),
                             "std": std, "se": se})
    return summary_rows, raw_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--targets-csv", default=None,
                   help="Override targets CSV (e.g. csvs/disease_targets.csv)")
    p.add_argument("--num-seeds", type=int, default=10)
    p.add_argument("--min-prevalence", type=float, default=0.02,
                   help="Minimum disease prevalence in intersected cohort (default 0.02 = 2%%)")
    p.add_argument("--embeddings-dir", default=U.EMBEDDINGS_DIR)
    p.add_argument("--tabular-csv", default="/path/to/dxa_filtered.csv")
    p.add_argument("--no-tab-cov", action="store_true",
                   help="Skip tab_cov (e.g. for cohorts without tabular)")
    p.add_argument("--no-dino-cov", action="store_true",
                   help="Skip dino_cov (e.g. when DINO embeddings unavailable)")
    p.add_argument("--no-cov-only", action="store_true",
                   help="Skip the covariates-only baseline arm")
    p.add_argument("--cls-auto-detect", action="store_true",
                   help="Treat any binary {0,1} column as a classification target.")
    p.add_argument("--results-csv",     default=f"{_RESULTS_DIR}/lp_cov_summary.csv")
    p.add_argument("--results-raw-csv", default=f"{_RESULTS_DIR}/lp_cov_raw.csv")
    p.add_argument("--diff-pen", action="store_true",
                   help="Differential per-block penalisation: leave the age/sex/BMI covariate "
                        "block unpenalised; tune the penalty only on the feature block.")
    p.add_argument("--region-pool", action="store_true",
                   help="Split SSL+cov arms into bone | tissue | regional | cov blocks, each "
                        "separately penalised (coordinate-descent scale sweep). Implies per-block. "
                        "Requires {model}_regionpool.pkl.")
    p.add_argument("--single-pen", action="store_true",
                   help="Include the regional block (like --region-pool) but fit ONE shared "
                        "penalty over [bone+tissue+regional+cov] instead of per-block scaling. "
                        "Isolates the regional information from the per-block regularisation change.")
    p.add_argument("--two-pen", action="store_true",
                   help="Two penalizers only: [bone+tissue+regional] | [cov], each block one "
                        "penalty (no bone/tissue/regional sub-splitting). Regional included.")
    p.add_argument("--group-lasso", action="store_true",
                   help="Group lasso over blocks [bone|tissue|regional|cov]: block-level L1 "
                        "(can zero a whole block) + within-block L2. sqrt(size)-weighted; "
                        "group_reg inner-CV-tuned. Includes the regional block.")
    p.add_argument("--bone-pool", action="store_true",
                   help="Pool regional INTO the bone view (bone* = mean(bone, regional)); no "
                        "separate block. Fits [bone*+tissue | cov] two-pen (decoupled cov penalty).")
    p.add_argument("--cov-free-scale", type=float, default=None,
                   help="Sensitivity analysis: fix the last/covariate block scale after "
                        "standardization instead of tuning it. Large values approximate "
                        "unpenalized covariates; try 100.")
    args = p.parse_args()

    global DIFF_PEN, REGION_POOL, SINGLE_PEN, TWO_PEN, GROUP_LASSO, BONE_POOL, COV_FREE_SCALE
    GROUP_LASSO = args.group_lasso
    BONE_POOL = args.bone_pool
    TWO_PEN = args.two_pen
    REGION_POOL = args.region_pool or args.single_pen or args.two_pen or args.group_lasso
    SINGLE_PEN = args.single_pen
    COV_FREE_SCALE = args.cov_free_scale
    DIFF_PEN = (args.diff_pen or args.region_pool or args.bone_pool) and not args.single_pen and not args.two_pen and not args.group_lasso
    if BONE_POOL:
        print("Bone-pool ON: bone* = mean(bone, regional); fit [bone*+tissue | cov], decoupled cov.")
    elif GROUP_LASSO:
        print("Group-lasso ON: blocks = [bone|tissue|regional|cov], block-L1 + within-block-L2.")
    elif TWO_PEN:
        print("Two-pen ON: blocks = [bone+tissue+regional] | [cov], one penalty each.")
    elif SINGLE_PEN:
        print("Single-pen region-pool ON: blocks = bone+tissue+regional+cov, ONE shared penalty.")
    elif REGION_POOL:
        print("Region-pool ON: SSL arms = bone | tissue | regional | cov, per-block penalised.")
    elif DIFF_PEN:
        print("Differential penalisation ON: covariate block left unpenalised.")
    if COV_FREE_SCALE is not None:
        print(f"Covariate-free sensitivity ON: last block fixed scale = {COV_FREE_SCALE:g}.")

    seeds = U.make_seeds(args.num_seeds)
    print(f"Seeds: {seeds}")

    ssl_models = ["lejepa"] + ([] if args.no_dino_cov else ["dino"])
    print(f"Loading {ssl_models} embeddings from {args.embeddings_dir}")
    embs = load_embeddings(args.embeddings_dir, ssl_models)

    if BONE_POOL:
        # Pool the regional (femur+lumbar) scans INTO the bone view rather than as a
        # separate block: bone* = mean(whole-body bone, regional-pooled). Same dim,
        # no extra block — enriches the bone channel without inflating capacity.
        for m in [mm for mm in ssl_models if mm in embs and "regionpool" in embs[mm]]:
            bone = embs[m]["bone"]
            rg = embs[m].pop("regionpool").reindex(bone.index)
            ok = rg.notna().all(axis=1)
            enriched = bone.copy()
            enriched.loc[ok] = (bone.loc[ok].values + rg.loc[ok].values) / 2.0
            embs[m]["bone"] = enriched
            print(f"  [{m}] bone-pool: enriched bone with regional on {int(ok.sum())}/{len(bone)} "
                  f"subjects (rest keep whole-body bone)")

    tabular_df = None
    if not args.no_tab_cov:
        print(f"Loading tabular from {args.tabular_csv}")
        tabular_df = pd.read_csv(args.tabular_csv).set_index(["RegistrationCode", "research_stage"])
        tabular_df = tabular_df[~tabular_df.index.duplicated(keep="first")]
        tabular_df.sort_index(inplace=True)
        tabular_df = tabular_df.select_dtypes(include="number")
        print(f"  Tabular: {len(tabular_df)} rows × {tabular_df.shape[1]} features")

    targets_csv = args.targets_csv or U.TARGETS_CSV
    target_df_full = pd.read_csv(targets_csv, index_col=[0, 1])
    target_df_full.sort_index(inplace=True)

    if args.cls_auto_detect:
        auto_cls = set()
        for c in target_df_full.columns:
            vals = set(target_df_full[c].dropna().unique())
            if vals.issubset({0.0, 1.0, 0, 1}):
                auto_cls.add(c)
        U.CLASSIFICATION_TARGETS = auto_cls
        print(f"Auto-detected {len(auto_cls)} classification targets")

    target_cols = args.targets or [
        c for c in target_df_full.columns
        if c.startswith("dis__") or c in U.CLASSIFICATION_TARGETS
        or pd.api.types.is_numeric_dtype(target_df_full[c])
    ]

    all_summary, all_raw = [], []
    for tc in target_cols:
        if tc not in target_df_full.columns:
            print(f"[Skip] {tc}: not in targets CSV")
            continue
        try:
            srows, rrows = run_target_cov(
                tc, target_df_full, embs, tabular_df, seeds,
                do_lejepa_cov=True,
                do_dino_cov=(not args.no_dino_cov and "dino" in embs),
                do_tab_cov=(not args.no_tab_cov and tabular_df is not None),
                do_cov_only=(not args.no_cov_only),
                min_prevalence=args.min_prevalence,
            )
            all_summary.extend(srows)
            all_raw.extend(rrows)
        except Exception as e:
            print(f"[ERROR] {tc}: {e}")
            import traceback; traceback.print_exc()

    if all_summary:
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
        pd.DataFrame(all_summary).to_csv(args.results_csv, index=False)
        pd.DataFrame(all_raw).to_csv(args.results_raw_csv, index=False)
        print(f"\nSaved summary → {args.results_csv}")
        print(f"Saved raw     → {args.results_raw_csv}")


if __name__ == "__main__":
    main()
