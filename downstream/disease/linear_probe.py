"""Ridge linear-probe comparison on pre-extracted embeddings.

Requires ``python -m model.extract_embeddings`` to have been run first.

Protocol per (target × model × seed):
  - 80/20 subject-level train/val split (per seed)
  - Early fusion: concatenate bone + tissue embeddings → single RidgeCV
    (5 log-spaced alphas, 2-fold CV within train set)
  - Covariates model: Ridge on [age, gender, bmi] from targets CSV
  - Ensemble model: LeJEPA bone+tissue + covariates → single RidgeCV

Outputs:
  lp_summary.csv  — mean ± std/SE across seeds per config
  lp_raw.csv      — raw per-seed scores
  lp_ttest.csv    — paired t-test: lejepa vs dino per target

Usage:
  python -m downstream.disease.linear_probe --targets age bmi
  python -m downstream.disease.linear_probe --models lejepa dino
"""

import argparse
import os
from itertools import product as _product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel, loguniform

import common.utils as U
from config import RESULTS_DIR

_RESULTS_DIR     = str(RESULTS_DIR)
_DEFAULT_SUMMARY = f"{_RESULTS_DIR}/lp_summary.csv"
_DEFAULT_RAW     = f"{_RESULTS_DIR}/lp_raw.csv"
_DEFAULT_TTEST   = f"{_RESULTS_DIR}/lp_ttest.csv"

_SSL_MODELS     = {"lejepa", "dino", "dino7b"}
_SPECIAL_MODELS = {"covariates", "ensemble", "tabular"}
_COV_COLS       = ["age", "gender", "bmi"]
_RAND_N_ITER: "int | None" = None  # set by --rand-search
_N_CS: int = 5                    # set by --n-cs

# DXA-derived targets: exclude sibling features to prevent leakage
_TABULAR_LEAKAGE_EXCLUSIONS = {
    "femur_neck_mean_bmd":       lambda cols: [c for c in cols if "femur_neck" in c],
    "body_comp_total_lean_mass": lambda cols: [c for c in cols if "lean" in c],
    "total_scan_vat_area":       lambda cols: [c for c in cols if "vat" in c],
    "spine_l1_l4_bmd":           lambda cols: [c for c in cols if "spine" in c or "l1" in c or "l4" in c],
    "total_scan_sat_area":       lambda cols: [c for c in cols if "sat" in c],
}

# Sex-specific targets: only analyse subjects of the given gender value (0=female, 1=male)
_SEX_FILTER_MAP = {
    "dis__breast_cancer":               0.0,  # female only
    "dis__endometriosis_and_adenomyosis": 0.0,
    "dis__polycystic_ovary_disease":    0.0,
    "dis__perimenopausal_disorders":    0.0,
    "dis__erectile_dysfunction":        1.0,  # male only
}


# ── LOAD PRE-EXTRACTED EMBEDDINGS ─────────────────────────────────────────────
BONE_ONLY   = False   # --bone-only: use only the bone-view embedding (skip tissue)
BMD_ONLY    = False   # --bmd-only: restrict the tabular arm to *_bmd columns
PER_BLOCK   = False   # --per-block: treat bone / tissue as separate penalized blocks
                      #              (per-block scale sweep; see _fit_predict_blocks).


def load_embeddings(embeddings_dir: str, models: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """Returns {model_name: {"bone": df, "tissue": df}} for each requested SSL model.
    Drops the Date level so the index is (RegistrationCode, research_stage)."""
    embs = {}
    for model in models:
        if model in _SPECIAL_MODELS:
            continue
        bone_path   = os.path.join(embeddings_dir, f"{model}_bone.pkl")
        tissue_path = os.path.join(embeddings_dir, f"{model}_tissue.pkl")
        if not os.path.exists(bone_path) or not os.path.exists(tissue_path):
            raise FileNotFoundError(
                f"Embeddings not found for '{model}' in {embeddings_dir}.\n"
                f"Run: python -m model.extract_embeddings --models {model}"
            )
        bone_df   = pd.read_pickle(bone_path)
        tissue_df = pd.read_pickle(tissue_path)

        if bone_df.index.nlevels == 3:
            bone_df   = bone_df.reset_index(level="Date", drop=True)
            tissue_df = tissue_df.reset_index(level="Date", drop=True)

        embs[model] = {"bone": bone_df, "tissue": tissue_df}
        n, d = bone_df.shape
        print(f"  [{model}] bone: {n} × {d}  |  tissue: {tissue_df.shape[0]} × {tissue_df.shape[1]}")
    return embs


def _impute(train_mat: np.ndarray, val_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median imputation using train statistics."""
    medians = np.nanmedian(train_mat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    train_out = np.where(np.isnan(train_mat), medians, train_mat)
    val_out   = np.where(np.isnan(val_mat),   medians, val_mat)
    return train_out, val_out


# ── RIDGE FIT + EVAL ──────────────────────────────────────────────────────────
def _fit_predict(X_tr, X_val, y_tr, y_val, is_cls: bool, seed: int = 0):
    """Fit Ridge/Logistic on train, return predictions on val."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    if is_cls:
        if _RAND_N_ITER:
            clf = RandomizedSearchCV(
                LogisticRegression(solver="lbfgs", max_iter=1000),
                {"C": loguniform(1e-3, 1e3)},
                n_iter=_RAND_N_ITER, cv=StratifiedKFold(2),
                scoring="roc_auc", random_state=seed, n_jobs=1,
            )
            clf.fit(X_tr, y_tr)
            return clf.best_estimator_.decision_function(X_val), None
        clf = LogisticRegressionCV(Cs=np.logspace(-3, 3, _N_CS), cv=U.RIDGE_CV_FOLDS,
                                   solver="lbfgs", max_iter=1000, n_jobs=1)
        clf.fit(X_tr, y_tr)
        return clf.decision_function(X_val), None
    else:
        if _RAND_N_ITER:
            reg = RandomizedSearchCV(
                Ridge(),
                {"alpha": loguniform(0.1, 1000)},
                n_iter=_RAND_N_ITER, cv=KFold(2),
                scoring="r2", random_state=seed, n_jobs=1,
            )
            reg.fit(X_tr, y_tr)
            return reg.best_estimator_.predict(X_val), reg.best_params_["alpha"]
        reg = RidgeCV(alphas=U.RIDGE_ALPHAS, cv=U.RIDGE_CV_FOLDS)
        reg.fit(X_tr, y_tr)
        return reg.predict(X_val), reg.alpha_


def _fit_predict_blocks(X_tr, X_val, y_tr, y_val, is_cls: bool, seed: int = 0,
                        block_sizes=None, fixed_block_scales=None):
    """Early fusion with TUNED per-block penalization across ≥2 column blocks, selected
    by COORDINATE DESCENT over discrete scale factors (linear in #blocks, not Cartesian).

    `block_sizes` gives each block's column count in order (e.g. bone | tissue | cov).
    The first block is the reference (scale 1); every later block is scaled by a
    factor s ∈ {1,4,16,64} (standardized columns ×s ⇒ ridge penalty 1/s² as hard, so s>1
    trusts that block more). Starting from all-1, each free block's scale is greedily set
    to its inner-CV-score argmax in turn (one pass). One block ⇒ a plain single fit.
    Generalizes the Cox-style per-block penalizer (compare_lp_cov._fit_base).

    `fixed_block_scales` skips the scale sweep and applies explicit post-standardization
    block scales. This is used for sensitivity analyses such as approximately
    unpenalized covariates."""
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    sizes = [b for b in (block_sizes or [X_tr.shape[1]]) if b > 0]
    bounds, start = [], 0
    for b in sizes:
        bounds.append((start, start + b)); start += b
    multi = len(bounds) >= 2
    free  = list(range(1, len(bounds)))                       # blocks after the reference
    grid  = (1.0, 4.0, 16.0, 64.0)

    def _fit(scales):
        if multi and any(s != 1.0 for s in scales):
            Xtr_s, Xval_s = X_tr.copy(), X_val.copy()
            for (lo, hi), s in zip(bounds, scales):
                if s != 1.0:
                    Xtr_s[:, lo:hi] *= s; Xval_s[:, lo:hi] *= s
        else:
            Xtr_s, Xval_s = X_tr, X_val
        if is_cls:
            m = LogisticRegressionCV(Cs=np.logspace(-3, 3, _N_CS), cv=U.RIDGE_CV_FOLDS,
                                     solver="lbfgs", max_iter=1000, tol=1e-3, n_jobs=1,
                                     scoring=("roc_auc" if multi else None))
            m.fit(Xtr_s, y_tr)
            cv = float(np.mean(next(iter(m.scores_.values())), axis=0).max()) if multi else 0.0
            return cv, m.decision_function(Xval_s), None
        m = RidgeCV(alphas=U.RIDGE_ALPHAS, cv=U.RIDGE_CV_FOLDS)
        m.fit(Xtr_s, y_tr)
        cv = float(getattr(m, "best_score_", 0.0)) if multi else 0.0
        return cv, m.predict(Xval_s), float(m.alpha_)

    scales = [1.0] * len(bounds)
    if fixed_block_scales is not None:
        if len(fixed_block_scales) != len(bounds):
            raise ValueError(
                f"fixed_block_scales has {len(fixed_block_scales)} values, "
                f"but block_sizes defines {len(bounds)} blocks"
            )
        _, best_pred, best_alpha = _fit([float(s) for s in fixed_block_scales])
        return best_pred, best_alpha

    best_cv, best_pred, best_alpha = _fit(scales)
    if multi:
        for bi in free:                                       # greedy coordinate descent
            for g in grid:
                if g == scales[bi]:
                    continue
                trial = scales.copy(); trial[bi] = g
                cv, pred, alpha = _fit(trial)
                if cv > best_cv:
                    best_cv, best_pred, best_alpha = cv, pred, alpha
                    scales[bi] = g
    return best_pred, best_alpha


def _append_result(results, raw_rows, model_key, score, target_col, metric_name, seed):
    cfg = results.setdefault(f"{model_key}_ridge", {})
    cfg.setdefault("fusion", []).append(float(score))
    raw_rows.append({
        "target": target_col, "metric": metric_name,
        "model": model_key, "mode": "ridge", "fusion": "early",
        "view": "fusion", "seed": seed, "score": float(score),
    })


# ── SINGLE TARGET ─────────────────────────────────────────────────────────────
def run_target(
    target_col: str,
    target_df_full: pd.DataFrame,
    embs: dict,
    models: list[str],
    seeds: list[int],
    tabular_df: "pd.DataFrame | None" = None,
    min_prevalence: float = 0.0,
) -> tuple[list, list]:

    is_cls = target_col.lower() in U.CLASSIFICATION_TARGETS
    metric_name = "auc" if is_cls else "pearson"

    target_df = target_df_full[[target_col]].dropna().copy()

    # Sex-specific filtering: restrict to subjects of the required sex
    if target_col in _SEX_FILTER_MAP and "gender" in target_df_full.columns:
        required_sex = _SEX_FILTER_MAP[target_col]
        sex_idx = target_df_full.index[target_df_full["gender"] == required_sex]
        target_df = target_df[target_df.index.isin(sex_idx)]
        print(f"  [sex filter] gender={int(required_sex)} ({'F' if required_sex == 0 else 'M'}) "
              f"→ {len(target_df)} rows")

    if len(target_df) < U.MIN_SAMPLES_PER_TARGET:
        print(f"[Skip] {target_col}: only {len(target_df)} labeled samples")
        return [], []

    if is_cls:
        uniq = np.sort(target_df[target_col].unique())
        if len(uniq) != 2:
            print(f"[Skip] {target_col}: not binary ({len(uniq)} classes)")
            return [], []
        target_df[target_col] = target_df[target_col].map({uniq[0]: 0.0, uniq[1]: 1.0}).astype(float)
        n_positive = int((target_df[target_col] == 1.0).sum())
        if n_positive < U.MIN_POSITIVE_CASES:
            print(f"[Skip] {target_col}: only {n_positive} positive cases (< {U.MIN_POSITIVE_CASES})")
            return [], []
    else:
        std = target_df[target_col].std()
        if std == 0 or pd.isna(std):
            print(f"[Skip] {target_col}: zero variance")
            return [], []
        # Do NOT standardize y here — standardize per split inside the seed loop
        # using training-set statistics to avoid leakage.

    # Determine active covariate columns (exclude target itself)
    avail_cov = [c for c in _COV_COLS if c in target_df_full.columns and c != target_col]

    # Determine which SSL models are needed (also lejepa if ensemble requested)
    ssl_models_needed = [m for m in models if m in _SSL_MODELS]
    if "ensemble" in models and "lejepa" not in ssl_models_needed:
        ssl_models_needed.append("lejepa")

    # Build subject → index mapping from labeled rows
    labeled_subjects: dict[str, list] = {}
    for idx in target_df.index:
        reg, stage = idx
        labeled_subjects.setdefault(reg, []).append(idx)

    # Filter subjects by SSL embedding presence
    for model in ssl_models_needed:
        usable_idx = set(embs[model]["bone"].index)
        labeled_subjects = {
            s: rows for s, rows in labeled_subjects.items()
            if any(r in usable_idx for r in rows)
        }

    valid_subjects = sorted(labeled_subjects)

    # When tabular is included, restrict to subjects that also pass tabular QC so all
    # models are evaluated on the same cohort.
    if "tabular" in models and tabular_df is not None:
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
        print(f"  [tabular QC] {len(valid_subjects)} subjects with valid tabular data")

    if len(valid_subjects) < 2:
        print(f"[Skip] {target_col}: insufficient subjects ({len(valid_subjects)})")
        return [], []

    # Min-cases check on the intersected cohort (not raw targets CSV)
    if is_cls:
        all_valid_idx = [r for s in valid_subjects for r in labeled_subjects[s]]
        n_pos_intersected = int((target_df.loc[all_valid_idx, target_col] == 1.0).sum())
        if n_pos_intersected < U.MIN_POSITIVE_CASES:
            print(f"[Skip] {target_col}: only {n_pos_intersected} positive cases in intersected cohort "
                  f"(< {U.MIN_POSITIVE_CASES})")
            return [], []

    print(f"\n>>> {target_col} | {len(valid_subjects)} subjects | {metric_name}")

    results: dict[str, dict[str, list[float]]] = {}
    raw_rows = []

    for seed in seeds:
        train_subs, val_subs = train_test_split(valid_subjects, test_size=0.2, random_state=seed)
        train_idx = [r for s in train_subs for r in labeled_subjects[s]]
        val_idx   = [r for s in val_subs   for r in labeled_subjects[s]]

        # Leakage guard
        train_regs = {r[0] for r in train_idx}
        val_regs   = {r[0] for r in val_idx}
        if train_regs & val_regs:
            raise RuntimeError(f"Subject leakage in {target_col} seed={seed}")

        train_idx = [r for r in train_idx if r in target_df.index]
        val_idx   = [r for r in val_idx   if r in target_df.index]

        y_tr  = target_df.loc[train_idx, target_col].values.astype(float)
        y_val = target_df.loc[val_idx,   target_col].values.astype(float)

        if not is_cls:
            _mu, _sd = y_tr.mean(), y_tr.std()
            if _sd > 0:
                y_tr  = (y_tr  - _mu) / _sd
                y_val = (y_val - _mu) / _sd

        print(f"  seed={seed} | train={len(train_subs)} subs ({len(train_idx)} rows) "
              f"| val={len(val_subs)} subs ({len(val_idx)} rows)")

        # Track lejepa data for ensemble construction
        lejepa_tr: tuple | None = None
        lejepa_val: tuple | None = None

        # ── SSL models (lejepa, dino) ──────────────────────────────────────
        for model in [m for m in models if m in _SSL_MODELS]:
            bone_df   = embs[model]["bone"]
            tissue_df = embs[model]["tissue"]

            tr_in_emb  = [r for r in train_idx if r in bone_df.index]
            val_in_emb = [r for r in val_idx   if r in bone_df.index]

            bone_tr   = bone_df.loc[tr_in_emb].values
            bone_val  = bone_df.loc[val_in_emb].values
            tissue_tr = tissue_df.loc[tr_in_emb].values
            tissue_val= tissue_df.loc[val_in_emb].values

            y_tr_m  = target_df.loc[tr_in_emb,  target_col].values.astype(float)
            y_val_m = target_df.loc[val_in_emb, target_col].values.astype(float)

            # View blocks for (optional) per-block penalization: bone and tissue as
            # separately penalized blocks.
            blocks_tr  = [bone_tr]
            blocks_val = [bone_val]
            block_sizes = [bone_tr.shape[1]]
            if not BONE_ONLY:
                blocks_tr.append(tissue_tr);  blocks_val.append(tissue_val)
                block_sizes.append(tissue_tr.shape[1])
            early_tr  = np.concatenate(blocks_tr,  axis=1)
            early_val = np.concatenate(blocks_val, axis=1)

            if PER_BLOCK and len(block_sizes) >= 2:
                pred_f, alpha_f = _fit_predict_blocks(early_tr, early_val, y_tr_m, y_val_m,
                                                      is_cls, seed=seed, block_sizes=block_sizes)
            else:
                pred_f, alpha_f = _fit_predict(early_tr, early_val, y_tr_m, y_val_m, is_cls, seed=seed)
            score = U.metric(y_val_m, pred_f, is_cls)
            if alpha_f is not None:
                print(f"    [{model}] alpha={alpha_f:.2f}")
            print(f"    [{model}] fusion={score:.4f}")
            _append_result(results, raw_rows, model, score, target_col, metric_name, seed)

            if model == "lejepa":
                lejepa_tr  = (bone_tr,  tissue_tr,  tr_in_emb,  y_tr_m)
                lejepa_val = (bone_val, tissue_val, val_in_emb, y_val_m)

        # ── Covariates model ───────────────────────────────────────────────
        if "covariates" in models and len(avail_cov) > 0:
            cov_tr_raw  = target_df_full.loc[train_idx, avail_cov].values.astype(float)
            cov_val_raw = target_df_full.loc[val_idx,   avail_cov].values.astype(float)
            cov_tr, cov_val = _impute(cov_tr_raw, cov_val_raw)
            pred, alpha = _fit_predict(cov_tr, cov_val, y_tr, y_val, is_cls, seed=seed)
            score = U.metric(y_val, pred, is_cls)
            if alpha is not None:
                print(f"    [covariates] alpha={alpha:.2f}")
            print(f"    [covariates] fusion={score:.4f}")
            _append_result(results, raw_rows, "covariates", score, target_col, metric_name, seed)

        # ── Ensemble model (LeJEPA + covariates) ──────────────────────────
        if "ensemble" in models and len(avail_cov) > 0:
            # Load lejepa embeddings if not already done (ensemble without lejepa in models list)
            if lejepa_tr is None:
                bone_df   = embs["lejepa"]["bone"]
                tissue_df = embs["lejepa"]["tissue"]
                tr_in_emb  = [r for r in train_idx if r in bone_df.index]
                val_in_emb = [r for r in val_idx   if r in bone_df.index]
                bone_tr   = bone_df.loc[tr_in_emb].values
                bone_val  = bone_df.loc[val_in_emb].values
                tissue_tr = tissue_df.loc[tr_in_emb].values
                tissue_val= tissue_df.loc[val_in_emb].values
                y_tr_m  = target_df.loc[tr_in_emb,  target_col].values.astype(float)
                y_val_m = target_df.loc[val_in_emb, target_col].values.astype(float)
                lejepa_tr  = (bone_tr,  tissue_tr,  tr_in_emb,  y_tr_m)
                lejepa_val = (bone_val, tissue_val, val_in_emb, y_val_m)

            b_tr, t_tr, tr_emb, y_tr_ens  = lejepa_tr
            b_val, t_val, val_emb, y_val_ens = lejepa_val

            cov_tr_raw  = target_df_full.loc[tr_emb,  avail_cov].values.astype(float)
            cov_val_raw = target_df_full.loc[val_emb, avail_cov].values.astype(float)
            cov_tr_ens, cov_val_ens = _impute(cov_tr_raw, cov_val_raw)

            early_tr  = np.concatenate([b_tr,  t_tr,  cov_tr_ens],  axis=1)
            early_val = np.concatenate([b_val, t_val, cov_val_ens], axis=1)
            pred, alpha = _fit_predict(early_tr, early_val, y_tr_ens, y_val_ens, is_cls, seed=seed)
            score = U.metric(y_val_ens, pred, is_cls)
            if alpha is not None:
                print(f"    [ensemble] alpha={alpha:.2f}")
            print(f"    [ensemble] fusion={score:.4f}")
            _append_result(results, raw_rows, "ensemble", score, target_col, metric_name, seed)

        # ── Tabular DXA baseline ──────────────────────────────────────────────
        if "tabular" in models and tabular_df is not None:
            try:
                feat_cols = [c for c in tabular_df.columns
                             if pd.api.types.is_numeric_dtype(tabular_df[c])]
                if BMD_ONLY:
                    feat_cols = [c for c in feat_cols if c.endswith('_bmd')]
                if target_col in _TABULAR_LEAKAGE_EXCLUSIONS:
                    exclude = set(_TABULAR_LEAKAGE_EXCLUSIONS[target_col](feat_cols))
                    feat_cols = [c for c in feat_cols if c not in exclude]
                if not feat_cols:
                    print(f"    [tabular] no features after leakage exclusion, skipping")
                else:
                    # Subjects already intersected with tabular QC upfront — use train/val directly
                    tab_tr  = tabular_df.loc[train_idx, feat_cols].copy()
                    tab_val = tabular_df.loc[val_idx,   feat_cols].copy()
                    tab_tr[target_col]  = target_df.loc[train_idx, target_col]
                    tab_val[target_col] = target_df.loc[val_idx,   target_col]
                    tab_tr  = tab_tr.dropna(subset=[target_col])
                    tab_val = tab_val.dropna(subset=[target_col])
                    if len(tab_val) < 10:
                        print(f"    [tabular] only {len(tab_val)} val rows, skipping")
                    else:
                        X_tr_t  = tab_tr[feat_cols].values.astype(float)
                        y_tr_t  = tab_tr[target_col].values.astype(float)
                        X_val_t = tab_val[feat_cols].values.astype(float)
                        y_val_t = tab_val[target_col].values.astype(float)
                        X_tr_t, X_val_t = _impute(X_tr_t, X_val_t)
                        pred_t, alpha_t = _fit_predict(X_tr_t, X_val_t, y_tr_t, y_val_t, is_cls, seed=seed)
                        score_t = U.metric(y_val_t, pred_t, is_cls)
                        if alpha_t is not None:
                            print(f"    [tabular] alpha={alpha_t:.2f}")
                        print(f"    [tabular] score={score_t:.4f}")
                        # Use view="tabular" to match existing tabular_ridge_raw.csv format
                        results.setdefault("tabular_ridge", {}).setdefault("tabular", []).append(float(score_t))
                        raw_rows.append({
                            "target": target_col, "metric": metric_name,
                            "model": "tabular", "mode": "ridge", "fusion": "n/a",
                            "view": "tabular", "seed": seed, "score": float(score_t),
                        })
            except Exception as e_tab:
                print(f"    [tabular] seed={seed} error: {e_tab}")

    # Summary table
    print(f"\n{'═'*60}")
    print(f"  {target_col} — {len(seeds)}-seed summary (val {metric_name})")
    print(f"{'═'*60}")
    summary_rows = []
    for key, views_dict in sorted(results.items()):
        model_key = key.rsplit("_ridge", 1)[0]
        for view_name, scores in sorted(views_dict.items()):
            arr = np.array(scores, dtype=np.float32)
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            se  = float(std / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            print(f"  {key:<24} | {view_name:<6} {arr.mean():.4f} ± {arr.std():.4f}  (n={len(arr)})")
            summary_rows.append({
                "target": target_col, "metric": metric_name,
                "model": model_key, "mode": "ridge", "fusion": "early", "view": view_name,
                "n": int(len(arr)), "mean": float(arr.mean()), "std": std, "se": se,
            })
    print()
    return summary_rows, raw_rows


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ridge LP comparison on pre-extracted embeddings")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+",
                        default=["lejepa", "dino", "covariates", "ensemble"],
                        choices=["lejepa", "dino", "dino7b", "covariates", "ensemble", "tabular"])
    parser.add_argument("--tabular-csv", default=None,
                        help="Path to DXA tabular CSV (required when 'tabular' is in --models)")
    parser.add_argument("--num-seeds", type=int, default=10,
                        help=f"How many seeds to use (picks first N from fixed pool, max {len(U._SEED_POOL)})")
    parser.add_argument("--min-prevalence", type=float, default=0.02,
                        help="Minimum disease prevalence in intersected cohort (default 0.02 = 2%%)")
    parser.add_argument("--embeddings-dir", default=U.EMBEDDINGS_DIR)
    parser.add_argument("--results-csv",       default=_DEFAULT_SUMMARY)
    parser.add_argument("--results-raw-csv",   default=_DEFAULT_RAW)
    parser.add_argument("--results-ttest-csv", default=_DEFAULT_TTEST)
    parser.add_argument("--targets-csv", default=None,
                        help="Override TARGETS_CSV (e.g. csvs/disease_targets.csv)")
    parser.add_argument("--cls-auto-detect", action="store_true",
                        help="Auto-detect classification targets: columns with only {0,1} values")
    parser.add_argument("--first-scan-only", action="store_true",
                        help="Keep only the earliest scan per subject (one row per subject, "
                             "matching HPP disease_classfication.py design)")
    parser.add_argument("--pca", type=int, default=0,
                        help="PCA components per embedding view (bone and tissue independently). "
                             "0 = no PCA. Applied to SSL embeddings only, not tabular/covariates.")
    parser.add_argument("--rand-search", action="store_true",
                        help="Use RandomizedSearchCV instead of fixed alpha grid")
    parser.add_argument("--rand-n-iter", type=int, default=5,
                        help="Random search iterations (default 5)")
    parser.add_argument("--n-cs", type=int, default=5,
                        help="Number of C values in the logistic regression grid (default 5)")
    parser.add_argument("--no-age", action="store_true",
                        help="Exclude age from covariate columns")
    parser.add_argument("--bone-only", action="store_true",
                        help="Use only the bone-view embedding (skip tissue) for SSL arms")
    parser.add_argument("--bmd-only", action="store_true",
                        help="Restrict the tabular arm to *_bmd columns (clinical BMD readout)")
    parser.add_argument("--per-block", action="store_true",
                        help="Treat bone and tissue as separate penalized blocks "
                             "(per-block scale sweep).")
    args  = parser.parse_args()
    for path in (args.results_csv, args.results_raw_csv, args.results_ttest_csv):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    seeds = U.make_seeds(args.num_seeds)

    global _RAND_N_ITER, _N_CS, BONE_ONLY, BMD_ONLY, PER_BLOCK
    BONE_ONLY   = args.bone_only
    BMD_ONLY    = args.bmd_only
    PER_BLOCK   = args.per_block
    _N_CS = args.n_cs
    if args.rand_search:
        _RAND_N_ITER = args.rand_n_iter
    if args.no_age and "age" in _COV_COLS:
        _COV_COLS.remove("age")

    # Load embeddings only for SSL models (+ lejepa if ensemble requested)
    ssl_to_load = [m for m in args.models if m in _SSL_MODELS]
    if "ensemble" in args.models and "lejepa" not in ssl_to_load:
        ssl_to_load.append("lejepa")

    print(f"Loading embeddings from: {args.embeddings_dir}")
    embs = load_embeddings(args.embeddings_dir, ssl_to_load)

    if args.pca > 0:
        from sklearn.decomposition import PCA as _PCA
        for model, views in embs.items():
            for view_name, df in views.items():
                n_comp = min(args.pca, df.shape[1], df.shape[0])
                pca = _PCA(n_components=n_comp, random_state=0)
                proj = pca.fit_transform(df.values.astype(float))
                embs[model][view_name] = pd.DataFrame(proj, index=df.index,
                                                      columns=[f"{view_name}_pc{i}" for i in range(n_comp)])
                var = pca.explained_variance_ratio_.sum()
                print(f"  [{model}/{view_name}] PCA {df.shape[1]}→{n_comp}  var={var:.1%}")

    # Load DXA tabular features if tabular model requested
    tabular_df = None
    if "tabular" in args.models:
        if not args.tabular_csv:
            parser.error("--tabular-csv is required when the tabular arm is requested")
        tab_path = args.tabular_csv
        print(f"Loading tabular features from: {tab_path}")
        tabular_df = pd.read_csv(tab_path).set_index(["RegistrationCode", "research_stage"])
        tabular_df = tabular_df[~tabular_df.index.duplicated(keep="first")]
        tabular_df.sort_index(inplace=True)
        tabular_df = tabular_df.select_dtypes(include="number")
        print(f"  Tabular: {len(tabular_df)} rows × {tabular_df.shape[1]} features")

    targets_csv = args.targets_csv or U.TARGETS_CSV
    target_df_full = pd.read_csv(targets_csv, index_col=[0, 1])
    target_df_full.sort_index(inplace=True)

    if args.first_scan_only:
        # Keep only the earliest scan per subject — mirrors HPP one-scan-per-subject design
        visit_order = {"baseline": 0, "00_01_visit": 1, "01_01_visit": 2,
                       "02_00_visit": 3, "03_01_visit": 4, "04_00_visit": 5,
                       "04_01_visit": 6, "05_01_visit": 7, "06_00_visit": 8, "06_02_visit": 9}
        stages = target_df_full.index.get_level_values("research_stage")
        ranks  = pd.Series(stages, index=target_df_full.index).map(
                     lambda s: visit_order.get(s, 99))
        keep = ranks.groupby(level=0).transform("min") == ranks
        target_df_full = target_df_full.loc[keep]
        print(f"[first-scan-only] {len(target_df_full)} rows kept "
              f"({target_df_full.index.get_level_values(0).nunique()} subjects)")

    if args.cls_auto_detect:
        auto_cls = set()
        for c in target_df_full.columns:
            vals = set(target_df_full[c].dropna().unique())
            if vals.issubset({0.0, 1.0, 0, 1}):
                auto_cls.add(c)
        U.CLASSIFICATION_TARGETS = auto_cls
        print(f"Auto-detected {len(auto_cls)} classification targets")

    if args.targets:
        target_cols = args.targets
    elif U.TARGET_COLUMNS == "all":
        target_cols = [c for c in target_df_full.columns if pd.api.types.is_numeric_dtype(target_df_full[c])]
    else:
        target_cols = [c.strip() for c in U.TARGET_COLUMNS.split(",") if c.strip()]

    print(f"Targets ({len(target_cols)}): {target_cols}")
    print(f"Models: {args.models} | Num seeds: {args.num_seeds} → {seeds}")
    print(f"Ridge alphas: {U.RIDGE_ALPHAS} | CV folds: {U.RIDGE_CV_FOLDS}")
    print(f"Covariate cols: {[c for c in _COV_COLS if c in target_df_full.columns]}")

    all_summary_rows, all_raw_rows = [], []
    for target_col in target_cols:
        try:
            summary_rows, raw_rows = run_target(
                target_col, target_df_full, embs, args.models, seeds,
                tabular_df=tabular_df,
                min_prevalence=args.min_prevalence,
            )
            all_summary_rows.extend(summary_rows)
            all_raw_rows.extend(raw_rows)
        except Exception as e:
            import traceback
            print(f"[Error] {target_col}: {e}")
            traceback.print_exc()

    U.save_results(all_summary_rows, all_raw_rows,
                   args.results_csv, args.results_raw_csv, args.results_ttest_csv,
                   use_wilcoxon=True)


if __name__ == "__main__":
    main()
