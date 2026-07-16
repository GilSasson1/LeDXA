import os
import pandas as pd
import numpy as np
import warnings
import argparse
import collections
import hashlib
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats
from typing import Dict, List, Tuple, Optional
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA

try:
    from sksurv.metrics import concordance_index_ipcw
    _HAS_SKSURV = True
except ImportError:
    _HAS_SKSURV = False

# --- Configuration ---
DEFAULT_EVENTS_PATH = "/path/to/project/ukbb_osteo_data_expanded_aligned.csv"
DEFAULT_TABULAR_PATH = "/path/to/project/ukbb_tabular_data_for_cox_with_baseline.csv"
DEFAULT_LEJEPA_PATH = "/data/hpp_labdata/Analyses/gilsa/embeddings/ukbb_comparison/lejepa_fusion.pkl"
DEFAULT_DINO_PATH = "/data/hpp_labdata/Analyses/gilsa/embeddings/ukbb_comparison/dino_fusion.pkl"

def load_clean_data(filepath: str, name: str) -> pd.DataFrame:
    print(f"Loading {name}...")
    df = pd.read_csv(filepath, index_col=0, low_memory=False) if filepath.endswith('.csv') else pd.read_parquet(filepath)
    df = df[df.index.notna() & ~df.index.isin(['error', 'skipped'])]
    df.index = df.index.astype(int)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = set(df[col].dropna().astype(str).unique())
        if {'Female', 'Male'}.intersection(unique_vals) or {'F', 'M'}.intersection(unique_vals):
            df[col] = df[col].map({'Female': 0, 'Male': 1, 'F': 0, 'M': 1})
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(how='all')
    print(f"  -> {name} shape: {df.shape}")
    return df

def _pick_visit_matched(columns: List[str], baseline_visit: int) -> List[str]:
    visit_tag = f'visit {baseline_visit}'
    matched = [c for c in columns if visit_tag in c.lower()]
    return matched if matched else columns

def _pick_covariate_columns(tabular_df: pd.DataFrame, baseline_visit: int) -> List[str]:
    cols = tabular_df.columns.tolist()
    age_candidates = [c for c in cols if 'age when attended assessment centre' in c.lower() or ' age ' in f' {c.lower()} ']
    sex_candidates = [c for c in cols if 'sex' in c.lower() or 'gender' in c.lower()]
    bmi_candidates = [c for c in cols if 'body mass index (bmi)' in c.lower() or ' bmi' in c.lower()]

    picked = []
    if age_c := _pick_visit_matched(age_candidates, baseline_visit): picked.append(age_c[0])
    if sex_c := _pick_visit_matched(sex_candidates, baseline_visit): picked.append(sex_c[0])
    if bmi_c := _pick_visit_matched(bmi_candidates, baseline_visit): picked.append(bmi_c[0])

    if len(picked) < 3:
        raise ValueError(f"Could not identify all 3 required covariates (Age, Sex, BMI). Found: {picked}")
    return picked

def _pick_dxa_columns(tabular_df: pd.DataFrame, baseline_visit: int, exclude_cols: Optional[List[str]] = None) -> List[str]:
    exclude_cols = exclude_cols or []
    visit_tag = f'visit {baseline_visit}'
    dxa_keywords = [
        'dxa', 'bmd', 'bmc', 'bone',
        'fat mass', 'lean mass', 'fat percentage', 'tissue fat percentage',
        'total mass', 'fat-free mass', 'tissue mass',
        'android', 'gynoid', 'spine', 'femur', 'pelvis', 'rib', 'vat',
        'l1-l4 area', 'average width', 'average height'
    ]
    exclude_tokens = ['date', 'report', 'code', 'icd', 'cancer', 'death', 'origin', 'format', 'measuring method', 'measurement completed', 'images', 'believed safe', 'assessment centre', 'month', 'year', 'scanner', 'device', 'instance', 'aliquot']

    dxa_cols = [c for c in tabular_df.columns if c not in exclude_cols
                and visit_tag in c.lower()
                and pd.api.types.is_numeric_dtype(tabular_df[c])
                and not any(tok in c.lower() for tok in exclude_tokens)
                and any(k in c.lower() for k in dxa_keywords)]

    if not dxa_cols:
        dxa_cols = [c for c in tabular_df.columns if c not in exclude_cols
                    and pd.api.types.is_numeric_dtype(tabular_df[c])
                    and not any(tok in c.lower() for tok in exclude_tokens)
                    and any(k in c.lower() for k in dxa_keywords)]

    if not dxa_cols:
        raise ValueError('Could not identify DXA features in tabular data.')
    return dxa_cols

def split_tabular_features(tabular_df: pd.DataFrame, baseline_visit: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cov_cols = _pick_covariate_columns(tabular_df, baseline_visit)
    dxa_cols = _pick_dxa_columns(tabular_df, baseline_visit, exclude_cols=cov_cols)
    print(f"  -> Covariates ({len(cov_cols)}): {', '.join(cov_cols)}")
    print(f"  -> DXA features ({len(dxa_cols)})")
    return tabular_df[cov_cols].copy(), tabular_df[dxa_cols].copy()

def get_survival_target(df_events: pd.DataFrame, event_col: str, common_ids: pd.Index,
                        baseline_visit: int = 2, death_col: str = None,
                        admin_date=None) -> pd.DataFrame:
    """Build (time, event) survival targets anchored at the visit-2 DXA scan.

    admin_date: fixed administrative censoring date (the data-extraction cutoff).
        This MUST be a single study-level constant, not derived from the events.
        If None, falls back to the global max observed event date for this endpoint
        (approximate — circular and endpoint-specific; supply the real cutoff).
    """
    baseline_col = f'Date of attending assessment centre - visit {baseline_visit}'
    cols = [baseline_col, event_col]
    if death_col and death_col in df_events.columns and death_col != event_col:
        cols.append(death_col)
    df = df_events.loc[common_ids, cols].copy()
    start = pd.to_datetime(df[baseline_col], errors='coerce')
    end   = pd.to_datetime(df[event_col],    errors='coerce')

    # exclude prevalent cases (event recorded before or at baseline DXA scan)
    prevalent = end.notna() & (end <= start)
    df    = df.loc[~prevalent]
    start = start.loc[~prevalent]
    end   = end.loc[~prevalent]

    # Administrative censoring date — a single fixed study cutoff. Using the max
    # observed event date is circular and makes follow-up endpoint-specific, so it
    # is only a last-resort fallback.
    if admin_date is None:
        admin_date = end[end.notna() & (end > start)].max()
    admin_date = pd.Timestamp(admin_date)
    if pd.isna(admin_date):
        return pd.DataFrame(columns=['time', 'event'])

    # Events after the administrative cutoff were not observed at extraction time
    # -> treat as censored at the cutoff.
    has_event = end.notna() & (end > start) & (end <= admin_date)

    # Censor at death when death precedes the admin cutoff (cause-specific hazard:
    # death from another cause is a censoring event, not the outcome).
    if death_col and death_col in df.columns:
        death_date = pd.to_datetime(df[death_col], errors='coerce')
        censor_date = death_date.where(death_date.notna() & (death_date < admin_date),
                                       admin_date)
    else:
        censor_date = pd.Series(admin_date, index=df.index)

    duration = (end - start).dt.days.where(has_event, (censor_date - start).dt.days)
    res = pd.DataFrame({'time': duration, 'event': has_event.astype(int)}, index=df.index)
    return res[(res['time'] > 0) & (start.notna())]

def _extract_icd10_prefix(series: pd.Series) -> pd.Series:
    """Extract ICD10 3-character prefix (e.g., I21, W19) from free-text code cells."""
    if series is None:
        return pd.Series(dtype='object')
    return series.astype(str).str.upper().str.extract(r'([A-Z][0-9]{2})', expand=False)

def add_cause_specific_death_events(df_events: pd.DataFrame, death_col: str, cause_col: str) -> List[str]:
    """Create synthetic date columns for cause-specific mortality endpoints."""
    if death_col not in df_events.columns or cause_col not in df_events.columns:
        return []

    death_date = pd.to_datetime(df_events[death_col], errors='coerce')
    cause_prefix = _extract_icd10_prefix(df_events[cause_col])

    # Cause-specific mortality endpoints. Rare endpoints are still created;
    # they are later filtered by --min-events.
    groups = {
        'Date of death due to cancer (C00-C97) - visit 0': [f'C{i:02d}' for i in range(98)],
        'Date of death due to circulatory disease (I00-I99) - visit 0': [f'I{i:02d}' for i in range(100)],
        'Date of death due to respiratory disease (J00-J99) - visit 0': [f'J{i:02d}' for i in range(100)],
        'Date of death due to ischemic heart disease (I20-I25) - visit 0': [f'I{i:02d}' for i in range(20, 26)],
        'Date of death due to myocardial infarction (I21-I22) - visit 0': ['I21', 'I22'],
        'Date of death due to stroke (I60-I69) - visit 0': [f'I{i:02d}' for i in range(60, 70)],
        'Date of death due to COPD (J44) - visit 0': ['J44'],
        'Date of death due to pneumonia (J12-J18) - visit 0': [f'J{i:02d}' for i in range(12, 19)],
        'Date of death due to dementia (F01/F03/G30) - visit 0': ['F01', 'F03', 'G30'],
        'Date of death due to Parkinson disease (G20) - visit 0': ['G20'],
        'Date of death due to falls (W00-W19) - visit 0': [f'W{i:02d}' for i in range(20)],
        'Date of death due to fragility injury (W00-W19 or X59) - visit 0': [f'W{i:02d}' for i in range(20)] + ['X59'],
        'Date of death due to osteoporosis (M80-M82) - visit 0': ['M80', 'M81', 'M82'],
        'Date of death due to frailty (R54) - visit 0': ['R54'],
    }

    created = []
    for col_name, prefixes in groups.items():
        if col_name in df_events.columns:
            continue
        mask = cause_prefix.isin(prefixes)
        df_events[col_name] = death_date.where(mask)
        created.append(col_name)

    return created

# Fixed PCA dims. A variance-target (~95%) was tried but reverted: it did not
# change the C-index (tested 50/100/150 PCs -> flat) yet pushed DINO to ~298 PCs
# and ~3x'd the joint-Cox runtime. 100/30 is the validated cox_full_results config.
EMB_PCA_COMPONENTS = 100  # embedding blocks
DXA_PCA_COMPONENTS = 30   # DXA tabular block

def preprocess_block(X_train, X_test, apply_pca=False, n_components=EMB_PCA_COMPONENTS):
    valid_cols = ~np.isnan(X_train).all(axis=0)
    if not valid_cols.any(): return None, None
    X_train, X_test = X_train[:, valid_cols], X_test[:, valid_cols]

    imp = SimpleImputer(strategy='median')
    X_train, X_test = imp.fit_transform(X_train), imp.transform(X_test)

    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    if apply_pca and X_train.shape[1] > n_components:
        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
        pca = PCA(n_components=n_comp)
        X_train, X_test = pca.fit_transform(X_train), pca.transform(X_test)

    return X_train, X_test

def fit_base_cox(X_train, X_test, t_train, e_train, penalizer=0.1):
    if X_train is None or X_train.shape[1] == 0: return None, None
    df_train = pd.DataFrame(X_train, columns=[str(i) for i in range(X_train.shape[1])])
    df_train['T'], df_train['E'] = t_train, e_train
    df_test = pd.DataFrame(X_test, columns=[str(i) for i in range(X_test.shape[1])])

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df_train, 'T', event_col='E')
        return cph.predict_log_partial_hazard(df_train), cph.predict_log_partial_hazard(df_test)
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return None, None


def fit_joint_cox(blocks, t_train, e_train):
    """Single Cox on concatenated preprocessed blocks with a per-column penalizer array.

    This is the proper covariate-ADJUSTED model: covariates and feature blocks are
    fit together in one Cox, with *differential penalization* — each block carries
    its own penalty, so the few interpretable covariates (age/sex/BMI) are penalized
    lightly while the high-dimensional feature block is regularised harder. A single
    scalar penalizer would either over-shrink the covariates or under-shrink the
    features.

    blocks: list of (X_train_block, X_test_block, penalizer).
    Returns the test-set log-partial-hazard (pd.Series) or None on failure.
    """
    X_tr_parts, X_te_parts, pen = [], [], []
    for X_tr_b, X_te_b, p in blocks:
        if X_tr_b is None or X_te_b is None or X_tr_b.shape[1] == 0:
            return None
        X_tr_parts.append(X_tr_b)
        X_te_parts.append(X_te_b)
        pen.extend([float(p)] * X_tr_b.shape[1])

    X_tr = np.hstack(X_tr_parts)
    X_te = np.hstack(X_te_parts)
    df_tr = pd.DataFrame(X_tr, columns=[str(i) for i in range(X_tr.shape[1])])
    df_tr['T'], df_tr['E'] = t_train, e_train
    df_te = pd.DataFrame(X_te, columns=[str(i) for i in range(X_te.shape[1])])

    try:
        cph = CoxPHFitter(penalizer=np.asarray(pen, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cph.fit(df_tr, 'T', event_col='E')
        return cph.predict_log_partial_hazard(df_te)
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return None


def _uno_c(y_time, y_event, risk_score):
    """Uno's IPCW concordance index. risk_score: higher value = higher predicted risk.

    Reported alongside Harrell's C: Harrell's C is biased under heavy censoring
    (it depends on the study censoring distribution); Uno's IPCW estimator corrects
    for this. The gap between the two is itself a censoring-bias diagnostic.
    """
    if not _HAS_SKSURV:
        return np.nan
    try:
        surv = np.array(list(zip(np.asarray(y_event).astype(bool),
                                 np.asarray(y_time).astype(float))),
                        dtype=[('event', '?'), ('time', '<f8')])
        return float(concordance_index_ipcw(surv, surv, np.asarray(risk_score, dtype=float))[0])
    except Exception:
        return np.nan


def _quartile_metrics(y_time, y_event, oof_vec, horizon_days):
    """Top-quartile (Q4) flagging metrics from OOF risk scores.

    For a low-prevalence screening use-case the global C-index dilutes tail
    performance; these summarise how well the model concentrates events into
    the flagged top-25%.  `oof_vec` follows the model convention (higher value =
    lower risk), so risk = -oof_vec.

    - Capture@Q4: fraction of all observed events whose subject is in Q4
      (sensitivity of a top-25% flag; no time horizon needed).
    - PPV@Q4 / Inc@Q1: cumulative incidence (1-KM) at `horizon_days` in the top
      and bottom risk quartile (censoring-adjusted; NOT competing-risk adjusted).
    - FoldEnrich: PPV@Q4 / Inc@Q1 — prevalence-robust event-concentration lift.
    """
    nan_res = {'Capture@Q4': np.nan, 'FoldEnrich': np.nan,
               'PPV@Q4': np.nan, 'Inc@Q1': np.nan}
    valid = np.isfinite(oof_vec)
    if valid.sum() < 8 or y_event[valid].sum() < 4:
        return nan_res
    t = y_time[valid].astype(float)
    e = y_event[valid].astype(int)
    risk = -oof_vec[valid]  # higher = more risk
    edges = np.quantile(risk, [0.25, 0.5, 0.75])
    quart = np.digitize(risk, edges)  # 0=Q1 (low risk) .. 3=Q4 (high risk)

    n_events = int(e.sum())
    capture = e[quart == 3].sum() / n_events if n_events > 0 else np.nan

    def _ci_at(mask):
        if mask.sum() < 2 or e[mask].sum() < 1:
            return np.nan
        kmf = KaplanMeierFitter().fit(t[mask], e[mask])
        return float(1.0 - kmf.survival_function_at_times(horizon_days).iloc[0])

    ppv_q4 = _ci_at(quart == 3)
    inc_q1 = _ci_at(quart == 0)
    enrich = (ppv_q4 / inc_q1) if (inc_q1 is not None and np.isfinite(inc_q1)
                                   and inc_q1 > 0) else np.nan
    return {'Capture@Q4': capture, 'FoldEnrich': enrich,
            'PPV@Q4': ppv_q4, 'Inc@Q1': inc_q1}

def _inner_oof_base_preds(X_tr, t_train, e_train, penalizer, n_inner=5):
    """Inner K-fold OOF predictions for the meta-learner training inputs.

    Avoids using in-sample base-model predictions when fitting the meta-Cox,
    which would leak training signal and bias the meta-learner weights.
    Returns an OOF array of length len(t_train); NaN where prediction failed.
    """
    oof = np.full(len(t_train), np.nan, dtype=float)
    if X_tr is None or X_tr.shape[1] == 0:
        return oof
    skf = StratifiedKFold(n_splits=n_inner, shuffle=False)
    for itr, ival in skf.split(X_tr, e_train):
        X_itr = X_tr[itr]; X_ival = X_tr[ival]
        df_itr = pd.DataFrame(X_itr, columns=[str(i) for i in range(X_itr.shape[1])])
        df_itr['T'], df_itr['E'] = t_train[itr], e_train[itr]
        df_ival = pd.DataFrame(X_ival, columns=[str(i) for i in range(X_ival.shape[1])])
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(df_itr, 'T', event_col='E')
            oof[ival] = cph.predict_log_partial_hazard(df_ival).values
        except (ConvergenceError, np.linalg.LinAlgError, ValueError):
            pass
    return oof

def fit_meta_cox_with_oof_risk(train_preds: dict, test_preds: dict, t_train, e_train, t_test, e_test,
                               penalizer=0.01):
    df_train = pd.DataFrame(train_preds)
    df_test = pd.DataFrame(test_preds)

    if df_train.empty or df_test.empty or df_train.shape[1] < 2: return np.nan, None

    train_mask = np.isfinite(df_train.values).all(axis=1)
    test_mask = np.isfinite(df_test.values).all(axis=1)
    if train_mask.sum() < 3 or test_mask.sum() == 0: return np.nan, None

    df_train = df_train.loc[train_mask].copy()
    df_test_valid = df_test.loc[test_mask].copy()

    t_train, e_train = np.asarray(t_train), np.asarray(e_train)
    t_test, e_test = np.asarray(t_test), np.asarray(e_test)

    df_train['T'], df_train['E'] = t_train[train_mask], e_train[train_mask]

    try:
        meta_cph = CoxPHFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta_cph.fit(df_train, 'T', event_col='E')

        fold_risk_valid = -np.asarray(meta_cph.predict_partial_hazard(df_test_valid), dtype=float).reshape(-1)
        cidx = concordance_index(t_test[test_mask], fold_risk_valid, e_test[test_mask])

        fold_risk_full = np.full(df_test.shape[0], np.nan, dtype=float)
        fold_risk_full[test_mask] = fold_risk_valid
        return float(cidx), fold_risk_full
    except (ConvergenceError, np.linalg.LinAlgError, ValueError):
        return np.nan, None

def _process_fold_eval(X_cov, X_dxa, emb_blocks, t_train, t_test, e_train, e_test, te_idx,
                       dxa_name, use_pca, do_ensemble, pens, region_blocks=None,
                       lean_primary=False, use_dxa_pca=True):
    """
    emb_blocks: list of (emb_name, (X_tr_raw, X_te_raw))
    pens: dict with keys 'cov', 'dxa', 'emb', 'meta' — per-block Cox penalizers.
    """
    X_tr_cov, X_te_cov = preprocess_block(X_cov[0], X_cov[1], apply_pca=False)
    X_tr_dxa, X_te_dxa = preprocess_block(X_dxa[0], X_dxa[1], apply_pca=use_dxa_pca,
                                          n_components=DXA_PCA_COMPONENTS)

    tr_pred_cov, te_pred_cov = fit_base_cox(X_tr_cov, X_te_cov, t_train, e_train, penalizer=pens['cov'])
    tr_pred_dxa, te_pred_dxa = fit_base_cox(X_tr_dxa, X_te_dxa, t_train, e_train, penalizer=pens['dxa'])

    res = {'__test_idx__': np.asarray(te_idx, dtype=int).tolist(), '__oof__': {}}

    def _store(key, te_pred):
        # Negate so higher score = longer survival, as required by concordance_index().
        if te_pred is not None:
            res['__oof__'][key] = (-np.exp(np.asarray(te_pred, dtype=float).reshape(-1))).tolist()

    # ── Standalone (unadjusted) reference arms ──
    _store('Covariates', te_pred_cov)
    if not lean_primary:
        _store(dxa_name, te_pred_dxa)

    # Fit each embedding block independently (standalone arms)
    emb_preds = {}  # name -> (X_tr_emb, X_te_emb, te_pred)
    for emb_name, (X_tr_raw, X_te_raw) in emb_blocks:
        emb_pen = pens.get('emb_by_name', {}).get(emb_name, pens['emb'])
        X_tr_emb, X_te_emb = preprocess_block(X_tr_raw, X_te_raw, apply_pca=use_pca)
        _, te_pred = fit_base_cox(X_tr_emb, X_te_emb, t_train, e_train, penalizer=emb_pen)
        emb_preds[emb_name] = (X_tr_emb, X_te_emb, te_pred)
        if not lean_primary:
            _store(emb_name, te_pred)

    # ── PRIMARY: joint covariate-adjusted Cox (differential penalization) ──
    # One Cox on [covariates ⊕ features], covariates lightly penalized. This is
    # the model for the claim "features add predictive value beyond age/sex/BMI".
    if X_tr_cov is not None and X_tr_dxa is not None:
        _store(f'{dxa_name} + Covariates',
               fit_joint_cox([(X_tr_cov, X_te_cov, pens['cov']),
                              (X_tr_dxa, X_te_dxa, pens['dxa'])], t_train, e_train))

    for emb_name, (X_tr_emb, X_te_emb, _tep) in emb_preds.items():
        emb_pen = pens.get('emb_by_name', {}).get(emb_name, pens['emb'])
        if X_tr_cov is not None and X_tr_emb is not None:
            _store(f'{emb_name} + Covariates',
                   fit_joint_cox([(X_tr_cov, X_te_cov, pens['cov']),
                                  (X_tr_emb, X_te_emb, emb_pen)], t_train, e_train))
            # + regional (femur+lumbar) block, separately penalised
            if (not lean_primary) and region_blocks and emb_name in region_blocks:
                Xr_tr, Xr_te = preprocess_block(region_blocks[emb_name][0],
                                                region_blocks[emb_name][1], apply_pca=use_pca)
                _store(f'{emb_name} + Regional + Covariates',
                       fit_joint_cox([(X_tr_cov, X_te_cov, pens['cov']),
                                      (X_tr_emb, X_te_emb, emb_pen),
                                      (Xr_tr, Xr_te, pens['regionpool'])], t_train, e_train))
        if (not lean_primary) and X_tr_cov is not None and X_tr_dxa is not None and X_tr_emb is not None:
            _store(f'{emb_name} + {dxa_name} + Covariates',
                   fit_joint_cox([(X_tr_cov, X_te_cov, pens['cov']),
                                  (X_tr_dxa, X_te_dxa, pens['dxa']),
                                  (X_tr_emb, X_te_emb, emb_pen)], t_train, e_train))

    # All embeddings together + Covariates (joint)
    if (not lean_primary) and len(emb_preds) > 1 and X_tr_cov is not None:
        blocks = [(X_tr_cov, X_te_cov, pens['cov'])]
        ok = True
        for en, (X_tr_emb, X_te_emb, _tep) in emb_preds.items():
            if X_tr_emb is None:
                ok = False
                break
            blocks.append((X_tr_emb, X_te_emb, pens.get('emb_by_name', {}).get(en, pens['emb'])))
        if ok:
            _store(' + '.join(emb_preds.keys()) + ' + Covariates',
                   fit_joint_cox(blocks, t_train, e_train))

    # ── SECONDARY: stacked meta-Cox ensemble (kept for comparison only) ──
    if do_ensemble:
        # Inner-fold OOF predictions for meta-learner training (avoids in-sample leakage)
        oof_cov = _inner_oof_base_preds(X_tr_cov, t_train, e_train, penalizer=pens['cov'])
        oof_dxa = _inner_oof_base_preds(X_tr_dxa, t_train, e_train, penalizer=pens['dxa'])

        if te_pred_cov is not None and te_pred_dxa is not None:
            _, meta_risk = fit_meta_cox_with_oof_risk(
                {'cov': oof_cov, 'dxa': oof_dxa}, {'cov': te_pred_cov, 'dxa': te_pred_dxa},
                t_train, e_train, t_test, e_test, penalizer=pens['meta'])
            if meta_risk is not None:
                res['__oof__'][f'{dxa_name} + Covariates (stacked)'] = meta_risk.tolist()

        oof_embs = {}  # cache inner OOF per embedding (avoid recomputing for combo arm)
        for emb_name, (X_tr_emb, X_te_emb, te_pred_emb) in emb_preds.items():
            oof_emb = _inner_oof_base_preds(X_tr_emb, t_train, e_train, penalizer=pens['emb'])
            oof_embs[emb_name] = oof_emb
            if te_pred_cov is not None and te_pred_emb is not None:
                _, meta_risk = fit_meta_cox_with_oof_risk(
                    {'cov': oof_cov, 'emb': oof_emb}, {'cov': te_pred_cov, 'emb': te_pred_emb},
                    t_train, e_train, t_test, e_test, penalizer=pens['meta'])
                if meta_risk is not None:
                    res['__oof__'][f'{emb_name} + Covariates (stacked)'] = meta_risk.tolist()

            if te_pred_cov is not None and te_pred_dxa is not None and te_pred_emb is not None:
                _, meta_risk = fit_meta_cox_with_oof_risk(
                    {'cov': oof_cov, 'dxa': oof_dxa, 'emb': oof_emb},
                    {'cov': te_pred_cov, 'dxa': te_pred_dxa, 'emb': te_pred_emb},
                    t_train, e_train, t_test, e_test, penalizer=pens['meta'])
                if meta_risk is not None:
                    res['__oof__'][f'{emb_name} + {dxa_name} + Covariates (stacked)'] = meta_risk.tolist()

        if len(emb_preds) > 1 and te_pred_cov is not None:
            all_valid = all(v[2] is not None for v in emb_preds.values())
            if all_valid:
                combined_tr = {'cov': oof_cov}
                combined_te = {'cov': te_pred_cov}
                for i, (en, (_, _, tep)) in enumerate(emb_preds.items()):
                    combined_tr[f'emb{i}'] = oof_embs[en]   # reuse cached inner OOF
                    combined_te[f'emb{i}'] = tep
                _, meta_risk = fit_meta_cox_with_oof_risk(combined_tr, combined_te,
                                                          t_train, e_train, t_test, e_test,
                                                          penalizer=pens['meta'])
                if meta_risk is not None:
                    combo_name = ' + '.join(emb_preds.keys()) + ' + Covariates (stacked)'
                    res['__oof__'][combo_name] = meta_risk.tolist()

    return res

def select_emb_penalizer(cov_df, emb_df, target_df, pens, candidates, random_seed=42,
                         use_pca=True, search_folds=5):
    """One-shot single-split search for the embedding-block Cox penalizer.

    Picks the penalizer maximising the validation C-index of the JOINT
    [covariates + embedding] Cox on a single stratified split. This
    directly targets the question "is the joint embedding arm being held back
    by over-shrinkage?". Cheap — ~len(candidates) Cox fits per endpoint, reused
    across all seeds — and deliberately NOT nested CV.

    Note: the split overlaps the main evaluation data, so the chosen penalizer
    carries mild optimism; acceptable for a coarse hyperparameter (audit-agreed
    tradeoff vs the ~25x cost of nested CV).
    """
    y_time  = target_df['time'].values
    y_event = target_df['event'].values
    X_cov = cov_df.loc[target_df.index].values.astype(float)
    X_emb = emb_df.loc[target_df.index].values.astype(float)

    skf = StratifiedKFold(n_splits=max(2, int(search_folds)),
                          shuffle=True, random_state=random_seed)
    tr_idx, va_idx = next(iter(skf.split(X_cov, y_event)))

    X_tr_cov, X_va_cov = preprocess_block(X_cov[tr_idx], X_cov[va_idx], apply_pca=False)
    X_tr_emb, X_va_emb = preprocess_block(X_emb[tr_idx], X_emb[va_idx], apply_pca=use_pca)
    if X_tr_cov is None or X_tr_emb is None:
        return pens['emb']

    t_tr, e_tr = y_time[tr_idx], y_event[tr_idx]
    t_va, e_va = y_time[va_idx], y_event[va_idx]

    best_p, best_c = pens['emb'], -np.inf
    for p in candidates:
        te_pred = fit_joint_cox([(X_tr_cov, X_va_cov, pens['cov']),
                                 (X_tr_emb, X_va_emb, p)], t_tr, e_tr)
        if te_pred is None:
            continue
        risk = -np.exp(np.asarray(te_pred, dtype=float).reshape(-1))
        try:
            c = concordance_index(t_va, risk, e_va)
        except Exception:
            continue
        if c > best_c:
            best_c, best_p = c, p
    return best_p


def select_cov_penalizer(cov_df, target_df, pens, candidates, random_seed=42, search_folds=5):
    """Per-endpoint penalizer search for the covariate-only block (age/sex/BMI).
    Fits a cov-only Cox on a single stratified split, picks penalizer maximising
    validation C-index. Run before emb/dxa searches so its result feeds into them."""
    y_time  = target_df['time'].values
    y_event = target_df['event'].values
    X_cov = cov_df.loc[target_df.index].values.astype(float)

    skf = StratifiedKFold(n_splits=max(2, int(search_folds)),
                          shuffle=True, random_state=random_seed)
    tr_idx, va_idx = next(iter(skf.split(X_cov, y_event)))
    X_tr_cov, X_va_cov = preprocess_block(X_cov[tr_idx], X_cov[va_idx], apply_pca=False)
    if X_tr_cov is None:
        return pens['cov']
    t_tr, e_tr = y_time[tr_idx], y_event[tr_idx]
    t_va, e_va = y_time[va_idx], y_event[va_idx]

    best_p, best_c = pens['cov'], -np.inf
    for p in candidates:
        _, te_pred = fit_base_cox(X_tr_cov, X_va_cov, t_tr, e_tr, penalizer=p)
        if te_pred is None:
            continue
        risk = -np.exp(np.asarray(te_pred, dtype=float).reshape(-1))
        try:
            c = concordance_index(t_va, risk, e_va)
        except Exception:
            continue
        if c > best_c:
            best_c, best_p = c, p
    return best_p


def select_dxa_penalizer(cov_df, dxa_df, target_df, pens, candidates, random_seed=42,
                         use_pca=True, search_folds=5, use_dxa_pca=True):
    """Symmetric counterpart to select_emb_penalizer for the DXA-tabular block, so
    the tabular baseline gets the SAME per-endpoint penalizer search as the embedding
    arm (otherwise the tuned-embedding vs fixed-tabular comparison is unfair). Single
    stratified split, joint [cov + dxa(PCA)] Cox, max validation C-index."""
    y_time  = target_df['time'].values
    y_event = target_df['event'].values
    X_cov = cov_df.loc[target_df.index].values.astype(float)
    X_dxa = dxa_df.loc[target_df.index].values.astype(float)

    skf = StratifiedKFold(n_splits=max(2, int(search_folds)),
                          shuffle=True, random_state=random_seed)
    tr_idx, va_idx = next(iter(skf.split(X_cov, y_event)))
    X_tr_cov, X_va_cov = preprocess_block(X_cov[tr_idx], X_cov[va_idx], apply_pca=False)
    X_tr_dxa, X_va_dxa = preprocess_block(X_dxa[tr_idx], X_dxa[va_idx], apply_pca=use_dxa_pca,
                                          n_components=DXA_PCA_COMPONENTS)
    if X_tr_cov is None or X_tr_dxa is None:
        return pens['dxa']
    t_tr, e_tr = y_time[tr_idx], y_event[tr_idx]
    t_va, e_va = y_time[va_idx], y_event[va_idx]

    best_p, best_c = pens['dxa'], -np.inf
    for p in candidates:
        te_pred = fit_joint_cox([(X_tr_cov, X_va_cov, pens['cov']),
                                 (X_tr_dxa, X_va_dxa, p)], t_tr, e_tr)
        if te_pred is None:
            continue
        risk = -np.exp(np.asarray(te_pred, dtype=float).reshape(-1))
        try:
            c = concordance_index(t_va, risk, e_va)
        except Exception:
            continue
        if c > best_c:
            best_c, best_p = c, p
    return best_p


def run_evaluation_single_seed(cov_df, dxa_df, emb_dfs, emb_names, dxa_name, target_df,
                                n_folds=5, use_pca=False, do_ensemble=False, random_seed=42,
                                pens=None, horizon_days=4 * 365.25, region_dfs=None,
                                lean_primary=False, include_uno=True, include_quartile=True,
                                holdout_test_size=None, use_dxa_pca=True):
    """K-fold stratified OOF evaluation, or one held-out split per seed.

    All events contribute to the pooled C-index, giving stable estimates even
    for rare events.  random_seed shuffles the data before folding so that
    running multiple seeds yields a fold-split stability estimate (NOT a
    population sampling-variance estimate — see audit M2).
    No parameter tuning is done inside folds (penalizers are fixed), so OOF is
    purely for evaluation.

    Returns {arm: {metric: value}} — metrics: 'C-Index' (Harrell), 'Uno C-Index',
    and the top-quartile flagging metrics from _quartile_metrics.
    """
    y_time  = target_df['time'].values
    y_event = target_df['event'].values
    n       = len(y_time)

    X_cov  = cov_df.loc[target_df.index].values.astype(float)
    X_dxa  = dxa_df.loc[target_df.index].values.astype(float)
    X_embs = [emb_df.loc[target_df.index].values.astype(float) for emb_df in emb_dfs]
    X_regs = ({name: rdf.reindex(target_df.index).values.astype(float)
               for name, rdf in region_dfs.items()} if region_dfs else {})

    if holdout_test_size is not None:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_test_size,
                                          random_state=random_seed)
        fold_iter = splitter.split(np.zeros(n), y_event)
    else:
        # Shuffle once with this seed, then stratify into folds
        rng          = np.random.default_rng(random_seed)
        shuffled_idx = rng.permutation(n)
        y_event_sh   = y_event[shuffled_idx]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=False)  # shuffle done above
        fold_iter = ((shuffled_idx[fold_tr], shuffled_idx[fold_te])
                     for fold_tr, fold_te in skf.split(shuffled_idx, y_event_sh))

    oof_preds = {}  # key -> length-n array, filled fold by fold

    for tr_idx, te_idx in fold_iter:
        if y_event[tr_idx].sum() < 2:
            continue

        fold_res = _process_fold_eval(
            (X_cov[tr_idx],  X_cov[te_idx]),
            (X_dxa[tr_idx],  X_dxa[te_idx]),
            [(name, (X_emb[tr_idx], X_emb[te_idx])) for name, X_emb in zip(emb_names, X_embs)],
            y_time[tr_idx], y_time[te_idx], y_event[tr_idx], y_event[te_idx],
            te_idx, dxa_name, use_pca, do_ensemble, pens,
            region_blocks={name: (Xr[tr_idx], Xr[te_idx]) for name, Xr in X_regs.items()},
            lean_primary=lean_primary, use_dxa_pca=use_dxa_pca,
        )

        te_idx_arr = np.asarray(fold_res.get('__test_idx__', []), dtype=int)
        for key, vals in fold_res.get('__oof__', {}).items():
            if key not in oof_preds:
                oof_preds[key] = np.full(n, np.nan, dtype=float)
            oof_preds[key][te_idx_arr] = np.asarray(vals, dtype=float)

    # Pooled OOF metrics per arm: concordance + top-quartile flagging metrics
    results = {}
    for key, oof_vec in oof_preds.items():
        valid = np.isfinite(oof_vec)
        m = {}
        if valid.sum() > 1 and y_event[valid].sum() > 0:
            # oof_vec: higher = lower hazard = longer survival
            m['C-Index'] = float(concordance_index(
                y_time[valid], oof_vec[valid], y_event[valid]))
            if include_uno:
                # Uno's C expects a risk score (higher = more risk) -> negate
                m['Uno C-Index'] = _uno_c(y_time[valid], y_event[valid], -oof_vec[valid])
        else:
            m['C-Index'] = np.nan
            if include_uno:
                m['Uno C-Index'] = np.nan
        if include_quartile:
            m.update(_quartile_metrics(y_time, y_event, oof_vec, horizon_days))
        results[key] = m

    return results

def _load_embeddings(path: str, name: str) -> pd.DataFrame:
    """Load an embedding file (.pkl or .csv), restrict to visit 2 if MultiIndex."""
    print(f"Loading {name} embeddings from {path}...")
    raw = pd.read_pickle(path) if path.endswith('.pkl') else pd.read_csv(path, index_col=0, low_memory=False)
    if isinstance(raw.index, pd.MultiIndex):
        visit_vals = raw.index.get_level_values(1).astype(str)
        raw = raw.loc[visit_vals == "2"].copy()
        numeric_0 = pd.to_numeric(raw.index.get_level_values(0), errors="coerce")
        raw = raw.loc[numeric_0.notna()].copy()
        raw.index = numeric_0[numeric_0.notna()].astype(int)
        print(f"  -> {name} restricted to visit 2. Shape: {raw.shape}")
    numeric = pd.to_numeric(raw.index, errors="coerce")
    raw = raw.loc[numeric.notna()].copy()
    raw.index = numeric[numeric.notna()].astype(int)
    print(f"  -> {name} shape: {raw.shape}")
    return raw

def main():
    parser = argparse.ArgumentParser(description='Seed-stability Cox comparison pipeline')
    parser.add_argument('--events-path', default=DEFAULT_EVENTS_PATH)
    parser.add_argument('--tabular-path', default=DEFAULT_TABULAR_PATH, help="Path to single CSV with Covariates and DXA features")
    parser.add_argument('--lejepa-path', default=DEFAULT_LEJEPA_PATH, help="Path to LeJEPA embeddings (.pkl or .csv)")
    parser.add_argument('--dino-path', default=DEFAULT_DINO_PATH, help="Path to DINO embeddings (.pkl or .csv); set to 'none' to skip")
    parser.add_argument('--lejepa-name', default='DXA SSL (LeJEPA)', help="Display name for LeJEPA embeddings")
    parser.add_argument('--dino-name', default='DXA SSL (DINO)', help="Display name for DINO embeddings")
    parser.add_argument('--event-keyword', default='all')
    parser.add_argument('--min-events', type=int, default=100,
                        help='Minimum incident events for an endpoint to be analysed. '
                             'Low-event endpoints give very noisy C-indices (audit M3).')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of evaluation folds per seed when --holdout-test-size is not set.')
    parser.add_argument('--n-seeds', type=int, default=5, help='Number of fold-shuffle seeds (fold-stability check)')
    parser.add_argument('--holdout-test-size', type=float, default=None,
                        help='If set, evaluate each seed as one stratified train/test split with this '
                             'test fraction instead of K-fold OOF. Example: 0.2 gives 10 independent '
                             'held-out 80/20 tests when used with --n-seeds 10.')
    parser.add_argument('--use-pca', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply PCA to the embedding blocks (default ON — raw '
                             'high-dimensional embeddings make Cox very slow). Use --no-pca to disable.')
    parser.add_argument('--use-dxa-pca', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply PCA to the DXA tabular block (default ON). Use --no-dxa-pca to '
                             'pass raw tabular features to Cox (117 features, L2 handles them fine).')
    parser.add_argument('--do-ensemble', action='store_true', help='Also run the secondary stacked meta-Cox arms')
    parser.add_argument('--lean-primary', action='store_true',
                        help='Run only Covariates, DXA Tabular + Covariates, and each embedding + '
                             'Covariates. Skip standalone feature arms, embedding+DXA combo arms, '
                             'regional arms, stacked ensembles, Uno C-index, and quartile/PPV metrics.')
    parser.add_argument('--admin-censor-date', default=None,
                        help='Administrative censoring date YYYY-MM-DD for first-occurrence (HES) endpoints '
                             '(the real UKBB data-extraction cutoff). If omitted, the global max event date '
                             'is used as an APPROXIMATE fallback — supply the real date for a correct analysis.')
    parser.add_argument('--death-censor-date', default=None,
                        help='Administrative censoring date YYYY-MM-DD for mortality endpoints '
                             '(death-registry cutoff). If omitted, the max observed death date is used.')
    parser.add_argument('--penalizer-cov', type=float, default=0.1,
                        help='Cox L2 penalizer for the covariate block. Default 0.1 — must be '
                             'small but NON-ZERO: an exact 0 in the mixed per-column penalizer '
                             'array degrades the joint Cox optimization and collapses the '
                             'embedding arm (~0.05 C-index loss on knee).')
    parser.add_argument('--penalizer-dxa', type=float, default=5.0,
                        help='Cox L2 penalizer for the (PCA-reduced) DXA tabular block')
    parser.add_argument('--penalizer-emb', type=float, default=0.3,
                        help='Cox L2 penalizer for the embedding PC block. Default 0.3 — the '
                             'value behind the validated cox_full_results run.')
    parser.add_argument('--penalizer-meta', type=float, default=0.01, help='Cox L2 penalizer for the stacked meta-Cox')
    parser.add_argument('--region-pool', action='store_true',
                        help='Add a femur+lumbar regional block to each embedding+covariate joint Cox arm.')
    parser.add_argument('--penalizer-regionpool', type=float, default=None,
                        help='Cox L2 penalizer for the regional block (default = --penalizer-emb).')
    parser.add_argument('--regionpool-lejepa-path',
                        default=DEFAULT_LEJEPA_PATH.replace('lejepa_fusion', 'lejepa_regionpool'))
    parser.add_argument('--regionpool-dino-path',
                        default=DEFAULT_DINO_PATH.replace('dino_fusion', 'dino_regionpool'))
    parser.add_argument('--sweep-cov-penalizer', action='store_true',
                        help='Per-endpoint penalizer search for the covariate block (age/sex/BMI) '
                             'using a cov-only Cox on a single stratified split. Uses the same '
                             'grid as --emb-penalizer-grid. Result feeds into the emb/dxa searches.')
    parser.add_argument('--sweep-emb-penalizer', action='store_true',
                        help='One-shot single-split search for the embedding-block penalizer, '
                             'per endpoint (cheap; not nested CV). Targets the joint covariate-'
                             'adjusted arm so the embedding block is not under-/over-shrunk.')
    parser.add_argument('--emb-penalizer-grid', default='0.1,0.3,1,3,10',
                        help='Comma-separated candidate embedding penalizers for --sweep-emb-penalizer.')
    parser.add_argument('--sweep-dxa-penalizer', action='store_true',
                        help='Symmetric per-endpoint penalizer search for the DXA-tabular block '
                             '(same grid/procedure as --sweep-emb-penalizer), so the tabular '
                             'baseline is tuned as fairly as the embedding arm.')
    parser.add_argument('--random-penalizer-search', type=int, default=0,
                        help='Use this many random log-uniform block-penalizer candidates per endpoint '
                             'for each feature block (DXA, LeJEPA, DINO). Implies symmetric DXA and '
                             'embedding penalizer search. Candidate values are sampled once per endpoint '
                             'from [--penalizer-random-low, --penalizer-random-high].')
    parser.add_argument('--penalizer-search-folds', type=int, default=5,
                        help='Number of stratified folds used to define the single parameter-selection '
                             'split. The first fold is used as validation; 2 means a 50/50 train/validation '
                             'split for choosing penalizers.')
    parser.add_argument('--penalizer-random-low', type=float, default=0.03)
    parser.add_argument('--penalizer-random-high', type=float, default=30.0)
    parser.add_argument('--quartile-horizon-years', type=float, default=4.0,
                        help='Time horizon (years) for the top-quartile PPV / fold-enrichment '
                             'metrics. Kept inside the well-observed follow-up range (median FU '
                             '~4 yr) — the curve tail is too sparse to read reliably.')
    parser.add_argument('--out-prefix', default='cox_ttest_results')
    args = parser.parse_args()

    if args.holdout_test_size is not None:
        if not (0 < args.holdout_test_size < 1):
            raise ValueError('--holdout-test-size must be between 0 and 1.')
    if args.penalizer_search_folds < 2:
        raise ValueError('--penalizer-search-folds must be at least 2.')

    pens = {'cov': args.penalizer_cov, 'dxa': args.penalizer_dxa,
            'emb': args.penalizer_emb, 'meta': args.penalizer_meta,
            'regionpool': (args.penalizer_regionpool if args.penalizer_regionpool is not None
                           else args.penalizer_emb)}
    emb_pen_grid = [float(x) for x in args.emb_penalizer_grid.split(',') if x.strip()]
    if args.random_penalizer_search > 0:
        args.sweep_emb_penalizer = True
        args.sweep_dxa_penalizer = True
    if not _HAS_SKSURV:
        print("  [WARNING] scikit-survival not installed — Uno's IPCW C-index will be NaN. "
              "Install with `uv pip install scikit-survival`.")

    events_df = pd.read_csv(args.events_path, low_memory=False, index_col='eid')
    events_df.index = events_df.index.astype(int)

    tabular_df = load_clean_data(args.tabular_path, "Tabular Data")
    cov_df, dxa_df = split_tabular_features(tabular_df)

    # Load embeddings
    emb_dfs = []
    emb_names = []

    lejepa_df = _load_embeddings(args.lejepa_path, args.lejepa_name)
    emb_dfs.append(lejepa_df)
    emb_names.append(args.lejepa_name)

    if args.dino_path.lower() != 'none':
        dino_df = _load_embeddings(args.dino_path, args.dino_name)
        emb_dfs.append(dino_df)
        emb_names.append(args.dino_name)

    # Optional regional (femur+lumbar) block, keyed by embedding name.
    region_dfs = {}
    if args.region_pool:
        rp_map = [(args.lejepa_name, args.regionpool_lejepa_path)]
        if args.dino_path.lower() != 'none':
            rp_map.append((args.dino_name, args.regionpool_dino_path))
        for nm, pth in rp_map:
            if os.path.exists(pth):
                region_dfs[nm] = _load_embeddings(pth, nm + " regional")
        print(f"  region-pool blocks: {list(region_dfs)}")

    dxa_name = "DXA Tabular"

    baseline_col = 'Date of attending assessment centre - visit 2'

    # Require complete covariate data (age, sex, BMI) — imputing covariates is invalid.
    cov_complete = cov_df.dropna()
    print(f"  -> Subjects with complete covariates: {len(cov_complete)} (dropped {len(cov_df) - len(cov_complete)} with any NaN covariate)")

    # Require >50% non-NaN DXA features.
    dxa_completeness = dxa_df.notna().mean(axis=1)
    dxa_sufficient = dxa_df[dxa_completeness > 0.5]
    print(f"  -> Subjects with >50% DXA features present: {len(dxa_sufficient)} (dropped {len(dxa_df) - len(dxa_sufficient)} below threshold)")

    common_ids = events_df[events_df[baseline_col].notna()].index
    common_ids = common_ids.intersection(cov_complete.index).intersection(dxa_sufficient.index)
    for emb_df in emb_dfs:
        common_ids = common_ids.intersection(emb_df.index)
    print(f"\nFinal Common Cohort: {len(common_ids)} (across tabular + {len(emb_dfs)} embedding source(s))")

    death_col = next((c for c in events_df.columns if 'date of death' in c.lower()), None)
    cause_col = next((c for c in events_df.columns if 'underlying (primary) cause of death: icd10' in c.lower()), None)
    if death_col:
        print(f"Competing-event censoring at death: '{death_col}'")

    if death_col and cause_col:
        created = add_cause_specific_death_events(events_df, death_col=death_col, cause_col=cause_col)
        if created:
            print(f"Added cause-specific mortality events ({len(created)}):")
            for c in created:
                n = pd.to_datetime(events_df[c], errors='coerce').notna().sum()
                print(f"  - {c}: {n} deaths")
    elif death_col and not cause_col:
        print("No ICD10 cause-of-death column found; skipping cause-specific mortality events.")

    excluded_event_tokens = ['date of attending assessment centre']
    event_cols = [c for c in events_df.columns if 'date' in c.lower() and c != baseline_col
                  and not any(tok in c.lower() for tok in excluded_event_tokens)
                  and (args.event_keyword == 'all' or args.event_keyword.lower() in c.lower())]

    # Always include all-cause mortality as an explicit event when a death date column exists.
    if death_col and death_col not in event_cols:
        event_cols.append(death_col)
        print(f"Added all-cause mortality event: '{death_col}'")

    # ── Administrative censoring dates (single fixed study cutoffs — audit C1) ──
    def _is_death_endpoint(col):
        return (death_col is not None and col == death_col) or ('death due to' in col.lower())

    if args.admin_censor_date:
        disease_admin = pd.Timestamp(args.admin_censor_date)
        print(f"HES first-occurrence censoring date: {disease_admin.date()}")
    else:
        disease_dates = [pd.to_datetime(events_df[c], errors='coerce').max()
                         for c in event_cols if not _is_death_endpoint(c)]
        disease_dates = [d for d in disease_dates if pd.notna(d)]
        disease_admin = max(disease_dates) if disease_dates else pd.NaT
        print(f"  [WARNING] --admin-censor-date not supplied; using the GLOBAL max event date "
              f"({disease_admin.date() if pd.notna(disease_admin) else 'NaT'}) as the HES censoring "
              f"cutoff for ALL endpoints. This is an approximation — supply the real UKBB "
              f"first-occurrence data-extraction date for a correct analysis.")

    if args.death_censor_date:
        death_admin = pd.Timestamp(args.death_censor_date)
        print(f"Death-registry censoring date: {death_admin.date()}")
    elif death_col:
        death_admin = pd.to_datetime(events_df[death_col], errors='coerce').max()
        print(f"  [WARNING] --death-censor-date not supplied; using max observed death date "
              f"({death_admin.date() if pd.notna(death_admin) else 'NaT'}).")
    else:
        death_admin = pd.NaT

    final_results = []
    perseed_rows = []   # long-format per-seed metrics (for seed dots / arbitrary Wilcoxon)
    base_seeds = [42 + i*100 for i in range(args.n_seeds)]
    horizon_days = args.quartile_horizon_years * 365.25

    # Sex-specific endpoint guardrail: restrict cohort to the relevant sex.
    # Values: 0 = Female, 1 = Male (UKBB Sex coding).
    _SEX_SPECIFIC_ENDPOINTS = {
        'breast':        0,   # female only
        'ovarian':       0,
        'cervical':      0,
        'endometriosis': 0,
        'prostate':      1,   # male only
        'testicular':    1,
    }
    _sex_col = next((c for c in cov_df.columns if 'sex' in c.lower() or 'gender' in c.lower()), None)

    for event_col in event_cols:
        # When mortality itself is the event, don't censor at death (self-referential).
        censor_col = None if (death_col and event_col == death_col) else death_col
        this_admin = death_admin if _is_death_endpoint(event_col) else disease_admin

        # Apply sex restriction for sex-specific endpoints
        endpoint_ids = common_ids
        if _sex_col is not None:
            col_lower = event_col.lower()
            for token, sex_val in _SEX_SPECIFIC_ENDPOINTS.items():
                if token in col_lower:
                    sex_ids = cov_df.index[cov_df[_sex_col] == sex_val]
                    endpoint_ids = common_ids.intersection(sex_ids)
                    print(f"  [sex-filter] {event_col}: restricted to sex={sex_val}, N={len(endpoint_ids)}")
                    break

        target_df = get_survival_target(events_df, event_col, endpoint_ids,
                                        death_col=censor_col, admin_date=this_admin)
        if target_df.empty or target_df['event'].sum() < args.min_events: continue

        # Per-endpoint embedding-penalizer selection (single-split, cheap)
        endpoint_pens = dict(pens)
        endpoint_pens['emb_by_name'] = {}
        if args.random_penalizer_search > 0:
            seed_bytes = hashlib.blake2b(event_col.encode('utf-8'), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(seed_bytes, 'little') % (2**32))
            random_grid = np.exp(rng.uniform(
                np.log(args.penalizer_random_low),
                np.log(args.penalizer_random_high),
                size=args.random_penalizer_search,
            ))
            emb_pen_grid_this = sorted({float(args.penalizer_emb), float(args.penalizer_dxa),
                                        *[float(x) for x in random_grid]})
        else:
            emb_pen_grid_this = emb_pen_grid
        if args.sweep_cov_penalizer:
            endpoint_pens['cov'] = select_cov_penalizer(cov_df, target_df, pens,
                                                         emb_pen_grid_this, random_seed=42,
                                                         search_folds=args.penalizer_search_folds)
        if args.sweep_emb_penalizer:
            for emb_name, emb_df in zip(emb_names, emb_dfs):
                best_p = select_emb_penalizer(cov_df, emb_df, target_df, endpoint_pens,
                                              emb_pen_grid_this, random_seed=42,
                                              use_pca=args.use_pca,
                                              search_folds=args.penalizer_search_folds)
                endpoint_pens['emb_by_name'][emb_name] = best_p
            endpoint_pens['emb'] = endpoint_pens['emb_by_name'].get(emb_names[0], pens['emb'])
            endpoint_pens['regionpool'] = endpoint_pens['emb']
        if args.sweep_dxa_penalizer:               # symmetric tabular-block search (same grid)
            endpoint_pens['dxa'] = select_dxa_penalizer(cov_df, dxa_df, target_df, endpoint_pens,
                                                         emb_pen_grid_this, random_seed=42,
                                                         use_pca=args.use_pca,
                                                         search_folds=args.penalizer_search_folds,
                                                         use_dxa_pca=args.use_dxa_pca)

        eval_desc = (f"held-out test_size={args.holdout_test_size}"
                     if args.holdout_test_size is not None
                     else f"{args.n_folds}-fold OOF")
        print(f"  Running Event: {event_col} (N={len(target_df)}, Events={target_df['event'].sum()}) "
              f"across {args.n_seeds} seeds ({eval_desc})  "
              f"[cov pen={endpoint_pens['cov']}, "
              f"emb pens={endpoint_pens.get('emb_by_name', {}) or endpoint_pens['emb']}, "
              f"dxa pen={endpoint_pens['dxa']}]")

        seed_results = collections.defaultdict(lambda: collections.defaultdict(list))
        for seed in base_seeds:
            scores = run_evaluation_single_seed(
                cov_df, dxa_df, emb_dfs, emb_names, dxa_name, target_df,
                n_folds=args.n_folds, use_pca=args.use_pca, do_ensemble=args.do_ensemble,
                random_seed=seed, pens=endpoint_pens, horizon_days=horizon_days,
                region_dfs=region_dfs, lean_primary=args.lean_primary,
                include_uno=not args.lean_primary, include_quartile=not args.lean_primary,
                holdout_test_size=args.holdout_test_size, use_dxa_pca=args.use_dxa_pca,
            )
            for arm, mdict in scores.items():
                for metric, val in mdict.items():
                    seed_results[arm][metric].append(val)

        if not seed_results: continue

        # Median follow-up — verify event rates are sensible after the C1 fix.
        med_fu = float(np.median(target_df['time'].values)) / 365.25
        row = {'Event': event_col, 'N': len(target_df), 'Total Events': int(target_df['event'].sum()),
               'Median Follow-up (yr)': round(med_fu, 2), 'Cov Penalizer': endpoint_pens['cov'],
               'Emb Penalizer': endpoint_pens['emb'], 'Dxa Penalizer': endpoint_pens['dxa']}
        for emb_name in emb_names:
            row[f'{emb_name} Penalizer'] = endpoint_pens.get('emb_by_name', {}).get(
                emb_name, endpoint_pens['emb'])
        if not args.lean_primary:
            row['PPV Horizon (yr)'] = args.quartile_horizon_years

        # Per-arm metric means + SE. SE is fold-shuffle stability, NOT a
        # population/sampling SE (audit M2). Metrics:
        #   C-Index, Uno C-Index   - global discrimination
        #   Capture@Q4             - % of events flagged by the top risk quartile
        #   PPV@Q4, Inc@Q1         - cumulative incidence at the horizon, top/bottom quartile
        #   FoldEnrich             - PPV@Q4 / Inc@Q1, prevalence-robust event concentration
        METRICS = ['C-Index'] if args.lean_primary else [
            'C-Index', 'Uno C-Index', 'Capture@Q4', 'FoldEnrich', 'PPV@Q4', 'Inc@Q1']
        for arm, md in seed_results.items():
            for metric in METRICS:
                vals = [v for v in md.get(metric, []) if not (v is None or np.isnan(v))]
                if vals:
                    row[f'{arm} {metric}'] = np.mean(vals)
                    row[f'{arm} {metric} SE'] = (np.std(vals, ddof=1) / np.sqrt(len(vals))
                                                 if len(vals) > 1 else 0.0)
                else:
                    row[f'{arm} {metric}'] = np.nan
                    row[f'{arm} {metric} SE'] = np.nan

        # Per-seed long-format dump — one row per (endpoint, arm, seed). Needed
        # for figures that show individual seeds as dots and for Wilcoxon tests
        # against an arbitrary reference arm (e.g. the per-endpoint best model).
        for arm, md in seed_results.items():
            for si, seed in enumerate(base_seeds):
                pr = {'Event': event_col, 'Arm': arm, 'Seed': seed}
                for metric in METRICS:
                    mv = md.get(metric, [])
                    pr[metric] = mv[si] if si < len(mv) else np.nan
                perseed_rows.append(pr)

        # One-tailed Wilcoxon signed-rank: H1 is model1 > model2, consistent with
        # HPP and UKBB disease classification comparisons.
        def _get_wilcoxon_p(model1, model2, metric='C-Index'):
            if model1 in seed_results and model2 in seed_results:
                v1 = seed_results[model1].get(metric, [])
                v2 = seed_results[model2].get(metric, [])
                valid_pairs = [(a, b) for a, b in zip(v1, v2)
                               if not (a is None or b is None or np.isnan(a) or np.isnan(b))]
                if len(valid_pairs) > 1:
                    a_vals, b_vals = zip(*valid_pairs)
                    if np.allclose(a_vals, b_vals): return 1.0
                    try:
                        _, p = stats.wilcoxon(a_vals, b_vals, alternative='two-sided')
                    except ValueError:
                        return 1.0
                    return p
            return np.nan

        # 1. Everything vs the Covariates baseline — on C-Index and on Capture@Q4
        for k in seed_results.keys():
            if k == 'Covariates': continue
            row[f'P-Value ({k} vs Covariates)'] = _get_wilcoxon_p(k, 'Covariates')
            if not args.lean_primary:
                row[f'P-Value Capture ({k} vs Covariates)'] = _get_wilcoxon_p(k, 'Covariates', 'Capture@Q4')

        if not args.lean_primary:
            # 2. Compare each embedding against DXA Tabular
            for emb_name in emb_names:
                row[f'P-Value ({emb_name} vs {dxa_name})'] = _get_wilcoxon_p(emb_name, dxa_name)

            # 3. Pairwise comparison between embeddings (if multiple)
            for i in range(len(emb_names)):
                for j in range(i + 1, len(emb_names)):
                    n1, n2 = emb_names[i], emb_names[j]
                    row[f'P-Value ({n1} vs {n2})'] = _get_wilcoxon_p(n1, n2)

        # 4. Joint covariate-adjusted cross-comparisons (C-Index and Capture@Q4)
        ens_dxa_key = f'{dxa_name} + Covariates'
        for emb_name in emb_names:
            ens_emb_key = f'{emb_name} + Covariates'
            row[f'P-Value ({ens_emb_key} vs {ens_dxa_key})'] = _get_wilcoxon_p(ens_emb_key, ens_dxa_key)
            if not args.lean_primary:
                row[f'P-Value Capture ({ens_emb_key} vs {ens_dxa_key})'] = \
                    _get_wilcoxon_p(ens_emb_key, ens_dxa_key, 'Capture@Q4')
                row[f'P-Value ({emb_name} vs {ens_dxa_key})'] = _get_wilcoxon_p(emb_name, ens_dxa_key)

        for i in range(len(emb_names)):
            for j in range(i + 1, len(emb_names)):
                ek1 = f'{emb_names[i]} + Covariates'
                ek2 = f'{emb_names[j]} + Covariates'
                row[f'P-Value ({ek1} vs {ek2})'] = _get_wilcoxon_p(ek1, ek2)

        final_results.append(row)

    if final_results:
        results_df = pd.DataFrame(final_results)
        # BH-FDR correction across endpoints, per comparison type
        pval_cols = [c for c in results_df.columns if c.startswith('P-Value')]
        for pc in pval_cols:
            raw = results_df[pc].values.astype(float)
            valid = ~np.isnan(raw)
            if valid.sum() > 1:
                _, padj, _, _ = multipletests(raw[valid], alpha=0.05, method='fdr_bh')
                adj = np.full(len(raw), np.nan)
                adj[valid] = padj
            else:
                adj = raw.copy()
            results_df[pc.replace('P-Value', 'P-Value-adj')] = adj
        results_df.to_csv(f"{args.out_prefix}.csv", index=False)
        print(f"\nSuccess! Results saved to {args.out_prefix}.csv")
    if perseed_rows:
        pd.DataFrame(perseed_rows).to_csv(f"{args.out_prefix}_perseed.csv", index=False)
        print(f"Per-seed metrics saved to {args.out_prefix}_perseed.csv")

if __name__ == '__main__':
    main()
