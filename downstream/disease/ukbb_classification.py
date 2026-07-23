"""Four-arm covariate-adjusted disease classification on UKBB visit-2 DXA.

Arms:
  Arm 1: Covariates only          (age / sex / BMI)
  Arm 2: DXA-FM + Covariates      (LeJEPA early fusion)
  Arm 3: DINOv3 + Covariates      (DINO early fusion)
  Arm 4: DXA Tabular + Covariates (tabular early fusion)

Protocol per disease × model × seed:
  - Cohort: 4-way intersection (labels ∩ LeJEPA ∩ DINO ∩ tabular) so all arms
    are evaluated on identical subjects and paired tests are valid
  - 80/20 subject-level stratified train/val split (shared across all arms)
  - Early fusion: concatenate feature block + covariates into one matrix
  - StandardScaler on train, applied to val; median imputation of NaNs
  - LogisticRegressionCV (C ∈ 5 log-spaced values, 2-fold inner CV)
  - Metric: ROC-AUC on val set
  - Minimum positives: 150

Statistics:
  - Two-tailed Wilcoxon signed-rank test across seeds (H1: test != ref, paired within seed)
  - Benjamini–Hochberg FDR correction across diseases per comparison

Outputs:
  ukbb_disease_4arm_seeds.csv   — per-seed AUC for all arms × diseases
  ukbb_disease_4arm_summary.csv — mean/SE + Wilcoxon p-values + FDR-adj p-values
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform
from lightgbm import LGBMClassifier
from config import DATA_ROOT, EMBEDDINGS_DIR, RESULTS_DIR

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

DISEASE_LABELS = str(DATA_ROOT / "ukbb" / "baseline_disease_targets.csv")
EMB_DIR = str(EMBEDDINGS_DIR / "ukbb")
TABULAR_CSV = str(DATA_ROOT / "ukbb" / "dxa_tabular.csv")

DEFAULT_OUT_SEEDS = str(RESULTS_DIR / "ukbb_disease_4arm_seeds.csv")
DEFAULT_OUT_SUMMARY = str(RESULTS_DIR / "ukbb_disease_4arm_summary.csv")
DEFAULT_N_SEEDS = 10
DEFAULT_PCA = 50
_SEED_POOL = [42, 73, 99, 123, 2024, 7, 17, 31, 137, 256,
              13, 21, 55, 89, 144, 233, 377, 610, 987, 1597]
MIN_POSITIVE = 100       # minimum positive cases
TRAIN_FRAC = 0.80
INNER_CV = 2
N_CS = 5

_DXA_KEYWORDS = [
    "dxa", "bmd", "bmc", "bone", "fat mass", "lean mass", "fat percentage",
    "tissue fat percentage", "total mass", "fat-free mass", "tissue mass",
    "android", "gynoid", "spine", "femur", "pelvis", "rib", "vat",
    "l1-l4", "average width", "average height",
]
_EXCLUDE_TOKENS = [
    "date", "report", "code", "icd", "cancer", "death", "origin",
    "format", "measuring method", "measurement completed", "images",
    "believed safe", "assessment centre", "month", "year", "scanner",
    "device", "instance", "aliquot",
]
_SEX_SPECIFIC_FEMALE = {"dis__pcos", "dis__endometriosis"}
_SEX_SPECIFIC_MALE: set = set()

MODELS = ["covariates", "lejepa_cov", "dino_cov", "tab_cov"]
MODEL_LABELS = {
    "covariates":  "Covariates (age/sex/BMI)",
    "lejepa_cov":  "DXA-FM + Covariates",
    "dino_cov":    "DINOv3 + Covariates",
    "tab_cov":     "DXA Tabular + Covariates",
}
COMPARISONS = [
    ("covariates",  "LeJEPAcov_vs_Cov",     "lejepa_cov"),
    ("covariates",  "DINOcov_vs_Cov",        "dino_cov"),
    ("covariates",  "Tabcov_vs_Cov",         "tab_cov"),
    ("tab_cov",     "LeJEPAcov_vs_Tabcov",   "lejepa_cov"),
    ("dino_cov",    "LeJEPAcov_vs_DINOcov",  "lejepa_cov"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = pvals[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = adj
    return np.minimum(out, 1.0)


def _impute_median(tr: np.ndarray, va: np.ndarray):
    medians = np.nanmedian(tr, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    tr_out = np.where(np.isnan(tr), medians, tr)
    va_out = np.where(np.isnan(va), medians, va)
    keep = tr_out.std(axis=0) > 0
    return tr_out[:, keep], va_out[:, keep]


def _fit_classify(X_tr, X_va, y_tr, y_va, rand_n_iter=None, seed=0,
                  model="logreg", n_cs=N_CS, inner_cv=INNER_CV,
                  c_lo=-3, c_hi=3, diff_pen=False, n_cov=0) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "lgbm":
            # Non-linear arm: gradient-boosted trees handle NaN natively, need no
            # scaling, and do NOT impose a single shared L2 penalty across the
            # covariate + imaging blocks — so this isolates whether the "drops"
            # come from linear mis-specification / uniform penalisation.
            search = RandomizedSearchCV(
                LGBMClassifier(class_weight="balanced", random_state=seed,
                               n_jobs=1, verbose=-1),
                {"n_estimators": [100, 300],
                 "learning_rate": loguniform(1e-2, 0.3),
                 "num_leaves": [15, 31],
                 "min_child_samples": [20, 50]},
                n_iter=(rand_n_iter or 12), cv=StratifiedKFold(max(inner_cv, 3)),
                scoring="roc_auc", random_state=seed, n_jobs=1,
            )
            search.fit(X_tr, y_tr)
            scores = search.best_estimator_.predict_proba(X_va)[:, 1]
        else:
            X_tr, X_va = _impute_median(X_tr, X_va)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
            cs = np.logspace(c_lo, c_hi, n_cs)
            has_cov_block = diff_pen and 0 < n_cov < X_tr.shape[1]
            if has_cov_block and not rand_n_iter:
                # Tuned per-block (differential) penalisation, mirroring the Cox
                # per-block penalizer: jointly CV-select the covariate-block looseness
                # s (covariates penalised 1/s² as hard as the imaging block) AND the
                # imaging penalty C. s=1 ≡ shared penalty; large s ≈ covariates free.
                # Avoids both the over-shrinkage of covariates (drops) and the
                # over-shrinkage of imaging (full-unpenalised hurt osteoporosis etc.).
                best_clf, best_s, best_score = None, 1.0, -np.inf
                for s in (1.0, 4.0, 16.0, 64.0):
                    Xs = X_tr.copy(); Xs[:, -n_cov:] *= s
                    c = LogisticRegressionCV(
                        Cs=cs, cv=inner_cv, penalty="l2", solver="lbfgs",
                        max_iter=2000, scoring="roc_auc", n_jobs=1,
                    )
                    c.fit(Xs, y_tr)
                    sc = float(np.mean(next(iter(c.scores_.values())), axis=0).max())
                    if sc > best_score:
                        best_score, best_s, best_clf = sc, s, c
                Xva = X_va.copy(); Xva[:, -n_cov:] *= best_s
                scores = best_clf.decision_function(Xva)
            elif rand_n_iter:
                if has_cov_block:   # fixed moderate looseness for the rand-search path
                    X_tr = X_tr.copy(); X_va = X_va.copy()
                    X_tr[:, -n_cov:] *= 16.0; X_va[:, -n_cov:] *= 16.0
                clf = RandomizedSearchCV(
                    LogisticRegression(solver="lbfgs", max_iter=2000, penalty="l2"),
                    {"C": loguniform(1e-3, 1e3)},
                    n_iter=rand_n_iter, cv=StratifiedKFold(inner_cv),
                    scoring="roc_auc", random_state=seed, n_jobs=1,
                )
                clf.fit(X_tr, y_tr)
                scores = clf.best_estimator_.decision_function(X_va)
            else:
                clf = LogisticRegressionCV(
                    Cs=cs, cv=inner_cv, penalty="l2", solver="lbfgs",
                    max_iter=2000, n_jobs=1,
                )
                clf.fit(X_tr, y_tr)
                scores = clf.decision_function(X_va)
    if scores.ndim > 1:
        scores = scores[:, 1]
    try:
        return {"auc": float(roc_auc_score(y_va, scores)),
                "pr":  float(average_precision_score(y_va, scores))}
    except ValueError:
        return {"auc": float("nan"), "pr": float("nan")}


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_embedding(pkl_path: str) -> pd.DataFrame:
    with open(pkl_path, "rb") as f:
        emb = pickle.load(f)
    if isinstance(emb.index, pd.MultiIndex):
        visit_vals = emb.index.get_level_values(1).astype(str)
        emb = emb.loc[visit_vals == "2"].copy()
        emb.index = pd.Index(pd.to_numeric(emb.index.get_level_values(0), errors="coerce"))
    else:
        emb.index = pd.to_numeric(emb.index, errors="coerce")
    emb = emb[emb.index.notna()].copy()
    emb.index = emb.index.astype(int)
    emb = emb.apply(pd.to_numeric, errors="coerce")
    return emb


def _load_tabular():
    print(f"  Loading tabular from {TABULAR_CSV} …")
    df = pd.read_csv(TABULAR_CSV, index_col=0, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    visit_tag = "visit 2"

    age_col = next((c for c in df.columns if "age when attended" in c.lower()), None)
    sex_col = next((c for c in df.columns if "sex" in c.lower() or "gender" in c.lower()), None)
    bmi_col = next((c for c in df.columns
                    if "body mass index (bmi)" in c.lower() and visit_tag in c.lower()), None)
    if not all([age_col, sex_col, bmi_col]):
        raise ValueError(f"Could not find age/sex/BMI. Found: age={age_col}, sex={sex_col}, bmi={bmi_col}")

    cov_cols = [age_col, sex_col, bmi_col]
    dxa_cols = [
        c for c in df.columns
        if c not in cov_cols
        and visit_tag in c.lower()
        and not any(tok in c.lower() for tok in _EXCLUDE_TOKENS)
        and any(k in c.lower() for k in _DXA_KEYWORDS)
    ]
    print(f"  Covariates ({len(cov_cols)}): {cov_cols}")
    print(f"  DXA features: {len(dxa_cols)}")

    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df[df.index.notna()].copy()
    df.index = df.index.astype(int)
    return df[cov_cols].copy(), df[dxa_cols].copy()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(n_seeds: int = DEFAULT_N_SEEDS,
         out_seeds: str = DEFAULT_OUT_SEEDS,
         out_summary: str = DEFAULT_OUT_SUMMARY,
         pca_components: int | None = DEFAULT_PCA,
         use_covariates: bool = True,
         rand_n_iter: "int | None" = None,
         model: str = "logreg",
         n_cs: int = N_CS,
         inner_cv: int = INNER_CV,
         c_lo: int = -3,
         c_hi: int = 3,
         diff_pen: bool = False,
         targets: "list | None" = None) -> None:
    for path in (out_seeds, out_summary):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    label = "covariate-adjusted" if use_covariates else "no-covariates"
    print(f"=== UKBB Disease Classification — 4-arm {label} ===\n")
    if n_seeds > len(_SEED_POOL):
        raise ValueError(f"n_seeds max is {len(_SEED_POOL)}, got {n_seeds}")
    seeds = _SEED_POOL[:n_seeds]
    print(f"Seeds ({n_seeds}): {seeds}\n")

    # ── Load labels ────────────────────────────────────────────────────────────
    print(f"Loading disease labels: {DISEASE_LABELS}")
    labels = pd.read_csv(DISEASE_LABELS, index_col=0)
    labels.index = pd.to_numeric(labels.index, errors="coerce")
    labels = labels[labels.index.notna()].copy()
    labels.index = labels.index.astype(int)
    disease_cols = [c for c in labels.columns if c.startswith("dis__")]
    print(f"  {len(labels):,} subjects × {len(disease_cols)} diseases")

    # Sex series for sex-specific restriction
    sex_col_name = next(
        (c for c in labels.columns if "sex" in c.lower() or "gender" in c.lower()), None
    )
    sex = None
    if sex_col_name:
        sex = pd.to_numeric(
            labels[sex_col_name].map({"Female": 0, "Male": 1, "F": 0, "M": 1}),
            errors="coerce",
        )

    # ── Load embeddings ────────────────────────────────────────────────────────
    print("\nLoading embeddings …")
    embeddings: dict[str, pd.DataFrame] = {}
    for name, fname in [("lejepa", "lejepa_fusion.pkl"), ("dino", "dino_fusion.pkl")]:
        path = os.path.join(EMB_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {name}")
            continue
        embeddings[name] = _load_embedding(path)
        print(f"  {name}: {embeddings[name].shape}")

    if not embeddings:
        raise FileNotFoundError(f"No embedding files found in {EMB_DIR}")

    # Optional PCA on embeddings (fit on full cohort — unsupervised, no label leakage).
    if pca_components is not None:
        for name, emb in embeddings.items():
            n_comp = min(pca_components, emb.shape[1], emb.shape[0] - 1)
            vals = emb.values.astype(float)
            finite_rows = np.all(np.isfinite(vals), axis=1)
            pca = PCA(n_components=n_comp, random_state=0)
            pca.fit(vals[finite_rows])
            proj = pca.transform(np.where(np.isfinite(vals), vals, 0.0))
            embeddings[name] = pd.DataFrame(proj, index=emb.index)
            print(f"  PCA {name}: {emb.shape[1]} → {n_comp} components "
                  f"({pca.explained_variance_ratio_.sum():.1%} var explained)")

    # ── Load tabular ──────────────────────────────────────────────────────────
    print("\nLoading tabular features …")
    cov_df, dxa_df = _load_tabular()

    # 4-way intersection: labels ∩ lejepa ∩ dino ∩ cov ∩ tabular.
    # Use data availability (non-NaN rows), not just index membership — the tabular CSV
    # covers all UKBB subjects so every subject passes an index check, even those missing
    # visit-2 DXA measurements or covariates.
    cov_valid = cov_df.dropna().index                                      # all 3 covariates present
    dxa_thresh = int(0.5 * dxa_df.shape[1])                               # ≤50% missing (matches HPP)
    dxa_valid = dxa_df.loc[dxa_df.isnull().sum(axis=1) <= dxa_thresh].index
    shared_idx = labels.index
    for name, emb in embeddings.items():
        shared_idx = shared_idx.intersection(emb.index)
    shared_idx = shared_idx.intersection(cov_valid)
    shared_idx = shared_idx.intersection(dxa_valid)
    n_intersected = len(shared_idx)
    disease_cols = [
        d for d in disease_cols
        if int(labels.loc[shared_idx, d].sum()) >= MIN_POSITIVE
    ]
    if targets:                                   # subset for parallel array runs
        disease_cols = [d for d in disease_cols if d in set(targets)]
    print(f"  cov_valid={len(cov_valid):,}  dxa_valid (≤50% missing)={len(dxa_valid):,}")
    print(f"  Intersected cohort (labels ∩ embeddings ∩ cov ∩ tabular): {n_intersected:,} subjects")
    print(f"  {len(disease_cols)} diseases pass ≥{MIN_POSITIVE} positives filter")

    # ── Run classification ─────────────────────────────────────────────────────
    seed_rows = []
    n_total = len(disease_cols) * len(MODELS) * n_seeds
    done = 0

    for dis_col in sorted(disease_cols):
        # Restrict to shared_idx so all arms see the same subjects.
        y_full = labels[dis_col].loc[shared_idx].dropna()

        # Sex restriction
        if dis_col in _SEX_SPECIFIC_FEMALE and sex is not None:
            y_full = y_full.loc[y_full.index.intersection(sex[sex == 0].index)]
        elif dis_col in _SEX_SPECIFIC_MALE and sex is not None:
            y_full = y_full.loc[y_full.index.intersection(sex[sex == 1].index)]

        n_pos = int((y_full == 1).sum())
        if n_pos < MIN_POSITIVE or n_pos > len(y_full) - MIN_POSITIVE:
            continue

        print(f"\n>>> {dis_col}  (n_pos={n_pos}, n={len(y_full)})")

        for seed in seeds:
            try:
                tr_idx, va_idx = train_test_split(
                    np.arange(len(y_full)),
                    test_size=1 - TRAIN_FRAC,
                    stratify=y_full.values,
                    random_state=seed,
                )
            except ValueError:
                continue

            tr_eids = y_full.index[tr_idx]
            va_eids = y_full.index[va_idx]
            y_tr = y_full.iloc[tr_idx].values.astype(float)
            y_va = y_full.iloc[va_idx].values.astype(float)

            cov_tr = cov_df.reindex(tr_eids).values.astype(float)
            cov_va = cov_df.reindex(va_eids).values.astype(float)

            # ── Arm 1: Covariates only ────────────────────────────────────────
            m_cov = _fit_classify(cov_tr, cov_va, y_tr, y_va, rand_n_iter=rand_n_iter, seed=seed, model=model, n_cs=n_cs, inner_cv=inner_cv, c_lo=c_lo, c_hi=c_hi, diff_pen=diff_pen, n_cov=cov_tr.shape[1])
            seed_rows.append({"disease": dis_col, "model": "covariates",
                               "seed": seed, "auc": m_cov["auc"], "pr": m_cov["pr"]})

            # ── Arm 2: LeJEPA (+ Covariates) ─────────────────────────────────
            if "lejepa" in embeddings:
                emb_tr = embeddings["lejepa"].reindex(tr_eids).values.astype(float)
                emb_va = embeddings["lejepa"].reindex(va_eids).values.astype(float)
                if use_covariates:
                    X_tr = np.concatenate([emb_tr, cov_tr], axis=1)
                    X_va = np.concatenate([emb_va, cov_va], axis=1)
                else:
                    X_tr, X_va = emb_tr, emb_va
                m_lc = _fit_classify(X_tr, X_va, y_tr, y_va, rand_n_iter=rand_n_iter, seed=seed, model=model, n_cs=n_cs, inner_cv=inner_cv, c_lo=c_lo, c_hi=c_hi, diff_pen=diff_pen, n_cov=cov_tr.shape[1])
                seed_rows.append({"disease": dis_col, "model": "lejepa_cov",
                                   "seed": seed, "auc": m_lc["auc"], "pr": m_lc["pr"]})

            # ── Arm 3: DINO (+ Covariates) ───────────────────────────────────
            if "dino" in embeddings:
                emb_tr_d = embeddings["dino"].reindex(tr_eids).values.astype(float)
                emb_va_d = embeddings["dino"].reindex(va_eids).values.astype(float)
                if use_covariates:
                    X_tr = np.concatenate([emb_tr_d, cov_tr], axis=1)
                    X_va = np.concatenate([emb_va_d, cov_va], axis=1)
                else:
                    X_tr, X_va = emb_tr_d, emb_va_d
                m_dc = _fit_classify(X_tr, X_va, y_tr, y_va, rand_n_iter=rand_n_iter, seed=seed, model=model, n_cs=n_cs, inner_cv=inner_cv, c_lo=c_lo, c_hi=c_hi, diff_pen=diff_pen, n_cov=cov_tr.shape[1])
                seed_rows.append({"disease": dis_col, "model": "dino_cov",
                                   "seed": seed, "auc": m_dc["auc"], "pr": m_dc["pr"]})

            # ── Arm 4: DXA Tabular (+ Covariates) ────────────────────────────
            tab_tr = dxa_df.reindex(tr_eids).values.astype(float)
            tab_va = dxa_df.reindex(va_eids).values.astype(float)
            if use_covariates:
                X_tr = np.concatenate([tab_tr, cov_tr], axis=1)
                X_va = np.concatenate([tab_va, cov_va], axis=1)
            else:
                X_tr, X_va = tab_tr, tab_va
            m_tc = _fit_classify(X_tr, X_va, y_tr, y_va, rand_n_iter=rand_n_iter, seed=seed, model=model, n_cs=n_cs, inner_cv=inner_cv, c_lo=c_lo, c_hi=c_hi, diff_pen=diff_pen, n_cov=cov_tr.shape[1])
            seed_rows.append({"disease": dis_col, "model": "tab_cov",
                               "seed": seed, "auc": m_tc["auc"], "pr": m_tc["pr"]})

            done += len(MODELS)
            if done % 200 == 0:
                print(f"  [{done}/{n_total}] {dis_col} seed={seed}")

    # ── Save seed-level results ────────────────────────────────────────────────
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out_seeds, index=False)
    print(f"\nSaved seed-level → {out_seeds}  ({len(seed_df)} rows)")

    # ── Aggregate + Wilcoxon + FDR ────────────────────────────────────────────
    diseases = sorted(seed_df["disease"].unique())

    def _paired_wilcoxon(disease, model_a, model_b):
        """Two-tailed: H1 is model_b != model_a (called as _paired_wilcoxon(ref, test));
        combine with the sign of mean(test - ref) to determine direction."""
        a = seed_df[(seed_df["disease"] == disease) & (seed_df["model"] == model_a)].set_index("seed")["auc"]
        b = seed_df[(seed_df["disease"] == disease) & (seed_df["model"] == model_b)].set_index("seed")["auc"]
        shared = sorted(a.index.intersection(b.index))
        if len(shared) < 2:
            return float("nan")
        diff = b.reindex(shared).values - a.reindex(shared).values  # test - ref
        if np.all(diff == 0):
            return 1.0
        _, p = wilcoxon(diff, alternative="two-sided")
        return float(p)

    summary_rows = []
    for dis in diseases:
        sub = seed_df[seed_df["disease"] == dis]
        y_dis = labels[dis].loc[shared_idx].dropna() if dis in labels.columns else pd.Series(dtype=float)
        n_pos   = int((y_dis == 1).sum())
        n_total = len(y_dis)
        prev    = round(float(n_pos / n_total), 5) if n_total > 0 else float("nan")
        row = {"disease": dis, "N_positives": n_pos, "N_total": n_total, "Prevalence": prev}
        for m in MODELS:
            msub = sub[sub["model"] == m]
            aucs = msub["auc"].dropna().values
            prs  = msub["pr"].dropna().values
            row[f"{MODEL_LABELS[m]}_mean"] = round(float(np.mean(aucs)), 4) if len(aucs) else float("nan")
            row[f"{MODEL_LABELS[m]}_SE"]   = round(float(np.std(aucs, ddof=1) / np.sqrt(len(aucs))), 4) if len(aucs) > 1 else float("nan")
            row[f"{MODEL_LABELS[m]}_N"]    = len(aucs)
            row[f"{MODEL_LABELS[m]}_PR_mean"] = round(float(np.mean(prs)), 4) if len(prs) else float("nan")
            row[f"{MODEL_LABELS[m]}_PR_SE"]   = round(float(np.std(prs, ddof=1) / np.sqrt(len(prs))), 4) if len(prs) > 1 else float("nan")
        for ref, col_name, test in COMPARISONS:
            row[f"P_{col_name}_raw"] = _paired_wilcoxon(dis, ref, test)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("disease")

    # BH-FDR per comparison column
    for _, col_name, _ in COMPARISONS:
        raw_vals = summary_df[f"P_{col_name}_raw"].values.astype(float)
        finite = np.isfinite(raw_vals)
        adj = np.full(len(raw_vals), float("nan"))
        if finite.sum() > 0:
            adj[finite] = _bh_adjust(raw_vals[finite])
        summary_df[f"P_{col_name}_adj"] = np.round(adj, 6)

    # Reorder columns
    base_cols = ["disease", "N_positives", "N_total", "Prevalence"]
    metric_cols = [f"{MODEL_LABELS[m]}_{s}" for m in MODELS
                   for s in ("mean", "SE", "N", "PR_mean", "PR_SE")]
    pval_cols = [f"P_{cn}_{s}" for _, cn, _ in COMPARISONS for s in ("raw", "adj")]
    summary_df = summary_df[base_cols + metric_cols + pval_cols]

    summary_df.to_csv(out_summary, index=False)
    print(f"Saved summary    → {out_summary}  ({len(summary_df)} rows)")

    # ── Quick results table ────────────────────────────────────────────────────
    cov_lbl  = MODEL_LABELS['covariates']
    lej_lbl  = MODEL_LABELS['lejepa_cov']
    dino_lbl = MODEL_LABELS['dino_cov']
    tab_lbl  = MODEL_LABELS['tab_cov']
    print("\n=== Results (FM vs Covariates; FDR-adj p) ===")
    print(f"{'Disease':<35} {'Cov':>6} {'FM+C':>6} {'DIN+C':>6} {'Tab+C':>6} {'p_adj':>8}")
    print("-" * 70)
    for _, r in summary_df.sort_values(f"{lej_lbl}_mean", ascending=False).iterrows():
        print(
            f"{r['disease']:<35} "
            f"{r[f'{cov_lbl}_mean']:>6.3f} "
            f"{r[f'{lej_lbl}_mean']:>6.3f} "
            f"{r[f'{dino_lbl}_mean']:>6.3f} "
            f"{r[f'{tab_lbl}_mean']:>6.3f} "
            f"{r['P_LeJEPAcov_vs_Cov_adj']:>8.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--out-seeds",   default=DEFAULT_OUT_SEEDS)
    parser.add_argument("--out-summary", default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--pca", type=int, default=DEFAULT_PCA,
                        help="PCA components for embeddings (0 = disabled)")
    parser.add_argument("--no-cov", action="store_true",
                        help="Run embedding/tabular arms without age/sex/BMI covariates")
    parser.add_argument("--rand-search", action="store_true",
                        help="Use RandomizedSearchCV instead of fixed C grid")
    parser.add_argument("--rand-n-iter", type=int, default=5,
                        help="Random search iterations (default 5)")
    parser.add_argument("--model", choices=["logreg", "lgbm"], default="logreg",
                        help="Classifier: linear logistic (default) or non-linear LightGBM")
    parser.add_argument("--n-cs", type=int, default=N_CS,
                        help=f"Number of C values in the logistic grid (default {N_CS})")
    parser.add_argument("--inner-cv", type=int, default=INNER_CV,
                        help=f"Inner-CV folds for hyperparameter selection (default {INNER_CV})")
    parser.add_argument("--c-lo", type=int, default=-3,
                        help="Lower log10 exponent of the C grid (default -3)")
    parser.add_argument("--c-hi", type=int, default=3,
                        help="Upper log10 exponent of the C grid (default 3)")
    parser.add_argument("--diff-pen", action="store_true",
                        help="Differential per-block penalisation: leave the age/sex/BMI "
                             "covariate block unpenalised; tune L2 only on the imaging/tabular block")
    parser.add_argument("--emb-dir", default=None,
                        help="Override the embeddings directory.")
    parser.add_argument("--disease-labels", default=DISEASE_LABELS,
                        help="CSV containing eid-indexed dis__ outcome columns.")
    parser.add_argument("--tabular-csv", default=TABULAR_CSV,
                        help="UK Biobank DXA tabular and covariate CSV.")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Subset of dis__ targets to run (for parallel array jobs).")
    args = parser.parse_args()
    DISEASE_LABELS = args.disease_labels
    TABULAR_CSV = args.tabular_csv
    if args.emb_dir:
        EMB_DIR = args.emb_dir
        print(f"[emb-dir override] {EMB_DIR}")
    # Auto-adjust output names for no-cov / pca / rand-search / model / grid runs
    _cov_tag = "_nocov" if args.no_cov else ""
    _pca_tag = f"_pca{args.pca}" if args.pca > 0 else ""
    _rs_tag  = f"_rs{args.rand_n_iter}" if args.rand_search else ""
    _model_tag = "" if args.model == "logreg" else f"_{args.model}"
    _grid_tag = (f"_grid{args.n_cs}cv{args.inner_cv}"
                 if (args.model == "logreg"
                     and (args.n_cs != N_CS or args.inner_cv != INNER_CV
                          or args.c_lo != -3 or args.c_hi != 3))
                 else "")
    _dp_tag = "_diffpen" if args.diff_pen else ""
    _suffix  = _cov_tag + _pca_tag + _rs_tag + _model_tag + _grid_tag + _dp_tag
    out_seeds   = args.out_seeds   if args.out_seeds   != DEFAULT_OUT_SEEDS   else DEFAULT_OUT_SEEDS.replace(".csv",   f"{_suffix}.csv") if _suffix else DEFAULT_OUT_SEEDS
    out_summary = args.out_summary if args.out_summary != DEFAULT_OUT_SUMMARY else DEFAULT_OUT_SUMMARY.replace(".csv", f"{_suffix}.csv") if _suffix else DEFAULT_OUT_SUMMARY
    rand_n_iter = args.rand_n_iter if args.rand_search else None
    main(n_seeds=args.n_seeds, out_seeds=out_seeds, out_summary=out_summary,
         pca_components=args.pca if args.pca > 0 else None,
         use_covariates=not args.no_cov,
         rand_n_iter=rand_n_iter,
         model=args.model, n_cs=args.n_cs, inner_cv=args.inner_cv,
         c_lo=args.c_lo, c_hi=args.c_hi, diff_pen=args.diff_pen,
         targets=args.targets)
