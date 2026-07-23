"""
compute_age_mae_imaging_only.py — HPP age MAE (years) for the IMAGING-ONLY arms.

Same protocol as compute_age_mae.py (80/20 subject split, StandardScaler, RidgeCV,
target z-scored, MAE-in-years = age_std × mean|pred − y|, 10 seeds) but the imaging
arms carry NO covariate block — they are embeddings / tabular features alone. The
covariates arm (sex+BMI) is kept as the baseline bar. Used by Fig 2 panel c when it
is rendered imaging-only.

Output: tables/age_mae_imaging_only.csv (cohort, model, n_seeds, pearson_mean/se, mae_yr_mean/se)
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
import common.utils as U
from config import TABLES_DIR
from downstream.disease.linear_probe import _COV_COLS, _TABULAR_LEAKAGE_EXCLUSIONS, _impute, load_embeddings

HPP_TABULAR = None
OUT_CSV = str(TABLES_DIR / "age_mae_imaging_only.csv")
ARMS = ["covariates", "lejepa", "dino", "tabular"]


def _fit_predict(blocks_tr, blocks_va, y_tr):
    X_tr = np.concatenate(blocks_tr, axis=1)
    X_va = np.concatenate(blocks_va, axis=1)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    reg = RidgeCV(alphas=U.RIDGE_ALPHAS, cv=U.RIDGE_CV_FOLDS)
    reg.fit(X_tr, y_tr)
    return reg.predict(X_va)


def _agg(rows):
    out, df = [], pd.DataFrame(rows)
    for m in ARMS:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        def ms(col):
            a = sub[col].dropna().values
            mean = float(np.mean(a)) if len(a) else float("nan")
            se = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
            return mean, se
        pm, ps = ms("pearson"); mm, msd = ms("mae_yr")
        out.append({"cohort": "HPP", "model": m, "n_seeds": len(sub),
                    "pearson_mean": round(pm, 4), "pearson_se": round(ps, 4),
                    "mae_yr_mean": round(mm, 3), "mae_yr_se": round(msd, 3)})
    return out


def run_hpp(n_seeds=10, bone_pool=False, first_scan=False):
    print(f"\n=== HPP age (imaging-only, bone+tissue{'; BONE-POOL' if bone_pool else ''}) ===")
    embs = load_embeddings(U.EMBEDDINGS_DIR, ["lejepa", "dino"])
    if bone_pool:
        for m in ["lejepa", "dino"]:
            if "regionpool" in embs[m]:
                bone = embs[m]["bone"]; rg = embs[m]["regionpool"].reindex(bone.index)
                ok = rg.notna().all(axis=1); enr = bone.copy()
                enr.loc[ok] = (bone.loc[ok].values + rg.loc[ok].values) / 2.0
                embs[m]["bone"] = enr
                print(f"  [{m}] bone-pool: enriched bone with regional on {int(ok.sum())}/{len(bone)}")
    tab = (pd.read_csv(HPP_TABULAR).set_index(["RegistrationCode", "research_stage"]))
    tab = tab[~tab.index.duplicated(keep="first")].sort_index().select_dtypes(include="number")
    tdf_full = pd.read_csv(U.TARGETS_CSV, index_col=[0, 1]).sort_index()

    target = "age"
    target_df = tdf_full[[target]].dropna().copy()
    age_std = float(target_df[target].std())
    age_mean = float(target_df[target].mean())
    target_df[target] = (target_df[target] - age_mean) / age_std
    avail_cov = [c for c in _COV_COLS if c in tdf_full.columns and c != target]  # gender, bmi
    print(f"  age_std={age_std:.3f} yr | covariates={avail_cov}")

    bone_idx = set(embs["lejepa"]["bone"].index)
    labeled = {}
    for idx in target_df.index:
        labeled.setdefault(idx[0], []).append(idx)
    labeled = {s: r for s, r in labeled.items() if any(x in bone_idx for x in r)}
    if first_scan:
        _ord = {'baseline': 0, '00_00_visit': 0, '02_00_visit': 2,
                '04_00_visit': 4, '06_00_visit': 6}
        labeled = {s: [min(r, key=lambda x: _ord.get(x[1], 99))] for s, r in labeled.items()}
        print("  first-scan-only: one (earliest) scan per subject")

    feat_cols_tab = [c for c in tab.columns if pd.api.types.is_numeric_dtype(tab[c])]
    if target in _TABULAR_LEAKAGE_EXCLUSIONS:
        excl = set(_TABULAR_LEAKAGE_EXCLUSIONS[target](feat_cols_tab))
        feat_cols_tab = [c for c in feat_cols_tab if c not in excl]
    thresh = int(0.5 * len(feat_cols_tab))
    valid_tab_idx = set(tab[feat_cols_tab].isnull().sum(axis=1)[lambda s: s <= thresh].index)
    valid_tab_rcs = {rc for rc, _ in valid_tab_idx}
    labeled = {s: [r for r in rows if r in valid_tab_idx]
               for s, rows in labeled.items() if s in valid_tab_rcs}
    labeled = {s: r for s, r in labeled.items() if r}
    valid_subjects = sorted(labeled)
    print(f"  subjects (emb ∩ label ∩ tabular): {len(valid_subjects)}")

    dino_idx = set(embs["dino"]["bone"].index)
    rows = []
    for seed in U.make_seeds(n_seeds):
        tr_s, va_s = train_test_split(valid_subjects, test_size=0.2, random_state=seed)
        tr = [r for s in tr_s for r in labeled[s] if r in target_df.index]
        va = [r for s in va_s for r in labeled[s] if r in target_df.index]

        def cov_blk(ix):
            return tdf_full.loc[ix, avail_cov].values.astype(float)

        for arm in ARMS:
            if arm == "covariates":
                tl, vl = [r for r in tr if r in bone_idx], [r for r in va if r in bone_idx]
                ct, cv = _impute(cov_blk(tl), cov_blk(vl))
                bt, bv = [ct], [cv]
            elif arm == "lejepa":
                tl, vl = [r for r in tr if r in bone_idx], [r for r in va if r in bone_idx]
                fm_t = np.concatenate([embs["lejepa"]["bone"].loc[tl].values,
                                       embs["lejepa"]["tissue"].loc[tl].values], axis=1)
                fm_v = np.concatenate([embs["lejepa"]["bone"].loc[vl].values,
                                       embs["lejepa"]["tissue"].loc[vl].values], axis=1)
                bt, bv = [fm_t], [fm_v]
            elif arm == "dino":
                tl = [r for r in tr if r in dino_idx]; vl = [r for r in va if r in dino_idx]
                fm_t = np.concatenate([embs["dino"]["bone"].loc[tl].values,
                                       embs["dino"]["tissue"].loc[tl].values], axis=1)
                fm_v = np.concatenate([embs["dino"]["bone"].loc[vl].values,
                                       embs["dino"]["tissue"].loc[vl].values], axis=1)
                bt, bv = [fm_t], [fm_v]
            else:  # tabular (imaging-derived DXA features, no cov)
                tl = [r for r in tr if r in bone_idx and r in valid_tab_idx]
                vl = [r for r in va if r in bone_idx and r in valid_tab_idx]
                tt, tv = _impute(tab.loc[tl, feat_cols_tab].values.astype(float),
                                 tab.loc[vl, feat_cols_tab].values.astype(float))
                bt, bv = [tt], [tv]
            if len(vl) < 10:
                continue
            y_tr = target_df.loc[tl, target].values.astype(float)
            y_va = target_df.loc[vl, target].values.astype(float)
            pred = _fit_predict(bt, bv, y_tr)
            pear = U.metric(y_va, pred, False)
            mae_yr = age_std * float(np.mean(np.abs(pred - y_va)))
            rows.append({"model": arm, "seed": seed, "pearson": pear, "mae_yr": mae_yr})
        print(f"  seed={seed} done")
    return _agg(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-csv", required=True)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--bone-pool", action="store_true")
    parser.add_argument("--first-scan-only", action="store_true")
    args = parser.parse_args()
    HPP_TABULAR = args.tabular_csv
    out = run_hpp(args.num_seeds, bone_pool=args.bone_pool,
                  first_scan=args.first_scan_only)
    sfx = ("_bonepool" if args.bone_pool else "") + (
        "_firstscan" if args.first_scan_only else "")
    out_csv = args.out_csv.replace(".csv", f"{sfx}.csv") if sfx else args.out_csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(pd.DataFrame(out).to_string(index=False))
