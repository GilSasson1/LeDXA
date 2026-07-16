"""
prepare_disease_targets.py

Builds disease_targets.csv and disease_targets_with_covs.csv: one binary (0/1)
column per disease, indexed by (RegistrationCode, research_stage) covering ALL
visits in the embedding pool.

Rationale: diseases are chronic. A subject diagnosed at baseline OR follow-up is
considered sick at all their visits. Labels are replicated across all visits a
subject has in the embedding pool. Both baseline_conditions_all.csv and
follow_up_conditions_all.csv are used (matching disease_classfication.py).

Also saves csvs/disease_display_names.json mapping sanitized column names
→ original disease names for use in plot labels.

Usage:
    python prepare_disease_targets.py
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

import common.utils as U

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Baseline-only: labels are prevalent-at-scan (consistent with UKBB prevalent-at-imaging design
# and with --first-scan-only in compare_lp.py). Follow-up diagnoses may be incident (post-scan),
# introducing label noise for the first-scan prediction task.
CONDITIONS_CSVS = [
    "/data/hpp_labdata/Data/10K/for_review/baseline_conditions_all.csv",
]
COV_PATH       = "/data/hpp_labdata/Analyses/10K_Trajectories/body_systems/Age_Gender_BMI.csv"
MIN_CASES      = 25   # minimum unique positive subjects in the embedding pool

_PARSER = argparse.ArgumentParser()
_PARSER.add_argument("--level", default="Consolidated name",
                     choices=["Consolidated name", "Group", "english_name"],
                     help="Disease label hierarchy level to use")
_ARGS = _PARSER.parse_args()
_LEVEL = _ARGS.level

# Output paths: add level suffix for non-default levels
_LEVEL_TAG = "" if _LEVEL == "Consolidated name" else f"_{_LEVEL.lower().replace(' ', '_')}"
OUT_CSV        = os.path.join(os.path.dirname(__file__), "csvs", f"disease_targets{_LEVEL_TAG}.csv")
OUT_WITH_COV   = os.path.join(os.path.dirname(__file__), "csvs", f"disease_targets{_LEVEL_TAG}_with_covs.csv")
OUT_NAMES_JSON = os.path.join(os.path.dirname(__file__), "csvs", f"disease_display_names{_LEVEL_TAG}.json")

# gender=1 → male, gender=0 → female (verified via creatinine/hemoglobin means)
# These diseases are biologically exclusive to one sex — evaluate within that sex only
# (labels for wrong-sex subjects are set to NaN so they are excluded from evaluation)
SEX_SPECIFIC_FEMALE = {
    "endometriosis and adenomyosis",
    "polycystic ovary disease",
    "breast cancer",
    "perimenopausal disorders",
    # Group-level equivalents
    "obgyn",
}
SEX_SPECIFIC_MALE = {
    "erectile dysfunction",
}

# DXA-derived conditions: BMD or fat% directly define these diagnoses.
# Kept in a separate panel rather than mixed with general diseases.
DXA_DIRECT_DISEASES = {
    "obesity",
    "osteopenia",
    "osteoporosis",
}


def sanitize_colname(name: str) -> str:
    """Convert disease name to a safe DataFrame column name."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return f"dis__{name}"


def main():
    # ── Load conditions (baseline + follow-up, matching disease_classfication.py) ─
    frames = []
    for path in CONDITIONS_CSVS:
        print(f"Loading conditions: {path}")
        df = pd.read_csv(path)
        df["RegistrationCode"] = (
            df["RegistrationCode"].astype(str)
            .apply(lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
        )
        frames.append(df)
    conditions = pd.concat(frames, ignore_index=True)
    # Normalize trailing/leading whitespace in the label column
    conditions[_LEVEL] = conditions[_LEVEL].str.strip()
    diseases = conditions[_LEVEL].dropna().unique()
    print(f"Total '{_LEVEL}' labels in source: {len(diseases)}")

    # ── Build base pool: (RC, research_stage) present in BOTH embedding models ─
    print("\nBuilding embedding base pool...")
    base_idx = None
    for model in ("lejepa", "dino"):
        emb_path = os.path.join(U.EMBEDDINGS_DIR, f"{model}_bone.pkl")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embedding not found: {emb_path}")
        emb = pd.read_pickle(emb_path)
        idx = emb.index
        if idx.nlevels == 3:
            idx = idx.droplevel("Date")
        print(f"  {model}_bone: {len(idx)} rows | {idx.get_level_values('RegistrationCode').nunique()} subjects")
        base_idx = idx if base_idx is None else base_idx.intersection(idx)

    print(f"  Intersection: {len(base_idx)} rows | {base_idx.get_level_values('RegistrationCode').nunique()} subjects")
    all_rcs = set(base_idx.get_level_values("RegistrationCode"))

    # ── Load gender labels for sex-stratification ──────────────────────────
    targets_df = pd.read_csv(U.TARGETS_CSV, index_col=[0, 1])
    targets_df.sort_index(inplace=True)
    gender_series = targets_df["gender"].reindex(base_idx)  # 1=male, 0=female
    male_rcs   = set(gender_series[gender_series == 1.0].index.get_level_values("RegistrationCode"))
    female_rcs = set(gender_series[gender_series == 0.0].index.get_level_values("RegistrationCode"))
    print(f"  Gender: {len(male_rcs)} male RCs | {len(female_rcs)} female RCs in pool")

    # ── Create binary disease columns ──────────────────────────────────────
    disease_df = pd.DataFrame(index=base_idx)
    col_to_name = {}
    col_to_group = {}   # "general" | "sex_specific" | "dxa_direct"
    skipped = []

    for disease in sorted(diseases):
        pos_rcs_raw = set(conditions[conditions[_LEVEL] == disease]["RegistrationCode"])
        pos_rcs = pos_rcs_raw & all_rcs  # restrict to subjects in embedding pool

        disease_lower = disease.lower().strip()

        # Determine sex restriction (if any)
        sex_restrict = None
        if disease_lower in {d.lower() for d in SEX_SPECIFIC_FEMALE}:
            sex_restrict = "female"
            pos_rcs = pos_rcs & female_rcs
        elif disease_lower in {d.lower() for d in SEX_SPECIFIC_MALE}:
            sex_restrict = "male"
            pos_rcs = pos_rcs & male_rcs

        if len(pos_rcs) < MIN_CASES:
            skipped.append((disease, len(pos_rcs)))
            continue

        col = sanitize_colname(disease)
        labels = (
            disease_df.index.get_level_values("RegistrationCode").isin(pos_rcs)
            .astype(np.float32)
        )

        # For sex-specific diseases: mask out wrong-sex subjects with NaN
        if sex_restrict == "female":
            wrong_sex = ~disease_df.index.get_level_values("RegistrationCode").isin(female_rcs)
            labels = labels.astype(object)
            labels[wrong_sex] = np.nan
        elif sex_restrict == "male":
            wrong_sex = ~disease_df.index.get_level_values("RegistrationCode").isin(male_rcs)
            labels = labels.astype(object)
            labels[wrong_sex] = np.nan

        disease_df[col] = labels

        # Tag disease group
        if disease_lower in {d.lower() for d in DXA_DIRECT_DISEASES}:
            group = "dxa_direct"
        elif sex_restrict is not None:
            group = "sex_specific"
        else:
            group = "general"

        col_to_name[col]  = disease
        col_to_group[col] = group
        sex_note = f" [sex={sex_restrict}]" if sex_restrict else ""
        n_labeled = int(pd.Series(labels).notna().sum())
        print(f"  [{group}] [{col}] '{disease}'{sex_note} — {len(pos_rcs)} pos / {n_labeled} labeled rows")

    print(f"\nKept: {len(col_to_name)} diseases | Skipped (<{MIN_CASES} cases): {len(skipped)}")
    if skipped:
        print(f"  Skipped: {[d for d, _ in skipped]}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    disease_df.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}  ({disease_df.shape})")

    with open(OUT_NAMES_JSON, "w") as f:
        json.dump(col_to_name, f, indent=2)
    print(f"Saved: {OUT_NAMES_JSON}")

    out_groups = OUT_NAMES_JSON.replace("disease_display_names", "disease_groups")
    with open(out_groups, "w") as f:
        json.dump(col_to_group, f, indent=2)
    print(f"Saved: {out_groups}")

    # ── Build disease_targets_with_covs.csv (join covariates) ─────────────
    cov_df = pd.read_csv(COV_PATH, index_col=[0, 1])
    with_cov = disease_df.join(cov_df[["age", "gender", "bmi"]], how="left")
    with_cov.to_csv(OUT_WITH_COV)
    print(f"Saved: {OUT_WITH_COV}  ({with_cov.shape})")
    n_missing_cov = with_cov["age"].isna().sum()
    if n_missing_cov:
        print(f"  Warning: {n_missing_cov} rows missing covariates (will be excluded from covariate model)")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n=== Prevalence summary (% positive rows) ===")
    prev = disease_df.mean().sort_values(ascending=False)
    for col, p in prev.items():
        print(f"  {col_to_name[col]:<45} {p*100:.1f}%")


if __name__ == "__main__":
    main()
