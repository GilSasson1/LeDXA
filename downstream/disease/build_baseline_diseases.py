"""
build_ukbb_baseline_diseases.py

Build ukbb_baseline_disease_targets.csv: binary prevalent-disease labels
at the visit-2 DXA scan date, for UKBB subjects with LeJEPA embeddings.

Prevalent = ICD-10 first-occurrence date <= visit-2 DXA scan date.

Sources:
  A) Events CSV  — ICD-10 date columns already in
     ukbb_osteo_data_expanded_aligned.csv
  B) UkbbLoader  — extra ICD-10 first-occurrence fields not in events CSV
     (hypertension, gout, depression, anxiety, IBS, etc.)

Baseline dates come from ukbb_tabular_data_for_cox_with_baseline.csv,
which covers all 502k UKBB subjects (wider than the events CSV).

Usage:
    python build_ukbb_baseline_diseases.py
    python build_ukbb_baseline_diseases.py --min-cases 50 --out my_labels.csv
"""

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, "/path/to")

EVENTS_CSV = "/path/to/project/ukbb_osteo_data_expanded_aligned.csv"
TABULAR_CSV = "/path/to/project/ukbb_tabular_data_for_cox_with_baseline.csv"
LEJEPA_EMB = (
    "/data/hpp_labdata/Analyses/gilsa/embeddings/"
    "ukbb_comparison/lejepa_fusion.pkl"
)
BASELINE_DATE_COL = "Date of attending assessment centre - visit 2"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "ukbb_baseline_disease_targets.csv")
DEFAULT_MIN_CASES = 50
EXTRA_CACHE = os.path.join(os.path.dirname(__file__), "ukbb_baseline_extra_icd_cache.csv")

# ── Disease definitions ───────────────────────────────────────────────────────

# (A) ICD-10 prefixes present as "Date {code} first reported (...) - visit 0"
# columns in EVENTS_CSV.
ICD10_FROM_EVENTS = {
    "diabetes":               ["E10", "E11"],
    "hypothyroidism":         ["E03"],
    "hyperthyroidism":        ["E05"],
    "ischemic_heart_disease": ["I20", "I21", "I22", "I25"],
    "heart_failure":          ["I50"],
    "atrial_fibrillation":    ["I48"],
    "stroke":                 ["I61", "I63", "I64"],
    "copd":                   ["J44", "J46"],
    "asthma":                 ["J45"],
    "liver_disease":          ["K76"],
    "rheumatoid_arthritis":   ["M05", "M06"],
    "osteoarthritis":         ["M16", "M17", "M19"],
    "back_pain":              ["M48", "M51"],
    "osteoporosis":           ["M80", "M81", "M82"],
    "fracture":               ["M84"],
    "renal_failure":          ["N17", "N18", "N19"],
}

# (B) Extra ICD-10 first-occurrence fields — read directly from UKBB parquet.
# Keys = short disease name; values = parquet column "{field_id}-0.0"
# Field IDs from ukbb_fields_metadata.csv (ukbb_base_metadata_dir).
UKBB_PARQUET = "/data/ukbb_raw/Data/ukb676772.parquet"
ICD10_EXTRA = {
    "hypertension":    "131286-0.0",
    "back_pain_m54":   "131928-0.0",
    "gout":            "131858-0.0",
    "psoriasis":       "131742-0.0",
    "ibs":             "131638-0.0",
    "celiac":          "131688-0.0",
    "gallstone":       "131674-0.0",
    "kidney_stones":   "132036-0.0",
    "anemia_iron":     "130622-0.0",
    "anemia_b12":      "130624-0.0",
    "anemia_folate":   "130626-0.0",
    "anemia_other":    "130648-0.0",
    "depression":      "130894-0.0",
    "depression_rec":  "130896-0.0",
    "anxiety":         "130906-0.0",
    "sleep_disorders": "131060-0.0",
    "hyperlipidemia":  "130814-0.0",
    "pcos":            "130736-0.0",
    "endometriosis":   "132122-0.0",
}

# OR-merge sub-codes into a single column after building all flags.
MERGE_AFTER = {
    "depression": ["depression", "depression_rec"],
    "back_pain":  ["back_pain", "back_pain_m54"],
    "anemia":     ["anemia_iron", "anemia_b12", "anemia_folate", "anemia_other"],
}


def _load_embedding_eids(emb_path: str) -> pd.Index:
    with open(emb_path, "rb") as f:
        emb = pickle.load(f)
    if isinstance(emb.index, pd.MultiIndex):
        visit_vals = emb.index.get_level_values(1).astype(str)
        eids = emb.index.get_level_values(0)[visit_vals == "2"]
    else:
        eids = emb.index
    return pd.Index(pd.to_numeric(eids, errors="coerce").dropna().astype(int).unique())


def _load_baseline_dates() -> pd.Series:
    """Visit-2 DXA scan dates indexed by eid (int), from the tabular file."""
    print(f"  Loading baseline dates from {TABULAR_CSV} …")
    df = pd.read_csv(TABULAR_CSV, index_col=0, usecols=["eid", BASELINE_DATE_COL],
                     low_memory=False)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df[df.index.notna()].copy()
    df.index = df.index.astype(int)
    s = pd.to_datetime(df[BASELINE_DATE_COL], errors="coerce")
    print(f"  Baseline dates non-null: {s.notna().sum():,} / {len(s):,}")
    return s


def _flags_from_events(ev: pd.DataFrame, baseline_dates: pd.Series) -> pd.DataFrame:
    flags = {}
    bl = baseline_dates.reindex(ev.index)
    has_baseline = bl.notna()
    for disease, prefixes in ICD10_FROM_EVENTS.items():
        date_cols = [c for c in ev.columns if any(f"Date {p} " in c for p in prefixes)]
        if not date_cols:
            print(f"  [events] no columns for {disease} ({prefixes}) — skipping")
            continue
        dates = ev[date_cols].apply(pd.to_datetime, errors="coerce")
        flag = dates.le(bl, axis=0).any(axis=1).astype(float)
        flag[~has_baseline] = float("nan")  # exclude subjects with no visit-2 date
        flags[disease] = flag
    return pd.DataFrame(flags, index=ev.index)


def _load_extra_icd_from_parquet() -> pd.DataFrame:
    """Read extra ICD-10 first-occurrence date columns directly from UKBB parquet."""
    if os.path.exists(EXTRA_CACHE):
        print(f"  Loading extra ICD-10 dates from cache: {EXTRA_CACHE}")
        df = pd.read_csv(EXTRA_CACHE, index_col=0)
        df.index = pd.to_numeric(df.index, errors="coerce")
        return df[df.index.notna()].copy()

    import pyarrow.parquet as pq
    cols_needed = ["eid"] + list(ICD10_EXTRA.values())
    print(f"  Reading {len(cols_needed) - 1} ICD-10 columns from parquet …")
    df = pq.read_table(UKBB_PARQUET, columns=cols_needed).to_pandas()
    df["eid"] = pd.to_numeric(df["eid"], errors="coerce")
    df = df[df["eid"].notna()].copy()
    df["eid"] = df["eid"].astype(int)
    df = df.set_index("eid")

    rename = {v: k for k, v in ICD10_EXTRA.items()}
    df = df.rename(columns=rename)
    for col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        print(f"    {col}: {df[col].notna().sum():,} non-null dates")

    df.to_csv(EXTRA_CACHE)
    print(f"  Cached → {EXTRA_CACHE}  ({len(df):,} rows × {df.shape[1]} cols)")
    return df


def _flags_from_extra(df_extra: pd.DataFrame, baseline_dates: pd.Series) -> pd.DataFrame:
    flags = {}
    has_baseline = baseline_dates.notna()
    for key in df_extra.columns:
        dates = pd.to_datetime(df_extra[key], errors="coerce").reindex(baseline_dates.index)
        flag = dates.le(baseline_dates).astype(float)
        flag[~has_baseline] = float("nan")
        flags[key] = flag
    return pd.DataFrame(flags, index=baseline_dates.index)


def _apply_merges(df: pd.DataFrame) -> pd.DataFrame:
    for merged_name, sources in MERGE_AFTER.items():
        available = [s for s in sources if s in df.columns]
        if not available:
            continue
        df[merged_name] = df[available].max(axis=1)
        drop = [s for s in available if s != merged_name]
        df = df.drop(columns=[c for c in drop if c in df.columns])
    return df


def main(min_cases: int = DEFAULT_MIN_CASES, out: str = DEFAULT_OUT) -> None:
    print("=== UKBB Baseline Disease Label Builder ===\n")

    print("Loading embedding eids …")
    emb_eids = _load_embedding_eids(LEJEPA_EMB)
    print(f"  Visit-2 embedding eids: {len(emb_eids):,}")

    print("\nLoading baseline visit-2 dates …")
    baseline_dates = _load_baseline_dates()

    print(f"\nLoading events CSV: {EVENTS_CSV}")
    ev = pd.read_csv(EVENTS_CSV, index_col=0, low_memory=False)
    ev = ev[ev.index.notna() & ~ev.index.isin(["error", "skipped"])].copy()
    ev.index = pd.to_numeric(ev.index, errors="coerce")
    ev = ev[ev.index.notna()].copy()
    ev.index = ev.index.astype(int)
    print(f"  Events CSV shape: {ev.shape}")

    print("\nPart A: Events CSV ICD-10 flags …")
    df_a = _flags_from_events(ev, baseline_dates)
    print(f"  Part A: {df_a.shape[1]} diseases")

    print("\nPart B: extra ICD-10 fields from parquet …")
    df_extra_dates = _load_extra_icd_from_parquet()
    if not df_extra_dates.empty:
        df_b = _flags_from_extra(df_extra_dates, baseline_dates)
    else:
        df_b = pd.DataFrame(index=baseline_dates.index)
    print(f"  Part B: {df_b.shape[1]} diseases")

    df_flags = df_a.join(df_b, how="outer")
    df_flags = _apply_merges(df_flags)
    df_flags.columns = [f"dis__{c}" if not c.startswith("dis__") else c for c in df_flags.columns]

    common = df_flags.index.intersection(emb_eids)
    df_flags = df_flags.loc[common].copy()
    print(f"\nAfter restricting to embedding eids: {len(df_flags):,} subjects")

    n_pos = (df_flags == 1).sum()
    keep = n_pos[n_pos >= min_cases].index.tolist()
    dropped = [c for c in df_flags.columns if c not in keep]
    df_flags = df_flags[keep]
    print(f"\nMin-cases threshold ({min_cases}): kept {len(keep)}, dropped {len(dropped)}: {dropped}")

    print("\nPrevalence summary:")
    for col in sorted(df_flags.columns):
        pos = int((df_flags[col] == 1).sum())
        total = int(df_flags[col].notna().sum())
        pct = 100 * pos / total if total > 0 else 0
        print(f"  {col:<45s}  {pos:>6,} / {total:>6,}  ({pct:.1f}%)")

    df_flags.index.name = "eid"
    df_flags.to_csv(out)
    print(f"\nSaved → {out}  ({len(df_flags):,} subjects × {len(df_flags.columns)} diseases)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    main(min_cases=args.min_cases, out=args.out)
