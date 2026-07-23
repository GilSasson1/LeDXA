"""
Prepare GWAS phenotype files for three modalities:
  1. DINOv3 fusion embeddings
  2. LeDXA fusion embeddings
  3. DXA tabular measurements

All outputs share the same format:
  Index: eid (str)
  Columns: sex, age, <feature_0>, <feature_1>, ...

All cohort-specific locations are command-line inputs. Run embedding extraction
first, then pass the resulting fusion pickles to this template.
"""

import os
import pandas as pd

# Columns to exclude from tabular (admin/metadata, not phenotypes)
TABULAR_EXCLUDE_KEYWORDS = [
    "date", "month", "year", "assessment centre", "dxa measuring method",
    "dxa measurement completed", "believed safe", "dxa images",
]


def load_age_sex(tabular_csv: str) -> pd.DataFrame:
    """Load age and sex from tabular CSV, indexed by eid (str)."""
    tab = pd.read_csv(
        tabular_csv,
        usecols=["eid", "Age when attended assessment centre - visit 2", "Sex - visit 0"],
    )
    tab = tab.rename(columns={
        "Age when attended assessment centre - visit 2": "age",
        "Sex - visit 0": "sex",
    })
    tab["eid"] = tab["eid"].astype(str)
    tab = tab.set_index("eid")
    tab = tab.dropna(subset=["age", "sex"])
    print(f"Age/sex table: {len(tab)} subjects with both age and sex")
    return tab


def embeddings_to_gwas(emb_path: str, age_sex: pd.DataFrame, out_path: str, name: str):
    """Load fusion embeddings pickle, filter to visit 2, join age/sex, save."""
    print(f"\n{'='*60}\nProcessing {name}\n{'='*60}")
    df = pd.read_pickle(emb_path)
    print(f"  Loaded: {df.shape}, index names: {df.index.names}")

    # Filter to visit 2
    visit_level = df.index.names[1]  # 'visit' or 'visit_index'
    v2 = df.loc[df.index.get_level_values(visit_level).astype(str) == "2"].copy()
    print(f"  Visit 2 only: {len(v2)} rows")

    # Drop visit level → single eid index
    v2.index = v2.index.get_level_values(0).astype(str)
    v2.index.name = "eid"

    # Join age/sex
    merged = age_sex[["sex", "age"]].join(v2, how="inner")
    print(f"  After joining age/sex: {len(merged)} rows, {merged.shape[1]} columns")

    merged.to_pickle(out_path)
    csv_path = out_path.replace(".pkl", ".csv")
    merged.to_csv(csv_path)
    print(f"  Saved pkl: {out_path}")
    print(f"  Saved csv: {csv_path}")
    return merged


def tabular_to_gwas(age_sex: pd.DataFrame, tabular_csv: str, out_path: str):
    """Load tabular CSV, filter to DXA phenotype columns, join age/sex, save."""
    print(f"\n{'='*60}\nProcessing DXA Tabular\n{'='*60}")
    tab = pd.read_csv(tabular_csv)
    tab["eid"] = tab["eid"].astype(str)
    tab = tab.set_index("eid")

    # Keep only visit-2 DXA measurement columns (exclude admin/metadata)
    keep_cols = []
    for col in tab.columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in TABULAR_EXCLUDE_KEYWORDS):
            continue
        # Also skip sex and age (we add them from age_sex)
        if col in ("Sex - visit 0", "Age when attended assessment centre - visit 2"):
            continue
        keep_cols.append(col)

    tab = tab[keep_cols]
    print(f"  Kept {len(keep_cols)} DXA measurement columns")

    # Join age/sex (inner join keeps only subjects with age+sex)
    merged = age_sex[["sex", "age"]].join(tab, how="inner")

    # Require ≥10% of DXA columns non-null — subjects with almost no tabular
    # data shouldn't enter GWAS even with imputation.
    dxa_cols = [c for c in merged.columns if c not in ("sex", "age")]
    dxa_completeness = merged[dxa_cols].notna().mean(axis=1)
    merged = merged[dxa_completeness >= 0.10]
    print(f"  After ≥10% DXA completeness filter: {len(merged)} rows, {merged.shape[1]} columns")

    merged.to_pickle(out_path)
    csv_path = out_path.replace(".pkl", ".csv")
    merged.to_csv(csv_path)
    print(f"  Saved pkl:  {out_path}")
    print(f"  Saved csv:  {csv_path}")
    return merged


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lejepa-embeddings", required=True,
                    help="Visit-indexed LeDXA fusion embedding pickle.")
    ap.add_argument("--dino-embeddings", required=True,
                    help="Visit-indexed DINOv3 fusion embedding pickle.")
    ap.add_argument("--tabular-csv", required=True,
                    help="Cohort table containing eid, visit-2 age/sex, and DXA measurements.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory for GWAS-ready pickle and CSV files.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Load age & sex
    age_sex = load_age_sex(args.tabular_csv)

    # Step 2: Process DINO → GWAS format
    dino_gwas = embeddings_to_gwas(
        args.dino_embeddings, age_sex,
        os.path.join(args.out_dir, "dino_huge_plus_gwas.pkl"),
        "DINO Huge+",
    )

    # Step 3: Process LeJEPA → GWAS format
    lejepa_gwas = embeddings_to_gwas(
        args.lejepa_embeddings, age_sex,
        os.path.join(args.out_dir, "lejepa_gwas.pkl"),
        "LeJEPA",
    )

    # Step 4: Process Tabular → GWAS format
    tabular_gwas = tabular_to_gwas(
        age_sex, args.tabular_csv,
        os.path.join(args.out_dir, "dxa_tabular_gwas.pkl"),
    )

    # Step 5: Verify consistency
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    for name, df in [("DINO Huge+", dino_gwas), ("LeJEPA", lejepa_gwas), ("DXA Tabular", tabular_gwas)]:
        print(f"  {name:15s}: {df.shape[0]:6d} rows × {df.shape[1]:4d} cols | "
              f"cols[0:3] = {df.columns[:3].tolist()}")

    # EID overlap
    d_eids = set(dino_gwas.index)
    l_eids = set(lejepa_gwas.index)
    t_eids = set(tabular_gwas.index)
    all_three = d_eids & l_eids & t_eids
    print(f"\n  EID overlap:")
    print(f"    DINO ∩ LeJEPA:       {len(d_eids & l_eids)}")
    print(f"    DINO ∩ Tabular:      {len(d_eids & t_eids)}")
    print(f"    LeJEPA ∩ Tabular:    {len(l_eids & t_eids)}")
    print(f"    All three:           {len(all_three)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
