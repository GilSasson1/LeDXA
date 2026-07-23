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


def _v2_by_eid(path: str):
    """Load an (eid, visit[_index]) embedding pkl, keep visit 2, index by eid(str)."""
    df = pd.read_pickle(path)
    vlvl = df.index.names[1]
    v2 = df.loc[df.index.get_level_values(vlvl).astype(str) == "2"].copy()
    v2.index = v2.index.get_level_values(0).astype(str)
    v2.index.name = "eid"
    return v2[~v2.index.duplicated(keep="first")]


def embeddings_to_gwas(emb_path: str, age_sex: pd.DataFrame, out_path: str, name: str,
                       region_path: str = None):
    """Load fusion embeddings pickle, filter to visit 2, join age/sex, save.
    If region_path given, append the mean-pooled regional (femur+lumbar) block
    (columns r*) so the GWAS PCA sees bone+tissue+regional."""
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

    if region_path:
        rg = _v2_by_eid(region_path).reindex(merged.index)
        n_nan = int(rg.isna().all(axis=1).sum())
        rg = rg.fillna(rg.mean())                      # impute missing-regional eids (PCA needs no NaN)
        merged = pd.concat([merged, rg], axis=1)
        print(f"  + regional block: +{rg.shape[1]} cols -> {merged.shape[1]} total "
              f"({n_nan} eids w/o regional, mean-imputed)")

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
    ap.add_argument("--region-dir",
                    help="Directory containing optional *_regionpool.pkl or bonepool fusion files.")
    ap.add_argument("--region-pool", action="store_true",
                    help="Append the femur+lumbar regional block (r*) to the embedding GWAS "
                         "phenotypes; writes _regionpool-suffixed files. Tabular has no regional "
                         "block and is left as-is.")
    ap.add_argument("--bone-pool", action="store_true",
                    help="Use the bone-pool fusion (regional mean-pooled INTO the bone view, no "
                         "separate block; same dim as whole-body). Writes _bonepool-suffixed files. "
                         "Tabular has no embedding pooling and is left as-is.")
    args = ap.parse_args()
    if args.region_pool and args.bone_pool:
        raise SystemExit("--region-pool and --bone-pool are mutually exclusive")
    os.makedirs(args.out_dir, exist_ok=True)
    if (args.region_pool or args.bone_pool) and not args.region_dir:
        ap.error("--region-dir is required with --region-pool or --bone-pool")

    # Resolve source fusion paths + suffix by regime.
    bonepool_dir = os.path.join(args.region_dir, "bonepool") if args.region_dir else None
    if args.bone_pool:
        sfx = "_bonepool"
        dino_path = os.path.join(bonepool_dir, "dino_fusion.pkl")
        lej_path  = os.path.join(bonepool_dir, "lejepa_fusion.pkl")
        rp_dino = rp_lej = None        # regional already folded into the bone view
    else:
        sfx = "_regionpool" if args.region_pool else ""
        dino_path = args.dino_embeddings
        lej_path  = args.lejepa_embeddings
        rp_dino = os.path.join(args.region_dir, "dino_regionpool.pkl") if args.region_pool else None
        rp_lej = os.path.join(args.region_dir, "lejepa_regionpool.pkl") if args.region_pool else None

    # Step 1: Load age & sex
    age_sex = load_age_sex(args.tabular_csv)

    # Step 2: Process DINO → GWAS format
    dino_gwas = embeddings_to_gwas(
        dino_path, age_sex,
        os.path.join(args.out_dir, f"dino_huge_plus_gwas{sfx}.pkl"),
        "DINO Huge+", region_path=rp_dino,
    )

    # Step 3: Process LeJEPA → GWAS format
    lejepa_gwas = embeddings_to_gwas(
        lej_path, age_sex,
        os.path.join(args.out_dir, f"lejepa_gwas{sfx}.pkl"),
        "LeJEPA", region_path=rp_lej,
    )

    if args.region_pool or args.bone_pool:
        regime = "region-pool" if args.region_pool else "bone-pool"
        print(f"\n[{regime}] DINO + LeJEPA embedding GWAS phenotypes written ({sfx}); "
              "tabular is embedding-agnostic, existing dxa_tabular_gwas stands.")
        return

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
