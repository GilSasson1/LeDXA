import argparse
import re
from typing import Dict, List

import pandas as pd


EXPANDED_EVENT_PATTERNS: Dict[str, List[str]] = {
    # Existing core groups
    "bone_joint": [
        r"Date M80 first reported", r"Date M81 first reported", r"Date M82 first reported",
        r"Date M05 first reported", r"Date M06 first reported", r"Date M16 first reported",
        r"Date M17 first reported", r"Date M19 first reported", r"Date M48 first reported",
        r"Date M51 first reported", r"Date M84 first reported",
    ],
    "neuro": [
        r"Date of all cause dementia report", r"Date of alzheimer's disease report",
        r"Date of all cause parkinsonism report", r"Date F01 first reported",
        r"Date F03 first reported", r"Date G20 first reported", r"Date G30 first reported",
    ],
    "metabolic": [
        r"Date E03 first reported", r"Date E05 first reported", r"Date E10 first reported",
        r"Date E11 first reported", r"Date E66 first reported", r"Date K76 first reported",
    ],
    # New high-priority outcomes
    "mortality": [
        r"date of death", r"underlying \(primary\) cause of death", r"contributory \(secondary\) cause of death",
        r"all-cause mortality", r"mortality"
    ],
    "cardio": [
        r"Date I21 first reported", r"Date I22 first reported", r"Date I20 first reported",
        r"Date I25 first reported", r"Date I50 first reported", r"Date I48 first reported",
        r"Date I63 first reported", r"Date I64 first reported", r"Date I61 first reported",
    ],
    "renal": [
        r"Date N18 first reported", r"Date N17 first reported", r"Date N19 first reported",
    ],
    "respiratory": [
        r"Date J44 first reported", r"Date J45 first reported", r"Date J46 first reported",
    ],
    "cancer": [
        r"Date C[0-9]{2} first reported", r"Date of malignant neoplasm",
    ],
}


def _parse_groups(value: str) -> List[str]:
    parts = [v.strip().lower() for v in value.split(",") if v.strip()]
    if not parts:
        return ["all"]
    return parts


def _build_pattern_list(group_names: List[str]) -> List[str]:
    if any(g == "all" for g in group_names):
        names = list(EXPANDED_EVENT_PATTERNS.keys())
    else:
        names = group_names

    missing = [g for g in names if g not in EXPANDED_EVENT_PATTERNS]
    if missing:
        raise ValueError(f"Unknown outcome groups: {missing}. Available: {list(EXPANDED_EVENT_PATTERNS.keys())}")

    pats: List[str] = []
    for g in names:
        pats.extend(EXPANDED_EVENT_PATTERNS[g])
    return pats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded UKBB Cox events file with baseline V2 anchor.")
    parser.add_argument("--ukbb-fields-file", required=True, help="Path to ukbb fields metadata CSV (must have 'title' column)")
    parser.add_argument("--out-path", default="/path/to/project/ukbb_osteo_data_expanded_aligned.csv")
    parser.add_argument("--groups", default="all", help="Comma list of outcome groups or 'all'")
    parser.add_argument("--add-visit3-date", action="store_true", help="Include Visit 3 assessment date as additional follow-up signal")
    args = parser.parse_args()

    # Late import so this script can be read without LabData env active.
    from LabData.DataLoaders import UkbbLoader

    group_names = _parse_groups(args.groups)
    selected_patterns = _build_pattern_list(group_names)

    ukbb_cols = pd.read_csv(args.ukbb_fields_file)
    if "title" not in ukbb_cols.columns:
        raise ValueError("ukbb fields metadata must contain a 'title' column")

    combined_regex = "|".join(f"(?:{p})" for p in selected_patterns)

    field_list = (
        ukbb_cols.loc[
            ukbb_cols["title"].str.contains(combined_regex, case=False, na=False, regex=True),
            "title",
        ]
        .drop_duplicates()
        .tolist()
    )

    print(f"Matched {len(field_list)} outcome-related fields from metadata.")

    ukb_loader = UkbbLoader.UkbbLoader()

    baseline_v2 = ukb_loader.load_ukbb(
        by="field_name",
        which=["Date of attending assessment centre"],
        instances=[2],
    ).set_index("eid - visit 2")
    baseline_v2.index.name = "eid"

    baseline_date_col = [c for c in baseline_v2.columns if "Date of attending assessment centre" in c][0]
    baseline_v2 = baseline_v2[[baseline_date_col]]

    event_df = ukb_loader.load_ukbb(
        by="field_name",
        which=field_list,
    ).set_index("eid - visit 0")
    event_df.index.name = "eid"

    cox_event_df = baseline_v2.join(event_df, how="left")

    if args.add_visit3_date:
        v3 = ukb_loader.load_ukbb(
            by="field_name",
            which=["Date of attending assessment centre"],
            instances=[3],
        ).set_index("eid - visit 3")
        v3.index.name = "eid"
        v3_cols = [c for c in v3.columns if "Date of attending assessment centre" in c]
        if v3_cols:
            cox_event_df = cox_event_df.join(v3[[v3_cols[0]]], how="left")
            print(f"Added Visit 3 follow-up date column: {v3_cols[0]}")

    cox_event_df.to_csv(args.out_path)

    print("Saved:", args.out_path)
    print("N subjects in aligned event file:", len(cox_event_df))
    print("N with baseline V2 date:", int(cox_event_df[baseline_date_col].notna().sum()))


if __name__ == "__main__":
    main()
