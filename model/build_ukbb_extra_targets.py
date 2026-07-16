"""
Build extra targets for Figure 3 external validation:
  1. Cancer incidence from the UKBB cancer register (fields 40005/40006).
     Extracts ALL C-code prefixes dynamically; also builds colorectal (C18+C19+C20).
     Output date columns follow the existing event naming convention so the Cox
     pipeline picks them up automatically:
       "Date cancer <label> - visit 0"
  2. Concurrent visit-2 biomarkers saved to ukbb_extra_biomarkers.csv
     (for the downstream regression pipeline).

Outputs:
  ukbb_cancer_events.csv      -- indexed by eid; "Date cancer * - visit 0" columns
  ukbb_extra_biomarkers.csv   -- indexed by eid; grip / spirometry values
"""

import argparse

import numpy as np
import pandas as pd

PARQUET = "/data/ukbb_raw/Data/ukb676772.parquet"
EVENTS_FILE = "/path/to/project/ukbb_osteo_data_expanded_aligned.csv"
CANCER_OUT = "/path/to/project/ukbb_cancer_events.csv"
BIO_OUT = "/path/to/project/ukbb_extra_biomarkers.csv"

N_CANCER_SLOTS = 10   # array instances 0..9 in the parquet

# Grouped / combined endpoints added on top of individual C-code prefixes
COMBINED_GROUPS = {
    "colorectal (C18-C20)": ["C18", "C19", "C20"],
    "any cancer (C00-C97)": [f"C{i:02d}" for i in range(98)],
}

# ICD10 3-char descriptions (for readable column names)
ICD10_LABELS = {
    "C00": "lip", "C01": "tongue_base", "C02": "tongue_other",
    "C03": "gum", "C04": "floor_mouth", "C05": "palate",
    "C06": "mouth_other", "C07": "parotid", "C08": "salivary",
    "C09": "tonsil", "C10": "oropharynx", "C11": "nasopharynx",
    "C12": "pyriform_sinus", "C13": "hypopharynx", "C14": "lip_oral_pharynx_other",
    "C15": "oesophagus", "C16": "stomach", "C17": "small_intestine",
    "C18": "colon", "C19": "rectosigmoid", "C20": "rectum",
    "C21": "anus", "C22": "liver", "C23": "gallbladder",
    "C24": "bile_duct", "C25": "pancreas", "C26": "gi_other",
    "C30": "nasal_cavity", "C31": "sinuses", "C32": "larynx",
    "C33": "trachea", "C34": "lung", "C37": "thymus",
    "C38": "heart_mediastinum", "C40": "bone_limb", "C41": "bone_other",
    "C43": "melanoma", "C44": "skin_nonmelanoma", "C45": "mesothelioma",
    "C46": "kaposi", "C47": "peripheral_nerve", "C48": "retroperitoneum",
    "C49": "soft_tissue", "C50": "breast", "C51": "vulva",
    "C52": "vagina", "C53": "cervix", "C54": "uterus",
    "C55": "uterus_unspec", "C56": "ovary", "C57": "female_genital_other",
    "C58": "placenta", "C60": "penis", "C61": "prostate",
    "C62": "testis", "C63": "male_genital_other", "C64": "kidney",
    "C65": "renal_pelvis", "C66": "ureter", "C67": "bladder",
    "C68": "urinary_other", "C69": "eye", "C70": "meninges",
    "C71": "brain", "C72": "cns_other", "C73": "thyroid",
    "C74": "adrenal", "C75": "endocrine_other", "C76": "ill_defined",
    "C77": "lymph_node_secondary", "C78": "respiratory_secondary",
    "C79": "other_secondary", "C80": "unknown_primary",
    "C81": "hodgkin", "C82": "follicular_lymphoma", "C83": "diffuse_lymphoma",
    "C84": "t_nk_cell_lymphoma", "C85": "nhl_other", "C86": "nhl_subtype",
    "C88": "malignant_immunoproliferative", "C90": "myeloma",
    "C91": "lymphoid_leukemia", "C92": "myeloid_leukemia",
    "C93": "monocytic_leukemia", "C94": "other_leukemia", "C95": "leukemia_unspec",
    "C96": "lymphoid_histiocytic_other", "C97": "multiple_primary",
}

# Grip strength, spirometry, and blood count field IDs (visit 2, array 0)
EXTRA_BIO_FIELDS = {
    # Grip strength
    "Hand grip strength (left) - visit 2":  "46-2.0",
    "Hand grip strength (right) - visit 2": "47-2.0",
    # Spirometry
    "Forced vital capacity (FVC) - visit 2":                  "3062-2.0",
    "Forced expiratory volume in 1-second (FEV1) - visit 2":  "3063-2.0",
    "Peak expiratory flow (PEF) - visit 2":                   "3064-2.0",
    # Blood count (visit 2 — sparse ~7.7% but independent of DXA)
    "White blood cell (leukocyte) count - visit 2":           "30000-2.0",
    "Red blood cell (erythrocyte) count - visit 2":           "30010-2.0",
    "Haemoglobin concentration - visit 2":                    "30020-2.0",
    "Haematocrit percentage - visit 2":                       "30030-2.0",
    "Mean corpuscular volume - visit 2":                      "30040-2.0",
    "Platelet count - visit 2":                               "30080-2.0",
    "Lymphocyte count - visit 2":                             "30120-2.0",
    "Neutrophill count - visit 2":                            "30140-2.0",
    "Monocyte count - visit 2":                               "30130-2.0",
    "Eosinophill count - visit 2":                            "30150-2.0",
}


def load_baseline_v2(events_file: str) -> pd.Series:
    df = pd.read_csv(events_file,
                     usecols=["eid", "Date of attending assessment centre - visit 2"],
                     index_col="eid")
    baseline = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    return baseline[baseline.notna()]


def extract_cancer(baseline_v2: pd.Series) -> pd.DataFrame:
    date_cols = [f"40005-{i}.0" for i in range(N_CANCER_SLOTS)]
    icd_cols  = [f"40006-{i}.0" for i in range(N_CANCER_SLOTS)]

    raw = pd.read_parquet(PARQUET, columns=["eid"] + date_cols + icd_cols).set_index("eid")
    raw = raw.loc[raw.index.isin(baseline_v2.index)]
    print(f"  Cancer register: {len(raw)} DXA subjects")

    # Collect all unique 3-char C-code prefixes seen in the data
    all_codes: set = set()
    for i in range(N_CANCER_SLOTS):
        col = raw[f"40006-{i}.0"].dropna().astype(str)
        codes = col.str[:3]
        all_codes.update(codes[codes.str.match(r"^C\d{2}$")].unique())

    print(f"  Unique cancer C-codes found: {sorted(all_codes)}")

    # first_prospective_date[code] -> Series indexed by eid
    first: dict = {}  # key -> pd.Series of datetime64

    def _update(key: str, eid_idx, dates):
        if key not in first:
            first[key] = pd.Series(pd.NaT, index=baseline_v2.index, dtype="datetime64[ns]")
        existing = first[key]
        for eid, d in zip(eid_idx, dates):
            if eid in existing.index:
                if pd.isna(existing[eid]) or d < existing[eid]:
                    existing[eid] = d

    for i in range(N_CANCER_SLOTS):
        slot = raw[[f"40005-{i}.0", f"40006-{i}.0"]].copy()
        slot.columns = ["date", "icd"]
        slot = slot[slot["icd"].notna() & slot["date"].notna()].copy()
        slot["date"] = pd.to_datetime(slot["date"], errors="coerce")
        slot = slot[slot["date"].notna()]
        slot["baseline"] = baseline_v2.reindex(slot.index)
        slot = slot[slot["date"] > slot["baseline"]]
        if slot.empty:
            continue
        slot["code3"] = slot["icd"].astype(str).str[:3]

        # Individual C-codes
        for code in all_codes:
            subset = slot[slot["code3"] == code]
            if not subset.empty:
                _update(code, subset.index.tolist(), subset["date"].tolist())

        # Combined groups
        for grp_name, prefixes in COMBINED_GROUPS.items():
            subset = slot[slot["code3"].isin(prefixes)]
            if not subset.empty:
                _update(grp_name, subset.index.tolist(), subset["date"].tolist())

    # Build output dataframe with Cox-pipeline-compatible column names
    out = pd.DataFrame(index=baseline_v2.index)
    all_keys = sorted(all_codes) + list(COMBINED_GROUPS.keys())
    for key in all_keys:
        if key not in first:
            continue
        dates = first[key]
        label = ICD10_LABELS.get(key, key.lower().replace(" ", "_"))
        col_name = f"Date cancer {label} ({key}) - visit 0"
        out[col_name] = dates
        n = int(dates.notna().sum())
        print(f"  {col_name}: {n} prospective events")

    return out


def extract_biomarkers(baseline_v2: pd.Series) -> pd.DataFrame:
    parquet_cols = list(EXTRA_BIO_FIELDS.values())
    raw = pd.read_parquet(PARQUET, columns=["eid"] + parquet_cols).set_index("eid")
    raw = raw.loc[raw.index.isin(baseline_v2.index)]
    rename = {v: k for k, v in EXTRA_BIO_FIELDS.items()}
    raw.rename(columns=rename, inplace=True)
    print(f"  Biomarkers: {len(raw)} subjects")
    for col in raw.columns:
        n = raw[col].notna().sum()
        print(f"    {col}: {n} non-null ({n/len(raw)*100:.1f}%)")
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer-out", default=CANCER_OUT)
    parser.add_argument("--bio-out", default=BIO_OUT)
    args = parser.parse_args()

    print("Loading baseline V2 dates...")
    baseline_v2 = load_baseline_v2(EVENTS_FILE)
    print(f"  {len(baseline_v2)} subjects with V2 date")

    print("\nExtracting cancer register...")
    cancer_df = extract_cancer(baseline_v2)
    cancer_df.index.name = "eid"
    cancer_df.to_csv(args.cancer_out)
    print(f"  Saved cancer events to {args.cancer_out}")

    print("\nExtracting visit-2 biomarkers (grip + spirometry)...")
    bio_df = extract_biomarkers(baseline_v2)
    bio_df.index.name = "eid"
    bio_df.to_csv(args.bio_out)
    print(f"  Saved biomarkers to {args.bio_out}")


if __name__ == "__main__":
    main()
