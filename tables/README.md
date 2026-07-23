# tables/

Curated, **de-identified cohort-level aggregate** result tables (no per-participant rows) for the
manuscript's numbered Supplementary Tables. Exploratory results, ablation sweeps, and
participant-level inputs are intentionally not included.

## Manuscript supplementary tables

| File | Supplementary Table |
|---|---|
| `table_1_hpp_disease.csv` | 1 — HPP prevalent-disease classification (AUROC) |
| `table_2_ukbb_disease.csv` | 2 — UKBB prevalent-disease classification (AUROC) |
| `table_3_hpp_disease_systems.csv` | 3 — HPP disease endpoints and organ-system mapping |
| `table_4_incident_disease.csv` | 4 — UKBB incident-disease discrimination (C-index) |
| `table_5_gwas_hits.csv` | 5 — LeDXA embedding GWAS hits |
| `table_6_medications.csv` | 6 — Paired biological-age-gap changes by ATC-3 medication group |
| `table_7_cluster_phenotypes.csv` | 7 — Matched female-cluster phenotypic differences |
| `table_8_cluster_omics.csv` | 8 — Female-cluster multi-omics hits |
| `table_9_dxa_features.csv` | 9 — HPP and UK Biobank DXA feature dictionary |

Table 3 reproduces the 37 chronic-disease endpoints used in the HPP analysis from the LabData
medical-condition metadata. The analyzed `Allergy` endpoint is listed once with both source systems:
the metadata groups allergy diagnoses under Immunology but also consolidates ICD-11 `ED80` (Acne)
into the same endpoint under Dermatology. Its positive count therefore matches the analyzed
composite rather than a retrospectively corrected label.

Table 6 contains the complete paired ATC-3 analysis (27 medication groups), including the HRT result
in women and the pooled antidepressant result.

Every file here is cohort-level aggregate with no participant rows. Run `python tools/check_no_pii.py`
before adding new outputs.
