# tables/

Curated, **de-identified cohort-level aggregate** result tables (no per-participant rows) — the
manuscript's numbered Supplementary Tables. Figure-regeneration inputs and exploratory/ablation
sweeps are intentionally not shipped; the figures themselves are provided as rendered PDFs in
[`../figures/`](../figures), and the code that produced them lives in `plotting/`, `downstream/`,
and `model/`.

## Manuscript supplementary tables

| File | Supplementary Table |
|---|---|
| `supp_table_1_hpp_prevalent_disease_classification_auroc.csv` | 1 — HPP prevalent-disease classification (AUROC) |
| `supp_table_2_ukbb_prevalent_disease_classification_auroc.csv` | 2 — UKBB prevalent-disease classification (AUROC) |
| `supp_table_3_hpp_disease_organ_system_grouping_PENDING.csv` | 3 — HPP disease organ-system grouping **[PENDING]** |
| `supp_table_4_ukbb_incident_disease_cox_discrimination_c_index.csv` | 4 — UKBB incident-disease Cox discrimination (C-index) |
| `supp_table_5_ledxa_embedding_gwas_hits.csv` | 5 — LeDXA embedding GWAS hits (genome-wide and suggestive) |
| `supp_table_6_atc3_medication_response_PENDING.csv` | 6 — Paired biological-age-gap change by ATC-3 medication class and sex **[PENDING]** |
| `supp_table_7_female_embedding_cluster_matched_phenotypic_differences.csv` | 7 — Female embedding-cluster matched phenotypic differences (DXA body-composition + bone-density, 299 rows) |
| `supp_table_8_female_embedding_cluster_multi_omics_hits.csv` | 8 — Female embedding-cluster multi-omics hits |
| `supp_table_9_dxa_tabular_feature_dictionary_hpp_and_uk_biobank.csv` | 9 — DXA tabular feature dictionary (HPP and UK Biobank) |

Tables 3 and 6 are pending — the diagnosis→organ-system-category mapping (3) and the reconciled
ATC-3 medication run (6) are not available yet; those files are stubs, not data.

Every file here is cohort-level aggregate with no participant rows. Run `python tools/check_no_pii.py`
before adding new outputs.
