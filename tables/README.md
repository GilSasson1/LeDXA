# tables/

Curated, **de-identified cohort-level aggregate** result tables (no per-participant rows) — the
manuscript's numbered Supplementary Tables. Figure-regeneration inputs and exploratory/ablation
sweeps are intentionally not shipped; the figures themselves are provided as rendered PDFs in
[`../figures/`](../figures), and the code that produced them lives in `plotting/`, `downstream/`,
and `model/`.

## Manuscript supplementary tables

| File | Supplementary Table |
|---|---|
| `table_1_hpp_disease.csv` | 1 — HPP prevalent-disease classification (AUROC) |
| `table_2_ukbb_disease.csv` | 2 — UKBB prevalent-disease classification (AUROC) |
| — | 3 — HPP disease organ-system grouping; not included because the source mapping is unavailable |
| `table_4_incident_disease.csv` | 4 — UKBB incident-disease discrimination (C-index) |
| `table_5_gwas_hits.csv` | 5 — LeDXA embedding GWAS hits |
| `table_6_medications.csv` | 6 — Paired biological-age-gap changes by ATC-3 medication group |
| `table_7_cluster_phenotypes.csv` | 7 — Matched female-cluster phenotypic differences |
| `table_8_cluster_omics.csv` | 8 — Female-cluster multi-omics hits |
| `table_9_dxa_features.csv` | 9 — HPP and UK Biobank DXA feature dictionary |

Table 3 remains an open manuscript item. No placeholder file is kept: it should be added only when
the diagnosis-to-organ-system mapping and case counts are available. Table 6 contains the complete
paired ATC-3 analysis (27 medication groups), including the HRT result in women and the pooled
antidepressant result.

Every file here is cohort-level aggregate with no participant rows. Run `python tools/check_no_pii.py`
before adding new outputs.
