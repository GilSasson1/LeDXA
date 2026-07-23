# tables/

Curated, **de-identified cohort-level aggregate** result tables (no per-participant rows).
These feed the figure scripts. For the figure → script → table provenance, see the
["Reproducing the figures"](../README.md#reproducing-the-figures) section of the top-level README.

Participant-level inputs and stale sweep/ablation runs have been removed for release.

## Manuscript supplementary tables

`supp_table_N_*.csv` are exact copies of the manuscript's numbered Supplementary Tables (as
assembled in `supplementary_tables.xlsx`), included here for direct reference alongside the code.
They are not themselves read by any plotting script — the pipeline files they were built from
(e.g. `supp_tableA_disease_auc_4arm_diffpentuned.csv`, `ukbb_pca0_diffpen_summary.csv`) remain in
this directory unchanged and are what `plotting/*.py` actually reads.

| File | Supplementary Table |
|---|---|
| `supp_table_1_hpp_prevalent_disease_classification_auroc.csv` | 1 — HPP prevalent-disease classification (AUROC) |
| `supp_table_2_ukbb_prevalent_disease_classification_auroc.csv` | 2 — UKBB prevalent-disease classification (AUROC) |
| `supp_table_3_hpp_disease_organ_system_grouping_PENDING.csv` | 3 — HPP disease organ-system grouping **[PENDING]** |
| `supp_table_4_ukbb_incident_disease_cox_discrimination_c_index.csv` | 4 — UKBB incident-disease Cox discrimination (C-index) |
| `supp_table_5_ledxa_embedding_gwas_hits.csv` | 5 — LeDXA embedding GWAS hits (genome-wide and suggestive) |
| `supp_table_6_atc3_medication_response_PENDING.csv` | 6 — Paired biological-age-gap change by ATC-3 medication class and sex **[PENDING]** |
| `supp_table_7_female_embedding_cluster_matched_phenotypic_differences.csv` | 7 — Female embedding-cluster matched phenotypic differences (currently DXA body-composition + bone-density only, 299 rows; non-DXA physiological systems not yet merged in — see `SUPPLEMENT_MANIFEST.md`) |
| `supp_table_8_female_embedding_cluster_multi_omics_hits.csv` | 8 — Female embedding-cluster multi-omics hits |
| `supp_table_9_dxa_tabular_feature_dictionary_hpp_and_uk_biobank.csv` | 9 — DXA tabular feature dictionary (HPP and UK Biobank) |

Tables 3 and 6 are pending — the diagnosis→organ-system-category mapping (3) and the reconciled
ATC-3 medication run (6) aren't available in either repo yet. Their files are stubs, not data. See
`supplement/SUPPLEMENT_MANIFEST.md` (in the parent `DEXA/` project) for the specifics.

The continuous organ-system biomarker (Pearson r) and chronological-age-accuracy tables were
dropped — those exact values are already stated in the main text (Fig. 2a, 2b), so a dedicated
supplementary table was redundant. The original Table 6/7 split (matched body-composition vs. a
separately-computed "multi-system summary") was consolidated to one table: the second one was
computed on the *unmatched* cohort (contradicting the paper's own matching methodology) and
didn't actually contain the non-DXA phenotypes its title promised — dropped rather than kept.
