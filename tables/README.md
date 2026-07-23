# tables/

Curated, **de-identified cohort-level aggregate** result tables (no per-participant rows).
Stripped to two categories only: the manuscript's numbered Supplementary Tables, and the files a
plotting script actually reads to regenerate a figure. Everything else (sensitivity/ablation
sweeps, superseded duplicates, unreferenced exploratory-analysis exports) has been removed.

## Manuscript supplementary tables

`supp_table_N_*.csv` are the manuscript's numbered Supplementary Tables. Three of them
(1, 2, 4) are also direct inputs to `plotting/fig2_heatmap.py` / `plotting/fig3_cox.py` — for
those, this is the one and only copy (no separate "pipeline" file with a different name).

| File | Supplementary Table |
|---|---|
| `supp_table_1_hpp_prevalent_disease_classification_auroc.csv` | 1 — HPP prevalent-disease classification (AUROC). Read by `plotting/fig2_heatmap.py`. |
| `supp_table_2_ukbb_prevalent_disease_classification_auroc.csv` | 2 — UKBB prevalent-disease classification (AUROC). Read by `plotting/fig2_heatmap.py`. |
| `supp_table_3_hpp_disease_organ_system_grouping_PENDING.csv` | 3 — HPP disease organ-system grouping **[PENDING]** |
| `supp_table_4_ukbb_incident_disease_cox_discrimination_c_index.csv` | 4 — UKBB incident-disease Cox discrimination (C-index). Written by `plotting/fig3_cox.py`. |
| `supp_table_5_ledxa_embedding_gwas_hits.csv` | 5 — LeDXA embedding GWAS hits (genome-wide and suggestive) |
| `supp_table_6_atc3_medication_response_PENDING.csv` | 6 — Paired biological-age-gap change by ATC-3 medication class and sex **[PENDING]** |
| `supp_table_7_female_embedding_cluster_matched_phenotypic_differences.csv` | 7 — Female embedding-cluster matched phenotypic differences (currently DXA body-composition + bone-density only, 299 rows; non-DXA physiological systems not yet merged in) |
| `supp_table_8_female_embedding_cluster_multi_omics_hits.csv` | 8 — Female embedding-cluster multi-omics hits |
| `supp_table_9_dxa_tabular_feature_dictionary_hpp_and_uk_biobank.csv` | 9 — DXA tabular feature dictionary (HPP and UK Biobank) |

Tables 3 and 6 are pending — the diagnosis→organ-system-category mapping (3) and the reconciled
ATC-3 medication run (6) aren't available in either repo yet. Their files are stubs, not data.
See `supplement/SUPPLEMENT_MANIFEST.md` (in the parent `DEXA/` project) for the specifics.

Dropped from the supplement entirely (values already stated in the main text, Fig. 2a/2b):
continuous organ-system biomarker Pearson r, chronological-age-accuracy. The original Table 6/7
split (matched body-composition vs. a separately-computed "multi-system summary") was consolidated
to one table (7): the dropped one was computed on the *unmatched* cohort, contradicting the
paper's own matching methodology, and didn't contain the non-DXA phenotypes its title promised.

## Figure-pipeline inputs (not themselves numbered supp tables)

| File | Used by |
|---|---|
| `age_mae_imaging_only.csv` | `plotting/fig5_bioage.py`, `downstream/bioage/age_mae.py` |
| `age_mae_imaging_only_wholebody.csv` | `plotting/fig2_heatmap.py` (Fig. 2a) |
| `age_prediction_analysis/medication_sexsplit_paired_gap.csv` | `plotting/fig5_bioage.py` (Fig. 5h) |
| `cox_ttest_results_bp_logsweep_nodxapca_perseed.csv` | `plotting/fig3_cox.py` |
| `disease_pairwise_diffpentuned.csv` | `plotting/fig2_heatmap.py`; written by `downstream/disease/topset.py` |
| `fig4c/fig4c_associations.tsv`, `fig4c/fig4c_primary.tsv` | `plotting/fig4_genetics.py`, `downstream/genetics/build_fig4c.py` |
| `tableD_bioage_disease_prevalence_extended.csv` | `plotting/fig5_bioage.py` (Fig. 5g) |
| `tableD_bioage_medication_paired_atc3.csv` | `plotting/fig5_bioage.py` |
| `tableD_bioage_phenotype_female.csv`, `tableD_bioage_phenotype_male.csv` | `plotting/fig5_bioage.py` (Fig. 5c, d) |
| `tableD_cluster_all_results.csv` | `plotting/fig6_clustering.py` |
| `tableE_bioage_gap_mortality_cox.csv` | `plotting/fig5_bioage.py` (Fig. 5e, f) |

Note: many of these files back specific numbers quoted in the main text (e.g. bio-age medication
or mortality figures) beyond just feeding a plot — that's incidental to why they're kept here,
which is strictly that a script reads them. Anything not read by a script and not a numbered
Supplementary Table was removed, even where it was the only source for a reported number.
