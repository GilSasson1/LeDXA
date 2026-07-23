# data/

Place your own data here. **Nothing in this directory is distributed** — it is
git-ignored, and the UK Biobank / Human Phenotype Project data used in the paper are
access-controlled and cannot be shared.

Expected layout (override any path via environment variables — see `config.py`):

```
data/
├── hpp/dxa_dataset.h5              # HPP DXA images (LEDXA_HPP_DXA_H5)
├── hpp/age_targets.csv             # HPP training targets (LEDXA_HPP_TARGETS_CSV)
├── hpp/downstream_targets.csv      # HPP probe targets (LEDXA_HPP_DOWNSTREAM_TARGETS_CSV)
├── ukbb/ukbb_dexa_dataset_v3.h5    # UK Biobank DXA images (LEDXA_UKBB_DXA_H5)
├── ukbb/age_targets.csv             # columns: eid, visit, age (LEDXA_UKBB_TARGETS_CSV)
├── ukbb/dxa_tabular.csv             # DXA measurements and analysis covariates
├── ukbb/incident_events.csv         # prepared baseline and endpoint dates
├── ukbb/baseline_disease_targets.csv # prepared dis__ classification outcomes
├── checkpoints/                    # trained model weights (LEDXA_CHECKPOINTS)
├── embeddings/                     # extracted frozen embeddings (LEDXA_EMBEDDINGS)
├── results/                        # local analysis outputs (LEDXA_RESULTS)
└── gwas_analysis/                  # GWAS summary stats / annotations (LEDXA_GWAS)
```

The prepared cohort tables are inputs to the public analysis templates; the
source-specific extraction and data-release process is deliberately not
included.

Data access:
- **UK Biobank** — via approved application at https://www.ukbiobank.ac.uk/
- **Human Phenotype Project** — via https://humanphenotypeproject.org/
