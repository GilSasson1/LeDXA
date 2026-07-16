# data/

Place your own data here. **Nothing in this directory is distributed** — it is
git-ignored, and the UK Biobank / Human Phenotype Project data used in the paper are
access-controlled and cannot be shared.

Expected layout (override any path via environment variables — see `config.py`):

```
data/
├── hpp/dxa_dataset.h5              # HPP DXA images (LEDXA_HPP_DXA_H5)
├── ukbb/ukbb_dexa_dataset_v3.h5    # UK Biobank DXA images (LEDXA_UKBB_DXA_H5)
├── checkpoints/                    # trained model weights (LEDXA_CHECKPOINTS)
├── embeddings/                     # extracted frozen embeddings (LEDXA_EMBEDDINGS)
└── gwas_analysis/                  # GWAS summary stats / annotations (LEDXA_GWAS)
```

Data access:
- **UK Biobank** — via approved application at https://www.ukbiobank.ac.uk/
- **Human Phenotype Project** — via https://humanphenotypeproject.org/
