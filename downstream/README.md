# Downstream analysis templates

This directory contains the analysis logic needed to reproduce the paper's
model comparisons after an authorized cohort has been converted to local
tables and LeDXA embeddings. It intentionally does not contain UK Biobank/HPP
downloaders, participant-level data builders, cluster job launchers, or
machine-specific paths.

Paths can be supplied directly on the command line. Shared defaults resolve
under `data/` and can be overridden with the environment variables documented
in [`config.py`](../config.py).

## Directory guide

- `disease/` — linear-probe, covariate-adjusted, fine-tuning, and UK Biobank
  disease-classification comparisons. Inputs are embedding pickles, a target
  CSV, and—when requested—a DXA tabular-feature CSV.
- `survival/` — cross-validated Cox models comparing LeDXA, DINOv3, DXA
  tabular measurements, and demographic covariates. The event table must
  already contain baseline dates and endpoint dates.
- `bioage/` — chronological-age prediction, regression-to-the-mean
  detrending, and mortality association of the resulting biological-age gap.
- `genetics/` — conversion of visit-indexed embeddings into GWAS phenotype
  matrices and the auditable GWAS-Catalog/LD annotation used for Figure 4c.
- `clustering/` — shared repository-relative paths and visual conventions for
  the Figure 6 generator. Participant-level UMAP and omics analyses remain
  controlled inputs.

## Example commands

Covariate-adjusted disease comparison:

```bash
python -m downstream.disease.linear_probe_cov \
  --embeddings-dir data/embeddings/hpp \
  --targets-csv data/hpp/downstream_targets.csv \
  --tabular-csv data/hpp/dxa_tabular.csv \
  --cls-auto-detect
```

Incident-outcome comparison:

```bash
python -m downstream.survival.cox_regression \
  --events-path data/ukbb/incident_events.csv \
  --tabular-path data/ukbb/dxa_tabular.csv \
  --lejepa-path data/embeddings/ukbb/lejepa_fusion.pkl \
  --dino-path data/embeddings/ukbb/dino_fusion.pkl
```

GWAS phenotype preparation:

```bash
python -m downstream.genetics.gwas_phenotypes \
  --lejepa-embeddings data/embeddings/ukbb/lejepa_fusion.pkl \
  --dino-embeddings data/embeddings/ukbb/dino_fusion.pkl \
  --tabular-csv data/ukbb/dxa_tabular.csv \
  --out-dir data/gwas_analysis/phenotypes
```

These commands are templates: column names and cohort indexing must match the
study schemas described in each module. Association testing itself is run with
the user's approved genotype-analysis environment; this repository provides
the phenotype preparation and downstream figure/annotation logic.
