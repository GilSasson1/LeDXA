# LeDXA

**A self-supervised foundation model for whole-body DXA scans.**
Code accompanying the paper *"[LeDXA: a self-supervised foundation model for dual-energy X-ray
absorptiometry]"* — 📄 *manuscript link / DOI: TBD*.

![Figure 1 — LeDXA study overview](assets/figure1.png)

The print-quality version is available as [PDF](assets/figure1.pdf).

## Overview

Whole-body DXA scans are routinely acquired to measure bone density and regional body composition,
leaving their spatial structure largely unused. **LeDXA** shows that self-supervised learning can
convert raw DXA images into general representations of systemic health. It is a vision model based on
a **joint-embedding predictive architecture (JEPA)**, trained by predicting image features in latent
space (rather than reconstructing pixels) and regularized with **SIGReg**. It was trained from scratch
on **11,540 unlabeled DXA scans** from the Human Phenotype Project (HPP) and evaluated internally and
on **47,400 external UK Biobank (UKBB)** scans.

Despite using ~5 orders of magnitude fewer training images and ~40× fewer parameters than DINOv3, the
frozen LeDXA embedding improves cross-cohort prediction of prevalent disease and physiological
biomarkers over scanner-derived DXA readouts and DINOv3; improves longitudinal prediction of incident
disease (notably hip/knee arthrosis and type-2 diabetes); yields a biological-age gap that tracks
disease burden and mortality; and produces an embedding space whose GWAS recovers known
body-composition and bone-density loci with higher SNP-heritability than DINOv3's.

## Architecture

| | |
|---|---|
| Backbone | ViT-Small/16 (`vit_small_patch16_384`, ~22M params) |
| Objective | LeJEPA (joint-embedding predictive) + SIGReg regularizer |
| Input | Whole-body DXA image, 384 × 128 |
| Embedding | 384-dimensional, used **frozen** for all downstream tasks |
| Pretraining corpus | 11,540 HPP DXA scans (unlabeled) |
| Baselines | DINOv3 (ViT-Huge) · scanner-derived tabular DXA measures |

## Repository structure

| Path | Function |
|------|----------|
| `model/` | Model architecture, SIGReg loss, image augmentations, HDF5 datasets, pretraining, and frozen-embedding extraction. |
| `common/` | Code shared by analyses: model factories, statistical helpers, Cox metadata, and the paper-wide plotting style. |
| `downstream/disease/` | Linear probes and fine-tuning for prevalent disease and continuous biomarker prediction. |
| `downstream/survival/` | Incident-event construction and Cox proportional-hazards analyses. |
| `downstream/bioage/` | Biological-age prediction, age-gap/aging-rate analyses, mortality, and medication associations. |
| `downstream/clustering/` | Unsupervised embedding clusters and body-composition phenotype characterization. |
| `downstream/genetics/` | GWAS phenotype preparation, lead-locus annotation, and Figure 4 inputs. |
| `plotting/` | Scripts that convert aggregate tables into main, supplementary, and extended-data figures. |
| `tables/` | De-identified aggregate results and figure inputs. No participant-level rows should be committed here. |
| `figures/` | Rendered paper figures. These are outputs; generation code remains in `plotting/`. |
| `assets/` | Documentation assets used by this README, including the inline Figure 1 PNG and print PDF. |
| `data/` | Git-ignored location for authorized local cohort data, checkpoints, embeddings, and caches. |
| `sample_data/` | Synthetic, participant-free smoke test for the encoder interface. |
| `tools/` | Repository safety utilities, currently the participant-data/PII guard. |
| `csvs/` | Small, non-participant metadata mappings such as disease groups and display names. |
| `config.py` | Central environment-variable-based paths and optional W&B configuration. |

### Reproducibility boundary

The installable model, synthetic smoke test, HPP/UKBB training entry points, and embedding extractor
do not require the authors' private Python packages. Some participant-level cohort-construction
scripts under `downstream/` are retained as analysis provenance and still expect authorized,
institution-specific data adapters (for example medication and raw UKBB field loaders). Those
adapters cannot be distributed with this repository; the committed aggregate tables and rendered
figures remain usable without them.

## Setup

```bash
git clone git@github.com:GilSasson1/LeDXA.git && cd LeDXA
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e .
```

Runtime dependencies have a single source of truth in `pyproject.toml`; `requirements.txt` installs
that project for compatibility with tools that expect a requirements file. `timm>=1.0.20` is required
for the DINOv3 model names used here. DINO weights are downloaded on first use and cached under
`data/hf_cache/` by default; override that with `LEDXA_HF_CACHE`.

## Quick check

```bash
python -m sample_data.demo     # builds the encoder, embeds a synthetic DXA batch
```

Expected output ends with `features (2, 384) -> projections (2, 128)`.

## Training

Pretrain from scratch on your own DXA scans:

```bash
python model/train.py          # LeJEPA, ViT-Small/16, 384×128 inputs
```

DXA images are read from an HDF5 store (see `model/datasets.py` for the expected format) and
augmented per `model/augmentations.py`. Frozen embeddings are then extracted with
`model/extract_embeddings.py` for the downstream analyses.

Paths are configured without editing source code:

```bash
export LEDXA_HPP_DXA_H5=/path/to/hpp_dxa_dataset.h5
export LEDXA_HPP_TARGETS_CSV=/path/to/hpp_age_targets.csv
export LEDXA_UKBB_DXA_H5=/path/to/ukbb_dexa_dataset.h5
export LEDXA_UKBB_TARGETS_CSV=/path/to/ukbb_age_targets.csv
export LEDXA_CHECKPOINTS=/path/to/checkpoints
export LEDXA_EMBEDDINGS=/path/to/embeddings
```

HPP targets used by `model/train.py` follow the dataset index expected by `model/datasets.py`.
UKBB targets used by `model/train_ukbb.py` must contain `eid`, `visit`, and `age` columns. W&B is
disabled by default; set `WANDB_ENTITY` (and optionally `WANDB_PROJECT`/`WANDB_MODE`) to enable it.

## Reproducing the figures

Analyses read **de-identified aggregate tables** from `tables/` (no participant-level data). Main
figures and their scripts:

| Figure | Script | Key inputs (in `tables/`) |
|--------|--------|---------------------------|
| **Fig 2** – disease & trait prediction | `plotting/fig2_heatmap.py` | `supp_tableA_disease_auc_4arm_diffpentuned.csv`, `ukbb_pca0_diffpen_summary.csv`, `disease_pairwise_diffpentuned.csv`, `age_mae_imaging_only_wholebody.csv` |
| **Fig 3** – incident-disease Cox | `plotting/fig3_cox.py` | `cox_ttest_results_bp_logsweep_nodxapca.csv` (+ `_perseed`) |
| **Fig 4** – embedding GWAS | `plotting/fig4_genetics.py`, `downstream/genetics/build_fig4c.py` | `fig4c/*.tsv`; GWAS summary stats *(external)* |
| **Fig 5** – biological age | `plotting/fig5_bioage.py` (via `downstream/bioage/run_section.py`) | `tableD_bioage_*`, `tableE_bioage_gap_mortality_cox.csv` |
| **Fig 6** – body-composition clusters | `plotting/fig6_clustering.py` (via `downstream/clustering/run_section.py`) | `tableD_cluster_*` |
| Fig 1 (schematic) | `plotting/fig1_model_panel.py`, `plotting/fig1_downstream_panel.py` | illustrative |

Disease-classification AUROC tables (Supplementary Tables 1–2) are `tables/tableA_hpp_disease_auc_4arm.csv`
and `tables/tableB_ukbb_disease_auc_4arm.csv`. Embedding-GWAS lead loci (genome-wide + suggestive,
P < 1×10⁻⁶) are in `tables/tableS_gwas_lejepa_hits.tsv`.

## Data availability

No participant-level data is included. Access is via the data owners:
**UK Biobank** (https://www.ukbiobank.ac.uk/) and the **Human Phenotype Project**
(https://humanphenotypeproject.org/). Place your data as described in [`data/README.md`](data/README.md).

## Citation

```bibtex
@article{ledxa,
  title  = {LeDXA: a self-supervised foundation model for dual-energy X-ray absorptiometry},
  author = {TBD},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

[MIT](LICENSE).
