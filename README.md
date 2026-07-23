# LeDXA

**A self-supervised foundation model for whole-body dual-energy X-ray absorptiometry (DXA) scans**

Official implementation of *LeDXA: a self-supervised foundation model for dual-energy X-ray
absorptiometry* — **Sasson et al. (2026)**

📄 **Paper:** coming soon <!-- Replace with: [Paper](MANUSCRIPT_URL) --> · [Citation](#citation)

[![LeDXA pretraining and downstream applications](assets/readme/overview.png)](assets/figure1.pdf)


## About

LeDXA uses **LeJEPA**, a joint-embedding predictive architecture, to learn from the spatial
structure of whole-body DXA scans. It was pretrained from scratch on **11,540 unlabeled HPP
scans** and evaluated externally on **47,400 UK Biobank scans**.

The frozen representation supports prevalent-disease and biomarker prediction, incident-disease
survival analysis, biological-age estimation, embedding GWAS, and unsupervised body-composition
phenotyping. This repository provides the model and training code, embedding extraction, a synthetic
smoke test, de-identified aggregate results, plotting code, and rendered manuscript figures.

## Model architecture

| Property | Value |
|---|---|
| Encoder | ViT-Small/16 (`vit_small_patch16_384`) |
| Parameters | **21,664,128** in the deployed encoder; **26,788,288** during pretraining with the projection head |
| Input | Bone and tissue DXA views processed separately at 384 × 128 pixels; each grayscale view is replicated across three channels |
| Patch sequence | 192 image patches (24 × 8) plus one class token |
| Projection head | 384 → 2,048 → 2,048 → 64; used only during pretraining |
| Representation | 384 dimensions per view; 768 dimensions when bone and tissue embeddings are concatenated for late fusion |

## Results

### Cross-cohort disease prediction

LeDXA representations retain strong discrimination across an internal HPP test set and the external
UK Biobank cohort, including cardiometabolic, musculoskeletal, hematological, and endocrine
conditions.

[![Selected HPP and UK Biobank disease-prediction results](assets/readme/cross_cohort_prediction.png)](figures/fig2_disease_heatmap.pdf)

### Prospective disease risk

Frozen LeDXA embeddings improve incident-disease prediction beyond demographic covariates and
scanner-derived DXA measurements, with particularly strong gains for hip and knee arthrosis and
type-2 diabetes. The preview shows these selected headline outcomes; click it for all evaluated
endpoints in Figure 3.

[![Selected incident hip arthrosis, knee arthrosis, and type-2 diabetes outcomes](assets/readme/incident_disease.png)](figures/fig3_cox_survival.pdf)

### Biological age and mortality

LeDXA predicts chronological age across HPP and UK Biobank. The resulting biological-age gap
stratifies subsequent mortality: participants in the oldest-appearing quartile have higher adjusted
mortality risk than those in the youngest-appearing quartile.

[![Biological-age prediction and mortality association](assets/readme/biological_age.png)](figures/fig5_biological_age.pdf)

## Quick start

LeDXA requires Python 3.10 or newer.

```bash
git clone https://github.com/GilSasson1/LeDXA.git
cd LeDXA
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m sample_data.demo
```

This command builds a randomly initialized encoder and runs a synthetic batch through it. It checks
that the installation and tensor shapes are correct; it does **not** produce trained LeDXA
embeddings. No pretrained checkpoint is currently distributed with the repository.

Expected output:

```text
input (2, 3, 384, 128) -> features (2, 384) -> projections (2, 128)
```

Dependencies are declared in `pyproject.toml`; `requirements.txt` is provided for compatibility.

## Repository layout

```text
LeDXA/
├── model/          architecture, datasets, augmentation, training, embedding extraction
├── downstream/     disease, survival, biological-age, clustering, and genetics
├── plotting/       manuscript figure generation
├── tables/         de-identified aggregate results and figure inputs
├── figures/        rendered manuscript figures
└── sample_data/    participant-free synthetic smoke test
```

Shared utilities and metadata support these main directories, while paths for controlled data and
outputs are configured through [`config.py`](config.py). Detailed data and output descriptions are in
[`data/README.md`](data/README.md), [`tables/README.md`](tables/README.md), and
[`figures/README.md`](figures/README.md).

## Adapting LeDXA to other data

The included data loader reflects the HDF5 structure used for this study and is not intended as a
universal DXA format. For another dataset, adapt [`model/datasets.py`](model/datasets.py) to provide
separate bone and tissue views; the encoder, pretraining loop, and embedding-extraction code can then
be reused. The study-specific pipeline is retained as a reference implementation in `model/`.

## Figures and reproducibility

The repository includes de-identified aggregate tables and the rendered main figures. Click any
preview above or use the links below for the complete publication-quality PDF.

| Figure | Scientific result | Public reproduction |
|---|---|---|
| [Figure 1](assets/figure1.pdf) | Study design and model overview | Rendered asset included |
| [Figure 2](figures/fig2_disease_heatmap.pdf) | Disease and physiological-trait prediction | Aggregate inputs included in `tables/` |
| [Figure 3](figures/fig3_cox_survival.pdf) | Incident-disease survival analysis | Render included; curves require participant-level follow-up data |
| [Figure 4](figures/fig4_genetics.pdf) | Embedding GWAS and SNP heritability | Render included; full regeneration requires external GWAS outputs |
| [Figure 5](figures/fig5_biological_age.pdf) | Biological age, health, and mortality | Render included; full regeneration requires participant-level predictions |
| [Figure 6](figures/fig6_female_clusters.pdf) | Body-composition phenotype discovery | Render included; UMAP regeneration requires participant-level embeddings |

Figure 2 can be regenerated from the committed aggregate inputs:

```bash
python -m plotting.fig2_heatmap
```

The remaining plotting scripts are retained as analysis provenance, but some require controlled
cohort inputs or institution-specific data adapters that cannot be distributed publicly. Aggregate
result tables contain no participant-level rows; run `python tools/check_no_pii.py` before publishing
new outputs.

## Data and model availability

No participant-level data or pretrained checkpoint is distributed in this repository. Researchers
can request data access from [UK Biobank](https://www.ukbiobank.ac.uk/) and the
[Human Phenotype Project](https://humanphenotypeproject.org/). The model and analysis code are
provided for adaptation to authorized DXA datasets.

## Citation

```bibtex
@article{ledxa,
  title  = {LeDXA: a self-supervised foundation model for dual-energy X-ray absorptiometry},
  author = {Sasson, Gil and others},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

This project is released under the [MIT License](LICENSE).
