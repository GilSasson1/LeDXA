"""Central path configuration for LeDXA.

All participant-level data lives OUTSIDE this repository and is **not** distributed
(UK Biobank and Human Phenotype Project data are access-controlled). Point the paths
below at your own data via environment variables, or edit the defaults.

Public scripts read paths from this module or explicit command-line arguments; no
machine-specific study paths are required.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Root for large inputs (DXA image HDF5s, embeddings, cohort tables). Never committed.
DATA_ROOT = Path(os.environ.get("LEDXA_DATA", REPO_ROOT / "data"))

# Key inputs — override via environment variables to match your setup.
HPP_DXA_H5      = Path(os.environ.get("LEDXA_HPP_DXA_H5",  DATA_ROOT / "hpp"  / "dxa_dataset.h5"))
UKBB_DXA_H5     = Path(os.environ.get("LEDXA_UKBB_DXA_H5", DATA_ROOT / "ukbb" / "ukbb_dexa_dataset_v3.h5"))
HPP_TARGETS_CSV = Path(os.environ.get("LEDXA_HPP_TARGETS_CSV", DATA_ROOT / "hpp" / "age_targets.csv"))
UKBB_TARGETS_CSV = Path(os.environ.get("LEDXA_UKBB_TARGETS_CSV", DATA_ROOT / "ukbb" / "age_targets.csv"))
HPP_DOWNSTREAM_TARGETS_CSV = Path(
    os.environ.get("LEDXA_HPP_DOWNSTREAM_TARGETS_CSV", DATA_ROOT / "hpp" / "downstream_targets.csv")
)
CHECKPOINTS_DIR = Path(os.environ.get("LEDXA_CHECKPOINTS", DATA_ROOT / "checkpoints"))
EMBEDDINGS_DIR  = Path(os.environ.get("LEDXA_EMBEDDINGS",  DATA_ROOT / "embeddings"))
GWAS_DIR        = Path(os.environ.get("LEDXA_GWAS",        DATA_ROOT / "gwas_analysis"))
RESULTS_DIR     = Path(os.environ.get("LEDXA_RESULTS",     DATA_ROOT / "results"))
LEJEPA_CHECKPOINT = Path(
    os.environ.get("LEDXA_CHECKPOINT", CHECKPOINTS_DIR / "hpp" / "best_model.pth")
)

# Repo-relative outputs (curated, de-identified aggregate results — safe to commit).
TABLES_DIR  = REPO_ROOT / "tables"
FIGURES_DIR = REPO_ROOT / "figures"


def out_table(name: str) -> str:
    TABLES_DIR.mkdir(exist_ok=True)
    return str(TABLES_DIR / name)


def out_figure(name: str) -> str:
    FIGURES_DIR.mkdir(exist_ok=True)
    return str(FIGURES_DIR / name)
