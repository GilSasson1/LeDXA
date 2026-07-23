"""Configurable inputs and repository-relative outputs for biological age."""
import os
from config import DATA_ROOT, FIGURES_DIR as _FIGURES_DIR, TABLES_DIR as _TABLES_DIR

TABLES_DIR = str(_TABLES_DIR)
FIGURES_DIR = str(_FIGURES_DIR)

PRED_CSV = os.environ.get(
    'BIOAGE_PRED_CSV', str(DATA_ROOT / 'ukbb' / 'age_predictions_with_visits.csv'))
EVENTS_CSV = os.environ.get(
    'BIOAGE_EVENTS_CSV', str(DATA_ROOT / 'ukbb' / 'incident_events.csv'))


def out_table(name: str) -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    return os.path.join(TABLES_DIR, name)


def out_figure(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)
