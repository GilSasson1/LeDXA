"""Canonical paths for the biological-age section.

All tables → ``dexa_fm/tables/``  (alongside ``tableA_*`` / ``tableB_*`` / ``tableC_*``)
All figures → ``dexa_fm/figures/`` (alongside ``fig2_*`` / ``fig3_*``)
"""
import os

DEXA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEXA_FM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TABLES_DIR = os.path.join(DEXA_FM_ROOT, 'tables')
FIGURES_DIR = os.path.join(DEXA_FM_ROOT, 'figures')

# Inputs that the bioage pipeline reads — these are existing artifacts.
PRED_CSV = os.environ.get('BIOAGE_PRED_CSV', os.path.join(DEXA_ROOT, 'age_prediction_ukbb', 'ukbb_age_predictions_with_visits.csv'))
EVENTS_CSV = os.path.join(DEXA_ROOT, 'ukbb_osteo_data_expanded_aligned.csv')
TABULAR_CSV = os.path.join(DEXA_ROOT, 'ukbb_tabular_data_for_cox_with_baseline.csv')
TARGETS_CSV = os.path.join(DEXA_FM_ROOT, 'ukbb', 'ukbb_targets_slim.csv')
HPP_ANALYSIS_DIR = os.path.join(DEXA_FM_ROOT, 'age_prediction_analysis')
PAPER_DRAFT = os.path.join(DEXA_ROOT, 'paper_draft.md')

# Working / cache files (large, not part of paper outputs)
WORK_DIR = os.path.join(DEXA_ROOT, 'age_prediction_ukbb')


def out_table(name: str) -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    return os.path.join(TABLES_DIR, name)


def out_figure(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)
