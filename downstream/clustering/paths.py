"""Canonical paths for the female-clustering section.

All tables → ``dexa_fm/tables/``  (tableD_cluster_*)
All figures → ``dexa_fm/figures/`` (fig6_* / supp_fig6_*)
"""
import os

DEXA_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DEXA_FM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TABLES_DIR  = os.path.join(DEXA_FM_ROOT, 'tables')
FIGURES_DIR = os.path.join(DEXA_FM_ROOT, 'figures')

# Working dir: where characterize_female_clusters.py writes its outputs.
WORK_DIR = os.path.join(DEXA_ROOT, 'volcano_plots', 'systems', 'women')

# Draft file (for auditing and numbering fixes).
PAPER_DRAFT = os.path.join(DEXA_ROOT, 'paper_draft.md')


def out_table(name: str) -> str:
    os.makedirs(TABLES_DIR, exist_ok=True)
    return os.path.join(TABLES_DIR, name)


def out_figure(name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    return os.path.join(FIGURES_DIR, name)


# ── Publish maps ──────────────────────────────────────────────────────────────
# Source path (relative to DEXA_ROOT) → destination filename in TABLES_DIR.
CSV_PUBLISH_MAP = {
    'volcano_plots/systems/women/cluster_summary.csv':           'tableD_cluster_summary.csv',
    'volcano_plots/systems/women/disease_bias_per_condition.csv':'tableD_cluster_disease_bias.csv',
    'volcano_plots/systems/women/disease_bias_sign_test.csv':    'tableD_cluster_disease_sign_test.csv',
    'volcano_plots/systems/women/disease_burden.csv':            'tableD_cluster_disease_burden.csv',
    'volcano_plots/systems/women/disease_burden_by_domain.csv':  'tableD_cluster_disease_burden_by_domain.csv',
    'volcano_plots/systems/women/family_history_burden.csv':     'tableD_cluster_family_history_burden.csv',
    'volcano_plots/systems/women/all_results.csv':               'tableD_cluster_all_results.csv',
    'volcano_plots/systems/women/supp_table_fig5_per_system.csv':'tableD_cluster_forest_per_system.csv',
    'volcano_plots/systems/women/supp_table_fig5_per_target.csv':'tableD_cluster_forest_per_target.csv',
}

# Source path (relative to DEXA_ROOT) → destination filename in FIGURES_DIR.
FIG_PUBLISH_MAP = {
    'volcano_plots/systems/women/figure_composite.pdf': 'supp_fig6_composite_legacy.pdf',
    'volcano_plots/systems/women/figure_composite.png': 'supp_fig6_composite_legacy.png',
    'volcano_plots/systems/women/forest_plot.pdf':      'supp_fig6_forest_detailed.pdf',
    'volcano_plots/systems/women/forest_plot.png':      'supp_fig6_forest_detailed.png',
    'volcano_plots/systems/women/figure_disease.pdf':   'supp_fig6_disease_bars.pdf',
    'volcano_plots/systems/women/figure_disease.png':   'supp_fig6_disease_bars.png',
}
