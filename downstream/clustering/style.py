"""Style helpers for the female-clustering section (Figure 6).

Cluster colours adopt the paper-wide MODEL_COLORS palette:
  Cluster A (adipose, larger group) → tabular green  (MODEL_COLORS['tabular'], #8ccbb3)
  Cluster B (lean, protective)      → lejepa blue    (#2166ac)

apply_paper_rcparams() and add_panel_letter() are re-exported from the bioage
style module so there is exactly one source of truth for Fig 2/3/5/6 styling.
"""
from downstream.bioage.style import apply_paper_rcparams, add_panel_letter  # noqa: F401
from common.plot_style import MODEL_COLORS                           # noqa: F401

__all__ = [
    'CLUSTER_COLORS', 'CLUSTER_NAMES',
    'MODEL_COLORS',
    'apply_paper_rcparams', 'add_panel_letter',
]

# Kept fixed (green A / blue B) independent of MODEL_COLORS, which now uses a
# saturated-hero / light-baseline palette for the model-comparison figures.
CLUSTER_COLORS = {
    'A': MODEL_COLORS['tabular'],   # adipose cluster — tabular green (#8ccbb3), matched to Fig 2
    'B': '#2166ac',                 # lean/protective cluster — blue
}

CLUSTER_NAMES = {
    'A': 'Cluster A (adipose)',
    'B': 'Cluster B (lean)',
}
