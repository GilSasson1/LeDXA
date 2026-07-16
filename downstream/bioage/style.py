"""Plot-style helpers aligned to Fig 2 / Fig 3 of the DEXA paper.

Two-fold purpose:

1. ``apply_paper_rcparams()`` sets matplotlib defaults identical to those
   already used by ``dexa_fm/plot_combined_figure.py`` (sans-serif, fontsize
   11/12, top+right spines off, dashed grid α=0.3).
2. Re-exports ``MODEL_COLORS`` (the canonical palette) plus quartile / sex
   palettes used by the biological-age figures.

Helpers:

* ``add_panel_letter(ax, letter)`` — bold panel letter at the Fig 2/3
  canonical location (top-left, transform=ax.transAxes).
* ``despine(ax)`` — convenience wrapper.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from common.plot_style import MODEL_COLORS, MODEL_LABELS, MODEL_ORDER  # noqa: E402

__all__ = [
    'MODEL_COLORS', 'MODEL_LABELS', 'MODEL_ORDER',
    'PALETTE_QUARTILE', 'PALETTE_SEX',
    'apply_paper_rcparams', 'add_panel_letter',
]

# Quartile palette (Q1 = biologically young → Q4 = biologically old).
# Identical to PALETTE_Q in ukbb_aging_pace_v2v3.py so existing figures
# don't shift hue when the new helper is dropped in.
PALETTE_QUARTILE = {'Q1': '#2471a3', 'Q2': '#76b7c8', 'Q3': '#f0a07a', 'Q4': '#c0392b'}

PALETTE_SEX = {'Female': '#c0392b', 'Male': '#2471a3'}


def apply_paper_rcparams() -> None:
    """Complement nature_double.mplstyle for multi-panel figures.

    Font sizes intentionally small (6/7 pt) — the mplstyle targets the final
    print width (7.09 in / 18 cm).  Larger fonts in 8-panel grids cause text
    collisions once the figure is scaled to page width.
    """
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.size':         6,
        'axes.titlesize':    7,
        'axes.labelsize':    6,
        'xtick.labelsize':   5,
        'ytick.labelsize':   5,
        'legend.fontsize':   5,
        'legend.frameon':    False,
        'figure.dpi':        150,
        'savefig.dpi':       400,
        'pdf.fonttype':      42,
        'ps.fonttype':       42,
        'lines.linewidth':   0.8,
        'axes.linewidth':    0.6,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linestyle':    '--',
    })


def add_panel_letter(ax, letter: str, *, fontsize: int = 12,
                     x: float = -0.14, y: float = 1.03) -> None:
    """Bold panel letter at the canonical Fig 2/3 position."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold',
            va='bottom', ha='left', clip_on=False)
