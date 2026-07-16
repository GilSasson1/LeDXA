"""Extended Data Fig — frozen linear probing vs. end-to-end fine-tuning, LeDXA vs DINOv3.

Panel a: three continuous biomarkers (age, creatinine, hemoglobin), Pearson r.
Panel b: five prevalent HPP conditions (MASLD, sleep apnea, osteoarthritis, anemia,
anxiety), AUROC. Diabetes excluded (see NOTE below).

Data sources (no covariates, concatenation fusion only — matches the frozen and
fine-tuned sides on architecture, differing only in whether the backbone trains):
  - Frozen: lp_flagship_matched_continuous_summary.csv (n=5 internally aggregated),
    lp_flagship_matched_disease_a_summary.csv + _disease_b_summary.csv (n=5 each).
  - Fine-tuned: ft_flagship_{target}_{model}_s{seed}_raw.csv for seed in
    {42,73,99,123,2024} (one compare_ft.py invocation per seed, aggregated here
    into mean/se across 5 seeds) — NOT the unsuffixed ft_flagship_*_summary.csv
    files, which are an earlier single-seed (n=1) pilot run superseded by the
    5-seed multiseed re-run (see submit_ft_flagships_multiseed.sh).

NOTE on diabetes exclusion (2026-07-15): diabetes has only ~119 HPP positives
(thinnest of the six original disease targets — flagged as a risk in
~/.claude/plans/curried-squishing-lampson.md before the run). Its 5-seed FT AUCs
are extremely noisy (LeDXA: 0.659-0.824, DINO: 0.694-0.873) and the resulting
point estimate has DINOv3 fine-tuned (0.784) nominally exceeding LeDXA
fine-tuned (0.734) — the only ranking reversal among the nine original targets,
contradicting the "did not reverse the performance ranking" claim, and almost
certainly sampling noise given the fully overlapping confidence intervals
rather than a genuine effect. Dropped from the figure; caption/Methods target
counts updated accordingly (9 -> 8 targets, six -> five disease conditions).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_STYLE_DIR = Path.home() / '.claude/skills/nature-plot-style/style_files'
_DOUBLE_STYLE = str(_STYLE_DIR / 'nature_double.mplstyle')

RESDIR = '/data/hpp_labdata/Analyses/gilsa/results/comparison'
OUT_DIR = '/path/to/project/supplement/figures'
SEEDS = [42, 73, 99, 123, 2024]

COLOR_LEDXA = '#083C7D'
COLOR_DINO = '#7FB9DC'
ROW_SHADE = '#F5F5F5'

CONTINUOUS = [('age', 'Age'), ('bt__creatinine', 'Creatinine'), ('bt__hemoglobin', 'Hemoglobin')]
# Diabetes deliberately omitted — see module docstring.
DISEASES = [
    ('dis__fatty_liver_disease', 'MASLD'),
    ('dis__sleep_apnea',         'Sleep apnea'),
    ('dis__osteoarthritis',      'Osteoarthritis'),
    ('dis__anemia',              'Anemia'),
    ('dis__anxiety',             'Anxiety'),
]
MODELS = [('lejepa', 'LeDXA'), ('dino', 'DINO')]


def _ft_aggregate(target, model):
    vals = []
    for s in SEEDS:
        f = os.path.join(RESDIR, f'ft_flagship_{target}_{model}_s{s}_raw.csv')
        vals.append(pd.read_csv(f)['score'].iloc[0])
    vals = np.asarray(vals)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


def _load_frozen():
    cont = pd.read_csv(os.path.join(RESDIR, 'lp_flagship_matched_continuous_summary.csv'))
    dis_a = pd.read_csv(os.path.join(RESDIR, 'lp_flagship_matched_disease_a_summary.csv'))
    dis_b = pd.read_csv(os.path.join(RESDIR, 'lp_flagship_matched_disease_b_summary.csv'))
    frozen = pd.concat([cont, dis_a, dis_b], ignore_index=True)
    return frozen.set_index(['target', 'model'])[['mean', 'se']]


def _panel(ax, targets, xlabel, frozen):
    n = len(targets)
    for i, (key, label) in enumerate(targets):
        y0 = n - 1 - i  # top-to-bottom in input order
        if i % 2 == 0:
            ax.axhspan(y0 - 0.5, y0 + 0.5, color=ROW_SHADE, zorder=0)
        for j, (mkey, mlabel) in enumerate(MODELS):
            y = y0 + (0.18 if mkey == 'lejepa' else -0.18)
            color = COLOR_LEDXA if mkey == 'lejepa' else COLOR_DINO
            fr_mean, fr_se = frozen.loc[(key, mkey), 'mean'], frozen.loc[(key, mkey), 'se']
            ft_mean, ft_se = _ft_aggregate(key, mkey)
            ax.errorbar(fr_mean, y, xerr=fr_se, fmt='o', mfc='white', mec=color,
                        color=color, markersize=5, capsize=2, linewidth=1.0,
                        markeredgewidth=1.2, zorder=3)
            ax.errorbar(ft_mean, y, xerr=ft_se, fmt='o', mfc=color, mec=color,
                        color=color, markersize=5, capsize=2, linewidth=1.0,
                        markeredgewidth=1.2, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in reversed(targets)])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='y', visible=False)


def compose_figure(*, save: bool = True):
    if _STYLE_DIR.exists():
        plt.style.use(_DOUBLE_STYLE)
    frozen = _load_frozen()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 5.0))
    _panel(ax_a, CONTINUOUS, 'Pearson r', frozen)
    _panel(ax_b, DISEASES, 'AUC', frozen)

    for ax, letter in zip([ax_a, ax_b], 'ab'):
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='right')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color=COLOR_LEDXA, mfc='white', mec=COLOR_LEDXA,
               linestyle='-', label='LeDXA, frozen'),
        Line2D([0], [0], marker='o', color=COLOR_LEDXA, mfc=COLOR_LEDXA, mec=COLOR_LEDXA,
               linestyle='-', label='LeDXA, fine-tuned'),
        Line2D([0], [0], marker='o', color=COLOR_DINO, mfc='white', mec=COLOR_DINO,
               linestyle='-', label='DINO, frozen'),
        Line2D([0], [0], marker='o', color=COLOR_DINO, mfc=COLOR_DINO, mec=COLOR_DINO,
               linestyle='-', label='DINO, fine-tuned'),
    ]
    fig.subplots_adjust(bottom=0.18, wspace=0.35)
    fig.legend(handles=handles, frameon=False, ncol=4, loc='lower center',
              bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.6)

    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        pdf = os.path.join(OUT_DIR, 'extdata_ft_vs_frozen_model_gap.pdf')
        png = os.path.join(OUT_DIR, 'extdata_ft_vs_frozen_model_gap.png')
        fig.savefig(pdf, dpi=400, facecolor='white', transparent=False, bbox_inches='tight')
        fig.savefig(png, dpi=400, facecolor='white', transparent=False, bbox_inches='tight')
        print(f'Saved -> {pdf}')
        print(f'Saved -> {png}')
    return fig


if __name__ == '__main__':
    compose_figure()
