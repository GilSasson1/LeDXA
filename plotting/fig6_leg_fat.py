"""plot_fig6_leg_fat.py

Fig 6 — Low-leg-fat body composition phenotype in women.

Two-panel layout (Nature double-column, 7.09 in wide):
  (a) PC1 × PC4 scatter for female subjects coloured by leg-fat residual.
      Confirms the DXA foundation model captures the fat distribution axis
      without being given leg-fat labels.
  (b) Forest plot: health associations of the low-leg-fat phenotype vs
      matched controls (same total fat, age, height; n=572 cases, 1:2).
      Systems: body composition → liver → blood lipids → cardiovascular
               → glycaemic → sleep.

Run from the DEXA root:
    python -m dexa_fm.hpp.clustering.plot_fig6_leg_fat
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault('MPLCONFIGDIR', '$HOME/.cache/tmp/matplotlib')

import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from downstream.clustering import paths
from downstream.clustering.style import apply_paper_rcparams, add_panel_letter

warnings.filterwarnings('ignore')

W_FULL    = 7.09
ROW_SHADE = '#F5F5F5'
C_HIGH    = '#d73027'   # red  = higher in low-leg-fat group (worse)
C_LOW     = '#2166ac'   # blue = higher in high-leg-fat group (protective)

FOREST_SYSTEMS = [
    ('body_composition',      'Body comp.'),
    ('liver',                 'Liver'),
    ('blood_tests_lipids',    'Blood lipids'),
    ('cardiovascular_system', 'Cardiovascular'),
    ('glycemic_status',       'Glycaemic'),
    ('sleep',                 'Sleep'),
]
MAX_PER_SYS = 2


def _prettify(feature: str) -> str:
    import re
    renames = {
        'total_scan_vat_volume':                       'VAT volume',
        'total_scan_vat_area':                         'VAT area',
        'body_comp_legs_fat_mass':                     'Leg fat mass',
        'body_comp_android_fat_mass':                  'Android fat',
        'body_comp_trunk_fat_mass':                    'Trunk fat',
        'body_comp_gynoid_fat_mass':                   'Gynoid fat',
        'bt__triglycerides':                           'Triglycerides',
        'bt__hdl_cholesterol':                         'HDL cholesterol',
        'bt__non_hdl_cholesterol':                     'Non-HDL cholesterol',
        'bt__total_cholesterol':                       'Total cholesterol',
        'liver_sound_speed':                           'Liver sound speed',
        'liver_attenuation':                           'Liver attenuation',
        'liver_viscosity':                             'Liver viscosity',
        'standing_three_min_blood_pressure_systolic':  'Systolic BP (3 min)',
        'standing_one_min_blood_pressure_systolic':    'Systolic BP (1 min)',
        'from_l_thigh_to_l_ankle_pwv':                 'Thigh–ankle PWV',
        'iglu_sddm':                                   'Glucose SD (CGM)',
        'bt__glucose':                                 'Fasting glucose',
        'iglu_1st_quartile':                           'Glucose 1st quartile',
        'ahi_during_supine':                           'AHI (supine)',
        'rdi_during_supine':                           'RDI (supine)',
        'rdi':                                         'RDI (overall)',
    }
    if feature in renames:
        return renames[feature]
    label = re.sub(r'^body_comp_', '', str(feature))
    label = re.sub(r'^body_', '', label)
    label = re.sub(r'^bt__', '', label)
    return label.replace('_', ' ').title()[:36]


# ── Panel (a): embedding scatter ───────────────────────────────────────────────

def _panel_scatter(ax: plt.Axes, fem_pcs: pd.DataFrame,
                   matched_pcs: pd.DataFrame) -> None:
    """PC1 × PC4 scatter of female subjects coloured by leg-fat residual."""
    rng  = np.random.default_rng(0)
    n    = min(3000, len(fem_pcs.dropna(subset=['leg_fat_resid'])))
    subs = fem_pcs.dropna(subset=['leg_fat_resid']).sample(n=n, random_state=0)

    v1 = subs['leg_fat_resid'].quantile(0.02)
    v2 = subs['leg_fat_resid'].quantile(0.98)

    sc = ax.scatter(
        subs['PC1'], subs['PC4'],
        c=subs['leg_fat_resid'],
        cmap='RdYlGn', vmin=v1, vmax=v2,
        s=4, alpha=0.45, linewidths=0, rasterized=True, zorder=2,
    )

    # Overlay matched cases (low leg fat) and controls with outlines
    low  = matched_pcs[matched_pcs['group'] == 'low_leg_fat']
    high = matched_pcs[matched_pcs['group'] == 'high_leg_fat']
    ax.scatter(low['PC1'],  low['PC4'],  s=6, facecolors='none',
               edgecolors='#d73027', linewidths=0.5, alpha=0.6, zorder=3,
               label=f'Low leg fat  (n={len(low):,})')
    ax.scatter(high['PC1'], high['PC4'], s=6, facecolors='none',
               edgecolors='#2166ac', linewidths=0.5, alpha=0.6, zorder=3,
               label=f'High leg fat  (n={len(high):,})')

    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
    cb.set_label('Leg fat residual\n(SD from expected)', fontsize=6)
    cb.ax.tick_params(labelsize=6)

    ax.legend(fontsize=6, frameon=False, loc='upper right',
              markerscale=1.5, handlelength=1.0)
    ax.set_xlabel('PC 1  (body size)', fontsize=8)
    ax.set_ylabel('PC 4  (fat distribution)', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    # Annotate PC4 separation
    d = abs(low['PC4'].mean() - high['PC4'].mean())
    ax.text(0.03, 0.97, f'PC4 separation = {d:.1f} SD',
            transform=ax.transAxes, fontsize=6, va='top', color='#444',
            style='italic')


# ── Panel (b): forest ──────────────────────────────────────────────────────────

def _panel_forest(ax: plt.Axes, results: pd.DataFrame) -> list:
    df = results.drop_duplicates(['system', 'feature']).copy()
    df = df[df['adj_p_value'] < 0.05].copy()
    df['abs_d'] = df['delta_z'].abs()
    df = df[~df['feature'].astype(str).str.contains('_diff', case=False)]

    entries = []
    sys_map = {s: lbl for s, lbl in FOREST_SYSTEMS}
    for sys, _ in FOREST_SYSTEMS:
        if sys not in df['system'].values:
            continue
        top = df[df['system'] == sys].nlargest(MAX_PER_SYS, 'abs_d')
        for _, r in top.iterrows():
            entries.append({
                'system':    sys,
                'sys_label': sys_map.get(sys, sys),
                'label':     _prettify(str(r['feature'])),
                'd':         float(r['delta_z']),
                'ci_lo':     float(r['ci_low']),
                'ci_hi':     float(r['ci_high']),
            })

    if not entries:
        ax.text(0.5, 0.5, 'No significant results', ha='center', va='center',
                transform=ax.transAxes)
        return []

    n = len(entries)
    sys_order = list(dict.fromkeys(e['system'] for e in entries))
    for i, e in enumerate(entries):
        if sys_order.index(e['system']) % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=ROW_SHADE, zorder=0, lw=0)

    for i, e in enumerate(entries):
        d, lo, hi = e['d'], e['ci_lo'], e['ci_hi']
        c = C_HIGH if d > 0 else C_LOW
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, alpha=0.55, zorder=2)
        ax.plot([0, d],   [i, i], color=c, lw=2.0, solid_capstyle='round',
                alpha=0.85, zorder=3)
        ax.scatter(d, i, color=c, s=45, zorder=4, edgecolors='none')

    prev_sys = None
    for i, e in enumerate(entries):
        if e['system'] != prev_sys and prev_sys is not None:
            ax.axhline(i - 0.5, color='#BBBBBB', lw=0.8, zorder=1)
        prev_sys = e['system']

    sys_mids: dict[str, list] = {}
    for i, e in enumerate(entries):
        sys_mids.setdefault(e['system'], []).append(i)
    for sys, ys in sys_mids.items():
        ax.text(1.01, np.mean(ys), sys_map.get(sys, sys), fontsize=6,
                transform=ax.get_yaxis_transform(), va='center', ha='left',
                color='#444', clip_on=False)

    ax.axvline(0, color='#555', lw=1.0, zorder=3)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([e['label'] for e in entries], fontsize=7)
    ax.set_ylim(-0.6, n + 0.1)
    xabs = max(abs(v) for e in entries for v in [e['ci_lo'], e['ci_hi'], 0]) * 1.18
    ax.set_xlim(-xabs, xabs)
    ax.set_xlabel("Cohen's $d$  (low vs high leg fat, adj. $p$ < 0.05)", fontsize=8)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='y', visible=False)

    return [
        Line2D([0], [0], color=C_HIGH, lw=2.5,
               label='Higher in low-leg-fat group'),
        Line2D([0], [0], color=C_LOW,  lw=2.5,
               label='Higher in high-leg-fat group'),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def compose_fig6() -> None:
    print("Loading data...")
    fem_pcs     = pd.read_csv(paths.out_table('leg_fat_female_pcs.csv'),
                              index_col=[0, 1])
    matched_pcs = pd.read_csv(paths.out_table('leg_fat_matched_pcs.csv'),
                              index_col=[0, 1])
    results     = pd.read_csv(paths.out_table('tableS_leg_fat_phenotype.csv'))

    scatter_h = 2.9
    forest_h  = max(3.5, len(FOREST_SYSTEMS) * MAX_PER_SYS * 0.26 + 1.0)
    apply_paper_rcparams()
    fig = plt.figure(figsize=(W_FULL, scatter_h + forest_h + 0.5))

    outer = mgs.GridSpec(2, 1, figure=fig,
                         height_ratios=[scatter_h, forest_h],
                         hspace=0.48)

    # (a) scatter — left half of row 1
    row1 = mgs.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                       wspace=0.40)
    ax_a = fig.add_subplot(row1[0])
    ax_b = fig.add_subplot(outer[1])

    # ── (a) Embedding scatter ─────────────────────────────────────────────────
    _panel_scatter(ax_a, fem_pcs, matched_pcs)
    ax_a.set_title(
        'DXA embedding captures fat\ndistribution (female subjects)',
        fontsize=7, pad=3,
    )
    add_panel_letter(ax_a, 'a')

    # ── (a right) leave blank or add a body comp bar chart ───────────────────
    # Show key body comp deltas to define the phenotype concisely
    ax_bar = fig.add_subplot(row1[1])
    bc_features = [
        ('body_comp_legs_fat_mass',  'Leg fat'),
        ('body_comp_gynoid_fat_mass','Gynoid fat'),
        ('body_comp_android_fat_mass','Android fat'),
        ('body_comp_trunk_fat_mass', 'Trunk fat'),
        ('total_scan_vat_volume',    'VAT volume'),
    ]
    sig = results[results['adj_p_value'] < 0.05].drop_duplicates('feature')
    feat_d = {r['feature']: r['delta_z'] for _, r in sig.iterrows()}
    labels_bar = [lbl for _, lbl in bc_features]
    vals_bar   = [feat_d.get(feat, np.nan) for feat, _ in bc_features]
    colors_bar = [C_HIGH if v > 0 else C_LOW for v in vals_bar]
    y_pos      = np.arange(len(labels_bar))
    ax_bar.barh(y_pos, vals_bar, color=colors_bar, alpha=0.85, height=0.6)
    ax_bar.axvline(0, color='#444', lw=0.8)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels_bar, fontsize=8)
    ax_bar.set_xlabel("Cohen's $d$  (low vs high leg fat)", fontsize=8)
    ax_bar.set_title('Body composition at\nmatched total fat mass', fontsize=7, pad=3)
    ax_bar.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.6)
    ax_bar.grid(axis='y', visible=False)
    ax_bar.tick_params(axis='x', labelsize=7)
    for spine in ['top', 'right']:
        ax_bar.spines[spine].set_visible(False)
    add_panel_letter(ax_bar, 'b')

    # ── (b) Forest ────────────────────────────────────────────────────────────
    h = _panel_forest(ax_b, results)
    ax_b.set_title(
        'Health associations of the low-leg-fat phenotype  '
        '(n=572 cases, 1:2 matched; same total fat, age, height)',
        fontsize=7.5, loc='left', pad=4,
    )
    add_panel_letter(ax_b, 'c')
    if h:
        ax_b.legend(handles=h, loc='lower right', fontsize=6.5, frameon=False)

    stem = paths.out_figure('fig6_leg_fat')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=400, facecolor='white',
                    bbox_inches='tight')
        print(f"Saved {stem}.{ext}")
    plt.close(fig)


if __name__ == '__main__':
    compose_fig6()
