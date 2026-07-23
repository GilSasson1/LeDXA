"""Figure 6 — Unsupervised female clustering (DEXA paper, Nature Medicine).

Four-panel main figure for the female-cluster results section:

    a  Full-cohort UMAP coloured by sex, with females split into their A/B
       cluster label (shows the female bimodality directly)
    b  Female UMAP coloured by BMI (the matching covariate): the split tracks an
       adiposity gradient, but Cluster A is heterogeneous and overlaps B in BMI —
       the similar-size women that age+BMI matching draws on
    c  Multi-system forest: what differs at matched age and BMI — lean, bone, grip,
       autonomic, haematology, liver, diet, activity (Cohen's d, B − A)
    d  Proteomics volcano (matched cohort): external DE, BH q<0.05

Visual conventions match Figs 2, 3, 5: nature_double style as base,
apply_paper_rcparams() font overrides, 26 pt bold panel letters via
add_panel_letter(), dashed y-grid (#CCCCCC, alpha=0.7), alternating
#F5F5F5 row backgrounds in the forest panel, MODEL_COLORS palette.
"""
from __future__ import annotations

import re
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DATA_ROOT

from downstream.clustering import paths
from downstream.clustering.style import (
    CLUSTER_COLORS, MODEL_COLORS, apply_paper_rcparams, add_panel_letter,
)

ROW_SHADE = '#F5F5F5'

# ── Omics companion data (matched-cohort external DE: unpaired MW + per-layer BH) ──
try:
    from adjustText import adjust_text
    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False

QSIG_OMICS = 0.05   # primary significance threshold (colour + solid sig line + labels)
C_NS = '#b8b8b8'
FEMALE_UMAP_CSV = DATA_ROOT / "clustering" / "female_umap.csv"
FULL_UMAP_CSV = DATA_ROOT / "clustering" / "full_cohort_umap.csv"
OMICS_CSV = DATA_ROOT / "clustering" / "matched_proteomics.csv"

from matplotlib.colors import LinearSegmentedColormap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prettify(feature: str) -> str:
    """Human-readable feature label for the forest panel."""
    feature = re.sub(r'\s*:\s*.*$', '', str(feature))   # strip ': median - TM3'-style suffixes
    renames = {
        'body_comp_legs_fat_mass':         'Legs fat mass',
        'body_comp_arms_fat_mass':         'Arms fat mass',
        'body_comp_leg_right_fat_mass':    'Leg fat (right)',
        'body_comp_leg_left_fat_mass':     'Leg fat (left)',
        'body_comp_gynoid_fat_mass':       'Gynoid fat mass',
        'body_comp_android_fat_mass':      'Android fat mass',
        'body_comp_trunk_fat_mass':        'Trunk fat mass',
        'body_comp_total_fat_mass':        'Total fat mass',
        'body_comp_total_lean_mass':       'Total lean mass',
        'body_comp_arms_lean_mass':        'Arms lean mass',
        'body_comp_trunk_fat_free_mass':   'Trunk fat-free mass',
        'body_pelvis_area':                'Pelvis area',
        'body_pelvis_bmc':                 'Pelvis BMC',
        'liver_sound_speed':               'Liver sound speed',
        'bt__alkaline_phosphatase':        'Alkaline phosphatase',
        'bt__creatinine':                  'Creatinine',
        'bt__neutrophils_abs':             'Neutrophils (abs)',
        'hr_bpm':                          'Resting heart rate',
        'heart_rate_min_during_rem':       'Min HR (REM sleep)',
        'heart_rate_mean_during_wake':     'Mean HR (wake)',
        'standing_one_min_blood_pressure_pulse_rate': 'Pulse rate (1 min)',
        'frailty_index':                   'Frailty index',
        'hand_grip_left':                  'Grip strength',
        'hand_grip_right':                 'Grip strength',
        'body_pelvis_bmc':                 'Pelvis BMC',
        'femur_troch_mean_bmc':            'Femoral trochanter BMC',
        'femur_troch_mean_bmd':            'Femoral trochanter BMD',
        'walking_speed_kmh':               'Walking speed',
        'Walking_speed_kmh':               'Walking speed',
        'heart_rate_mean_during_rem':      'Heart rate (REM sleep)',
        'heart_rate_mean_during_nrem':     'Heart rate (NREM sleep)',
        'heart_rate_mean_during_sleep':    'Heart rate (sleep)',
        'r_r_ms':                          'R–R interval (HRV)',
        'bt__mch':                         'MCH',
        'bt__mchc':                        'MCHC',
        'bt__mcv':                         'MCV',
        'bt__rdw':                         'RDW',
        'bt__hemoglobin':                  'Haemoglobin',
        'physical_activity_maderate_days_a_week': 'Moderate activity (days/wk)',
        'Energy / BMR':                    'Energy intake / BMR',
        'IMEDAS_score_per_day':            'Mediterranean diet score',
    }
    if feature in renames:
        return renames[feature]
    label = re.sub(r'^body_comp_', '', feature)
    label = re.sub(r'^body_', '', label)
    label = re.sub(r'^bt__', '', label)
    label = label.replace('_', ' ').title()
    return label[:38]


def _load_umap():
    """Load participant-level female UMAP coordinates, BMI, and cluster labels."""
    required = {"UMAP1", "UMAP2", "cluster", "bmi"}
    df_f = pd.read_csv(FEMALE_UMAP_CSV)
    missing = required.difference(df_f.columns)
    if missing:
        raise ValueError(f"{FEMALE_UMAP_CSV} is missing columns: {sorted(missing)}")
    if df_f.groupby('cluster')['bmi'].mean().idxmax() != 0:
        df_f['cluster'] = 1 - df_f['cluster']
    df_f['sex_label'] = 'Female'
    return df_f


def _load_fullcohort_umap(female_cluster: pd.DataFrame | None = None):
    """Load a shared full-cohort UMAP with a ``sex_label`` column."""
    required = {"UMAP1", "UMAP2", "sex_label"}
    df = pd.read_csv(FULL_UMAP_CSV)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{FULL_UMAP_CSV} is missing columns: {sorted(missing)}")
    df = df.dropna(subset=['sex_label'])
    return df


# ── Panel builders ────────────────────────────────────────────────────────────

def _panel_a_fullcohort_umap(ax, umap_all: pd.DataFrame) -> None:
    """Full-cohort UMAP coloured by sex — the unfiltered embedding (no DBSCAN
    outlier drop) shows the female population split into two spatially distinct
    islands purely from position, without needing to colour by cluster."""
    sex_palette = {'Male': '#9aa0a6', 'Female': MODEL_COLORS['lejepa']}
    for sex_label, color in sex_palette.items():
        sub = umap_all[umap_all['sex_label'] == sex_label]
        if len(sub) == 0:
            continue
        n_plot = min(800, len(sub))
        samp = sub.sample(n_plot, random_state=42)
        ax.scatter(samp['UMAP1'], samp['UMAP2'], c=color, s=5, alpha=0.55,
                   label=f'{sex_label} (n={len(sub):,})', rasterized=True, linewidths=0)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(frameon=False, markerscale=2.5, loc='lower left', fontsize=8.5)
    ax.margins(0.08)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _panel_b_female_bmi(ax, umap_female: pd.DataFrame) -> None:
    """Female UMAP coloured by BMI, the matching covariate (panel b).

    Shows that the A/B split tracks an adiposity gradient AND that the two groups
    overlap in BMI — at any given mid-range BMI, women fall into both clusters.
    That BMI-overlap zone is exactly the population the age+BMI matching draws on,
    motivating the question 'why is a woman of the same size and age in A vs B?'.
    Outliers (e.g. a few high-BMI women sitting inside the lean cluster) are visible
    rather than hidden inside a cluster mean.
    """
    sub = umap_female.dropna(subset=['bmi'])
    samp = sub.sample(min(1800, len(sub)), random_state=42)
    vmin, vmax = samp['bmi'].quantile([0.01, 0.99])
    # Truncated 'Blues' (drop the near-white low end) so low-BMI / Cluster-B points
    # stay visible against the white background while keeping a single-hue, on-palette
    # sequential ramp (light blue = lean → dark blue = adipose).
    blues = LinearSegmentedColormap.from_list(
        'blues_trunc', plt.get_cmap('Blues')(np.linspace(0.22, 1.0, 256)))
    sc = ax.scatter(samp['UMAP1'], samp['UMAP2'], c=samp['bmi'], cmap=blues,
                    s=6, alpha=0.9, vmin=float(vmin), vmax=float(vmax),
                    rasterized=True, linewidths=0)
    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('BMI (kg/m$^2$)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    # Orient the reader: label each cluster's centroid (A = adipose, B = lean)
    # to tie the blobs to the A/B naming used in panels c and d. K-means IDs are
    # not deterministic across runs, so assign letters by actual mean BMI
    # (higher-BMI cluster = A/adipose) rather than by raw cluster ID.
    mean_bmi = sub.groupby('cluster')['bmi'].mean()
    adipose_id = mean_bmi.idxmax()
    for cid in mean_bmi.index:
        name = 'A' if cid == adipose_id else 'B'
        cl = sub[sub['cluster'] == cid]
        if cl.empty:
            continue
        ax.annotate(name, (cl['UMAP1'].median(), cl['UMAP2'].max()),
                    fontsize=12, fontweight='bold', color='#222', ha='center',
                    va='bottom')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(False)


def _panel_c_diverging(ax, all_results_csv: str) -> None:
    """Curated signed-effect forest (panel c).

    ~2 features per system, grouped by system with alternating shading.
    x-axis = matched-cohort Cohen's d (positive = B > A, negative = A > B).
    Features hand-curated to represent the biological story and avoid
    redundant laterality pairs (e.g. keep bilateral total, not left+right).

    Data source: the master ``all_results.csv`` (matched cohort, all systems),
    rebuilt directly so panel c always matches the current cohort.
    """
    # Best feature(s) per system, chosen automatically by |Cohen's d| among the
    # matched-cohort hits (q<0.05) — no hand-picked features. Restricted to the
    # systems that carry the muscle/bone/functional story (off-narrative systems
    # such as liver/diet are not shown), excluding laterality/diff duplicates;
    # body composition contributes its top lean (B↑) and top peripheral-fat (B↓)
    # so both halves of the composition shift are visible.
    df = pd.read_csv(all_results_csv).drop_duplicates(['system', 'feature'])
    df = df[df['adj_p_value'] < 0.05].copy()
    df['absd'] = df['delta_z'].abs()
    df['feat'] = df['feature'].astype(str)
    df = df[~df['feat'].str.contains('_diff', case=False)]   # drop L–R difference noise

    def _pick(sys_key, sign=None, inc=None, exc=None, drop_lat=False, n=1):
        """Top-|d| significant feature(s) for a system. drop_lat removes L/R
        laterality (only where a bilateral total exists, e.g. body comp/bone —
        NOT grip). inc/exc are name include/exclude filters to keep the metric
        on-story (e.g. bone = BMD/BMC, not area)."""
        sub = df[df['system'] == sys_key]
        if drop_lat:
            sub = sub[~sub['feat'].str.contains('left|right', case=False)]
        if sign is not None:
            sub = sub[np.sign(sub['delta_z']) == sign]
        if inc:
            sub = sub[sub['feat'].str.contains(inc, case=False)]
        if exc:
            sub = sub[~sub['feat'].str.contains(exc, case=False)]
        return sub.sort_values('absd', ascending=False).head(n)

    # ~1–2 complementary features per system with signal (no near-duplicates),
    # spanning the domains discussed in the text: body composition, bone, muscle
    # function, cardiovascular/autonomic, haematology, liver, renal, diet and
    # activity. Glycemic/lipid panels carry 0 hits and are omitted.
    picks = [
        _pick('body_composition', sign=+1, inc='lean|fat_free', exc='android|gynoid', drop_lat=True),
        _pick('body_composition', sign=-1, inc='fat', exc='fat_free|lean', drop_lat=True),
        _pick('bone_density',     sign=+1, inc='bmc', drop_lat=True),   # top BMC site
        _pick('bone_density',     sign=+1, inc='bmd', drop_lat=True),   # top BMD site
        _pick('frailty',          inc='grip'),                          # grip (no bilateral total → keep L/R)
        _pick('cardiovascular_system', inc='hr_bpm'),                   # resting heart rate
        _pick('cardiovascular_system', inc='r_r_ms'),                   # heart-rate variability
        _pick('sleep',            inc='heart_rate_mean_during_rem'),     # nocturnal HR
        _pick('hematopoietic_system', n=2),                             # haemoglobin + MCH
        _pick('liver',            inc='sound'),                         # liver sound speed (↓ liver fat)
        _pick('renal_function',   inc='creatinine'),                    # creatinine (↑ muscle mass)
        _pick('high_level_diet',  inc='imedas'),                        # Mediterranean diet score
        _pick('high_level_diet',  inc='energy'),                        # energy intake / BMR
        _pick('lifestyle',        inc='physical_activity'),             # moderate physical activity
    ]
    entries = []
    for sub in picks:
        for _, r in sub.iterrows():
            entries.append({
                'system': r['system'], 'label': _prettify(str(r['feature'])),
                'd': float(r['delta_z']), 'ci_lo': float(r['ci_low']),
                'ci_hi': float(r['ci_high']), 'q': float(r['adj_p_value']),
                'sig': True,
            })

    n = len(entries)
    y = np.arange(n)

    # Alternating shading per system group
    sys_order = list(dict.fromkeys(e['system'] for e in entries))
    for i, e in enumerate(entries):
        if sys_order.index(e['system']) % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=ROW_SHADE, zorder=0, linewidth=0)

    c_b = CLUSTER_COLORS['B']
    c_a = CLUSTER_COLORS['A']

    for i, e in enumerate(entries):
        d, lo, hi = e['d'], e['ci_lo'], e['ci_hi']
        c = c_b if d > 0 else c_a
        # CI line
        ax.plot([lo, hi], [i, i], color=c, lw=1.6, alpha=0.55, zorder=2)
        # Stem from zero
        ax.plot([0, d], [i, i], color=c, lw=2.0, solid_capstyle='round',
                zorder=3, alpha=0.85)
        # Dot
        ax.scatter(d, i, color=c, s=50, zorder=4, edgecolors='none')

    # System group dividers
    prev_sys = None
    for i, e in enumerate(entries):
        if e['system'] != prev_sys and prev_sys is not None:
            ax.axhline(i - 0.5, color='#BBBBBB', lw=0.8, ls='-', zorder=1)
        prev_sys = e['system']

    ax.axvline(0, color='#555', lw=1.0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([e['label'] for e in entries])
    ax.set_ylim(-0.55, n + 0.2)

    xabs = max(abs(e['ci_lo']) for e in entries + [{'ci_lo': 0}] if True)
    xabs = max(xabs, max(e['ci_hi'] for e in entries)) * 1.15
    ax.set_xlim(-xabs, xabs)
    ax.set_xlabel("Cohen's $d$  (B − A; all adj. $p$ < 0.05)")
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='y', visible=False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=c_b, lw=2.5, label='Higher in Cluster B'),
        Line2D([0], [0], color=c_a, lw=2.5, label='Higher in Cluster A'),
    ]
    # Legend returned so caller can place it as a shared figure-level legend.
    return handles


def _panel_omics_volcano(ax, layer_csv: str, title: str, label_fn,
                         n_label: int = 8, xlabel: str = 'Mean difference (B − A)') -> None:
    """Volcano for one omics layer (matched cohort, mean_diff = B − A)."""
    d = pd.read_csv(layer_csv)
    d['is_sig'] = d['qval'] < QSIG_OMICS
    sig = d['is_sig']
    up = sig & (d['mean_diff'] > 0)
    dn = sig & (d['mean_diff'] < 0)
    c_b, c_a = CLUSTER_COLORS['B'], CLUSTER_COLORS['A']
    ax.scatter(d.loc[~sig, 'mean_diff'], d.loc[~sig, 'neg_log10q'],
               s=6, color=C_NS, alpha=0.5, linewidths=0, zorder=1)
    ax.scatter(d.loc[up, 'mean_diff'], d.loc[up, 'neg_log10q'],
               s=9, color=c_b, alpha=0.9, linewidths=0, zorder=2)
    ax.scatter(d.loc[dn, 'mean_diff'], d.loc[dn, 'neg_log10q'],
               s=9, color=c_a, alpha=0.9, linewidths=0, zorder=2)
    ax.axhline(-np.log10(QSIG_OMICS), color='black', ls='--', lw=0.8, zorder=1)
    ax.axvline(0, color='#555', lw=0.6, alpha=0.6, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$-\\log_{10}$ adj. $p$')
    if title:
        ax.set_title(title)
    lab = d[sig].sort_values('qval').head(n_label)
    texts = [ax.text(r['mean_diff'], r['neg_log10q'], label_fn(r['feature']),
                     fontsize=10, zorder=3) for _, r in lab.iterrows()]
    if _HAS_ADJUST and texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.3))


# ── Composer ──────────────────────────────────────────────────────────────────

def compose_figure(*, save: bool = True, use_cached_umap: bool = True):
    apply_paper_rcparams()
    # Fig 6 has only 4 large panels — larger fonts fill the space and stay legible.
    plt.rcParams.update({
        'font.size':        10,
        'axes.titlesize':   11,
        'axes.labelsize':   10,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  10,
    })

    # Load UMAP — this requires the HPP embeddings to be accessible.
    print('Loading UMAP data …')
    umap_female = _load_umap()                      # female-only clustering UMAP (panels b/c/d)
    umap_all = _load_fullcohort_umap(umap_female)   # full-cohort shared UMAP by sex + A/B (panel a)
    print(f'  Female: {len(umap_female):,} | Full cohort: {len(umap_all):,} '
          f'(Male: {(umap_all["sex_label"]=="Male").sum():,})')

    fig = plt.figure(figsize=(7.09, 6.84))   # Nature double-column width (180 mm); authored at final scale
    # 2×2: UMAPs on top (a = sex, b = female coloured by BMI), then the matched
    # multi-system forest (c) and the proteomics volcano (d) below.
    gs = fig.add_gridspec(2, 2, hspace=0.26, wspace=0.22,
                          height_ratios=[1.0, 1.7])

    ax_a = fig.add_subplot(gs[0, 0])   # full-cohort UMAP by sex
    ax_b = fig.add_subplot(gs[0, 1])   # female UMAP coloured by BMI
    ax_c = fig.add_subplot(gs[1, 0])   # matched-cohort multi-system forest
    ax_d = fig.add_subplot(gs[1, 1])   # proteomics volcano

    _panel_a_fullcohort_umap(ax_a, umap_all)
    _panel_b_female_bmi(ax_b, umap_female)
    legend_handles = _panel_c_diverging(ax_c, paths.out_table('tableD_cluster_all_results.csv'))
    _panel_omics_volcano(ax_d, str(OMICS_CSV),
                         '', lambda f: f, n_label=5, xlabel='log$_2$ fold-change (B − A)')

    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d], 'abcd'):
        add_panel_letter(ax, letter)

    # Reserve bottom margin so the shared B/A legend sits clear below the
    # bottom-row (c, d) x-axis labels.
    fig.subplots_adjust(bottom=0.13, left=0.04, right=0.97, top=0.95)
    fig.legend(handles=legend_handles, frameon=False, ncol=2,
               loc='lower center', bbox_to_anchor=(0.5, 0.015),
               handlelength=1.6, columnspacing=2.0, fontsize=10)

    if save:
        pdf = paths.out_figure('fig6_female_clusters.pdf')
        png = paths.out_figure('fig6_female_clusters.png')
        fig.savefig(pdf, format='pdf', dpi=400, facecolor='white',
                    transparent=False, bbox_inches='tight')
        fig.savefig(png, dpi=400, facecolor='white',
                    transparent=False, bbox_inches='tight')
        print(f'Saved  →  {pdf}')
        print(f'Saved  →  {png}')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--female-umap", type=Path, default=FEMALE_UMAP_CSV,
                        help="CSV with UMAP1, UMAP2, cluster, and bmi columns.")
    parser.add_argument("--full-umap", type=Path, default=FULL_UMAP_CSV,
                        help="CSV with UMAP1, UMAP2, and sex_label columns.")
    parser.add_argument("--omics", type=Path, default=OMICS_CSV,
                        help="Matched proteomics results with feature, mean_diff, qval, neg_log10q.")
    args = parser.parse_args()
    FEMALE_UMAP_CSV = args.female_umap
    FULL_UMAP_CSV = args.full_umap
    OMICS_CSV = args.omics
    compose_figure()
