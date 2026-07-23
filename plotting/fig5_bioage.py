"""Figure 5 — biological-age results.

Composes the main figure from aggregate tables in ``tables/`` and controlled
participant-level prediction/event inputs configured in ``config.py``.

Layout (4 rows × 2 cols)
------------------------
    a  HPP age prediction (scatter + identity)        b  UKBB age prediction
    c  ♀ sex-divergent Q4 − Q1 phenotype forest        d  ♂ phenotype forest
    e  Mortality HR forest (continuous + Q1–Q4)        f  KM by gap quartile
    g  Disease prevalence Q1 vs Q4 (bar chart)         h  Paired pre/post medication

Visual conventions are pulled verbatim from Fig 2 / Fig 3 (``plot_fig2_4arm.py``
and ``ukbb/plot_fig3_cox.py``): sans-serif rcParams, 26 pt bold panel letters
at (-0.14, 1.03), dashed y-grid, top+right spines hidden, ``MODEL_COLORS``
palette, alternating ``#F5F5F5`` row backgrounds for lollipop forests, KM line
widths 2.2/1.4, PercentFormatter y-axis on KM panels.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set2 palette — used wherever colours are not already semantically fixed
# (PALETTE_QUARTILE and MODEL_COLORS remain their canonical values).
_SET2 = sns.color_palette('Set2', n_colors=8)
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from downstream.bioage import paths
from downstream.bioage.style import (
    MODEL_COLORS, PALETTE_QUARTILE, add_panel_letter, apply_paper_rcparams,
)


# ── Panel-specific constants ──────────────────────────────────────────────────
TOP_PHENO_N = 9         # phenotypes shown per sex panel (c, d)
DISEASE_TOP_N = 8       # diseases shown in panel g (Q1 vs Q4)
ROW_SHADE = '#F5F5F5'   # Fig 3 alternating-row background

# ICD-10 sub-conditions that decompose the three composite buckets (cardiovascular
# disease, osteoarthritis, renal failure). They live in the extended prevalence
# table alongside sarcopenia, but panel g shows only the composite buckets — the
# sub-condition breakdown is a supplementary figure (extend_disease_panel.py).
DECOMP_SUBCONDITIONS = {
    'Angina', 'Myocardial infarction', 'Chronic ischaemic heart disease', 'Heart failure',
    'Hip arthrosis', 'Knee arthrosis', 'Other arthrosis',
    'Acute renal failure', 'Chronic kidney disease', 'Unspecified renal failure',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shade_alternate_rows(ax, n_rows: int) -> None:
    for idx in range(n_rows):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color=ROW_SHADE, zorder=0, linewidth=0)


def _shorten_label(s: str) -> str:
    import re
    s = s.split(' - ')[0].strip()
    if 'visceral adipose' in s.lower():
        return 'VAT mass'
    # Strip the descriptive parentheticals and the laterality suffix — sides
    # are highly correlated, so showing "(left)" / "(right)" in the figure adds
    # noise without information.
    s = s.replace('(bone mineral density)', '').replace('(bone mineral content)', '')
    s = re.sub(r'\s*\((left|right)\)\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+(left|right)\s*$', '', s, flags=re.IGNORECASE)
    # Collapse the duplicated short-name ("Total BMD BMD" → "Total BMD",
    # "Ribs BMC BMC" → "Ribs BMC") that survives the parenthetical removal.
    s = re.sub(r'\b(BMD)\s+\1\b', r'\1', s)
    s = re.sub(r'\b(BMC)\s+\1\b', r'\1', s)
    s = ' '.join(s.split())
    parts = s.split()
    return ' '.join(parts[-5:]) if len(parts) > 5 else s


def _top_phenotypes(csv_path: str, n: int = TOP_PHENO_N) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df['Adjusted_P_Value'] < 0.05].copy()
    df['abs_d'] = df['Cohens_d'].abs()
    df = df[~df['Label'].str.lower().str.contains('t-score|z-score|bone area', na=False)]

    def _cat(label):
        lb = label.lower()
        if 'vat' in lb or 'visceral' in lb:    return 'VAT'
        if 'bmc' in lb:                         return 'BMC'
        if 'bmd' in lb or 'bone mineral' in lb: return 'BMD'
        if 'lean' in lb or 'fat-free' in lb:    return 'Lean'
        if 'fat' in lb:                         return 'Fat'
        return 'Other'

    df['cat'] = df['Label'].apply(_cat)
    df = df.sort_values('abs_d', ascending=False)

    # Dedupe redundant "mass" vs "volume" reporting of the same VAT measurement
    # (they have identical Cohen's d). Keep the mass version — clinically
    # standard and matches the draft's wording.
    is_vat = df['Label'].str.lower().str.contains('vat|visceral')
    is_volume = df['Label'].str.lower().str.contains('volume')
    df = df[~(is_vat & is_volume)]

    # Collapse laterality duplicates: "Arm BMD (left)" and "Arm BMD (right)"
    # both shorten to "Arm BMD" — keep the stronger of the two and discard
    # the weaker so the panel doesn't show two rows with the same label.
    df['short_label'] = df['Label'].apply(_shorten_label)
    df = df.drop_duplicates(subset='short_label', keep='first')

    # Also collapse "Arm" vs "Arms" / "Leg" vs "Legs": singular = one side,
    # plural = bilateral sum. The signals are highly correlated; keep the
    # plural (whole-region) version, drop the singular if both survive.
    def _norm_key(lbl):
        import re
        return re.sub(r'^(Arms?|Legs?)\b', lambda m: m.group(1).rstrip('s') + 's', lbl)
    df['_dup_key'] = df['short_label'].apply(_norm_key)
    df = df.drop_duplicates(subset='_dup_key', keep='first').drop(columns='_dup_key')

    # Per-category caps. VAT collapses to one entry; everything else allows two
    # so the negative-d (loss) and positive-d (paradoxical gain) signals coexist.
    cat_caps = {'VAT': 1, 'BMC': 2, 'BMD': 2, 'Lean': 2, 'Fat': 2, 'Other': 1}
    picks = []
    for cat, cap in cat_caps.items():
        picks.append(df[df['cat'] == cat].head(cap))
    df = pd.concat(picks, ignore_index=True)

    # If we still exceed the panel budget, keep the most extreme |d| entries
    # but guarantee any positive-d hits stay in (those carry the draft's
    # "paradoxical compensatory increase" claim).
    if len(df) > n:
        pos = df[df['Cohens_d'] > 0]
        neg = df[df['Cohens_d'] <= 0].sort_values('abs_d', ascending=False).head(n - len(pos))
        df = pd.concat([pos, neg], ignore_index=True)

    df = df.sort_values('Cohens_d').reset_index(drop=True)
    return df


def _scatter_panel(ax, df: pd.DataFrame, age_col: str, pred_col: str,
                   *, title: str, max_per_q: int = 400,
                   ann_r: float = None, ann_mae: float = None, mae_dec: int = 2) -> None:
    """Predicted vs chronological age, coloured by quartile (panels a, b).
    ann_r/ann_mae override the annotation with a pre-computed (e.g. canonical
    10-seed cross-validated) metric so it matches the value reported elsewhere
    (Fig 2c); the scatter itself still shows the per-subject predictions."""
    df = df.dropna(subset=[age_col, pred_col, 'gap_quartile']).copy()
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        sub = df[df['gap_quartile'] == q]
        if len(sub) > max_per_q:
            sub = sub.sample(max_per_q, random_state=42)
        ax.scatter(sub[age_col], sub[pred_col], c=PALETTE_QUARTILE[q],
                   s=6, alpha=0.55, label=q, rasterized=True, linewidth=0)
    lo, hi = df[age_col].min() - 1, df[age_col].max() + 1
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel('Chronological age (yr)')
    ax.set_ylabel('Predicted age (yr)')
    r = ann_r if ann_r is not None else pearsonr(df[age_col], df[pred_col])[0]
    mae = ann_mae if ann_mae is not None else mean_absolute_error(df[age_col], df[pred_col])
    ax.text(0.97, 0.05, f'r = {r:.2f}\nMAE = {mae:.{mae_dec}f} yr',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=MODEL_COLORS['lejepa'], alpha=0.92, linewidth=0.8))
    ax.set_title(title, fontweight='bold', pad=8)
    ax.legend(title='Gap quartile', markerscale=2.0, frameon=False, loc='upper left')


def _phenotype_forest(ax, df: pd.DataFrame, *, sex_label: str) -> None:
    """Lollipop forest of Cohen's d (Q4 − Q1) per phenotype (panels c, d)."""
    n = len(df)
    _shade_alternate_rows(ax, n)
    color = MODEL_COLORS['lejepa']
    y = np.arange(n)
    for i, row in df.iterrows():
        d = row['Cohens_d']
        c = color if d > 0 else '#c0392b'
        ax.plot([0, d], [i, i], color=c, lw=2.4, solid_capstyle='round', zorder=3, alpha=0.85)
        ax.scatter(d, i, color=c, s=60, zorder=4, edgecolors='none')
    ax.axvline(0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels(df['short_label'].tolist())
    # Bump phenotype labels above the global 5 pt so they stay sharp when the
    # large PNG is downscaled to screen width.
    ax.tick_params(axis='y', labelsize=7)
    ax.set_ylim(-0.55, n + 0.3)
    # Tight xlim — add 20% padding each side so stars and dots aren't clipped.
    dmin = df['Cohens_d'].min(); dmax = df['Cohens_d'].max()
    pad = max(abs(dmin), abs(dmax)) * 0.20
    ax.set_xlim(dmin - pad, dmax + pad)
    ax.set_xlabel("Cohen's $d$  (Q4 − Q1)")
    ax.set_title(sex_label, fontweight='bold', pad=8)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='y', visible=False)


def _mortality_forest(ax, cox_csv: str) -> None:
    """Forest of HR (continuous gap + Q2/Q3/Q4 vs Q1) — panel e."""
    df = pd.read_csv(cox_csv)
    df = df[df['covariate'].isin(['bioage_gap', 'Q2', 'Q3', 'Q4'])].copy()
    labels = {
        'bioage_gap': 'Per +1 yr gap',
        'Q2':         'Q2 vs Q1',
        'Q3':         'Q3 vs Q1',
        'Q4':         'Q4 vs Q1',
    }
    df['label'] = df['covariate'].map(labels)
    df = df.iloc[::-1].reset_index(drop=True)  # show Q4 at top
    n = len(df)
    _shade_alternate_rows(ax, n)

    for i, row in df.iterrows():
        hr = row['exp(coef)']
        lo = row['exp(coef) lower 95%']
        hi = row['exp(coef) upper 95%']
        p = row['p']
        c = MODEL_COLORS['lejepa'] if (p < 0.05) and (hr > 1) else '#888'
        ax.errorbar([hr], [i], xerr=[[hr - lo], [hi - hr]], fmt='o',
                    color=c, markersize=7, ecolor=c, elinewidth=1.6, capsize=4, zorder=4)
        p_str = f'{p:.1e}' if p < 1e-4 else f'{p:.3f}'
        annot = f'HR {hr:.2f} [{lo:.2f}–{hi:.2f}]\np={p_str}'
        ax.text(hi + 0.02, i, annot,
                va='center', ha='left', fontsize=6, color='#222', clip_on=True,
                linespacing=1.3)
    ax.axvline(1.0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.set_yticks(range(n)); ax.set_yticklabels(df['label'].tolist())
    ax.set_ylim(-0.5, n)   # explicit top margin so two-line annotation at top row isn't y-clipped
    ax.set_xlabel('Hazard ratio (age, sex adjusted)')
    xmax = float(df['exp(coef) upper 95%'].max()) * 1.45
    # Left limit must clear the smallest lower-CI whisker (Q2 dips to ~0.74),
    # otherwise the interval is clipped at the axis edge.
    xmin = min(0.85, float(df['exp(coef) lower 95%'].min()) - 0.05)
    ax.set_xlim(xmin, max(2.0, xmax))
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='y', visible=False)


def _km_panel(ax, gap_df: pd.DataFrame) -> None:
    """Cumulative all-cause mortality KM by gap quartile (panel f).

    Purely descriptive — panel e carries the formal statistical claim via the
    age+sex-adjusted Cox model.
    """
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = gap_df['gap_quartile'] == q
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(gap_df.loc[mask, 'time_yr'], gap_df.loc[mask, 'event'],
                label=f'{q} (n={mask.sum():,})')
        lw = 2.2 if q in ('Q1', 'Q4') else 1.4
        ls = '-' if q == 'Q4' else ('--' if q == 'Q1' else '-')
        kmf.plot_cumulative_density(ax=ax, ci_show=False,
                                    color=PALETTE_QUARTILE[q], lw=lw, linestyle=ls)

    ax.set_xlabel('Years from baseline DXA scan')
    ax.set_ylabel('Cumulative all-cause mortality')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(axis='y', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper left', title='Gap quartile')


def _disease_panel(ax, disease_csv: str, *, top_n: int = DISEASE_TOP_N) -> None:
    """Q1 vs Q4 prevalence bar chart, sex-pooled, top diseases by RR (panel g)."""
    df = pd.read_csv(disease_csv)
    df = df[df['Adj_P_Value'] < 0.05].copy()
    df = df[~df['Condition'].isin(DECOMP_SUBCONDITIONS)].copy()  # composite buckets only
    # average RR across sexes; show top_n by avg RR (with ≥1% Q1 prev each sex)
    pivot = df.pivot_table(index='Condition', columns='Sex',
                            values=['Prev_Q1', 'Prev_Q4', 'RR'])
    valid = (pivot['Prev_Q1'].min(axis=1) >= 1.0) & pivot['Prev_Q4'].notna().all(axis=1)
    pivot = pivot.loc[valid].copy()
    pivot[('RR', 'mean')] = pivot['RR'].mean(axis=1)
    pivot = pivot.sort_values(('RR', 'mean'), ascending=False).head(top_n)

    _ABBREV = {
        'Rheumatoid arthritis': 'Rheum. arthritis',
        'Atrial fibrillation':  'Atrial fibrill.',
        'Cardiovascular disease': 'Cardiovasc. dis.',
        'Sarcopenia (EWGSOP2)': 'Sarcopenia',
    }
    conds = [_ABBREV.get(c, c) for c in pivot.index.tolist()]
    x = np.arange(len(conds))
    w = 0.20
    palette = {
        ('Female', 'Q1'): ('#D81B60', 0.45),
        ('Female', 'Q4'): ('#D81B60', 0.92),
        ('Male',   'Q1'): ('#1565C0', 0.45),
        ('Male',   'Q4'): ('#1565C0', 0.92),
    }
    legend_lookup = {
        ('Female', 'Q1'): 'Female Q1 (young)',
        ('Female', 'Q4'): 'Female Q4 (old)',
        ('Male',   'Q1'): 'Male Q1 (young)',
        ('Male',   'Q4'): 'Male Q4 (old)',
    }
    for j, (sex, q) in enumerate([('Female', 'Q1'), ('Female', 'Q4'),
                                  ('Male', 'Q1'),   ('Male', 'Q4')]):
        col, alpha = palette[(sex, q)]
        vals = pivot[(f'Prev_{q}', sex)].values
        ax.bar(x + (j - 1.5) * w, vals, width=w,
               color=col, alpha=alpha, edgecolor='black' if q == 'Q4' else 'none',
               linewidth=0.5 if q == 'Q4' else 0,
               label=legend_lookup[(sex, q)])
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=30, ha='right')
    ax.set_ylabel('Prevalence (%)')
    ymax_bar = pivot[['Prev_Q1', 'Prev_Q4']].max().max()
    ax.set_ylim(0, ymax_bar * 1.35)
    ax.legend(frameon=False, ncol=1, loc='upper right',
              handlelength=1.2, handletextpad=0.4, labelspacing=0.3, borderaxespad=0.3)
    ax.grid(axis='y', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='x', visible=False)


def _medication_panel(ax, paired_csv: str) -> None:
    """Δgap before → after for FDR-significant ATC-3 groups (panel h)."""
    df = pd.read_csv(paired_csv)
    sig = df[df['adjusted_p_value'] < 0.05].copy()

    sig = sig.sort_values('mean_delta')

    entries = []
    for _, row in sig.iterrows():
        name = f"{row['drug']} ({row['demo']})"
        entries.append({
            'label':  name,
            'before': row['mean_before'],
            'after':  row['mean_after'],
            'delta':  row['mean_delta'],
            'N':      int(row['N_paired']),
            'q':      row['adjusted_p_value'],
        })

    if not entries:
        ax.text(0.5, 0.5, 'No FDR-significant medications', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        return

    x = np.arange(len(entries))
    w = 0.34
    for i, e in enumerate(entries):
        ax.bar(i - w / 2, e['before'], width=w, color='#BDBDBD', alpha=0.85,
               edgecolor='black', linewidth=0.4,
               label='Before' if i == 0 else None)
        ax.bar(i + w / 2, e['after'], width=w, color=MODEL_COLORS['lejepa'], alpha=0.92,
               edgecolor='black', linewidth=0.4,
               label='After'  if i == 0 else None)
        star = '***' if e['q'] < 0.001 else ('**' if e['q'] < 0.01 else '*')
        # Tiny significance marker above whichever bar is taller
        y_top = max(e['before'], e['after'])
        ax.text(i, y_top + 0.15, star,
                ha='center', va='bottom', fontsize=6, fontweight='bold', color='#444')

    ax.axhline(0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([e['label'] for e in entries])
    ax.set_ylabel('Biological-age gap (yr)')
    subtitle = '   ·   '.join(
        f"{e['label'].split(' ')[0]} (n={e['N']}): Δ = {e['delta']:+.2f} yr"
        for e in entries
    )
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6, color='#222')

    # Headroom so the significance star and legend don't crowd the bars.
    ymax_bar = max(max(e['before'], e['after']) for e in entries)
    ymin_bar = min(min(e['before'], e['after'], 0) for e in entries)
    ax.set_ylim(ymin_bar - 0.4, ymax_bar * 1.20 + 0.4)

    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='y', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.grid(axis='x', visible=False)


def _load_mortality_followup() -> pd.DataFrame:
    """Reconstruct (eid, gap, gap_quartile, time_yr, event) for KM panel."""
    pred = pd.read_csv(paths.PRED_CSV)
    pred = pred[pred['visit'].astype(str) == '2'].drop_duplicates('eid').set_index('eid')

    from downstream.bioage.rtm import detrend_gap, bin_quartiles
    pred['bioage_gap']   = detrend_gap(pred)
    pred['gap_quartile'] = bin_quartiles(pred['bioage_gap'])

    ev = pd.read_csv(paths.EVENTS_CSV, index_col=0, low_memory=False)
    ev = ev[ev.index.notna() & ~ev.index.isin(['error', 'skipped'])]
    ev.index = ev.index.astype(int)
    baseline_col = 'Date of attending assessment centre - visit 2'
    death_col    = 'Date of death - visit 0'
    ev = ev[[baseline_col, death_col]]
    df = pred[['bioage_gap', 'gap_quartile']].join(ev, how='inner')
    df = df.dropna(subset=[baseline_col])
    start = pd.to_datetime(df[baseline_col], errors='coerce')
    death = pd.to_datetime(df[death_col],    errors='coerce')
    keep = ~(death.notna() & (death <= start))
    df, start, death = df[keep], start[keep], death[keep]
    event = death.notna() & (death > start)
    admin = death[event].max()
    end = death.where(event, pd.Timestamp(admin))
    df['time_yr'] = (end - start).dt.days / 365.25
    df['event']   = event.astype(int)
    df = df[df['time_yr'] > 0]
    return df.dropna(subset=['gap_quartile'])


# ── Composer ──────────────────────────────────────────────────────────────────

def compose_figure(*, save: bool = True):
    # Load the nature_double mplstyle as the base (thin strokes, embedded fonts,
    # dpi=400), then apply_paper_rcparams() overrides font sizes upward so the
    # panels stay legible at this figure's 18×26 in dimensions.
    apply_paper_rcparams()

    # ── load inputs ─────────────────────────────────────────────────────────
    ukbb_preds = pd.read_csv(paths.out_table('tableD_bioage_predictions_ukbb.csv'))
    hpp_preds  = pd.read_csv(paths.out_table('tableD_bioage_predictions_hpp.csv'))
    hpp_preds  = hpp_preds.rename(columns={'age_residual': 'bioage_gap'})
    hpp_preds  = hpp_preds.dropna(subset=['age_true', 'age_pred_lejepa', 'bioage_gap'])
    from downstream.bioage.rtm import bin_quartiles
    hpp_preds['gap_quartile']  = bin_quartiles(hpp_preds['bioage_gap'])
    ukbb_preds['gap_quartile'] = ukbb_preds['gap_quartile'].astype('category')

    df_f = _top_phenotypes(paths.out_table('tableD_bioage_phenotype_female.csv'))
    df_m = _top_phenotypes(paths.out_table('tableD_bioage_phenotype_male.csv'))

    # ── build figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 13))
    gs = fig.add_gridspec(4, 2, hspace=0.75, wspace=0.55,
                          height_ratios=[1.0, 1.3, 1.0, 1.2])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1])

    # Annotate the HPP scatter with the canonical 10-seed cross-validated metric
    # (= the value reported in Fig 2c, from age_mae_imaging_only.csv) rather than the
    # single-split in-plot value, so the two figures report the same HPP age accuracy.
    _hpp_age = pd.read_csv(os.path.join(paths.TABLES_DIR, 'age_mae_imaging_only.csv'))
    _hl = _hpp_age[(_hpp_age['cohort'] == 'HPP') & (_hpp_age['model'] == 'lejepa')].iloc[0]
    hpp_ann_r, hpp_ann_mae = float(_hl['pearson_mean']), float(_hl['mae_yr_mean'])

    # ── panels ───────────────────────────────────────────────────────────────
    _scatter_panel(ax_a, hpp_preds, 'age_true', 'age_pred_lejepa', title='HPP',
                   ann_r=hpp_ann_r, ann_mae=hpp_ann_mae, mae_dec=2)
    _scatter_panel(ax_b, ukbb_preds, 'age_true', 'age_pred_lejepa', title='UKBB', mae_dec=2)
    _phenotype_forest(ax_c, df_f, sex_label='Female (UKBB)')
    _phenotype_forest(ax_d, df_m, sex_label='Male (UKBB)')
    _mortality_forest(ax_e, paths.out_table('tableE_bioage_gap_mortality_cox.csv'))
    gap_df = _load_mortality_followup()
    _km_panel(ax_f, gap_df)
    _disease_panel(ax_g, paths.out_table('tableD_bioage_disease_prevalence_extended.csv'))
    _medication_panel(ax_h, paths.out_table('table_6_medications.csv'))

    # ── panel letters ────────────────────────────────────────────────────────
    for ax, letter in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h],
                          ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']):
        add_panel_letter(ax, letter)

    if save:
        pdf = paths.out_figure('fig5_biological_age.pdf')
        png = paths.out_figure('fig5_biological_age.png')
        fig.savefig(pdf, format='pdf', dpi=400, facecolor='white',
                    transparent=False, bbox_inches='tight')
        fig.savefig(png, dpi=400, facecolor='white',
                    transparent=False, bbox_inches='tight')
        print(f'Saved  →  {pdf}')
        print(f'Saved  →  {png}')
    return fig


if __name__ == '__main__':
    compose_figure()
