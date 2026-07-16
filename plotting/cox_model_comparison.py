import re
import textwrap
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# --- Event filtering / pretty-printing ---
_NON_DISEASE_EVENTS = {'__derived_admin_censor_date__'}
_NON_DISEASE_PATTERNS = [re.compile(r'assessment centre', re.IGNORECASE)]

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common.cox_utils import (COX_EVENT_SHORT_NAMES as _EVENT_SHORT_NAMES,
                       COX_EVENT_CATEGORIES  as _EVENT_CATEGORIES,
                       COX_CATEGORY_NAMES    as _CATEGORY_NAMES)

# --- Model name mapping: CSV column names → display labels ---
_MODEL_RENAME = {
    'Covariates':                      'Covariates (age/sex/BMI)',
    'DXA Tabular':                     'DXA Tabular',
    'DXA Tabular + Covariates':        'DXA Tabular + Cov',
    'DeepDXA':                         'DeepDXA',
    'DXA SSL (DINO)':                  'DINOv3 (Frozen)',
    'DeepDXA + Covariates':            'DeepDXA + Cov',
}

# --- Color palette — matched to plot_comparison.py MODEL_COLORS ---
_MODEL_COLORS = {
    'DeepDXA':                    '#083c7d',
    'DeepDXA + Cov':              '#083c7d',
    'DINOv3 (Frozen)':            '#7fb9dc',
    'DXA Tabular':                '#8ccbb3',
    'DXA Tabular + Cov':          '#8ccbb3',
    'Covariates (age/sex/BMI)':   '#bdbdbd',
}
_FALLBACK_COLORS = ['#b35806', '#E67E22', '#16A085', '#C0392B', '#2980B9']

def _model_color(model_name: str, fallback_idx: int = 0) -> str:
    return _MODEL_COLORS.get(model_name,
           _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)])

def _is_ensemble(model_name: str) -> bool:
    return ' + ' in model_name

def _prettify_event(raw: str) -> Optional[str]:
    raw = raw.strip()
    if raw in _NON_DISEASE_EVENTS: return None
    if any(p.search(raw) for p in _NON_DISEASE_PATTERNS): return None
    low = raw.lower()
    if 'death due to cancer' in low or 'c00-c97' in low: return 'Cancer Death'
    if re.search(r'\bdate of death\b', low): return 'All-Cause Death'
    if 'all cause dementia' in low: return 'Dementia'
    if 'parkinsonism' in low: return 'Parkinsonism'
    if 'alzheimer' in low and 'report' in low: return None
    if 'alzheimer' in low: return "Alzheimer's Disease"
    m = re.search(r'\(([^)]+)\)', raw)
    if m:
        name = m.group(1)
        name = re.sub(r'\s*\[.*?\]', '', name).strip().title()
        return _EVENT_SHORT_NAMES.get(name, name)
    name = re.sub(r'\s*-\s*visit\s*\d+', '', raw, flags=re.IGNORECASE).strip()
    name = re.sub(r'^Date\s+(of\s+)?', '', name, flags=re.IGNORECASE).strip().title()
    return name if name else raw

def _significance_stars(p: float) -> str:
    if pd.isna(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

# --- Column helpers ---
_DESIRED_MODELS = [
    'Covariates',
    'DXA Tabular',
    'DXA Tabular + Covariates',
    'DXA SSL (DINO)',
    'DeepDXA',
    'DeepDXA + Covariates',
]

def _extract_base_model_names(df: pd.DataFrame, include_ensembles: bool = False) -> List[str]:
    """Return display names for models in the desired order, renaming via _MODEL_RENAME."""
    available = {c.replace(' C-Index', '') for c in df.columns
                 if c.endswith(' C-Index') and not c.endswith(' C-Index SE')}
    models = [m for m in _DESIRED_MODELS if m in available]
    if not include_ensembles:
        models = [m for m in models if not _is_ensemble(m)]
    return [_MODEL_RENAME.get(m, m) for m in models]

def _resolve_se(df: pd.DataFrame, model: str) -> pd.Series:
    se_col = f'{model} C-Index SE'
    if se_col in df.columns:
        return pd.to_numeric(df[se_col], errors='coerce').fillna(0.0)
    if 'N' in df.columns:
        cvals = pd.to_numeric(df[f'{model} C-Index'], errors='coerce')
        nvals = pd.to_numeric(df['N'], errors='coerce').clip(lower=1)
        return np.sqrt((cvals * (1.0 - cvals)) / nvals).fillna(0.0)
    return pd.Series(0.0, index=df.index)

def _get_pvalue(df: pd.DataFrame, model_a: str, model_b: str) -> pd.Series:
    for col in [f'P-Value ({model_a} vs {model_b})', f'P-Value ({model_b} vs {model_a})']:
        if col in df.columns:
            return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(np.nan, index=df.index)

def _prepare_wide(df: pd.DataFrame, models: List[str], min_events: int) -> pd.DataFrame:
    if 'Total Events' in df.columns:
        df = df[pd.to_numeric(df['Total Events'], errors='coerce') >= min_events].copy()
    df['EventLabel'] = df['Event'].apply(_prettify_event)
    df = df.dropna(subset=['EventLabel'])
    # Drop rows where all model C-Indices are NaN
    ci_cols = [f'{m} C-Index' for m in models if f'{m} C-Index' in df.columns]
    df = df.dropna(subset=ci_cols, how='all')
    df['Category'] = df['EventLabel'].map(lambda x: _EVENT_CATEGORIES.get(x, 99))
    df['BestScore'] = df[ci_cols].max(axis=1)
    df = df.sort_values(by=['Category', 'BestScore'], ascending=[True, False])
    return df.reset_index(drop=True)

# --- Horizontal bar chart (all models, grouped by disease) ---
def plot_bar_all_models(df: pd.DataFrame, models: List[str], out_path: Path,
                        ref_model: str = 'DXA Tabular') -> None:
    """Grouped horizontal bar chart: one row per disease, one bar per model."""
    # Rename CSV columns to display names so lookups use display names throughout
    rename_map = {}
    for csv_name, display_name in _MODEL_RENAME.items():
        for suffix in [' C-Index', ' C-Index SE']:
            old = f'{csv_name}{suffix}'
            new = f'{display_name}{suffix}'
            if old in df.columns:
                rename_map[old] = new
    df = df.rename(columns=rename_map)
    work = _prepare_wide(df, models, min_events=0)
    if work.empty:
        print("  No data for bar chart.")
        return

    n_models = len(models)
    colors = [_model_color(m, i) for i, m in enumerate(models)]

    # Build y positions with category gaps
    y_positions, labels, cat_centers, dividers = [], [], [], []
    y_curr = 0
    for cat, group in work.groupby('Category', sort=False):
        cat_start = y_curr
        for _, row in group.iterrows():
            y_positions.append(y_curr)
            labels.append(row['EventLabel'])
            y_curr += 1
        cat_centers.append(((cat_start + y_curr - 1) / 2.0, cat))
        dividers.append(y_curr - 0.5)
        y_curr += 0.6
    dividers = dividers[:-1]
    y = np.array(y_positions)

    bar_h = min(0.75 / n_models, 0.18)
    offsets = np.linspace(-(n_models - 1) / 2 * bar_h, (n_models - 1) / 2 * bar_h, n_models)

    fig_h = max(8.0, 0.45 * y_curr + 1.5)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))

    ax.axvline(0.5, color='#AAAAAA', linestyle=':', linewidth=1.5, zorder=1)
    for d in dividers:
        ax.axhline(d, color='#EEEEEE', linestyle='-', linewidth=2, zorder=0)

    bars_per_model = {}
    for i, (model, color, offset) in enumerate(zip(models, colors, offsets)):
        ci_col = f'{model} C-Index'
        if ci_col not in work.columns:
            continue
        vals = pd.to_numeric(work[ci_col], errors='coerce').to_numpy()
        ses = _resolve_se(work, model).to_numpy()
        ax.barh(y + offset, vals, height=bar_h, color=color, alpha=0.85,
                xerr=ses, ecolor='#333333', capsize=2, zorder=3)
        bars_per_model[model] = (vals, ses)

    all_vals = np.concatenate([v for v, _ in bars_per_model.values()])
    all_ses  = np.concatenate([s for _, s in bars_per_model.values()])
    xmin = min(0.48, float(np.nanmin(all_vals - all_ses)) - 0.02)
    xmax = min(1.02, float(np.nanmax(all_vals + all_ses)) + 0.04)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.6, y_curr - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('C-index', fontsize=9)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.9)
    ax.grid(axis='y', visible=False)

    for y_cen, cat in cat_centers:
        cat_name = _CATEGORY_NAMES.get(cat, 'Other')
        ax.text(xmax - 0.002, y_cen, cat_name, ha='right', va='center',
                fontsize=7.5, fontweight='bold', color='#999999', style='italic')

    patches = [mpatches.Patch(color=_model_color(m, i), label=m) for i, m in enumerate(models)]
    ax.legend(handles=patches, frameon=False, loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")

# --- Radar chart (all models overlaid) ---
def plot_radar_all_models(df: pd.DataFrame, models: List[str], out_path: Path) -> None:
    work = _prepare_wide(df, models, min_events=0)
    if work.empty:
        return

    ci_cols = [f'{m} C-Index' for m in models if f'{m} C-Index' in work.columns]
    valid_models = [c.replace(' C-Index', '') for c in ci_cols]
    mat = work.set_index('EventLabel')[ci_cols].rename(columns=lambda c: c.replace(' C-Index', ''))
    mat = mat.dropna(how='all')
    if len(mat) < 3:
        print("  Too few events for radar chart.")
        return

    # Sort by category then best score
    mat['_cat'] = mat.index.map(lambda x: _EVENT_CATEGORIES.get(x, 99))
    mat['_best'] = mat[valid_models].max(axis=1)
    mat = mat.sort_values(['_cat', '_best'], ascending=[True, False]).drop(columns=['_cat', '_best'])

    labels = mat.index.tolist()
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.plot(angles, [0.5] * (N + 1), color='#AAAAAA', linestyle=':', linewidth=1.5, zorder=1)

    all_vals_flat = []
    for i, model in enumerate(valid_models):
        color = _model_color(model, i)
        vals = mat[model].tolist() + mat[model].tolist()[:1]
        all_vals_flat.extend(mat[model].dropna().tolist())
        ax.plot(angles, vals, color=color, linewidth=2.5, label=model, zorder=3)
        ax.fill(angles, vals, color=color, alpha=0.10, zorder=2)

    r_min = min(0.48, min(all_vals_flat) - 0.03) if all_vals_flat else 0.45
    r_max = min(1.0, max(all_vals_flat) + 0.05) if all_vals_flat else 1.0
    ax.set_ylim(r_min, r_max)

    r_ticks = np.linspace(r_min, r_max, 4)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{v:.2f}" for v in r_ticks], fontsize=9, color='black')
    ax.set_rlabel_position(0)

    ax.set_thetagrids(np.degrees(angles[:-1]), [''] * N)
    ax.tick_params(axis='x', pad=0)
    ax.spines['polar'].set_visible(False)
    ax.grid(color='#CCCCCC', linestyle='--', linewidth=1)

    label_r = r_max + (r_max - r_min) * 0.25
    cat_angles: Dict[int, list] = {}
    eps = 12
    for i, (angle_rad, label) in enumerate(zip(angles[:-1], labels)):
        cat = _EVENT_CATEGORIES.get(label, 99)
        cat_angles.setdefault(cat, []).append(angle_rad)
        angle_deg = np.degrees(angle_rad) % 360
        wrapped = '\n'.join(textwrap.wrap(label, width=13))
        if abs(angle_deg - 90) < eps:    ha, va = 'center', 'bottom'
        elif abs(angle_deg - 270) < eps: ha, va = 'center', 'top'
        elif angle_deg < 180:            ha, va = 'left',   'center'
        else:                            ha, va = 'right',  'center'
        ax.text(angle_rad, label_r, wrapped, ha=ha, va=va, fontsize=8,
                color='black', linespacing=1.2)

    cat_r = r_max + (r_max - r_min) * 0.55
    for cat, angs in cat_angles.items():
        if cat == 99: continue
        mean_ang = np.mean(angs)
        cat_name = _CATEGORY_NAMES.get(cat, '').upper()
        angle_deg = np.degrees(mean_ang) % 360
        va = 'bottom' if angle_deg < 180 else 'top'
        ax.text(mean_ang, cat_r, cat_name, ha='center', va=va, fontsize=11,
                fontweight='bold', color='#888888', alpha=0.6)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=min(3, len(valid_models)),
              frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")

# --- Pairwise scatter: model A vs model B per disease ---
def plot_pairwise_scatter(df: pd.DataFrame, model_a: str, model_b: str, out_path: Path) -> None:
    ci_a, ci_b = f'{model_a} C-Index', f'{model_b} C-Index'
    if ci_a not in df.columns or ci_b not in df.columns:
        print(f"  Skipping scatter ({model_a} vs {model_b}): columns missing.")
        return
    work = df[['Event', ci_a, ci_b]].copy()
    work['EventLabel'] = work['Event'].apply(_prettify_event)
    work = work.dropna(subset=['EventLabel', ci_a, ci_b])
    work[ci_a] = pd.to_numeric(work[ci_a], errors='coerce')
    work[ci_b] = pd.to_numeric(work[ci_b], errors='coerce')
    work = work.dropna()
    if work.empty:
        return

    a_vals, b_vals = work[ci_a].to_numpy(), work[ci_b].to_numpy()
    color_a = _model_color(model_a, 0)
    color_b = _model_color(model_b, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    lims = [min(a_vals.min(), b_vals.min()) - 0.02, max(a_vals.max(), b_vals.max()) + 0.02]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.4)

    categories = work['EventLabel'].map(lambda x: _EVENT_CATEGORIES.get(x, 99))
    scatter = ax.scatter(a_vals, b_vals, c=categories, cmap='tab10', s=60, zorder=3, edgecolors='white', linewidths=0.5)

    ax.set_xlabel(f'{model_a} C-index', fontsize=9)
    ax.set_ylabel(f'{model_b} C-index', fontsize=9)
    ax.set_title(f'{model_a} vs {model_b}', fontsize=10, fontweight='bold')
    ax.set_xlim(lims); ax.set_ylim(lims)

    for _, row in work.iterrows():
        ax.annotate(row['EventLabel'], (row[ci_a], row[ci_b]),
                    fontsize=6, alpha=0.7, textcoords='offset points', xytext=(4, 2))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {out_path}")

# --- Main ---
DEFAULT_RESULTS_CSV = Path('/path/to/cox_ttest_results.csv')
DEFAULT_OUTPUT_DIR  = Path('/path/to/project/results/cox_plots')

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Plot Cox regression comparison results')
    parser.add_argument('--results-csv', type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--min-events', type=int, default=150)
    parser.add_argument('--include-ensembles', action='store_true',
                        help='Include ensemble models (X + Covariates) in plots')
    parser.add_argument('--ref-model', default='DXA Tabular',
                        help='Reference model for significance stars in bar chart')
    args = parser.parse_args()

    if not args.results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if 'Total Events' in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df['Total Events'], errors='coerce') >= args.min_events]
        print(f"Filtered to events with N >= {args.min_events}: {before} → {len(df)} rows")

    models = _extract_base_model_names(df, include_ensembles=True)
    print(f"Models detected ({len(models)}): {models}")

    if not models:
        print("No model C-Index columns found. Exiting.")
        return

    print("\nGenerating Cox C-index bar chart...")
    plot_bar_all_models(df, models, args.out_dir / 'cox_bar_all_models.png', ref_model=args.ref_model)
    print(f"\nDone. Plots saved to {args.out_dir}")

if __name__ == '__main__':
    main()
