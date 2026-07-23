"""
plot_fig2_heatmap.py — Figure 2 redesign (Virchow Fig. 2a idiom)

Panel a : Disease-classification AUROC heatmap for diagnoses available in both
          HPP and UK Biobank. Rows = 4 covariate-adjusted model arms; columns =
          diseases. AUROC printed in every cell; cell shade encodes within-disease
          model rank.
Panel b : Cohort-specific disease AUROC heatmaps for diagnoses shown only in HPP
          or only in UK Biobank.
Panel c : HPP chronological-age prediction, MAE bars.
Panel d : HPP continuous physiological biomarkers, Pearson r bars with SE error
          bars and FDR-corrected DeepDXA-vs-all-comparators significance markers.

No Cox/KM (lives in Fig 3).

Usage:
  python plot_fig2_heatmap.py
  python plot_fig2_heatmap.py --verify           # print matrices, skip render
  python plot_fig2_heatmap.py --out path.png
"""
import argparse, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from common.plot_style import MODEL_COLORS, _draw_sig_markers
except Exception:  # pragma: no cover — fallback if import chain breaks
    MODEL_COLORS = {"lejepa": "#083c7d", "ensemble": "#083c7d",
                    "dino": "#7fb9dc", "tabular": "#8ccbb3",
                    "covariates": "#bdbdbd"}
    _draw_sig_markers = None

# Arm colors = the heatmap rank-shade ramp (Blues at 0.95/0.66/0.40/0.16): DeepDXA
# boldest, then DINOv3, then DXA-Tabular (features), then Age/sex/BMI (covariates)
# lightest — so the bar colors echo the within-disease rank shading. Local override.
MODEL_COLORS = {**MODEL_COLORS, "lejepa": "#083c7d", "ensemble": "#083c7d",
                "dino": "#7fb9dc", "tabular": "#8ccbb3",
                "covariates": "#bdbdbd"}

FIG_WIDTH_CM = 18.0
PT_MAIN = 10
PT_SMALL = 9
PT_HEATMAP_ANN = 9
PT_HEATMAP_COMPACT = 7.5
PT_PANEL = 12

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root
DEXA  = os.environ.get('DEXA_ROOT', os.path.dirname(_ROOT))  # parent dir holds external (non-distributed) inputs
# Canonical Figure 2 disease panels use whole-body DXA scan embeddings only:
# [bone+tissue] | [age/sex/BMI covariates], with the per-block differential
# penalisation tuned in CV. Regional-pool, two-pen, and bone-pool tables are
# ablations and must be opted into explicitly via FIG2_ALLOW_ABLATION=1.
FIG2_SOURCE_REGIME = 'diffpentuned'
# In-repo canonical inputs (repo-relative):
SUPP_A    = os.environ.get('FIG2_HPP_TABLE', os.path.join(_ROOT, 'tables', 'supp_tableA_disease_auc_4arm_diffpentuned.csv'))
UKBB_AUC  = os.environ.get('FIG2_UKBB_TABLE', os.path.join(_ROOT, 'tables', 'ukbb_pca0_diffpen_summary.csv'))
AGE_MAE_CSV  = os.environ.get('FIG2_AGE_MAE', os.path.join(_ROOT, 'tables', 'age_mae_imaging_only_wholebody.csv'))  # per-model age (imaging-only, multi-visit)
OUT_DEF   = os.path.join(_ROOT, 'figures', 'fig2_disease_heatmap.png')
# External inputs (participant-level or not distributed — supply via env / DEXA_ROOT):
SUPP_B    = os.environ.get('FIG2_REG_TABLE', f'{DEXA}/supp_tableB_systems_mv_wholebody.csv')  # systems biomarkers; panels c & d
AGE_PRED  = os.environ.get('FIG2_AGE_PRED', f'{DEXA}/age_prediction_ukbb/ukbb_age_predictions_with_visits.csv')
HPP_AGE_PRED = os.environ.get('FIG2_HPP_AGE_PRED', os.path.join(_ROOT, 'tables', 'tableD_bioage_predictions_hpp.csv'))  # per-participant; not shipped

# ── Model rows (top → bottom); DeepDXA is the hero ──────────────────────────────
MODELS = [
    ('lejepa',     'LeDXA + cov'),
    ('dino',       'DINOv3 + cov'),
    ('tabular',    'DXA-Tab + cov'),
    ('covariates', 'Age, sex & BMI\n(covariates)'),
]
# Per-cohort wide-column lookup: key -> mean column in that cohort's table.
HPP_COLS = {  # exporter labels DINOv3 as "DINOv3"; relabelled to DINOv3 on the row axis
    'lejepa':     'DXA-FM + Covariates_mean',
    'dino':       'DINOv3 + Covariates_mean',
    'tabular':    'DXA Tabular + Covariates_mean',
    'covariates': 'Covariates (age/sex/BMI)_mean',
}
UKBB_COLS = {  # NOTE: UKBB summary labels DINOv3 as "DINOv3"; we relabel to DINOv3.
    'lejepa':     'DXA-FM + Covariates_mean',
    'dino':       'DINOv3 + Covariates_mean',
    'tabular':    'DXA Tabular + Covariates_mean',
    'covariates': 'Covariates (age/sex/BMI)_mean',
}
# Table-B arm -> mean/SE columns (regression panel + age bars).
# Imaging-only arms (no covariate fusion): DeepDXA/DINOv3/DXA-Tabular embeddings
# alone, plus a covariates-only baseline bar. Source table summarises 10 seeds.
B_COLS = {
    'lejepa':     ('DeepDXA_mean',      'DeepDXA_SE'),
    'dino':       ('DINOv3_mean',       'DINOv3_SE'),
    'tabular':    ('DXA Tabular_mean',  'DXA Tabular_SE'),
    'covariates': ('Covariates_mean',   'Covariates_SE'),
}
B_PCOLS = {  # comparison -> (pcol, mean-col-of-comparator) for sig markers
    'covariates': ('P_DeepDXA_vs_Cov_adj',  'Covariates_mean'),
    'tabular':    ('P_DeepDXA_vs_Tab_adj',  'DXA Tabular_mean'),
    'dino':       ('P_DeepDXA_vs_DINO_adj', 'DINOv3_mean'),
}

# ── Curated columns for panel c ─────────────────────────────────────────────────
# ── Panel c — diseases present in BOTH cohorts (paired HPP/UKBB, same name) ──────
# Each col: (hpp_display, ukbb_key, hpp_label, ukbb_label).
# hpp_label shown above HPP sub-grid; ukbb_label shown below UKBB sub-grid.
# When phenotype names differ between cohorts, use the cohort-specific term.
# Chronic disorders with good HPP AUROC and an exact/broadly equivalent UKBB endpoint.
REPLICATED = [
    ('Anemia',               'dis__anemia',                'Anemia',          'Anemia'),
    ('Anxiety',              'dis__anxiety',               'Anxiety',         'Anxiety'),
    ('Diabetes',             'dis__diabetes',              'Diabetes',        'Diabetes'),
    ('Gallstones',           'dis__gallstone',             'Gallstones',      'Gallstones'),
    ('Hyperlipidemia',       'dis__hyperlipidemia',        'Hyperlipidemia',  'Hyperlipidemia'),
    ('Hypertension',         'dis__hypertension',          'Hypertension',    'Hypertension'),
    ('Hyperthyroidism',      'dis__hyperthyroidism',       'Hyperthyroidism', 'Hyperthyroidism'),
    ('Osteoarthritis',       'dis__osteoarthritis',        'Osteoarthritis',  'Osteoarthritis'),
    ('Urinary Stones',       'dis__kidney_stones',         'Kidney stones',   'Kidney stones'),
]
HPP_SPECIFIC = [  # Reserved for paired-panel HPP-specific columns
]
UKBB_SPECIFIC = [  # Reserved for paired-panel UKBB-specific columns
]
# Column blocks in display order: (list, group-key, header label — no "win")
COLUMN_BLOCKS = [
    (REPLICATED,   'repl', 'Replicated disorders'),
    (HPP_SPECIFIC, 'hpp',  'HPP-specific Wins'),
    (UKBB_SPECIFIC,'ukbb', 'UKBB-specific Wins'),
]

# ── Cohort-specific gains (compact per-cohort heatmaps) ─────────────
# DeepDXA gains highlighted in one cohort after the replicated set is shown above.
HPP_ONLY = [   # HPP gain examples
    ('Fatty Liver',         'MASLD'),
    ('Fibromyalgia',        'Fibromyalgia'),
    ('Migraine',            'Migraine'),
    ('Peptic Ulcer',        'Peptic Ulcer'),
    ('Prediabetes',         'Prediabetes'),
    ('Sleep Apnea',         'Sleep Apnea'),
]
UKBB_ONLY = [  # UKBB gain examples
    ('dis__sleep_disorders',      'Sleep Disorders'),
    ('dis__liver_disease',        'Liver Disease'),
    ('dis__copd',                 'COPD'),
    ('dis__asthma',               'Asthma'),
    ('dis__gout',                 'Gout'),
]

# ── Continuous-trait order handled dynamically (sorted by DeepDXA r) ─────────────

# Heatmap color encodes RANK WITHIN A DISEASE (winner darkest), not absolute AUROC.
_CMAP = matplotlib.colormaps['Blues']
_REL_FLOOR = 0.18  # lightest shade (worst model in a column)


# ── Data loading ────────────────────────────────────────────────────────────────

def load_tables():
    hpp = pd.read_csv(SUPP_A)
    hpp = hpp.set_index(hpp.columns[0])           # 'Disease'
    ukbb = pd.read_csv(UKBB_AUC).set_index('disease')
    return hpp, ukbb


def _auc(df, idx, col):
    if idx in df.index and col in df.columns:
        v = pd.to_numeric(df.loc[idx, col], errors='coerce')
        return float(v) if np.isfinite(v) else None
    return None


# ── Significance vs covariates (binary overlay) ───────────────────────────────────
# For each (cohort, disease_key, imaging_arm): True if arm sig. beats covariates
# (two-tailed Wilcoxon FDR<0.05). Drawn as a bold border on the cell.
_PAIRS_CSV = os.environ.get('FIG2_PAIRS', os.path.join(_ROOT, 'tables', 'disease_pairwise_diffpentuned.csv'))
_HPP_KEY_ALIAS = {
    'Gallstone disease':    'Gallstones',
    'Urinary tract stones': 'Urinary Stones',
    'Fatty Liver Disease':  'Fatty Liver',
    'Urinary tract infection': 'UTI',
}
ALPHA = 0.05


def _assert_canonical_sources():
    if os.environ.get('FIG2_ALLOW_ABLATION') == '1':
        return
    checked = {
        'FIG2_HPP_TABLE': SUPP_A,
        'FIG2_UKBB_TABLE': UKBB_AUC,
        'FIG2_PAIRS': _PAIRS_CSV,
    }
    banned = ('regionpool', 'twopen')
    problems = []
    for name, path in checked.items():
        base = os.path.basename(path).lower()
        if any(token in base for token in banned):
            problems.append(f'{name}={path}')
        if 'diffpen' not in base and 'bonepool' not in base and 'lsw' not in base:
            problems.append(f'{name}={path} (missing regime marker)')
    if problems:
        joined = '\n  '.join(problems)
        raise RuntimeError(
            'Figure 2 must use canonical sources (diffpentuned or bonepool_lsw).\n'
            f'Unexpected source(s):\n  {joined}\n'
            'Set FIG2_ALLOW_ABLATION=1 only for an intentional ablation render.'
        )


def _load_sig_vs_cov():
    """Returns {(cohort, display_key, arm): bool} — True where arm sig. > covariates."""
    out = {}
    if not os.path.exists(_PAIRS_CSV):
        return out
    df = pd.read_csv(_PAIRS_CSV)
    for _, r in df.iterrows():
        # only rows involving covariates as the weaker model
        if 'covariates' not in (r.model_a, r.model_b):
            continue
        imaging = r.model_a if r.model_b == 'covariates' else r.model_b
        mu_img = r.mean_a if r.model_b == 'covariates' else r.mean_b
        mu_cov = r.mean_b if r.model_b == 'covariates' else r.mean_a
        sig = (np.isfinite(r.p_two_tailed_adj) and
               r.p_two_tailed_adj < ALPHA and
               mu_img > mu_cov)
        out[(str(r.cohort), str(r.key), str(imaging))] = bool(sig)
    return out


_SIG_VS_COV = _load_sig_vs_cov()


def _load_sig_all():
    """Full pairwise dict: {(cohort, key, model_a, model_b): bool} — True = model_a sig. beats model_b."""
    out = {}
    if not os.path.exists(_PAIRS_CSV):
        return out
    df = pd.read_csv(_PAIRS_CSV)
    for _, r in df.iterrows():
        key, cohort = str(r.key), str(r.cohort)
        sig_ab = (np.isfinite(r.p_two_tailed_adj) and r.p_two_tailed_adj < ALPHA
                  and r.mean_a > r.mean_b)
        sig_ba = (np.isfinite(r.p_two_tailed_adj) and r.p_two_tailed_adj < ALPHA
                  and r.mean_b > r.mean_a)
        out.setdefault((cohort, key, str(r.model_a), str(r.model_b)), bool(sig_ab))
        out.setdefault((cohort, key, str(r.model_b), str(r.model_a)), bool(sig_ba))
    return out


_SIG_ALL = _load_sig_all()


def _dominance_count(cohort, key, model_key):
    """How many of the other three model arms does model_key significantly beat (FDR < 0.05)?"""
    k = _HPP_KEY_ALIAS.get(str(key), str(key)) if cohort == 'HPP' else str(key)
    return sum(_SIG_ALL.get((cohort, k, model_key, om), False)
               for om, _ in MODELS if om != model_key)


def _build_dom_mat(cohort, keys):
    """Return (n_models × n_keys) int matrix: dominance count per (model, disease) cell."""
    mat = np.zeros((len(MODELS), len(keys)), dtype=int)
    for j, key in enumerate(keys):
        for i, (mkey, _) in enumerate(MODELS):
            mat[i, j] = _dominance_count(cohort, key, mkey)
    return mat


def _build_win_tensor(cohort, keys):
    """Return (n_models × n_models × n_keys) bool array: wins[a,b,j] = model a sig. beats model b for disease j."""
    n_m, n_k = len(MODELS), len(keys)
    tensor = np.zeros((n_m, n_m, n_k), dtype=bool)
    for j, key in enumerate(keys):
        k = _HPP_KEY_ALIAS.get(str(key), str(key)) if cohort == 'HPP' else str(key)
        for a, (ma, _) in enumerate(MODELS):
            for b, (mb, _) in enumerate(MODELS):
                if a != b:
                    tensor[a, b, j] = _SIG_ALL.get((cohort, k, ma, mb), False)
    return tensor


def _hasse_rank(wins_j, finite_rows):
    """Iterative Hasse-diagram ranking for one disease column.
    Tier 0: models not significantly beaten by anyone still in play.
    Tier 1: same, from the remaining. Ties automatically share a tier.
    Returns {row_idx: tier} for finite rows only."""
    ranks, remaining = {}, set(finite_rows)
    tier = 0
    while remaining:
        top = [i for i in remaining if not any(wins_j[k, i] for k in remaining if k != i)]
        if not top:            # cycle fallback (shouldn't occur in practice)
            for i in remaining:
                ranks[i] = tier
            break
        for i in top:
            ranks[i] = tier
        remaining -= set(top)
        tier += 1
    return ranks


def _col_sig_vs_cov(cohort, key):
    """Bool list (per model row) — True where that arm sig. beats covariates."""
    key = str(key)
    if cohort == 'HPP':
        key = _HPP_KEY_ALIAS.get(key, key)
    return [_SIG_VS_COV.get((cohort, key, mk), False) for mk, _ in MODELS]


# Stub kept so build_matrix callers don't need changes (returns None = no topset)
def _col_topset(cohort, key):
    return None


def _col_gain(hpp, ukbb, h, u):
    """Mean DeepDXA−DXA-tabular AUROC gain across available cohorts, for column ordering."""
    gains = []
    if h:
        l, t = _auc(hpp, h, HPP_COLS['lejepa']), _auc(hpp, h, HPP_COLS['tabular'])
        if l is not None and t is not None:
            gains.append(l - t)
    if u:
        l, t = _auc(ukbb, u, UKBB_COLS['lejepa']), _auc(ukbb, u, UKBB_COLS['tabular'])
        if l is not None and t is not None:
            gains.append(l - t)
    return sum(gains) / len(gains) if gains else -1e9


def build_matrix(hpp, ukbb):
    """Return columns list, group boundaries/headers, and per-cohort AUC matrices.
    Replicated diseases are ordered by descending HPP DeepDXA AUROC; any
    cohort-specific blocks remain ordered by descending DeepDXA−tabular gain."""
    defs = []
    for block, grp, _ in COLUMN_BLOCKS:
        block_defs = [(h, u, hl, ul, grp) for (h, u, hl, ul) in block]
        if grp == 'repl':
            block_defs.sort(
                key=lambda d: _auc(hpp, d[0], HPP_COLS['lejepa']) or -1e9,
                reverse=True,
            )
        else:
            block_defs.sort(key=lambda d: _col_gain(hpp, ukbb, d[0], d[1]), reverse=True)
        defs.extend(block_defs)
    hpp_col_labels  = [hl for _, _, hl, _, _ in defs]
    ukbb_col_labels = [ul for _, _, _, ul, _ in defs]
    cols      = [(hl, grp) for _, _, hl, _, grp in defs]
    hpp_keys  = [h for h, _, _, _, _ in defs]
    ukbb_keys = [u for _, u, _, _, _ in defs]

    hpp_mat, ukbb_mat = [], []
    for mkey, _ in MODELS:
        hpp_mat.append([_auc(hpp, hk, HPP_COLS[mkey]) if hk else None for hk in hpp_keys])
        ukbb_mat.append([_auc(ukbb, uk, UKBB_COLS[mkey]) if uk else None for uk in ukbb_keys])

    hpp_ts  = [_col_sig_vs_cov('HPP',  hk) if hk else [False]*len(MODELS) for hk in hpp_keys]
    ukbb_ts = [_col_sig_vs_cov('UKBB', uk) if uk else [False]*len(MODELS) for uk in ukbb_keys]

    # block boundaries (start index) and header labels, skipping empty blocks
    bounds, start = [], 0
    for block, _, header in COLUMN_BLOCKS:
        if block:
            bounds.append((start, start + len(block), header))
            start += len(block)
    return (cols, bounds, np.array(hpp_mat, dtype=object), np.array(ukbb_mat, dtype=object),
            hpp_ts, ukbb_ts, hpp_col_labels, ukbb_col_labels)


# ── Panel c — heatmap ────────────────────────────────────────────────────────────

_LEJEPA_IDX  = next(i for i, (mk, _) in enumerate(MODELS) if mk == 'lejepa')
_DINO_IDX    = next(i for i, (mk, _) in enumerate(MODELS) if mk == 'dino')
_TABULAR_IDX = next(i for i, (mk, _) in enumerate(MODELS) if mk == 'tabular')


def _draw_heatmap(ax, mat, col_labels, seps, cohort_label, show_xlabels,
                  topset=None, show_xlabels_top=False, win_tensor=None,
                  ann_fontsize=PT_HEATMAP_ANN):
    """Cell shading: within-disease AUROC rank (darkest = highest).
    The winning arm is darkest blue. Other arms statistically tied with the
    winner stay colored in lighter blues. Arms significantly worse than the
    winner are grey."""
    n_rows, n_cols = mat.shape
    if n_cols == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_rows)
        ax.invert_yaxis()
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels([lbl for _, lbl in MODELS], fontsize=PT_SMALL)
        ax.set_xticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if cohort_label:
            ax.text(0.0, 1.03, cohort_label.replace('\n', ' '),
                    transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=PT_MAIN, fontweight='bold', clip_on=False)
        return
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    rank_shades = [_CMAP(0.95), _CMAP(0.66), _CMAP(0.40), _CMAP(0.16)]

    for j in range(n_cols):
        col_vals   = [mat[i, j] for i in range(n_rows)]
        finite_rows = sorted([i for i in range(n_rows) if col_vals[i] is not None],
                             key=lambda i: -col_vals[i])

        # The best-scoring arm anchors the top set; any arm NOT significantly beaten
        # by it (two-sided Wilcoxon FDR>=0.05) joins the top set. Top-set arms are
        # colored and shaded by score rank. Statistically worse arms are greyed out.
        winner_row = finite_rows[0] if finite_rows else None
        top_set = []
        if winner_row is not None:
            # Walk arms in descending score; an arm joins the coloured top set only if the
            # winner does NOT significantly beat it. Stop at the first significantly-beaten
            # arm so the coloured set is a contiguous top-by-score block — this prevents a
            # lower-scoring arm being coloured while a higher-scoring one is not.
            for i in finite_rows:
                if (i == winner_row or win_tensor is None
                        or not bool(win_tensor[winner_row, i, j])):
                    top_set.append(i)              # tied with (not beaten by) the best
                else:
                    break                          # first beaten arm → it + all lower are "not coloured"
        top_pos = {i: p for p, i in enumerate(top_set)}   # score rank within top set

        for i in range(n_rows):
            v = mat[i, j]
            if v is None:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor='#f2f2f2',
                                       edgecolor='white', lw=0.6, zorder=1))
                ax.text(j + 0.5, i + 0.5, '–', ha='center', va='center',
                        color='#bbbbbb', fontsize=ann_fontsize, zorder=2)
                continue
            if i in top_pos:                       # colored, shaded by score rank
                pos = top_pos[i]
                shade = rank_shades[min(pos, 3)]
                txt = 'white' if pos <= 1 else '#333333'
            else:                                  # significantly worse than winner
                shade = '#eeeeee'
                txt = '#555555'
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=shade,
                                   edgecolor='white', lw=0.6, zorder=1))
            ax.text(j + 0.5, i + 0.5, f'{v:.3f}', ha='center', va='center',
                    color=txt, fontsize=ann_fontsize, fontweight='normal', zorder=3)

    for s in (seps or []):
        if 0 < s < n_cols:
            ax.plot([s, s], [0, n_rows], color='#444', lw=1.0, ls=(0, (3, 2)), zorder=5)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([lbl for _, lbl in MODELS], fontsize=PT_SMALL)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    if show_xlabels:
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=PT_SMALL)
    else:
        ax.set_xticklabels([])
    if show_xlabels_top:
        ax.xaxis.set_tick_params(which='both', labeltop=True, labelbottom=False)
        ax.set_xticklabels(col_labels, rotation=45, ha='left', fontsize=PT_SMALL)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if cohort_label:
        ax.text(0.0, 1.03, cohort_label.replace('\n', ' '),
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=PT_MAIN, fontweight='bold', clip_on=False)


def draw_panel_a(ax_hpp, ax_ukbb, hpp, ukbb):
    cols, bounds, hpp_mat, ukbb_mat, hpp_ts, ukbb_ts, hpp_lbls, ukbb_lbls = build_matrix(hpp, ukbb)
    defs = [(h, u, hl, ul, grp) for block, grp, _ in COLUMN_BLOCKS
            for (h, u, hl, ul) in block]
    hpp_keys  = [h for h, _, _, _, _ in defs]
    ukbb_keys = [u for _, u, _, _, _ in defs]
    hpp_wt  = _build_win_tensor('HPP',  hpp_keys)
    ukbb_wt = _build_win_tensor('UKBB', ukbb_keys)
    _draw_heatmap(ax_hpp,  hpp_mat,  hpp_lbls,  None, 'HPP\n(internal, held-out)',
                  show_xlabels=False, topset=hpp_ts, win_tensor=hpp_wt)
    _draw_heatmap(ax_ukbb, ukbb_mat, ukbb_lbls, None, 'UK Biobank\n(external)',
                  show_xlabels=True, topset=ukbb_ts, win_tensor=ukbb_wt)
    ax_hpp.text(-0.005, 1.16, 'c', transform=ax_hpp.transAxes, fontsize=PT_PANEL,
                fontweight='bold', va='bottom', ha='right', clip_on=False)


# ── Panels d, e — single-cohort notable wins (compact per-cohort heatmaps) ────────

def _single_mat(df, items, cols_map, cohort):
    """items: list of (idx, label). Return (labels, matrix[model x disease], topset)."""
    labels = [lbl for _, lbl in items]
    mat = []
    for mkey, _ in MODELS:
        mat.append([_auc(df, idx, cols_map[mkey]) for idx, _ in items])
    ts = [_col_sig_vs_cov(cohort, idx) for idx, _ in items]
    return labels, np.array(mat, dtype=object), ts


def _sort_items_by_gain(df, items, cols_map):
    """Order (idx, label) items by descending DeepDXA−tabular AUROC gain (largest left)."""
    def g(it):
        l, t = _auc(df, it[0], cols_map['lejepa']), _auc(df, it[0], cols_map['tabular'])
        return (l - t) if (l is not None and t is not None) else -1e9
    return sorted(items, key=g, reverse=True)


def draw_panel_b_single(ax_hpp, ax_ukbb, hpp, ukbb):
    hpp_items  = _sort_items_by_gain(hpp,  HPP_ONLY,  HPP_COLS)
    ukbb_items = _sort_items_by_gain(ukbb, UKBB_ONLY, UKBB_COLS)
    h_lbls, h_mat, h_ts = _single_mat(hpp,  hpp_items,  HPP_COLS,  'HPP')
    u_lbls, u_mat, u_ts = _single_mat(ukbb, ukbb_items, UKBB_COLS, 'UKBB')
    h_wt = _build_win_tensor('HPP',  [idx for idx, _ in hpp_items])
    u_wt = _build_win_tensor('UKBB', [idx for idx, _ in ukbb_items])
    _draw_heatmap(ax_hpp,  h_mat, h_lbls, None, '', show_xlabels=True, topset=h_ts,
                  win_tensor=h_wt, ann_fontsize=PT_HEATMAP_COMPACT)
    _draw_heatmap(ax_ukbb, u_mat, u_lbls, None, '', show_xlabels=True, topset=u_ts,
                  win_tensor=u_wt, ann_fontsize=PT_HEATMAP_COMPACT)
    ax_ukbb.set_yticklabels([])           # share model labels with the HPP block
    # captions sit ABOVE each sub-grid (titles), not below — keeps them attached to
    # panels d/e instead of drifting into panels a/b.
    ax_hpp.text(0.5, 1.04, 'HPP only', transform=ax_hpp.transAxes, ha='center',
                va='bottom', fontsize=PT_MAIN, fontweight='bold', clip_on=False)
    ax_ukbb.text(0.5, 1.04, 'UKBB only', transform=ax_ukbb.transAxes, ha='center',
                 va='bottom', fontsize=PT_MAIN, fontweight='bold', clip_on=False)
    ax_hpp.text(-0.02, 1.18, 'd', transform=ax_hpp.transAxes, fontsize=PT_PANEL,
                fontweight='bold', va='bottom', ha='right', clip_on=False)
    ax_ukbb.text(-0.02, 1.18, 'e', transform=ax_ukbb.transAxes, fontsize=PT_PANEL,
                 fontweight='bold', va='bottom', ha='right', clip_on=False)


# ── Panel a — age prediction ─────────────────────────────────────────────────────

def draw_panel_c_age(ax, supp_b):
    """Age prediction as a compact Virchow-style bar panel: MAE per model,
    SE error bars, value labelled at the bar end. No scatter."""
    # Imaging-only age: MAE and r both come from AGE_MAE_CSV (HPP), so the
    # age panel stays tied to one consistent run
    # (compute_age_mae_imaging_only.py). supp_b is unused here.
    order = ['lejepa', 'dino', 'tabular']           # no covariates baseline for age
    labs  = ['LeDXA', 'DINOv3', 'DXA-Tab']
    md = pd.read_csv(AGE_MAE_CSV)
    md = md[md['cohort'] == 'HPP'].set_index('model')
    vals = [float(md.loc[m, 'mae_yr_mean']) for m in order]
    ses  = [float(md.loc[m, 'mae_yr_se']) for m in order]
    cols = [MODEL_COLORS[m] for m in order]
    y = np.arange(len(order))[::-1] * 0.6           # bars close together
    ax.barh(y, vals, color=cols, height=0.5, xerr=ses, ecolor='#333',
            error_kw=dict(lw=0.7, capsize=1.8, capthick=0.7), zorder=3)
    for yi, m, v, s in zip(y, order, vals, ses):
        ax.text(v + s + 0.18, yi, f'{v:.2f}', va='center', ha='left', fontsize=PT_MAIN)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=PT_MAIN)
    ax.set_ylim(y.min() - 0.4, y.max() + 0.4)
    x_max = max(v + s for v, s in zip(vals, ses)) + 0.9
    ax.set_xlim(0, x_max)
    ax.set_xlabel('MAE (years)', fontsize=PT_MAIN)
    ax.tick_params(labelsize=PT_MAIN, length=0)
    ax.grid(axis='x', ls='--', color='#ccc', alpha=0.6)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.text(-0.42, 1.0, 'a', transform=ax.transAxes, fontsize=PT_PANEL,
            fontweight='bold', va='bottom', ha='right', clip_on=False)


# ── Panel b — continuous-trait regression bars ───────────────────────────────────

def draw_panel_c(ax, supp_b):
    df = supp_b[supp_b['Target'] != 'Age'].copy()
    df['__dd'] = pd.to_numeric(df[B_COLS['lejepa'][0]], errors='coerce')
    df = df.sort_values('__dd', ascending=True).reset_index(drop=True)  # ascending: best at top after invert
    traits = df['Target'].tolist()
    GROUP_SPACING = 2.0        # center-to-center between trait groups
    GROUP_SPAN    = 1.5        # total height the 4 bars occupy → ~0.5 gap between groups, thicker bars
    y = np.arange(len(traits)) * GROUP_SPACING   # generous whitespace between biomarkers

    n_m = len(MODELS)
    slot = GROUP_SPAN / n_m    # per-bar slot within the group
    offsets = np.linspace((n_m - 1) / 2 * slot, -(n_m - 1) / 2 * slot, n_m)
    bar_h = slot * 0.92        # tiny gap keeps SE caps from colliding

    # compute data range for tight x-axis
    all_vals, all_ends = [], []
    for mkey, _ in MODELS:
        mc, sc = B_COLS[mkey]
        v = pd.to_numeric(df[mc], errors='coerce').to_numpy()
        s = pd.to_numeric(df[sc], errors='coerce').fillna(0.0).to_numpy()
        all_vals.extend(v[np.isfinite(v)])
        all_ends.extend((v + s)[np.isfinite(v)])
    x_lo = max(0, np.min(all_vals) - 0.04)
    x_hi = np.max(all_ends) + 0.06   # room for sig markers

    bar_ends = {}
    for i, (mkey, _) in enumerate(MODELS):
        mc, sc = B_COLS[mkey]
        vals = pd.to_numeric(df[mc], errors='coerce').to_numpy()
        ses = pd.to_numeric(df[sc], errors='coerce').fillna(0.0).to_numpy()
        ax.barh(y + offsets[i], vals, height=bar_h, color=MODEL_COLORS[mkey],
                alpha=0.9, xerr=ses, ecolor='#333', capsize=1.8,
                error_kw=dict(lw=0.8, capthick=0.8), lw=0, zorder=3)
        for t, v, s in zip(traits, vals, ses):
            if np.isfinite(v):
                bar_ends[(t, mkey)] = float(v + (s if np.isfinite(s) else 0.0))

    # Significance marker per trait: DeepDXA FDR-significantly beats every
    # competing model. The displayed star tier reflects the weakest adjusted
    # q-value across the three pairwise contrasts.
    _P_COL_MAP = {
        'tabular':    'P_DeepDXA_vs_Tab_adj',
        'dino':       'P_DeepDXA_vs_DINO_adj',
        'covariates': 'P_DeepDXA_vs_Cov_adj',
    }
    def _stars(p):
        return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < 0.05 else ''
    lej_off = offsets[0]
    for yi, (_, row) in zip(y, df.iterrows()):
        deep_val = pd.to_numeric(row[B_COLS['lejepa'][0]], errors='coerce')
        pvals, beats_all = [], np.isfinite(deep_val)
        for mk, _ in MODELS[1:]:
            comp_val = pd.to_numeric(row[B_COLS[mk][0]], errors='coerce')
            beats_all = beats_all and np.isfinite(comp_val) and deep_val > comp_val
            p_col = _P_COL_MAP.get(mk)
            pvals.append(pd.to_numeric(row.get(p_col), errors='coerce') if p_col else np.nan)
        p = max(pvals) if pvals and all(np.isfinite(pv) for pv in pvals) else np.nan
        end = bar_ends.get((row['Target'], 'lejepa'))
        if beats_all and np.isfinite(p) and end is not None and _stars(p):
            ax.text(end + 0.012, yi + lej_off, _stars(p), va='center', ha='left',
                    fontsize=PT_MAIN, color='#222', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(traits, fontsize=PT_MAIN)
    ax.set_ylim(y.min() - GROUP_SPACING * 0.6, y.max() + GROUP_SPACING * 0.6)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel('Pearson r', fontsize=PT_MAIN)
    ax.tick_params(labelsize=PT_MAIN, length=0)
    ax.grid(axis='x', ls='--', color='#ccc', alpha=0.6)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.text(-0.18, 1.0, 'b', transform=ax.transAxes, fontsize=PT_PANEL,
            fontweight='bold', va='bottom', ha='right', clip_on=False)


# ── Legend + assembly ─────────────────────────────────────────────────────────────

def _arm_legend(fig, y):
    """Model-color key placed just above the bottom row, where the regression bars
    (panel d) carry no row labels of their own."""
    # Plain model names: this legend sits above panels c/d, which show the
    # imaging-only arms (no covariate fusion). Panel a/b carry their own "+ Cov"
    # row labels; the per-panel feature sets are spelled out in the caption.
    _LEGEND_LABELS = {'lejepa': 'LeDXA', 'dino': 'DINOv3',
                      'tabular': 'DXA-Tabular', 'covariates': 'Age, sex & BMI (covariates)'}
    handles = [Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]) for m, _ in MODELS]
    labels = [_LEGEND_LABELS[m] for m, _ in MODELS]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.53, y),
               ncol=4, frameon=False, fontsize=PT_MAIN, handlelength=1.4, columnspacing=1.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_DEF)
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()

    _assert_canonical_sources()
    hpp, ukbb = load_tables()
    supp_b = pd.read_csv(SUPP_B)              # imaging-only arms → panels c & d

    if args.verify:
        print(f"Figure 2 source regime: {FIG2_SOURCE_REGIME}")
        print(f"HPP table: {SUPP_A}")
        print(f"UKBB table: {UKBB_AUC}")
        print(f"Pairwise significance: {_PAIRS_CSV}")
        cols, bounds, hpp_mat, ukbb_mat, hpp_ts, ukbb_ts, hpp_lbls, ukbb_lbls = build_matrix(hpp, ukbb)
        print("Blocks:", [(h, f'{s}:{e}') for s, e, h in bounds])
        print("Columns:", [c for c, _ in cols])
        for label, mat in [('HPP', hpp_mat), ('UKBB', ukbb_mat)]:
            print(f"\n{label} AUC matrix (rows={[m for m,_ in MODELS]}):")
            for i, (mk, _) in enumerate(MODELS):
                print(f"  {mk:11s}", [None if v is None else round(v, 3) for v in mat[i]])
        print("\nTable B targets:", supp_b['Target'].tolist())
        return

    # Nature-style output: 18 cm wide, 10 pt main text, 8 pt only where the
    # heatmaps are dense, 12 pt bold panel letters, dpi=400, pdf.fonttype=42.
    plt.rcParams.update({'font.size': PT_MAIN, 'axes.labelsize': PT_MAIN,
                         'xtick.labelsize': PT_MAIN, 'ytick.labelsize': PT_MAIN})
    fig = plt.figure(figsize=(FIG_WIDTH_CM / 2.54, 11.4))
    # Health → disease narrative: age/biomarkers first, then disease discrimination.
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.5, 1.7, 1.0],
                     hspace=0.65, left=0.155, right=0.97, top=0.90, bottom=0.10)

    # row 0: panel a (age bars, narrow) | panel b (regression bars)
    top = outer[0].subgridspec(1, 2, width_ratios=[0.6, 1.6], wspace=0.72)
    ax_age = fig.add_subplot(top[0, 0])
    ax_reg = fig.add_subplot(top[0, 1])
    draw_panel_c_age(ax_age, supp_b)      # age (panel a) — imaging-only arms
    draw_panel_c(ax_reg, supp_b)          # regression (panel b)

    # legend centered just above the top row (relevant to panels a/b colors)
    row0_top = top.get_grid_positions(fig)[1][0]  # top edge of top row, fig coords
    _arm_legend(fig, y=row0_top + 0.068)

    # row 1: panel c — both-cohort heatmap (HPP over UKBB), color = dominance count
    mid = outer[1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.20)
    ax_hpp  = fig.add_subplot(mid[0, 0])
    ax_ukbb = fig.add_subplot(mid[1, 0])
    draw_panel_a(ax_hpp, ax_ukbb, hpp, ukbb)

    # row 2: panels d, e — single-cohort notable wins, full-width (no spacer cols)
    bot = outer[2].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.3)
    ax_h2 = fig.add_subplot(bot[0, 0])
    ax_u2 = fig.add_subplot(bot[0, 1])
    draw_panel_b_single(ax_h2, ax_u2, hpp, ukbb)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    base = os.path.splitext(args.out)[0]
    for ext in ('png', 'pdf'):
        fig.savefig(f'{base}.{ext}', dpi=400, facecolor='white',
                    transparent=False, bbox_inches='tight')
    print(f"Saved → {base}.png / .pdf")


if __name__ == '__main__':
    main()
