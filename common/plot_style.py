"""
plot_combined_figure.py

3-panel combined figure for the DXA-FM paper:
  Panel A — LP continuous targets (Pearson r)
  Panel B — Disease classification AUC  (top N by LeJEPA AUC, grouped by category)
  Panel C — Cox survival C-index
"""

import json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ukbb"))
from common.cox_utils import (COX_EVENT_SHORT_NAMES  as _COX_EVENT_SHORT_NAMES_IMPORTED,
                       COX_EVENT_CATEGORIES   as _COX_EVENT_CATEGORIES_IMPORTED,
                       COX_CATEGORY_NAMES_ABBREV as _COX_CATEGORY_NAMES_IMPORTED)

# ── PATHS ─────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
_REPO_ROOT  = os.path.dirname(_HERE)
_METADATA_DIR = os.path.join(_REPO_ROOT, "metadata")
RESULTS_DIR = "/data/hpp_labdata/Analyses/gilsa/results/comparison"
FIGURES_DIR = "/data/hpp_labdata/Analyses/gilsa/figures/comparison"

LP_CSV              = os.path.join(RESULTS_DIR, "lp_summary.csv")
LP_EXTENDED_CSV     = os.path.join(RESULTS_DIR, "lp_extended_summary.csv")
LP_RAW_CSV          = os.path.join(RESULTS_DIR, "lp_all_raw.csv")
LP_EXTENDED_RAW_CSV      = os.path.join(RESULTS_DIR, "lp_extended_raw.csv")
LP_NEW_BATCH_RAW_CSV     = os.path.join(RESULTS_DIR, "lp_new_batch_raw.csv")
LP_NIGHTINGALE_RAW_CSV   = os.path.join(RESULTS_DIR, "lp_nightingale_raw.csv")
LP_TABULAR_MISSING_RAW_CSV = os.path.join(RESULTS_DIR, "lp_tabular_cont_missing_raw.csv")
TABULAR_RAW_CSV     = os.path.join(RESULTS_DIR, "tabular_ridge_raw.csv")
TABULAR_NEW_RAW_CSV = os.path.join(RESULTS_DIR, "tabular_ridge_new_raw.csv")
DIS_RAW_CSV         = os.path.join(RESULTS_DIR, "lp_disease_raw.csv")
TABULAR_DIS_RAW_CSV = os.path.join(RESULTS_DIR, "tabular_disease_raw.csv")
TABULAR_CSV         = os.path.join(RESULTS_DIR, "tabular_summary.csv")
DISEASE_CSV         = os.path.join(RESULTS_DIR, "lp_disease_summary.csv")
DISEASE_COV_ENS_CSV     = os.path.join(RESULTS_DIR, "lp_disease_cov_ens_summary.csv")
DISEASE_COV_ENS_RAW_CSV = os.path.join(RESULTS_DIR, "lp_disease_cov_ens_raw.csv")
TABULAR_DIS_CSV     = os.path.join(RESULTS_DIR, "tabular_disease_summary.csv")
DISEASE_TARGETS_CSV = os.environ.get(
    "LEDXA_DISEASE_TARGETS_CSV",
    os.path.join(_REPO_ROOT, "data", "hpp", "disease_targets.csv"),
)
_COX_CSV_ENS  = "ukbb/cox_ttest_results_min48_ensemble.csv"
_COX_CSV_ORIG = "ukbb/cox_ttest_results_min48.csv"
COX_CSV = _COX_CSV_ENS if os.path.exists(_COX_CSV_ENS) else _COX_CSV_ORIG

# Curated Panel B targets: AUC > 0.65, gain over Tab AND Cov, no gender-specific.
# Bone (osteopenia/osteoporosis) intentionally excluded from main panel — covered in text
# as the "matches the gold-standard tabular readout" parity case.
_DIS_KEEP = {
    "dis__osteoarthritis",          # 0.750 | dTab=0.13  dCov=0.04  (largest tabular gain)
    "dis__fatty_liver_disease",     # 0.776 | dTab=0.05  dCov=0.06
    "dis__prediabetes",             # 0.725 | dTab=0.05  dCov=0.06
    "dis__hypertension",            # 0.788 | dTab=0.04  dCov=0.02
    "dis__sleep_apnea",             # 0.784 | dTab=0.04  dCov=0.03
    "dis__hyperlipidemia",          # 0.724 | dTab=0.03  dCov=0.04
}

_SEX_SPECIFIC_DIS = {
    "dis__breast_cancer",
    "dis__endometriosis_and_adenomyosis",
    "dis__polycystic_ovary_disease",
    "dis__erectile_dysfunction",
    "dis__perimenopausal_disorders",
}

def _dis_n_positives():
    """Return {target: n_positives} from disease_targets.csv."""
    if not os.path.exists(DISEASE_TARGETS_CSV):
        return {}
    dt = pd.read_csv(DISEASE_TARGETS_CSV)
    dis_cols = [c for c in dt.columns if c.startswith('dis__')]
    return dt[dis_cols].sum().to_dict()

# Disease → category for grouped display in Panel B
_DIS_CATEGORY = {
    "dis__anemia":                        1,
    "dis__b12_deficiency":                1,
    "dis__hypertension":                  2,
    "dis__ischemic_heart_disease":        2,
    "dis__prediabetes":                   3,
    "dis__diabetes":                      3,
    "dis__hyperlipidemia":                3,
    "dis__hypothyroidism":                3,
    "dis__fatty_liver_disease":           4,
    "dis__gallstone_disease":             4,
    "dis__haemorrhoids":                  4,
    "dis__peptic_dis":                    4,
    "dis__peptic_ulcer_disease":          4,
    "dis__irritable_bowel_syndrome_ibs":  4,
    "dis__back_pain":                     5,
    "dis__fracture":                      5,
    "dis__osteoarthritis":                5,
    "dis__osteopenia":                    5,
    "dis__hearing_loss":                  5,
    "dis__asthma":                        6,
    "dis__sleep_apnea":                   6,
    "dis__allergy":                       6,
    "dis__chronic_sinusitis":             6,
    "dis__covid_19":                      6,
    "dis__depression":                    7,
    "dis__anxiety":                       7,
    "dis__attention_deficit_disorder_adhd": 7,
    "dis__migraine":                      7,
    "dis__headache":                      7,
    "dis__episodic_vertigo":              7,
    "dis__endometriosis_and_adenomyosis": 8,
    "dis__polycystic_ovary_disease":      8,
    "dis__breast_cancer":                 8,
    "dis__atopic_dermatitis":             9,
    "dis__basal_cell_carcinoma":          9,
    "dis__psoriasis":                     9,
    "dis__oral_aphthae":                  9,
    "dis__urinary_tract_infection":       10,
    "dis__urinary_tract_stones":          10,
    "dis__anal_fissure":                  11,
}
_DIS_CATEGORY_NAMES = {
    1: "Hematol.",  2: "Cardiovasc.",  3: "Metabolic",
    4: "GI",        5: "Musculosk.",   6: "Respiratory",
    7: "Neuropsych.", 8: "Hormonal",   9: "Dermatol.",
    10: "Urology",  11: "GI (other)",
}

_DISEASE_NAMES_JSON  = os.path.join(_METADATA_DIR, "disease_display_names.json")
_DISEASE_GROUPS_JSON = os.path.join(_METADATA_DIR, "disease_groups.json")

# ── STYLE ─────────────────────────────────────────────────────────────────────
MODEL_ORDER  = ["ensemble", "lejepa", "dino", "tabular", "covariates"]
MODEL_COLORS = {
    # Paper-wide model palette: DeepDXA is the visual anchor; comparators are muted.
    "lejepa":     "#083c7d",
    "ensemble":   "#083c7d",
    "dino":       "#7fb9dc",
    "tabular":    "#8ccbb3",
    "covariates": "#bdbdbd",
}
MODEL_LABELS = {
    "lejepa":     "DXA-FM (ours)",
    "dino":       "DINOv3 (Frozen)",
    "tabular":    "DXA Tabular",
    "covariates": "Covariates (age/sex/BMI)",
    "ensemble":   "DXA-FM + Covariates",
}

# Curated continuous targets: significant gain over Tab AND Cov.
# Three storylines: (1) age/aging, (2) hepatic soft-tissue, (3) cardiometabolic + sleep.
# GlycA pending tabular run — add once available.
PANEL_A_TARGETS = [
    "age",                          # r=0.89 | dTab=0.16  dCov=0.77  (headline)
    "liver_attenuation",            # r=0.42 | dTab=0.04  dCov=0.12
    "liver_elasticity",             # r=0.45 | dTab=0.04  dCov=0.08
    "bt__triglycerides",            # r=0.49 | dTab=0.01  dCov=0.17
    "ahi",                          # r=0.54 | dTab=0.02  dCov=0.04
    "hr_bpm",                       # r=0.43 | dTab=0.03  dCov=0.17
]
TARGET_LABELS = {
    "age":                            "Age",
    "ahi":                            "AHI",
    "rdi":                            "RDI",
    "saturation_mean":                "O₂ Saturation",
    "bt__creatinine":                 "Creatinine",
    "bt__hemoglobin":                 "Hemoglobin",
    "bt__lymphocytes_abs":            "Lymphocytes",
    "bt__rbc":                        "RBC",
    "bt__rdw":                        "RDW",
    "bt__mch":                        "MCH",
    "bt__ferritin":                   "Ferritin",
    "bt__alt_gpt":                    "ALT",
    "sitting_blood_pressure_systolic":"Systolic BP",
    "sitting_blood_pressure_diastolic":"Diastolic BP",
    "bt__triglycerides":              "Triglycerides",
    "bt__hdl_cholesterol":            "HDL Cholesterol",
    "hr_bpm":                         "Heart Rate",
    "sleep_efficiency":               "Sleep Efficiency",
    "GlycA":                          "GlycA (NMR)",
    "liver_attenuation":              "Liver Att.",
    "liver_elasticity":               "Liver Elasticity",
    "liver_viscosity":                "Liver Viscosity",
    "hand_grip_left":                 "Hand Grip (L)",
    "hand_grip_right":                "Hand Grip",
    "walking_speed_kmh":              "Walking Speed",
    "from_r_thigh_to_r_ankle_pwv":    "PWV",
    "r_abi":                          "ABI",
    "happiness_level":                "Happiness",
    "iglu_ea1c":                      "Est. HbA1c",
    "iglu_gmi":                       "Glucose Mgmt. Idx",
    "intima_media_th_mm_1_intima_media_thickness": "Intima-Media Thickness",
    "HDL_C":                          "HDL (NMR)",
    "LDL_C":                          "LDL (NMR)",
    "ApoB":                           "ApoB (NMR)",
    "bt__non_hdl_cholesterol":        "Non-HDL Chol.",
    "bt__total_cholesterol":          "Cholesterol",
    "bt__glucose":                    "Glucose",
    "bt__hba1c":                      "HbA1c",
}

# Cox ── model name mapping
_COX_MODEL_RENAME = {
    'Covariates':                    'Covariates (age/sex/BMI)',
    'DXA Tabular':                   'DXA Tabular',
    'DXA SSL (LeJEPA)':              'DXA-FM (ours)',
    'DXA SSL (DINO)':                'DINOv3 (pre-trained)',
    'DXA SSL (LeJEPA) + Covariates': 'DXA-FM + Covariates',
}
_COX_DESIRED = [
    "Covariates (age/sex/BMI)",
    "DXA Tabular",
    "DXA-FM (ours)",
    "DINOv3 (pre-trained)",
    "DXA-FM + Covariates",
]
_COX_DISPLAY_TO_KEY = {
    "DXA-FM (ours)":            "lejepa",
    "DINOv3 (pre-trained)":     "dino",
    "DXA Tabular":              "tabular",
    "Covariates (age/sex/BMI)": "covariates",
    "DXA-FM + Covariates":      "ensemble",
}
# Cox events dropped: loses to DINOv3/tabular, or not central enough for DXA story
# NOTE: use prettified labels (output of _prettify_cox), not raw event strings
_COX_EXCLUDE_LABELS = {
    # loses to DINOv3 or tabular
    'Asthma', 'Stroke', "Alzheimer's", 'Hypothyroidism',
    'Intracerebral Haemorrhage', 'Dementia',
    # redundant / weak / not central
    'Angina Pectoris', 'Ischaemic Heart Disease', 'Myocardial Infarction',
    'Liver Disease', 'Intervertebral Disk Disease',
    'Arthrosis', 'Spondylopathy',
    'Obesity', 'Hyperthyroidism',
    'Rheumatoid Arthritis',   # N=79 too small
    'Acute Renal Failure',    # less central than chronic
    'Cerebral Infarction',    # less central for DXA story
}

# Whitelist for main figure Panel C — only these 4 outcomes shown
_COX_PANEL_C_KEEP = {
    'Knee Arthrosis',      # FM beats both baselines; largest gain
    'Hip Arthrosis',       # FM ≈ covariates; large tabular gain
    'Osteoporosis',        # FM beats covariates; primary clinical DXA use case
    'Type 2 Diabetes',     # FM beats both baselines; high absolute C-index
    'Atrial Fibrillation', # best cardiac outcome; large tabular gain
    'Death',               # all-cause mortality; largest tabular gain (+0.106)
}

_COX_EVENT_SHORT_NAMES = _COX_EVENT_SHORT_NAMES_IMPORTED
_COX_EVENT_CATEGORIES  = _COX_EVENT_CATEGORIES_IMPORTED
_COX_CATEGORY_NAMES    = _COX_CATEGORY_NAMES_IMPORTED

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    10,
    "axes.labelsize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       400,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   0.8,
    "axes.linewidth":    0.6,
    "legend.frameon":    False,
})

# ── SIGNIFICANCE HELPERS ──────────────────────────────────────────────────────
def _compute_lp_pvalues(raw_csv, tab_raw_csv=None, cov_raw_csv=None, extra_raw_csvs=None):
    """Two-tailed paired t-tests (LeJEPA vs each baseline). Returns {target: {model: p}}.
    Tabular scores are read from any raw file (model=='tabular'); extra_raw_csvs lists
    additional raw files to merge in (lejepa/dino/covariates/tabular rows)."""
    if not (raw_csv and os.path.exists(raw_csv)):
        return {}
    frames = [pd.read_csv(raw_csv, encoding='latin-1')]
    for p in (extra_raw_csvs or []):
        if p and os.path.exists(p):
            frames.append(pd.read_csv(p, encoding='latin-1'))
    all_raw = pd.concat(frames, ignore_index=True)
    if 'seed' in all_raw.columns:
        all_raw = all_raw.drop_duplicates(subset=['target', 'model', 'seed'])

    # SSL + covariates rows
    raw = all_raw[(all_raw['mode'] == 'ridge') & (all_raw['view'] == 'fusion')]

    # Merge in covariates raw scores if provided separately
    if cov_raw_csv and os.path.exists(cov_raw_csv):
        cov_raw = pd.read_csv(cov_raw_csv, encoding='latin-1')
        cov_raw = cov_raw[(cov_raw['mode'] == 'ridge') & (cov_raw['view'] == 'fusion')]
        raw = pd.concat([raw, cov_raw], ignore_index=True)

    # Tabular: prefer rows already in raw_csv; fall back to separate file
    tab_in_main = all_raw[all_raw['model'] == 'tabular']
    if len(tab_in_main) > 0:
        tab_raw = tab_in_main
    elif tab_raw_csv and os.path.exists(tab_raw_csv):
        tr = pd.read_csv(tab_raw_csv, encoding='latin-1')
        tab_raw = tr[tr['view'].isin(['tabular', 'fusion'])]
    else:
        tab_raw = None

    def _paired(lej_rows, other_rows, min_pairs=2):
        """Paired t-test on shared seeds; returns p-value or None."""
        lej_s   = lej_rows.set_index('seed')['score']
        other_s = other_rows.set_index('seed')['score']
        shared  = sorted(lej_s.index.intersection(other_s.index))
        if len(shared) < min_pairs:
            return None
        _, p = ttest_rel(lej_s.reindex(shared).values, other_s.reindex(shared).values)
        return float(p)

    result = {}
    for target, grp in raw.groupby('target'):
        lej_rows = grp[grp['model'] == 'lejepa'].sort_values('seed')
        if len(lej_rows) < 2:
            continue
        result[target] = {}
        for model in ['dino', 'covariates']:
            p = _paired(lej_rows, grp[grp['model'] == model].sort_values('seed'))
            if p is not None:
                result[target][model] = p
        if tab_raw is not None:
            tg = tab_raw[tab_raw['target'] == target]
            tab_rows = tg[tg['model'] == 'tabular'].sort_values('seed')
            p = _paired(lej_rows, tab_rows)
            if p is not None:
                result[target]['tabular'] = p
    return result


def _fdr_pass(pval_dict, model='tabular', alpha=0.05):
    """Return set of targets passing BH FDR correction on LeJEPA-vs-{model} p-values."""
    if not pval_dict:
        return None
    pairs = [(t, pval_dict[t][model]) for t in pval_dict if model in pval_dict.get(t, {})]
    if not pairs:
        return None
    targets_list, pvals_list = zip(*pairs)
    m = len(pvals_list)
    order = np.argsort(pvals_list)
    k_star = -1
    for rank, idx in enumerate(order, start=1):
        if pvals_list[idx] <= rank / m * alpha:
            k_star = rank
    return {targets_list[order[r - 1]] for r in range(1, k_star + 1)} if k_star > 0 else set()


def _sig_sym(p):
    if p is None or p >= 0.05: return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    return '*'

def _draw_sig_markers(ax, y_pos, ordered_targets, pval_dict, offsets, models, xmax, xmin=0, bar_ends=None, width_ratio=1.0):
    """Colored asterisks at each comparator bar's y-position, just right of target bars."""
    if not pval_dict or 'lejepa' not in models:
        return
    comparators = [m for m in ['dino', 'tabular', 'covariates'] if m in models]
    r = xmax - xmin
    
    # --- UNIVERSAL PHYSICAL SCALING ---
    # These constants guarantee the exact same physical pixel spacing across all panels 
    # by normalizing against the panel's data range (r) and its grid width_ratio.
    gap    = 0.022 * (r / width_ratio)  # Initial space after longest bar
    tick   = 0.007 * (r / width_ratio)  # Horizontal bracket tick length
    step   = 0.015 * (r / width_ratio)  # Base space between brackets
    char_w = 0.011 * (r / width_ratio)  # Space allocated per asterisk character
    tgap   = 0.004 * (r / width_ratio)  # Gap between bracket line and text

    row_max = {}
    if bar_ends:
        for (t, _m), v in bar_ends.items():
            if t in set(ordered_targets):
                row_max[t] = max(row_max.get(t, xmin), v)

    for yp, target in zip(y_pos, ordered_targets):
        tpvals = pval_dict.get(target, {})
        y_lej = yp + offsets[models.index('lejepa')]
        
        current_x = row_max.get(target, xmax) + gap if bar_ends else xmax + gap

        for model in comparators:
            sym = _sig_sym(tpvals.get(model))
            if not sym:
                continue
            y_m = yp + offsets[models.index(model)]
            color = MODEL_COLORS[model]
            y_lo, y_hi = sorted([y_lej, y_m])

            ax.plot([current_x, current_x], [y_lo, y_hi], color=color, lw=1.2, clip_on=False, zorder=6)
            ax.plot([current_x - tick, current_x], [y_lej, y_lej], color=color, lw=1.2, clip_on=False, zorder=6)
            ax.plot([current_x - tick, current_x], [y_m, y_m], color=color, lw=1.2, clip_on=False, zorder=6)
            ax.text(current_x + tgap, (y_lo + y_hi) / 2, sym, ha='left', va='center',
                    fontsize=10, color=color, fontweight='bold', clip_on=False, zorder=6)
            
            # Advance exactly enough for the next bracket
            current_x += step + (len(sym) * char_w)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _load(path):
    return pd.read_csv(path, encoding="latin-1") if path and os.path.exists(path) else None

def _load_disease_df():
    """Merge base disease CSV (lejepa+dino) with cov+ens CSV when available."""
    base  = _load(DISEASE_CSV)
    extra = _load(DISEASE_COV_ENS_CSV)
    if base is None:
        return extra
    if extra is None:
        return base
    merged = pd.concat([base, extra], ignore_index=True)
    merged = merged.drop_duplicates(subset=["target", "model", "mode", "view"])
    return merged

def _fusion_pivot(df, mode):
    is_tab = df["model"].eq("tabular")
    mask   = (((df["mode"] == mode) & ~is_tab) | is_tab) & df["view"].isin(["fusion", "tabular"])
    return df[mask].copy().pivot_table(index="target", columns="model", values=["mean", "se"])

def _merge_tabular(piv, tabular_df):
    if tabular_df is None:
        return piv
    tab_piv = _fusion_pivot(tabular_df, "ridge")
    for cg in ["mean", "se"]:
        if cg in tab_piv.columns.get_level_values(0):
            piv[(cg, "tabular")] = tab_piv[(cg, "tabular")]
    return piv.sort_index(axis=1)

def _prettify_cox(event_str):
    m = re.match(r'Date\s+\w+\s+first reported\s*\(([^)]+)\)\s*-\s*visit\s*\d+',
                 str(event_str), re.IGNORECASE)
    if m:
        raw = m.group(1).strip().title()
        return _COX_EVENT_SHORT_NAMES.get(raw, raw)
    s = str(event_str).lower()
    if 'death' in s:       return 'Death'
    if 'dementia' in s:    return 'Dementia'
    if 'alzheimer' in s:   return "Alzheimer's"
    if 'parkinson' in s:   return "Parkinson's"
    return None

def _dis_label(col, display_names):
    raw  = display_names.get(col, col.replace("dis__", "").replace("_", " "))
    name = raw.title()
    for pat, rep in [
        (r"Endometriosis And Adenomyosis",  "Endometriosis"),
        (r"Polycystic Ovary Disease",       "PCOS"),
        (r"Perimenopausal Disorders",       "Perimenopause"),
        (r"Erectile Dysfunction",           "Erectile Dysfunct."),
        (r"Urinary Tract Infection",        "UTI"),
        (r"Urinary Tract Stones",           "Urinary Stones"),
        (r"Attention Deficit Disorder.*",   "ADHD"),
        (r"Irritable Bowel Syndrome.*",     "IBS"),
        (r"Intervertebral Disc Disease",    "Disc Disease"),
        (r"Primary Hypercholesterolaemia",  "Hypercholest."),
        (r"Squamous Call Carcinoma",        "SCC"),
        (r"Fatty Liver Disease",            "Fatty Liver"),
        (r"Gallstone Disease",              "Gallstones"),
        (r"Hypercoagulability",             "Hypercoagul."),
        (r"Ischemic Heart Disease",         "Ischemic HD"),
        (r"Basal Cell Carcinoma",           "BCC"),
        (r"Atopic Dermatitis",              "Atopic Derm."),
        (r"Diaphragmatic Hernia",           "Diaphr. Hernia"),
        (r"Peptic Ulcer Disease",           "Peptic Ulcer"),
        (r"Chronic Sinusitis",              "Chr. Sinusitis"),
        (r"B12 Deficiency",                 "B12 Deficiency"),
        (r"Covid 19",                       "COVID-19"),
    ]:
        name = re.sub(pat, rep, name, flags=re.IGNORECASE)
    return name

def _draw_grouped_bars(ax, ordered_targets, y_pos, piv, models, xmin, xmax, x_label,
                       y_labels, cat_centers, dividers, cat_names_map, panel_letter,
                       baseline_x=None, pval_dict=None):
    """Draw horizontal grouped bars with category separators and right-side labels."""
    y = np.array(y_pos)
    n_m = len(models)
    bar_h = min(0.75 / n_m, 0.14)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_h, (n_m - 1) / 2 * bar_h, n_m)

    # Alternating row background shading (every other row, very subtle)
    for idx, yp in enumerate(y_pos):
        if idx % 2 == 0:
            ax.axhspan(yp - 0.5, yp + 0.5, color='#F5F5F5', zorder=0, linewidth=0)

    if baseline_x is not None:
        ax.axvline(baseline_x, color='#AAAAAA', linestyle=':', linewidth=1.5, zorder=1)

    for d in dividers:
        ax.axhline(d, color='#CCCCCC', linestyle='-', linewidth=2.0, zorder=2)

    all_vals = []
    bar_ends = {}
    for i, model in enumerate(models):
        means = np.array([piv["mean"][model].get(t, np.nan)
                          if model in piv["mean"].columns else np.nan
                          for t in ordered_targets])
        ses   = np.array([piv["se"][model].get(t, 0.0)
                          if model in piv["se"].columns else 0.0
                          for t in ordered_targets])
        ax.barh(y + offsets[i], means, bar_h, color=MODEL_COLORS[model],
                alpha=0.85, zorder=3)
        ax.errorbar(means, y + offsets[i], xerr=ses,
                    fmt="none", color="#444444", capsize=2, linewidth=0.8, zorder=4)
        all_vals.append(means)
        for target, mean, se in zip(ordered_targets, means, ses):
            if np.isfinite(mean): bar_ends[(target, model)] = float(mean + (se if np.isfinite(se) else 0.0))

    y_max = max(y_pos) + 0.8 if y_pos else 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, y_max + 0.3)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(x_label)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.9)
    ax.grid(axis='y', visible=False)


    ax.text(-0.14, 1.03, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)

    _draw_sig_markers(ax, y_pos, ordered_targets, pval_dict, offsets, models, xmax, xmin=xmin, bar_ends=bar_ends, width_ratio=1.0)


# ── PANEL A: LP continuous targets ────────────────────────────────────────────
_PANEL_B_EXCLUDE = {'rdi', 'health_satisfaction'}

def draw_panel_a(ax, lp_df, tabular_df, pval_dict=None, fdr_passed=None, top_n=None, panel_letter="a"):
    piv = _merge_tabular(_fusion_pivot(lp_df, "ridge"), tabular_df)
    if fdr_passed is not None:
        candidates = [t for t in piv.index if t in fdr_passed and t not in _PANEL_B_EXCLUDE]
        if (top_n is not None
                and "lejepa" in piv["mean"].columns
                and "tabular" in piv["mean"].columns):
            def _gain(t):
                lej = piv["mean"]["lejepa"].get(t, np.nan)
                tab = piv["mean"]["tabular"].get(t, np.nan)
                return lej - tab if not (np.isnan(lej) or np.isnan(tab)) else -np.inf
            candidates = sorted(candidates, key=_gain, reverse=True)[:top_n]
    else:
        candidates = [t for t in PANEL_A_TARGETS if t in piv.index]
    targets = candidates
    models  = [m for m in MODEL_ORDER if m in piv["mean"].columns]

    n   = len(targets)
    n_m = len(models)
    bar_h   = min(0.85 / n_m, 0.20)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_h, (n_m - 1) / 2 * bar_h, n_m)
    y = np.arange(n)

    # Alternating row shading
    for idx in range(n):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color='#F5F5F5', zorder=0, linewidth=0)

    bar_ends = {}  # Track bar lengths for the significance brackets
    for i, model in enumerate(models):
        means = np.array([piv["mean"][model].get(t, np.nan)
                          if model in piv["mean"].columns else np.nan
                          for t in targets])
        ses   = np.array([piv["se"][model].get(t, 0.0)
                          if model in piv["se"].columns else 0.0
                          for t in targets])
        ax.barh(y + offsets[i], means, bar_h, color=MODEL_COLORS[model], alpha=0.85, zorder=3)
        ax.errorbar(means, y + offsets[i], xerr=ses,
                    fmt="none", color="#444444", capsize=2, linewidth=0.8, zorder=4)
        
        # Populate bar_ends with the max extension of the bar + error
        for target, mean, se in zip(targets, means, ses):
            if np.isfinite(mean):
                bar_ends[(target, model)] = float(mean + (se if np.isfinite(se) else 0.0))

    ax.set_yticks(y)
    ax.set_yticklabels([TARGET_LABELS.get(t, t) for t in targets])
    ax.invert_yaxis()
    all_means = [piv["mean"][m].get(t, np.nan)
                 for m in models if m in piv["mean"].columns
                 for t in targets]
    raw_max = np.nanmax(all_means) if all_means else 1.0
    xmax_a  = round(np.ceil(raw_max / 0.05) * 0.05 + 0.02, 2)

    ax.set_xlabel("Pearson r")
    ax.set_xlim(0, xmax_a)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.9)
    ax.grid(axis='y', visible=False)
    ax.text(-0.14, 1.03, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)
    
    _draw_sig_markers(ax, list(y), targets, pval_dict, offsets, models, xmax_a, xmin=0, bar_ends=bar_ends, width_ratio=1.5)
    


# ── PANEL B: Disease classification AUC ───────────────────────────────────────
def draw_panel_b(ax, disease_df, tabular_df, display_names, groups, top_n=16, pval_dict=None,
                 fdr_passed=None, panel_letter="b"):
    piv = _merge_tabular(_fusion_pivot(disease_df, "ridge"), tabular_df)
    if "lejepa" not in piv["mean"].columns:
        ax.set_visible(False)
        return

    if fdr_passed is not None:
        top_targets = {t for t in piv.index if t in fdr_passed}
        # Keep only targets with N positives > 150
        n_pos = _dis_n_positives()
        if n_pos:
            top_targets = {t for t in top_targets if n_pos.get(t, 0) > 150}
        # Require LeJEPA beats BOTH tabular AND covariates baselines
        for baseline in ('tabular', 'covariates'):
            if baseline in piv["mean"].columns and "lejepa" in piv["mean"].columns:
                top_targets = {t for t in top_targets
                               if piv["mean"]["lejepa"].get(t, np.nan) >
                                  piv["mean"][baseline].get(t, np.nan)}
        # Require LeJEPA AUC >= 0.6
        if "lejepa" in piv["mean"].columns:
            top_targets = {t for t in top_targets
                           if piv["mean"]["lejepa"].get(t, np.nan) >= 0.6}
        # Top-N by AUC gain over tabular baseline
        if (top_n is not None
                and "tabular" in piv["mean"].columns
                and "lejepa" in piv["mean"].columns):
            def _auc_gain(t):
                lej = piv["mean"]["lejepa"].get(t, np.nan)
                tab = piv["mean"]["tabular"].get(t, np.nan)
                return lej - tab if not (np.isnan(lej) or np.isnan(tab)) else -np.inf
            top_targets = set(sorted(top_targets, key=_auc_gain, reverse=True)[:top_n])
    else:
        top_targets = {t for t in _DIS_KEEP if t in piv.index}

    # Assign each selected target to a category (99 = uncategorized)
    def _cat(t): return _DIS_CATEGORY.get(t, 99)

    # Gather categories present in top_targets, sorted by category id
    cats_present = sorted(set(_cat(t) for t in top_targets))

    # Build ordered list of targets: within each category, sort by AUC descending
    ordered_targets, y_pos, y_labels = [], [], []
    cat_centers, dividers = [], []
    y_curr = 0
    for cat in cats_present:
        grp = sorted([t for t in top_targets if _cat(t) == cat],
                     key=lambda t: -piv["mean"]["lejepa"].get(t, 0.0))
        cat_start = y_curr
        for t in grp:
            ordered_targets.append(t)
            y_pos.append(y_curr)
            y_labels.append(_dis_label(t, display_names))
            y_curr += 1
        cat_centers.append(((cat_start + y_curr - 1) / 2.0, cat))
        dividers.append(y_curr - 0.5)
        y_curr += 0.25
    dividers = dividers[:-1]

    models = [m for m in MODEL_ORDER if m in piv["mean"].columns]

    all_b_vals = [piv["mean"][m].get(t, np.nan)
                  for m in models if m in piv["mean"].columns
                  for t in ordered_targets]
    b_max  = np.nanmax(all_b_vals) if all_b_vals else 1.0
    xmax_b = round(np.ceil(b_max / 0.05) * 0.05 + 0.05, 2)

    _draw_grouped_bars(
        ax, ordered_targets, y_pos, piv, models,
        xmin=0.5, xmax=xmax_b, x_label="AUC",
        y_labels=y_labels,
        cat_centers=cat_centers, dividers=dividers,
        cat_names_map=_DIS_CATEGORY_NAMES,
        panel_letter=panel_letter,
        baseline_x=0.5,
        pval_dict=pval_dict,
    )
    tick_step = 0.1 if (xmax_b - 0.5) > 0.25 else 0.05
    ax.set_xticks(np.arange(0.5, xmax_b + 0.01, tick_step))


# ── PANEL C: Cox C-index ──────────────────────────────────────────────────────
def draw_panel_c(ax, cox_df, min_events=150, show_ensemble=True):
    rename_map = {}
    for csv_name, disp_name in _COX_MODEL_RENAME.items():
        for suf in [' C-Index', ' C-Index SE']:
            old = f'{csv_name}{suf}'
            if old in cox_df.columns:
                rename_map[old] = f'{disp_name}{suf}'
    cox_df = cox_df.rename(columns=rename_map)

    if 'Total Events' in cox_df.columns:
        cox_df = cox_df[pd.to_numeric(cox_df['Total Events'], errors='coerce') >= min_events].copy()

    cox_df['EventLabel'] = cox_df['Event'].apply(_prettify_cox)
    cox_df = cox_df.dropna(subset=['EventLabel'])
    cox_df = cox_df[~cox_df['EventLabel'].isin(_COX_EXCLUDE_LABELS)]
    cox_df = cox_df[cox_df['EventLabel'].isin(_COX_PANEL_C_KEEP)]
    # deduplicate: if same label appears twice keep the one with more events
    cox_df = cox_df.sort_values('Total Events', ascending=False).drop_duplicates(subset=['EventLabel'])
    cox_df['Category']   = cox_df['EventLabel'].map(lambda x: _COX_EVENT_CATEGORIES.get(x, 99))
    models_present = [m for m in _COX_DESIRED if f'{m} C-Index' in cox_df.columns]
    # Panel C order: ensemble first (topmost), then lejepa, dino, tabular, covariates
    _cox_order = ["ensemble", "lejepa", "dino", "tabular", "covariates"]
    _cox_rank  = {k: i for i, k in enumerate(_cox_order)}
    if not show_ensemble:
        models_present = [m for m in models_present
                          if _COX_DISPLAY_TO_KEY.get(m, m) != "ensemble"]
    models_present.sort(key=lambda m: _cox_rank.get(_COX_DISPLAY_TO_KEY.get(m, m), 99))
    ci_cols = [f'{m} C-Index' for m in models_present]
    cox_df['BestScore']  = cox_df[ci_cols].apply(pd.to_numeric, errors='coerce').max(axis=1)
    cox_df = cox_df.sort_values(['Category', 'BestScore'], ascending=[True, False]).reset_index(drop=True)

    # Build y positions with category gaps
    y_pos, labels, cat_centers, dividers = [], [], [], []
    y_curr = 0
    ordered_events = []
    for cat, grp in cox_df.groupby('Category', sort=False):
        cat_start = y_curr
        for _, row in grp.iterrows():
            y_pos.append(y_curr)
            labels.append(row['EventLabel'])
            ordered_events.append(row['EventLabel'])
            y_curr += 1
        cat_centers.append(((cat_start + y_curr - 1) / 2.0, cat))
        dividers.append(y_curr - 0.5)
        y_curr += 0.25
    dividers = dividers[:-1]
    y = np.array(y_pos)

    n_m     = len(models_present)
    bar_h   = min(0.85 / n_m, 0.20)
    offsets = np.linspace(-(n_m - 1) / 2 * bar_h, (n_m - 1) / 2 * bar_h, n_m)

    # Alternating row shading
    for idx, yp in enumerate(y_pos):
        if idx % 2 == 0:
            ax.axhspan(yp - 0.5, yp + 0.5, color='#F5F5F5', zorder=0, linewidth=0)

    ax.axvline(0.5, color='#AAAAAA', linestyle=':', linewidth=1.5, zorder=1)
    for d in dividers:
        ax.axhline(d, color='#CCCCCC', linestyle='-', linewidth=2.0, zorder=2)

    all_v, all_s = [], []
    cox_bar_ends = {}
    for i, model in enumerate(models_present):
        ci_col = f'{model} C-Index'
        key    = _COX_DISPLAY_TO_KEY.get(model, model)
        color  = MODEL_COLORS.get(key, "#999999")
        vals   = pd.to_numeric(cox_df[ci_col], errors='coerce').to_numpy()
        se_col = f'{model} C-Index SE'
        if se_col in cox_df.columns:
            ses = pd.to_numeric(cox_df[se_col], errors='coerce').fillna(0.0).to_numpy()
        elif 'N' in cox_df.columns:
            nv  = pd.to_numeric(cox_df['N'], errors='coerce').clip(lower=1).to_numpy()
            ses = np.where(np.isnan(vals), 0.0, np.sqrt(vals * (1 - vals) / nv))
        else:
            ses = np.zeros(len(vals))
        ax.barh(y + offsets[i], vals, height=bar_h, color=color, alpha=0.85,
                xerr=ses, ecolor='#444444', capsize=2, zorder=3)
        all_v.append(vals); all_s.append(ses)
        for event_label, val, se in zip(ordered_events, vals, ses):
            if np.isfinite(val): cox_bar_ends[(event_label, key)] = float(val + (se if np.isfinite(se) else 0.0))

    all_v_cat = np.concatenate(all_v)
    all_s_cat = np.concatenate(all_s)
    xmin = 0.5
    xmax = 0.8
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, y_curr - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('C-index')
    ax.grid(axis='x', linestyle='--', color='#CCCCCC', alpha=0.9)
    ax.grid(axis='y', visible=False)


    ax.text(-0.14, 1.03, "c", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)

    # Significance brackets for Cox: one-tailed (two-tailed p / 2 when lejepa > other)
    # P-value cols keep original names; C-Index cols were renamed via rename_map
    _lej_ci  = 'DXA-FM (ours) C-Index'
    _pcol = {
        'dino':       ('P-Value (DXA SSL (LeJEPA) vs DXA SSL (DINO))',          'DINOv3 (pre-trained) C-Index'),
        'tabular':    ('P-Value (DXA SSL (LeJEPA) vs DXA Tabular)',              'DXA Tabular C-Index'),
        'covariates': ('P-Value (DXA SSL (LeJEPA) vs Covariates)',               'Covariates (age/sex/BMI) C-Index'),
        'ensemble':   ('P-Value (DXA SSL (LeJEPA) + Covariates vs Covariates)',  'DXA-FM + Covariates C-Index'),
    }
    cox_pvals = {}
    for _, row in cox_df.iterrows():
        label = row['EventLabel']
        cox_pvals[label] = {}
        lej_val = pd.to_numeric(row.get(_lej_ci), errors='coerce')
        for mkey, (pcol, ci_col) in _pcol.items():
            if pcol not in cox_df.columns:
                continue
            p2 = pd.to_numeric(row.get(pcol), errors='coerce')
            oth_val = pd.to_numeric(row.get(ci_col), errors='coerce')
            if not (np.isnan(p2) or np.isnan(lej_val) or np.isnan(oth_val)):
                cox_pvals[label][mkey] = p2 / 2 if lej_val > oth_val else 1.0

    model_keys = [_COX_DISPLAY_TO_KEY.get(m, m) for m in models_present]
    _draw_sig_markers(ax, y_pos, ordered_events, cox_pvals, offsets, model_keys, xmax, xmin=xmin, bar_ends=cox_bar_ends, width_ratio=1.5)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    lp_df      = _load(LP_CSV)
    lp_ext_df  = _load(LP_EXTENDED_CSV)
    if lp_df is not None and lp_ext_df is not None:
        lp_df = pd.concat([lp_df, lp_ext_df], ignore_index=True).drop_duplicates(
            subset=["target", "model", "mode", "view"])
    elif lp_ext_df is not None:
        lp_df = lp_ext_df
    # Tabular scores are now included in lp_df (compare_lp.py saves all models together).
    # Fall back to the legacy separate CSV only if lp_df has no tabular rows.
    tabular_df = None
    if lp_df is not None and "tabular" not in lp_df["model"].values:
        tabular_df = _load(TABULAR_CSV)
    disease_df = _load_disease_df()
    # Use tabular already in disease_df (new unified run); fall back to legacy file only if absent
    tab_dis_df = None
    if disease_df is not None and "tabular" not in disease_df["model"].values:
        tab_dis_df = _load(TABULAR_DIS_CSV)
    cox_df     = _load(COX_CSV)

    display_names, groups = {}, {}
    if os.path.exists(_DISEASE_NAMES_JSON):
        with open(_DISEASE_NAMES_JSON) as f:
            display_names = json.load(f)
    if os.path.exists(_DISEASE_GROUPS_JSON):
        with open(_DISEASE_GROUPS_JSON) as f:
            groups = json.load(f)

    fig, axes = plt.subplots(
        1, 3, figsize=(7.09, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.5, 1.5], "wspace": 0.35},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.05, wspace=0.35)

    pval_a   = _compute_lp_pvalues(LP_RAW_CSV,  TABULAR_RAW_CSV)
    pval_b   = _compute_lp_pvalues(DIS_RAW_CSV, TABULAR_DIS_RAW_CSV, DISEASE_COV_ENS_RAW_CSV)

    if lp_df is not None:
        draw_panel_a(axes[0], lp_df, tabular_df, pval_dict=pval_a)
    if disease_df is not None:
        draw_panel_b(axes[1], disease_df, tab_dis_df, display_names, groups, pval_dict=pval_b)
    if cox_df is not None:
        draw_panel_c(axes[2], cox_df)

    # Shift Panel C right to clear Panel B's significance brackets (which extend past xmax)
    pos = axes[2].get_position()
    axes[2].set_position([pos.x0 + 0.04, pos.y0, pos.width, pos.height])

    # Shared legend at top
    handles = [mpatches.Patch(facecolor=MODEL_COLORS[m], label=MODEL_LABELS[m], alpha=0.85)
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.97), framealpha=0.95, fontsize=9,
               columnspacing=1.5, handlelength=1.8, edgecolor='#DDDDDD')

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(FIGURES_DIR, f"fig_combined_3panel.{ext}")
        if ext == "png":
            fig.savefig(out, dpi=800)
        else:
            fig.savefig(out)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
