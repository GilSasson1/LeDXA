"""Shared plotting constants and helpers for the LeDXA manuscript figures.

Provides the paper-wide model palette/labels/order, the disease-label prettifier,
and the significance-marker helper used by the Figure 2 and Figure 6 panels.

(A legacy standalone "3-panel combined figure" generator formerly lived here; it
has been removed. The current per-figure scripts are ``plotting/fig2_heatmap.py``
and ``plotting/fig3_cox.py``.)
"""
import os
import re

from config import RESULTS_DIR as _CONFIG_RESULTS_DIR

# ── Paths ───────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(__file__)
_REPO_ROOT    = os.path.dirname(_HERE)
_METADATA_DIR = os.path.join(_REPO_ROOT, "metadata")
RESULTS_DIR   = str(_CONFIG_RESULTS_DIR)

DISEASE_TARGETS_CSV = os.environ.get(
    "LEDXA_DISEASE_TARGETS_CSV",
    os.path.join(_REPO_ROOT, "data", "hpp", "disease_targets.csv"),
)
_DISEASE_NAMES_JSON = os.path.join(_METADATA_DIR, "disease_display_names.json")

# ── Model palette / labels ────────────────────────────────────────────────────
MODEL_ORDER  = ["ensemble", "lejepa", "dino", "tabular", "covariates"]
MODEL_COLORS = {
    # Paper-wide model palette: LeDXA is the visual anchor; comparators are muted.
    "lejepa":     "#083c7d",
    "ensemble":   "#083c7d",
    "dino":       "#7fb9dc",
    "tabular":    "#8ccbb3",
    "covariates": "#bdbdbd",
}
MODEL_LABELS = {
    "lejepa":     "LeDXA",
    "dino":       "DINOv3 (Frozen)",
    "tabular":    "DXA Tabular",
    "covariates": "Covariates (age/sex/BMI)",
    "ensemble":   "LeDXA + Covariates",
}


# ── Significance helpers ──────────────────────────────────────────────────────
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


# ── Disease-label prettifier ──────────────────────────────────────────────────
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
