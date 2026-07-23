"""
plot_supp_disease_groups.py

Supplementary figure: 4-arm AUROC comparison across 20 organ-system disease groups
(HPP cohort). One horizontal bar per arm, sorted by DeepDXA vs. DXA Tabular gap.
Significance brackets annotate DeepDXA comparisons.

Usage:
    python plot_supp_disease_groups.py
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from common.plot_style import MODEL_COLORS

# ── Paths ─────────────────────────────────────────────────────────────────────
_RESULTS_DIR = "/data/hpp_labdata/Analyses/gilsa/results/comparison"
_ROOT = Path(__file__).resolve().parents[1]
_METADATA_DIR = _ROOT / "metadata"
SUMMARY_CSV  = os.environ.get("GRP_SUMMARY", os.path.join(_RESULTS_DIR, "lp_disease_4arm_group_summary.csv"))
WILCOX_CSV   = os.environ.get("GRP_WILCOX",  os.path.join(_RESULTS_DIR, "lp_disease_4arm_group_wilcoxon.csv"))
TARGETS_CSV = os.environ.get(
    "LEDXA_GROUP_DISEASE_TARGETS_CSV",
    str(_ROOT / "data" / "hpp" / "disease_targets_group_with_covs.csv"),
)
INDIVIDUAL_TARGETS_CSV = os.environ.get(
    "LEDXA_DISEASE_TARGETS_WITH_COVS_CSV",
    str(_ROOT / "data" / "hpp" / "disease_targets_with_covs.csv"),
)
INDIVIDUAL_NAMES_JSON  = str(_METADATA_DIR / "disease_display_names.json")
INDIVIDUAL_GROUPS_JSON = str(_METADATA_DIR / "disease_groups.json")
CONDITIONS_CSV = "/data/hpp_labdata/Data/10K/for_review/baseline_conditions_all.csv"
OUT_DIR      = "figures"
OUT_NAME     = os.environ.get("GRP_OUT_NAME", "supp_disease_groups")
MIN_CHRONIC_CASES = 100

# ── Style ─────────────────────────────────────────────────────────────────────
_style_dir  = Path.home() / ".claude/skills/nature-plot-style/style_files"
double_style = str(_style_dir / "nature_double.mplstyle")

# Arm colours — aligned with MODEL_COLORS (plot_combined_figure.py / Fig 2)
PALETTE = {
    "covariates": MODEL_COLORS["covariates"],
    "tabular":    MODEL_COLORS["tabular"],
    "dino":       MODEL_COLORS["dino"],
    "lejepa":     MODEL_COLORS["lejepa"],
}
ARM_ORDER  = ["covariates", "tabular", "dino", "lejepa"]
ARM_LABELS = {
    "covariates": "Covariates (age/sex/BMI)",
    "tabular":    "DXA Tabular + cov",
    "dino":       "DINOv3 + cov",
    "lejepa":     "LeDXA + cov",
}

# ── Disease group display names ───────────────────────────────────────────────
GROUP_DISPLAY = {
    "dis__cardiovascular":    "Cardiovascular",
    "dis__dermatology":       "Dermatology",
    "dis__endocrinology":     "Endocrinology",
    "dis__ent":               "ENT",
    "dis__eye_disorder":      "Eye disorder",
    "dis__gastro":            "Gastro",
    "dis__hematological":     "Hematological",
    "dis__immunology":        "Immunology",
    "dis__infectious_disease":"Infectious Disease",
    "dis__metabolic":         "Metabolic",
    "dis__neurologic":        "Neurologic",
    "dis__obgyn":             "OBGyn",
    "dis__oncology":          "Oncology",
    "dis__orthopedic":        "Orthopedic",
    "dis__other":             "Other",
    "dis__pulmonology":       "Pulmonology",
    "dis__rheumatology":      "Rheumatology",
    "dis__sleep":             "Sleep",
    "dis__surgery":           "Surgery",
    "dis__urology":           "Urology",
}

CHRONIC_EXCLUDE = {
    # Non-chronic or not used in the 37-condition HPP chronic-disease count.
    "dis__covid_19",
    "dis__fracture",
    "dis__urinary_tract_infection",
    "dis__oral_aphthae",
    "dis__basal_cell_carcinoma",
    "dis__surgery",
    # DXA-direct / separate bone-density paragraph.
    "dis__obesity",
    "dis__osteopenia",
    "dis__osteoporosis",
    # Sex-specific endpoints excluded from the pooled chronic-condition count.
    "dis__breast_cancer",
    "dis__endometriosis_and_adenomyosis",
    "dis__polycystic_ovary_disease",
}

GROUP_EXCLUDE = {
    # Acute/procedural category; not part of the chronic grouped-disease count.
    "dis__surgery",
}


def _sanitize_colname(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return f"dis__{name}"


def _included_chronic_condition_counts():
    """Return {system_target: number of included chronic HPP conditions}.

    This mirrors the 37-condition count used in the Results text: individual
    HPP diseases with >=100 positives, excluding DXA-direct, sex-specific, and
    non-chronic / less central endpoints.
    """
    if not all(os.path.exists(p) for p in (
        INDIVIDUAL_TARGETS_CSV, INDIVIDUAL_NAMES_JSON,
        INDIVIDUAL_GROUPS_JSON, CONDITIONS_CSV,
    )):
        return {}

    with open(INDIVIDUAL_GROUPS_JSON) as f:
        disease_groups = json.load(f)

    targets = pd.read_csv(INDIVIDUAL_TARGETS_CSV)
    disease_cols = [c for c in targets.columns if c.startswith("dis__")]
    positives = targets[disease_cols].sum(skipna=True)
    included = {
        c for c in disease_cols
        if positives.get(c, 0) >= MIN_CHRONIC_CASES
        and disease_groups.get(c) == "general"
        and c not in CHRONIC_EXCLUDE
    }

    conditions = pd.read_csv(CONDITIONS_CSV, usecols=["Consolidated name", "Group"])
    conditions = conditions.dropna(subset=["Consolidated name", "Group"]).copy()
    conditions["disease_col"] = conditions["Consolidated name"].map(_sanitize_colname)
    conditions["system_col"] = conditions["Group"].map(_sanitize_colname)
    disease_to_system = dict(
        conditions.drop_duplicates("disease_col")
        .set_index("disease_col")["system_col"]
    )

    counts = {}
    for disease_col in included:
        system_col = disease_to_system.get(disease_col)
        if system_col:
            counts[system_col] = counts.get(system_col, 0) + 1
    return counts

# ── Load data ─────────────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)
wilcox  = pd.read_csv(WILCOX_CSV)

# Restrict to disease groups only (exclude age/gender/bmi regression targets)
dis_targets = [t for t in summary["target"].unique() if t.startswith("dis__")]
chronic_condition_counts = _included_chronic_condition_counts()
if chronic_condition_counts:
    dis_targets = [t for t in dis_targets if chronic_condition_counts.get(t, 0) > 0]
dis_targets = [t for t in dis_targets if t not in GROUP_EXCLUDE]
summary = summary[summary["target"].isin(dis_targets)]
wilcox  = wilcox[wilcox["target"].isin(dis_targets)]

MODEL_ALIAS = {
    "lejepa_cov": "lejepa",
    "dino_cov": "dino",
    "tab_cov": "tabular",
}
summary["model"] = summary["model"].replace(MODEL_ALIAS)
if "model_a" in wilcox.columns:
    wilcox["model_a"] = wilcox["model_a"].replace(MODEL_ALIAS)
if "model_b" in wilcox.columns:
    wilcox["model_b"] = wilcox["model_b"].replace(MODEL_ALIAS)

# Pivot to (target × model) → mean, se
pivot_mean = summary.pivot(index="target", columns="model", values="mean")
pivot_se   = summary.pivot(index="target", columns="model", values="se")

# ── Compute Δ(LeJEPA − Tabular) for sorting ───────────────────────────────────
delta_tab = pivot_mean["lejepa"] - pivot_mean["tabular"]

# ── Significance (BH-FDR-adjusted p < 0.05, two-sided Wilcoxon) ──────────────
# Use the FDR-adjusted p-value (p_adj) to match the figure caption / paper text;
# falls back to raw p only if an adjusted column is unavailable.
_PCOL = "p_adj" if "p_adj" in wilcox.columns else "p_raw"

# ── N_positives from grouped targets file; this is participant count, not
# constituent-diagnosis count.
tgt_df = pd.read_csv(TARGETS_CSV, index_col=[0, 1])
n_pos = {
    col: int(tgt_df[col].sum())
    for col in dis_targets
    if col in tgt_df.columns
}


def _pval(target, comparison):
    row = wilcox[(wilcox["target"] == target) & (wilcox["comparison"] == comparison)]
    return float(row[_PCOL].values[0]) if len(row) else 1.0

def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

BRACKET_CAP = 0.010

def _draw_v_bracket(ax, y_lej, y_comp, x_bracket, cap=BRACKET_CAP):
    """
    Vertical bracket for a horizontal bar chart.
    Draws a capped line from y_comp to y_lej at x_bracket.
    """
    y_lo, y_hi = min(y_lej, y_comp), max(y_lej, y_comp)
    ax.plot([x_bracket, x_bracket], [y_lo, y_hi],
            lw=0.9, color="#444444", clip_on=False, zorder=5)
    ax.plot([x_bracket - cap, x_bracket], [y_lo, y_lo],
            lw=0.9, color="#444444", clip_on=False, zorder=5)
    ax.plot([x_bracket - cap, x_bracket], [y_hi, y_hi],
            lw=0.9, color="#444444", clip_on=False, zorder=5)

# ── Sort groups by Δtab ascending (best DeepDXA advantage at top) ─────────────
order = delta_tab.sort_values(ascending=True).index.tolist()

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.style.use(double_style)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400,
                     "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

n_groups  = len(order)
bar_height = 0.08
group_gap  = 0.07   # gap between group clusters
y_step     = len(ARM_ORDER) * bar_height + group_gap

fig_h = max(4.8, n_groups * y_step * 1.08)
fig, ax = plt.subplots(figsize=(5.5, fig_h))

X_FLOOR = 0.48
ax.set_xlim(X_FLOOR - 0.005, 0.88)

# Measure the actual rendered width (in data coords) of a rotated star label,
# so bracket columns can be packed tightly with a guaranteed no-overlap gap
# instead of a hand-tuned constant.
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_probe = ax.text(0, 0, "***", fontsize=7, rotation=90)
_bbox_disp = _probe.get_window_extent(renderer=_renderer)
_probe.remove()
_inv = ax.transData.inverted()
_x0_data, _ = _inv.transform((_bbox_disp.x0, _bbox_disp.y0))
_x1_data, _ = _inv.transform((_bbox_disp.x1, _bbox_disp.y1))
STAR_X_EXTENT = abs(_x1_data - _x0_data)

for gi, target in enumerate(order):
    y_center = gi * y_step
    display  = GROUP_DISPLAY.get(target, target.replace("dis__", "").title())
    n = n_pos.get(target, 0)

    for ai, arm in enumerate(ARM_ORDER):
        y_pos  = y_center + ai * bar_height - (len(ARM_ORDER) - 1) * bar_height / 2
        mean_v = pivot_mean.loc[target, arm] if arm in pivot_mean.columns else np.nan
        se_v   = pivot_se.loc[target, arm]   if arm in pivot_se.columns   else np.nan
        if np.isnan(mean_v):
            continue
        bar_len = max(mean_v - X_FLOOR, 0)
        ax.barh(y_pos, bar_len, left=X_FLOOR, height=bar_height * 0.82,
                color=PALETTE[arm], edgecolor="none", zorder=2)
        ax.errorbar(mean_v, y_pos, xerr=se_v,
                    fmt="none", color="black", capsize=1.5,
                    linewidth=0.6, capthick=0.6, zorder=3)

    # Y-tick label: number of positive participants in this grouped endpoint.
    ax.text(-0.01, y_center, f"{display}  (n={n:,})",
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8)

    # Per-comparator significance brackets (nearest → farthest from DeepDXA)
    # ARM_ORDER = [covariates(ai=0), tabular(ai=1), dino(ai=2), lejepa(ai=3)]
    y_lej = y_center + 1.5 * bar_height   # ai=3 → +0.27
    _COMP_BRACKETS = [
        ("lejepa_vs_dino",       2),   # dino:       Δy = 0.18  (nearest)
        ("lejepa_vs_tabular",    1),   # tabular:    Δy = 0.36
        ("lejepa_vs_covariates", 0),   # covariates: Δy = 0.54  (farthest)
    ]
    # x_right = rightmost bar tip (mean + SE) across all arms in this group
    x_right = max(
        (pivot_mean.loc[target, arm] + (pivot_se.loc[target, arm]
         if arm in pivot_se.columns and not np.isnan(pivot_se.loc[target, arm]) else 0.0))
        for arm in ARM_ORDER if arm in pivot_mean.columns and not np.isnan(pivot_mean.loc[target, arm])
    )
    BASE_OFF = 0.018
    STAR_GAP = 0.004
    # Minimum center-to-center step between adjacent bracket columns that
    # guarantees the star label of one bracket never reaches the cap of the
    # next: gap-to-star + measured star width + the next bracket's cap width,
    # plus a small clearance buffer.
    STEP_OFF = STAR_GAP + STAR_X_EXTENT + BRACKET_CAP + 0.002
    level = 0
    for comparison, ai_comp in _COMP_BRACKETS:
        p = _pval(target, comparison)
        s = _stars(p)
        if not s:
            continue
        y_comp = y_center + ai_comp * bar_height - 1.5 * bar_height
        x_b = x_right + BASE_OFF + level * STEP_OFF
        _draw_v_bracket(ax, y_lej, y_comp, x_b)
        ax.text(x_b + STAR_GAP, (y_lej + y_comp) / 2, s,
                ha="left", va="center", fontsize=7, color="#444444",
                rotation=90, clip_on=False, zorder=5)
        level += 1

# ── Grid, floor line, axes ────────────────────────────────────────────────────
ax.axvline(X_FLOOR, color="black", linewidth=0.5, zorder=1)
ax.set_ylim(-y_step * 0.6, (n_groups - 1) * y_step + y_step * 0.6)
ax.set_yticks([])
ax.set_xlabel("AUROC", fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
ax.grid(axis="x", linewidth=0.4, alpha=0.5, zorder=0)

# ── (significance footnote removed; described in the figure caption instead) ────

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [mpatches.Patch(facecolor=PALETTE[arm], label=ARM_LABELS[arm])
           for arm in ARM_ORDER]
ax.legend(handles=handles, loc="lower right", fontsize=8,
          frameon=False, handlelength=1.2, handleheight=0.8,
          borderpad=0.3, labelspacing=0.3)

# ── Save ──────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    path = os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}")
    plt.savefig(path, dpi=400, facecolor="white", transparent=False,
                bbox_inches="tight", format=ext if ext == "pdf" else None)
    print(f"Saved: {path}")

plt.show()
