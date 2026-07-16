"""Supplementary Fig. 2 | HPP replication of the biological-age-gap Q4-vs-Q1
DXA body-composition associations (sex-stratified), mirroring Fig. 5c,d (UKBB).
Two panels (female, male): Cohen's d (Q4 - Q1) for the top FDR-significant DXA
features, after greedily de-duplicating collinear phenotypes (|r| >= CORR_THRESH
against every feature already kept) so the panel reflects independent signals
rather than repeated left/right/mean or area/volume variants of the same
underlying measurement."""
import sys
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, re
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downstream.bioage.style import MODEL_COLORS, apply_paper_rcparams

plt.style.use(str(Path.home() / ".claude/skills/nature-plot-style/style_files/nature_double.mplstyle"))
apply_paper_rcparams()
plt.rcParams["figure.dpi"] = 150

# Colours/style matched to Fig. 5c,d (UKBB Q4-vs-Q1 phenotype forests): navy
# for positive Cohen's d, red for negative, alternating row shading, dashed grid.
POS_COLOR = MODEL_COLORS["lejepa"]
NEG_COLOR = "#c0392b"
ROW_SHADE = "#F5F5F5"

T = "tables"
OUT = "/path/to/project/supplement/supplementary_figures/Supp_Fig2_hpp_replication"
TOPN = 14
CORR_THRESH = 0.8
DXA_TABULAR_CSV = "/path/to/dxa_filtered.csv"
EXTRA_ANTHRO_CSV = "/path/to/project/targets_for_downstream_full.csv"
# Known-problematic columns to exclude from candidate phenotypes regardless of
# significance/effect size (e.g. unreliable derivation, QC issues).
EXCLUDE_FEATURES = {"total_scan_vat_mass"}


def _load_feature_matrix():
    """Raw per-participant HPP feature values, used only to measure collinearity
    between candidate phenotypes (not for the Cohen's d values themselves,
    which come from the precomputed replication tables)."""
    dxa = pd.read_csv(DXA_TABULAR_CSV)
    extra = pd.read_csv(EXTRA_ANTHRO_CSV, usecols=["RegistrationCode", "research_stage",
                                                     "height", "hips", "waist"])
    merged = dxa.merge(extra, on=["RegistrationCode", "research_stage"], how="left")
    return merged[~merged.index.duplicated()]


def select_decorrelated_top(sig_df, feature_matrix, topn=TOPN, corr_thresh=CORR_THRESH):
    """Greedily walk phenotypes in descending |Cohen's d| order, keeping a
    feature only if it correlates below corr_thresh with every feature already
    kept. Stops once topn independent features are collected."""
    sig_df = sig_df.copy()
    sig_df["absd"] = sig_df["Cohens_d"].abs()
    sig_df = sig_df.sort_values("absd", ascending=False)

    kept_rows = []
    for _, row in sig_df.iterrows():
        p = row["Phenotype"]
        if p not in feature_matrix.columns:
            continue
        is_redundant = any(
            feature_matrix[p].corr(feature_matrix[k]) >= corr_thresh
            for k in (r["Phenotype"] for r in kept_rows)
        )
        if not is_redundant:
            kept_rows.append(row)
        if len(kept_rows) >= topn:
            break
    return pd.DataFrame(kept_rows)

def prettify(s):
    s = str(s)
    for pre in ("body_comp_", "total_scan_", "body_"):
        s = s.replace(pre, "")
    s = s.replace("_", " ")
    for a, b in [("bmd", "BMD"), ("bmc", "BMC"), ("vat", "VAT"), ("sat", "SAT"),
                 ("l1 l4", "L1–L4"), ("mean", ""), ("upper neck", "upper-neck")]:
        s = re.sub(rf"\b{a}\b", b, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:]

def _shade_alternate_rows(ax, n_rows):
    for idx in range(n_rows):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color=ROW_SHADE, zorder=0, linewidth=0)


feature_matrix = _load_feature_matrix()

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
for ax, sex in zip(axes, ["female", "male"]):
    d = pd.read_csv(f"{T}/hpp_gap_dxa_replication_{sex}.csv")
    d = d[d["Adjusted_P_Value"] < 0.05].copy()
    d = d[~d["Phenotype"].isin(EXCLUDE_FEATURES)]
    d = select_decorrelated_top(d, feature_matrix, topn=TOPN, corr_thresh=CORR_THRESH)
    d = d.sort_values("Cohens_d").reset_index(drop=True)

    n = len(d)
    _shade_alternate_rows(ax, n)
    y = np.arange(n)
    for i, row in d.iterrows():
        val = row["Cohens_d"]
        c = POS_COLOR if val > 0 else NEG_COLOR
        ax.plot([0, val], [i, i], color=c, lw=2.4, solid_capstyle="round", zorder=3, alpha=0.85)
        ax.scatter(val, i, color=c, s=60, zorder=4, edgecolors="none")
    ax.axvline(0, color="#888", lw=0.9, ls="--", alpha=0.6)
    ax.set_yticks(y); ax.set_yticklabels([prettify(p) for p in d["Phenotype"]], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylim(-0.55, n + 0.3)
    dmin, dmax = d["Cohens_d"].min(), d["Cohens_d"].max()
    pad = max(abs(dmin), abs(dmax)) * 0.20
    ax.set_xlim(dmin - pad, dmax + pad)
    ax.set_xlabel("Cohen's $d$  (Q4 − Q1)")
    ax.set_title(f"{sex.capitalize()} (HPP)", fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle="--", color="#CCCCCC", alpha=0.7)
    ax.grid(axis="y", visible=False)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=400, facecolor="white", transparent=False, bbox_inches="tight")
print("saved", OUT)
