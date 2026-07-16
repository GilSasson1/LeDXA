"""
Extended Data Fig. 1 | Study flow and cohort selection.
Two-arm CONSORT-style flow diagram starting from the FULL cohorts:
  full cohort -> underwent whole-body DXA -> format/quality selection -> covariate-complete analytic sample.

Numbers computed fresh from source:
  HPP  baseline cohort (10k_cov_mean_updated_baseline.csv): 13,456 participants
        -> archived whole-body DXA (DICOM archive): 8,920 participants / 11,700 scan-sessions
        -> usable HDF5 dxa_dataset.h5: 8,820 participants / 11,540 scans (pretraining + internal eval)
        -> complete age/sex/BMI: 8,759 analytic
  UKBB full release (ukbb_tabular_data_for_cox_with_baseline.csv): 502,244 participants
        -> instance-2 whole-body DXA zips (dexa_images/*_2_0.zip): 65,904 participants
        -> MONOCHROME2 + QC + age/sex: 47,400 selected
        -> complete BMI: 45,789 analytic

Font sizes are larger than the nature-plot-style default (6 pt) by explicit request for
a readable standalone flow chart; other style elements (Set2 palette, thin 0.6-pt strokes,
PNG+PDF at 400 dpi, no title) follow the skill.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from pathlib import Path

palette = sns.color_palette("Set2", n_colors=8)
_style_dir = Path.home() / ".claude/skills/nature-plot-style/style_files"
double_style = str(_style_dir / "nature_double.mplstyle")
plt.rcParams["figure.dpi"] = 150

OUT = "figures/extended_data_fig1_cohort_flow"

plt.style.use(double_style)              # Set2 palette, 0.6-pt strokes, 400-dpi save
fig, ax = plt.subplots(figsize=(14, 9.5))
ax.set_xlim(-0.37, 1.37); ax.set_ylim(0, 1); ax.axis("off")

BW, BH = 0.42, 0.14          # main box width/height
EW, EH = 0.40, 0.11          # excluded box
FS = 11                       # body font (enlarged per request)
HFS = 14                      # header
CFS = 10                      # caption/footnote
XL, XR = 0.27, 0.73          # column centres

def box(cx, cy, w, h, text, facecolor, edgecolor="black", fs=FS, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=0.6, edgecolor=edgecolor, facecolor=facecolor, clip_on=False))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight,
            clip_on=False, linespacing=1.35)

def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0, color="black", clip_on=False))

def elbow(x0, y, x1):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=9, linewidth=0.8, color="0.4", clip_on=False))

cols = {
    "HPP": dict(x=XL, sign=-1, fill=palette[0], header="Internal — HPP",
                rows=["HPP cohort (baseline)\n13,456 participants",
                      "Whole-body DXA\nGE Lunar Prodigy Advance\n11,700 scans, 8,920 participants",
                      "Usable scans\n11,540 scans, 8,820 participants\n(pretraining + internal eval)",
                      "Analytic sample\n8,759 participants\n(complete age, sex, BMI)"],
                excl=["No whole-body DXA\nscan available\n4,536 participants",
                      "Excluded: 160 scans /\n100 participants\n(degraded / partial scans)",
                      "Excluded: 61 participants\n(missing age, sex or BMI)"]),
    "UKBB": dict(x=XR, sign=+1, fill=palette[1], header="External — UK Biobank",
                 rows=["UK Biobank cohort\n502,244 participants",
                       "Whole-body DXA\nGE Lunar iDXA\n65,904 participants",
                       "Selected cohort\n47,400 participants\n(MONOCHROME2, imaging visit)",
                       "Analytic sample\n45,789 participants\n(complete age, sex, BMI)"],
                 excl=["No instance-2\nwhole-body DXA\n436,340 participants",
                       "Excluded: 18,504 participants\nnon-MONOCHROME2 / failed QC\nor missing age or sex",
                       "Excluded: 1,611 participants\n(missing BMI)"]),
}

ys = [0.895, 0.625, 0.365, 0.105]        # 4 main rows
mids = [(ys[i] + ys[i + 1]) / 2 for i in range(3)]

for c in cols.values():
    x, s = c["x"], c["sign"]
    ax.text(x, 0.985, c["header"], ha="center", va="center", fontsize=HFS, weight="bold",
            clip_on=False)
    for i, y in enumerate(ys):
        box(x, y, BW, BH, c["rows"][i], c["fill"])
    for i, my in enumerate(mids):
        arrow(x, ys[i] - BH / 2, ys[i + 1] + BH / 2)
        elbow(x, my, x + s * (BW / 2 + 0.005))
        box(x + s * (BW / 2 + 0.005 + EW / 2), my, EW, EH, c["excl"][i], "0.92", edgecolor="0.4")

ax.text(0.5, 0.008,
        "No diagnosis- or medication-based exclusions were applied; "
        "exclusions were limited to scan format/quality and covariate completeness.",
        ha="center", va="center", fontsize=CFS, style="italic", color="0.25", clip_on=False)

plt.savefig(OUT + ".png", dpi=400, facecolor="white", transparent=False, bbox_inches="tight")
plt.savefig(OUT + ".pdf", dpi=400, facecolor="white", transparent=False, bbox_inches="tight", format="pdf")
print("saved:", OUT + ".png / .pdf")
