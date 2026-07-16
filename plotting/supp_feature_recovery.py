"""
Supplementary / Extended Data: direct recovery of scanner-measured anthropometrics
and DXA body-composition measurements by the imaging embeddings (HPP).
Horizontal bars: Pearson r per measurement, DeepDXA vs DINOv3 (imaging-only).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
RES = "/data/hpp_labdata/Analyses/gilsa/results/comparison"
RAW = os.path.join(RES, "lp_recovery_raw.csv")
OUT_DIR = "figures"

COL = {"lejepa": "#083c7d", "dino": "#7fb9dc"}
LABELS = {"lejepa": "DeepDXA", "dino": "DINOv3"}
DISPLAY = {
    "bmi": "BMI", "total_fat_mass": "Total fat mass", "total_lean_mass": "Total lean mass",
    "android_fat_mass": "Android fat", "gynoid_fat_mass": "Gynoid fat",
    "appendicular_lean_mass": "Appendicular lean", "total_bmd": "Total BMD", "vat_mass": "VAT mass",
}

raw = pd.read_csv(RAW)
g = raw.groupby(["target", "model"])["score"]
mean = g.mean().unstack(); se = (g.std() / np.sqrt(g.count())).unstack()

# order by DeepDXA r (descending → highest recovery at top)
order = mean["lejepa"].sort_values(ascending=True).index.tolist()

_style = Path.home() / ".claude/skills/nature-plot-style/style_files/nature_single.mplstyle"
if _style.exists():
    plt.style.use(str(_style))
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 400, "font.size": 8})

arms = ["dino", "lejepa"]   # draw DINOv3 then DeepDXA (DeepDXA on top within group)
bar_h = 0.36
fig, ax = plt.subplots(figsize=(4.6, 0.55 * len(order) + 0.8))
for i, t in enumerate(order):
    for ai, arm in enumerate(arms):
        y = i + (ai - 0.5) * bar_h
        ax.barh(y, mean.loc[t, arm], height=bar_h * 0.92, color=COL[arm],
                edgecolor="none", zorder=2)
        ax.errorbar(mean.loc[t, arm], y, xerr=se.loc[t, arm], fmt="none",
                    color="black", capsize=1.5, lw=0.6, capthick=0.6, zorder=3)
        ax.text(mean.loc[t, arm] + se.loc[t, arm] + 0.02, y, f"{mean.loc[t, arm]:.2f}",
                va="center", ha="left", fontsize=6.5, color="#222")

ax.set_yticks(range(len(order)))
ax.set_yticklabels([DISPLAY.get(t, t) for t in order], fontsize=8)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Pearson r", fontsize=8)
ax.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
handles = [mpatches.Patch(color=COL[a], label=LABELS[a]) for a in ["lejepa", "dino"]]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8,
          handlelength=1.2, borderpad=0.3)
plt.tight_layout()
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"supp_feature_recovery.{ext}")
    plt.savefig(p, dpi=400, facecolor="white", bbox_inches="tight")
    print("Saved:", p)
