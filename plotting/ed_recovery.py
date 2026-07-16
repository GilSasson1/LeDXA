"""
Extended Data: LeDXA recovers DXA-derived phenotypes from raw images.
(a) osteopenia ROC, (b) osteoporosis ROC — LeDXA-bone vs BMD-readout vs DINOv3-bone
    (imaging/readout only, no covariates; representative 80/20 split).
(c) continuous body-composition recovery (Pearson r), LeDXA vs DINOv3.
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc as auc_fn, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common.utils as U
from downstream.disease.linear_probe import load_embeddings, _fit_predict, _fit_predict_blocks, _impute

DXA_TAB = "/path/to/dxa_filtered.csv"
TGT     = "csvs/disease_targets_with_covs.csv"
RES     = "/data/hpp_labdata/Analyses/gilsa/results/comparison"
OUT_DIR = "figures"
SEED    = 42
ROC_COL = {"lejepa": "#083c7d", "bmd": "#8ccbb3", "dino": "#7fb9dc"}
ROC_LBL = {"lejepa": "LeDXA", "bmd": "BMD readout", "dino": "DINOv3"}

# Env overrides: point at baseline, bone-pool, or region-pool CSVs / output name.
# ED_BONE_POOL=1 folds the mean-pooled lumbar/femur regional scan embedding into
# the bone-view embedding, matching compare_lp.py --bone-pool.
# ED_REGION_ONLY=1 uses only the mean-pooled lumbar/femur regional scan embedding,
# matching compare_lp.py --region-only.
# ED_REGION_POOL=1 appends the mean-pooled regional-scan block to the SSL ROC arms
# (per-block penalized), matching the lp_strict_bone_regionpool analysis.
BONE_POOL   = os.environ.get("ED_BONE_POOL", "0") == "1"
REGION_ONLY = os.environ.get("ED_REGION_ONLY", "0") == "1"
REGION_POOL = os.environ.get("ED_REGION_POOL", "0") == "1"
STRICT_CSV  = os.environ.get("ED_STRICT_CSV", f"{RES}/lp_strict_bone_raw.csv")
REC_CSV     = os.environ.get("ED_REC_CSV",    f"{RES}/lp_recovery_raw.csv")
OUT_NAME    = os.environ.get("ED_OUT_NAME",   "ed_feature_recovery")
if sum(bool(x) for x in (BONE_POOL, REGION_ONLY, REGION_POOL)) > 1:
    raise SystemExit("Use only one of ED_BONE_POOL=1, ED_REGION_ONLY=1, or ED_REGION_POOL=1.")

print("loading embeddings…")
embs = load_embeddings(U.EMBEDDINGS_DIR, ["lejepa", "dino"])
dt  = pd.read_csv(TGT).set_index(["RegistrationCode", "research_stage"])
tab = pd.read_csv(DXA_TAB).set_index(["RegistrationCode", "research_stage"])
tab = tab[~tab.index.duplicated(keep="first")]
bmd_cols = [c for c in tab.columns if c.endswith("_bmd")]


def _first_scan(idx):
    f = idx.to_frame(index=False).sort_values("research_stage").drop_duplicates("RegistrationCode", keep="first")
    return pd.MultiIndex.from_frame(f)


def roc_for(tcol):
    """Pool out-of-sample predictions across the 10 analysis splits → stable ROC;
    legend AUC = mean per-split AUC (matches the strict-bone analysis).
    Under REGION_ONLY the SSL arms are the lumbar/femur regional embedding only.
    Under BONE_POOL the SSL arms are [bone* + tissue], where bone* is the mean
    of whole-body bone and lumbar/femur regional embeddings. Under REGION_POOL
    the SSL arms become [bone | tissue | regional-pool], each a separately
    penalized block. Mirrors compare_lp.py for these analysis regimes."""
    y = pd.to_numeric(dt[tcol], errors="coerce").dropna()
    blb, bdb = embs["lejepa"]["bone"], embs["dino"]["bone"]
    idx = y.index.intersection(blb.index).intersection(bdb.index).intersection(tab.index)
    idx = _first_scan(idx)
    yv = y.loc[idx].astype(int).values

    def ssl_arm(mk):
        if REGION_ONLY:
            rp = embs[mk]["regionpool"].reindex(idx).astype(float)
            ok = ~rp.isna().all(axis=1)
            return rp.loc[ok].values, [rp.shape[1]], ok.values
        bone = embs[mk]["bone"].loc[idx].astype(float).copy()
        if BONE_POOL:
            rp = embs[mk]["regionpool"].reindex(idx).astype(float)
            ok = rp.notna().all(axis=1)
            bone.loc[ok] = (bone.loc[ok].values + rp.loc[ok].values) / 2.0
        cols = [bone.values, embs[mk]["tissue"].loc[idx].values.astype(float)]
        sizes = [cols[0].shape[1], cols[1].shape[1]]
        if REGION_POOL:
            rp = embs[mk]["regionpool"].reindex(idx).values.astype(float)
            cols.append(rp); sizes.append(rp.shape[1])
        return np.concatenate(cols, axis=1), sizes, None

    Xlej, sz_lej, ok_lej = ssl_arm("lejepa")
    Xdin, sz_din, ok_din = ssl_arm("dino")
    if REGION_ONLY:
        ok = ok_lej & ok_din
        idx = idx[ok]
        yv = y.loc[idx].astype(int).values
        Xlej, sz_lej, _ = ssl_arm("lejepa")
        Xdin, sz_din, _ = ssl_arm("dino")
    arms  = {"lejepa": Xlej, "dino": Xdin, "bmd": tab.loc[idx, bmd_cols].values.astype(float)}
    sizes = {"lejepa": sz_lej, "dino": sz_din, "bmd": [arms["bmd"].shape[1]]}

    pooled = {k: ([], []) for k in arms}
    per_seed_auc = {k: [] for k in arms}
    for sd in U.make_seeds(10):
        tr, va = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=sd)
        for k, X in arms.items():
            Xtr, Xva = _impute(X[tr], X[va])
            if REGION_POOL and k in {"lejepa", "dino"}:
                s, _ = _fit_predict_blocks(Xtr, Xva, yv[tr], yv[va], True, seed=sd, block_sizes=sizes[k])
            else:
                s, _ = _fit_predict(Xtr, Xva, yv[tr], yv[va], True, seed=sd)
            pooled[k][0].extend(yv[va].tolist()); pooled[k][1].extend(s.tolist())
            per_seed_auc[k].append(roc_auc_score(yv[va], s))
    out = {}
    am = {"lejepa": "lejepa", "dino": "dino", "bmd": "tabular"}  # legend AUC = canonical strict-bone mean
    for k in arms:
        fpr, tpr, _ = roc_curve(pooled[k][0], pooled[k][1])
        out[k] = (fpr, tpr, float(STRICT_AUC.loc[(tcol, am[k])]))
    return out, int((yv == 1).sum())


STRICT_AUC = pd.read_csv(STRICT_CSV).groupby(["target", "model"])["score"].mean()

print("osteopenia…"); roc_open, n_open = roc_for("dis__osteopenia")
print("osteoporosis…"); roc_oporo, n_oporo = roc_for("dis__osteoporosis")

# ── recovery bars ──────────────────────────────────────────────────────────────
raw = pd.read_csv(REC_CSV)
g = raw.groupby(["target", "model"])["score"]
rmean = g.mean().unstack(); rse = (g.std() / np.sqrt(g.count())).unstack()
DISP = {"bmi": "BMI", "total_fat_mass": "Total fat mass", "total_lean_mass": "Total lean mass",
        "android_fat_mass": "Android fat", "gynoid_fat_mass": "Gynoid fat",
        "appendicular_lean_mass": "Appendicular lean", "vat_volume": "VAT volume",
        "femur_neck_bmd": "Femoral neck BMD", "femur_total_bmd": "Total hip BMD",
        "lumbar_l1l4_bmd": "Lumbar BMD (L1–L4)"}
border = rmean["lejepa"].sort_values(ascending=True).index.tolist()

_style = Path.home() / ".claude/skills/nature-plot-style/style_files/nature_double.mplstyle"
plt.style.use(str(_style))
plt.rcParams.update({"figure.dpi": 150, "font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7})

fig = plt.figure(figsize=(8.2, 5.8))
gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1], wspace=0.34, hspace=0.5)
ax_a = fig.add_subplot(gs[:, 0]); ax_b = fig.add_subplot(gs[0, 1]); ax_c = fig.add_subplot(gs[1, 1])


def _draw_roc(ax, roc, title, n):
    for k in ["lejepa", "bmd", "dino"]:
        fpr, tpr, a = roc[k]
        ax.plot(fpr, tpr, color=ROC_COL[k], lw=0.9, label=f"{ROC_LBL[k]} ({a:.2f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.6, ls="--")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{title} (n={n})")
    ax.legend(loc="lower right", handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)


_draw_roc(ax_b, roc_open, "Osteopenia", n_open)
_draw_roc(ax_c, roc_oporo, "Osteoporosis", n_oporo)

arms = ["dino", "lejepa"]; bar_h = 0.36
RC = {"lejepa": "#083c7d", "dino": "#7fb9dc"}
for i, t in enumerate(border):
    for ai, arm in enumerate(arms):
        yb = i + (ai - 0.5) * bar_h
        ax_a.barh(yb, rmean.loc[t, arm], height=bar_h * 0.92, color=RC[arm], zorder=2)
        ax_a.errorbar(rmean.loc[t, arm], yb, xerr=rse.loc[t, arm], fmt="none", color="black",
                      capsize=1.3, lw=0.6, capthick=0.6, zorder=3)
        ax_a.text(rmean.loc[t, arm] + rse.loc[t, arm] + 0.02, yb, f"{rmean.loc[t, arm]:.2f}",
                  va="center", ha="left", fontsize=6.5, color="#222")
ax_a.set_yticks(range(len(border))); ax_a.set_yticklabels([DISP[t] for t in border])
ax_a.set_xlim(0, 1.12); ax_a.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_a.set_xlabel("Pearson r"); ax_a.grid(axis="x", lw=0.4, alpha=0.5, zorder=0)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.legend(handles=[mpatches.Patch(color=RC[a], label={"lejepa": "LeDXA", "dino": "DINOv3"}[a])
                     for a in ["lejepa", "dino"]], loc="upper center",
            bbox_to_anchor=(0.5, -0.08), ncol=2)

for ax, lab, dx in [(ax_a, "a", -0.30), (ax_b, "b", -0.17), (ax_c, "c", -0.17)]:
    ax.text(dx, 1.05, lab, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")

for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"{OUT_NAME}.{ext}")
    plt.savefig(p, dpi=400, facecolor="white", transparent=False, bbox_inches="tight")
    print("Saved:", p)
