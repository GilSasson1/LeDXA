"""
plot_fig3_cox.py — combined Cox Figure 3.

Panel (a): C-index for endpoints where DeepDXA+cov significantly beats DXA
Tabular+cov specifically (FDR q<SIG_Q) — the core "adds value over the existing
clinical DXA measurement" claim, not necessarily beating DINOv3 too — plus
Osteoporosis (FORCE_INCLUDE), shown despite no significant win because DXA-measured
BMD IS the diagnostic gold standard for that disease, so matching it (marked 'ns')
is itself the point. Bars = across-seed mean, error bars = across-seed SE. Bracket
markers show every comparator (DINOv3/Tabular/Covariates) DeepDXA beats
significantly: '*' q<SIG_Q, '**' q<0.01 — each comparison type FDR-corrected
separately across all endpoints by cox_regression_comparison.py itself (the correct
testing family per claim, reused as-is rather than re-derived). Bracket height
clears every bar spanned between the two arms being compared, not just those two,
so a bracket never visually collides with an intervening bar (e.g. DINOv3 sitting
between DeepDXA and Tabular). The full endpoint set (including nulls) is in the
companion Supplementary Table (export_supp_table) — all 4 arms and all 3
DeepDXA-vs-comparator tests (raw p, FDR q), N/events/follow-up — so nothing from
the analysis is silently dropped, without a second lossy summary figure alongside it.

Panel (b): cumulative incidence by risk quartile for a deliberate contrast set
of four endpoints (PANEL_B_ENDPOINTS) — hip and knee arthrosis (the two large
wins), type 2 diabetes (a modest-but-real win, showing the effect is graded not
binary), and osteoporosis (a deliberate negative control: DXA-measured BMD IS
osteoporosis's diagnostic criterion, so no gain is mechanistically expected) —
top-risk-quartile (Q4) cumulative incident-event capture, DeepDXA+Cov vs
DXA Tabular+Cov; capture@Q4 annotated at years 1/2/3 and end of follow-up.

Colours follow the fig2 palette (plot_combined_figure.MODEL_COLORS).

Panel (a) and the supplementary table both read the main (non-perseed) summary file
emitted by cox_regression_comparison.py (`<out_prefix>.csv` — MAIN_CSV). Panel (b)
risk scores are recomputed here as out-of-fold joint Cox.

Panel (b) significance is a PAIRED SUBJECT-LEVEL BOOTSTRAP of the Capture@Q4
difference (DeepDXA - DXA Tabular): both arms are scored on the same bootstrap
resample of subjects, the top-risk quartile is re-thresholded within each resample
(Q4 is defined relative to the cohort), and the difference yields a 95% percentile
CI and a two-sided bootstrap p at each time cutoff. This quantifies uncertainty from
the finite patient cohort — the claim the panel actually makes. It replaces the old
Wilcoxon over CV fold-shuffle seeds, which only tested fold-partition stability (not
population significance) and had a hard two-tailed p floor of 2/2**n_seeds (n=10 ->
0.002). p-values are BH-FDR corrected across the four PANEL_B_ENDPOINTS at each
cutoff — the same DeepDXA-vs-Tabular testing family panel (a) corrects over.
"""
import os
import sys
import importlib.util
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, FuncFormatter, LogLocator, NullFormatter
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # repo root
sys.path.insert(0, _ROOT)


def _load(mod_name, path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


_crc = _load("crc", os.path.join(_ROOT, "downstream", "survival", "cox_regression.py"))
_cmp = _load("cox_cmp", os.path.join(_HERE, "cox_model_comparison.py"))
get_survival_target = _crc.get_survival_target
preprocess_block = _crc.preprocess_block
fit_joint_cox = _crc.fit_joint_cox
load_clean_data = _crc.load_clean_data
split_tabular_features = _crc.split_tabular_features
_load_embeddings = _crc._load_embeddings
_prettify_event = _cmp._prettify_event
_EVENT_CATEGORIES = _cmp._EVENT_CATEGORIES
_CATEGORY_NAMES = _cmp._CATEGORY_NAMES

EVENTS_PATH  = "/path/to/project/ukbb_osteo_data_expanded_aligned.csv"
# Canonical Fig 3 regime = bp_logsweep: BONE-POOL fusion, PCA(100 emb / 30 dxa), per-endpoint
# SYMMETRIC penalty SELECTION over a log grid {0.1,0.3,1,3,10,30} (grid search) — both the
# embedding and tabular block penalties chosen per endpoint by the same procedure (fair). cov=0.1.
# Panel a reads the grid-search per-seed CSV; panel b recomputes risk with the same per-endpoint
# selectors (FIG3_FIXED_PEN unset) over the matching SWEEP_GRID. NB panel-a CSV is the fast
# lean/holdout grid-search run; panel-b recompute is 5-fold OOF — protocols differ.
_BONEPOOL = "/data/hpp_labdata/Analyses/gilsa/embeddings/ukbb_comparison/bonepool"
LEJEPA_PATH  = f"{_BONEPOOL}/lejepa_fusion.pkl"
REGION_PATH  = None
TABULAR_PATH = "/path/to/project/ukbb_tabular_data_for_cox_with_baseline.csv"
PERSEED_CSV  = os.environ.get("FIG3_PERSEED",
    os.path.join(_ROOT, "tables", "cox_ttest_results_bp_logsweep_nodxapca_perseed.csv"))
# Main (non-perseed) summary output from the same run — has per-arm mean/SE and
# all pairwise DeepDXA-vs-comparator raw p / FDR-adjusted q (each comparison type
# corrected separately across endpoints), plus N/events/follow-up. Panel (a), the
# extended-data figure, and the supplementary table all read from this file.
MAIN_CSV = os.environ.get("FIG3_MAIN_CSV", PERSEED_CSV.replace("_perseed.csv", ".csv"))
# Panel b: leave FIG3_FIXED_PEN UNSET so it selects emb & tabular penalties per-endpoint over
# SWEEP_GRID (matches the grid-search panel a). Set the env var to a number to force a fixed pen.
FIG3_FIXED_PEN = os.environ.get("FIG3_FIXED_PEN", "") or None
OUT_PATH        = os.path.join(_ROOT, "figures", "fig3_cox_survival")
SUPP_TABLE_PATH = os.path.join(_ROOT, "tables", "fig3_supp_table_all_endpoints.csv")

BASELINE_COL = "Date of attending assessment centre - visit 2"
DEATH_COL    = "Date of death - visit 0"
ADMIN_CENSOR_DATE = "2023-03-31"
PEN_COV, PEN_DXA, PEN_EMB = 0.1, 1.0, 1.0   # cov fixed 0.1; emb/dxa are per-endpoint-selected fallbacks
# Fallback only (when FIG3_FIXED_PEN is unset): per-endpoint symmetric selection of the
# embedding AND tabular block penalties with the same deterministic selectors / grid as
# panel a, so the tabular baseline is not handicapped by a fixed heavy penalty.
SWEEP_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]   # matches the bp_logsweep grid-search
N_FOLDS, SEED = 5, 42

# Palette aligned with Fig 2 MODEL_COLORS (plot_combined_figure.py)
# Bar order matches Fig 2: DeepDXA → DINOv3 → Tabular → Covariates (left to right)
ARMS = [  # (csv arm name, display, colour)
    ("DXA SSL (LeJEPA) + Covariates", "LeDXA + cov",       "#083c7d"),
    ("DXA SSL (DINO) + Covariates",   "DINOv3 + cov",      "#7fb9dc"),
    ("DXA Tabular + Covariates",      "DXA Tabular + cov", "#8ccbb3"),
    ("Covariates",                    "Covariates",        "#bdbdbd"),
]
COL_DXAFM, COL_TAB = "#083c7d", "#8ccbb3"   # panel b recall-curve colours

DEEP_ARM = ARMS[0][0]     # DeepDXA+cov arm name, shared by panel (a) and the supplementary
TABULAR_ARM = ARMS[2][0]  # DXA Tabular+cov — the specific comparator panel-a inclusion gates on
SIG_Q = 0.05               # FDR-adjusted significance threshold for panel-a inclusion
# Shown in panel (a) even without a significant win vs Tabular: DXA-measured BMD is the
# diagnostic gold standard for osteoporosis, so "not significantly different" IS the story
# (the learned embedding matches the definitional clinical measurement for this one).
FORCE_INCLUDE = {"Osteoporosis"}

_SHORT_LABELS = {
    "Intervertebral Disk Disease": "Intervert. disk",
    "Ischemic Heart Disease":      "Ischaemic HD",
    "Ischaemic Heart Disease":     "Ischaemic HD",
    "Chronic Renal Failure":       "Chr. renal fail.",
    "Atrial Fibrillation":         "Atrial fibr.",
    "Type 2 Diabetes":             "Type 2 diabetes",
    "Cerebral Infarction":         "Cerebral infarct.",
    "Knee Arthrosis":              "Knee arthrosis",
    "Hip Arthrosis":               "Hip arthrosis",
    "Liver Disease":               "Liver disease",
    "Disk Disease":                "Disk disease",
}


def _short_label(label):
    return _SHORT_LABELS.get(label, label)


# Deliberate contrast set, not a mechanical top-N-by-gain list: two large wins
# (hip/knee), one modest-but-real win (T2D, showing the effect is graded rather
# than all-or-nothing), and one negative control (osteoporosis — see below).
PANEL_B_ENDPOINTS = [
    ("Hip arthrosis (M16)",      "Date M16 first reported (coxarthrosis [arthrosis of hip]) - visit 0"),
    ("Knee arthrosis (M17)",     "Date M17 first reported (gonarthrosis [arthrosis of knee]) - visit 0"),
    ("Type 2 Diabetes (E11)",    "Date E11 first reported (non-insulin-dependent diabetes mellitus) - visit 0"),
    ("Osteoporosis (M81)",       "Date M81 first reported (osteoporosis without pathological fracture) - visit 0"),
]


# Paired subject-level bootstrap for the panel-b Capture@Q4 CI. Risk scores are the
# single-seed OOF joint Cox already drawn as the curves; the bootstrap resamples the
# PATIENT cohort (not CV seeds), so the CI reflects sampling of subjects.
# 10k resamples so the two-sided bootstrap p (min ~2/(N_BOOT+1)) can resolve past
# 0.001 even after FDR; the 95% CI, the primary reported quantity, has no such floor.
N_BOOT, BOOT_SEED = 10000, 12345


def stars(p):
    if p is None or not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def cox_oof_joint(blocks_raw, t, e, n_folds=N_FOLDS, seed=SEED):
    """blocks_raw: list of (X_raw, apply_pca, n_components, penalizer)."""
    oof = np.full(len(t), np.nan)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(blocks_raw[0][0], e):
        blocks, ok = [], True
        for X_raw, apply_pca, n_comp, pen in blocks_raw:
            X_tr, X_te = preprocess_block(X_raw[tr], X_raw[te], apply_pca=apply_pca,
                                          n_components=n_comp)
            if X_tr is None:
                ok = False
                break
            blocks.append((X_tr, X_te, pen))
        if not ok:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = fit_joint_cox(blocks, t[tr], e[tr])
        if pred is not None:
            oof[te] = np.asarray(pred, dtype=float)
    return oof


def cumulative_capture_curve(risk, t_days, e):
    """
    Cumulative events captured in Q4 (top 25%) and Q1 (bottom 25%)
    normalised by total events ever — running recall vs. time.
    y(t) = events_in_group_up_to_t / total_events_ever.
    Starts at 0, ends at recall@group.
    """
    v = np.isfinite(risk)
    risk_v, t_yr, ev = risk[v], t_days[v] / 365.25, e[v]
    in_q4 = risk_v >= np.quantile(risk_v, 0.75)
    in_q1 = risk_v <= np.quantile(risk_v, 0.25)

    total = int(ev.sum())
    if total == 0:
        return None

    emask   = ev == 1
    etimes  = t_yr[emask]
    q4_flag = in_q4[emask].astype(float)
    q1_flag = in_q1[emask].astype(float)

    order   = np.argsort(etimes)
    etimes  = etimes[order]
    q4_flag = q4_flag[order]
    q1_flag = q1_flag[order]

    cum_q4 = np.cumsum(q4_flag) / total
    cum_q1 = np.cumsum(q1_flag) / total

    times = np.concatenate([[0.0], etimes])
    cq4   = np.concatenate([[0.0], cum_q4])
    cq1   = np.concatenate([[0.0], cum_q1])

    last_ev_t = float(t_yr[ev == 1].max()) if ev.sum() > 0 else float(t_yr.max())
    return times, cq4, cq1, float(cum_q4[-1]), float(cum_q1[-1]), last_ev_t, total


CAPTURE_TIMEPOINTS_YR = [1, 2, 3, None]   # None = full follow-up


def bootstrap_capture_diff(risk_fm, risk_tab, t_days, e,
                           timepoints=CAPTURE_TIMEPOINTS_YR, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired subject-level bootstrap of the Capture@Q4 difference (DeepDXA - Tabular).

    Both arms are scored on the SAME resample of subjects each iteration (paired), so
    the difference isolates the model effect from subject-sampling noise; the top-risk
    quartile is re-thresholded within each resample because Q4 is defined relative to
    the cohort. This measures uncertainty from the finite patient sample — the claim
    the panel makes — unlike a test over CV fold-shuffles.

    Returns {tc: dict(mean_fm, mean_tab, diff, lo, hi, p, n_events)} where diff is the
    full-cohort point estimate, [lo, hi] the 95% percentile CI, and p a two-sided
    bootstrap p-value (percentile min-tail, +1 corrected so it never hits exactly 0).
    """
    v = np.isfinite(risk_fm) & np.isfinite(risk_tab)
    rf, rt = risk_fm[v], risk_tab[v]
    t_yr, ev = t_days[v] / 365.25, e[v].astype(int)
    n = len(rf)
    rng = np.random.default_rng(seed)

    def _cap(in_q4, emask):
        tot = int(emask.sum())
        return (float(in_q4[emask].sum()) / tot) if tot else np.nan

    # Full-cohort point estimates (identical definition to cumulative_capture_curve's endpoint).
    q4_fm = rf >= np.quantile(rf, 0.75)
    q4_tab = rt >= np.quantile(rt, 0.75)
    point, n_ev = {}, {}
    for tc in timepoints:
        emask = (ev == 1) if tc is None else ((t_yr <= tc) & (ev == 1))
        n_ev[tc] = int(emask.sum())
        point[tc] = (_cap(q4_fm, emask), _cap(q4_tab, emask))

    boot = {tc: np.full(n_boot, np.nan) for tc in timepoints}
    for b in range(n_boot):
        bi = rng.integers(0, n, n)                       # same draw for both arms → paired
        rfb, rtb, tyb, evb = rf[bi], rt[bi], t_yr[bi], ev[bi]
        q4fb = rfb >= np.quantile(rfb, 0.75)
        q4tb = rtb >= np.quantile(rtb, 0.75)
        for tc in timepoints:
            emask = (evb == 1) if tc is None else ((tyb <= tc) & (evb == 1))
            tot = int(emask.sum())
            if tot:
                boot[tc][b] = q4fb[emask].sum() / tot - q4tb[emask].sum() / tot

    out = {}
    for tc in timepoints:
        cf, ct = point[tc]
        d = boot[tc][np.isfinite(boot[tc])]
        if d.size < 2:
            out[tc] = dict(mean_fm=cf, mean_tab=ct, diff=cf - ct,
                           lo=np.nan, hi=np.nan, p=np.nan, n_events=n_ev[tc])
            continue
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2.0 * min((np.sum(d <= 0) + 1) / (d.size + 1),
                      (np.sum(d >= 0) + 1) / (d.size + 1))
        out[tc] = dict(mean_fm=cf, mean_tab=ct, diff=cf - ct,
                       lo=float(lo), hi=float(hi), p=min(float(p), 1.0),
                       n_events=n_ev[tc])
    return out


def _bh_fdr(pvals):
    """Benjamini-Hochberg adjusted q-values; NaN entries are excluded from the ranking
    and passed through as NaN."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = idx.size
    if m == 0:
        return q
    order = idx[np.argsort(p[idx])]
    adj = p[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]        # enforce monotone non-decreasing
    q[order] = np.clip(adj, 0, 1)
    return q


# ── Shared: per-endpoint means + DeepDXA-vs-comparator significance, read directly
# from cox_regression_comparison.py's own summary output. Each comparison type (vs
# Tabular, vs DINOv3, vs Covariates) is FDR-corrected THERE separately across all
# endpoints for that one comparison — the correct testing family for a specific claim
# like "does DeepDXA beat Tabular?" — so it's reused as-is rather than re-derived from
# a pooled, mixed-comparator correction. ─────────────────────────────────────────────
def compute_endpoint_significance(main_df):
    """Per-endpoint arm means/SEs and DeepDXA-vs-each-comparator raw p / FDR q, pulled
    from the main (non-perseed) cox_regression_comparison.py output.

    Returns a list of row dicts: event, label, cat, means, ses, best/best_score (display
    ordering only), ranked_comps (non-deep arms, best mean first), raw_p/adj_p (dict per
    comparator), n, events, followup_yr, qualifies (beats DXA Tabular+cov at FDR q<SIG_Q),
    wins (list of (comp_arm, 'fdr') for every comparator DeepDXA beats at FDR q<SIG_Q).
    """
    arm_names = [a[0] for a in ARMS]
    _PANEL_A_SKIP = {"Obesity"}   # definitional endpoint: BMI in covariate arm

    rows = []
    for _, row in main_df.iterrows():
        label = _prettify_event(row["Event"])
        if label is None or label in _PANEL_A_SKIP:
            continue
        means = {a: row.get(f"{a} C-Index", np.nan) for a in arm_names}
        ses   = {a: row.get(f"{a} C-Index SE", np.nan) for a in arm_names}
        if all(pd.isna(v) for v in means.values()):
            continue
        cat = _EVENT_CATEGORIES.get(label, 99)
        best = max((a for a in arm_names if pd.notna(means[a])), key=lambda a: means[a])
        ranked_comps = sorted(
            [(a, means[a]) for a in arm_names if a != DEEP_ARM and pd.notna(means[a])],
            key=lambda t: -t[1])
        raw_p = {comp_arm: row.get(f"P-Value ({DEEP_ARM} vs {comp_arm})", np.nan)
                 for comp_arm, _ in ranked_comps}
        adj_p = {comp_arm: row.get(f"P-Value-adj ({DEEP_ARM} vs {comp_arm})", np.nan)
                 for comp_arm, _ in ranked_comps}
        rows.append(dict(event=row["Event"], label=label, cat=cat, means=means, ses=ses,
                          best=best, best_score=means[best], ranked_comps=ranked_comps,
                          raw_p=raw_p, adj_p=adj_p,
                          n=row.get("N", np.nan), events=row.get("Total Events", np.nan),
                          followup_yr=row.get("Median Follow-up (yr)", np.nan)))
    rows.sort(key=lambda r: (r["cat"], -r["best_score"]))

    for r in rows:
        tab_mean = r["means"].get(TABULAR_ARM, np.nan)
        tab_q = r["adj_p"].get(TABULAR_ARM, np.nan)
        r["qualifies"] = bool(pd.notna(tab_mean) and pd.notna(tab_q)
                               and r["means"][DEEP_ARM] > tab_mean and tab_q < SIG_Q)
        # Wins vs EACH individual comparator (for the per-arm brackets in panel a) —
        # FDR q<SIG_Q only; no nominal/uncorrected tier.
        r["wins"] = [(comp_arm, "fdr") for comp_arm, comp_mean in r["ranked_comps"]
                     if r["means"][DEEP_ARM] > comp_mean
                     and pd.notna(r["adj_p"].get(comp_arm)) and r["adj_p"][comp_arm] < SIG_Q]
    return rows


# ── Panel (a): C-index for endpoints where DeepDXA+cov beats DXA Tabular+cov
# specifically (FDR q<SIG_Q) — the core "adds value over the clinical DXA
# measurement" claim, not necessarily beating DINOv3 too ──────────────────────
def draw_panel_a(ax, main_df):
    all_rows = compute_endpoint_significance(main_df)
    rows = [r for r in all_rows if r["qualifies"] or r["label"] in FORCE_INCLUDE]

    arm_names = [a[0] for a in ARMS]
    n_arms = len(ARMS)
    bar_w = 0.17
    offsets = np.linspace(-0.27, 0.27, n_arms)
    deep_idx = next(i for i, (a, _, _) in enumerate(ARMS) if a == DEEP_ARM)

    x_pos, labels, cat_centers, dividers = [], [], [], []
    x = 0.0
    last_cat = None
    EVENT_SPACING = 1.0
    GAP = 0.58
    for r in rows:
        if last_cat is not None and r["cat"] != last_cat:
            dividers.append(x + (GAP - EVENT_SPACING) / 2)
            x += GAP
        if last_cat != r["cat"]:
            cat_centers.append([x, x, r["cat"]])
        cat_centers[-1][1] = x
        last_cat = r["cat"]
        x_pos.append(x)
        labels.append(_short_label(r["label"]))
        x += EVENT_SPACING

    scores_all = []
    for xp, r in zip(x_pos, rows):
        for (arm, disp, color), off in zip(ARMS, offsets):
            m = r["means"][arm]
            if not np.isfinite(m):
                continue
            se = r["ses"].get(arm, 0.0)
            se = se if np.isfinite(se) else 0.0
            ax.bar(xp + off, m, width=bar_w, color=color, alpha=0.92,
                    yerr=se, ecolor="#444444", capsize=1.5, zorder=3,
                    error_kw=dict(lw=0.8))
            scores_all.append(m + se)

    ymin = max(0.55, float(np.nanmin([r["means"][a] for r in rows
               for a in arm_names if np.isfinite(r["means"][a])])) - 0.025)
    ymax = max(0.90, float(np.nanmax(scores_all)) + 0.09)

    # Brackets reuse the per-comparator FDR q already computed (each comparison type
    # corrected separately across all endpoints) by compute_endpoint_significance.
    # Height must clear EVERY bar between the two arms being bracketed (not just the
    # two endpoints of the bracket) — e.g. a DeepDXA-vs-Tabular bracket spans over the
    # DINOv3 bar sitting between them on the x-axis, so it must clear that bar too.
    # running_top makes multiple brackets on the same endpoint stack monotonically.
    # bkt_gap = clearance above the bars for the first (lowest) bracket; bkt_step = the
    # bigger clearance needed between STACKED bracket levels, since each level's star
    # sits a few points above its own line and needs room before the next line up.
    bkt_gap = 0.013
    bkt_step = 0.041
    brackets = []   # (x_comp, x_deep, y, sym)

    def _bar_top(r, arm):
        se = r["ses"].get(arm, 0.0)
        se = se if np.isfinite(se) else 0.0
        return r["means"][arm] + se

    def _span_top(r, off_a, off_b):
        lo, hi = min(off_a, off_b), max(off_a, off_b)
        return max(_bar_top(r, a) for (a, _, _), off in zip(ARMS, offsets)
                   if lo <= off <= hi and np.isfinite(r["means"][a]))

    for xp, r in zip(x_pos, rows):
        deep_mean = r["means"].get(DEEP_ARM, np.nan)
        if not np.isfinite(deep_mean):
            continue
        deep_off = offsets[deep_idx]
        running_top = _bar_top(r, DEEP_ARM)

        win_arms = {a for a, _ in r["wins"]}
        level = 0
        for comp_arm, disp, _ in ARMS[1:]:   # fixed left-to-right order = increasing offset distance
            if comp_arm not in win_arms or not np.isfinite(r["means"].get(comp_arm, np.nan)):
                continue
            comp_idx = next(i for i, (a, _, _) in enumerate(ARMS) if a == comp_arm)
            comp_off = offsets[comp_idx]
            gap = bkt_gap if level == 0 else bkt_step
            y = max(_span_top(r, deep_off, comp_off), running_top) + gap
            running_top = y
            q = r["adj_p"][comp_arm]
            sym = "**" if q < 0.01 else "*"
            brackets.append((xp + comp_off, xp + deep_off, y, sym))
            ymax = max(ymax, y + 0.050)
            level += 1

        if (TABULAR_ARM not in win_arms and r["label"] in FORCE_INCLUDE
                and np.isfinite(r["means"].get(TABULAR_ARM, np.nan))):
            # Explicit "ns" marker vs Tabular: shown despite no significant win because
            # matching (not necessarily beating) the diagnostic gold standard is the point.
            tab_idx = next(i for i, (a, _, _) in enumerate(ARMS) if a == TABULAR_ARM)
            tab_off = offsets[tab_idx]
            gap = bkt_gap if level == 0 else bkt_step
            y = max(_span_top(r, deep_off, tab_off), running_top) + gap
            brackets.append((xp + tab_off, xp + deep_off, y, "ns"))
            ymax = max(ymax, y + 0.050)

    from matplotlib.transforms import blended_transform_factory
    tick = (ymax - ymin) * 0.025   # symmetric tick length
    for x_comp, x_deep, y_bkt, sym in brackets:
        lo, hi = min(x_comp, x_deep), max(x_comp, x_deep)
        is_ns = (sym == "ns")
        ax.plot([lo, lo, hi, hi], [y_bkt - tick, y_bkt, y_bkt, y_bkt - tick],
                color="#999999" if is_ns else "#444444", lw=0.6,
                ls=":" if is_ns else "-", clip_on=False)
        # '*'/'**' glyphs render vertically centered on their anchor rather than sitting
        # on a baseline, so a small offset lets the tick line cut through the middle of
        # the star — needs a much bigger gap (and real bbox padding) than normal text.
        ax.annotate(sym, xy=((lo + hi) / 2, y_bkt), xytext=(0, 2.5 if is_ns else 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5 if is_ns else 8, color="#777777" if is_ns else "#222222",
                    fontweight="normal" if is_ns else "bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.95),
                    clip_on=False)

    ax.set_xlim(x_pos[0] - 0.6, x_pos[-1] + 0.6)
    ax.set_ylim(ymin, ymax)
    sep_top = ymin + (ymax - ymin) * 0.80
    for d in dividers:
        ax.vlines(d, ymin, sep_top, color="#E0E0E0", lw=1.8, zorder=0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("C-index", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", ls="--", color="#CCCCCC", alpha=0.8)
    ax.grid(axis="x", ls="--", color="#DDDDDD", alpha=0.35)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    _CAT_DISPLAY = {
        1: "Musculoskel.", 2: "Metabolic", 3: "Cardio.",
        4: "Resp.", 5: "Hep.&Renal", 6: "Neurol.",
    }
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for x0, x1, cat in cat_centers:
        mid = (x0 + x1) / 2
        pad = max(0.28, (x1 - x0) * 0.06)
        ax.plot([x0 - pad, x1 + pad], [1.022, 1.022],
                transform=trans, color="#aaaaaa", lw=0.6, clip_on=False)
        ax.text(mid, 1.036, _CAT_DISPLAY.get(cat, _CATEGORY_NAMES.get(cat, "Other")),
                transform=trans, ha="center", va="bottom",
                fontsize=6.5, fontweight="bold",
                color="#888888", style="italic", clip_on=False)


# ── Supplementary Table: full numeric breakdown backing panel (a) — all 4 arms x
# all 3 DeepDXA-vs-comparator tests, N/events/follow-up, for every endpoint tested.
def export_supp_table(main_df, out_csv):
    rows = compute_endpoint_significance(main_df)
    out = []
    for r in rows:
        rec = {"Disease": r["label"], "Category": _CATEGORY_NAMES.get(r["cat"], "Other"),
               "N": r["n"], "Events": r["events"], "Median follow-up (yr)": r["followup_yr"]}
        for arm_name, disp, _ in ARMS:
            rec[f"{disp} C-index"] = r["means"].get(arm_name, np.nan)
            rec[f"{disp} SE"] = r["ses"].get(arm_name, np.nan)
        for comp_arm, disp, _ in ARMS[1:]:
            rec[f"DeepDXA vs {disp}: raw p"] = r["raw_p"].get(comp_arm, np.nan)
            rec[f"DeepDXA vs {disp}: FDR q"] = r["adj_p"].get(comp_arm, np.nan)
        rec["Significant vs Tabular (FDR q<0.05)"] = r["qualifies"]
        out.append(rec)
    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


# ── Panel (b): cumulative events captured / total events vs. time ───────────────
def _annot_row(lbl, r):
    """Old-style row: 'Y1  75/43% p<0.001' — the two arms' Capture@Q4 and the p-value
    (raw two-sided bootstrap p; FDR q is still computed and logged for the caption)."""
    mf = f"{r['mean_fm']*100:.0f}" if np.isfinite(r['mean_fm']) else "–"
    mt = f"{r['mean_tab']*100:.0f}" if np.isfinite(r['mean_tab']) else "–"
    p = r.get('p', np.nan)
    if not np.isfinite(p):  ps = "n/a"
    elif p < 0.001:         ps = "p<0.001"
    elif p < 0.01:          ps = "p<0.01"
    elif p < 0.05:          ps = "p<0.05"
    else:                   ps = f"p={p:.2f}"
    return f"{lbl:<4}{mf}/{mt}% {ps}"


def draw_panel_b(axes, events, cov_df, dxa_df, emb_df, common, reg_df=None):
    n_cols_b = 2 if len(PANEL_B_ENDPOINTS) > 3 else len(PANEL_B_ENDPOINTS)

    # Pass 1: fit each endpoint, draw the Capture@Q4 curves, bootstrap the paired
    # DeepDXA-vs-Tabular difference. Annotations are deferred to pass 2 so the p-values
    # can be BH-FDR corrected across endpoints (the family panel a also corrects over).
    pending = []   # (ax, label, boot_results)
    for i, (ax, (label, event_col)) in enumerate(zip(axes, PANEL_B_ENDPOINTS)):
        is_left   = (i % n_cols_b) == 0
        is_bottom = i >= (len(PANEL_B_ENDPOINTS) - n_cols_b)
        target = get_survival_target(events, event_col, common, baseline_visit=2,
                                     death_col=DEATH_COL, admin_date=ADMIN_CENSOR_DATE)
        idx = target.index
        t = target["time"].values.astype(float)
        e = target["event"].values.astype(int)
        cov_raw = cov_df.loc[idx].values.astype(float)
        dxa_raw = dxa_df.loc[idx].values.astype(float)
        emb_raw = emb_df.loc[idx].values.astype(float)
        print(f"  panel b — {label}: N={len(idx):,} events={int(e.sum())}")

        # Fair per-endpoint penalties: same deterministic selectors / grid as panel a
        # (symmetric-tuned), so the high-dim embedding and low-dim tabular block each get
        # their CV-selected penalty and the tabular baseline is not handicapped.
        if FIG3_FIXED_PEN is not None:
            fair_emb = fair_dxa = float(FIG3_FIXED_PEN)   # shared equal penalty on both blocks
        else:
            _pens0 = {'cov': PEN_COV, 'emb': PEN_EMB, 'dxa': PEN_DXA, 'regionpool': PEN_EMB}
            fair_emb = _crc.select_emb_penalizer(cov_df, emb_df, target, _pens0, SWEEP_GRID, random_seed=42)
            fair_dxa = _crc.select_dxa_penalizer(cov_df, dxa_df, target, _pens0, SWEEP_GRID, random_seed=42)
        print(f"    panel-b pens — emb={fair_emb}, dxa={fair_dxa}")

        # DeepDXA arm = bone-pool fusion (regional already merged); tabular = DXA scalars.
        blocks_fm  = [(cov_raw, False, 0, PEN_COV), (emb_raw, True, 100, fair_emb)]
        if reg_df is not None:
            reg_raw = reg_df.loc[idx].values.astype(float)
            blocks_fm.append((reg_raw, True, 100, fair_emb))
        blocks_tab = [(cov_raw, False, 0, PEN_COV), (dxa_raw, False, 0,  fair_dxa)]
        risk_fm  = cox_oof_joint(blocks_fm,  t, e)
        risk_tab = cox_oof_joint(blocks_tab, t, e)

        tm_fm,  cq4_fm,  _, _, _, tmax_fm,  _ = cumulative_capture_curve(risk_fm,  t, e)
        tm_tab, cq4_tab, _, _, _, tmax_tab, _ = cumulative_capture_curve(risk_tab, t, e)
        tmax = min(tmax_fm, tmax_tab)

        # Q4 (top-risk quartile) only — Q1 dropped, wasn't adding visual information
        ax.plot(tm_fm,  cq4_fm,  color=COL_DXAFM, lw=1.2, ls="-",  drawstyle="steps-post")
        ax.plot(tm_tab, cq4_tab, color=COL_TAB,   lw=1.2, ls="-",  drawstyle="steps-post")

        ax.set_xlim(0, tmax)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

        # Paired subject-level bootstrap of the Capture@Q4 difference at each cutoff.
        boot = bootstrap_capture_diff(risk_fm, risk_tab, t, e)
        pending.append((ax, label, boot))

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Years from baseline" if is_bottom else "", fontsize=10)
        ax.set_ylabel("Cumulative incident cases captured (%)" if is_left else "", fontsize=10)
        ax.tick_params(labelsize=8)
        if not is_left:
            ax.tick_params(axis="y", labelleft=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Pass 2: BH-FDR across the endpoints at each cutoff, then annotate.
    for tc in CAPTURE_TIMEPOINTS_YR:
        qs = _bh_fdr([boot[tc]["p"] for _, _, boot in pending])
        for (_, _, boot), q in zip(pending, qs):
            boot[tc]["q"] = float(q) if np.isfinite(q) else np.nan

    for ax, label, boot in pending:
        for tc in CAPTURE_TIMEPOINTS_YR:
            r = boot[tc]
            lbl = f"Y{tc}" if tc is not None else "End"
            print(f"    {label} {lbl}: Δ={r['diff']*100:+.1f}pp "
                  f"[{r['lo']*100:+.1f},{r['hi']*100:+.1f}] "
                  f"p={r['p']:.4f} q={r.get('q', np.nan):.4f} (events={r['n_events']})")
        # Header names the metric: the X/Y numbers are the sensitivity (recall) of
        # the top-25%-risk rule at each horizon — DeepDXA / Tabular.
        header = "Sensitivity, top-25%\n(DeepDXA/Tab)"
        ann = "\n".join([header] +
                        [_annot_row(f"Y{tc}" if tc is not None else "End", boot[tc])
                         for tc in CAPTURE_TIMEPOINTS_YR])
        # Lower-right: these cumulative curves are monotonic, so the space below
        # them at late follow-up is always empty — the box never occludes a curve
        # (top-left would, now that the header makes the box taller).
        ax.text(0.97, 0.03, ann,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#333333", family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc",
                          lw=0.4, alpha=1.0))


def main():
    print("Loading main summary + cohort data ...")
    main_df = pd.read_csv(MAIN_CSV)

    # ── Supplementary table first: only needs main_df, cheap, fails fast before
    # the heavier panel-b cohort/embedding loading ──
    export_supp_table(main_df, SUPP_TABLE_PATH)

    events = pd.read_csv(EVENTS_PATH, low_memory=False, index_col="eid")
    events.index = events.index.astype(int)
    emb_df = _load_embeddings(LEJEPA_PATH, "DeepDXA")   # bone-pool enriched fusion
    reg_df = _load_embeddings(REGION_PATH, "DeepDXA-Regional") if REGION_PATH else None
    tabular = load_clean_data(TABULAR_PATH, "Tabular")
    cov_df, dxa_df = split_tabular_features(tabular)
    common = (events[events[BASELINE_COL].notna()].index
              .intersection(cov_df.dropna().index)
              .intersection(dxa_df[dxa_df.notna().mean(axis=1) > 0.5].index)
              .intersection(emb_df.index))
    if reg_df is not None:
        common = common.intersection(reg_df.index)
    print(f"  cohort: {len(common):,}")

    _style = os.path.join(os.path.expanduser('~'), '.claude/skills/nature-plot-style/style_files/nature_double.mplstyle')
    plt.style.use(_style)
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    n_b = len(PANEL_B_ENDPOINTS)
    n_cols_b = 2 if n_b > 3 else n_b
    n_rows_b = (n_b + n_cols_b - 1) // n_cols_b
    fig = plt.figure(figsize=(18 / 2.54, 10.5))
    gs = GridSpec(1 + n_rows_b, 1,
                  height_ratios=[2.15] + [1.6] * n_rows_b, hspace=1.10,
                  left=0.10, right=0.98, bottom=0.12, top=0.92)
    ax_a = fig.add_subplot(gs[0])
    # 4-col scaffold so each panel spans 2 cols; a lone last panel (odd count) is
    # centered in its row (cols 1:3) instead of left-aligned with an empty cell.
    gs_b = gs[1:].subgridspec(n_rows_b, n_cols_b * 2, wspace=0.34, hspace=0.70)
    axes_b = []
    for i in range(n_b):
        r = i // n_cols_b
        if i == n_b - 1 and (n_b % n_cols_b) != 0:
            ax = fig.add_subplot(gs_b[r, 1:3])           # centered lone panel
        else:
            c = (i % n_cols_b) * 2
            ax = fig.add_subplot(gs_b[r, c:c + 2])
        axes_b.append(ax)

    draw_panel_a(ax_a, main_df)
    draw_panel_b(axes_b, events, cov_df, dxa_df, emb_df, common, reg_df=reg_df)

    # panel letters: same figure-x (left of left panel edge), each just above its panel
    fig.canvas.draw()   # force layout so get_position() is accurate
    bb_a  = ax_a.get_position()
    bb_b0 = axes_b[0].get_position()
    lx = min(bb_a.x0, bb_b0.x0) - 0.030   # shared left x
    fig.text(lx, bb_a.y1  + 0.008, "a", fontsize=12, fontweight="bold",
             va="bottom", ha="right")
    fig.text(lx, bb_b0.y1 + 0.008, "b", fontsize=12, fontweight="bold",
             va="bottom", ha="right")

    # arm legend: inside panel a, pinned to the very top-right corner
    arm_leg = [Patch(fc=c, label=d) for _, d, c in ARMS]
    ax_a.legend(handles=arm_leg, loc="lower right", ncol=2, frameon=False,
                fontsize=9, borderpad=0.5,
                bbox_to_anchor=(1.0, 1.13))

    # panel b legend: top-risk-quartile (Q4) capture curves only
    b_leg = [
        Line2D([0],[0], color=COL_DXAFM, lw=1.2, ls="-",  label="LeDXA + cov"),
        Line2D([0],[0], color=COL_TAB,   lw=1.2, ls="-",  label="DXA Tabular + cov"),
    ]
    fig.legend(handles=b_leg, ncol=2, frameon=False, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.54, 0.03))

    fig.savefig(OUT_PATH + ".png", dpi=400, facecolor='white', transparent=False)
    fig.savefig(OUT_PATH + ".pdf", dpi=400, facecolor='white', transparent=False)
    print(f"Saved {OUT_PATH}.png / .pdf")


if __name__ == "__main__":
    main()
