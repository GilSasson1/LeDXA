"""
topset.py — two-tailed "statistically-tied top set" per disease.

For Fig 2 (Virchow Fig 2a idiom): for each disease, the best-AUROC model and any
model NOT significantly different from it (two-tailed paired Wilcoxon across seeds,
BH-FDR within cohort) form the 'top set'. The figure colors the top set identically
and boxes a lone significant winner; non-top models are greyed.

Two-tailed is required: it correctly flags a competitor that is significantly BETTER
(e.g. DINOv3 > DeepDXA on COPD), which a one-tailed DeepDXA>X test would mask.

Inputs : HPP  raw  = {RESULTS_DIR}/lp_cov_disease_4arm_raw.csv  (per-seed 'score')
         UKBB seeds = tables/ukbb_disease_4arm_seeds.csv        (per-seed 'auc')
         Neither raw per-seed file is distributed in this repo (participant-level
         provenance); point --hpp-raw/--ukbb-seeds at your own reproduction of them.
Output : tables/disease_pairwise_diffpentuned.csv and tables/disease_top_set.csv
         (cohort, key, raw_key, arm, mean, in_top_set, sole_winner, argmax) — named
         to match the canonical inputs plotting/fig2_heatmap.py reads by default.
"""
import os, sys, json, argparse, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from common.plot_style import _dis_label, _DISEASE_NAMES_JSON, RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
HPP_RAW_DEFAULT = os.path.join(RESULTS_DIR, 'lp_cov_disease_4arm_raw.csv')
UKBB_SEEDS = os.path.join(REPO_ROOT, 'tables', 'ukbb_disease_4arm_seeds.csv')
OUT        = os.path.join(REPO_ROOT, 'tables', 'disease_top_set.csv')
OUT_PAIRS  = os.path.join(REPO_ROOT, 'tables', 'disease_pairwise_diffpentuned.csv')

ARM_MAP = {'lejepa_cov': 'lejepa', 'dino_cov': 'dino', 'tab_cov': 'tabular', 'covariates': 'covariates'}
ARMS = ['lejepa', 'dino', 'tabular', 'covariates']
ALPHA = 0.05
_display_names = json.load(open(_DISEASE_NAMES_JSON)) if os.path.exists(_DISEASE_NAMES_JSON) else {}


def _bh(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    o = np.empty(n); o[order] = adj
    return np.minimum(o, 1.0)


def _two_tailed(a, b):
    """Paired two-tailed Wilcoxon across shared seeds (b vs a)."""
    shared = sorted(set(a.index) & set(b.index))
    if len(shared) < 2:
        return np.nan
    diff = b.reindex(shared).values - a.reindex(shared).values
    if np.all(diff == 0):
        return 1.0
    _, p = wilcoxon(diff, alternative='two-sided')
    return float(p)


def compute(raw, key_col, score_col, name_fn, cohort):
    """Returns (topset_rows, pairwise_rows).

    All 6 model pairs are tested two-tailed (paired Wilcoxon across seeds), BH-FDR
    across every (disease × pair) in the cohort. The 'top set' for coloring is the
    best model plus any model NOT significantly different from it (i.e. tied)."""
    diseases = sorted(raw[key_col].unique())
    means, ser_by_d = {}, {}
    pair_recs = []  # (disease, a, b, p_raw)  for ALL pairs, a<b in ARMS order
    for d in diseases:
        sub = raw[raw[key_col] == d]
        ser = {}
        for m_raw, m in ARM_MAP.items():
            s = sub[sub['model'] == m_raw].set_index('seed')[score_col].dropna()
            if len(s):
                ser[m] = s
        if 'lejepa' not in ser or len(ser) < 2:
            continue
        means[d] = {m: float(s.mean()) for m, s in ser.items()}
        ser_by_d[d] = ser
        for a, b in itertools.combinations([m for m in ARMS if m in ser], 2):
            pair_recs.append((d, a, b, _two_tailed(ser[a], ser[b])))

    ps = np.array([r[3] for r in pair_recs], float)
    finite = np.isfinite(ps)
    adj = np.full(len(ps), np.nan)
    if finite.sum() > 0:
        adj[finite] = _bh(ps[finite])
    padj = {(d, a, b): pa for (d, a, b, _), pa in zip(pair_recs, adj)}

    def _p(d, x, y):  # symmetric lookup
        return padj.get((d, x, y), padj.get((d, y, x), np.nan))

    topset_rows, pair_rows = [], []
    for d, mu in means.items():
        best = max(mu, key=mu.get)
        top = {best}
        for m in mu:
            if m == best:
                continue
            pa = _p(d, best, m)
            if not np.isfinite(pa) or pa >= ALPHA:   # not significantly different → tied
                top.add(m)
        sole = len(top) == 1
        # Significance TIERS: greedy from the top — each tier's leader (highest mean
        # among the remaining) plus every remaining model NOT significantly different
        # from it share a tier; significantly-worse models drop to the next tier.
        tier_of = {}
        remaining = sorted(mu, key=lambda m: -mu[m])
        tier = 1
        while remaining:
            leader = remaining[0]
            group = [leader] + [m for m in remaining[1:]
                                if (not np.isfinite(_p(d, leader, m))) or _p(d, leader, m) >= ALPHA]
            for m in group:
                tier_of[m] = tier
            remaining = [m for m in remaining if m not in group]
            tier += 1
        for m, v in mu.items():
            topset_rows.append({'cohort': cohort, 'key': name_fn(d), 'raw_key': d, 'arm': m,
                                'mean': round(v, 4), 'in_top_set': m in top, 'tier': tier_of[m],
                                'sole_winner': bool(sole and m == best), 'argmax': m == best})
        for a, b in itertools.combinations([m for m in ARMS if m in mu], 2):
            pair_rows.append({'cohort': cohort, 'key': name_fn(d), 'raw_key': d,
                              'model_a': a, 'model_b': b,
                              'mean_a': round(mu[a], 4), 'mean_b': round(mu[b], 4),
                              'p_two_tailed_adj': round(float(_p(d, a, b)), 5)
                              if np.isfinite(_p(d, a, b)) else np.nan})
    return topset_rows, pair_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hpp-raw', default=HPP_RAW_DEFAULT,
                    help='HPP per-seed disease raw (target/model/seed/score)')
    ap.add_argument('--out-pairs', default=OUT_PAIRS, help='output pairwise CSV')
    ap.add_argument('--out', default=OUT, help='output top-set CSV')
    ap.add_argument('--ukbb-seeds', default=UKBB_SEEDS, help='UKBB per-seed disease raw')
    args = ap.parse_args()

    top_rows, pair_rows = [], []
    if os.path.exists(args.hpp_raw):
        hpp = pd.read_csv(args.hpp_raw, encoding='latin-1')
        hpp = hpp[hpp['target'].astype(str).str.startswith('dis__')]
        if 'metric' in hpp.columns:               # disease rows are AUC; drop any pearson leftovers
            hpp = hpp[hpp['metric'] == 'auc']
        t, p = compute(hpp, 'target', 'score', lambda d: _dis_label(d, _display_names), 'HPP')
        top_rows += t; pair_rows += p
        print(f"HPP diseases: {len({r['key'] for r in t})}  (raw={args.hpp_raw})")
    else:
        print(f"[warn] missing {args.hpp_raw}")
    if os.path.exists(args.ukbb_seeds):
        ukbb = pd.read_csv(args.ukbb_seeds)
        t, p = compute(ukbb, 'disease', 'auc', lambda d: d, 'UKBB')
        top_rows += t; pair_rows += p
        print(f"UKBB diseases: {len({r['key'] for r in t})}  (seeds={args.ukbb_seeds})")
    else:
        print(f"[warn] missing {args.ukbb_seeds}")

    df = pd.DataFrame(top_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.DataFrame(pair_rows).to_csv(args.out_pairs, index=False)
    print(f"Saved → {args.out}  ({len(df)} rows) and {args.out_pairs} ({len(pair_rows)} pairs)")

    for cohort, k in [('UKBB', 'dis__copd'), ('UKBB', 'dis__gout'),
                      ('UKBB', 'dis__diabetes'), ('HPP', 'Hypertension')]:
        sub = df[(df.cohort == cohort) & (df.key == k)]
        if len(sub):
            print(f"  {cohort} {k}: top_set={sub[sub.in_top_set].arm.tolist()}  "
                  f"sole={bool(sub.sole_winner.any())}")


if __name__ == '__main__':
    main()
