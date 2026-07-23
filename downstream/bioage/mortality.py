"""UKBB cross-cohort validation: does DXA-FM age-residual (biological-age gap)
predict all-cause mortality?

Loads the existing UKBB age-prediction residuals (visit 2, DXA imaging
baseline), builds time-to-event from existing aligned event file, fits a Cox
PH model with quartile-binned residuals + age + sex covariates, and reports
HR per quartile + KM curves.

This is independent of Fig. 2's all-cause-mortality Cox panel: that uses the
full DXA-FM embedding as predictor; this uses only the chronological-age-
orthogonal residual, asking whether the age-orthogonal signal is itself
prognostic.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import pearsonr
from config import DATA_ROOT, RESULTS_DIR

PRED_CSV = os.environ.get(
    'BIOAGE_PRED_CSV', str(DATA_ROOT / 'ukbb' / 'age_predictions_with_visits.csv'))
SUFFIX     = os.environ.get('BIOAGE_SUFFIX', '')
EVENTS_CSV = str(DATA_ROOT / 'ukbb' / 'incident_events.csv')
TABULAR_CSV = str(DATA_ROOT / 'ukbb' / 'dxa_tabular.csv')
OUTPUT_DIR = str(RESULTS_DIR / 'bioage')

BASELINE_VISIT = 2
N_QUANTILES    = 4
DEATH_COL      = 'Date of death - visit 0'
BASELINE_COL   = f'Date of attending assessment centre - visit {BASELINE_VISIT}'

def main():
    print("Loading UKBB DXA-FM age predictions...")
    pred = pd.read_csv(PRED_CSV)
    pred = pred[pred['visit'] == BASELINE_VISIT].drop_duplicates('eid').set_index('eid')
    print(f"  Visit-{BASELINE_VISIT} predictions: {len(pred)} subjects, age {pred['age_true'].min():.0f}–{pred['age_true'].max():.0f} (mean {pred['age_true'].mean():.1f})")
    r, _ = pearsonr(pred['age_true'], pred['age_pred_lejepa'])
    print(f"  Age prediction r = {r:.3f}, MAE = {(pred['age_true'] - pred['age_pred_lejepa']).abs().mean():.2f} yr")

    # Apply same RTM-style detrending as HPP: regress age_pred on
    # polynomial(age_true) → take residuals. The CSV already has age_residual
    # but it's just (pred - true); we replace with polynomial-detrended residual
    # so the Q-bins align with the HPP "biological-age gap" definition.
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    Xage = PolynomialFeatures(degree=2).fit_transform(pred[['age_true']])
    expected = LinearRegression().fit(Xage, pred['age_pred_lejepa']).predict(Xage)
    pred['bioage_gap'] = pred['age_pred_lejepa'] - expected
    print(f"  Polynomial-detrended residual: mean={pred['bioage_gap'].mean():+.3f}, sd={pred['bioage_gap'].std():.2f}")

    # Quartile-bin
    pred['Q'] = pd.qcut(pred['bioage_gap'].rank(method='first'), q=N_QUANTILES,
                        labels=[f'Q{i+1}' for i in range(N_QUANTILES)])
    print(f"  Quartile sizes:", pred['Q'].value_counts().sort_index().to_dict())

    print("\nLoading UKBB event table...")
    ev = pd.read_csv(EVENTS_CSV, index_col=0, low_memory=False)
    ev = ev[ev.index.notna() & ~ev.index.isin(['error', 'skipped'])]
    ev.index = ev.index.astype(int)
    needed = [BASELINE_COL, DEATH_COL]
    ev = ev[needed].copy()
    print(f"  Event table: {len(ev)} subjects")

    # Pull sex from tabular
    print("Loading sex covariate from tabular...")
    tab = pd.read_csv(TABULAR_CSV, index_col=0, low_memory=False, usecols=lambda c: c == 'eid' or 'sex' in c.lower() or 'gender' in c.lower())
    tab = tab[tab.index.notna() & ~tab.index.isin(['error', 'skipped'])]
    tab.index = tab.index.astype(int)
    sex_cands = [c for c in tab.columns if 'sex' in c.lower() or 'gender' in c.lower()]
    # Prefer non-visit-suffixed sex column (genetic sex is constant)
    sex_col = next((c for c in sex_cands if 'visit' not in c.lower() and 'genetic' in c.lower()), sex_cands[0] if sex_cands else None)
    if sex_col is None:
        raise ValueError("No sex column found in tabular file")
    print(f"  Using sex column: {sex_col}")
    tab[sex_col] = pd.to_numeric(tab[sex_col].replace({'Female': 0, 'Male': 1, 'F': 0, 'M': 1}), errors='coerce')

    # Merge
    df = pred[['age_true', 'bioage_gap', 'Q']].join(ev, how='inner').join(tab[[sex_col]].rename(columns={sex_col: 'sex'}), how='inner')
    df = df.dropna(subset=['age_true', 'bioage_gap', 'sex', BASELINE_COL])
    print(f"\nMerged cohort: {len(df)} subjects")

    # Build time-to-event
    start = pd.to_datetime(df[BASELINE_COL], errors='coerce')
    death = pd.to_datetime(df[DEATH_COL], errors='coerce')
    # Drop subjects with prevalent death (death before baseline DXA — none expected)
    prevalent = death.notna() & (death <= start)
    df = df.loc[~prevalent]; start = start.loc[~prevalent]; death = death.loc[~prevalent]

    event = death.notna() & (death > start)
    admin_date = death[event].max()
    print(f"  Admin censoring date: {admin_date.date()}, total deaths: {int(event.sum())}")
    censor_date = pd.Series(pd.Timestamp(admin_date), index=df.index)
    end_date = death.where(event, censor_date)
    df['time_yr'] = (end_date - start).dt.days / 365.25
    df['event']   = event.astype(int)
    df = df[df['time_yr'] > 0].copy()
    print(f"  After time>0 filter: {len(df)} subjects, {int(df['event'].sum())} deaths, "
          f"median follow-up {df['time_yr'].median():.2f} yr")

    # ─── Cox: Q1 reference, dummy-encoded ───────────────────────────────────
    print("\n=== Cox PH: time-to-death ~ Q-bin + age + sex (Q1 reference) ===")
    cox_df = df[['time_yr', 'event', 'age_true', 'sex', 'Q']].copy()
    for q in ['Q2', 'Q3', 'Q4']:
        cox_df[q] = (cox_df['Q'] == q).astype(int)
    cox_df = cox_df.drop(columns='Q')
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='time_yr', event_col='event')
    print(cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).to_string())
    summary_out = cph.summary.copy()
    summary_out.to_csv(os.path.join(OUTPUT_DIR, f'ukbb_bioage_mortality_cox{SUFFIX}.csv'))

    # Trend test: continuous bioage_gap (per 1 yr of biological-age gap)
    cox_cont = df[['time_yr', 'event', 'age_true', 'sex', 'bioage_gap']].copy()
    cph2 = CoxPHFitter()
    cph2.fit(cox_cont, duration_col='time_yr', event_col='event')
    print("\n=== Cox PH: continuous bioage_gap + age + sex ===")
    print(cph2.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].round(4).to_string())

    # ─── KM by quartile ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5))
    PALETTE = {'Q1': '#2471a3', 'Q2': '#76b7c8', 'Q3': '#f0a07a', 'Q4': '#c0392b'}
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = df['Q'] == q
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, 'time_yr'], df.loc[mask, 'event'], label=f'{q} (n={mask.sum()}, d={int(df.loc[mask, "event"].sum())})')
        kmf.plot_cumulative_density(ax=ax, ci_show=False, color=PALETTE[q], lw=2)
    ax.set_xlabel('Years from DXA baseline')
    ax.set_ylabel('Cumulative all-cause mortality')
    ax.set_title('UKBB: bio-age gap quartiles → all-cause mortality')
    ax.legend(frameon=False, fontsize=10, loc='upper left')
    ax.grid(alpha=0.2, linestyle='--')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'ukbb_bioage_mortality_km{SUFFIX}.pdf'))
    fig.savefig(os.path.join(OUTPUT_DIR, f'ukbb_bioage_mortality_km{SUFFIX}.png'), dpi=200)
    plt.close()
    print(f"\nSaved → ukbb_bioage_mortality_cox{SUFFIX}.csv + ukbb_bioage_mortality_km{SUFFIX}.{{pdf,png}} in {OUTPUT_DIR}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pred-csv', default=PRED_CSV)
    ap.add_argument('--events-csv', default=EVENTS_CSV)
    ap.add_argument('--tabular-csv', default=TABULAR_CSV)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--suffix', default='', help='Suffix for output files (avoid clobbering canonical).')
    a = ap.parse_args()
    PRED_CSV = a.pred_csv
    EVENTS_CSV = a.events_csv
    TABULAR_CSV = a.tabular_csv
    OUTPUT_DIR = a.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SUFFIX = a.suffix
    main()
