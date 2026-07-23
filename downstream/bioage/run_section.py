"""End-to-end orchestrator for the biological-age section.

Runs the section analyses in dependency order and then copies (with paper-style
renaming) every canonical CSV / figure into ``dexa_fm/tables/`` and
``dexa_fm/figures/``.

The existing analysis scripts are run as-is via subprocess so this orchestrator
introduces zero regression risk for the standalone workflows. The bioage module
is the single source of truth for downstream code (plot_fig5, audit_claims).

Phases
------
A. compute_canonical_gap   : predictions CSV → tableD_bioage_predictions_ukbb.csv
                              (uses bioage.rtm, the canonical poly-2 RTM).
B. ridge_refit             : refit Ridge on UKBB embeddings (--with-refit only).
C. aging_pace_phenotypes   : ukbb_aging_pace_v2v3.py (mortality + phenotypes +
                              disease prevalence + the legacy Figure 5).
D. sarcopenia              : sarcopenia_ukbb.py (Q1→Q4 EWGSOP2 gradient).
E. disease_extended        : extend_disease_panel.py (sarcopenia + HF + MI + CKD + hip OA).
F. medications_ukbb        : meds_bio_age_ukbb.py (UKBB HRT / OCP).
G. mortality               : dexa_fm/ukbb/ukbb_bioage_mortality.py.
H. publish                 : copy CSVs → dexa_fm/tables/, PDFs → dexa_fm/figures/
                              with paper naming (tableD_* / tableE_* / fig5_*).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from . import paths
from .rtm import detrend_gap, bin_quartiles


_HERE = os.path.dirname(__file__)
DEXA_ROOT = paths.DEXA_ROOT


# ── Canonical output naming map ───────────────────────────────────────────────
# Source file  →  paper-style destination filename in dexa_fm/tables/ or figures/
CSV_PUBLISH_MAP = {
    # ukbb_aging_pace_v2v3.py outputs (gap-only; pace analysis removed)
    'age_prediction_ukbb/ukbb_gap_mortality_cox.csv':            'tableE_bioage_gap_mortality_cox.csv',
    'age_prediction_ukbb/gap_female_phenotype_results.csv':      'tableD_bioage_phenotype_female.csv',
    'age_prediction_ukbb/gap_male_phenotype_results.csv':        'tableD_bioage_phenotype_male.csv',
    'age_prediction_ukbb/gap_disease_prevalence_results.csv':    'tableD_bioage_disease_prevalence.csv',
    # sarcopenia
    'age_prediction_ukbb/sarcopenia_prevalence_by_gap_quartile.csv': 'tableD_bioage_sarcopenia_by_quartile.csv',
    'age_prediction_ukbb/sarcopenia_q4_vs_q1.csv':                   'tableD_bioage_sarcopenia_q4_vs_q1.csv',
    'age_prediction_ukbb/sarcopenia_logistic_on_gap.csv':            'tableD_bioage_sarcopenia_logistic.csv',
    # extended disease panel
    'age_prediction_ukbb/gap_disease_prevalence_results_extended.csv': 'tableD_bioage_disease_prevalence_extended.csv',
    # mortality (dedicated script)
    'age_prediction_ukbb/ukbb_bioage_mortality_cox.csv':         'tableE_bioage_mortality_cox_dedicated.csv',
    # medication HPP outputs (canonical for Fig 5h)
    'tables/age_prediction_analysis/medications_atc3_paired_before_after.csv':       'tableD_bioage_medication_paired_atc3.csv',
    'tables/age_prediction_analysis/medications_significant_hits_phenotype_changes_by_sex.csv': 'tableD_bioage_medication_hits_by_sex.csv',
    'tables/age_prediction_analysis/medications_crosssectional_users_vs_nonusers.csv': 'tableD_bioage_medication_users_vs_nonusers.csv',
    'tables/age_prediction_analysis/drug_bodycomp_G03F_female.csv':                    'tableD_bioage_medication_G03F_female.csv',
    'tables/age_prediction_analysis/drug_bodycomp_N06A_female.csv':                    'tableD_bioage_medication_N06A_female.csv',
    'tables/age_prediction_analysis/drug_bodycomp_N06A_male.csv':                      'tableD_bioage_medication_N06A_male.csv',
    'tables/age_prediction_analysis/drug_bodycomp_pcorr_G03F_female.csv':              'tableD_bioage_medication_pcorr_G03F_female.csv',
    'tables/age_prediction_analysis/drug_bodycomp_pcorr_N06A_female.csv':              'tableD_bioage_medication_pcorr_N06A_female.csv',
    'tables/age_prediction_analysis/drug_bodycomp_pcorr_N06A_male.csv':                'tableD_bioage_medication_pcorr_N06A_male.csv',
    # HPP age predictions (Fig 5a)
    'tables/age_prediction_analysis/age_prediction_rate_partitions.csv':               'tableD_bioage_predictions_hpp.csv',
}

# Supplementary figures from existing scripts; the *main* fig5 is built by plot_fig5.py.
FIG_PUBLISH_MAP = {
    'age_prediction_ukbb/figure5_supp_sarcopenia.pdf':            'supp_fig5_sarcopenia.pdf',
    'age_prediction_ukbb/figure5_supp_sarcopenia.png':            'supp_fig5_sarcopenia.png',
    'age_prediction_ukbb/gap_disease_prevalence_bars_extended.pdf': 'supp_fig5_disease_panel_extended.pdf',
    'age_prediction_ukbb/gap_disease_prevalence_bars_extended.png': 'supp_fig5_disease_panel_extended.png',
    'age_prediction_ukbb/ukbb_bioage_mortality_km.pdf':           'supp_fig5_mortality_km_dedicated.pdf',
}


# ── Phase A: canonical gap table ─────────────────────────────────────────────

def compute_canonical_gap() -> pd.DataFrame:
    """Recompute the canonical gap from the predictions CSV using bioage.rtm."""
    print('\n=== A. Compute canonical bioage gap ===')
    if not os.path.exists(paths.PRED_CSV):
        raise FileNotFoundError(f'Missing predictions: {paths.PRED_CSV}')
    df = pd.read_csv(paths.PRED_CSV)
    df['eid']   = df['eid'].astype(int)
    df['visit'] = df['visit'].astype(str)

    df['bioage_gap'] = detrend_gap(df, age_col='age_true', pred_col='age_pred_lejepa', degree=2)

    v2 = df[df['visit'] == '2'].copy()
    v3 = df[df['visit'] == '3'].copy()
    r2, _ = pearsonr(v2['age_true'], v2['age_pred_lejepa'])
    mae2 = mean_absolute_error(v2['age_true'], v2['age_pred_lejepa'])
    print(f'  Visit-2: n={len(v2):,}, r={r2:.4f}, MAE={mae2:.2f} yr')
    if len(v3):
        r3, _ = pearsonr(v3['age_true'], v3['age_pred_lejepa'])
        mae3 = mean_absolute_error(v3['age_true'], v3['age_pred_lejepa'])
        print(f'  Visit-3: n={len(v3):,}, r={r3:.4f}, MAE={mae3:.2f} yr')

    v2['gap_quartile'] = bin_quartiles(v2['bioage_gap'])
    out = v2[['eid', 'visit', 'age_true', 'age_pred_lejepa', 'bioage_gap', 'gap_quartile']]
    dest = paths.out_table('tableD_bioage_predictions_ukbb.csv')
    out.to_csv(dest, index=False)
    print(f'  → {dest}  ({len(out):,} rows)')

    # Also write a tiny summary table that the audit will consume.
    summary = pd.DataFrame([
        dict(metric='ukbb_v2_pearson_r',  value=float(r2),  n=int(len(v2))),
        dict(metric='ukbb_v2_mae_yr',     value=float(mae2), n=int(len(v2))),
        dict(metric='ukbb_v2_n_subjects', value=float(len(v2)), n=int(len(v2))),
    ])
    if len(v3):
        summary = pd.concat([summary, pd.DataFrame([
            dict(metric='ukbb_v3_pearson_r',  value=float(r3),  n=int(len(v3))),
            dict(metric='ukbb_v3_mae_yr',     value=float(mae3), n=int(len(v3))),
        ])], ignore_index=True)
    summary_path = paths.out_table('tableD_bioage_prediction_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f'  → {summary_path}')
    return df


# ── Phase B–G: run existing scripts in dependency order ──────────────────────

def _run_script(name: str, script_path: str, *, env_extra=None) -> bool:
    print(f'\n=== {name} ===')
    print(f'  $ python {script_path}')
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            cwd=DEXA_ROOT, env=env, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
    except Exception as exc:
        print(f'  failed to launch: {exc}')
        return False
    elapsed = time.time() - t0
    tail = '\n'.join(r.stdout.splitlines()[-20:])
    print(f'--- last 20 lines (exit={r.returncode}, {elapsed:.1f}s) ---')
    print(tail)
    return r.returncode == 0


def run_ridge_refit() -> bool:
    return _run_script(
        'B. Refit Ridge age model on UKBB embeddings',
        os.path.join(DEXA_ROOT, 'ukbb_biological_age.py'),
    )


def run_aging_pace() -> bool:
    return _run_script(
        'C. UKBB bio-age gap: phenotypes + disease + mortality',
        os.path.join(DEXA_ROOT, 'ukbb_aging_pace_v2v3.py'),
    )


def run_sarcopenia() -> bool:
    return _run_script(
        'D. UKBB sarcopenia by gap quartile',
        os.path.join(DEXA_ROOT, 'sarcopenia_ukbb.py'),
    )


def run_disease_extended() -> bool:
    return _run_script(
        'E. Extended disease panel (sarcopenia + HF + MI + CKD + hip OA)',
        os.path.join(DEXA_ROOT, 'extend_disease_panel.py'),
    )


def run_medications_ukbb() -> bool:
    return _run_script(
        'F. UKBB medications (HRT / OCP)',
        os.path.join(DEXA_ROOT, 'meds_bio_age_ukbb.py'),
    )


def run_mortality_dedicated() -> bool:
    return _run_script(
        'G. Dedicated bioage-gap mortality Cox',
        os.path.join(DEXA_ROOT, 'dexa_fm', 'ukbb', 'ukbb_bioage_mortality.py'),
    )


# ── Phase H: publish to dexa_fm/{tables,figures}/ ────────────────────────────

def publish(*, force: bool = False) -> dict:
    """Copy canonical outputs into paper dirs with paper-style names."""
    print('\n=== H. Publish to dexa_fm/{tables,figures}/ ===')
    report = {'copied': [], 'missing': [], 'skipped': []}
    for src_rel, dst_name in CSV_PUBLISH_MAP.items():
        src = os.path.join(DEXA_ROOT, src_rel)
        dst = paths.out_table(dst_name)
        if not os.path.exists(src):
            report['missing'].append(src_rel)
            print(f'  [missing] {src_rel}')
            continue
        if (not force) and os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            report['skipped'].append(dst_name)
            continue
        shutil.copy2(src, dst)
        report['copied'].append(dst_name)
        print(f'  CSV  {src_rel}  →  tables/{dst_name}')
    for src_rel, dst_name in FIG_PUBLISH_MAP.items():
        src = os.path.join(DEXA_ROOT, src_rel)
        dst = paths.out_figure(dst_name)
        if not os.path.exists(src):
            report['missing'].append(src_rel)
            print(f'  [missing] {src_rel}')
            continue
        if (not force) and os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            report['skipped'].append(dst_name)
            continue
        shutil.copy2(src, dst)
        report['copied'].append(dst_name)
        print(f'  FIG  {src_rel}  →  figures/{dst_name}')
    print(f'\n  Copied: {len(report["copied"])} | Skipped (up-to-date): {len(report["skipped"])} | Missing: {len(report["missing"])}')
    if report['missing']:
        print('  Missing inputs (re-run earlier phases or check paths):')
        for m in report['missing']:
            print(f'    · {m}')
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-refit', action='store_true',
                    help='Refit Ridge age model (requires UKBB embeddings pickle).')
    ap.add_argument('--skip-aging-pace', action='store_true')
    ap.add_argument('--skip-sarcopenia', action='store_true')
    ap.add_argument('--skip-disease-extended', action='store_true')
    ap.add_argument('--skip-meds', action='store_true')
    ap.add_argument('--skip-mortality', action='store_true')
    ap.add_argument('--publish-only', action='store_true',
                    help='Skip analysis re-runs; only re-publish existing outputs.')
    ap.add_argument('--force-publish', action='store_true',
                    help='Overwrite paper-dir outputs even if newer than source.')
    args = ap.parse_args(argv)

    print(f'DEXA_ROOT       = {DEXA_ROOT}')
    print(f'tables_dir      = {paths.TABLES_DIR}')
    print(f'figures_dir     = {paths.FIGURES_DIR}')

    if not args.publish_only:
        # A. canonical gap (always runs; fast, deterministic)
        compute_canonical_gap()

        if args.with_refit:
            run_ridge_refit()

        if not args.skip_aging_pace:
            run_aging_pace()
        if not args.skip_sarcopenia:
            run_sarcopenia()
        if not args.skip_disease_extended:
            run_disease_extended()
        if not args.skip_meds:
            run_medications_ukbb()
        if not args.skip_mortality:
            run_mortality_dedicated()

    publish(force=args.force_publish)
    print('\nOrchestrator done.')


if __name__ == '__main__':
    main()
