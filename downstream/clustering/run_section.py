"""End-to-end orchestrator for the female-clustering section.

Runs characterize_female_clusters.py (females, --gender 0) as a subprocess,
then publishes canonical outputs to dexa_fm/tables/ and dexa_fm/figures/.

CLI
---
python -m dexa_fm.hpp.clustering.run_section [--skip-rerun] [--publish-only] [--force-publish]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

from . import paths

DEXA_ROOT = paths.DEXA_ROOT


def _run_analysis() -> bool:
    script = os.path.join(DEXA_ROOT, 'characterize_female_clusters.py')
    print(f'\n=== Running female clustering analysis ===')
    print(f'  $ python {script} --gender 0')
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, script, '--gender', '0'],
        cwd=DEXA_ROOT, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - t0
    tail = '\n'.join(r.stdout.splitlines()[-25:])
    print(f'--- last 25 lines (exit={r.returncode}, {elapsed:.1f}s) ---')
    print(tail)
    return r.returncode == 0


def publish(*, force: bool = False) -> dict:
    """Copy canonical outputs into dexa_fm/{tables,figures}/."""
    print('\n=== Publishing to dexa_fm/{tables,figures}/ ===')
    report = {'copied': [], 'missing': [], 'skipped': []}

    for src_rel, dst_name in paths.CSV_PUBLISH_MAP.items():
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

    for src_rel, dst_name in paths.FIG_PUBLISH_MAP.items():
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

    print(f'\n  Copied: {len(report["copied"])} | '
          f'Skipped (up-to-date): {len(report["skipped"])} | '
          f'Missing: {len(report["missing"])}')
    if report['missing']:
        print('  Missing inputs — re-run without --publish-only:')
        for m in report['missing']:
            print(f'    · {m}')
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skip-rerun',    action='store_true',
                    help='Skip characterize_female_clusters.py; only publish existing outputs.')
    ap.add_argument('--publish-only',  action='store_true',
                    help='Alias for --skip-rerun.')
    ap.add_argument('--force-publish', action='store_true',
                    help='Overwrite paper-dir outputs even if newer than source.')
    args = ap.parse_args(argv)

    print(f'DEXA_ROOT   = {DEXA_ROOT}')
    print(f'tables_dir  = {paths.TABLES_DIR}')
    print(f'figures_dir = {paths.FIGURES_DIR}')

    if not (args.skip_rerun or args.publish_only):
        _run_analysis()

    publish(force=args.force_publish)
    print('\nOrchestrator done.')


if __name__ == '__main__':
    main()
