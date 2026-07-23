"""Fine-tuning comparison: LeDXA vs DINOv3.

Full end-to-end fine-tuning with cosine LR schedule and linear warmup.
Best checkpoint is saved per run; embeddings are extracted from best model.

Usage:
  python -m downstream.disease.finetune --targets age bmi
  python -m downstream.disease.finetune --fusions late concat
"""

import argparse
import os
import h5py
import pandas as pd

import common.utils as U
from config import RESULTS_DIR

_RESULTS_DIR = str(RESULTS_DIR)
_DEFAULT_SUMMARY_CSV = f"{_RESULTS_DIR}/ft_summary.csv"
_DEFAULT_RAW_CSV     = f"{_RESULTS_DIR}/ft_raw.csv"
_DEFAULT_TTEST_CSV   = f"{_RESULTS_DIR}/ft_ttest.csv"


def main():
    parser = argparse.ArgumentParser(description="FT comparison: LeJEPA vs DINOv3")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=["lejepa", "dino"], choices=["lejepa", "dino"])
    parser.add_argument("--fusions", nargs="+", default=["late"], choices=["late", "concat"])
    parser.add_argument("--num-seeds", type=int, default=U.DEFAULT_NUM_SEEDS,
                        help=f"How many seeds to use (picks first N from fixed pool, max {len(U._SEED_POOL)})")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seed value(s), overriding --num-seeds/the fixed-pool prefix "
                             "(e.g. --seeds 73 for a single fine-grained, checkpointable run)")
    parser.add_argument("--lejepa-checkpoint", default=U.LEJEPA_CHECKPOINT)
    parser.add_argument("--epochs-ft", type=int, default=U.NUM_EPOCHS_FT)
    parser.add_argument("--batch-size", type=int, default=U.BATCH_SIZE)
    parser.add_argument("--results-csv", default=_DEFAULT_SUMMARY_CSV)
    parser.add_argument("--results-raw-csv", default=_DEFAULT_RAW_CSV)
    parser.add_argument("--results-ttest-csv", default=_DEFAULT_TTEST_CSV)
    parser.add_argument("--targets-csv", default=None,
                        help="Override TARGETS_CSV (e.g. csvs/disease_targets_with_covs.csv)")
    parser.add_argument("--cls-auto-detect", action="store_true",
                        help="Auto-detect classification targets: columns with only {0,1} values")
    args  = parser.parse_args()
    for path in (args.results_csv, args.results_raw_csv, args.results_ttest_csv):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    seeds = args.seeds if args.seeds is not None else U.make_seeds(args.num_seeds)

    U.NUM_EPOCHS_FT = args.epochs_ft
    U.BATCH_SIZE = args.batch_size

    print(f"Scanning HDF5: {U.HDF5_PATH}")
    with h5py.File(U.HDF5_PATH, "r") as f:
        all_keys = list(f.keys())
    print(f"Total HDF5 keys: {len(all_keys)}")

    targets_csv = args.targets_csv or U.TARGETS_CSV
    target_df_full = pd.read_csv(targets_csv, index_col=[0, 1])
    target_df_full.sort_index(inplace=True)

    if args.cls_auto_detect:
        auto_cls = set()
        for c in target_df_full.columns:
            vals = set(target_df_full[c].dropna().unique())
            if vals.issubset({0.0, 1.0, 0, 1}):
                auto_cls.add(c)
        U.CLASSIFICATION_TARGETS = auto_cls
        print(f"Auto-detected {len(auto_cls)} classification targets")

    if args.targets:
        target_cols = args.targets
    elif U.TARGET_COLUMNS == "all":
        target_cols = [c for c in target_df_full.columns if pd.api.types.is_numeric_dtype(target_df_full[c])]
    else:
        target_cols = [c.strip() for c in U.TARGET_COLUMNS.split(",") if c.strip()]

    print(f"Device: {U.DEVICE}")
    print(f"Targets ({len(target_cols)}): {target_cols}")
    print(f"Models: {args.models} | Fusions: {args.fusions}")
    print(f"Num seeds: {len(seeds)} → {seeds} | Epochs FT: {U.NUM_EPOCHS_FT} | Batch: {U.BATCH_SIZE}")

    all_summary_rows, all_raw_rows = [], []
    for target_col in target_cols:
        try:
            summary_rows, raw_rows = U.run_one_target(
                all_keys, target_df_full, target_col,
                lejepa_ckpt=args.lejepa_checkpoint,
                models=args.models,
                fusions=args.fusions, seeds=seeds,
            )
            all_summary_rows.extend(summary_rows)
            all_raw_rows.extend(raw_rows)
        except Exception as e:
            print(f"[Error] {target_col}: {e}")

    U.save_results(all_summary_rows, all_raw_rows,
                   args.results_csv, args.results_raw_csv, args.results_ttest_csv)


if __name__ == "__main__":
    main()
