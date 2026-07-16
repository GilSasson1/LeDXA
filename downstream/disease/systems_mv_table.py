"""Build the Figure 2 systems biomarker display table from LP raw/summary CSVs."""
import argparse

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


TARGET_LABELS = {
    "hr_bpm": "Heart Rate",
    "bt__hdl_cholesterol": "HDL Cholesterol",
    "bt__glucose": "Glucose",
    "liver_attenuation": "Liver Fat",
    "bt__creatinine": "Creatinine",
    "bt__hemoglobin": "Hemoglobin",
    "ahi": "AHI",
    "bt__wbc": "WBC",
}

ORDER = [
    "hr_bpm",
    "bt__hdl_cholesterol",
    "bt__glucose",
    "liver_attenuation",
    "bt__creatinine",
    "bt__hemoglobin",
    "ahi",
    "bt__wbc",
]

MODELS = {
    "lejepa": "DeepDXA",
    "dino": "DINOv3",
    "tabular": "DXA Tabular",
    "covariates": "Covariates",
}

CONTRASTS = {
    "Cov": "covariates",
    "Tab": "tabular",
    "DINO": "dino",
}


def _padj_by_contrast(raw: pd.DataFrame) -> dict[str, dict[str, float]]:
    raw = raw[raw["metric"] == "pearson"]
    p_raw = {name: {} for name in CONTRASTS}
    for target in sorted(raw["target"].unique()):
        piv = raw[raw["target"] == target].pivot_table(
            index="seed", columns="model", values="score", aggfunc="mean"
        )
        for name, base in CONTRASTS.items():
            if "lejepa" not in piv.columns or base not in piv.columns:
                continue
            paired = piv[["lejepa", base]].dropna()
            if len(paired) < 2:
                continue
            diff = paired["lejepa"].values - paired[base].values
            if (diff == 0).all():
                p_raw[name][target] = 1.0
            else:
                _, p_raw[name][target] = wilcoxon(diff, alternative="two-sided")

    out = {}
    for name, by_target in p_raw.items():
        targets = list(by_target)
        if not targets:
            out[name] = {}
            continue
        qvals = multipletests([by_target[t] for t in targets], method="fdr_bh")[1]
        out[name] = dict(zip(targets, qvals))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    summary = summary[summary["metric"] == "pearson"].copy()
    raw = pd.read_csv(args.raw)
    padj = _padj_by_contrast(raw)

    rows = []
    for target in ORDER:
        sub = summary[summary["target"] == target]
        if sub.empty:
            print(f"[warn] missing target: {target}")
            continue
        row = {"Target": TARGET_LABELS[target]}
        for model, label in MODELS.items():
            m = sub[sub["model"] == model]
            row[f"{label}_mean"] = float(m["mean"].iloc[0]) if len(m) else float("nan")
            row[f"{label}_SE"] = float(m["se"].iloc[0]) if len(m) else float("nan")
        row["P_DeepDXA_vs_Cov_adj"] = padj.get("Cov", {}).get(target, float("nan"))
        row["P_DeepDXA_vs_Tab_adj"] = padj.get("Tab", {}).get(target, float("nan"))
        row["P_DeepDXA_vs_DINO_adj"] = padj.get("DINO", {}).get(target, float("nan"))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.round(4).to_string(index=False))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
