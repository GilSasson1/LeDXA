"""
Build main Figure 4 from the GWAS analysis outputs.

The source panel images were copied into:
  /data/embeddings/big_gil

The tabular inputs used by the source scripts live in:
  /data/gwas_analysis
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parent.parent  # repo root
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

BIG_GIL_DIR = Path("/data/embeddings/big_gil")
ANALYSIS_DIR = Path("/data/gwas_analysis")
EMBEDDING_DIR = Path("/data/embeddings_qc")

MANHATTAN_PATH = ROOT / "figures" / "fig_manhattan_ledxa.png"
ANNOTATED_HITS_PATH = ANALYSIS_DIR / "annotated_hits.tsv"
CATALOG_HITS_PATH = ANALYSIS_DIR / "gwas_catalog_hits.tsv"
HERITABILITY_PATH = ANALYSIS_DIR / "heritability.tsv"
GW_THRESHOLD = 5e-8
SUGGESTIVE_THRESHOLD = 1e-5

TAB_C = "#8ccbb3"
DINO_C = "#7fb9dc"
DEEPDXA_C = "#083c7d"
ALL3_C = "#777777"
LABEL_COLORS = {
    TAB_C: "#222222",
    DINO_C: "#276f9f",
    DEEPDXA_C: "#f2f2f2",
    ALL3_C: "#555555",
}
LEADER_LABEL_COLORS = {
    TAB_C: "#3f7f69",
    DINO_C: "#276f9f",
    DEEPDXA_C: "#083c7d",
    ALL3_C: "#555555",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.02) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
    )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        p
        for p in (ANNOTATED_HITS_PATH, CATALOG_HITS_PATH, HERITABILITY_PATH)
        if not p.exists()
    ]
    if missing:
        formatted = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(f"Missing Figure 4 input(s):\n{formatted}")

    annotated = pd.read_csv(ANNOTATED_HITS_PATH, sep="\t")
    catalog = pd.read_csv(CATALOG_HITS_PATH, sep="\t")
    h2 = pd.read_csv(HERITABILITY_PATH, sep="\t").dropna(subset=["h2"])

    # rs6954210 is flagged in_jepa=True in this file, but the raw JEPA_18
    # summary statistics give P=6.26568e-08 (> 5e-8) -- it does not reach
    # genome-wide significance in any of the 20 LeDXA PCs and is tabular-only.
    # hit_summary.tsv (used for the manuscript text counts) already has this
    # correct; annotated_hits.tsv does not. Correcting here until the
    # upstream pipeline artifact is regenerated.
    annotated.loc[annotated["ID"] == "rs6954210", ["in_jepa", "best_pheno_jepa", "best_p_jepa"]] = [
        False,
        np.nan,
        np.nan,
    ]

    # rs62104180's only tabular hit is on the duplicated "Body_mass_index_(BMI)"
    # phenotype column -- BMI is not a DXA-derived measurement and was excluded
    # from the tabular phenotype set; drop this artifact so the tabular count
    # matches the manuscript text (1,470, not 1,471).
    annotated = annotated[annotated["ID"] != "rs62104180"].reset_index(drop=True)

    return annotated, catalog, h2


def _overlap_counts(annotated: pd.DataFrame) -> dict[str, int]:
    tab = annotated["in_tabular"].astype(bool)
    dino = annotated["in_dino"].astype(bool)
    deepdxa = annotated["in_jepa"].astype(bool)
    return {
        "tab_only": int((tab & ~dino & ~deepdxa).sum()),
        "dino_only": int((~tab & dino & ~deepdxa).sum()),
        "tab_dino": int((tab & dino & ~deepdxa).sum()),
        "deepdxa_only": int((~tab & ~dino & deepdxa).sum()),
        "tab_deepdxa": int((tab & ~dino & deepdxa).sum()),
        "dino_deepdxa": int((~tab & dino & deepdxa).sum()),
        "all_three": int((tab & dino & deepdxa).sum()),
    }


def _jepa_pc_files() -> list[Path]:
    files = []
    for path in EMBEDDING_DIR.glob("gwas_results.JEPA_*.glm.linear"):
        match = re.search(r"gwas_results\.JEPA_(\d+)\.glm\.linear$", path.name)
        if match:
            files.append((int(match.group(1)), path))
    return [path for _, path in sorted(files)]


def _plot_manhattan_from_image(ax: plt.Axes) -> None:
    import matplotlib.image as mpimg

    image = mpimg.imread(MANHATTAN_PATH)
    cropped = image[105:, :, :]
    ax.imshow(cropped, aspect="auto")
    ax.axis("off")


# Canonical gene symbol (or None) for the top LeDXA-PC lead loci, matching the
# published annotated Manhattan plot. rsID -> gene symbol, derived from
# annotated_hits.tsv (lead SNP per 500kb-clumped locus, in_jepa==True, sorted
# by best_p_jepa); loci with no GWAS-catalog/RefSeq gene call are labeled by
# rsID alone.
MANHATTAN_LOCUS_LABELS: dict[str, str | None] = {
    "rs9594738": None,
    "rs2062375": "COLEC10",
    "rs72961013": "LOC105377989",
    "rs78110303": None,
    "rs917727": "FAM3C",
    "rs1128249": "COBLL1",
    "rs4711750": "POLR1C",
    "rs1524068": None,
    "rs1421085": "FTO",
    "rs7949030": "GANAB",
    "rs9687846": "C5orf67",
    "rs183211": "NSF",
    "rs7979524": "CCDC91",
    "rs6426749": None,
}


def draw_manhattan_panel(ax: plt.Axes) -> None:
    files = _jepa_pc_files()
    if not files:
        if MANHATTAN_PATH.exists():
            _plot_manhattan_from_image(ax)
            _panel_label(ax, "a", x=-0.045, y=1.02)
            return
        raise FileNotFoundError(f"Missing Figure 4 Manhattan input: {MANHATTAN_PATH}")

    base = pd.read_csv(files[0], sep="\t", usecols=["#CHROM", "POS", "ID", "P"])
    chrom = pd.to_numeric(base["#CHROM"], errors="coerce").to_numpy(dtype=float)
    pos = pd.to_numeric(base["POS"], errors="coerce").to_numpy(dtype=float)
    snp_id = base["ID"].to_numpy()
    min_p = pd.to_numeric(base["P"], errors="coerce").to_numpy(dtype=float)

    for path in files[1:]:
        p_values = pd.to_numeric(
            pd.read_csv(path, sep="\t", usecols=["P"])["P"], errors="coerce"
        ).to_numpy(dtype=float)
        if len(p_values) != len(min_p):
            raise ValueError(f"{path} has {len(p_values)} rows, expected {len(min_p)}")
        min_p = np.fmin(min_p, p_values)

    valid = np.isfinite(chrom) & np.isfinite(pos) & np.isfinite(min_p) & (min_p > 0)
    chrom = chrom[valid].astype(int)
    pos = pos[valid]
    snp_id = snp_id[valid]
    log_p = -np.log10(np.clip(min_p[valid], 1e-300, 1.0))

    order = np.argsort(chrom * 1_000_000_000 + pos, kind="stable")
    chrom = chrom[order]
    pos = pos[order]
    snp_id = snp_id[order]
    log_p = log_p[order]

    x_pos = np.zeros(len(pos), dtype=float)
    tick_positions = []
    tick_labels = []
    running = 0.0
    for chromosome in range(1, 23):
        mask = chrom == chromosome
        if not mask.any():
            continue
        chr_pos = pos[mask]
        span = chr_pos.max() - chr_pos.min() + 1
        x_pos[mask] = running + (chr_pos - chr_pos.min())
        tick_positions.append(running + span / 2)
        tick_labels.append(str(chromosome))
        tail_gap = 28_000_000 if chromosome >= 18 else 0
        running += span + span * 0.02 + tail_gap

    for chromosome in range(1, 23):
        mask = chrom == chromosome
        if not mask.any():
            continue
        color = DEEPDXA_C if chromosome % 2 else "#9a9a9a"
        ax.scatter(
            x_pos[mask],
            log_p[mask],
            s=2.2,
            color=color,
            alpha=0.8,
            rasterized=True,
            linewidths=0,
        )

    significant = log_p >= -np.log10(GW_THRESHOLD)
    ax.scatter(
        x_pos[significant],
        log_p[significant],
        s=5.5,
        color=DEEPDXA_C,
        rasterized=True,
        linewidths=0,
        zorder=4,
    )
    ax.axhline(
        -np.log10(GW_THRESHOLD),
        linestyle="--",
        linewidth=0.9,
        color="#444444",
        label="Genome-wide (P = 5×10⁻⁸)",
    )
    ax.axhline(
        -np.log10(SUGGESTIVE_THRESHOLD),
        linestyle=":",
        linewidth=0.75,
        color="#b8b8b8",
    )

    # xlim/ylim must be set before placing labels below — collision avoidance
    # needs the final data->pixel transform to measure rendered label boxes.
    ax.set_xlim(-running * 0.005, running * 1.005)
    ax.set_ylim(0, log_p.max() * 1.16)

    id_to_idx = {sid: i for i, sid in enumerate(snp_id)}
    label_points = []
    for rsid, gene in MANHATTAN_LOCUS_LABELS.items():
        i = id_to_idx.get(rsid)
        if i is None:
            continue
        label_points.append((rsid, gene, x_pos[i], log_p[i]))
    label_points.sort(key=lambda t: t[2])  # left to right, so bumps stack predictably

    renderer = ax.figure.canvas.get_renderer()
    placed_bboxes = []
    for rsid, gene, x, y in label_points:
        label = f"{gene}\n{rsid}" if gene else rsid
        # Centered labels near the left/right plot edge overhang past the axis
        # spine (e.g. rs6426749 on chr1) — anchor those to the inward side instead.
        if x < running * 0.03:
            ha, dx = "left", 3
        elif x > running * 0.97:
            ha, dx = "right", -3
        else:
            ha, dx = "center", 0

        dy = 4
        for _ in range(6):  # a handful of vertical bumps resolves any nearby-locus overlap
            ann = ax.annotate(
                label,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="bottom",
                fontsize=9,
                color="#222222",
                linespacing=1.1,
            )
            bbox = ann.get_window_extent(renderer=renderer).expanded(1.05, 1.15)
            if not any(bbox.overlaps(prev) for prev in placed_bboxes):
                placed_bboxes.append(bbox)
                break
            ann.remove()
            dy += 9  # one label-line's worth of vertical space per bump
        else:
            placed_bboxes.append(bbox)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Chromosome", fontsize=15)
    ax.set_ylabel(r"$-\log_{10}$(min P across PCs)", fontsize=15)
    ax.tick_params(axis="x", labelsize=8.5, pad=2)
    ax.tick_params(axis="y", labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    _panel_label(ax, "a", x=-0.045, y=1.02)


def draw_overlap_panel(ax: plt.Axes, annotated: pd.DataFrame) -> None:
    counts = _overlap_counts(annotated)
    methods = ["Tabular", "DINOv3", "LeDXA"]
    segments = {
        "Tabular": [
            ("Tabular only", counts["tab_only"], TAB_C),
            ("Tabular + DINOv3", counts["tab_dino"], DINO_C),
            ("Tabular + LeDXA", counts["tab_deepdxa"], DEEPDXA_C),
            ("All three", counts["all_three"], ALL3_C),
        ],
        "DINOv3": [
            ("DINOv3 only", counts["dino_only"], DINO_C),
            ("Tabular + DINOv3", counts["tab_dino"], TAB_C),
            ("DINOv3 + LeDXA", counts["dino_deepdxa"], DEEPDXA_C),
            ("All three", counts["all_three"], ALL3_C),
        ],
        "LeDXA": [
            ("LeDXA only", counts["deepdxa_only"], DEEPDXA_C),
            ("Tabular + LeDXA", counts["tab_deepdxa"], TAB_C),
            ("DINOv3 + LeDXA", counts["dino_deepdxa"], DINO_C),
            ("All three", counts["all_three"], ALL3_C),
        ],
    }

    x = np.arange(len(methods))
    width = 0.54
    totals = []
    thin_labels = []
    for xi, method in zip(x, methods):
        bottom = 0
        for _, value, color in segments[method]:
            if value <= 0:
                continue
            ax.bar(
                xi,
                value,
                width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.65,
            )
            if value >= 40:
                label_y = bottom + value / 2
                if bottom == 0 and value < 80:
                    label_y = bottom + value * 0.62
                label_color = LABEL_COLORS.get(color, "#222222")
                text = ax.text(
                    xi,
                    label_y,
                    f"{value}",
                    ha="center",
                    va="center",
                    fontsize=11.2,
                    color="#666666",
                    fontweight="bold" if color == DEEPDXA_C else "normal",
                )
                text.set_color(label_color)
                if color == DEEPDXA_C:
                    text.set_path_effects(
                        [path_effects.withStroke(linewidth=1.1, foreground="#083c7d")]
                    )
            else:
                thin_labels.append((xi, bottom + value / 2, int(value), color))
            bottom += value
        totals.append(bottom)
        ax.text(
            xi,
            bottom + 35,
            f"{bottom}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#444444",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("# genome-wide-significant SNPs", fontsize=10.5)
    ymax = max(totals)
    ax.set_ylim(0, ymax * 1.08)
    ax.tick_params(axis="y", labelsize=9)
    ax.margins(x=0.06)
    ax.set_xlim(-0.35, 2.88)

    legend_handles = [
        mpatches.Patch(facecolor=TAB_C, label="Tabular"),
        mpatches.Patch(facecolor=DINO_C, label="DINOv3"),
        mpatches.Patch(facecolor=DEEPDXA_C, label="LeDXA"),
        mpatches.Patch(facecolor=ALL3_C, label="All three"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.01),
        frameon=False,
        fontsize=9,
        borderpad=0.2,
        handlelength=1.0,
    )

    min_gap = ymax * 0.055
    floor = ymax * 0.055
    for xi in x:
        group = sorted([label for label in thin_labels if label[0] == xi], key=lambda t: t[1])
        placed = []
        for _, y_anchor, value, color in group:
            y_text = max(y_anchor, floor)
            if placed and y_text < placed[-1] + min_gap:
                y_text = placed[-1] + min_gap
            placed.append(y_text)
            ax.annotate(
                str(value),
                xy=(xi + width / 2, y_anchor),
                xytext=(xi + width / 2 + 0.18, y_text),
                ha="left",
                va="center",
                fontsize=10.8,
                color=LEADER_LABEL_COLORS.get(color, color),
                fontweight="bold",
                clip_on=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": LEADER_LABEL_COLORS.get(color, color),
                    "lw": 0.8,
                },
            )
    _panel_label(ax, "b", x=-0.18, y=1.04)


# Panel c is derived reproducibly from fig4c/fig4c_primary.tsv: each of the 18
# LeDXA-specific loci is assigned ONE primary organ-system domain from its lead
# SNP's GWAS-Catalog associations (LD proxies, 1000G EUR r2>=0.8, are used only
# when the lead SNP is itself uncatalogued), preferring the strongest on-target
# {Bone / Body composition / Height} association. See fig4c/build_fig4c.py for
# the pipeline and fig4c/fig4c_associations.tsv for the full multi-domain
# evidence (Supplement). Loci with no catalogued association in any domain are
# shown as their own "Catalog-absent" bar rather than being dropped.
PRIMARY_TSV = ROOT / "tables" / "fig4c" / "fig4c_primary.tsv"
# on-target (musculoskeletal / body-composition) domains, pinned to the top of
# the panel in this order; all remaining domains follow, sorted by descending count.
ON_TARGET_TOP = [
    "Body composition",
    "Bone",
    "Height/Anthropometric",
    "Joint/connective-tissue",
]
# sentence-case display labels for the y-axis tick text
DISPLAY_LABEL = {
    "Height/Anthropometric": "Height/anthropometric",
    "Joint/connective-tissue": "Joint/connective tissue",
    "Metabolic/Lipid": "Metabolic/lipid",
    "Reproductive/Endocrine": "Reproductive/endocrine",
    "Neuro/Behavioral": "Neuro/behavioral",
    "Immune/Hematologic": "Immune/hematologic",
}


def _locus_domain_counts() -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(PRIMARY_TSV, sep="\t")
    n_loci = len(df)
    counts = Counter(df.loc[df["primary_domain"] != "(none/novel)", "primary_domain"])
    n_novel = int((df["primary_domain"] == "(none/novel)").sum())
    top = [d for d in ON_TARGET_TOP if counts.get(d, 0) > 0]
    rest = sorted((d for d in counts if d not in ON_TARGET_TOP),
                  key=lambda d: (-counts[d], d))
    rows = [{"Category": DISPLAY_LABEL.get(d, d), "Count": counts[d], "novel": False}
            for d in top + rest]
    if n_novel:
        rows.append({"Category": "Catalog-absent", "Count": n_novel, "novel": True})
    return pd.DataFrame(rows), n_loci


def draw_catalog_panel(ax: plt.Axes, annotated: pd.DataFrame, catalog: pd.DataFrame) -> None:
    cat_df, n_loci = _locus_domain_counts()
    cat_df = cat_df.iloc[::-1].reset_index(drop=True)

    bar_colors = [ALL3_C if novel else DEEPDXA_C for novel in cat_df["novel"]]
    bars = ax.barh(
        cat_df["Category"],
        cat_df["Count"],
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    for bar, value in zip(bars, cat_df["Count"]):
        ax.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{int(value)}",
            ha="left",
            va="center",
            fontsize=12,
        )
    ax.set_xlabel("Number of loci", fontsize=10.5)
    ax.set_xlim(0, cat_df["Count"].max() + 1)
    ax.set_xticks(range(0, int(cat_df["Count"].max()) + 2))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10.4)
    _panel_label(ax, "c", x=-0.055, y=1.02)


def _ordered_h2(h2: pd.DataFrame, group: str) -> pd.DataFrame:
    out = h2[h2["group"] == group].copy()
    out["idx"] = out["pheno"].str.split("_").str[-1].astype(int)
    return out.sort_values("idx")


def draw_heritability_panel(ax: plt.Axes, h2: pd.DataFrame) -> None:
    dino = _ordered_h2(h2, "dino")
    deepdxa = _ordered_h2(h2, "jepa")
    if dino.empty or deepdxa.empty:
        raise ValueError("Heritability table is missing dino or jepa PC rows")

    dino_x = np.arange(len(dino))
    deepdxa_x = np.arange(len(deepdxa)) + len(dino) + 1.25
    ax.bar(
        dino_x,
        dino["h2"],
        yerr=dino["h2_se"],
        color=DINO_C,
        edgecolor="#333333",
        linewidth=0.35,
        capsize=2,
        error_kw={"lw": 0.65},
        label="DINOv3",
    )
    ax.bar(
        deepdxa_x,
        deepdxa["h2"],
        yerr=deepdxa["h2_se"],
        color=DEEPDXA_C,
        edgecolor="#333333",
        linewidth=0.35,
        capsize=2,
        error_kw={"lw": 0.65},
        label="LeDXA",
    )
    ax.hlines(
        dino["h2"].mean(),
        dino_x[0] - 0.45,
        dino_x[-1] + 0.45,
        linestyle="--",
        color="black",
        linewidth=0.8,
    )
    ax.hlines(
        deepdxa["h2"].mean(),
        deepdxa_x[0] - 0.45,
        deepdxa_x[-1] + 0.45,
        linestyle="--",
        color="black",
        linewidth=0.8,
    )

    all_x = np.concatenate([dino_x, deepdxa_x])
    ax.set_xticks(all_x)
    ax.set_xticklabels(
        [f"PC{i}" for i in dino["idx"]] + [f"PC{i}" for i in deepdxa["idx"]],
        rotation=90,
        fontsize=8,
    )
    ax.set_ylabel(r"SNP $h^2$ (observed scale)", fontsize=14)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 0.32)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("LeDXA"), labels.index("DINOv3")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        frameon=False,
        fontsize=11,
    )

    t_stat, p_value = ttest_ind(dino["h2"], deepdxa["h2"], equal_var=False)
    ax.text(
        0.52,
        0.98,
        f"Welch's t = {t_stat:.2f}, P = {p_value:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#b8b8b8",
            "linewidth": 0.7,
        },
    )
    _panel_label(ax, "d", x=-0.105, y=1.08)


def build_figure() -> plt.Figure:
    annotated, catalog, h2 = _load_inputs()

    fig = plt.figure(figsize=(12.6, 14.8), dpi=300)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.5, 2.28, 1.02],
        hspace=0.26,
        wspace=0.42,
    )

    ax_a = fig.add_subplot(grid[0, :])
    draw_manhattan_panel(ax_a)

    ax_b = fig.add_subplot(grid[1, 0])
    draw_overlap_panel(ax_b, annotated)

    ax_c = fig.add_subplot(grid[1, 1])
    draw_catalog_panel(ax_c, annotated, catalog)

    ax_d = fig.add_subplot(grid[2, :])
    draw_heritability_panel(ax_d, h2)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.985, bottom=0.075)
    return fig


def main() -> None:
    fig = build_figure()
    for ext in ("png", "pdf"):
        out_path = OUT_DIR / f"fig4_genetics.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
