#!/usr/bin/env python
"""Generate the current quality-control figures.

Outputs:
- figures/quality_control/panel_outlier_pca_raw.png
- figures/quality_control/qc_panel_preprocessing.png
- figures/quality_control/qc_panel_plots/*.png

This replaces the older lettered one-off QC panel scripts. The standalone
outlier PCA and the combined preprocessing panel share the same raw PCA
calculation so the highlighted outliers are consistent across both figures.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cancerscout.paths import (
    ACTIVE_RAW_NORMICS_DIR,
    FIGURES_DIR,
    QUALITY_CONTROL_RESULTS_DIR,
)
from cancerscout.plot_style import (
    SEQUENTIAL_CMAP,
    font_pt,
    savefig,
    set_house_style,
)
from cancerscout.qc_data import (
    PLATE_COLOR,
    RAW_BID_FILE,
    STAGE_COLOR,
    STAGES,
    aligned,
    kde_line,
    load_data,
    pca2,
    sorted_plates,
)

RAW_META_FILE = ACTIVE_RAW_NORMICS_DIR / "intermediate/sample_metadata_selected.csv"
OUTLIER_IDS = {
    "I_D201218_MDIA_P2568_SExp01-BID0751_R01",
    "I_D201218_MDIA_P2568_SExp01-BID0762_R01",
}
DEFAULT_FIGURES_DIR = FIGURES_DIR / "quality_control"
OUTLIER_PCA_NAME = "panel_outlier_pca_raw.png"
PREPROCESSING_PANEL_NAME = "qc_panel_preprocessing.png"
PREPROCESSING_PLOTS_DIR_NAME = "qc_panel_plots"
RAW_DETECTED_TSV = QUALITY_CONTROL_RESULTS_DIR / "sample_qc_metrics.tsv"
RAW_GROUP_ORDER = ["BID", "BatchQC", "QC-pool", "Other"]


@dataclass
class RawOutlierPca:
    scores: np.ndarray
    explained_variance: np.ndarray
    plates: pd.Series
    missing_fraction: pd.Series
    sample_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--only",
        choices=["all", "outlier-pca", "preprocessing-panel"],
        default="all",
        help="Limit generation to one figure. The default writes both requested QC figures.",
    )
    return parser.parse_args()


def _ylabel_yticks(ax: plt.Axes, label: str, show: bool) -> None:
    if show:
        ax.set_ylabel(label)


def draw_detected_proteins(ax: plt.Axes, show_ylabel: bool = True) -> None:
    """Detected protein groups per sample, by raw sample group.

    Mirrors the standalone `raw_detected_proteins_by_sample_group.png` written by
    `run_raw_qc`, so the panel opens on assay depth rather than on a missingness
    density that the neighbouring panels already cover.
    """
    raw_sample = pd.read_csv(RAW_DETECTED_TSV, sep="\t")
    raw_sample = raw_sample[raw_sample["matrix"].eq("raw reshaped")]
    order = [
        g for g in RAW_GROUP_ORDER if g in set(raw_sample["inferred_sample_group"])
    ]
    palette = sns.color_palette(SEQUENTIAL_CMAP, n_colors=max(len(order), 3))
    group_colors = dict(zip(order, palette))

    sns.boxplot(
        raw_sample,
        x="inferred_sample_group",
        y="n_detected",
        order=order,
        palette=group_colors,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("")
    _ylabel_yticks(ax, "Detected protein groups", show_ylabel)
    ax.tick_params(axis="x", rotation=20)


def draw_protein_std(
    ax: plt.Axes,
    matrices: list[pd.DataFrame],
    meta: pd.DataFrame,
    show_ylabel: bool = True,
) -> None:
    for df, stage in zip(matrices, STAGES):
        sub, _ = aligned(df, meta)
        kde_line(ax, sub.std(axis=0, skipna=True).values, STAGE_COLOR[stage], stage)
    ax.set_xlabel("Per-protein std (log2)")
    _ylabel_yticks(ax, "Density", show_ylabel)
    ax.set_xlim(left=0)
    ax.legend()


def draw_batch_signal(
    ax: plt.Axes,
    matrices: list[pd.DataFrame],
    meta: pd.DataFrame,
    show_ylabel: bool = True,
) -> None:
    for df, stage in zip(matrices, STAGES):
        sub, plates = aligned(df, meta)
        sds = sub.groupby(plates).median().std(axis=0, skipna=True).values
        kde_line(ax, sds, STAGE_COLOR[stage], stage)
    ax.set_xlabel("SD of plate medians (log2)")
    _ylabel_yticks(ax, "Density", show_ylabel)
    ax.set_xlim(left=0)
    ax.legend()


def draw_plate_boxes(
    ax: plt.Axes,
    df: pd.DataFrame,
    meta: pd.DataFrame,
    stage: str,
    ylim: tuple[float, float],
    show_ylabel: bool,
    show_yticks: bool,
) -> None:
    sub, plates = aligned(df, meta)
    order = sorted_plates(plates)
    per_sample = sub.median(axis=1, skipna=True)
    data = [per_sample[plates == p].dropna().values for p in order]

    bp = ax.boxplot(
        data,
        positions=range(len(order)),
        widths=0.6,
        patch_artist=True,
        flierprops=dict(marker=".", markersize=2.5, alpha=0.4, linestyle="none"),
        medianprops=dict(linewidth=1.5, color="black"),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
    )
    for patch, plate in zip(bp["boxes"], order):
        patch.set_facecolor(PLATE_COLOR[plate])
        patch.set_alpha(0.8)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Plate")
    ax.set_ylim(ylim)
    if show_ylabel:
        ax.set_ylabel("Median log2 abundance")
    if not show_yticks:
        ax.set_yticklabels([])
    ax.text(
        0.03,
        0.97,
        stage,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="dimgray",
        style="italic",
        fontsize=font_pt("tick"),
    )


def _scatter_plates(
    ax: plt.Axes,
    scores: np.ndarray,
    plates: pd.Series,
    skip_ids: set[str] = frozenset(),
    show_legend: bool = False,
) -> None:
    plate_vals = plates.to_numpy()
    sample_ids = plates.index.astype(str).tolist()
    for plate in sorted_plates(plates):
        mask = np.array(
            [
                plate_vals[i] == plate and sample_ids[i] not in skip_ids
                for i in range(len(plate_vals))
            ]
        )
        if mask.any():
            ax.scatter(
                scores[mask, 0],
                scores[mask, 1],
                c=[PLATE_COLOR[plate]],
                s=8,
                linewidths=0,
                alpha=0.6,
                label=f"P{plate}",
            )
    if show_legend:
        ax.legend(loc="lower right", ncol=2, title="Plate", markerscale=1.8)


def _bid_label(sample_id: str) -> str:
    match = re.search(r"BID\d+", sample_id)
    return match.group() if match else sample_id


def build_raw_outlier_pca() -> RawOutlierPca:
    raw = pd.read_csv(RAW_BID_FILE, index_col=0).astype(float)
    raw_l2 = np.log2(raw.replace(0.0, np.nan))
    meta_all = pd.read_csv(RAW_META_FILE, index_col=0)
    meta_bid = meta_all[meta_all["sample_group"] == "BID"].copy()
    meta_bid["Plate"] = meta_bid["Plate"].astype(int).astype(str)

    shared = raw_l2.index.intersection(meta_bid.index)
    sub = raw_l2.loc[shared]
    plates = meta_bid.loc[shared, "Plate"]
    scores, exp = pca2(sub)

    return RawOutlierPca(
        scores=scores,
        explained_variance=exp,
        plates=plates,
        missing_fraction=sub.isna().mean(axis=1),
        sample_ids=sub.index.astype(str).tolist(),
    )


def draw_raw_outlier_pca(
    ax: plt.Axes,
    raw_pca: RawOutlierPca,
    show_legend: bool,
    show_stage_label: bool,
    annotation_fontsize: float,
    annotation_offsets: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Raw PCA over all samples, with the two excluded outliers called out.

    Leader lines rather than a legend: both outliers sit in the sparse top-right
    of the panel, so a label anchored there points at empty space, whereas a
    legend has to sit somewhere on the dense cloud.
    """
    sample_to_idx = {sample_id: i for i, sample_id in enumerate(raw_pca.sample_ids)}
    outlier_ids = {sample_id for sample_id in OUTLIER_IDS if sample_id in sample_to_idx}
    offsets = annotation_offsets or {}

    _scatter_plates(
        ax,
        raw_pca.scores,
        raw_pca.plates,
        skip_ids=outlier_ids,
        show_legend=show_legend,
    )
    plate_vals = raw_pca.plates.to_numpy()
    for sample_id in sorted(outlier_ids, key=raw_pca.sample_ids.index):
        i = sample_to_idx[sample_id]
        ax.scatter(
            raw_pca.scores[i, 0],
            raw_pca.scores[i, 1],
            c=[PLATE_COLOR[plate_vals[i]]],
            s=8,
            linewidths=0,
            alpha=0.7,
            zorder=4,
        )
        missing_pct = raw_pca.missing_fraction.loc[sample_id] * 100
        ax.annotate(
            f"{_bid_label(sample_id)}\n({missing_pct:.1f}% missing)",
            xy=(raw_pca.scores[i, 0], raw_pca.scores[i, 1]),
            xytext=offsets.get(sample_id, (-8, -30)),
            textcoords="offset points",
            color="dimgray",
            fontsize=annotation_fontsize,
            ha="right",
            va="top",
            linespacing=1.0,
            arrowprops=dict(arrowstyle="-", color="dimgray", lw=0.6),
        )

    ax.set_xlabel(f"PC1 ({raw_pca.explained_variance[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({raw_pca.explained_variance[1] * 100:.1f}%)")
    if show_stage_label:
        ax.text(
            0.03,
            0.97,
            "Raw (all samples)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color="dimgray",
            style="italic",
            fontsize=font_pt("tick"),
        )


def draw_pca_plate(
    ax: plt.Axes,
    df: pd.DataFrame,
    meta: pd.DataFrame,
    label: str,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    sub, plates = aligned(df, meta)
    scores, exp = pca2(sub)
    _scatter_plates(ax, scores, plates, show_legend=show_legend)
    ax.set_xlabel(f"PC1 ({exp[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({exp[1] * 100:.1f}%)")
    ax.yaxis.label.set_visible(show_ylabel)
    ax.text(
        0.03,
        0.97,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="dimgray",
        style="italic",
        fontsize=font_pt("tick"),
    )


def write_outlier_pca(figures_dir: Path, raw_pca: RawOutlierPca) -> Path:
    out = figures_dir / OUTLIER_PCA_NAME
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    draw_raw_outlier_pca(
        ax,
        raw_pca,
        show_legend=True,
        show_stage_label=False,
        annotation_fontsize=font_pt("label"),
        annotation_offsets={
            "I_D201218_MDIA_P2568_SExp01-BID0751_R01": (-14, -30),
            "I_D201218_MDIA_P2568_SExp01-BID0762_R01": (-14, -72),
        },
    )
    fig.tight_layout()
    savefig(fig, out)
    return out


def write_preprocessing_panel(figures_dir: Path, raw_pca: RawOutlierPca) -> Path:
    raw_l2, norm, final, meta_bid = load_data()
    matrices = [raw_l2, norm, final]

    all_meds = np.concatenate(
        [
            aligned(df, meta_bid)[0].median(axis=1, skipna=True).dropna().values
            for df in matrices
        ]
    )
    pad = 0.05 * (all_meds.max() - all_meds.min())
    box_ylim = (all_meds.min() - pad, all_meds.max() + pad)

    fig = plt.figure(figsize=(7.2, 7.2))
    # No plate legend: the plate identity of an individual point is not the
    # message here, the loss of plate structure across stages is.
    gs = gridspec.GridSpec(3, 12, height_ratios=[4, 4, 3], hspace=0.38, wspace=0.90)

    ax_summary = [fig.add_subplot(gs[0, j * 4 : (j + 1) * 4]) for j in range(3)]
    ax_box = [fig.add_subplot(gs[1, j * 4 : (j + 1) * 4]) for j in range(3)]
    ax_pca = [fig.add_subplot(gs[2, j * 3 : (j + 1) * 3]) for j in range(4)]

    print("  Row 1: summary")
    draw_detected_proteins(ax_summary[0], show_ylabel=True)
    draw_protein_std(ax_summary[1], matrices, meta_bid, show_ylabel=False)
    draw_batch_signal(ax_summary[2], matrices, meta_bid, show_ylabel=False)

    print("  Row 2: boxplots")
    for j, (df, stage) in enumerate(zip(matrices, STAGES)):
        draw_plate_boxes(
            ax_box[j],
            df,
            meta_bid,
            stage,
            ylim=box_ylim,
            show_ylabel=(j == 0),
            show_yticks=(j == 0),
        )

    print("  Row 3: PCA")
    draw_raw_outlier_pca(
        ax_pca[0],
        raw_pca,
        show_legend=False,
        show_stage_label=True,
        annotation_fontsize=font_pt("annot"),
        annotation_offsets={
            "I_D201218_MDIA_P2568_SExp01-BID0751_R01": (-12, -20),
            "I_D201218_MDIA_P2568_SExp01-BID0762_R01": (-10, -34),
        },
    )
    draw_pca_plate(
        ax_pca[1], raw_l2, meta_bid, "Raw (997)", show_ylabel=False, show_legend=False
    )
    draw_pca_plate(
        ax_pca[2], norm, meta_bid, "Normics/VSN", show_ylabel=False, show_legend=False
    )
    draw_pca_plate(
        ax_pca[3],
        final,
        meta_bid,
        "Normics+limma",
        show_ylabel=False,
        show_legend=False,
    )

    out = figures_dir / PREPROCESSING_PANEL_NAME
    savefig(fig, out, close=False)
    write_preprocessing_plots(
        figures_dir / PREPROCESSING_PLOTS_DIR_NAME,
        fig,
        [*ax_summary, *ax_box, *ax_pca],
    )
    plt.close(fig)
    return out


def write_preprocessing_plots(
    figures_dir: Path,
    fig: plt.Figure,
    axes: list[plt.Axes],
) -> None:
    """Write clean, title-free standalone exports from the preprocessing figure."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    panel_specs = [
        ("detected_protein_groups", "Sample group", "Detected protein groups"),
        ("per_protein_std", "Per-protein std (log2)", "Density"),
        ("plate_median_sd", "SD of plate medians (log2)", "Density"),
        ("plate_abundance_raw", "Plate", "Median log2 abundance"),
        ("plate_abundance_normics_vsn", "Plate", "Median log2 abundance"),
        ("plate_abundance_normics_limma", "Plate", "Median log2 abundance"),
        ("pca_raw_all_samples", None, None),
        ("pca_raw_997", None, None),
        ("pca_normics_vsn", None, None),
        ("pca_normics_limma", None, None),
    ]
    for axis, (name, xlabel, ylabel) in zip(axes, panel_specs):
        visibility = [other.get_visible() for other in axes]
        original_position = axis.get_position()
        original_xlabel = axis.get_xlabel()
        original_ylabel = axis.get_ylabel()
        xlabel_visible = axis.xaxis.label.get_visible()
        ylabel_visible = axis.yaxis.label.get_visible()
        original_title = axis.get_title(loc="left")
        hidden_stage_labels = [
            artist
            for artist in axis.texts
            if artist.get_text()
            in {
                "Raw",
                "Raw (all samples)",
                "Raw (997)",
                "Normics/VSN",
                "Normics+limma",
            }
        ]
        for other in axes:
            other.set_visible(other is axis)
        # The source axes occupy narrow composite-grid slots. Give each export
        # the same square canvas so labels and legends are not clipped by the
        # source slot, especially for the rightmost panels.
        axis.set_position([0.18, 0.16, 0.76, 0.76])
        for artist in hidden_stage_labels:
            artist.set_visible(False)
        axis.set_title("")
        axis.set_xlabel(xlabel if xlabel is not None else original_xlabel)
        axis.set_ylabel(ylabel if ylabel is not None else original_ylabel)
        axis.xaxis.label.set_visible(True)
        axis.yaxis.label.set_visible(True)
        savefig(fig, figures_dir / name, close=False)
        axis.set_title(original_title, loc="left")
        axis.set_xlabel(original_xlabel)
        axis.set_ylabel(original_ylabel)
        axis.set_position(original_position)
        axis.xaxis.label.set_visible(xlabel_visible)
        axis.yaxis.label.set_visible(ylabel_visible)
        for artist in hidden_stage_labels:
            artist.set_visible(True)
        for other, visible in zip(axes, visibility):
            other.set_visible(visible)


def main() -> None:
    args = parse_args()
    set_house_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw PCA data...")
    raw_pca = build_raw_outlier_pca()

    written: list[Path] = []
    if args.only in {"all", "outlier-pca"}:
        print("Writing standalone raw outlier PCA...")
        written.append(write_outlier_pca(args.figures_dir, raw_pca))
    if args.only in {"all", "preprocessing-panel"}:
        print("Writing combined preprocessing QC panel...")
        written.append(write_preprocessing_panel(args.figures_dir, raw_pca))

    for path in written:
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
