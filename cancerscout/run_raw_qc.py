"""Generate raw/preprocessing QC tables, figures, and report.

This is a lightweight, paper-facing QC layer for the current R14 Goettingen
CancerScout extraction. It does not rerun Normics or limma; it audits the raw
reshaped matrix and the final Normics+limma matrix used downstream.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cancerscout.io import save_text, write_tsv
from cancerscout.plot_style import (
    font_pt,
    MSI_STATUS_COLORS,
    SEQUENTIAL_CMAP,
    clean_ax,
    make_color_map,
    place_legend,
    savefig,
    set_house_style,
)
from cancerscout.paths import (
    ACTIVE_RAW_NORMICS_DIR,
    FIGURES_DIR,
    NORMALIZED_LIMMA_CSV,
    QUALITY_CONTROL_RESULTS_DIR,
    REPORTS_DIR,
)


RAW_SAMPLE_COUNTS_FIGURE = "raw_sample_counts_by_group.png"
RAW_DETECTED_PROTEINS_FIGURE = "raw_detected_proteins_by_sample_group.png"
RAW_DETECTED_PROTEINS_FILTERED_FIGURE = (
    "raw_detected_proteins_by_sample_group_no_low_bid_outliers.png"
)
RAW_BASIC_QC_SUMMARY = "raw_proteomics_basic_qc_summary.tsv"
RAW_GROUP_ORDER = ["BID", "BatchQC", "QC-pool", "Other"]
LOW_BID_DETECTED_PROTEIN_THRESHOLD = 7600


def infer_raw_sample_group(sample: str) -> str:
    sample = str(sample)
    if re.search(r"BID\d+", sample, flags=re.IGNORECASE):
        return "BID"
    if re.search(r"BatchQC", sample, flags=re.IGNORECASE):
        return "BatchQC"
    if re.search(r"QC[-_ ]?pool", sample, flags=re.IGNORECASE):
        return "QC-pool"
    return "Other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normics-dir", type=Path, default=ACTIVE_RAW_NORMICS_DIR)
    parser.add_argument("--out-dir", type=Path, default=QUALITY_CONTROL_RESULTS_DIR)
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR / "quality_control"
    )
    parser.add_argument(
        "--analysable-matrix",
        type=Path,
        default=NORMALIZED_LIMMA_CSV,
        help="Processed Normics+limma matrix for the analysable cohort (997 tumours).",
    )
    parser.add_argument("--top-variable-proteins", type=int, default=2000)
    return parser.parse_args()


def read_raw_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, low_memory=False)
    annotation_rows = raw.iloc[:4].copy()
    annotation_rows = annotation_rows.rename(columns={"r_file_name": "annotation"})
    matrix = raw.iloc[4:].copy()
    matrix = matrix.rename(columns={"r_file_name": "sample"})
    matrix = matrix.set_index("sample")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    matrix = matrix.mask(matrix <= 0)
    return matrix, annotation_rows


def read_final_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0).T
    matrix.index.name = "sample"
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    return matrix


def sample_qc(
    matrix: pd.DataFrame, label: str, meta: pd.DataFrame | None = None
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "sample": matrix.index.astype(str),
            "matrix": label,
            "n_proteins": matrix.shape[1],
            "n_detected": matrix.notna().sum(axis=1).to_numpy(),
            "missing_fraction": matrix.isna().mean(axis=1).to_numpy(),
            "median_abundance": matrix.median(axis=1, skipna=True).to_numpy(),
            "mean_abundance": matrix.mean(axis=1, skipna=True).to_numpy(),
        }
    )
    if meta is not None and "sample" in meta.columns:
        cols = [
            c
            for c in [
                "sample",
                "sample_group",
                "bid",
                "bid_numeric",
                "acquisition_batch",
                "Plate",
                "MSI_final",
            ]
            if c in meta.columns
        ]
        out = out.merge(meta[cols], on="sample", how="left")
    if label == "raw reshaped":
        out["inferred_sample_group"] = out["sample"].map(infer_raw_sample_group)
    else:
        out["inferred_sample_group"] = pd.NA
    return out


def protein_qc(matrix: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "protein": matrix.columns.astype(str),
            "matrix": label,
            "n_samples": matrix.shape[0],
            "n_detected": matrix.notna().sum(axis=0).to_numpy(),
            "missing_fraction": matrix.isna().mean(axis=0).to_numpy(),
            "median_abundance": matrix.median(axis=0, skipna=True).to_numpy(),
            "mean_abundance": matrix.mean(axis=0, skipna=True).to_numpy(),
        }
    )


def run_pca(matrix: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = matrix.loc[:, matrix.notna().mean(axis=0) >= 0.5].copy()
    if usable.shape[1] > top_n:
        variable = (
            usable.var(axis=0, skipna=True)
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        usable = usable.loc[:, variable]
    usable = usable.apply(lambda s: s.fillna(s.median()), axis=0)
    x = StandardScaler().fit_transform(usable)
    pca = PCA(n_components=5, random_state=0)
    coords = pd.DataFrame(
        pca.fit_transform(x),
        index=usable.index,
        columns=[f"PC{i}" for i in range(1, 6)],
    )
    variance = pd.DataFrame(
        {
            "component": coords.columns,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return coords, variance


def raw_basic_qc_summary(
    raw_sample: pd.DataFrame, raw_protein: pd.DataFrame
) -> pd.DataFrame:
    rows = []

    def add(section: str, metric: str, value: float, group: str = "all") -> None:
        rows.append(
            {"section": section, "metric": metric, "group": group, "value": value}
        )

    add("sample_runs", "n_raw_runs", int(raw_sample.shape[0]))
    for group in RAW_GROUP_ORDER:
        n = int(raw_sample["inferred_sample_group"].eq(group).sum())
        if n > 0 or group == "Other":
            add("sample_runs", "n_raw_runs", n, group)

    add("protein_groups", "n_protein_groups", int(raw_protein.shape[0]))
    for label, threshold in [
        ("detected_any_run", 1),
        ("detected_ge_50pct_runs", 0.50),
        ("detected_ge_90pct_runs", 0.90),
        ("detected_ge_95pct_runs", 0.95),
        ("detected_ge_99pct_runs", 0.99),
        ("detected_100pct_runs", 1.00),
    ]:
        if label == "detected_any_run":
            value = int(raw_protein["n_detected"].ge(threshold).sum())
        else:
            coverage = 1.0 - raw_protein["missing_fraction"]
            value = int(coverage.ge(threshold).sum())
        add("protein_groups", label, value)

    add(
        "per_sample",
        "median_detected_proteins_per_run",
        float(raw_sample["n_detected"].median()),
    )
    add(
        "per_sample",
        "median_sample_missing_fraction",
        float(raw_sample["missing_fraction"].median()),
    )
    add(
        "per_sample",
        "p95_sample_missing_fraction",
        float(raw_sample["missing_fraction"].quantile(0.95)),
    )
    add(
        "per_protein",
        "median_protein_missing_fraction",
        float(raw_protein["missing_fraction"].median()),
    )
    add(
        "per_protein",
        "p95_protein_missing_fraction",
        float(raw_protein["missing_fraction"].quantile(0.95)),
    )
    add(
        "per_protein",
        "p99_protein_missing_fraction",
        float(raw_protein["missing_fraction"].quantile(0.99)),
    )
    return pd.DataFrame(rows)


def plot_detected_proteins_by_group(
    raw_sample: pd.DataFrame,
    order: list[str],
    group_colors: dict[str, tuple[float, float, float]],
    out_path: Path,
) -> None:
    fig, ax_detected = plt.subplots(figsize=(4.8, 3.4))
    sns.boxplot(
        raw_sample,
        x="inferred_sample_group",
        y="n_detected",
        order=order,
        showfliers=False,
        palette=group_colors,
        ax=ax_detected,
    )
    sns.stripplot(
        raw_sample,
        x="inferred_sample_group",
        y="n_detected",
        order=order,
        color="0.20",
        alpha=0.22,
        size=1.8,
        jitter=0.25,
        ax=ax_detected,
    )
    ax_detected.set_xlabel("")
    ax_detected.set_ylabel("Detected protein groups")
    ax_detected.tick_params(axis="x", rotation=25)
    clean_ax(ax_detected)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_raw_basic_qc(
    raw_sample: pd.DataFrame,
    raw_protein: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
) -> pd.DataFrame:
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = raw_basic_qc_summary(raw_sample, raw_protein)
    write_tsv(summary, out_dir / RAW_BASIC_QC_SUMMARY)

    group_counts = (
        raw_sample["inferred_sample_group"]
        .value_counts()
        .reindex(RAW_GROUP_ORDER)
        .dropna()
        .astype(int)
    )
    order = group_counts.index.tolist()
    palette = sns.color_palette(SEQUENTIAL_CMAP, n_colors=max(len(order), 3))
    group_colors = dict(zip(order, palette))

    fig, ax_counts = plt.subplots(figsize=(4.2, 3.2))
    ax_counts.bar(
        group_counts.index,
        group_counts.values,
        color=[group_colors[g] for g in group_counts.index],
        edgecolor="white",
        linewidth=0.7,
    )
    ax_counts.set_ylabel("Number of samples")
    ax_counts.set_xlabel("")
    ax_counts.tick_params(axis="x", rotation=25)
    for x, value in enumerate(group_counts.values):
        ax_counts.text(
            x, value, f"{value:,}", ha="center", va="bottom", fontsize=font_pt("label")
        )
    clean_ax(ax_counts)
    fig.tight_layout()
    savefig(fig, fig_dir / RAW_SAMPLE_COUNTS_FIGURE)

    plot_detected_proteins_by_group(
        raw_sample,
        order=order,
        group_colors=group_colors,
        out_path=fig_dir / RAW_DETECTED_PROTEINS_FIGURE,
    )
    filtered_raw_sample = raw_sample[
        ~(
            raw_sample["inferred_sample_group"].eq("BID")
            & raw_sample["n_detected"].lt(LOW_BID_DETECTED_PROTEIN_THRESHOLD)
        )
    ].copy()
    plot_detected_proteins_by_group(
        filtered_raw_sample,
        order=order,
        group_colors=group_colors,
        out_path=fig_dir / RAW_DETECTED_PROTEINS_FILTERED_FIGURE,
    )
    return summary


def plot_qc(
    sample_stats: pd.DataFrame,
    protein_stats: pd.DataFrame,
    raw: pd.DataFrame,
    final: pd.DataFrame,
    final_997: pd.DataFrame,
    meta: pd.DataFrame,
    out_dir: Path,
    fig_dir: Path,
    top_n: int,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.histplot(
        sample_stats,
        x="missing_fraction",
        hue="matrix",
        element="step",
        stat="density",
        common_norm=False,
        bins=45,
        ax=ax,
    )
    ax.set_xlabel("Sample missing fraction")
    ax.set_ylabel("Density")
    clean_ax(ax)
    place_legend(ax, mode="inside", title=None)
    savefig(fig, fig_dir / "sample_missing_fraction_raw_vs_final.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sns.histplot(
        protein_stats,
        x="missing_fraction",
        hue="matrix",
        element="step",
        stat="density",
        common_norm=False,
        bins=45,
        ax=ax,
    )
    ax.set_xlabel("Protein missing fraction")
    ax.set_ylabel("Density")
    clean_ax(ax)
    place_legend(ax, mode="inside", title=None)
    savefig(fig, fig_dir / "protein_missing_fraction_raw_vs_final.png")

    # The analysable-cohort PCA is a separate entry because it uses the processed
    # 997-sample matrix, not the 999-sample `final` matrix read from the raw
    # Normics output: the manuscript reports 997 tumour proteomes.
    for label, matrix, hue, stem in [
        ("raw", raw, "sample_group", "pca_raw"),
        (
            "final_normics_limma",
            final,
            "acquisition_batch",
            "pca_final_normics_limma",
        ),
        ("final_limma_997", final_997, "MSI_final", "pca_final_limma_by_msi"),
    ]:
        coords, variance = run_pca(matrix, top_n)
        write_tsv(
            coords.reset_index().rename(columns={"index": "sample"}),
            out_dir / f"pca_{label}_coordinates.tsv",
        )
        write_tsv(variance, out_dir / f"pca_{label}_variance.tsv")
        plot = coords.reset_index().rename(columns={"index": "sample"})
        plot = plot.merge(
            meta[["sample", "acquisition_batch", "sample_group", "MSI_final"]],
            on="sample",
            how="left",
        )
        if hue == "MSI_final":
            # Exact counts in the legend, matching the convention used by the
            # Kaplan-Meier panels.
            counts = plot[hue].dropna().astype(str).value_counts()
            plot[hue] = plot[hue].map(
                lambda v: f"{v} (n={counts[str(v)]:,})" if pd.notna(v) else v
            )
            palette = {
                f"{level} (n={count:,})": MSI_STATUS_COLORS[level]
                for level, count in counts.items()
            }
        else:
            palette = make_color_map(plot[hue].dropna().astype(str).unique())
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        sns.scatterplot(
            plot,
            x="PC1",
            y="PC2",
            hue=hue,
            palette=palette,
            s=12,
            linewidth=0,
            alpha=0.8,
            ax=ax,
        )
        ax.set_xlabel(f"PC1 ({variance.loc[0, 'explained_variance_ratio'] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({variance.loc[1, 'explained_variance_ratio'] * 100:.1f}%)")
        clean_ax(ax)
        if hue == "MSI_final":
            place_legend(ax, mode="inside", title="Microsatellite status")
        else:
            place_legend(ax, mode="outside", title=hue)
        savefig(fig, fig_dir / f"{stem}.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    final_long = (
        final.stack(future_stack=True)
        .dropna()
        .sample(min(250000, int(final.notna().sum().sum())), random_state=0)
    )
    raw_long = (
        raw.stack(future_stack=True)
        .dropna()
        .sample(min(250000, int(raw.notna().sum().sum())), random_state=0)
    )
    density = pd.concat(
        [
            pd.DataFrame(
                {
                    "abundance": np.log2(raw_long.to_numpy()),
                    "matrix": "raw log2 quantity",
                }
            ),
            pd.DataFrame(
                {"abundance": final_long.to_numpy(), "matrix": "final Normics+limma"}
            ),
        ],
        ignore_index=True,
    )
    sns.kdeplot(density, x="abundance", hue="matrix", common_norm=False, ax=ax)
    ax.set_xlabel("Abundance")
    ax.set_ylabel("Density")
    clean_ax(ax)
    place_legend(ax, mode="inside", title=None)
    savefig(fig, fig_dir / "abundance_distribution_raw_vs_final.png")


def write_report(
    args: argparse.Namespace,
    manifest: dict,
    sample_stats: pd.DataFrame,
    protein_stats: pd.DataFrame,
) -> None:
    raw_s = sample_stats[sample_stats["matrix"].eq("raw reshaped")]
    final_s = sample_stats[sample_stats["matrix"].eq("final Normics+limma")]
    raw_p = protein_stats[protein_stats["matrix"].eq("raw reshaped")]
    final_p = protein_stats[protein_stats["matrix"].eq("final Normics+limma")]
    lines = [
        "# Raw Data and Preprocessing QC",
        "",
        "This report audits the current R14 Goettingen extraction before downstream MuVI analysis.",
        "",
        "## Source",
        "",
        f"- Raw long-format source: `{manifest.get('source_tsv', 'NA')}`",
        f"- Reshaped raw matrix: `{args.normics_dir / 'intermediate/r14_goettingen_raw_long_reshaped_samples_by_protein.csv'}`",
        f"- Final Normics+limma matrix: `{args.normics_dir / 'final/normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv'}`",
        f"- BID patient samples: {manifest.get('n_bid_samples', 'NA')}",
        f"- Total raw samples/runs retained in reshaped matrix: {manifest.get('n_samples', 'NA')}",
        f"- Raw proteins: {manifest.get('n_proteins', 'NA')}",
        f"- Missing BID numbers 1-1000: {manifest.get('missing_bid_numbers_1_to_1000', [])}",
        "",
        "## QC Summary",
        "",
        "| Matrix | Samples | Proteins | Median sample missing fraction | Median protein missing fraction |",
        "|---|---:|---:|---:|---:|",
        f"| Raw reshaped | {raw_s.shape[0]} | {raw_p.shape[0]} | {raw_s['missing_fraction'].median():.3f} | {raw_p['missing_fraction'].median():.3f} |",
        f"| Final Normics+limma | {final_s.shape[0]} | {final_p.shape[0]} | {final_s['missing_fraction'].median():.3f} | {final_p['missing_fraction'].median():.3f} |",
        "",
        "## Preprocessing Interpretation",
        "",
        "Raw `pg_quantity` values were pivoted into a samples-by-protein matrix. BID patient samples were then processed through Normics/VSN normalization and limma plate-batch correction. The final matrix used downstream contains only the 997 BID samples (BID0751 and BID0762 removed as outliers) and is the input to the MuVI handoff object.",
        "",
        "The QC figures check missingness, detected-protein counts, abundance distributions, and low-dimensional structure before and after normalization/batch correction. These plots are exploratory diagnostics; they do not by themselves define sample exclusion rules.",
        "",
        "## Figures",
        "",
        "The current manuscript-facing QC figures are generated by `python -m cancerscout.run_raw_qc` and `python -m cancerscout.run_quality_control_figures`:",
        "",
        f"- `{args.figures_dir / RAW_SAMPLE_COUNTS_FIGURE}`",
        f"- `{args.figures_dir / RAW_DETECTED_PROTEINS_FIGURE}`",
        f"- `{args.figures_dir / RAW_DETECTED_PROTEINS_FILTERED_FIGURE}`",
        f"- `{args.figures_dir / 'panel_outlier_pca_raw.png'}`",
        f"- `{args.figures_dir / 'pca_final_limma_by_msi.png'}`",
        f"- `{args.figures_dir / 'qc_panel_preprocessing.png'}`",
    ]
    save_text("\n".join(lines) + "\n", REPORTS_DIR / "quality_control.md")


def main() -> None:
    args = parse_args()
    set_house_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = (
        args.normics_dir
        / "intermediate/r14_goettingen_raw_long_reshaped_samples_by_protein_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    raw, raw_annotations = read_raw_matrix(
        args.normics_dir
        / "intermediate/r14_goettingen_raw_long_reshaped_samples_by_protein.csv"
    )
    final = read_final_matrix(
        args.normics_dir
        / "final/normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"
    )
    # The analysable cohort after outlier exclusion; `final` above still carries
    # the two high-missingness PCA outliers that the processed matrix drops.
    final_997 = read_final_matrix(args.analysable_matrix)
    meta = pd.read_csv(args.normics_dir / "intermediate/sample_metadata_selected.csv")

    sample_stats = pd.concat(
        [
            sample_qc(raw, "raw reshaped", meta),
            sample_qc(final, "final Normics+limma", meta),
        ],
        ignore_index=True,
    )
    protein_stats = pd.concat(
        [protein_qc(raw, "raw reshaped"), protein_qc(final, "final Normics+limma")],
        ignore_index=True,
    )
    write_tsv(sample_stats, args.out_dir / "sample_qc_metrics.tsv")
    write_tsv(protein_stats, args.out_dir / "protein_qc_metrics.tsv")
    write_tsv(raw_annotations, args.out_dir / "raw_matrix_annotation_rows.tsv")

    summary = pd.DataFrame(
        [
            {
                "matrix": label,
                "n_samples": int(sub_s.shape[0]),
                "n_proteins": int(sub_p.shape[0]),
                "median_sample_missing_fraction": float(
                    sub_s["missing_fraction"].median()
                ),
                "median_protein_missing_fraction": float(
                    sub_p["missing_fraction"].median()
                ),
                "median_detected_proteins_per_sample": float(
                    sub_s["n_detected"].median()
                ),
            }
            for label, sub_s, sub_p in [
                (
                    "raw reshaped",
                    sample_stats[sample_stats["matrix"].eq("raw reshaped")],
                    protein_stats[protein_stats["matrix"].eq("raw reshaped")],
                ),
                (
                    "final Normics+limma",
                    sample_stats[sample_stats["matrix"].eq("final Normics+limma")],
                    protein_stats[protein_stats["matrix"].eq("final Normics+limma")],
                ),
            ]
        ]
    )
    write_tsv(summary, args.out_dir / "qc_summary.tsv")
    plot_raw_basic_qc(
        sample_stats[sample_stats["matrix"].eq("raw reshaped")],
        protein_stats[protein_stats["matrix"].eq("raw reshaped")],
        args.out_dir,
        args.figures_dir,
    )
    plot_qc(
        sample_stats,
        protein_stats,
        raw,
        final,
        final_997,
        meta,
        args.out_dir,
        args.figures_dir,
        args.top_variable_proteins,
    )
    write_report(args, manifest, sample_stats, protein_stats)
    (args.out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n"
    )
    print(f"Wrote QC outputs to {args.out_dir.resolve()}")
    print(f"Wrote QC figures to {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
