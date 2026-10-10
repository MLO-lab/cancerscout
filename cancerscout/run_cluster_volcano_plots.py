"""One-vs-rest volcano plots per Leiden consensus cluster (normics+limma proteomics).

Usage:
    python -m cancerscout.run_cluster_volcano_plots [options]

Saves:
    results/cluster_interpretation/volcano_marker_rankings.tsv  — full differential table
    figures/cluster_interpretation/volcano/volcano_C<N>.png      — one per cluster
    figures/cluster_interpretation/volcano/volcano_grid_3x2.png  — combined C1-C6 grid
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D

from cancerscout.io import write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CLUSTER_INTERPRETATION_RESULTS_DIR,
    FIGURES_DIR,
    NORMALIZED_LIMMA_CSV,
    ORDERED_MDATA_H5MU,
)
from cancerscout.plot_style import (
    CLUSTER_COLORS,
    cluster_display_label,
    savefig,
    set_house_style,
)
from cancerscout.stats import differential_abundance

EXPR_CSV = NORMALIZED_LIMMA_CSV
MDATA_PATH = ORDERED_MDATA_H5MU
GRID_VOLCANO_NAME = "volcano_grid_3x2.png"


def _bid_from_run_id(run_id: str) -> str | None:
    m = re.search(r"BID(\d+)", run_id)
    return str(int(m.group(1))) if m else None


def load_expression() -> tuple[pd.DataFrame, pd.Series]:
    """Return (samples × proteins expr, uniprot→gene_symbol Series).

    Expr index matches metadata integer BID strings.
    Falls back to uniprot IDs as symbols if h5mu is unavailable.
    """
    raw = pd.read_csv(EXPR_CSV, index_col=0)  # proteins × run-IDs
    raw.columns = [_bid_from_run_id(c) for c in raw.columns]
    raw = raw.loc[:, [c for c in raw.columns if c is not None]]
    expr = raw.T  # samples × proteins

    gene_symbols = pd.Series(expr.columns, index=expr.columns, dtype=str)
    try:
        import mudata as mu

        mdata = mu.read_h5mu(str(MDATA_PATH))
        sym = mdata["Proteome"].var["gene_symbol"].astype(str)
        gene_symbols = sym.reindex(expr.columns).fillna(
            pd.Series(expr.columns, index=expr.columns, dtype=str)
        )
    except Exception:
        pass

    return expr, gene_symbols


def volcano_plot(
    marker_df: pd.DataFrame,
    cluster: str,
    gene_symbols: pd.Series,
    top_n: int = 10,
    fdr_alpha: float = 0.05,
    lfc_threshold: float = 0.5,
    figsize: tuple[int, int] = (6, 6),
    out_dir: Path | None = None,
) -> None:
    """Plot one-vs-rest volcano for a single cluster and save to out_dir."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_volcano(
        ax,
        marker_df=marker_df,
        cluster=cluster,
        gene_symbols=gene_symbols,
        top_n=top_n,
        fdr_alpha=fdr_alpha,
        lfc_threshold=lfc_threshold,
        show_legend=True,
        label_fontsize=7,
        title_fontsize=11,
    )
    if out_dir is not None:
        fname = out_dir / f"volcano_C{cluster}.png"
        savefig(fig, fname)
        print(f"  Saved {fname.name}")
    plt.close(fig)


def draw_volcano(
    ax: plt.Axes,
    marker_df: pd.DataFrame,
    cluster: str,
    gene_symbols: pd.Series,
    top_n: int = 10,
    fdr_alpha: float = 0.05,
    lfc_threshold: float = 0.5,
    show_legend: bool = True,
    label_fontsize: int = 7,
    title_fontsize: int = 11,
) -> None:
    """Draw one-vs-rest volcano for a single cluster on an existing axis."""
    color_cluster = CLUSTER_COLORS.get(cluster, "#d62728")
    label_cluster = cluster_display_label(cluster, include_id=True)

    df = marker_df.copy()
    # mean_diff = log2(cluster mean) – log2(rest mean), i.e. log2FC since data is log2
    df = df.rename(columns={"mean_diff": "log_fc", "p_value": "pval"})
    df["neglog10p"] = -np.log10(df["pval"].clip(lower=1e-300))
    df["sig"] = (df["FDR"] < fdr_alpha) & (df["log_fc"].abs() > lfc_threshold)

    # FDR threshold line: the largest raw p-value still passing FDR < fdr_alpha.
    # This correctly marks where FDR significance actually starts on the y-axis,
    # rather than the misleading raw p=0.05 convention.
    fdr_sig_pvals = df.loc[df["FDR"] < fdr_alpha, "pval"]
    fdr_line_y = (
        float(-np.log10(fdr_sig_pvals.max())) if not fdr_sig_pvals.empty else None
    )

    # Top hits: up to top_n per direction so neither side monopolises labels
    sig = df[df["sig"]]
    top_up = sig[sig["log_fc"] > 0].nlargest(top_n, "log_fc")
    top_dn = sig[sig["log_fc"] < 0].nsmallest(top_n, "log_fc")
    top_hits = pd.concat([top_up, top_dn])

    ns = df[~df["sig"]]
    ax.scatter(
        ns["log_fc"],
        ns["neglog10p"],
        color="lightgray",
        alpha=0.15,
        linewidths=0,
        s=12,
        rasterized=True,
    )

    up = df[df["sig"] & (df["log_fc"] > 0)]
    ax.scatter(
        up["log_fc"],
        up["neglog10p"],
        color=color_cluster,
        alpha=0.85,
        linewidths=0,
        s=18,
    )

    dn = df[df["sig"] & (df["log_fc"] < 0)]
    ax.scatter(
        dn["log_fc"],
        dn["neglog10p"],
        color="#888888",
        alpha=0.85,
        linewidths=0,
        s=18,
    )

    texts = []
    for _, r in top_hits.iterrows():
        sym = str(gene_symbols.get(r["protein"], r["protein"]))
        texts.append(ax.text(r["log_fc"], r["neglog10p"], sym, fontsize=label_fontsize))
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    xmax = df["log_fc"].abs().max() * 1.05
    ax.set_xlim(-xmax, xmax)

    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.axvline(lfc_threshold, color="black", ls=":", lw=0.6)
    ax.axvline(-lfc_threshold, color="black", ls=":", lw=0.6)
    if fdr_line_y is not None:
        ax.axhline(fdr_line_y, color="black", ls=":", lw=0.6)
        ax.text(
            0.02,
            fdr_line_y,
            f"\nFDR={fdr_alpha}",
            va="top",
            fontsize=label_fontsize,
            color="0.4",
            transform=ax.get_yaxis_transform(),
        )

    ax.set_xlabel(
        f"log₂FC ({label_cluster} vs rest)\n← Up in rest    Up in {label_cluster} →"
    )

    if show_legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color_cluster,
                markersize=7,
                label=f"Up in {label_cluster}",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#888888",
                markersize=7,
                label="Up in rest",
            ),
        ]
        ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="upper left")

    ax.set_ylabel("−log₁₀(p)")
    ax.set_title(f"{label_cluster} vs rest", fontsize=title_fontsize)

    ax.text(
        0.98,
        0.02,
        f"↑{len(up)} ↓{len(dn)} (FDR<{fdr_alpha}, |log₂FC|>{lfc_threshold})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=label_fontsize,
        color="0.4",
    )


def volcano_grid(
    marker_df: pd.DataFrame,
    clusters: list[str],
    gene_symbols: pd.Series,
    out_dir: Path,
    top_n: int = 6,
    fdr_alpha: float = 0.05,
    lfc_threshold: float = 0.5,
    out_name: str = GRID_VOLCANO_NAME,
) -> Path:
    """Save a 3x2 grid of one-vs-rest volcano plots."""
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.6), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax, cluster in zip(flat_axes, clusters):
        sub = marker_df[marker_df["cluster"].astype(str) == str(cluster)]
        if sub.empty:
            ax.axis("off")
            continue
        draw_volcano(
            ax,
            marker_df=sub,
            cluster=str(cluster),
            gene_symbols=gene_symbols,
            top_n=top_n,
            fdr_alpha=fdr_alpha,
            lfc_threshold=lfc_threshold,
            show_legend=False,
            label_fontsize=6,
            title_fontsize=10,
        )

    for ax in flat_axes[len(clusters) :]:
        ax.axis("off")

    out_path = out_dir / out_name
    savefig(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path.name}")
    return out_path


def gene_symbols_from_rankings(marker_df: pd.DataFrame) -> pd.Series:
    if {"protein", "gene_symbol"}.issubset(marker_df.columns):
        symbols = marker_df.drop_duplicates("protein").set_index("protein")[
            "gene_symbol"
        ]
        return symbols.astype(str)
    return pd.Series(
        marker_df["protein"].astype(str).unique(),
        index=marker_df["protein"].astype(str).unique(),
    )


def cluster_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    try:
        return (0, f"{int(float(text)):03d}")
    except Exception:
        return (1, text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ACTIVE_MODEL_DIR))
    parser.add_argument("--out-dir", default=str(CLUSTER_INTERPRETATION_RESULTS_DIR))
    parser.add_argument(
        "--figures-dir", default=str(FIGURES_DIR / "cluster_interpretation" / "volcano")
    )
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--lfc", type=float, default=0.5)
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N hits labelled per direction (up and down separately)",
    )
    parser.add_argument(
        "--grid-top-n",
        type=int,
        default=6,
        help="Top N hits labelled per direction in the combined 3x2 grid.",
    )
    parser.add_argument("--cluster-col", default="leiden_cons")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip differential abundance and regenerate plots from an existing rankings TSV.",
    )
    parser.add_argument(
        "--rankings-file",
        type=Path,
        default=CLUSTER_INTERPRETATION_RESULTS_DIR / "volcano_marker_rankings.tsv",
    )
    parser.add_argument("--grid-name", default=GRID_VOLCANO_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_house_style()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.figures_dir)
    out_dir = Path(args.out_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        print(f"Loading existing marker rankings from {args.rankings_file}...")
        marker_df = pd.read_csv(args.rankings_file, sep="\t")
        marker_df["cluster"] = marker_df["cluster"].astype(str)
        gene_symbols = gene_symbols_from_rankings(marker_df)
        clusters = sorted(marker_df["cluster"].unique(), key=cluster_sort_key)
    else:
        print("Loading expression matrix...")
        expr, gene_symbols = load_expression()
        print(f"  {expr.shape[0]} samples, {expr.shape[1]} proteins")

        meta = pd.read_csv(results_dir / "metadata.csv", index_col=0)
        meta.index = meta.index.astype(str)
        clusters = meta[args.cluster_col].astype(str)

        shared = expr.index.intersection(clusters.index)
        expr = expr.loc[shared]
        clusters = clusters.loc[shared]
        print(f"  {len(shared)} samples after alignment, {clusters.nunique()} clusters")

        print(
            "Running one-vs-rest differential abundance (Welch t-test, FDR per cluster)..."
        )
        marker_df = differential_abundance(expr, clusters, gene_symbols=gene_symbols)
        tsv_path = out_dir / "volcano_marker_rankings.tsv"
        write_tsv(marker_df, tsv_path)
        print(f"  Saved marker rankings → {tsv_path}")
        clusters = sorted(clusters.unique(), key=cluster_sort_key)

    print("Plotting volcanoes...")
    for cluster in clusters:
        sub = marker_df[marker_df["cluster"] == cluster]
        if sub.empty:
            continue
        volcano_plot(
            sub,
            cluster=cluster,
            gene_symbols=gene_symbols,
            top_n=args.top_n,
            fdr_alpha=args.fdr,
            lfc_threshold=args.lfc,
            out_dir=fig_dir,
        )

    print("Plotting combined 3x2 volcano grid...")
    volcano_grid(
        marker_df=marker_df,
        clusters=clusters,
        gene_symbols=gene_symbols,
        out_dir=fig_dir,
        top_n=args.grid_top_n,
        fdr_alpha=args.fdr,
        lfc_threshold=args.lfc,
        out_name=args.grid_name,
    )

    print(f"\nDone. Figures → {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
