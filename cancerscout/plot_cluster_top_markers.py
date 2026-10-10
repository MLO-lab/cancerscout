"""Top-3 marker heatmaps for volcano and RFECV signatures.

Both heatmaps display the **same Welch t-test view** (cell colour =
``mean_diff`` / log2-FC, stars = volcano ``FDR``). They differ only in how
the top-N genes per cluster are picked:

- Volcano: rank by ``mean_diff`` per cluster, filter FDR < 0.05.
- RFECV up/down: rank by ``train_cluster_minus_rest_z`` per cluster, filter
  ``selection_frequency`` >= 0.3.

The RFECV input is the per-cluster RFECV ranking table outer-joined to the
volcano table so every (selected_gene, cluster) cell has a defined
``mean_diff`` and ``FDR``.

Usage:
    python -m cancerscout.plot_cluster_top_markers
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from cancerscout.paths import (
    CLUSTER_INTERPRETATION_RESULTS_DIR,
    CLUSTER_MARKER_RESULTS_DIR,
    FIGURES_DIR,
)
from cancerscout.plotting import top_markers_heatmap


VOLCANO_TSV = CLUSTER_INTERPRETATION_RESULTS_DIR / "volcano_marker_rankings.tsv"
RFECV_TSV = CLUSTER_MARKER_RESULTS_DIR / "cluster_marker_feature_rankings.tsv"
VOLCANO_FIG_DIR = FIGURES_DIR / "cluster_interpretation"
RFECV_FIG_DIR = FIGURES_DIR / "cluster_marker_rfecv_min50_strict10_uncapped_997"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--volcano-tsv", type=Path, default=VOLCANO_TSV)
    parser.add_argument("--rfecv-tsv", type=Path, default=RFECV_TSV)
    parser.add_argument("--volcano-fig-dir", type=Path, default=VOLCANO_FIG_DIR)
    parser.add_argument("--rfecv-fig-dir", type=Path, default=RFECV_FIG_DIR)
    parser.add_argument("--skip-volcano", action="store_true")
    parser.add_argument("--skip-rfecv", action="store_true")
    parser.add_argument(
        "--up-down-only",
        action="store_true",
        help="Only write the combined up/down volcano heatmap.",
    )
    return parser.parse_args()


def build_rfecv_merged_tsv(rfecv_tsv: Path, volcano_tsv: Path, out_path: Path) -> None:
    """Build a dense (gene × all clusters) table merging volcano + RFECV.

    The RFECV table is sparse: only (gene, cluster) pairs the per-cluster
    RFECV actually evaluated have rows. To get a dense heatmap we expand to
    the Cartesian product of (every RFECV gene) × (every cluster present in
    volcano), then left-join volcano's ``mean_diff`` / ``FDR`` (always
    defined) and RFECV's ``selection_frequency`` / ``train_cluster_minus_rest_z``
    / ``mean_abs_shap_test`` (NaN where RFECV did not evaluate).
    """
    rfecv = pd.read_csv(rfecv_tsv, sep="\t")
    volcano = pd.read_csv(volcano_tsv, sep="\t")
    rfecv["cluster"] = rfecv["cluster"].astype(str)
    volcano["cluster"] = volcano["cluster"].astype(str)
    rfecv["gene_symbol"] = rfecv["gene_symbol"].astype(str)
    volcano["gene_symbol"] = volcano["gene_symbol"].astype(str)

    rfecv_genes = sorted(rfecv["gene_symbol"].unique())
    clusters = sorted(
        volcano["cluster"].unique(), key=lambda x: int(x) if x.isdigit() else x
    )
    grid = pd.MultiIndex.from_product(
        [rfecv_genes, clusters], names=["gene_symbol", "cluster"]
    ).to_frame(index=False)

    rfecv_cols = [
        "cluster",
        "gene_symbol",
        "selection_frequency",
        "train_cluster_minus_rest_z",
        "mean_abs_shap_test",
    ]
    merged = grid.merge(
        volcano[["cluster", "gene_symbol", "mean_diff", "FDR"]],
        on=["cluster", "gene_symbol"],
        how="left",
    ).merge(rfecv[rfecv_cols], on=["cluster", "gene_symbol"], how="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    # Both heatmaps display the Welch t-test: cell color = mean_diff (log2 FC),
    # stars = volcano FDR. The two analyses differ only in how the per-cluster
    # top-N is selected.
    legend_title = "log2 FC"
    if not args.skip_volcano:
        for direction in () if args.up_down_only else ("up", "down"):
            top_markers_heatmap(
                input_tsv=args.volcano_tsv,
                value_col="mean_diff",
                sig_col="FDR",
                sig_mode="fdr",
                direction=direction,
                output=args.volcano_fig_dir
                / f"volcano_top{args.top_n}_{direction}_heatmap.png",
                top_n=args.top_n,
                legend_title=legend_title,
            )
        top_markers_heatmap(
            input_tsv=args.volcano_tsv,
            value_col="mean_diff",
            sig_col="FDR",
            sig_mode="fdr",
            direction="both",
            output=args.volcano_fig_dir
            / f"volcano_top{args.top_n}_up_down_heatmap.png",
            top_n=args.top_n,
            legend_title=legend_title,
        )
    if not args.skip_rfecv:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_tsv = Path(tmpdir) / "rfecv_with_volcano.tsv"
            build_rfecv_merged_tsv(args.rfecv_tsv, args.volcano_tsv, merged_tsv)
            for direction in ("up", "down"):
                top_markers_heatmap(
                    input_tsv=merged_tsv,
                    value_col="mean_diff",  # cell colour (volcano t-test mean diff)
                    selection_col="train_cluster_minus_rest_z",  # ranks candidates per cluster (RFECV signal)
                    filter_col="selection_frequency",  # gates eligibility per cluster (RFECV)
                    filter_mode="frequency",
                    sig_col="FDR",  # stars (volcano t-test)
                    sig_mode="fdr",
                    direction=direction,
                    output=args.rfecv_fig_dir
                    / f"rfecv_top{args.top_n}_{direction}_heatmap.png",
                    top_n=args.top_n,
                    tiebreak="mean_abs_shap_test",
                    legend_title=legend_title,
                )


if __name__ == "__main__":
    main()
