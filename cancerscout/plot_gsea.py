"""Plot side of the GSEA/ssGSEA pipeline.

Renders the manuscript-facing paired GSEA/ssGSEA figure and a title-free
heatmap of the top-N significant positive and negative non-redundant pathways
per cluster. **No compute** -- everything comes from the TSVs ``run_gsea.py``
wrote, so these views can be replotted without re-running the enrichment.

Usage:
    python -m cancerscout.plot_gsea \\
        --out-dir results/gsea/gobp --figures-dir figures/gsea/gobp
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cancerscout.paths import FIGURES_DIR, REPO_ROOT
from cancerscout.plot_style import set_house_style
from cancerscout.plotting import run_r_figure


COMPLEXHEATMAP_SCRIPT = REPO_ROOT / "r" / "gsea_combined_heatmap.R"
TOP_PATHWAYS_SCRIPT = REPO_ROOT / "r" / "gsea_top_pathways_heatmap.R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory containing TSVs from run_gsea.py.",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR / "gsea" / "reactome"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Significant positive and negative pathways per cluster.",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Only write the top pathway heatmap, not gsea_main_nes_heatmap.",
    )
    return parser.parse_args()


def combined_gsea_ssgsea_figure(
    results_dir: Path,
    fig_dir: Path,
    top_n: int,
) -> None:
    """Render the manuscript-facing GSEA NES heatmap with ComplexHeatmap."""
    out_path = fig_dir / "gsea_main_nes_heatmap.png"
    missing = [
        path
        for path in (
            results_dir / "gsea_nonredundant_significant_terms.tsv",
            results_dir / "gsea_prerank_all_terms.tsv",
            results_dir / "ssgsea_cluster_mean_shifts.tsv",
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing GSEA result tables for the main NES heatmap: "
            + ", ".join(str(path) for path in missing)
        )

    fig_dir.mkdir(parents=True, exist_ok=True)
    run_r_figure(COMPLEXHEATMAP_SCRIPT, [results_dir, out_path, top_n])


def top_pathway_figure(
    results_dir: Path,
    fig_dir: Path,
    top_n: int,
) -> None:
    """Render the title-free positive/negative GSEA pathway heatmap."""
    required = [
        results_dir / "gsea_nonredundant_significant_terms.tsv",
        results_dir / "gsea_prerank_all_terms.tsv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing GSEA result tables for the top pathway heatmap: "
            + ", ".join(str(path) for path in missing)
        )

    out_path = fig_dir / f"gsea_top{top_n}_up_down_heatmap.png"
    run_r_figure(TOP_PATHWAYS_SCRIPT, [results_dir, out_path, top_n])


def main() -> None:
    args = parse_args()
    set_house_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # Keep the manuscript-facing paired figure and add a clean standalone view
    # of the top positive/negative significant pathways per cluster.
    if not args.skip_combined:
        combined_gsea_ssgsea_figure(args.out_dir, args.figures_dir, args.top_n)
    top_pathway_figure(args.out_dir, args.figures_dir, args.top_n)
    print(f"Wrote pathway figures to {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
