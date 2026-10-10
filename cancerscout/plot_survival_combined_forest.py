"""Render the combined overall-survival Cox forest from saved result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cancerscout.paths import FIGURES_DIR, RESULTS_DIR, SURVIVAL_RESULTS_DIR
from cancerscout.plot_style import export_style
from cancerscout.plotting import run_r_figure


DEFAULT_CLUSTER_RESULTS = SURVIVAL_RESULTS_DIR / "cluster_survival_cox_models.tsv"
DEFAULT_CLINICAL_RESULTS = (
    SURVIVAL_RESULTS_DIR / "clinical_covariates_adjusted_cox_forest.tsv"
)
DEFAULT_COMBINED_RESULTS = SURVIVAL_RESULTS_DIR / "combined_os_cox_forest.tsv"
DEFAULT_OUTPUT = FIGURES_DIR / "survival" / "cluster_survival_adjusted_cox_forest.png"


def build_combined_forest(
    cluster_results_path: Path, clinical_results_path: Path
) -> pd.DataFrame:
    """Combine the saved OS cluster and clinical Cox rows for the legacy view."""
    cluster_results = pd.read_csv(cluster_results_path, sep="\t")
    clinical_results = pd.read_csv(clinical_results_path, sep="\t")

    cluster = cluster_results.loc[
        cluster_results["endpoint"].eq("OS")
        & cluster_results["model"].eq("cluster_vs_rest_adjusted")
        & cluster_results["cluster"].notna()
    ].copy()
    clinical = clinical_results.loc[
        clinical_results["endpoint"].eq("OS")
        & clinical_results["model"].eq("clinical_covariates_adjusted")
    ].copy()

    if len(cluster) != 6:
        raise ValueError(
            "Expected six OS cluster-vs-rest rows, "
            f"found {len(cluster)} in {cluster_results_path}"
        )
    if len(clinical) != 6:
        raise ValueError(
            "Expected six OS clinical-covariate rows, "
            f"found {len(clinical)} in {clinical_results_path}"
        )

    cluster["source_model"] = cluster["model"]
    cluster["term_kind"] = "cluster"
    cluster["cluster"] = pd.to_numeric(cluster["cluster"]).astype("Int64").astype(str)
    cluster["row_label"] = cluster["cluster_label"]

    clinical["source_model"] = clinical["model"]
    clinical["term_kind"] = "covariate"
    clinical["cluster"] = pd.Series(pd.NA, index=clinical.index, dtype="object")
    clinical["row_label"] = clinical["plot_label"]

    combined = pd.concat([cluster, clinical], ignore_index=True, sort=False)
    combined["model"] = "combined_os"
    combined = combined.sort_values("hazard_ratio", kind="stable").reset_index(
        drop=True
    )

    required = {
        "endpoint",
        "model",
        "term",
        "hazard_ratio",
        "ci_lower",
        "ci_upper",
        "p",
        "term_kind",
        "row_label",
    }
    missing = required.difference(combined.columns)
    if missing:
        raise ValueError(f"Combined Cox table is missing columns: {sorted(missing)}")
    return combined


def render_combined_os_forest(
    cluster_results_path: Path,
    clinical_results_path: Path,
    combined_results_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Write the combined provenance table and render PNG/PDF through the R style."""
    combined = build_combined_forest(cluster_results_path, clinical_results_path)
    combined_results_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_results_path, sep="\t", index=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_style(RESULTS_DIR / "plot_style")
    run_r_figure(
        "cox_forest_plot.R",
        [
            "--tsv",
            str(combined_results_path),
            "--model",
            "combined_os",
            "--endpoint",
            "OS",
            "--row-col",
            "row_label",
            "--output",
            str(output_path),
        ],
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the combined OS Cox forest from saved tables."
    )
    parser.add_argument("--cluster-results", type=Path, default=DEFAULT_CLUSTER_RESULTS)
    parser.add_argument(
        "--clinical-results", type=Path, default=DEFAULT_CLINICAL_RESULTS
    )
    parser.add_argument(
        "--combined-results", type=Path, default=DEFAULT_COMBINED_RESULTS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = render_combined_os_forest(
        args.cluster_results,
        args.clinical_results,
        args.combined_results,
        args.output,
    )
    print(f"Wrote {len(combined)} combined OS Cox rows to {args.combined_results}")
    print(f"Wrote combined OS forest to {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
