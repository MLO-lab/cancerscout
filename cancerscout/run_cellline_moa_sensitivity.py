"""Compare CRC65 predicted drug sensitivities by CancerScout state.

Frejno Table EV3A contains predicted standardized AUC values; lower values
mean greater predicted sensitivity. Table EV6B supplies semicolon-separated
target/activity annotations. This stage uses the unified CRC65 assignments,
collapses dataset-specific drug models to compounds, then compounds to
mechanisms, and tests C1-C4 against the remaining classified cell lines.

Usage:
    conda run -n proj python -m cancerscout.run_cellline_moa_sensitivity \
        --frejno-zip data/external/new_samples/crc_drug.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from cancerscout.io import write_tsv
from cancerscout.paths import (
    CELLLINE_MOA_RESULTS_DIR,
    EXTERNAL_TRANSFER_RESULTS_DIR,
    FIGURES_DIR,
    FREJNO_DRUG_ZIP,
)
from cancerscout.plot_style import set_house_style
from cancerscout.plotting import complex_heatmap
from cancerscout.run_pharmacoproteomics import (
    FIGURE_SUFFIX_RE,
    PHARMA_ROW_LABEL_MM,
    collapse_predicted_auc,
    drug_model_and_mechanism_tables,
    load_auc_predictions,
    load_drug_annotations,
    normalize_compound,
    open_ev_workbooks,
)


CLUSTERS_TO_TEST = ["1", "2", "3", "4"]
UNKNOWN_CLUSTER = "Unknown"
DEFAULT_FIGURES_DIR = FIGURES_DIR / "cellline_moa_sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frejno-zip", type=Path, default=FREJNO_DRUG_ZIP)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=EXTERNAL_TRANSFER_RESULTS_DIR / "crc65_assignments.tsv",
    )
    parser.add_argument("--out-dir", type=Path, default=CELLLINE_MOA_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--min-compounds", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--top-n-compounds",
        type=int,
        default=3,
        help="Top-N sensitive and resistant compounds per cluster for the drug-level panel.",
    )
    parser.add_argument(
        "--compound-rank-by",
        choices=("fdr", "delta"),
        default="fdr",
        help=(
            "Rank compounds by BH q-value ('fdr', the default) or by effect size "
            "('delta', matching the mechanism panel)."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw the mechanism and compound heatmaps from saved rankings.",
    )
    return parser.parse_args()


def join_audits(
    assignments: pd.DataFrame,
    auc: pd.DataFrame,
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment_ids = set(assignments["sample_id"].astype(str))
    ev3a_ids = set(auc.columns.astype(str))
    sample_rows = []
    cluster_column = (
        "predicted_cluster_thresholded"
        if "predicted_cluster_thresholded" in assignments
        else "predicted_cluster"
    )
    cluster_lookup = assignments.set_index("sample_id")[cluster_column].astype(str)
    for sample_id in sorted(assignment_ids | ev3a_ids):
        in_assignments = sample_id in assignment_ids
        in_ev3a = sample_id in ev3a_ids
        sample_rows.append(
            {
                "sample_id": sample_id,
                "in_unified_crc65_assignments": in_assignments,
                "in_ev3a": in_ev3a,
                "included": in_assignments and in_ev3a,
                "predicted_cluster": cluster_lookup.get(sample_id, np.nan),
                "reason": (
                    "included"
                    if in_assignments and in_ev3a
                    else "crc65_without_ev3a_prediction"
                    if in_assignments
                    else "ev3a_non_crc65_sample"
                ),
            }
        )

    auc_drugs = set(auc.index.astype(str))
    annotation_drugs = set(annotations["drug"].astype(str))
    drug_rows = []
    for drug in sorted(auc_drugs | annotation_drugs):
        in_ev3a = drug in auc_drugs
        in_ev6b = drug in annotation_drugs
        compound, dataset = normalize_compound(drug)
        drug_rows.append(
            {
                "drug_model": drug,
                "compound": compound,
                "source_dataset": dataset,
                "in_ev3a": in_ev3a,
                "in_ev6b": in_ev6b,
                "included": in_ev3a and in_ev6b,
                "reason": (
                    "included"
                    if in_ev3a and in_ev6b
                    else "missing_ev6b_annotation"
                    if in_ev3a
                    else "missing_ev3a_prediction"
                ),
            }
        )
    return pd.DataFrame(sample_rows), pd.DataFrame(drug_rows)


def cluster_mechanism_tests(
    moa_scores: pd.DataFrame,
    assignments: pd.DataFrame,
    mechanism_coverage: pd.DataFrame,
) -> pd.DataFrame:
    sample_cluster = assignments.set_index("sample_id")["predicted_cluster"].astype(str)
    scored = moa_scores.copy()
    scored["predicted_cluster"] = scored["sample_id"].map(sample_cluster)
    scored = scored.dropna(subset=["predicted_cluster"])
    rows = []
    for cluster in CLUSTERS_TO_TEST:
        for mechanism, sub in scored.groupby("mechanism", sort=True):
            values = pd.to_numeric(sub["predicted_standardized_auc"], errors="coerce")
            in_cluster = values[sub["predicted_cluster"].eq(cluster)].dropna()
            rest = values[~sub["predicted_cluster"].eq(cluster)].dropna()
            status = (
                "ok"
                if len(in_cluster) >= 3 and len(rest) >= 3
                else "too_few_cell_lines"
            )
            if status == "ok":
                statistic, p_value = mannwhitneyu(
                    in_cluster, rest, alternative="two-sided"
                )
            else:
                statistic, p_value = np.nan, np.nan
            median_cluster = in_cluster.median() if len(in_cluster) else np.nan
            median_rest = rest.median() if len(rest) else np.nan
            rows.append(
                {
                    "cluster": cluster,
                    "mechanism": mechanism,
                    "n_cluster": len(in_cluster),
                    "n_rest": len(rest),
                    "median_auc_cluster": median_cluster,
                    "median_auc_rest": median_rest,
                    "mean_auc_cluster": in_cluster.mean()
                    if len(in_cluster)
                    else np.nan,
                    "mean_auc_rest": rest.mean() if len(rest) else np.nan,
                    "delta_auc_cluster_minus_rest": median_cluster - median_rest,
                    "mann_whitney_u": statistic,
                    "p_value": p_value,
                    "status": status,
                    "effect_interpretation": (
                        "predicted_sensitivity"
                        if pd.notna(median_cluster)
                        and pd.notna(median_rest)
                        and median_cluster < median_rest
                        else "predicted_resistance"
                        if pd.notna(median_cluster)
                        and pd.notna(median_rest)
                        and median_cluster > median_rest
                        else "no_difference"
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_bh_global"] = np.nan
    tested = out["p_value"].notna()
    if tested.any():
        out.loc[tested, "fdr_bh_global"] = multipletests(
            out.loc[tested, "p_value"], method="fdr_bh"
        )[1]
    out = out.merge(mechanism_coverage, on="mechanism", how="left")
    return out.sort_values(["cluster", "delta_auc_cluster_minus_rest", "fdr_bh_global"])


def cluster_compound_tests(
    compound_scores: pd.DataFrame, assignments: pd.DataFrame
) -> pd.DataFrame:
    """Per-compound cluster-vs-rest tests, mirroring the mechanism-level analysis.

    Kept in a BH family of its own (``fdr_bh_compound``). Pooling the compound
    tests with the mechanism tests would inflate every mechanism q-value and
    silently change the mechanism panel.
    """
    sample_cluster = assignments.set_index("sample_id")["predicted_cluster"].astype(str)
    scored = compound_scores.copy()
    scored["predicted_cluster"] = scored["sample_id"].map(sample_cluster)
    scored = scored.dropna(subset=["predicted_cluster"])
    rows = []
    for cluster in CLUSTERS_TO_TEST:
        for (compound_key, compound), sub in scored.groupby(
            ["compound_key", "compound"], sort=True
        ):
            values = pd.to_numeric(sub["predicted_standardized_auc"], errors="coerce")
            in_cluster = values[sub["predicted_cluster"].eq(cluster)].dropna()
            rest = values[~sub["predicted_cluster"].eq(cluster)].dropna()
            status = (
                "ok"
                if len(in_cluster) >= 3 and len(rest) >= 3
                else "too_few_cell_lines"
            )
            if status == "ok":
                statistic, p_value = mannwhitneyu(
                    in_cluster, rest, alternative="two-sided"
                )
            else:
                statistic, p_value = np.nan, np.nan
            median_cluster = in_cluster.median() if len(in_cluster) else np.nan
            median_rest = rest.median() if len(rest) else np.nan
            rows.append(
                {
                    "cluster": cluster,
                    "compound": compound,
                    "compound_key": compound_key,
                    "n_cluster": len(in_cluster),
                    "n_rest": len(rest),
                    "median_auc_cluster": median_cluster,
                    "median_auc_rest": median_rest,
                    "mean_auc_cluster": in_cluster.mean()
                    if len(in_cluster)
                    else np.nan,
                    "mean_auc_rest": rest.mean() if len(rest) else np.nan,
                    "delta_auc_cluster_minus_rest": median_cluster - median_rest,
                    "mann_whitney_u": statistic,
                    "p_value": p_value,
                    "status": status,
                    "effect_interpretation": (
                        "predicted_sensitivity"
                        if pd.notna(median_cluster)
                        and pd.notna(median_rest)
                        and median_cluster < median_rest
                        else "predicted_resistance"
                        if pd.notna(median_cluster)
                        and pd.notna(median_rest)
                        and median_cluster > median_rest
                        else "no_difference"
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out["fdr_bh_compound"] = np.nan
    tested = out["p_value"].notna()
    if tested.any():
        out.loc[tested, "fdr_bh_compound"] = multipletests(
            out.loc[tested, "p_value"], method="fdr_bh"
        )[1]
    return out.sort_values(
        ["cluster", "delta_auc_cluster_minus_rest", "fdr_bh_compound"]
    )


def select_top_compounds(
    rankings: pd.DataFrame, top_n: int, rank_by: str = "delta"
) -> pd.DataFrame:
    """Top-N most sensitive and most resistant compounds per cluster.

    ``rank_by="delta"`` matches the mechanism panel and surfaces the largest
    effects. ``rank_by="fdr"`` ranks by significance first; this controls display
    ordering and does not imply that any compound passes the global correction.
    """
    selected = []
    usable = rankings[rankings["status"].eq("ok")].copy()
    sort_sensitivity = (
        (["fdr_bh_compound", "delta_auc_cluster_minus_rest"], [True, True])
        if rank_by == "fdr"
        else (["delta_auc_cluster_minus_rest", "fdr_bh_compound"], [True, True])
    )
    sort_resistance = (
        (["fdr_bh_compound", "delta_auc_cluster_minus_rest"], [True, False])
        if rank_by == "fdr"
        else (["delta_auc_cluster_minus_rest", "fdr_bh_compound"], [False, True])
    )
    for cluster in CLUSTERS_TO_TEST:
        sub = usable[usable["cluster"].eq(cluster)]
        directions = [
            (
                "sensitivity",
                sub[sub["delta_auc_cluster_minus_rest"].lt(0)].sort_values(
                    sort_sensitivity[0], ascending=sort_sensitivity[1]
                ),
            ),
            (
                "resistance",
                sub[sub["delta_auc_cluster_minus_rest"].gt(0)].sort_values(
                    sort_resistance[0], ascending=sort_resistance[1]
                ),
            ),
        ]
        for direction, candidates in directions:
            chosen = candidates.head(top_n).copy()
            chosen["display_direction"] = direction
            chosen["display_rank"] = np.arange(1, len(chosen) + 1)
            selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def select_top_mechanisms(rankings: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = []
    usable = rankings[rankings["status"].eq("ok")].copy()
    for cluster in CLUSTERS_TO_TEST:
        sub = usable[usable["cluster"].eq(cluster)]
        directions = [
            (
                "sensitivity",
                sub[sub["delta_auc_cluster_minus_rest"].lt(0)].sort_values(
                    ["delta_auc_cluster_minus_rest", "fdr_bh_global"]
                ),
            ),
            (
                "resistance",
                sub[sub["delta_auc_cluster_minus_rest"].gt(0)].sort_values(
                    ["delta_auc_cluster_minus_rest", "fdr_bh_global"],
                    ascending=[False, True],
                ),
            ),
        ]
        for direction, candidates in directions:
            chosen = candidates.head(top_n).copy()
            chosen["display_direction"] = direction
            chosen["display_rank"] = np.arange(1, len(chosen) + 1)
            selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def fdr_stars(value: object) -> str:
    if not np.isfinite(value):
        return ""
    if value <= 0.001:
        return "***"
    if value <= 0.01:
        return "**"
    if value <= 0.05:
        return "*"
    return ""


def plot_heatmap(
    rankings: pd.DataFrame,
    selected: pd.DataFrame,
    output: Path,
    *,
    index_col: str = "mechanism",
    q_col: str = "fdr_bh_global",
) -> None:
    if selected.empty:
        raise ValueError(f"No {index_col} passed the top-direction selection")
    rankings = rankings.copy()
    selected = selected.copy()
    rankings["cluster"] = rankings["cluster"].astype(str)
    selected["cluster"] = selected["cluster"].astype(str)
    ordered = selected.sort_values(
        ["cluster", "display_direction", "display_rank"],
        key=lambda s: (
            s.map({"sensitivity": 0, "resistance": 1}).fillna(s)
            if s.name == "display_direction"
            else s
        ),
    ).reset_index(drop=True)
    order = ordered[index_col].tolist()
    # Keep one row for every selected cluster/direction entry, even when the
    # same mechanism or compound is selected in more than one cluster. A row is
    # labelled with the cluster whose top-N selected it, matching how the
    # pharmaco panels block their rows by dominant cluster.
    row_groups = ordered["cluster"].tolist()
    direction_arrows = (
        ordered["display_direction"]
        .map({"sensitivity": "↑", "resistance": "↓"})
        .fillna("")
    )
    row_labels = (direction_arrows + " " + ordered[index_col].astype(str)).tolist()
    matrix = rankings.pivot(
        index=index_col, columns="cluster", values="delta_auc_cluster_minus_rest"
    ).reindex(index=order, columns=CLUSTERS_TO_TEST)
    q_values = rankings.pivot(index=index_col, columns="cluster", values=q_col).reindex(
        index=order, columns=CLUSTERS_TO_TEST
    )
    stars = q_values.map(fdr_stars)
    matrix.columns = [f"C{c}" for c in matrix.columns]
    stars.columns = matrix.columns
    if any(FIGURE_SUFFIX_RE.search(str(label)) for label in matrix.index):
        raise ValueError(f"Dataset suffix remained in a {index_col} figure label")
    if not matrix.notna().any().any():
        raise ValueError(f"{index_col} heatmap contains no finite effect estimates")

    # Plot predicted sensitivity rather than raw delta AUC: lower AUC means more
    # sensitive, so negating puts "more sensitive" at the high end and lets the
    # shared red-is-high ramp apply unchanged. Cell colours are unchanged from
    # the previous inverted-ramp rendering; the printed scale flips sign.
    complex_heatmap(
        -matrix,
        output,
        legend_title="Predicted\nsensitivity\n(-\u0394AUC)",
        stars=stars,
        row_groups=row_groups,
        row_labels=row_labels,
        annotate="value_stars",
        row_label_max_mm=PHARMA_ROW_LABEL_MM,
    )


def render_saved(out_dir: Path, figures_dir: Path) -> None:
    plot_heatmap(
        pd.read_csv(out_dir / "cluster_moa_rankings.tsv", sep="\t"),
        pd.read_csv(out_dir / "top_moa_per_cluster.tsv", sep="\t"),
        figures_dir / "crc65_moa_sensitivity_heatmap.png",
    )
    compound_rankings = out_dir / "cluster_compound_rankings.tsv"
    if compound_rankings.exists():
        plot_heatmap(
            pd.read_csv(compound_rankings, sep="\t"),
            pd.read_csv(out_dir / "top_compounds_per_cluster.tsv", sep="\t"),
            figures_dir / "crc65_compound_sensitivity_heatmap.png",
            index_col="compound",
            q_col="fdr_bh_compound",
        )


def main() -> None:
    args = parse_args()
    set_house_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        render_saved(args.out_dir, args.figures_dir)
        print(
            f"Redrew CRC65 mechanism and compound heatmaps -> {args.figures_dir.resolve()}"
        )
        return

    missing = [
        path for path in (args.frejno_zip, args.assignments) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing CRC65 MOA input(s): " + ", ".join(map(str, missing))
        )

    assignments = pd.read_csv(args.assignments, sep="\t")
    required = {"sample_id", "predicted_cluster"}
    if missing_columns := sorted(required - set(assignments.columns)):
        raise KeyError(f"Assignments are missing columns: {missing_columns}")
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    assignments["predicted_cluster"] = assignments["predicted_cluster"].astype(str)
    if "predicted_cluster_thresholded" not in assignments:
        raise KeyError(
            "CRC65 assignments must contain calibrated "
            "predicted_cluster_thresholded labels"
        )
    assignments["predicted_cluster_thresholded"] = assignments[
        "predicted_cluster_thresholded"
    ].astype(str)
    if assignments["sample_id"].duplicated().any():
        raise ValueError(
            "Unified CRC65 assignments contain duplicate sample identifiers"
        )

    workbooks = open_ev_workbooks(args.frejno_zip)
    auc = load_auc_predictions(workbooks)
    annotations = load_drug_annotations(workbooks)
    auc.index = auc.index.astype(str)
    auc.columns = auc.columns.astype(str)

    sample_audit, drug_audit = join_audits(assignments, auc, annotations)
    sample_audit["analysis_included"] = sample_audit["included"] & sample_audit[
        "predicted_cluster"
    ].ne(UNKNOWN_CLUSTER)
    write_tsv(sample_audit, args.out_dir / "ev3a_sample_join_audit.tsv")
    write_tsv(drug_audit, args.out_dir / "ev3a_ev6b_drug_join_audit.tsv")
    if not drug_audit["included"].all():
        raise ValueError("Some EV3A/EV6B drug models were not accounted for")

    models, compound_moa, mechanism_coverage = drug_model_and_mechanism_tables(
        annotations
    )
    mechanism_coverage["eligible"] = mechanism_coverage["n_unique_compounds"].ge(
        args.min_compounds
    )
    eligible = set(mechanism_coverage.loc[mechanism_coverage["eligible"], "mechanism"])
    included_samples = sample_audit.loc[
        sample_audit["analysis_included"], "sample_id"
    ].tolist()
    analysis_assignments = assignments.loc[
        assignments["predicted_cluster_thresholded"].ne(UNKNOWN_CLUSTER)
    ].copy()
    analysis_assignments["predicted_cluster"] = analysis_assignments[
        "predicted_cluster_thresholded"
    ]
    compound_scores, moa_scores = collapse_predicted_auc(
        auc, models, compound_moa, included_samples, eligible
    )
    if compound_scores.duplicated(["sample_id", "compound_key"]).any():
        raise ValueError(
            "Compound collapse produced duplicate sample/compound rows; "
            "compound case variants were not canonicalized"
        )
    rankings = cluster_mechanism_tests(
        moa_scores, analysis_assignments, mechanism_coverage
    )
    selected = select_top_mechanisms(rankings, args.top_n)
    compound_rankings = cluster_compound_tests(compound_scores, analysis_assignments)
    selected_compounds = select_top_compounds(
        compound_rankings, args.top_n_compounds, args.compound_rank_by
    )

    write_tsv(models, args.out_dir / "drug_model_annotations.tsv")
    write_tsv(compound_moa, args.out_dir / "compound_mechanism_map.tsv")
    write_tsv(mechanism_coverage, args.out_dir / "mechanism_coverage.tsv")
    write_tsv(compound_scores, args.out_dir / "cellline_compound_auc.tsv")
    write_tsv(moa_scores, args.out_dir / "cellline_moa_auc.tsv")
    write_tsv(rankings, args.out_dir / "cluster_moa_rankings.tsv")
    write_tsv(selected, args.out_dir / "top_moa_per_cluster.tsv")
    write_tsv(compound_rankings, args.out_dir / "cluster_compound_rankings.tsv")
    write_tsv(selected_compounds, args.out_dir / "top_compounds_per_cluster.tsv")

    manifest = {
        "frejno_zip": str(args.frejno_zip),
        "assignments": str(args.assignments),
        "ev3a_semantics": "predicted standardized AUC; lower values indicate predicted sensitivity",
        "ev6b_annotation": "Target or activity split on semicolons",
        "n_crc65_assignments": len(assignments),
        "n_crc65_known_assignments": int(
            assignments["predicted_cluster_thresholded"].ne(UNKNOWN_CLUSTER).sum()
        ),
        "n_crc65_unknown_assignments": int(
            assignments["predicted_cluster_thresholded"].eq(UNKNOWN_CLUSTER).sum()
        ),
        "n_crc65_with_ev3a": len(included_samples),
        "n_crc65_known_with_ev3a": len(
            sample_audit.loc[sample_audit["analysis_included"]]
        ),
        "n_ev3a_non_crc65_samples": int(
            sample_audit["reason"].eq("ev3a_non_crc65_sample").sum()
        ),
        "n_drug_models": int(auc.index.nunique()),
        "n_compounds": int(models["compound_key"].nunique()),
        "n_eligible_mechanisms": len(eligible),
        "minimum_unique_compounds_per_mechanism": args.min_compounds,
        "clusters_tested": CLUSTERS_TO_TEST,
        "test": "two-sided Mann-Whitney U, each C1-C4 versus all other known CRC65 lines; Unknown excluded",
        "effect": "cluster median predicted standardized AUC minus rest median",
        "multiple_testing_mechanism": "one Benjamini-Hochberg family across all cluster-mechanism tests",
        "multiple_testing_compound": "one Benjamini-Hochberg family across all cluster-compound tests, separate from the mechanism family",
        "n_mechanism_tests": int(rankings["p_value"].notna().sum()),
        "n_compound_tests": int(compound_rankings["p_value"].notna().sum()),
        "n_selected_moa_rows": len(selected),
        "n_selected_compound_rows": len(selected_compounds),
        "top_n_per_direction": args.top_n,
        "top_n_compounds_per_direction": args.top_n_compounds,
        "compound_display_ranking": args.compound_rank_by,
        "figure_colour_direction": (
            "heatmaps display -delta_auc: positive/red means predicted greater "
            "sensitivity and negative/blue means predicted resistance; selection "
            "uses raw delta_auc"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    plot_heatmap(
        rankings,
        selected,
        args.figures_dir / "crc65_moa_sensitivity_heatmap.png",
    )
    plot_heatmap(
        compound_rankings,
        selected_compounds,
        args.figures_dir / "crc65_compound_sensitivity_heatmap.png",
        index_col="compound",
        q_col="fdr_bh_compound",
    )
    print(f"Wrote CRC65 mechanism and compound analysis -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
