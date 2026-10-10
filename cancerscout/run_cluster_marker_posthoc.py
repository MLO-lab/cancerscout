"""Post-hoc summaries for locked RFECV cluster marker signatures.

This script intentionally does not rerun RFECV, Optuna, or model training. It
only reads completed marker outputs and writes manuscript-facing summaries,
diagnostics, and compact figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cancerscout.io import load_downstream_data, read_metadata, write_tsv
from cancerscout.plot_style import (
    clean_ax,
    cluster_display_label,
    cluster_palette,
    savefig,
    set_house_style,
)
from cancerscout.run_cluster_marker_rfecv import fit_preprocessor, transform_df
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CLUSTER_MARKER_RESULTS_DIR,
    FIGURES_DIR,
    GSEA_RESULTS_DIR,
    RESULTS_DIR,
)


DEFAULT_MARKER_DIR = CLUSTER_MARKER_RESULTS_DIR
DEFAULT_FIGURES_DIR = FIGURES_DIR / CLUSTER_MARKER_RESULTS_DIR.name
DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize locked RFECV marker signatures without rerunning models."
    )
    parser.add_argument("--marker-dir", default=str(DEFAULT_MARKER_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--cluster-column", default="leiden_cons")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-n-supplement", type=int, default=50)
    parser.add_argument("--cluster6-label", default="6")
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def require_inputs(marker_dir: Path) -> dict[str, Path]:
    inputs = {
        "rankings": marker_dir / "cluster_marker_feature_rankings.tsv",
        "selected": marker_dir / "cluster_marker_selected_features.tsv",
        "reduced": marker_dir / "cluster_marker_reduced_panel_metrics.tsv",
        "metrics": marker_dir / "cluster_marker_model_metrics.tsv",
        "predictions": marker_dir / "cluster_marker_predictions.tsv",
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required RFECV outputs:\n" + "\n".join(missing)
        )
    return inputs


def fmt_p(value: object) -> str:
    try:
        value = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(value):
        return "NA"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{value:.3f}"


def sort_markers(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        c
        for c in [
            "cluster",
            "rank",
            "selection_frequency",
            "mean_abs_shap_test",
            "xgb_gain_importance",
            "mean_rfecv_rank",
            "mean_outer_fold_selection_fraction",
            "gene_symbol",
        ]
        if c in df.columns
    ]
    ascending = []
    for col in sort_cols:
        ascending.append(
            False
            if col
            in {
                "selection_frequency",
                "mean_abs_shap_test",
                "xgb_gain_importance",
                "mean_outer_fold_selection_fraction",
            }
            else True
        )
    return df.sort_values(sort_cols, ascending=ascending, na_position="last")


def make_supplement_table(
    rankings: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    selected = selected.copy()
    selected["cluster"] = selected["cluster"].astype(str)
    rankings = rankings.copy()
    rankings["cluster"] = rankings["cluster"].astype(str)
    table = selected.merge(rankings, on=["cluster", "gene_symbol"], how="left")
    cols = [
        "cluster",
        "rank",
        "gene_symbol",
        "source_protein_ids",
        "marker_direction",
        "selection_frequency",
        "mean_rfecv_rank",
        "mean_outer_fold_selection_fraction",
        "mean_abs_shap_test",
        "xgb_gain_importance",
        "single_feature_auc_test",
        "train_cluster_minus_rest_z",
    ]
    return sort_markers(table[[c for c in cols if c in table.columns]])


def summarize_reduced_panels(
    metrics: pd.DataFrame, reduced: pd.DataFrame
) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    base_cols = [
        "cluster",
        "n_features_selected",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "f1",
    ]
    base = test[base_cols].rename(
        columns={
            "n_features_selected": "full_signature_size",
            "roc_auc": "full_roc_auc",
            "average_precision": "full_average_precision",
            "balanced_accuracy": "full_balanced_accuracy",
            "f1": "full_f1",
        }
    )
    out = reduced.merge(base, on="cluster", how="left")
    for metric in ["roc_auc", "average_precision", "balanced_accuracy", "f1"]:
        out[f"delta_{metric}_vs_full"] = out[metric] - out[f"full_{metric}"]
        out[f"fraction_{metric}_of_full"] = out[metric] / out[f"full_{metric}"]
    return out.sort_values(["cluster", "panel_size_used"])


def plot_reduced_panel_delta(summary: pd.DataFrame, fig_dir: Path) -> None:
    long = summary.melt(
        id_vars=["cluster", "panel_size_used"],
        value_vars=[
            "delta_roc_auc_vs_full",
            "delta_average_precision_vs_full",
            "delta_balanced_accuracy_vs_full",
            "delta_f1_vs_full",
        ],
        var_name="metric",
        value_name="delta_vs_full",
    )
    long["metric"] = (
        long["metric"]
        .str.replace("delta_", "", regex=False)
        .str.replace("_vs_full", "", regex=False)
        .str.replace("_", " ")
    )
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    long["cluster_label"] = long["cluster"].astype(str).map(cluster_display_label)
    sns.lineplot(
        data=long,
        x="panel_size_used",
        y="delta_vs_full",
        hue="metric",
        style="cluster_label",
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.axhline(0, color="0.45", lw=0.8, ls="--")
    ax.set_xlabel("Top-N marker panel")
    ax.set_ylabel("Delta vs full signature")
    clean_ax(ax)
    ax.legend(
        frameon=False, fontsize=6, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    savefig(fig, fig_dir / "reduced_panel_delta_summary.png")


def top_marker_tables(
    supplement: pd.DataFrame, top_n: int, top_n_supplement: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Manuscript-facing top-marker summaries follow the same predictive-importance
    # ordering used by the combined heatmap. The full supplement still retains
    # consensus RFECV rank in the `rank` column.
    sort_cols = [
        c
        for c in [
            "mean_abs_shap_test",
            "xgb_gain_importance",
            "selection_frequency",
            "mean_rfecv_rank",
        ]
        if c in supplement.columns
    ]
    ascending = [False if c != "mean_rfecv_rank" else True for c in sort_cols]
    ordered = supplement.sort_values(
        ["cluster", *sort_cols], ascending=[True, *ascending], na_position="last"
    ).copy()
    ordered["predictive_summary_rank"] = ordered.groupby("cluster").cumcount() + 1
    top10 = ordered.groupby("cluster", group_keys=False).head(top_n).copy()
    top50 = ordered.groupby("cluster", group_keys=False).head(top_n_supplement).copy()
    return top10, top50


def signature_overlap(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = selected.copy()
    selected["cluster"] = selected["cluster"].astype(str)
    sets = {
        cluster: set(group["gene_symbol"].astype(str))
        for cluster, group in selected.groupby("cluster")
    }
    rows = []
    for a, set_a in sets.items():
        for b, set_b in sets.items():
            union = set_a | set_b
            rows.append(
                {
                    "cluster_a": a,
                    "cluster_b": b,
                    "n_a": len(set_a),
                    "n_b": len(set_b),
                    "overlap_n": len(set_a & set_b),
                    "jaccard": len(set_a & set_b) / len(union) if union else np.nan,
                    "fraction_a_overlapping_b": len(set_a & set_b) / len(set_a)
                    if set_a
                    else np.nan,
                    "fraction_b_overlapping_a": len(set_a & set_b) / len(set_b)
                    if set_b
                    else np.nan,
                }
            )
    marker_clusters = (
        selected.groupby("gene_symbol")["cluster"]
        .agg(
            lambda s: ";".join(
                sorted(set(map(str, s)), key=lambda x: int(x) if x.isdigit() else x)
            )
        )
        .reset_index()
    )
    marker_clusters["n_clusters"] = marker_clusters["cluster"].str.split(";").str.len()
    shared = marker_clusters.loc[marker_clusters["n_clusters"] > 1].sort_values(
        ["n_clusters", "gene_symbol"], ascending=[False, True]
    )
    return pd.DataFrame(rows), shared


def direction_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    return (
        rankings.groupby(["cluster", "marker_direction"], dropna=False)
        .size()
        .reset_index(name="n_markers")
        .merge(
            rankings.groupby("cluster").size().rename("total_markers").reset_index(),
            on="cluster",
            how="left",
        )
        .assign(fraction=lambda d: d["n_markers"] / d["total_markers"])
        .sort_values(["cluster", "marker_direction"])
    )


def plot_direction_summary(summary: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    plot = summary.copy()
    plot["cluster_label"] = plot["cluster"].astype(str).map(cluster_display_label)
    order = [
        cluster_display_label(c)
        for c in sorted(
            summary["cluster"].astype(str).unique(),
            key=lambda x: int(float(x))
            if str(x).replace(".", "", 1).isdigit()
            else str(x),
        )
    ]
    sns.barplot(
        data=plot,
        x="cluster_label",
        y="fraction",
        hue="marker_direction",
        order=order,
        palette="colorblind",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of selected markers")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    clean_ax(ax)
    ax.legend(frameon=False, title="")
    savefig(fig, fig_dir / "marker_direction_barplot.png")


def load_proteome_z(
    results_dir: Path, marker_genes: list[str], cluster_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    data = load_downstream_data(results_dir)
    if data.proteome is None:
        raise RuntimeError("Could not load current Proteome modality.")
    x = data.proteome.copy()
    x.index = x.index.astype(str)
    if data.gene_symbols is not None:
        genes = data.gene_symbols.reindex(x.columns)
        cleaned = []
        for col in x.columns:
            text = str(genes.loc[col]).strip() if pd.notna(genes.loc[col]) else str(col)
            if not text or text.lower() in {"nan", "none"}:
                text = str(col)
            cleaned.append(text.split(";")[0].split(",")[0].strip())
        x.columns = cleaned
        x = x.T.groupby(level=0).median().T
    marker_genes = [g for g in marker_genes if g in x.columns]
    metadata = read_metadata(results_dir)
    metadata.index = metadata.index.astype(str)
    labels = metadata[cluster_col].astype(str)
    common = x.index.intersection(labels.index)
    x = x.loc[common, marker_genes]
    labels = labels.loc[common]
    # Use all available samples for visualization/audit z-scores; model metrics remain unchanged.
    pipe = fit_preprocessor(x)
    return transform_df(pipe, x), labels


def cluster6_audit(
    predictions: pd.DataFrame,
    rankings: pd.DataFrame,
    results_dir: Path,
    cluster_col: str,
    cluster_label: str,
    fig_dir: Path,
    top_n: int,
) -> pd.DataFrame:
    pred = predictions.loc[
        predictions["cluster"].astype(str).eq(str(cluster_label))
    ].copy()
    pred["call_type"] = np.select(
        [
            pred["y_true"].eq(1) & pred["y_pred_0_5"].eq(1),
            pred["y_true"].eq(1) & pred["y_pred_0_5"].eq(0),
            pred["y_true"].eq(0) & pred["y_pred_0_5"].eq(1),
            pred["y_true"].eq(0) & pred["y_pred_0_5"].eq(0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="NA",
    )
    top = (
        rankings.loc[rankings["cluster"].astype(str).eq(str(cluster_label))]
        .head(top_n)["gene_symbol"]
        .astype(str)
        .tolist()
    )
    if top:
        x_z, _ = load_proteome_z(results_dir, top, cluster_col)
        shared_ids = pred["analysis_sample_id"].astype(str).tolist()
        available = [sid for sid in shared_ids if sid in x_z.index]
        for gene in top:
            if gene in x_z.columns:
                pred[f"{gene}_z"] = (
                    pred["analysis_sample_id"].astype(str).map(x_z.loc[available, gene])
                )
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    sns.stripplot(
        data=pred,
        x="split",
        y="y_score",
        hue="call_type",
        dodge=True,
        jitter=0.2,
        palette="colorblind",
        ax=ax,
    )
    ax.axhline(0.5, color="0.45", lw=0.8, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel(f"Predicted probability: {cluster_display_label(cluster_label)}")
    clean_ax(ax)
    ax.legend(frameon=False, title="", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(
        fig, fig_dir / f"cluster{cluster_label}_prediction_probability_stripplot.png"
    )
    return pred.sort_values(
        ["split", "call_type", "y_score"], ascending=[True, True, False]
    )


def first_existing(paths: list[str | Path]) -> Path | None:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def join_terms(df: pd.DataFrame, cluster: str, n: int = 3) -> str:
    if df.empty:
        return "NA"
    frame = df.loc[df["cluster"].astype(str).eq(str(cluster))].copy()
    if frame.empty:
        return "NA"
    if "NES" in frame.columns:
        frame = frame.reindex(frame["NES"].abs().sort_values(ascending=False).index)
    terms = (
        frame.head(n)
        .get("Term", pd.Series(dtype=str))
        .astype(str)
        .str.replace("_", " ", regex=False)
        .tolist()
    )
    return "; ".join(terms) if terms else "NA"


def biology_summary(rankings: pd.DataFrame, top_markers: pd.DataFrame) -> pd.DataFrame:
    clusters = sorted(
        rankings["cluster"].astype(str).unique(),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    gsea_path = first_existing(
        [
            GSEA_RESULTS_DIR / "reactome/gsea_top5_up_down_significant_terms.tsv",
            GSEA_RESULTS_DIR / "gobp/gsea_top5_up_down_significant_terms.tsv",
        ]
    )
    gsea = pd.read_csv(gsea_path, sep="\t") if gsea_path else pd.DataFrame()
    cptac_path = first_existing(
        [
            RESULTS_DIR
            / "cptac_classification/locked_leiden_cons_new_cptac_distribution_fraction.tsv"
        ]
    )
    cptac = pd.read_csv(cptac_path, sep="\t") if cptac_path else pd.DataFrame()
    msi_path = first_existing(
        [
            RESULTS_DIR
            / "cptac_classification/leiden_cluster_msi_comparison_locked_093414.tsv"
        ]
    )
    msi = pd.read_csv(msi_path, sep="\t") if msi_path else pd.DataFrame()
    drug_path = first_existing(
        [RESULTS_DIR / "pharmacoproteomics/cluster_drug_rankings.tsv"]
    )
    drugs = pd.read_csv(drug_path, sep="\t") if drug_path else pd.DataFrame()

    rows = []
    for cluster in clusters:
        top_cluster_markers = (
            top_markers.loc[top_markers["cluster"].astype(str).eq(cluster)]
            .sort_values(
                "predictive_summary_rank"
                if "predictive_summary_rank" in top_markers.columns
                else "rank"
            )["gene_symbol"]
            .astype(str)
            .tolist()
        )
        cptac_note = "NA"
        if not cptac.empty:
            hit = cptac.loc[cptac["leiden_cons"].astype(str).eq(cluster)]
            if not hit.empty:
                vals = (
                    hit.drop(columns=["leiden_cons"])
                    .iloc[0]
                    .sort_values(ascending=False)
                )
                cptac_note = "; ".join(
                    f"{idx}={val:.2f}" for idx, val in vals.head(3).items()
                )
        msi_note = "NA"
        if not msi.empty:
            hit = msi.loc[msi["cluster_label"].astype(str).eq(cluster)]
            if not hit.empty:
                r = hit.iloc[0]
                msi_note = f"MSI fraction={r.get('cluster_msi_fraction_precision', np.nan):.2f}; recall={r.get('msi_recall_sensitivity', np.nan):.2f}"
        drug_note = "NA"
        if not drugs.empty:
            hit = drugs.loc[
                drugs["cluster"].astype(str).eq(cluster)
                & drugs["predicted_direction"].astype(str).eq("predicted_sensitive")
            ].copy()
            if not hit.empty:
                hit = hit.sort_values(
                    "delta_auc_score_cluster_minus_rest", ascending=True
                ).head(3)
                drug_note = "; ".join(
                    f"{r.drug} ({r.get('target_or_activity', 'NA')})"
                    for _, r in hit.iterrows()
                )
        rows.append(
            {
                "cluster": cluster,
                "cluster_name": cluster_display_label(cluster),
                "top10_markers": "; ".join(top_cluster_markers),
                "dominant_marker_direction": rankings.loc[
                    rankings["cluster"].astype(str).eq(cluster), "marker_direction"
                ]
                .value_counts()
                .idxmax(),
                "pathway_themes": join_terms(gsea, cluster),
                "cptac_distribution": cptac_note,
                "msi_summary": msi_note,
                "top_predicted_sensitive_drugs": drug_note,
                "interpretation_flag": "exploratory_low_n"
                if cluster == "6"
                else "standard",
            }
        )
    return pd.DataFrame(rows)


def write_biology_report(
    summary: pd.DataFrame, marker_dir: Path, fig_dir: Path, path: Path
) -> None:
    lines = [
        "# Cluster Marker Biology Report",
        "",
        f"Locked marker directory: `{marker_dir}`",
        f"Figure directory: `{fig_dir}`",
        "",
        "This report summarizes RFECV-selected marker signatures. It is descriptive and should not be read as clinical diagnostic or treatment guidance.",
        "",
        "## Cluster Summaries",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"### Cluster {row['cluster']}: {row.get('cluster_name', cluster_display_label(row['cluster']))}",
                "",
                f"- Top RFECV markers: {row['top10_markers']}",
                f"- Dominant marker direction: {row['dominant_marker_direction']}",
                f"- Pathway themes: {row['pathway_themes']}",
                f"- CPTAC distribution: {row['cptac_distribution']}",
                f"- MSI summary: {row['msi_summary']}",
                f"- Pharmacoproteomic hypotheses: {row['top_predicted_sensitive_drugs']}",
                f"- Interpretation flag: {row['interpretation_flag']}",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    marker_dir = Path(args.marker_dir)
    fig_dir = Path(args.figures_dir)
    posthoc_dir = marker_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_house_style()

    inputs = require_inputs(marker_dir)
    rankings = read_tsv(inputs["rankings"])
    selected = read_tsv(inputs["selected"])
    reduced = read_tsv(inputs["reduced"])
    metrics = read_tsv(inputs["metrics"])
    predictions = read_tsv(inputs["predictions"])
    rankings["cluster"] = rankings["cluster"].astype(str)
    selected["cluster"] = selected["cluster"].astype(str)

    observed_sizes = (
        selected.groupby("cluster")["gene_symbol"].nunique().astype(int).to_dict()
    )
    print(f"Observed selected signature sizes: {observed_sizes}")

    supplement = make_supplement_table(rankings, selected)
    write_tsv(supplement, posthoc_dir / "supplement_cluster_marker_signatures.tsv")

    reduced_summary = summarize_reduced_panels(metrics, reduced)
    write_tsv(reduced_summary, posthoc_dir / "reduced_panel_summary.tsv")
    plot_reduced_panel_delta(reduced_summary, fig_dir)

    top10, top50 = top_marker_tables(supplement, args.top_n, args.top_n_supplement)
    write_tsv(top10, posthoc_dir / "top10_marker_table.tsv")
    write_tsv(top50, posthoc_dir / "top50_marker_table.tsv")

    cluster6 = cluster6_audit(
        predictions,
        rankings,
        Path(args.results_dir),
        args.cluster_column,
        args.cluster6_label,
        fig_dir,
        args.top_n,
    )
    write_tsv(cluster6, posthoc_dir / "cluster6_prediction_audit.tsv")

    overlap, shared = signature_overlap(selected)
    write_tsv(overlap, posthoc_dir / "signature_overlap_jaccard.tsv")
    write_tsv(shared, posthoc_dir / "shared_marker_table.tsv")

    directions = direction_summary(rankings)
    write_tsv(directions, posthoc_dir / "marker_direction_summary.tsv")
    plot_direction_summary(directions, fig_dir)

    bio = biology_summary(rankings, top10)
    write_tsv(bio, posthoc_dir / "cluster_marker_biology_summary.tsv")
    write_biology_report(
        bio, marker_dir, fig_dir, posthoc_dir / "cluster_marker_biology_report.md"
    )

    print(f"Wrote posthoc outputs to {posthoc_dir}")
    print(f"Wrote posthoc figures to {fig_dir}")


if __name__ == "__main__":
    main()
