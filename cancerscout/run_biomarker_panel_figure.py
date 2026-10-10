"""AUC vs panel size figure for the RFECV reduced biomarker panels.

Shows how classifier AUC degrades as the panel shrinks from the full RFECV
signature down to single proteins, one line per cluster.

Panels are nested top-N subsets of each cluster's RFECV signature, ordered by
mean |SHAP| of the full-signature model on the training+validation tumours. The
test split is used only to score the refitted panels.

Usage:
    python -m cancerscout.run_biomarker_panel_figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cancerscout.io import read_metadata, write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CLUSTER_MARKER_RESULTS_DIR,
    FIGURES_DIR,
    MODEL_SPLITS_DIR,
    RESULTS_DIR,
)
from cancerscout.plot_style import (
    CLUSTER_COLORS,
    font_pt,
    inches,
    set_house_style,
    savefig,
)
from cancerscout.run_cluster_marker_rfecv import (
    binary_metrics,
    compute_shap_summary,
    fit_preprocessor,
    load_proteome_by_gene,
    load_splits,
    prefilter_features,
    train_xgb,
    transform_df,
)

DEFAULT_MARKER_DIR = CLUSTER_MARKER_RESULTS_DIR
DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR
DEFAULT_SPLIT_DIR = MODEL_SPLITS_DIR
OUT_FIGURE = FIGURES_DIR / "cluster_interpretation" / "biomarker_panel_auc_vs_size.png"
PANEL_SIZES = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500]
# Same 56 mm height and line/marker weights as the classifier CV panels
# (plot_multiclass_consensus_classifier) it sits beside, 1.3x their width.
PANEL_FIGSIZE = (inches(78.0), inches(56.0))
PANEL_LINEWIDTH = 0.9
PANEL_MARKERSIZE = 2.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-dir", type=Path, default=DEFAULT_MARKER_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--out", type=Path, default=OUT_FIGURE)
    parser.add_argument(
        "--panel-sizes",
        default=",".join(map(str, PANEL_SIZES)),
        help="Comma-separated top-N panel sizes to evaluate.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip the reduced-panel classifier training; re-render the figure from the existing reduced_panel_summary.tsv.",
    )
    return parser.parse_args()


def parse_panel_sizes(text: str) -> list[int]:
    return sorted({int(item.strip()) for item in str(text).split(",") if item.strip()})


def best_params_by_cluster(metrics: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    out = {}
    test = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    for row in test.itertuples(index=False):
        params = getattr(row, "best_xgb_params", "{}")
        out[str(row.cluster)] = json.loads(params) if isinstance(params, str) else {}
    return out


def full_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics.loc[metrics["split"].astype(str).eq("test")].copy()
    test["cluster"] = test["cluster"].astype(str)
    cols = [
        "cluster",
        "n_features_selected",
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "f1",
    ]
    return test[cols].rename(
        columns={
            "n_features_selected": "full_signature_size",
            "roc_auc": "full_roc_auc",
            "average_precision": "full_average_precision",
            "balanced_accuracy": "full_balanced_accuracy",
            "f1": "full_f1",
        }
    )


def ordered_markers(
    rankings: pd.DataFrame, cluster: str, shap_train: pd.Series
) -> list[str]:
    sub = rankings.loc[rankings["cluster"].astype(str).eq(cluster)].copy()
    sub["mean_abs_shap_train"] = sub["gene_symbol"].astype(str).map(shap_train)
    return (
        sub.sort_values(
            [
                "mean_abs_shap_train",
                "xgb_gain_importance",
                "selection_frequency",
                "mean_rfecv_rank",
            ],
            ascending=[False, False, False, True],
            na_position="last",
        )["gene_symbol"]
        .drop_duplicates()
        .astype(str)
        .tolist()
    )


def prepare_model_input(
    results_dir: Path,
    split_dir: Path,
    run_config: dict[str, object],
) -> tuple[pd.DataFrame, pd.Series, object]:
    metadata = read_metadata(results_dir)
    metadata.index = metadata.index.astype(str)
    cluster_column = str(run_config.get("cluster_column", "leiden_cons"))
    labels = metadata[cluster_column].astype(str)

    x_gene, _ = load_proteome_by_gene(results_dir, run_config.get("mdata_path"))
    common = labels.index.intersection(x_gene.index.astype(str))
    labels = labels.loc[common]
    x_gene = x_gene.loc[common]
    splits = load_splits(split_dir, set(common))

    split_names = {sid: "train" for sid in splits.train}
    split_names.update({sid: "val" for sid in splits.val})
    split_names.update({sid: "test" for sid in splits.test})
    used_ids = [sid for sid in common if sid in split_names]
    x_gene = x_gene.loc[used_ids]
    labels = labels.loc[used_ids]

    x_pref, _, _ = prefilter_features(
        x_gene.loc[splits.train],
        x_gene,
        float(run_config.get("missing_threshold", 0.5)),
        int(run_config.get("max_features_for_rfecv", 1000)),
    )
    pipe = fit_preprocessor(x_pref.loc[splits.train])
    return transform_df(pipe, x_pref), labels, splits


def build_reduced_panel_summary(
    marker_dir: Path,
    results_dir: Path,
    split_dir: Path,
    panel_sizes: list[int],
) -> pd.DataFrame:
    run_config_path = marker_dir / "run_config.json"
    run_config = (
        json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    )
    metrics = pd.read_csv(marker_dir / "cluster_marker_model_metrics.tsv", sep="\t")
    metrics["cluster"] = metrics["cluster"].astype(str)
    rankings = pd.read_csv(marker_dir / "cluster_marker_feature_rankings.tsv", sep="\t")
    rankings["cluster"] = rankings["cluster"].astype(str)

    x_z, labels, splits = prepare_model_input(results_dir, split_dir, run_config)
    params = best_params_by_cluster(metrics)
    model_args = SimpleNamespace(
        seed=int(run_config.get("seed", 260510)),
        threads=int(run_config.get("threads", 8)),
    )
    final_train_ids = [
        sid for sid in list(splits.train) + list(splits.val) if sid in x_z.index
    ]
    test_ids = [sid for sid in splits.test if sid in x_z.index]

    rows = []
    order_rows = []
    for cluster in sorted(rankings["cluster"].unique(), key=lambda x: int(float(x))):
        y_train = (labels.loc[final_train_ids] == cluster).astype(int)
        y_test = (labels.loc[test_ids] == cluster).astype(int)
        # Order the signature without touching the test split: SHAP of the
        # full-signature model on the tumours it was fitted on.
        signature = (
            rankings.loc[rankings["cluster"].eq(cluster), "gene_symbol"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        full_model = train_xgb(
            x_z.loc[final_train_ids, signature],
            y_train,
            model_args,
            params=params.get(cluster, {}),
        )
        shap_summary, _ = compute_shap_summary(
            full_model, x_z.loc[final_train_ids, signature], cluster
        )
        if shap_summary is None:
            raise RuntimeError(f"SHAP is required to order the cluster {cluster} panel")
        shap_train = shap_summary.set_index("gene_symbol")["mean_abs_shap_test"]
        ordered = ordered_markers(rankings, cluster, shap_train)
        order_rows.extend(
            {
                "cluster": cluster,
                "rank": rank,
                "gene_symbol": gene,
                "mean_abs_shap_train": shap_train[gene],
            }
            for rank, gene in enumerate(ordered, start=1)
        )
        seen_sizes = set()
        for requested_size in panel_sizes:
            genes = ordered[: min(requested_size, len(ordered))]
            used_size = len(genes)
            if not genes or used_size in seen_sizes:
                continue
            seen_sizes.add(used_size)
            if y_train.sum() < 2 or y_test.nunique() < 2:
                continue
            model = train_xgb(
                x_z.loc[final_train_ids, genes],
                y_train,
                model_args,
                params=params.get(cluster, {}),
            )
            score = model.predict_proba(x_z.loc[test_ids, genes])[:, 1]
            row = binary_metrics(y_test, score)
            row.update(
                {
                    "cluster": cluster,
                    "panel_size_requested": requested_size,
                    "panel_size_used": used_size,
                }
            )
            rows.append(row)

    reduced = pd.DataFrame(rows)
    summary = reduced.merge(full_metrics(metrics), on="cluster", how="left")
    for metric in ["roc_auc", "average_precision", "balanced_accuracy", "f1"]:
        summary[f"delta_{metric}_vs_full"] = summary[metric] - summary[f"full_{metric}"]
        summary[f"fraction_{metric}_of_full"] = (
            summary[metric] / summary[f"full_{metric}"]
        )
    summary = summary.sort_values(["cluster", "panel_size_used"])

    write_tsv(reduced, marker_dir / "cluster_marker_reduced_panel_metrics.tsv")
    posthoc_dir = marker_dir / "posthoc"
    posthoc_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(summary, posthoc_dir / "reduced_panel_summary.tsv")
    write_tsv(pd.DataFrame(order_rows), posthoc_dir / "reduced_panel_protein_order.tsv")
    return summary


def _grouped_tick_positions(
    panel_sizes: list[int],
    group_thresholds: tuple[int, int] = (5, 100),
    intra_steps: tuple[float, float, float] = (0.2, 0.4, 0.8),
    inter_gap: float = 1.1,
) -> np.ndarray:
    """Place panel-size ticks in 3 tight groups: (=<5), (6..100), (>100).

    Within each group ticks are evenly spaced at ``intra_steps[group_idx]``.
    Between groups the visual gap is ``inter_gap`` (broken-axis markers are
    drawn by the caller). Returns positions aligned 1:1 with ``panel_sizes``.
    """
    low, high = group_thresholds
    positions: list[float] = []
    x = 0.0
    prev_group: int | None = None
    for size in panel_sizes:
        if size <= low:
            group = 0
        elif size <= high:
            group = 1
        else:
            group = 2
        if prev_group is None:
            pass
        elif group == prev_group:
            x += intra_steps[group]
        else:
            x += inter_gap
        positions.append(x)
        prev_group = group
    return np.asarray(positions, dtype=float)


def _break_marker_positions(
    panel_sizes: list[int],
    tick_positions: np.ndarray,
    group_thresholds: tuple[int, int] = (5, 100),
) -> list[float]:
    """X-coords (between two ticks) where a broken-axis marker should be drawn."""
    low, high = group_thresholds
    out: list[float] = []
    for i in range(1, len(panel_sizes)):
        prev_size, this_size = panel_sizes[i - 1], panel_sizes[i]
        prev_grp = 0 if prev_size <= low else 1 if prev_size <= high else 2
        this_grp = 0 if this_size <= low else 1 if this_size <= high else 2
        if prev_grp != this_grp:
            out.append((tick_positions[i - 1] + tick_positions[i]) / 2.0)
    return out


def plot_auc_vs_panel_size(df: pd.DataFrame, out: Path, panel_sizes: list[int]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    tick_positions = _grouped_tick_positions(panel_sizes)
    tick_sizes = np.asarray(panel_sizes, dtype=float)

    def x_position(values: pd.Series | list[float] | np.ndarray) -> np.ndarray:
        """Place requested panel sizes evenly, with full-size values interpolated."""
        values = np.asarray(values, dtype=float)
        return np.interp(np.log10(values), np.log10(tick_sizes), tick_positions)

    ax.axhline(0.9, color="0.6", lw=0.6, ls=(0, (2, 2)), zorder=0)
    for cid in sorted(df["cluster"].astype(str).unique(), key=lambda x: int(float(x))):
        sub = (
            df.loc[df["cluster"].astype(str).eq(cid)]
            .sort_values("panel_size_used")
            .drop_duplicates("panel_size_used")
        )
        color = CLUSTER_COLORS.get(cid, "#4C4C4C")
        full_auc = sub["full_roc_auc"].iloc[0]
        full_size = sub["full_signature_size"].iloc[0]

        compact = sub.loc[sub["panel_size_used"].le(50)]
        extended = sub.loc[sub["panel_size_used"].ge(50)]
        if not compact.empty:
            ax.plot(
                x_position(compact["panel_size_used"]),
                compact["roc_auc"],
                color=color,
                lw=PANEL_LINEWIDTH,
                marker="o",
                ms=PANEL_MARKERSIZE,
                label=f"C{cid}",
                clip_on=False,
            )
        if len(extended) > 1:
            ax.plot(
                x_position(extended["panel_size_used"]),
                extended["roc_auc"],
                color=color,
                lw=PANEL_LINEWIDTH,
                marker="o",
                ms=PANEL_MARKERSIZE,
                alpha=0.4,
                clip_on=False,
            )

        ax.plot(
            x_position([full_size])[0],
            full_auc,
            marker="D",
            ms=3.2,
            mec=color,
            mfc="white",
            mew=0.8,
            zorder=5,
            clip_on=False,
        )

    ax.set_xticks(tick_positions, [str(size) for size in panel_sizes])
    ax.set_xlabel("Panel size (proteins)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xlim(tick_positions[0] - 0.15, tick_positions[-1] + 0.2)
    ax.set_ylim(0.6, 1.01)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["0.6", "0.7", "0.8", "0.9", "1"])

    # broken-axis markers between group transitions, anchored to the bottom
    # spine using a blended transform (x in data coords, y in axes fraction).
    blended = ax.get_xaxis_transform()
    for x_break in _break_marker_positions(panel_sizes, tick_positions):
        ax.axvline(x_break, color="white", linewidth=3, zorder=4)
        for offset in (-0.04, 0.04):
            ax.plot(
                [x_break + offset - 0.04, x_break + offset + 0.04],
                [-0.012, 0.012],
                transform=blended,
                color="0.4",
                linewidth=0.7,
                clip_on=False,
                zorder=5,
            )

    handles, labels = ax.get_legend_handles_labels()
    extended_handle = Line2D(
        [0],
        [0],
        color="0.35",
        lw=PANEL_LINEWIDTH,
        alpha=0.4,
        marker="o",
        markersize=PANEL_MARKERSIZE,
    )
    full_handle = Line2D(
        [0],
        [0],
        marker="D",
        lw=0,
        mec="0.35",
        mfc="white",
        markersize=3.2,
        markeredgewidth=0.8,
    )
    reference_handle = Line2D([0], [0], color="0.6", lw=0.6, ls=(0, (2, 2)))
    # Same legend box as the CV ROC/PR/reliability panels.
    ax.legend(
        handles=handles + [extended_handle, full_handle, reference_handle],
        labels=labels + [">50 proteins", "Full RFECV panel", "AUC = 0.9"],
        loc="lower right",
        alignment="left",
        fontsize=font_pt("micro"),
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.2,
        borderaxespad=0.3,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )

    fig.tight_layout(pad=0.4)
    savefig(fig, out)


def main() -> None:
    args = parse_args()
    set_house_style()
    panel_sizes = parse_panel_sizes(args.panel_sizes)
    summary_path = args.marker_dir / "posthoc" / "reduced_panel_summary.tsv"
    if args.plots_only:
        if not summary_path.exists():
            raise FileNotFoundError(
                f"--plots-only requires existing {summary_path}; run without --plots-only first."
            )
        summary = pd.read_csv(summary_path, sep="\t")
    else:
        summary = build_reduced_panel_summary(
            args.marker_dir,
            args.results_dir,
            args.split_dir,
            panel_sizes,
        )
    plot_auc_vs_panel_size(summary, args.out, panel_sizes)
    print(f"Saved -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
