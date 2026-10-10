"""Render figures for the cclf multi-class classifier.

Reads `results/multiclass_consensus_classifier/` and writes:

- `confusion_matrix_test.png` — compact counts + row-% matrix.
- `auroc_curves_test.png` — one-vs-rest AUROC curves on the untouched test set.
- `pr_curves_test.png` — one-vs-rest precision-recall curves on the untouched
  test set.
- `reliability_overlay_{uniform,quantile}_test.png` — overlaid one-vs-rest
  reliability curves on equal-width and equal-frequency bins.
- `reliability_diagrams_test.png` — one-vs-rest reliability diagrams for C1–C6.
- `reliability_top_label_test.png` — top-label reliability diagram.
- `shap_top3_up_heatmap.png` / `shap_top3_down_heatmap.png` — canonical
  ComplexHeatmap top-3 markers per cluster, ranked by SHAP signed mean.
- `volcano_selected_top3_up_heatmap.png` /
  `volcano_selected_top3_down_heatmap.png` — differential-abundance top-3
  markers among proteins exposed to the canonical multiclass ensemble.

Usage:
    conda run -n proj python -m cancerscout.plot_multiclass_consensus_classifier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

from cancerscout.io import write_tsv
from cancerscout.paths import (
    FIGURES_DIR,
    RESULTS_DIR,
)
from cancerscout.plot_style import (
    font_pt,
    inches,
    sequential_blue_cmap,
    CLUSTER_COLORS,
    savefig,
    set_house_style,
)
from cancerscout.plotting import top_markers_heatmap
from cancerscout.run_multiclass_consensus_classifier import calibration_report


DEFAULT_RESULTS_DIR = RESULTS_DIR / "multiclass_consensus_classifier"
DEFAULT_FIGURES_DIR = FIGURES_DIR / "multiclass_consensus_classifier"
DEFAULT_VOLCANO_TSV = (
    RESULTS_DIR / "cluster_interpretation" / "volcano_marker_rankings.tsv"
)
# Drawn at print size: three panels fit across a 183 mm double column, and
# every font comes from the house type tiers via set_house_style().
PANEL_FIGSIZE = (inches(60.0), inches(56.0))
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 260520
RELIABILITY_BINS = 10
CURVE_LW = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--output-suffix",
        default="test",
        help="Suffix for evaluation-specific outputs, e.g. 'test' or 'cv'.",
    )
    parser.add_argument(
        "--evaluation-label",
        default="held-out test split",
        help="Label used in evaluation-specific plot titles.",
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--volcano-tsv", type=Path, default=DEFAULT_VOLCANO_TSV)
    parser.add_argument("--skip-volcano-selected", action="store_true")
    return parser.parse_args()


def load_confusion_matrix(path: Path) -> pd.DataFrame:
    cm = pd.read_csv(path, sep="\t", index_col=0)
    cm.index = [str(i).removeprefix("true_") for i in cm.index]
    cm.columns = [str(c).removeprefix("pred_") for c in cm.columns]
    return cm.astype(int)


def plot_confusion(cm: pd.DataFrame, out_path: Path) -> None:
    cm = cm.loc[cm.sum(axis=1).gt(0)]
    row_sums = cm.sum(axis=1).replace(0, np.nan)
    cm_rownorm = cm.div(row_sums, axis=0).fillna(0.0)
    classes = cm.index.tolist()

    cmap = sequential_blue_cmap()
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    im = ax.imshow(
        cm_rownorm.values,
        aspect="equal",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="none",
    )
    tick_labels = ["Unknown" if c.casefold() == "unknown" else f"C{c}" for c in classes]
    ax.set_xticks(range(len(classes)), tick_labels)
    ax.set_yticks(range(len(classes)), tick_labels)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True cluster")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm.iloc[i, j])
            fraction = float(cm_rownorm.iloc[i, j])
            ax.text(
                j,
                i,
                f"{count}\n{fraction:.0%}" if count else "0",
                ha="center",
                va="center",
                fontsize=font_pt("annot"),
                linespacing=0.95,
                color="white" if fraction > 0.55 else "black",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, aspect=18, pad=0.03)
    cbar.set_label("Recall (row %)")
    cbar.set_ticks([0, 0.5, 1.0], labels=["0", "50", "100"])
    cbar.ax.tick_params(length=2)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout(pad=0.4)
    savefig(fig, out_path)


def ovr_scores(
    y_true: np.ndarray, prob: np.ndarray, classes: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class one-vs-rest AUROC and AP; NaN where a class is absent."""
    auroc, aupr = [], []
    for k, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        if np.unique(y_bin).size < 2:
            auroc.append(np.nan)
            aupr.append(np.nan)
            continue
        auroc.append(roc_auc_score(y_bin, prob[:, k]))
        aupr.append(average_precision_score(y_bin, prob[:, k]))
    return np.asarray(auroc), np.asarray(aupr)


def compute_ovr_curve_metrics(
    y_true: pd.Series, prob_df: pd.DataFrame, classes: list[str]
) -> pd.DataFrame:
    """Point estimates plus class-stratified bootstrap 95% CIs.

    Resampling within each true class keeps every class's support fixed, so a
    replicate never loses the 17 C6 tumours and the AUROC stays defined.
    """
    y = pd.Series(y_true).astype(str).reindex(prob_df.index).to_numpy()
    prob = prob_df.reindex(columns=classes).to_numpy(dtype=float)
    auroc, aupr = ovr_scores(y, prob, classes)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    members = [np.flatnonzero(y == cls) for cls in classes]
    boot = np.empty((N_BOOTSTRAP, 2, len(classes) + 1))
    for b in range(N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(m, m.size) for m in members if m.size])
        b_auroc, b_aupr = ovr_scores(y[idx], prob[idx], classes)
        boot[b, 0] = np.append(b_auroc, np.nanmean(b_auroc))
        boot[b, 1] = np.append(b_aupr, np.nanmean(b_aupr))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

    return pd.DataFrame(
        {
            "cluster": [*classes, "macro"],
            "support": [*(m.size for m in members), len(y)],
            "auroc": np.append(auroc, np.nanmean(auroc)),
            "auroc_ci_low": lo[0],
            "auroc_ci_high": hi[0],
            "aupr": np.append(aupr, np.nanmean(aupr)),
            "aupr_ci_low": lo[1],
            "aupr_ci_high": hi[1],
            "prevalence": [*(m.size / len(y) for m in members), np.nan],
        }
    )


def compute_ovr_metrics_by_fold(
    preds_df: pd.DataFrame, classes: list[str]
) -> pd.DataFrame:
    """Per-fold AUROC/AP, to check that pooling out-of-fold scores is fair."""
    rows = []
    for fold, sub in preds_df.groupby("fold"):
        y = sub["true_cluster"].astype(str).to_numpy()
        prob = sub[[f"prob_{cls}" for cls in classes]].to_numpy(dtype=float)
        auroc, aupr = ovr_scores(y, prob, classes)
        for k, cls in enumerate(classes):
            rows.append(
                {
                    "fold": fold,
                    "cluster": cls,
                    "support": int((y == cls).sum()),
                    "auroc": auroc[k],
                    "aupr": aupr[k],
                }
            )
    return pd.DataFrame(rows)


def legend_with_ci(metrics: pd.DataFrame, cls: str, metric: str) -> str:
    row = metrics.set_index("cluster").loc[cls]
    return (
        f"C{cls}  {row[metric]:.2f} "
        f"({row[f'{metric}_ci_low']:.2f}–{row[f'{metric}_ci_high']:.2f})"
    )


def plot_multiclass_curves(
    y_true: pd.Series,
    y_prob: pd.DataFrame,
    classes: list[str],
    color_map: dict[str, str],
    curve: str,
    metrics: pd.DataFrame,
    ax,
):
    """Single-panel one-vs-rest ROC / PR curves, cluster-coloured.

    Legend entries carry the point estimate and bootstrap 95% CI from
    ``metrics``. The PR panel draws each class's prevalence as its no-skill
    baseline, since PR chance level is not shared across classes (C6 sits at
    1.7%).
    """
    if curve not in {"roc", "pr"}:
        raise ValueError("curve must be 'roc' or 'pr'")

    y_true = pd.Series(y_true).astype(str)
    display_cls = RocCurveDisplay if curve == "roc" else PrecisionRecallDisplay
    metric = "auroc" if curve == "roc" else "aupr"

    for cls in classes:
        if cls not in y_prob.columns:
            continue
        y_bin = (y_true == cls).astype(int)
        probs = y_prob[cls].reindex(y_true.index)
        color = color_map.get(cls, "#4C4C4C")
        display = display_cls.from_predictions(
            y_bin, probs, color=color, lw=CURVE_LW, ax=ax, drop_intermediate=False
        )
        display.line_.set_label(legend_with_ci(metrics, cls, metric))
        if curve == "pr":
            ax.axhline(
                y_bin.mean(), color=color, lw=0.6, ls=(0, (2, 2)), alpha=0.8, zorder=0
            )

    if curve == "roc":
        ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color="0.6", lw=0.6, zorder=0)
        style_unit_panel(
            ax,
            "False-positive rate",
            "True-positive rate",
            "AUROC (95% CI)",
            "lower right",
        )
    else:
        style_unit_panel(ax, "Recall", "Precision", "AP (95% CI)", "lower left")
    return ax


def style_unit_panel(ax, xlabel: str, ylabel: str, legend_title: str, loc: str):
    """Shared frame for the ROC, PR and reliability panels, which sit side by
    side: square [0, 1] axes, identical ticks and an identical legend box."""
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1"])
    ax.set_aspect("equal")
    ax.legend(
        loc=loc,
        title=legend_title,
        alignment="left",
        fontsize=font_pt("micro"),
        title_fontsize=font_pt("micro"),
        handlelength=1.2,
        handletextpad=0.4,
        labelspacing=0.2,
        borderaxespad=0.3,
        # Opaque box so baselines and diagonals don't run through the text.
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )


def plot_curve_figure(
    y_true: pd.Series,
    prob_df: pd.DataFrame,
    classes: list[str],
    color_map: dict[str, str],
    curve: str,
    metrics: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    plot_multiclass_curves(y_true, prob_df, classes, color_map, curve, metrics, ax)
    fig.tight_layout(pad=0.4)
    savefig(fig, out_path)


def plot_reliability_diagrams(
    per_class_ece: pd.DataFrame,
    per_class_bins: pd.DataFrame,
    out_path: Path,
    evaluation_label: str = "held-out test split",
) -> None:
    ece_lookup = dict(
        zip(per_class_ece["cluster"].astype(str), per_class_ece["ece"].astype(float))
    )
    support_lookup = dict(
        zip(
            per_class_ece["cluster"].astype(str),
            per_class_ece["support"].astype(int),
        )
    )
    clusters = sorted(
        per_class_bins["cluster"].astype(str).unique(),
        key=lambda v: int(v) if v.isdigit() else v,
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True)
    for ax, cluster in zip(axes.flat, clusters):
        sub = per_class_bins[per_class_bins["cluster"].astype(str) == cluster].copy()
        sub = sub.dropna(subset=["mean_confidence", "accuracy"])
        sub = sub.sort_values("mean_confidence")
        color = CLUSTER_COLORS.get(cluster, "#4C4C4C")
        ece = ece_lookup.get(cluster, float("nan"))
        ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.5, lw=0.8)
        ax.plot(
            sub["mean_confidence"].values,
            sub["accuracy"].values,
            marker="o",
            color=color,
        )
        ax.set_title(f"C{cluster} (n={support_lookup.get(cluster, 0)}, ECE={ece:.03f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    for ax in axes[1, :]:
        ax.set_xlabel("Mean predicted probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed frequency")
    fig.suptitle(f"One-vs-rest reliability — {evaluation_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.94], pad=1.0)
    savefig(fig, out_path)


def plot_reliability_overlay(
    per_class_ece: pd.DataFrame,
    per_class_bins: pd.DataFrame,
    out_path: Path,
) -> None:
    """Overlaid one-vs-rest reliability curves, one line per cluster.

    Drawn for both binnings. Equal-width bins cover the whole probability
    range but leave the high-probability bins of rare classes with a handful
    of samples; equal-frequency (quantile) bins give every point the same
    support but, since most one-vs-rest scores sit near zero, crowd their
    points into the low range.
    """
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.plot([0, 1], [0, 1], ls=(0, (2, 2)), color="0.6", lw=0.6, zorder=0)

    ece_lookup = per_class_ece.set_index(per_class_ece["cluster"].astype(str))["ece"]
    clusters = sorted(
        per_class_bins["cluster"].astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    for cluster in clusters:
        sub = per_class_bins[per_class_bins["cluster"].astype(str) == cluster]
        sub = sub[sub["count"].gt(0)].sort_values("mean_confidence")
        ax.plot(
            sub["mean_confidence"].values,
            sub["accuracy"].values,
            color=CLUSTER_COLORS.get(cluster, "#4C4C4C"),
            lw=CURVE_LW,
            marker="o",
            ms=2.2,
            clip_on=False,
            label=f"C{cluster}  {ece_lookup[cluster]:.3f}",
        )

    style_unit_panel(
        ax, "Predicted probability", "Observed frequency", "ECE", "lower right"
    )
    fig.tight_layout(pad=0.4)
    savefig(fig, out_path)


def plot_top_label_reliability(
    top_label_bins: pd.DataFrame,
    out_path: Path,
    evaluation_label: str = "held-out test split",
) -> None:
    sub = top_label_bins.dropna(subset=["mean_confidence", "accuracy"]).sort_values(
        "mean_confidence"
    )
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.6, lw=0.8, label="Perfect")
    ax.plot(
        sub["mean_confidence"].values,
        sub["accuracy"].values,
        marker="o",
        color="#4C4C4C",
        label=f"Top label (ECE={float(sub['ece'].iloc[0]):.03f})",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed accuracy",
        title=f"Top-label reliability — {evaluation_label}",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.4)
    savefig(fig, out_path)


def render_shap_heatmap(
    shap_long_tsv: Path,
    direction: str,
    top_n: int,
    output: Path,
) -> None:
    top_markers_heatmap(
        shap_long_tsv,
        output,
        direction=direction,
        top_n=top_n,
        value_col="shap_signed",
        selection_col="shap_signed",
        sig_col="shap_abs_norm",
        sig_mode="frequency",
        filter_col="always_pass",
        filter_mode="frequency",
        legend_title="mean SHAP",
    )


def build_classifier_volcano_table(
    volcano_tsv: Path,
    exposure_tsv: Path,
) -> pd.DataFrame:
    volcano = pd.read_csv(volcano_tsv, sep="\t")
    exposure = pd.read_csv(exposure_tsv, sep="\t")
    required_volcano = {"cluster", "gene_symbol", "mean_diff", "FDR"}
    required_exposure = {"gene_symbol", "exposure_count"}
    if missing := sorted(required_volcano - set(volcano.columns)):
        raise KeyError(f"Volcano table is missing columns: {missing}")
    if missing := sorted(required_exposure - set(exposure.columns)):
        raise KeyError(f"Feature-exposure table is missing columns: {missing}")

    volcano = volcano.copy()
    exposure = exposure.copy()
    volcano["cluster"] = volcano["cluster"].astype(str)
    volcano["gene_symbol"] = volcano["gene_symbol"].astype(str)
    exposure["gene_symbol"] = exposure["gene_symbol"].astype(str)
    volcano["mean_diff"] = pd.to_numeric(volcano["mean_diff"], errors="coerce")
    volcano["FDR"] = pd.to_numeric(volcano["FDR"], errors="coerce")
    exposure["exposure_count"] = pd.to_numeric(
        exposure["exposure_count"], errors="coerce"
    ).fillna(0)
    exposure = exposure.groupby("gene_symbol", as_index=False)["exposure_count"].max()

    table = volcano.merge(
        exposure,
        on="gene_symbol",
        how="inner",
        validate="many_to_one",
    )
    table = table[
        table["exposure_count"].gt(0)
        & table["mean_diff"].notna()
        & table["FDR"].notna()
        & ~table["gene_symbol"].str.casefold().isin({"", "nan", "none"})
    ].copy()
    table["_abs_mean_diff"] = table["mean_diff"].abs()
    table = (
        table.sort_values(
            ["cluster", "gene_symbol", "FDR", "_abs_mean_diff"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["cluster", "gene_symbol"])
        .drop(columns="_abs_mean_diff")
        .sort_values(["cluster", "FDR", "mean_diff"])
        .reset_index(drop=True)
    )
    if table.empty:
        raise ValueError(
            "No differential-abundance rows overlap proteins exposed to the "
            "canonical multiclass ensemble"
        )
    return table


def select_volcano_top_markers(table: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = []
    for cluster in sorted(
        table["cluster"].astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else value,
    ):
        sub = table[table["cluster"].astype(str).eq(cluster) & table["FDR"].lt(0.05)]
        for direction, candidates, ascending in (
            ("up", sub[sub["mean_diff"].gt(0)], False),
            ("down", sub[sub["mean_diff"].lt(0)], True),
        ):
            chosen = candidates.sort_values(
                ["mean_diff", "FDR"],
                ascending=[ascending, True],
            ).head(top_n)
            if chosen.empty:
                continue
            chosen = chosen.copy()
            chosen.insert(1, "direction", direction)
            chosen.insert(2, "rank", np.arange(1, len(chosen) + 1))
            selected.append(chosen)
    return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()


def main() -> None:
    args = parse_args()
    set_house_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(args.output_suffix).strip()
    if not suffix or any(char in suffix for char in "/\\"):
        raise ValueError("--output-suffix must be a non-empty filename-safe value")

    # ---- confusion matrix ----
    cm = load_confusion_matrix(args.results_dir / f"confusion_matrix_{suffix}.tsv")
    cm_path = args.figures_dir / f"confusion_matrix_{suffix}.png"
    plot_confusion(cm, cm_path)
    print(f"Wrote: {cm_path}")

    # ---- separate AUROC and precision-recall one-vs-rest curves ----
    preds_path = args.results_dir / "predictions.tsv"
    if preds_path.exists():
        preds_df = pd.read_csv(preds_path, sep="\t")
        prob_cols = sorted(
            [c for c in preds_df.columns if c.startswith("prob_")],
            key=lambda c: int(c.removeprefix("prob_"))
            if c.removeprefix("prob_").isdigit()
            else c,
        )
        classes = [c.removeprefix("prob_") for c in prob_cols]
        prob_df = preds_df.set_index("sample_id")[prob_cols].copy()
        prob_df.columns = classes
        y_true = preds_df.set_index("sample_id")["true_cluster"].astype(str)

        ovr_metrics = compute_ovr_curve_metrics(y_true, prob_df, classes)
        write_tsv(ovr_metrics, args.results_dir / f"ovr_curve_metrics_{suffix}.tsv")
        if "fold" in preds_df.columns:
            write_tsv(
                compute_ovr_metrics_by_fold(preds_df, classes),
                args.results_dir / f"ovr_curve_metrics_by_fold_{suffix}.tsv",
            )

        for curve, filename in (
            ("roc", f"auroc_curves_{suffix}.png"),
            ("pr", f"pr_curves_{suffix}.png"),
        ):
            curve_path = args.figures_dir / filename
            plot_curve_figure(
                y_true, prob_df, classes, CLUSTER_COLORS, curve, ovr_metrics, curve_path
            )
            print(f"Wrote: {curve_path}")

        for binning in ("uniform", "quantile"):
            bin_ece, bin_table, _ = calibration_report(
                y_true, prob_df, classes, RELIABILITY_BINS, binning
            )
            write_tsv(
                bin_ece,
                args.results_dir / f"calibration_per_class_ece_{binning}_{suffix}.tsv",
            )
            write_tsv(
                bin_table,
                args.results_dir / f"calibration_per_class_bins_{binning}_{suffix}.tsv",
            )
            overlay_path = (
                args.figures_dir / f"reliability_overlay_{binning}_{suffix}.png"
            )
            plot_reliability_overlay(bin_ece, bin_table, overlay_path)
            print(f"Wrote: {overlay_path}")
    else:
        print("Skipped AUROC/PR/reliability curves — predictions.tsv not found.")

    # ---- reliability diagrams ----
    per_class_ece_path = args.results_dir / "calibration_per_class_ece.tsv"
    per_class_bins_path = args.results_dir / "calibration_per_class_bins.tsv"
    if per_class_ece_path.exists() and per_class_bins_path.exists():
        per_class_ece = pd.read_csv(per_class_ece_path, sep="\t")
        per_class_bins = pd.read_csv(per_class_bins_path, sep="\t")
        rel_path = args.figures_dir / f"reliability_diagrams_{suffix}.png"
        plot_reliability_diagrams(
            per_class_ece,
            per_class_bins,
            rel_path,
            evaluation_label=args.evaluation_label,
        )
        print(f"Wrote: {rel_path}")
        top_label_path = args.results_dir / "calibration_top_label_bins.tsv"
        if top_label_path.exists():
            top_label_fig = args.figures_dir / f"reliability_top_label_{suffix}.png"
            plot_top_label_reliability(
                pd.read_csv(top_label_path, sep="\t"),
                top_label_fig,
                evaluation_label=args.evaluation_label,
            )
            print(f"Wrote: {top_label_fig}")
    else:
        print("Skipped reliability diagrams — calibration TSVs not found.")

    # ---- SHAP top markers ----
    shap_long_path = args.results_dir / "shap_class_long.tsv"
    if not args.skip_shap and shap_long_path.exists():
        for direction in ("up", "down"):
            out = args.figures_dir / f"shap_top{args.top_n}_{direction}_heatmap.png"
            render_shap_heatmap(shap_long_path, direction, args.top_n, out)
            print(f"Wrote: {out}")
    else:
        print("Skipped SHAP heatmaps (skip-shap set or shap_class_long.tsv missing).")

    # ---- volcano-selected markers among classifier-exposed proteins ----
    exposure_path = args.results_dir / "shap_feature_exposure.tsv"
    if (
        not args.skip_volcano_selected
        and args.volcano_tsv.exists()
        and exposure_path.exists()
    ):
        classifier_volcano = build_classifier_volcano_table(
            args.volcano_tsv,
            exposure_path,
        )
        classifier_volcano_path = (
            args.results_dir / "classifier_exposed_volcano_long.tsv"
        )
        write_tsv(classifier_volcano, classifier_volcano_path)
        selected = select_volcano_top_markers(classifier_volcano, args.top_n)
        if selected.empty:
            raise ValueError(
                "No FDR-significant up/down volcano markers were available for "
                "classifier-exposed proteins"
            )
        write_tsv(
            selected,
            args.results_dir / "volcano_selected_top_markers_per_cluster.tsv",
        )
        for direction in ("up", "down"):
            out = (
                args.figures_dir
                / f"volcano_selected_top{args.top_n}_{direction}_heatmap.png"
            )
            top_markers_heatmap(
                classifier_volcano_path,
                out,
                direction=direction,
                top_n=args.top_n,
                value_col="mean_diff",
                selection_col="mean_diff",
                sig_col="FDR",
                sig_mode="fdr",
                filter_col="FDR",
                filter_mode="fdr",
                legend_title="Δ abundance",
            )
            print(f"Wrote: {out}")
    else:
        print(
            "Skipped volcano-selected classifier heatmaps "
            "(skip set or required TSV missing)."
        )


if __name__ == "__main__":
    main()
