"""Plot consensus-clustering quality diagnostics to justify the k=6 choice.

Computes multi-metric diagnostics from the 100 repeated Leiden runs
(``leiden.csv``) for k=2..max_k via average-linkage hclust + cutree, then
renders one multi-panel PNG:

  - mean and median silhouette vs k
  - 1-PAC-style agreement with decisive consensus pairs
  - pairwise consensus concordance
  - adjacent-k ARI/Jaccard with min cluster size

plus a six-panel ``k_selection_evidence.png`` that states the k choice as four
independent criteria (silhouette peak, resolved partition ambiguity, whether the
partition actually changes, and minimum cluster support), with clean standalone
exports under ``figures/muvi/k_selection_evidence_plots/``.

The hamming-distance consensus implementation matches ``r/consensus_clustering.R``.
The extended scores are written as a sibling TSV.

Usage:
    python -m cancerscout.plot_consensus_quality \\
        [--results-dir models/production_997_seed_2605161932] \\
        [--max-k 8] \\
        [--out figures/muvi/consensus_quality_k_selection.png] \\
        [--out-median figures/muvi/consensus_median_silhouette_k_selection.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_samples

from cancerscout.paths import ACTIVE_MODEL_DIR, FIGURES_DIR
from cancerscout.plot_style import (
    font_pt,
    BG_GREY,
    SEQUENTIAL_CMAP,
    clean_ax,
    place_legend,
    savefig,
    set_house_style,
)


SELECTED_COLOR = "#111111"
METRIC_PALETTE = sns.color_palette(SEQUENTIAL_CMAP, n_colors=6)
SIL_MEAN_COLOR = METRIC_PALETTE[1]
SIL_MEDIAN_COLOR = METRIC_PALETTE[4]
PAC_COLOR = METRIC_PALETTE[4]
CONCORDANCE_COLOR = METRIC_PALETTE[4]
WITHIN_COLOR = METRIC_PALETTE[2]
ARI_COLOR = METRIC_PALETTE[2]
JACCARD_COLOR = METRIC_PALETTE[4]
MIN_SIZE_COLOR = BG_GREY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=FIGURES_DIR / "muvi" / "consensus_quality_k_selection.png",
    )
    parser.add_argument(
        "--out-median",
        type=Path,
        default=FIGURES_DIR / "muvi" / "consensus_median_silhouette_k_selection.png",
        help="Single-panel median silhouette PNG destination.",
    )
    parser.add_argument(
        "--out-evidence",
        type=Path,
        default=FIGURES_DIR / "muvi" / "k_selection_evidence.png",
        help="Six-panel k-selection evidence PNG destination.",
    )
    parser.add_argument(
        "--out-scores",
        type=Path,
        default=None,
        help="Optional sibling TSV destination. Defaults to <results-dir>/consensus_scores_by_k_extended.tsv.",
    )
    return parser.parse_args()


def hamming_distance_matrix(leiden_mat: np.ndarray) -> np.ndarray:
    """Pairwise hamming distance across the repeated Leiden runs.

    leiden_mat is samples x n_runs. Distance is the fraction of runs where
    two samples were assigned to different clusters. Matches the R script's
    `compute_hamming_dist`.
    """
    return squareform(pdist(leiden_mat, metric="hamming"))


def write_scores_tsv(scores: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, sep="\t", index=False)


def adjacent_pair_jaccard(prev_labels: np.ndarray, labels: np.ndarray) -> float:
    """Pairwise Jaccard index between adjacent-k partitions."""
    iu, ju = np.triu_indices(len(labels), k=1)
    prev_same = prev_labels[iu] == prev_labels[ju]
    curr_same = labels[iu] == labels[ju]
    union = prev_same | curr_same
    if not np.any(union):
        return np.nan
    return float(np.mean((prev_same & curr_same)[union]))


def pairwise_concordance(
    labels: np.ndarray,
    consensus_matrix: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[float, float, float]:
    """Label-invariant concordance between a partition and the consensus matrix.

    Same-cluster pairs should have high consensus co-assignment. Different-cluster
    pairs should have high consensus separation, i.e. 1 - consensus co-assignment.
    The reported concordance is the mean of those two terms.
    """
    same = labels[pair_i] == labels[pair_j]
    consensus_values = consensus_matrix[pair_i, pair_j]

    within_consensus = (
        float(np.nanmean(consensus_values[same])) if np.any(same) else np.nan
    )
    between_separation = (
        float(np.nanmean(1.0 - consensus_values[~same])) if np.any(~same) else np.nan
    )
    pairwise_agreement = float(np.nanmean([within_consensus, between_separation]))
    return pairwise_agreement, within_consensus, between_separation


def evaluate_k(
    linkage_z: np.ndarray,
    dist_matrix: np.ndarray,
    consensus_matrix: np.ndarray,
    k_values: range,
    pac_low: float = 0.1,
    pac_high: float = 0.9,
) -> pd.DataFrame:
    """Per-k diagnostics for the consensus k-selection dashboard.

    The consensus matrix ``C[i,j]`` is the fraction of Leiden runs where
    samples i and j share a cluster (= 1 - hamming distance). For each k, the
    candidate partition is obtained from average-linkage hclust + cutree(k)
    on the hamming distance.

    ``pac`` is a k-specific, cola-inspired partition ambiguity score: the
    fraction of decisive off-diagonal pairs whose hclust partition contradicts
    the repeated-run consensus:
      - same-cluster pairs with C < ``pac_low``  (over-grouping)
      - different-cluster pairs with C > ``pac_high``  (over-splitting)
    Pairs with C in (pac_low, pac_high) are excluded from the decisive set.
    """
    n = consensus_matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    c_off = consensus_matrix[iu, ju]
    high_mask = c_off >= pac_high
    low_mask = c_off <= pac_low
    decisive = high_mask | low_mask  # ignore pairs in the (low, high) ambiguous zone

    rows = []
    prev_labels = None
    for k in k_values:
        labels = fcluster(linkage_z, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        sil = silhouette_samples(dist_matrix, labels, metric="precomputed")
        same = labels[iu] == labels[ju]
        # decisive partition disagreements with the consensus
        over_grouping = (
            same & low_mask
        )  # partition glued together pairs that never co-clustered
        over_splitting = (
            ~same
        ) & high_mask  # partition separated pairs that almost always co-clustered
        n_decisive = int(decisive.sum())
        pac_k = float((over_grouping.sum() + over_splitting.sum()) / max(n_decisive, 1))
        pair_conc, within_cons, between_sep = pairwise_concordance(
            labels, consensus_matrix, iu, ju
        )

        if prev_labels is None:
            adjacent_ari = np.nan
            adjacent_jaccard = np.nan
        else:
            adjacent_ari = float(adjusted_rand_score(prev_labels, labels))
            adjacent_jaccard = adjacent_pair_jaccard(prev_labels, labels)

        rows.append(
            {
                "k": k,
                "mean_silhouette": float(np.nanmean(sil)),
                "median_silhouette": float(np.nanmedian(sil)),
                "one_minus_pac": 1.0 - pac_k,
                "pac": pac_k,
                "pairwise_concordance": pair_conc,
                "within_consensus": within_cons,
                "between_separation": between_sep,
                "adjacent_ari": adjacent_ari,
                "adjacent_jaccard": adjacent_jaccard,
                "min_cluster_size": int(np.min(np.bincount(labels)[1:])),
                "cluster_sizes": ";".join(
                    str(int(v)) for v in sorted(np.bincount(labels)[1:], reverse=True)
                ),
            }
        )
        prev_labels = labels
    return pd.DataFrame(rows)


def add_selected_k(ax, selected_k: int) -> None:
    ax.axvline(selected_k, color=SELECTED_COLOR, linewidth=0.9, linestyle=":")


def plot_dashboard(scores: pd.DataFrame, selected_k: int, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex=True)
    ax_sil, ax_pac, ax_conc, ax_stability = axes.ravel()

    ax_sil.plot(
        scores["k"],
        scores["mean_silhouette"],
        marker="o",
        color=SIL_MEAN_COLOR,
        label="Mean",
    )
    ax_sil.plot(
        scores["k"],
        scores["median_silhouette"],
        marker="o",
        color=SIL_MEDIAN_COLOR,
        label="Median",
    )
    add_selected_k(ax_sil, selected_k)
    ax_sil.set_title("Silhouette")
    ax_sil.set_ylabel("Hamming-distance silhouette")
    ax_sil.set_ylim(0, 1.02)
    place_legend(ax_sil, loc="lower right", ncol=1)

    ax_pac.plot(scores["k"], scores["one_minus_pac"], marker="o", color=PAC_COLOR)
    add_selected_k(ax_pac, selected_k)
    ax_pac.set_title("1 - consensus-cut discordance")
    ax_pac.set_ylabel("Agreement with clear-cut pairs")
    ax_pac.set_ylim(0, 1.02)

    ax_conc.plot(
        scores["k"],
        scores["pairwise_concordance"],
        marker="o",
        color=CONCORDANCE_COLOR,
        label="Concordance",
    )
    ax_conc.plot(
        scores["k"],
        scores["within_consensus"],
        marker="o",
        color=WITHIN_COLOR,
        alpha=0.75,
        label="Within",
    )
    add_selected_k(ax_conc, selected_k)
    ax_conc.set_title("Pairwise Consensus Concordance")
    ax_conc.set_ylabel("Agreement")
    ax_conc.set_xlabel("Number of clusters (k)")
    ax_conc.set_ylim(0, 1.02)
    place_legend(ax_conc, loc="lower right", ncol=1)

    ax_min = ax_stability.twinx()
    ax_min.set_zorder(0)
    ax_stability.set_zorder(1)
    ax_stability.patch.set_visible(False)
    ax_min.bar(
        scores["k"],
        scores["min_cluster_size"],
        width=0.58,
        color=MIN_SIZE_COLOR,
        edgecolor="white",
        linewidth=0.6,
        label="Min cluster size",
        zorder=0,
    )
    ax_stability.plot(
        scores["k"],
        scores["adjacent_ari"],
        marker="o",
        color=ARI_COLOR,
        label="Adjacent ARI",
    )
    ax_stability.plot(
        scores["k"],
        scores["adjacent_jaccard"],
        marker="o",
        color=JACCARD_COLOR,
        label="Adjacent Jaccard",
    )
    add_selected_k(ax_stability, selected_k)
    ax_stability.set_title("Adjacent-k Stability")
    ax_stability.set_ylabel("Similarity to k - 1")
    ax_stability.set_xlabel("Number of clusters (k)")
    ax_stability.set_ylim(0, 1.02)
    ax_min.set_ylabel("Min cluster size")
    ax_min.set_ylim(0, max(20, float(scores["min_cluster_size"].max()) * 1.22))
    ax_min.tick_params(axis="y", labelsize=font_pt("label"), colors="0.35")
    ax_min.spines["right"].set_color("0.6")

    handles, labels = ax_stability.get_legend_handles_labels()
    handles.append(Patch(facecolor=MIN_SIZE_COLOR, edgecolor="white"))
    labels.append("Min cluster size")
    ax_stability.legend(
        handles, labels, frameon=False, loc="upper left", fontsize=font_pt("label")
    )

    for ax in [ax_sil, ax_pac, ax_conc, ax_stability]:
        ax.set_xticks(scores["k"])
        ax.set_xlim(scores["k"].min() - 0.3, scores["k"].max() + 0.3)
        clean_ax(ax)
    clean_ax(ax_min)
    ax_min.spines["right"].set_visible(True)

    fig.suptitle(
        "Consensus clustering quality vs k", y=0.995, fontsize=font_pt("title")
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.4, h_pad=1.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig(fig, out)


def plot_median_silhouette(scores: pd.DataFrame, selected_k: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(
        scores["k"],
        scores["median_silhouette"],
        marker="o",
        color=SIL_MEDIAN_COLOR,
        label="Median silhouette",
    )
    add_selected_k(ax, selected_k)
    ax.set_title("Median Silhouette")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Hamming-distance silhouette")
    ax.set_xticks(scores["k"])
    ax.set_xlim(scores["k"].min() - 0.3, scores["k"].max() + 0.3)
    ax.set_ylim(0, 1.02)
    clean_ax(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    savefig(fig, out)


MIN_SUPPORT = 10  # ~1% of the 997-tumour cohort
NOVELTY_CEILING = 0.95  # adjacent-k ARI above this = no new partition structure
PAC_RESOLVED = 1e-3


def k_selection_criteria(scores: pd.DataFrame) -> pd.DataFrame:
    """Per-k pass/fail on the four k-selection criteria used in the figure."""
    best_mean = scores["mean_silhouette"].max()
    best_median = scores["median_silhouette"].max()
    # adjacent_ari is NaN at the first k (no predecessor); treat that as novel.
    novelty = scores["adjacent_ari"].fillna(0.0) < NOVELTY_CEILING
    return pd.DataFrame(
        {
            "k": scores["k"],
            "Silhouette peak": (scores["mean_silhouette"] >= best_mean)
            & (scores["median_silhouette"] >= best_median),
            "Cut agrees with consensus": scores["pac"] <= PAC_RESOLVED,
            "Adds new structure": novelty,
            f"All clusters n>={MIN_SUPPORT}": scores["min_cluster_size"] >= MIN_SUPPORT,
        }
    )


def plot_k_selection_evidence(scores: pd.DataFrame, selected_k: int, out: Path) -> None:
    """Six-panel reviewer-facing argument for the selected k."""
    # 3x2, not 2x3: panel F's criterion labels are long enough that three
    # columns overflow the double-column bound even after tight_layout.
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4))
    ax_sil, ax_pac, ax_novel, ax_cascade, ax_minsize, ax_verdict = axes.ravel()
    ks = scores["k"].to_numpy()

    # A. Silhouette: both statistics peak at the selected k.
    ax_sil.plot(
        ks, scores["mean_silhouette"], marker="o", color=SIL_MEAN_COLOR, label="Mean"
    )
    ax_sil.plot(
        ks,
        scores["median_silhouette"],
        marker="o",
        color=SIL_MEDIAN_COLOR,
        label="Median",
    )
    add_selected_k(ax_sil, selected_k)
    ax_sil.set_title("A  Silhouette peaks at k=%d" % selected_k, loc="left")
    ax_sil.set_ylabel("Hamming-distance silhouette")
    ax_sil.set_ylim(0, 1.02)
    place_legend(ax_sil)

    # B. PAC on a log axis: the linear 1-PAC view hides a 3.5-decade drop.
    pac = scores["pac"].clip(lower=1e-6)
    ax_pac.plot(ks, pac, marker="o", color=PAC_COLOR)
    ax_pac.set_yscale("log")
    add_selected_k(ax_pac, selected_k)
    ax_pac.set_title("B  Consensus-cut discordance (log scale)", loc="left")
    ax_pac.set_ylabel("Discordant clear-cut pairs")

    # C. Adjacent-k novelty: k>selected barely repartitions the cohort.
    ax_novel.axhspan(NOVELTY_CEILING, 1.02, color=BG_GREY, alpha=0.35, zorder=0)
    ax_novel.plot(ks, scores["adjacent_ari"], marker="o", color=ARI_COLOR, label="ARI")
    ax_novel.plot(
        ks,
        scores["adjacent_jaccard"],
        marker="o",
        color=JACCARD_COLOR,
        label="Jaccard",
    )
    ax_novel.text(
        ks.min(),
        0.985,
        "no new structure",
        ha="left",
        va="center",
        fontsize=font_pt("tick"),
        color="0.35",
    )
    add_selected_k(ax_novel, selected_k)
    ax_novel.set_title("C  Does k add new structure?", loc="left")
    ax_novel.set_ylabel("Agreement with the k-1 partition")
    ax_novel.set_ylim(0, 1.02)
    # Opaque so the selected-k line never runs through the legend text.
    ax_novel.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="none",
        fontsize=font_pt("label"),
    )

    # D. Where the splits go: the cascade of cluster sizes.
    palette = sns.color_palette(SEQUENTIAL_CMAP, n_colors=int(ks.max()))
    for row, (k, sizes) in enumerate(zip(ks, scores["cluster_sizes"])):
        left = 0.0
        for i, size in enumerate(int(v) for v in str(sizes).split(";")):
            ax_cascade.barh(
                k, size, left=left, color=palette[i], edgecolor="white", linewidth=0.5
            )
            left += size
    ax_cascade.set_title("D  Cluster-size cascade", loc="left")
    ax_cascade.set_xlabel("Samples")
    ax_cascade.set_ylabel("Number of clusters (k)")
    ax_cascade.set_yticks(ks)
    ax_cascade.invert_yaxis()

    # E. Minimum support per cluster.
    colors = [
        "#B03A2E" if v < MIN_SUPPORT else MIN_SIZE_COLOR
        for v in scores["min_cluster_size"]
    ]
    ax_minsize.bar(ks, scores["min_cluster_size"], color=colors)
    ax_minsize.axhline(MIN_SUPPORT, color="#B03A2E", linestyle="--", linewidth=0.9)
    ax_minsize.text(
        ks.min(),
        MIN_SUPPORT * 1.15,
        f"minimum support n={MIN_SUPPORT}",
        fontsize=font_pt("tick"),
        color="#B03A2E",
    )
    add_selected_k(ax_minsize, selected_k)
    ax_minsize.set_title("E  Smallest cluster at each k", loc="left")
    ax_minsize.set_ylabel("Samples in smallest cluster")

    # F. Verdict matrix: the selected k should be the only all-pass column.
    criteria = k_selection_criteria(scores).set_index("k")
    grid = criteria.to_numpy(dtype=float).T
    ax_verdict.imshow(grid, cmap="Greens", vmin=-0.35, vmax=1.35, aspect="auto")
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            ax_verdict.text(
                c,
                r,
                "✓" if grid[r, c] else "✗",
                ha="center",
                va="center",
                color="white" if grid[r, c] else "#B03A2E",
                fontsize=font_pt("title"),
            )
    ax_verdict.set_xticks(range(len(criteria.index)))
    ax_verdict.set_xticklabels(criteria.index)
    ax_verdict.set_yticks(range(len(criteria.columns)))
    ax_verdict.set_yticklabels(criteria.columns, fontsize=font_pt("label"))
    ax_verdict.set_title("F  Criterion verdict", loc="left")
    ax_verdict.set_xlabel("Number of clusters (k)")

    for ax in (ax_sil, ax_pac, ax_novel, ax_minsize):
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(ks)
        ax.set_xlim(ks.min() - 0.3, ks.max() + 0.3)
        clean_ax(ax)
    clean_ax(ax_cascade)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    savefig(fig, out, close=False)
    write_k_selection_plots(
        out.parent / "k_selection_evidence_plots",
        fig,
        axes.ravel().tolist(),
    )
    plt.close(fig)


def write_k_selection_plots(
    figures_dir: Path,
    fig: plt.Figure,
    axes: list[plt.Axes],
) -> None:
    """Write clean standalone exports of the six evidence panels."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    panel_specs = [
        (
            "silhouette_vs_k",
            "Number of clusters (k)",
            "Hamming-distance silhouette",
        ),
        (
            "consensus_cut_discordance_vs_k",
            "Number of clusters (k)",
            "Discordant clear-cut pairs",
        ),
        (
            "adjacent_k_novelty_vs_k",
            "Number of clusters (k)",
            "Agreement with the k-1 partition",
        ),
        (
            "cluster_size_cascade",
            "Samples",
            "Number of clusters (k)",
        ),
        (
            "minimum_cluster_support_vs_k",
            "Number of clusters (k)",
            "Samples in smallest cluster",
        ),
        (
            "criterion_verdict_by_k",
            "Criterion",
            "Number of clusters (k)",
        ),
    ]
    for axis, (name, xlabel, ylabel) in zip(axes, panel_specs):
        visibility = [other.get_visible() for other in axes]
        original_title = axis.get_title(loc="left")
        original_xlabel = axis.get_xlabel()
        original_ylabel = axis.get_ylabel()
        xlabel_visible = axis.xaxis.label.get_visible()
        ylabel_visible = axis.yaxis.label.get_visible()
        hidden_annotations = []
        hidden_patches = []
        verdict_state = None
        for other in axes:
            other.set_visible(other is axis)
        if name == "adjacent_k_novelty_vs_k":
            hidden_annotations = [
                artist
                for artist in axis.texts
                if artist.get_text() == "no new structure"
            ]
            for artist in hidden_annotations:
                artist.set_visible(False)
            hidden_patches = list(axis.patches)
            for patch in hidden_patches:
                patch.set_visible(False)
        if name == "criterion_verdict_by_k":
            image = axis.images[0]
            verdict_state = {
                "array": image.get_array().copy(),
                "extent": image.get_extent(),
                "xlim": axis.get_xlim(),
                "ylim": axis.get_ylim(),
                "xticks": axis.get_xticks().copy(),
                "yticks": axis.get_yticks().copy(),
                "xticklabels": [label.get_text() for label in axis.get_xticklabels()],
                "yticklabels": [label.get_text() for label in axis.get_yticklabels()],
                "text_positions": [
                    (artist, artist.get_position()) for artist in axis.texts
                ],
            }
            image.set_data(verdict_state["array"].T)
            image.set_extent((-0.5, 3.5, 6.5, -0.5))
            for artist, (x, y) in verdict_state["text_positions"]:
                artist.set_position((y, x))
            axis.set_xlim(-0.5, 3.5)
            axis.set_ylim(6.5, -0.5)
            axis.set_xticks(range(4))
            axis.set_xticklabels(
                [
                    "Silhouette peak",
                    "Cut agrees with\nconsensus",
                    "Adds new\nstructure",
                    "All clusters\nn≥10",
                ],
                rotation=35,
                ha="right",
                rotation_mode="anchor",
            )
            axis.set_yticks(range(7))
            axis.set_yticklabels(verdict_state["xticklabels"])
        axis.set_title("", loc="left")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.xaxis.label.set_visible(True)
        axis.yaxis.label.set_visible(True)
        savefig(fig, figures_dir / name, close=False)
        axis.set_title(original_title, loc="left")
        axis.set_xlabel(original_xlabel)
        axis.set_ylabel(original_ylabel)
        axis.xaxis.label.set_visible(xlabel_visible)
        axis.yaxis.label.set_visible(ylabel_visible)
        if verdict_state is not None:
            image.set_data(verdict_state["array"])
            image.set_extent(verdict_state["extent"])
            axis.set_xlim(*verdict_state["xlim"])
            axis.set_ylim(*verdict_state["ylim"])
            axis.set_xticks(verdict_state["xticks"])
            axis.set_xticklabels(verdict_state["xticklabels"])
            axis.set_yticks(verdict_state["yticks"])
            axis.set_yticklabels(verdict_state["yticklabels"])
            for artist, position in verdict_state["text_positions"]:
                artist.set_position(position)
        for artist in hidden_annotations:
            artist.set_visible(True)
        for patch in hidden_patches:
            patch.set_visible(True)
        for other, visible in zip(axes, visibility):
            other.set_visible(visible)


def main() -> None:
    args = parse_args()
    set_house_style()

    summary = pd.read_csv(args.results_dir / "consensus_summary.csv")
    selected_k = int(summary["selected_best_k"].iloc[0])

    leiden = pd.read_csv(args.results_dir / "leiden.csv", index_col=0)
    # Treat each Leiden run as a categorical assignment; hamming distance is
    # the fraction of runs where two samples disagree on cluster identity.
    leiden_mat = leiden.to_numpy(dtype=float)
    dist_matrix = hamming_distance_matrix(leiden_mat)
    condensed = squareform(dist_matrix, checks=False)
    linkage_z = linkage(condensed, method="average")

    # Consensus matrix C[i,j] = fraction of Leiden runs where i,j share a cluster
    consensus_matrix = 1.0 - dist_matrix

    scores = evaluate_k(
        linkage_z, dist_matrix, consensus_matrix, range(2, args.max_k + 1)
    )
    out_scores = args.out_scores or (
        args.results_dir / "consensus_scores_by_k_extended.tsv"
    )
    write_scores_tsv(scores, out_scores)

    plot_dashboard(scores, selected_k, args.out)
    plot_median_silhouette(scores, selected_k, args.out_median)
    plot_k_selection_evidence(scores, selected_k, args.out_evidence)
    print(f"Wrote {args.out.resolve()}")
    print(f"Wrote {args.out_median.resolve()}")
    print(f"Wrote {args.out_evidence.resolve()}")


if __name__ == "__main__":
    main()
