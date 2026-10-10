"""CPTAC subtype concordance analysis for the six Leiden consensus clusters.

The concordance view is restricted to the three named CPTAC UMS classes
(CIN, MSI, and Mesenchymal). External CPTAC samples with no UMS label are
excluded, and their raw nearest CancerScout assignment is used as a hard
C1--C6 mapping.

Produces:
  figures/subtype_concordance/cluster_cptac_confusion.png   — count + CPTAC-column-% heatmaps
  figures/subtype_concordance/cluster_cptac_bilateral_composition.png — directional UMS transfer composite
  figures/subtype_concordance/cancerscout_to_cptac_ums.png — CancerScout-to-CPTAC standalone
  figures/subtype_concordance/cptac_ums_to_cancerscout.png — CPTAC-to-CancerScout standalone
  figures/subtype_concordance/cluster_cptac_reverse_stacked_bar.png — CPTAC-normalised supplementary view
  figures/subtype_concordance/cluster_cptac_sankey.html     — interactive Sankey
  figures/subtype_concordance/cluster_cptac_sankey.png      — static Sankey
  figures/subtype_concordance/cluster_cptac_alluvial.png    — pill-styled alluvial alternative to the Sankey
  results/subtype_concordance/concordance_stats.tsv          — per-cell enrichment + global stats
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score

from cancerscout.paths import (
    CPTAC_RESULTS_DIR,
    EXTERNAL_TRANSFER_RESULTS_DIR,
    FIGURES_DIR,
    SUBTYPE_CONCORDANCE_RESULTS_DIR,
)
from cancerscout.plot_style import (
    CLUSTER_COLORS,
    CLUSTER_SHORT_NAMES,
    STYLE,
    figure_size,
    font_pt,
    pt_to_css_px,
    save_plotly,
    set_house_style,
    savefig,
)
from cancerscout.plotting import run_r_figure

OUT_FIGURES = FIGURES_DIR / "subtype_concordance"
OUT_RESULTS = SUBTYPE_CONCORDANCE_RESULTS_DIR
EXTERNAL_ASSIGNMENTS = EXTERNAL_TRANSFER_RESULTS_DIR / "cptac_assignments.tsv"
REVERSE_PREDICTIONS = CPTAC_RESULTS_DIR / "sample_leiden_cons_cptac_prediction.csv"

CPTAC_ORDER = ["CIN", "MSI", "Mesenchymal"]
CLUSTER_ORDER = [str(i) for i in range(1, 7)]
CPTAC_COLORS = {
    "CIN": CLUSTER_COLORS["3"],
    "MSI": CLUSTER_COLORS["4"],
    "Mesenchymal": CLUSTER_COLORS["1"],
}
TRANSFER_CLUSTER_ORDER = CLUSTER_ORDER
TRANSFER_CLUSTER_COLORS = CLUSTER_COLORS.copy()

FIG_TITLE_SIZE = font_pt("title") + 4.0
FIG_LABEL_SIZE = font_pt("label") + 4.0
FIG_TICK_SIZE = font_pt("tick") + 3.0
FIG_ANNOT_SIZE = font_pt("annot") + 3.0


def load_data() -> pd.DataFrame:
    if not REVERSE_PREDICTIONS.exists():
        raise FileNotFoundError(
            "Missing saved CPTAC-trained predictions: " f"{REVERSE_PREDICTIONS}"
        )
    predictions = pd.read_csv(REVERSE_PREDICTIONS)
    required = {"leiden_cons", "cptac_prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Saved CPTAC-trained prediction table is missing columns: "
            f"{sorted(missing)}"
        )
    keep = predictions[["leiden_cons", "cptac_prediction"]].dropna().copy()
    keep["cluster"] = (
        pd.to_numeric(keep["leiden_cons"], errors="raise").astype(int).astype(str)
    )
    keep["cptac"] = keep["cptac_prediction"].astype(str).str.strip()
    return keep.loc[keep["cptac"].isin(CPTAC_ORDER)].copy()


def load_external_assignments() -> pd.DataFrame:
    if not EXTERNAL_ASSIGNMENTS.exists():
        raise FileNotFoundError(
            "Missing saved CancerScout-to-CPTAC assignments: " f"{EXTERNAL_ASSIGNMENTS}"
        )
    assignments = pd.read_csv(EXTERNAL_ASSIGNMENTS, sep="\t")
    required = {"cohort", "UMS", "predicted_cluster"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(
            "Saved external-transfer table is missing columns: " f"{sorted(missing)}"
        )
    assignments = assignments.loc[assignments["cohort"].eq("CPTAC")].copy()
    assignments["ums_display"] = assignments["UMS"].fillna("").astype(str).str.strip()
    assignments = assignments.loc[assignments["ums_display"].isin(CPTAC_ORDER)].copy()
    assignments["predicted_cluster_hard"] = (
        pd.to_numeric(assignments["predicted_cluster"], errors="raise")
        .astype(int)
        .astype(str)
    )
    return assignments


def cluster_label(cid: str) -> str:
    return f"C{cid} {CLUSTER_SHORT_NAMES.get(cid, '')}"


def cluster_short_label(cid: str) -> str:
    return f"C{cid}"


def plot_confusion(
    ct: pd.DataFrame,
    ct_norm_cluster: pd.DataFrame,
    ct_norm_cptac: pd.DataFrame,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    panels = [
        (
            axes[0],
            ct,
            ct.columns.tolist(),
            [cluster_short_label(c) for c in ct.index],
            "CPTAC subtype",
            "Sample counts",
            "d",
            None,
            35,
            FIG_ANNOT_SIZE,
        ),
        (
            axes[1],
            ct_norm_cptac,
            [cluster_short_label(c) for c in ct_norm_cptac.columns],
            ct_norm_cptac.index.tolist(),
            "Leiden cluster",
            "Column % (per CPTAC subtype)",
            ".0%",
            1.0,
            35,
            FIG_ANNOT_SIZE,
        ),
    ]

    for (
        ax,
        data,
        x_labels,
        y_labels,
        xlabel,
        title,
        fmt,
        vmax,
        x_rotation,
        text_size,
    ) in panels:
        n_rows, n_cols = data.shape
        im = ax.imshow(
            data.values,
            aspect="auto",
            cmap="Blues",
            vmin=0,
            vmax=vmax if vmax else data.values.max(),
        )
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(
            x_labels, fontsize=FIG_TICK_SIZE, rotation=x_rotation, ha="right"
        )
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=FIG_TICK_SIZE)
        ax.set_xlabel(xlabel, fontsize=FIG_LABEL_SIZE)
        ax.set_title(title, fontsize=FIG_TITLE_SIZE)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=FIG_TICK_SIZE)

        for i in range(n_rows):
            for j in range(n_cols):
                val = data.values[i, j]
                text = f"{val:{fmt}}"
                color = (
                    "white" if val > (0.6 * (vmax or data.values.max())) else "black"
                )
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=text_size,
                    color=color,
                )

    fig.suptitle("Leiden clusters × CPTAC subtypes", fontsize=FIG_TITLE_SIZE, y=1.01)
    fig.tight_layout()
    savefig(fig, out_path)


def _annotated_stacked_bar(
    ax: plt.Axes,
    values: pd.DataFrame,
    source_counts: pd.Series,
    colors: dict[str, str],
    xlabel: str,
    legend_title: str,
) -> None:
    categories = values.index.tolist()
    segments = values.columns.tolist()
    y = np.arange(len(categories))
    left = np.zeros(len(categories), dtype=float)
    for segment in segments:
        vals = values[segment].to_numpy(dtype=float)
        bars = ax.barh(
            y,
            vals,
            left=left,
            color=colors.get(segment, "#BDBDBD"),
            label=f"C{segment}" if segment in CLUSTER_ORDER else segment,
            height=0.68,
            edgecolor="white",
            linewidth=0.35,
        )
        for bar, value in zip(bars, vals):
            if value >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FIG_ANNOT_SIZE,
                    color=_contrast_text(colors.get(segment, "#BDBDBD")),
                )
        left += vals
    labels = [
        f"{category} (n={int(source_counts[category])})" for category in categories
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FIG_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, fontsize=FIG_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=FIG_TICK_SIZE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0, -0.28),
        ncol=2,
        fontsize=FIG_TICK_SIZE,
        frameon=False,
        title=legend_title,
        title_fontsize=FIG_LABEL_SIZE,
    )


def _contrast_text(color: str) -> str:
    red, green, blue = to_rgb(color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "black" if luminance >= 0.62 else "white"


def _composition_table(
    frame: pd.DataFrame,
    row_col: str,
    row_order: list[str],
    segment_col: str,
    segment_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(frame[row_col], frame[segment_col]).reindex(
        index=row_order, columns=segment_order, fill_value=0
    )
    counts = counts.astype(int)
    totals = counts.sum(axis=1)
    if (totals <= 0).any():
        missing = totals[totals <= 0].index.tolist()
        raise ValueError(f"Composition rows have no samples: {missing}")
    fractions = counts.div(totals, axis=0)
    if not np.allclose(fractions.sum(axis=1), 1.0):
        raise ValueError("Directional composition rows do not sum to one")
    return counts, fractions


def _directional_panel(
    ax: plt.Axes,
    counts: pd.DataFrame,
    fractions: pd.DataFrame,
    colors: dict[str, str],
    xlabel: str,
    legend_side: bool = False,
    bar_height: float = 0.68,
) -> None:
    segments = fractions.columns.tolist()
    rows = fractions.index.tolist()
    y = np.arange(len(rows))
    left = np.zeros(len(rows), dtype=float)

    for segment in segments:
        values = fractions[segment].to_numpy(dtype=float)
        color = colors.get(segment, STYLE.palette.missing)
        bars = ax.barh(
            y,
            values,
            left=left,
            color=color,
            label=(f"C{segment}" if segment in CLUSTER_ORDER else segment),
            height=bar_height,
            edgecolor="white",
            linewidth=0.45,
        )
        for bar, value in zip(bars, values):
            if value >= 0.08:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FIG_ANNOT_SIZE,
                    color=_contrast_text(color),
                )
        left += values

    row_totals = counts.sum(axis=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"C{row} (n={int(row_totals[row])})"
            if row in CLUSTER_ORDER
            else f"{row} (n={int(row_totals[row])})"
            for row in rows
        ],
        fontsize=FIG_TICK_SIZE,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(xlabel, fontsize=FIG_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=FIG_TICK_SIZE)
    if legend_side:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            fontsize=FIG_TICK_SIZE,
            frameon=False,
            handlelength=1.1,
            handletextpad=0.35,
            borderaxespad=0,
        )
    else:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0, -0.22),
            ncol=4 if len(segments) > 4 else len(segments),
            fontsize=FIG_TICK_SIZE,
            frameon=False,
            handlelength=1.1,
            columnspacing=0.8,
            handletextpad=0.35,
            borderaxespad=0,
        )


def _directional_figure(
    counts: pd.DataFrame,
    fractions: pd.DataFrame,
    colors: dict[str, str],
    xlabel: str,
    out_path: Path,
    tier: str,
    legend_side: bool = False,
    bar_height: float = 0.68,
    square_axes: bool = False,
) -> None:
    width_in, height_in = figure_size(tier, height_mm=125)
    if legend_side:
        width_in *= 1.2
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    _directional_panel(
        ax,
        counts,
        fractions,
        colors,
        xlabel,
        legend_side=legend_side,
        bar_height=bar_height,
    )
    if square_axes:
        ax.set_box_aspect(1)
    fig.subplots_adjust(
        left=0.29,
        right=0.82 if legend_side else 0.98,
        top=0.95,
        bottom=0.22 if legend_side else 0.42,
    )
    savefig(fig, out_path)


def plot_bilateral_composition(
    forward_counts: pd.DataFrame,
    forward_fractions: pd.DataFrame,
    reverse_counts: pd.DataFrame,
    reverse_fractions: pd.DataFrame,
    out_path: Path,
) -> None:
    """Render the two reciprocal CPTAC transfer directions side by side."""
    width_in, height_in = figure_size("double", height_mm=125)
    fig, axes = plt.subplots(1, 2, figsize=(width_in * 0.90, height_in))
    _directional_panel(
        axes[0],
        forward_counts,
        forward_fractions,
        TRANSFER_CLUSTER_COLORS,
        "Proportion of\nCPTAC samples",
    )
    _directional_panel(
        axes[1],
        reverse_counts,
        reverse_fractions,
        CPTAC_COLORS,
        "Proportion of\nCancerScout samples",
    )
    fig.subplots_adjust(left=0.17, right=0.99, top=0.95, bottom=0.42, wspace=0.52)
    savefig(fig, out_path)


def plot_cancerscout_to_cptac(
    counts: pd.DataFrame, fractions: pd.DataFrame, out_path: Path
) -> None:
    _directional_figure(
        counts,
        fractions,
        TRANSFER_CLUSTER_COLORS,
        "Proportion of cohort samples",
        out_path,
        "single",
        legend_side=True,
        bar_height=0.56,
        square_axes=True,
    )


def plot_cptac_to_cancerscout(
    counts: pd.DataFrame, fractions: pd.DataFrame, out_path: Path
) -> None:
    _directional_figure(
        counts,
        fractions,
        CPTAC_COLORS,
        "Proportion of\nCancerScout samples",
        out_path,
        "single",
    )


def plot_reverse_stacked_bar(ct: pd.DataFrame, out_path: Path) -> None:
    """Render the CPTAC-normalised direction as the S4C supplementary asset."""
    values = ct.div(ct.sum(axis=0), axis=1).T
    counts = ct.sum(axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _annotated_stacked_bar(
        ax,
        values,
        counts,
        {cluster: CLUSTER_COLORS[cluster] for cluster in CLUSTER_ORDER},
        "Proportion of\nCPTAC class",
        "Leiden state",
    )
    ax.set_title("CPTAC classes → Leiden clusters", fontsize=FIG_TITLE_SIZE)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.86, bottom=0.42)
    savefig(fig, out_path)


def plot_sankey(df: pd.DataFrame, out_png: Path) -> None:
    """Cluster -> CPTAC Sankey, stripped to the two label columns.

    Node positions are fixed rather than solved so the labels can be drawn as
    annotations outside the diagram: plotly's own node labels sit on top of the
    ribbons, which is unreadable at print size.
    """
    import plotly.graph_objects as go

    cluster_ids = sorted(df["cluster"].unique(), key=int)
    subtypes = [s for s in CPTAC_ORDER if (df["cptac"] == s).any()]
    counts = {
        (c, s): int(((df["cluster"] == c) & (df["cptac"] == s)).sum())
        for c in cluster_ids
        for s in subtypes
    }

    left_total = {c: sum(counts[(c, s)] for s in subtypes) for c in cluster_ids}
    right_total = {s: sum(counts[(c, s)] for c in cluster_ids) for s in subtypes}
    total = sum(left_total.values())

    # Order the right axis by the flow-weighted mean rank of its incoming
    # ribbons: with C1..C6 fixed top-down on the left, this is what stops the
    # dominant flows from crossing each other.
    rank = {c: i for i, c in enumerate(cluster_ids)}
    subtypes = sorted(
        subtypes,
        key=lambda s: sum(rank[c] * counts[(c, s)] for c in cluster_ids)
        / max(right_total[s], 1),
    )

    def stack(order, sizes, gap):
        """Node centres in [0, 1] top-down, with a fixed gap between nodes."""
        span = 1.0 - gap * (len(order) - 1)
        centres, cursor = {}, 0.0
        for key in order:
            height = sizes[key] / total * span
            centres[key] = cursor + height / 2.0
            cursor += height + gap
        return centres

    left_y = stack(cluster_ids, left_total, 0.025)
    right_y = stack(subtypes, right_total, 0.025)

    nodes = [("L", c) for c in cluster_ids] + [("R", s) for s in subtypes]
    index = {node: i for i, node in enumerate(nodes)}
    node_x = [0.001] * len(cluster_ids) + [0.999] * len(subtypes)
    node_y = [left_y[c] for c in cluster_ids] + [right_y[s] for s in subtypes]
    node_color = [CLUSTER_COLORS.get(c, "#4C4C4C") for c in cluster_ids] + [
        CPTAC_COLORS.get(s, "#BDBDBD") for s in subtypes
    ]

    sources, targets, values, link_colors = [], [], [], []
    for c in cluster_ids:
        base = CLUSTER_COLORS.get(c, "#4C4C4C")
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        for s in subtypes:
            if counts[(c, s)] == 0:
                continue
            sources.append(index[("L", c)])
            targets.append(index[("R", s)])
            values.append(counts[(c, s)])
            link_colors.append(f"rgba({r},{g},{b},0.45)")

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                pad=10,
                thickness=9,
                line=dict(width=0),
                label=[""] * len(nodes),
                x=node_x,
                y=node_y,
                color=node_color,
                hoverinfo="none",
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                line=dict(width=0),
                hoverinfo="none",
            ),
        )
    )

    # Match the enlarged 12-point-equivalent labels used in the UMAP panels.
    label_px = pt_to_css_px("label") * 1.5
    # Sankey node.y runs top-down; paper-coordinate annotations run bottom-up.
    annotations = [
        dict(
            x=-0.012,
            y=1.0 - left_y[c],
            xref="paper",
            yref="paper",
            text=f"C{c}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=label_px, color="#1A1A1A"),
        )
        for c in cluster_ids
    ] + [
        dict(
            x=1.012,
            y=1.0 - right_y[s],
            xref="paper",
            yref="paper",
            text=s,
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(size=label_px, color="#1A1A1A"),
        )
        for s in subtypes
    ]

    fig.update_layout(
        annotations=annotations,
        font=dict(family=STYLE.typography.family, size=label_px),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=88, r=144, t=8, b=8),
    )
    save_plotly(fig, out_png, width_in=7.0, height_in=4.2)
    fig.write_html(str(out_png.with_suffix(".html")))


def plot_stacked_bar(ct_norm: pd.DataFrame, out_path: Path) -> None:
    """Horizontal stacked bar — proportion of each cluster going to each CPTAC subtype."""
    cluster_ids = ct_norm.index.tolist()
    cptac_cols = ct_norm.columns.tolist()

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    lefts = np.zeros(len(cluster_ids))
    y = np.arange(len(cluster_ids))

    for sub in cptac_cols:
        vals = ct_norm[sub].values
        bars = ax.barh(
            y,
            vals,
            left=lefts,
            color=CPTAC_COLORS.get(sub, "#BDBDBD"),
            label=sub,
            height=0.65,
        )
        for bar, val in zip(bars, vals):
            if val > 0.07:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=FIG_ANNOT_SIZE,
                    color="white",
                )
        lefts += vals

    ax.set_yticks(y)
    ax.set_yticklabels(
        [cluster_short_label(c) for c in cluster_ids], fontsize=FIG_TICK_SIZE
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of cluster", fontsize=FIG_LABEL_SIZE)
    ax.set_title("Leiden clusters → CPTAC subtypes", fontsize=FIG_TITLE_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=FIG_TICK_SIZE)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=FIG_TICK_SIZE,
        frameon=False,
        title="CPTAC subtype",
        title_fontsize=FIG_LABEL_SIZE,
    )
    fig.tight_layout()
    savefig(fig, out_path)


def compute_stats(df: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    chi2, p, dof, expected = chi2_contingency(ct.values)
    n = ct.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    ari = adjusted_rand_score(df["cluster"], df["cptac"])

    # Standardised Pearson residuals per cell
    rows = []
    for i, cid in enumerate(ct.index):
        for j, sub in enumerate(ct.columns):
            obs = ct.values[i, j]
            exp = expected[i, j]
            resid = (obs - exp) / np.sqrt(exp) if exp > 0 else np.nan
            rows.append(
                {
                    "cluster": cid,
                    "cluster_label": cluster_label(cid),
                    "cptac_subtype": sub,
                    "observed": int(obs),
                    "expected": round(exp, 1),
                    "std_residual": round(resid, 2),
                    "enriched": resid > 2,
                    "depleted": resid < -2,
                }
            )
    stats_df = pd.DataFrame(rows)
    stats_df.attrs["chi2"] = chi2
    stats_df.attrs["p"] = p
    stats_df.attrs["dof"] = dof
    stats_df.attrs["cramers_v"] = cramers_v
    stats_df.attrs["ari"] = ari
    return stats_df


def main() -> None:
    set_house_style()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    forward = load_external_assignments()
    print(f"Loaded {len(df)} saved CPTAC-trained predictions")
    print(f"Loaded {len(forward)} saved CancerScout-to-CPTAC assignments")

    ct = pd.crosstab(df["cluster"], df["cptac"]).reindex(
        index=CLUSTER_ORDER, columns=CPTAC_ORDER, fill_value=0
    )
    ct_norm_cluster = ct.div(ct.sum(axis=1), axis=0)
    ct_norm_cptac = ct.div(ct.sum(axis=0), axis=1).T

    expected_clusters = {"1": 155, "2": 297, "3": 212, "4": 196, "5": 67, "6": 15}
    expected_cptac = {"CIN": 658, "MSI": 168, "Mesenchymal": 116}
    if len(df) != 942 or df["cluster"].value_counts().to_dict() != expected_clusters:
        raise ValueError(
            "Saved CPTAC-trained predictions do not contain the expected 942 known-label "
            f"cluster counts: {df['cluster'].value_counts().to_dict()}"
        )
    if df["cptac"].value_counts().to_dict() != expected_cptac:
        raise ValueError(
            "Saved CPTAC-trained predictions do not contain the expected class "
            f"counts: {df['cptac'].value_counts().to_dict()}"
        )
    if int(ct.to_numpy().sum()) != 942:
        raise ValueError(
            f"Reverse CPTAC cross-tabulation sums to {int(ct.to_numpy().sum())}, not 942"
        )
    if not np.allclose(ct_norm_cluster.sum(axis=1), 1.0):
        raise ValueError("Leiden-normalised CPTAC composition does not sum to one")
    if not np.allclose(ct_norm_cptac.sum(axis=1), 1.0):
        raise ValueError("CPTAC-normalised Leiden composition does not sum to one")

    expected_ums = {"CIN": 46, "MSI": 15, "Mesenchymal": 17}
    if (
        len(forward) != 78
        or forward["ums_display"].value_counts().to_dict() != expected_ums
    ):
        raise ValueError(
            "Saved CPTAC assignments do not contain the expected UMS support: "
            f"{forward['ums_display'].value_counts().to_dict()}"
        )
    if not forward["predicted_cluster_hard"].isin(TRANSFER_CLUSTER_ORDER).all():
        raise ValueError(
            "CPTAC hard assignments contain an unexpected cluster: "
            f"{sorted(forward['predicted_cluster_hard'].unique())}"
        )

    forward_counts, forward_fractions = _composition_table(
        forward,
        "ums_display",
        CPTAC_ORDER,
        "predicted_cluster_hard",
        TRANSFER_CLUSTER_ORDER,
    )
    reverse_counts = ct.copy()
    reverse_fractions = ct_norm_cluster.copy()

    print("\nConfusion matrix (counts):")
    print(ct.to_string())

    stats = compute_stats(df, ct)
    print(
        f"\nChi-square: {stats.attrs['chi2']:.1f}, p={stats.attrs['p']:.2e}, "
        f"Cramér's V={stats.attrs['cramers_v']:.3f}, ARI={stats.attrs['ari']:.3f}"
    )

    print("\nSignificantly enriched (|std residual| > 2):")
    sig = stats.loc[stats["enriched"] | stats["depleted"]].sort_values(
        "std_residual", ascending=False
    )
    print(
        sig[
            ["cluster_label", "cptac_subtype", "observed", "expected", "std_residual"]
        ].to_string(index=False)
    )

    # Save stats with global metrics as header rows
    header = pd.DataFrame(
        [
            {
                "cluster": "GLOBAL",
                "cluster_label": "",
                "cptac_subtype": f"chi2={stats.attrs['chi2']:.1f} p={stats.attrs['p']:.2e} "
                f"CramersV={stats.attrs['cramers_v']:.3f} ARI={stats.attrs['ari']:.3f}",
                "observed": "",
                "expected": "",
                "std_residual": "",
                "enriched": "",
                "depleted": "",
            }
        ]
    )
    pd.concat([header, stats], ignore_index=True).to_csv(
        OUT_RESULTS / "concordance_stats.tsv", sep="\t", index=False
    )
    ct.to_csv(OUT_RESULTS / "bilateral_counts.tsv", sep="\t")
    ct_norm_cluster.to_csv(OUT_RESULTS / "bilateral_by_leiden.tsv", sep="\t")
    ct_norm_cptac.to_csv(OUT_RESULTS / "bilateral_by_cptac.tsv", sep="\t")

    print("\nPlotting confusion matrix...")
    plot_confusion(
        ct,
        ct_norm_cluster,
        ct_norm_cptac,
        OUT_FIGURES / "cluster_cptac_confusion.png",
    )

    print("Plotting Sankey...")
    plot_sankey(df, OUT_FIGURES / "cluster_cptac_sankey.png")
    print("Plotting alluvial...")
    run_r_figure(
        "cluster_cptac_alluvial.R",
        [
            OUT_RESULTS / "bilateral_counts.tsv",
            OUT_FIGURES / "cluster_cptac_alluvial.png",
            *(f"{subtype}={color}" for subtype, color in CPTAC_COLORS.items()),
        ],
    )

    print("Plotting stacked bar...")
    plot_stacked_bar(ct_norm_cluster, OUT_FIGURES / "cluster_cptac_stacked_bar.png")
    print("Plotting bilateral composition...")
    plot_bilateral_composition(
        forward_counts,
        forward_fractions,
        reverse_counts,
        reverse_fractions,
        OUT_FIGURES / "cluster_cptac_bilateral_composition.png",
    )
    print("Plotting directional standalones...")
    plot_cancerscout_to_cptac(
        forward_counts,
        forward_fractions,
        OUT_FIGURES / "cancerscout_to_cptac_ums.png",
    )
    plot_cptac_to_cancerscout(
        reverse_counts,
        reverse_fractions,
        OUT_FIGURES / "cptac_ums_to_cancerscout.png",
    )
    plot_reverse_stacked_bar(ct, OUT_FIGURES / "cluster_cptac_reverse_stacked_bar.png")

    print(f"\nDone. Figures → {OUT_FIGURES.resolve()}")


if __name__ == "__main__":
    main()
