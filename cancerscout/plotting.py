"""Plotting utilities used by the downstream scripts and notebooks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cancerscout.paths import REPO_ROOT, RSCRIPT_CONSENSUS_R_BIN, RESULTS_DIR
from cancerscout.plot_style import (
    clean_ax,
    place_legend,
    relabel_cluster_axis,
    savefig as styled_savefig,
    export_style,
)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styled_savefig(fig, path)


def blank_plot(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    savefig(fig, path)


def run_r_figure(
    script: str | Path,
    args: Sequence[str],
    *,
    cwd: Path | None = REPO_ROOT,
) -> None:
    """Run an ``r/*.R`` figure script through the pinned consensus-r Rscript.

    `script` is either a bare filename resolved under `r/` or an explicit path.
    stdout and stderr are echoed either way; a non-zero exit raises so a failed
    render can never pass silently as an up-to-date figure.
    """
    path = Path(script)
    if not path.is_absolute() and path.parent == Path("."):
        path = REPO_ROOT / "r" / path
    if not path.exists():
        raise FileNotFoundError(f"Missing R script: {path}")
    if not Path(RSCRIPT_CONSENSUS_R_BIN).exists():
        raise FileNotFoundError(f"Missing Rscript binary: {RSCRIPT_CONSENSUS_R_BIN}")

    cmd = [str(RSCRIPT_CONSENSUS_R_BIN), str(path), *[str(a) for a in args]]
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"R rendering failed ({path.name}): exit {result.returncode}"
        )


def top_markers_heatmap(
    input_tsv: Path,
    output: Path,
    *,
    direction: str,
    top_n: int,
    value_col: str,
    sig_col: str,
    sig_mode: str,
    selection_col: str | None = None,
    filter_col: str | None = None,
    filter_mode: str | None = None,
    legend_title: str | None = None,
    tiebreak: str | None = None,
) -> None:
    """Render one direction or a combined view of ``r/cluster_top_markers_heatmap.R``.

    Shared by the volcano/RFECV marker panels and the classifier SHAP panels:
    all of them show the same Welch t-test view and differ only in how the
    top-N genes per cluster are picked.
    """
    if not input_tsv.exists():
        raise FileNotFoundError(f"Missing input TSV: {input_tsv}")

    args = [
        "--input",
        str(input_tsv),
        "--value-col",
        value_col,
        "--sig-col",
        sig_col,
        "--sig-mode",
        sig_mode,
        "--direction",
        direction,
        "--output",
        str(output),
        "--top-n",
        str(top_n),
    ]
    for flag, value in (
        ("--selection-col", selection_col),
        ("--filter-col", filter_col),
        ("--filter-mode", filter_mode),
        ("--legend-title", legend_title),
        ("--tiebreak", tiebreak),
    ):
        if value:
            args += [flag, value]
    output.parent.mkdir(parents=True, exist_ok=True)
    run_r_figure("cluster_top_markers_heatmap.R", args)


def complex_heatmap(
    values: pd.DataFrame,
    path: Path,
    *,
    title: str = "",
    legend_title: str = "value",
    stars: pd.DataFrame | None = None,
    row_labels: list[str] | pd.Series | None = None,
    row_groups: list[str] | pd.Series | None = None,
    annotate: str = "none",
    color_limit_quantile: float | None = None,
    color_limit_floor: float | None = None,
    row_label_max_mm: float | None = None,
) -> None:
    """Render a matrix heatmap through R/ComplexHeatmap.

    Style, cell geometry, and output formats come from the shared spec; the
    optional arguments only cover per-figure data concerns (how aggressively to
    clip the colour scale, how much width to allow for long row labels).
    """
    if values.empty:
        blank_plot(path, f"No data for {title or path.stem}")
        return

    export_style(RESULTS_DIR / "plot_style")
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = values.copy()
    frame.columns = [str(c) for c in frame.columns]
    row_ids = [f"row_{idx + 1:04d}" for idx in range(frame.shape[0])]
    labels = (
        [str(v) for v in frame.index]
        if row_labels is None
        else [str(v) for v in list(row_labels)]
    )
    if len(labels) != len(row_ids):
        raise ValueError("row_labels must match the number of heatmap rows")
    meta = pd.DataFrame({"row_id": row_ids, "row_label": labels})
    if row_groups is not None:
        groups = [str(v) for v in list(row_groups)]
        if len(groups) != len(row_ids):
            raise ValueError("row_groups must match the number of heatmap rows")
        meta["row_group"] = groups

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        matrix_path = tmp / "matrix.tsv"
        meta_path = tmp / "row_meta.tsv"
        matrix = frame.copy()
        matrix.insert(0, "row_id", row_ids)
        matrix.to_csv(matrix_path, sep="\t", index=False)
        meta.to_csv(meta_path, sep="\t", index=False)

        args = [
            "--matrix",
            str(matrix_path),
            "--row-meta",
            str(meta_path),
            "--output",
            str(path),
            "--title",
            title,
            "--legend-title",
            legend_title,
            "--annotate",
            annotate,
        ]
        for flag, value in (
            ("--color-limit-quantile", color_limit_quantile),
            ("--color-limit-floor", color_limit_floor),
            ("--row-label-max-mm", row_label_max_mm),
        ):
            if value is not None:
                args += [flag, str(value)]
        if stars is not None:
            star_frame = (
                stars.reindex(index=values.index, columns=values.columns)
                .fillna("")
                .copy()
            )
            star_frame.columns = [str(c) for c in star_frame.columns]
            star_frame.insert(0, "row_id", row_ids)
            stars_path = tmp / "stars.tsv"
            star_frame.to_csv(stars_path, sep="\t", index=False)
            args += ["--stars", str(stars_path)]

        run_r_figure("generic_cluster_heatmap.R", args)


def heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        blank_plot(path, f"No data for {title}")
        return
    title_lower = str(title).lower()
    if "nes" in title_lower:
        legend_title = "NES"
    elif (
        "z-score" in title_lower or "zscore" in title_lower or "z score" in title_lower
    ):
        legend_title = "z-score"
    elif "shift" in title_lower:
        legend_title = "shift"
    else:
        legend_title = "value"
    complex_heatmap(df, path, title="", legend_title=legend_title, annotate="none")


def categorical_fraction_bar(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    path: Path,
    title: str | None,
    palette: dict[str, str] | list[str] | None = None,
    hue_order: list[str] | None = None,
) -> None:
    tmp = df[[group_col, value_col]].dropna().copy()
    if tmp.empty:
        blank_plot(path, f"No data for {title}")
        return
    frac = pd.crosstab(
        tmp[group_col].astype(str), tmp[value_col].astype(str), normalize="index"
    )
    if "cluster" in group_col.lower():
        frac = relabel_cluster_axis(frac, axis="index", wrapped=True)
    long = frac.reset_index().melt(
        id_vars=group_col, var_name=value_col, value_name="fraction"
    )
    fig, ax = plt.subplots(figsize=(max(7, 0.7 * frac.shape[0]), 4.5))
    sns.barplot(
        long,
        x=group_col,
        y="fraction",
        hue=value_col,
        hue_order=hue_order,
        ax=ax,
        palette=palette,
    )
    if title:
        ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    clean_ax(ax)
    place_legend(ax, mode="outside")
    fig.tight_layout()
    savefig(fig, path)
