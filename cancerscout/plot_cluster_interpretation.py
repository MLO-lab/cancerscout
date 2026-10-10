"""Plot side of the locked-cluster interpretation pipeline.

Reads the TSVs written by ``run_cluster_interpretation.py`` and renders the
figure set (UMAP scatter, per-metadata box/bar plots, KM curves, cell-type
heatmaps, pathway heatmap). **No compute.**

Usage:
    python -m cancerscout.plot_cluster_interpretation \\
        --out-dir results/cluster_interpretation \\
        --figures-dir figures/cluster_interpretation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from cancerscout.paths import CLUSTER_INTERPRETATION_RESULTS_DIR, FIGURES_DIR
from cancerscout.plot_style import (
    CLUSTER_COLORS,
    MSI_STATUS_COLORS,
    cluster_display_label,
    cluster_palette,
    font_pt,
    relabel_cluster_axis,
    savefig,
    set_house_style,
)
from cancerscout.plotting import blank_plot, complex_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=CLUSTER_INTERPRETATION_RESULTS_DIR,
        help="Directory containing TSVs from run_cluster_interpretation.py.",
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR / "cluster_interpretation"
    )
    parser.add_argument("--top-n", type=int, default=5)
    return parser.parse_args()


def prettify_pathway_name(term: object, max_len: int = 52) -> str:
    text = str(term)
    db = "R" if text.startswith("REACTOME_") else "G"
    text = re.sub(r"^(REACTOME_|GOBP_|GOCC_|GOMF_|HALLMARK_)", "", text)
    text = text.replace("_", " ").title()
    replacements = {
        "Rrna": "rRNA",
        "Trna": "tRNA",
        "Rna": "RNA",
        "Mrna": "mRNA",
        "Dna": "DNA",
        "Ecm": "ECM",
        "Hiv": "HIV",
        "Cftr": "CFTR",
        "Ros": "ROS",
        "Rns": "RNS",
        "Eif2Ak4": "EIF2AK4",
        "Gcn2": "GCN2",
        "Mtor": "mTOR",
        "Lkb1": "LKB1",
        "Ampk": "AMPK",
    }
    for old, new in replacements.items():
        text = re.sub(rf"\b{old}\b", new, text)
    suffix = f" [{db}]"
    if len(text) + len(suffix) > max_len:
        text = text[: max_len - len(suffix) - 3].rstrip() + "..."
    return text + suffix


METADATA_OVERREPRESENTATION_VARIABLES = (
    "MSI_status",
    "CPTAC_prediction",
    "UICC_stage",
    "tumor_location",
    "grade",
    "sex",
)
METADATA_QC_VARIABLES = ("acquisition_batch",)
MISSING_CATEGORY_LABELS = {
    "",
    "na",
    "nan",
    "none",
    "null",
    "missing",
    "unknown",
    "nicht bekannt",
}
MIN_OVERREPRESENTATION_CATEGORY_TOTAL = 5
METADATA_LABELS = {
    "MSI_status": "MSI status",
    "CPTAC_prediction": "CPTAC class",
    "UICC_stage": "UICC stage",
    "tumor_location": "Tumour location",
    "grade": "Grade",
    "sex": "Sex",
    "age": "Age",
    "acquisition_batch": "Acquisition batch",
}


def _is_missing_category(value: object) -> bool:
    return str(value).strip().lower() in MISSING_CATEGORY_LABELS


def _fdr_stars(q_value: float) -> str:
    if not np.isfinite(q_value):
        return ""
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def _collapse_uicc_major(table: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for column in table.columns:
        match = re.match(r"^(I|II|III|IV)\b", str(column).strip())
        mapping[column] = match.group(1) if match else str(column)
    return table.T.groupby(pd.Series(mapping)).sum().T


def _categorical_metadata_rows(
    out_dir: Path, variables: tuple[str, ...]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable in variables:
        counts_tsv = out_dir / f"metadata_counts_{variable}.tsv"
        if not counts_tsv.exists():
            continue
        counts = pd.read_csv(counts_tsv, sep="\t")
        if counts.empty or "cluster" not in counts.columns:
            continue
        counts["cluster"] = counts["cluster"].astype(str)
        value_cols = [
            c for c in counts.columns if c != "cluster" and not _is_missing_category(c)
        ]
        if not value_cols:
            continue
        table = (
            counts.set_index("cluster")[value_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )
        if variable == "UICC_stage":
            table = _collapse_uicc_major(table)
        table = table.loc[:, table.sum(axis=0) >= MIN_OVERREPRESENTATION_CATEGORY_TOTAL]
        if table.empty:
            continue
        cluster_totals = table.sum(axis=1)
        grand_total = int(cluster_totals.sum())
        if grand_total == 0:
            continue
        for category in table.columns:
            category_total = int(table[category].sum())
            if category_total < MIN_OVERREPRESENTATION_CATEGORY_TOTAL:
                continue
            for cluster_id in table.index.astype(str):
                a = int(table.loc[cluster_id, category])
                cluster_total = int(cluster_totals.loc[cluster_id])
                b = cluster_total - a
                c = category_total - a
                d = (grand_total - cluster_total) - c
                if cluster_total == 0 or grand_total == cluster_total:
                    continue
                odds_ratio, p_value = fisher_exact(
                    [[a, b], [c, d]], alternative="two-sided"
                )
                corrected_or = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
                log2_or = float(np.log2(corrected_or))
                rows.append(
                    {
                        "variable": variable,
                        "category": str(category),
                        "cluster": cluster_id,
                        "n_cluster": cluster_total,
                        "n_rest": grand_total - cluster_total,
                        "n_category_cluster": a,
                        "n_category_rest": c,
                        "fraction_cluster": a / cluster_total
                        if cluster_total
                        else np.nan,
                        "fraction_rest": c / (grand_total - cluster_total)
                        if grand_total != cluster_total
                        else np.nan,
                        "odds_ratio": odds_ratio,
                        "odds_ratio_haldane": corrected_or,
                        "log2_odds_ratio": log2_or,
                        "p_value": p_value,
                        "effect_definition": "category present vs absent; cluster vs rest",
                    }
                )
    return rows


def _age_metadata_rows(out_dir: Path) -> list[dict[str, object]]:
    path = out_dir / "metadata_numeric_age_samples.tsv"
    if not path.exists():
        return []
    age = pd.read_csv(path, sep="\t")
    if age.empty or not {"cluster", "value_num"}.issubset(age.columns):
        return []
    age["cluster"] = age["cluster"].astype(str)
    age["value_num"] = pd.to_numeric(age["value_num"], errors="coerce")
    age = age.dropna(subset=["cluster", "value_num"])
    sd = float(age["value_num"].std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return []
    age_z = (age["value_num"] - age["value_num"].mean()) / sd
    rows = []
    for cluster in sorted(
        age["cluster"].unique(), key=lambda x: int(x) if x.isdigit() else x
    ):
        y = age["cluster"].eq(cluster).astype(int)
        try:
            fit = sm.Logit(y, sm.add_constant(age_z)).fit(disp=False)
            coef = float(fit.params["value_num"])
            p_value = float(fit.pvalues["value_num"])
        except Exception:
            continue
        rows.append(
            {
                "variable": "age",
                "category": f"per {sd:.1f} years (1 SD)",
                "cluster": cluster,
                "n_cluster": int(y.sum()),
                "n_rest": int((1 - y).sum()),
                "n_category_cluster": np.nan,
                "n_category_rest": np.nan,
                "fraction_cluster": np.nan,
                "fraction_rest": np.nan,
                "odds_ratio": float(np.exp(coef)),
                "odds_ratio_haldane": float(np.exp(coef)),
                "log2_odds_ratio": float(coef / np.log(2)),
                "p_value": p_value,
                "effect_definition": "odds of cluster membership per 1-SD older age",
            }
        )
    return rows


def _write_and_plot_metadata_effects(
    rows: list[dict[str, object]],
    table_path: Path,
    figure_path: Path,
    significant_only: bool,
) -> None:
    if not rows:
        pd.DataFrame().to_csv(table_path, sep="\t", index=False)
        blank_plot(figure_path, "No metadata association results available")
        return

    df = pd.DataFrame(rows)
    df["FDR"] = multipletests(df["p_value"].fillna(1), method="fdr_bh")[1]
    df["direction"] = np.where(df["log2_odds_ratio"] >= 0, "enriched", "depleted")
    df.to_csv(table_path, sep="\t", index=False)

    df["label"] = (
        df["variable"].map(METADATA_LABELS).fillna(df["variable"])
        + ": "
        + df["category"].astype(str)
    )
    display = df.copy()
    if significant_only:
        keep = display.groupby("label")["FDR"].min()
        display = display[display["label"].isin(keep[keep < 0.05].index)]
    if display.empty:
        blank_plot(figure_path, "No metadata associations pass FDR < 0.05")
        return

    clusters = sorted(
        display["cluster"].astype(str).unique(),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    variable_order = {
        variable: index
        for index, variable in enumerate(
            (*METADATA_OVERREPRESENTATION_VARIABLES, "age", *METADATA_QC_VARIABLES)
        )
    }
    label_order = (
        display.assign(
            _variable_order=display["variable"].map(variable_order).fillna(999),
            _max_abs=display.groupby("label")["log2_odds_ratio"].transform(
                lambda values: values.abs().max()
            ),
        )
        .sort_values(
            ["_variable_order", "_max_abs", "label"], ascending=[True, False, True]
        )["label"]
        .drop_duplicates()
        .tolist()
    )
    plot_df = display.pivot_table(
        index="label", columns="cluster", values="log2_odds_ratio", aggfunc="first"
    ).reindex(index=label_order, columns=clusters)
    star_df = display.pivot_table(
        index="label", columns="cluster", values="FDR", aggfunc="first"
    ).reindex(index=label_order, columns=clusters)
    star_df = star_df.map(_fdr_stars).fillna("")
    plot_df.columns = [f"C{c}" for c in plot_df.columns.astype(str)]
    star_df.columns = plot_df.columns

    # The floor keeps a panel of small effects from being colour-saturated;
    # the quantile keeps one large effect from flattening everything else.
    complex_heatmap(
        plot_df,
        figure_path,
        legend_title="log2 odds\nratio",
        stars=star_df,
        annotate="stars",
        color_limit_floor=1.0,
    )


def plot_metadata_overrepresentation(out_dir: Path, fig_dir: Path) -> None:
    main_rows = _categorical_metadata_rows(
        out_dir, METADATA_OVERREPRESENTATION_VARIABLES
    )
    main_rows.extend(_age_metadata_rows(out_dir))
    _write_and_plot_metadata_effects(
        main_rows,
        out_dir / "metadata_overrepresentation_by_cluster.tsv",
        fig_dir / "metadata_overrepresentation_by_cluster.png",
        significant_only=True,
    )

    qc_rows = _categorical_metadata_rows(out_dir, METADATA_QC_VARIABLES)
    _write_and_plot_metadata_effects(
        qc_rows,
        out_dir / "metadata_overrepresentation_acquisition_batch.tsv",
        fig_dir / "metadata_overrepresentation_acquisition_batch.png",
        significant_only=False,
    )


def _short_cluster_label(value: object) -> str:
    match = re.match(r"^C?(\d+)", str(value).strip())
    return f"C{match.group(1)}" if match else str(value)


def plot_umap_panels(umap: pd.DataFrame, fig_dir: Path) -> None:
    if umap.empty:
        return
    cluster_labels = [f"C{c}" for c in sorted(CLUSTER_COLORS, key=int)]
    palettes = {
        "cluster": {
            f"C{c}": CLUSTER_COLORS[c] for c in sorted(CLUSTER_COLORS, key=int)
        },
        "CPTAC_prediction": {
            "CIN": CLUSTER_COLORS["3"],
            "MSI": CLUSTER_COLORS["4"],
            "Mesenchymal": CLUSTER_COLORS["1"],
            "Other": "#BDBDBD",
            "NA": "#BDBDBD",
        },
        "MSI_status": {key: MSI_STATUS_COLORS[key] for key in ["MSI", "MSS", "NA"]},
    }
    hue_orders = {
        "cluster": cluster_labels,
        "CPTAC_prediction": ["CIN", "MSI", "Mesenchymal", "Other", "NA"],
        "MSI_status": ["MSS", "MSI", "NA"],
    }
    for label in ("cluster", "CPTAC_prediction", "MSI_status"):
        if label not in umap.columns:
            continue
        plot = umap[["UMAP1", "UMAP2", label]].copy()
        if label == "cluster":
            plot[label] = plot[label].map(_short_cluster_label)
        else:
            plot[label] = plot[label].astype(str)
        fig, ax = plt.subplots(figsize=(4.4, 3.7))
        order = [x for x in hue_orders[label] if x in set(plot[label])]
        sns.scatterplot(
            plot,
            x="UMAP1",
            y="UMAP2",
            hue=label,
            hue_order=order,
            palette=palettes[label],
            s=18,
            linewidth=0,
            legend=False,
            ax=ax,
        )
        centers = plot.groupby(label)[["UMAP1", "UMAP2"]].median()
        label_offsets = {
            "CPTAC_prediction": {"MSI": (0.0, 0.25), "Other": (0.0, -0.25)},
            "MSI_status": {"MSS": (0.0, 0.18), "NA": (0.0, -0.18)},
        }.get(label, {})
        for category in order:
            center = centers.loc[category]
            x_offset, y_offset = label_offsets.get(category, (0.0, 0.0))
            ax.text(
                center["UMAP1"] + x_offset,
                center["UMAP2"] + y_offset,
                category,
                ha="center",
                va="center",
                color="0.1",
                fontweight="bold",
                fontsize=font_pt("label") + 4.0,
                path_effects=[
                    path_effects.withStroke(linewidth=2.5, foreground="white")
                ],
                zorder=3,
            )
        ax.set_xlabel("UMAP1", fontsize=font_pt("label") + 4.0)
        ax.set_ylabel("UMAP2", fontsize=font_pt("label") + 4.0)
        ax.tick_params(axis="both", which="major", labelsize=font_pt("tick") + 3.0)
        savefig(fig, fig_dir / f"umap_by_{label}.png")


def plot_metadata_panels(out_dir: Path, fig_dir: Path) -> None:
    plot_dir = fig_dir / "metadata"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Numeric: read *_samples.tsv (cluster, value_num per sample) and draw box/strip plots.
    for samples_tsv in sorted(out_dir.glob("metadata_numeric_*_samples.tsv")):
        label = samples_tsv.stem[len("metadata_numeric_") : -len("_samples")]
        df = pd.read_csv(samples_tsv, sep="\t")
        if df.empty:
            continue
        df["cluster_label"] = df["cluster"].astype(str).map(cluster_display_label)
        order = [
            cluster_display_label(c)
            for c in sorted(
                df["cluster"].astype(str).unique(),
                key=lambda x: int(float(x))
                if str(x).replace(".", "", 1).isdigit()
                else str(x),
            )
        ]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(
            df,
            x="cluster_label",
            y="value_num",
            order=order,
            ax=ax,
            showfliers=False,
            palette=cluster_palette(order),
        )
        sns.stripplot(
            df,
            x="cluster_label",
            y="value_num",
            order=order,
            color="black",
            alpha=0.25,
            size=2,
            ax=ax,
        )
        ax.set_title(f"{label} by cluster")
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        savefig(fig, plot_dir / f"{label}_by_cluster.png")
        plt.close(fig)

    # Categorical: read metadata_fraction_<label>.tsv and draw composition bars.
    for frac_tsv in sorted(out_dir.glob("metadata_fraction_*.tsv")):
        label = frac_tsv.stem[len("metadata_fraction_") :]
        frac = pd.read_csv(frac_tsv, sep="\t").set_index("cluster")
        if frac.empty:
            continue
        frac_plot = relabel_cluster_axis(frac, axis="index")
        long = frac_plot.reset_index().melt(
            id_vars="cluster", var_name=label, value_name="fraction"
        )
        fig, ax = plt.subplots(figsize=(max(7, 0.75 * frac.shape[0]), 4.5))
        sns.barplot(long, x="cluster", y="fraction", hue=label, ax=ax)
        ax.set_title(f"{label} composition by cluster")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        savefig(fig, plot_dir / f"{label}_by_cluster.png")
        plt.close(fig)


def plot_pathway_heatmap(out_dir: Path, fig_dir: Path) -> None:
    top_tsv = out_dir / "top_pathways_per_cluster.tsv"
    if not top_tsv.exists():
        blank_plot(
            fig_dir / "top_pathways_per_cluster_heatmap.png",
            "Pathway enrichment table unavailable",
        )
        return
    top = pd.read_csv(top_tsv, sep="\t")
    if top.empty:
        blank_plot(
            fig_dir / "top_pathways_per_cluster_heatmap.png",
            "Pathway enrichment table unavailable",
        )
        return
    term_col = (
        "Term" if "Term" in top.columns else "Name" if "Name" in top.columns else None
    )
    if term_col is None or "NES" not in top.columns:
        blank_plot(
            fig_dir / "top_pathways_per_cluster_heatmap.png",
            "Pathway enrichment table missing required columns",
        )
        return
    top = top.copy()
    top["cluster"] = top["cluster"].astype(str)
    top["_row_order"] = range(len(top))
    mat = top.pivot_table(
        index=term_col, columns="cluster", values="NES", aggfunc="mean"
    ).fillna(0)
    term_order = top.sort_values("_row_order")[term_col].drop_duplicates().tolist()
    cluster_order = sorted(
        mat.columns.astype(str), key=lambda x: int(x) if x.isdigit() else x
    )
    mat = mat.reindex(index=term_order, columns=cluster_order).fillna(0)
    row_labels = [prettify_pathway_name(term) for term in mat.index]
    row_groups = (
        top.sort_values("_row_order")
        .drop_duplicates(term_col)["cluster"]
        .astype(str)
        .tolist()
    )
    complex_heatmap(
        mat,
        fig_dir / "top_pathways_per_cluster_heatmap.png",
        title="",
        legend_title="NES",
        row_labels=row_labels,
        row_groups=row_groups,
        annotate="none",
    )


def main() -> None:
    args = parse_args()
    set_house_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    umap_tsv = args.out_dir / "umap_coordinates.tsv"
    if umap_tsv.exists():
        # keep_default_na=False so the literal "NA" string survives the round-trip
        # for samples without MSI/CPTAC annotation (otherwise they vanish from the scatter).
        plot_umap_panels(
            pd.read_csv(umap_tsv, sep="\t", keep_default_na=False, na_values=[""]),
            args.figures_dir,
        )

    plot_metadata_panels(args.out_dir, args.figures_dir)
    plot_metadata_overrepresentation(args.out_dir, args.figures_dir)
    plot_pathway_heatmap(args.out_dir, args.figures_dir)

    print(f"Wrote cluster interpretation figures to {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()
