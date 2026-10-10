"""Mutation enrichment extended with TP53/APC/KRAS co-occurrence columns.

Adds three binary co-occurrence features (all members must be pathogenic):
  TP53_APC_pathogenic, APC_KRAS_pathogenic, TP53_APC_KRAS_pathogenic

Rows are ordered by overall cohort prevalence, descending, within two groups:
single-gene mutations first, tuples second (dashed separator). The final row
order is saved to results/mutation_enrichment/mutation_row_order.json, which is
what run_mutation_complex_oncoprint.py reads to lay out Figure 4C — the
canonical oncoprint. This module draws the enrichment heatmap only.

Runs Fisher enrichment (same method as run_mutation_enrichment.py) and
re-draws the enrichment heatmap to a NEW path only — existing outputs are
never touched.

Only genotyped tumours enter the tests: those with any call in the MuData
Mutation modality, the same availability the oncoprint shows (69 of the 997
have none and would otherwise count as wild-type). The heatmap shows a
Haldane–Anscombe log2 OR (+0.5 per cell) for tables with a zero cell, so a
significant depletion to zero mutated tumours stays visible; Fisher p-values
use the raw counts.

Outputs:
  results/mutation_enrichment/mutation_enrichment_with_tuples.tsv
  results/mutation_enrichment/mutation_frequency_with_tuples.tsv
  results/mutation_enrichment/mutation_row_order.json
  figures/cluster_interpretation/mutation_enrichment_heatmap_with_tuples.png
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    FIGURES_DIR,
    MUTATION_H5AD,
    ORDERED_MDATA_H5MU,
)
from cancerscout.plot_style import (
    CLUSTER_COLORS,
    CLUSTER_SHORT_NAMES,
    MSI_STATUS_COLORS,
    figure_size,
    savefig,
    set_house_style,
)
from cancerscout.plotting import complex_heatmap
from cancerscout.run_mutation_complex_oncoprint import load_mutation_calls

OUT_RESULTS = Path("results/mutation_enrichment")
OUT_ENR_FIG = (
    FIGURES_DIR
    / "cluster_interpretation"
    / "mutation_enrichment_heatmap_with_tuples.png"
)

TUPLES = [
    ("TP53_APC_pathogenic", ["TP53", "APC"]),
    ("APC_KRAS_pathogenic", ["APC", "KRAS"]),
    ("TP53_APC_KRAS_pathogenic", ["TP53", "APC", "KRAS"]),
]

# Genes shown in the oncoprint (original set + tuples)
ONCOPRINT_GENES = ["APC", "BRAF", "CTNNB1", "KRAS", "MLH1", "PTEN", "TP53"]

MIN_PATHOGENIC = 5


def add_tuple_columns(malign: pd.DataFrame) -> pd.DataFrame:
    for col_name, members in TUPLES:
        present = [m for m in members if m in malign.columns]
        if len(present) == len(members):
            malign[col_name] = malign[present].all(axis=1)
        else:
            print(f"  Skipping {col_name}: missing {set(members) - set(present)}")
    return malign


def load_malign_extended(model_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    mut = ad.read_h5ad(MUTATION_H5AD)
    malign = mut.obsm["malign"].copy()
    malign.index = mut.obs_names.astype(str)

    meta = pd.read_csv(model_dir / "metadata.csv", index_col=0)
    meta.index = meta.index.astype(str)

    malign = malign.reindex(meta.index).astype(bool)
    malign = add_tuple_columns(malign)

    clusters = meta["leiden_cons"].astype(str).reindex(meta.index)
    calls, _ = load_mutation_calls(ORDERED_MDATA_H5MU)
    genotyped = calls.reindex(meta.index).notna().any(axis=1)
    return malign.loc[genotyped], clusters.loc[genotyped]


def run_enrichment(
    malign: pd.DataFrame, clusters: pd.Series, genes: list[str]
) -> pd.DataFrame:
    rows = []
    for gene in genes:
        if gene not in malign.columns:
            continue
        col = malign[gene].astype(bool)
        if col.sum() < MIN_PATHOGENIC:
            continue
        for cid in sorted(clusters.dropna().unique(), key=int):
            in_c = clusters == cid
            mut = col & in_c
            wt = ~col & in_c
            mut_o = col & ~in_c
            wt_o = ~col & ~in_c
            table = np.array([[mut.sum(), mut_o.sum()], [wt.sum(), wt_o.sum()]])
            odds, p = fisher_exact(table, alternative="two-sided")
            freq_cluster = col[in_c].mean()
            freq_rest = col[~in_c].mean()
            log2_or = np.log2(odds) if (np.isfinite(odds) and odds > 0) else np.nan
            corrected = table + 0.5 if (table == 0).any() else table
            log2_or_display = np.log2(
                corrected[0, 0] * corrected[1, 1] / (corrected[0, 1] * corrected[1, 0])
            )
            rows.append(
                {
                    "gene": gene,
                    "cluster": cid,
                    "n_cluster": int(in_c.sum()),
                    "n_pathogenic_cluster": int(mut.sum()),
                    "n_pathogenic_total": int(col.sum()),
                    "freq_cluster": round(freq_cluster, 4),
                    "freq_rest": round(freq_rest, 4),
                    "odds_ratio": odds,
                    "log2_OR": log2_or,
                    "log2_OR_display": log2_or_display,
                    "p_value": p,
                }
            )
    df = pd.DataFrame(rows)
    df["FDR"] = df.groupby("gene")["p_value"].transform(
        lambda ps: multipletests(ps.fillna(1), method="fdr_bh")[1]
    )
    df["significant"] = df["FDR"] < 0.05
    return df.sort_values(["gene", "cluster"])


def run_frequency(
    malign: pd.DataFrame, clusters: pd.Series, genes: list[str]
) -> pd.DataFrame:
    rows = []
    for gene in genes:
        if gene not in malign.columns:
            continue
        col = malign[gene].astype(bool)
        rows.append(
            {
                "cluster": "ALL",
                "gene": gene,
                "n": len(col),
                "n_pathogenic": int(col.sum()),
                "freq": round(col.mean(), 4),
            }
        )
        for cid in sorted(clusters.dropna().unique(), key=int):
            sub = col[clusters == cid]
            rows.append(
                {
                    "cluster": cid,
                    "gene": gene,
                    "n": len(sub),
                    "n_pathogenic": int(sub.sum()),
                    "freq": round(sub.mean(), 4),
                }
            )
    return pd.DataFrame(rows)


def _fdr_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_enrichment_heatmap(
    enr: pd.DataFrame, genes: list[str], out_path: Path
) -> None:
    cluster_ids = [str(c) for c in sorted(enr["cluster"].unique(), key=int)]
    genes_present = [g for g in genes if g in enr["gene"].values]
    if not genes_present:
        print("  No significant genes to plot.")
        return

    lor = pd.DataFrame(index=genes_present, columns=cluster_ids, dtype=float)
    fdr = pd.DataFrame(index=genes_present, columns=cluster_ids, dtype=float)
    for _, row in enr.iterrows():
        cid = str(row["cluster"])
        if row["gene"] in lor.index and cid in lor.columns:
            lor.loc[row["gene"], cid] = row["log2_OR_display"]
            fdr.loc[row["gene"], cid] = row["FDR"]

    stars = fdr.applymap(_fdr_stars).fillna("")
    # Tuple rows get a "+"-joined display label; singletons keep the gene symbol.
    row_labels = [g.replace("_pathogenic", "").replace("_", "+") for g in genes_present]
    complex_heatmap(
        lor,
        out_path,
        legend_title="log2(OR)",
        stars=stars,
        row_labels=row_labels,
        annotate="auto",
    )


def prevalence_order(genes: list[str], malign: pd.DataFrame) -> list[str]:
    """Return genes sorted by descending overall prevalence."""
    prev = malign[genes].mean()
    return prev.sort_values(ascending=False).index.tolist()


def build_row_orders(
    malign: pd.DataFrame, sig_genes: list[str]
) -> dict[str, list[str]]:
    tuple_cols = [t[0] for t in TUPLES]

    # Enrichment heatmap: only significant genes, split and sorted by prevalence
    sig_single = [g for g in sig_genes if g not in tuple_cols]
    sig_tuples = [g for g in sig_genes if g in tuple_cols]
    enr_order = prevalence_order(sig_single, malign) + prevalence_order(
        sig_tuples, malign
    )

    # Oncoprint: all ONCOPRINT_GENES present + selected top tuples, sorted by prevalence
    opr_single = [g for g in ONCOPRINT_GENES if g in malign.columns]
    opr_tuples = [t for t in tuple_cols if t in malign.columns]
    opr_order = prevalence_order(opr_single, malign) + prevalence_order(
        opr_tuples, malign
    )

    return {"enrichment_heatmap": enr_order, "oncoprint": opr_order}


def main() -> None:
    set_house_style()
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_ENR_FIG.parent.mkdir(parents=True, exist_ok=True)

    malign, clusters = load_malign_extended(ACTIVE_MODEL_DIR)
    tuple_cols = [t[0] for t in TUPLES]
    all_genes = malign.columns.tolist()
    print(f"Loaded {len(malign)} samples, {len(all_genes)} genes+tuples in panel")
    for tc in tuple_cols:
        if tc in malign.columns:
            print(
                f"  {tc}: n_pathogenic={malign[tc].sum()}  "
                f"({malign[tc].mean()*100:.1f}%)"
            )

    enr = run_enrichment(malign, clusters, all_genes)
    freq = run_frequency(malign, clusters, all_genes)

    enr.to_csv(
        OUT_RESULTS / "mutation_enrichment_with_tuples.tsv", sep="\t", index=False
    )
    freq.to_csv(
        OUT_RESULTS / "mutation_frequency_with_tuples.tsv", sep="\t", index=False
    )
    print(f"Saved TSVs → {OUT_RESULTS.resolve()}")

    sig_genes = enr[enr["significant"]]["gene"].unique().tolist()
    orders = build_row_orders(malign, sig_genes)

    order_path = OUT_RESULTS / "mutation_row_order.json"
    with open(order_path, "w") as fh:
        json.dump(orders, fh, indent=2)
    print(f"Row order saved → {order_path}")
    print(f"  Enrichment heatmap rows: {orders['enrichment_heatmap']}")

    plot_enrichment_heatmap(enr, orders["enrichment_heatmap"], OUT_ENR_FIG)
    print(f"Enrichment heatmap → {OUT_ENR_FIG.resolve()}")


if __name__ == "__main__":
    main()
