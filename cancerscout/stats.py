"""Statistical helpers for downstream CRC analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind
from statsmodels.stats.multitest import multipletests


def bool_like(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.isin({"1", "true", "yes", "y", "mutated", "pathogenic", "msi", "dmmr"})


def mutation_matrix_from_metadata(meta: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    out = {}
    lower = {c.lower(): c for c in meta.columns}
    for gene in genes:
        for cand in [f"malign_{gene}", f"mutation_{gene}", gene]:
            col = lower.get(cand.lower())
            if col:
                out[gene] = bool_like(meta[col])
                break
    return pd.DataFrame(out, index=meta.index)


def mutation_frequency(mut: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    aligned = mut.join(clusters.rename("cluster"), how="inner")
    rows = []
    for gene in mut.columns:
        rows.append({"cluster": "ALL", "gene": gene, "n": len(aligned), "mutated_n": int(aligned[gene].sum()), "frequency": aligned[gene].mean()})
        for cluster, sub in aligned.groupby("cluster"):
            rows.append({"cluster": cluster, "gene": gene, "n": len(sub), "mutated_n": int(sub[gene].sum()), "frequency": sub[gene].mean()})
    return pd.DataFrame(rows)


def mutation_enrichment(mut: pd.DataFrame, clusters: pd.Series, min_mutated: int = 5) -> pd.DataFrame:
    aligned = mut.join(clusters.rename("cluster"), how="inner")
    rows = []
    for gene in mut.columns:
        if aligned[gene].sum() < min_mutated:
            continue
        for cluster in sorted(aligned["cluster"].dropna().astype(str).unique()):
            in_cluster = aligned["cluster"].astype(str).eq(cluster)
            mutated = aligned[gene].astype(bool)
            table = np.array(
                [[(mutated & in_cluster).sum(), (mutated & ~in_cluster).sum()],
                 [(~mutated & in_cluster).sum(), (~mutated & ~in_cluster).sum()]]
            )
            odds, p = fisher_exact(table, alternative="greater")
            rows.append({"gene": gene, "cluster": cluster, "odds_ratio": odds, "log2_odds_ratio": np.log2(odds) if np.isfinite(odds) and odds > 0 else np.nan, "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = multipletests(out["p_value"].fillna(1), method="fdr_bh")[1]
    return out


def differential_abundance(expr: pd.DataFrame, clusters: pd.Series, gene_symbols: pd.Series | None = None, min_n: int = 3) -> pd.DataFrame:
    expr = expr.reindex(clusters.index)
    rows = []
    for cluster in sorted(clusters.dropna().astype(str).unique()):
        in_cluster = clusters.astype(str).eq(cluster)
        if in_cluster.sum() < min_n or (~in_cluster).sum() < min_n:
            continue
        x = expr.loc[in_cluster]
        y = expr.loc[~in_cluster]
        for protein in expr.columns:
            stat, p = ttest_ind(x[protein], y[protein], equal_var=False, nan_policy="omit")
            rows.append(
                {
                    "cluster": cluster,
                    "protein": protein,
                    "gene_symbol": gene_symbols.get(protein, protein) if gene_symbols is not None else protein,
                    "mean_in": x[protein].mean(skipna=True),
                    "mean_out": y[protein].mean(skipna=True),
                    "mean_diff": x[protein].mean(skipna=True) - y[protein].mean(skipna=True),
                    "t_stat": stat if np.isfinite(stat) else 0,
                    "p_value": p if np.isfinite(p) else 1,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = out.groupby("cluster")["p_value"].transform(lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
        out = out.sort_values(["cluster", "FDR", "p_value", "mean_diff"], ascending=[True, True, True, False])
    return out

