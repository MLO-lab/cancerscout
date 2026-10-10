"""Compute side of the GSEA/ssGSEA pipeline.

Reads the trained-model handoff and final normalized proteome, runs preranked
GSEA, redundancy pruning, ssGSEA, and Kruskal-Wallis tests, and writes the
result TSVs into ``--out-dir``. **No figures.** Use ``plot_gsea.py`` to render
figures from the saved TSVs.

Usage:
    python -m cancerscout.run_gsea \\
        --gene-set gobp --out-dir results/gsea/gobp \\
        --permutations 1000 --threads 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mudata as mu
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, kruskal, spearmanr, t as student_t
from statsmodels.stats.multitest import multipletests

from cancerscout.io import build_sample_id_map, find_col, read_metadata, resolve_mdata_path, save_text, write_tsv
from cancerscout.paths import ACTIVE_MODEL_DIR, GENE_SET_DIR, GSEA_RESULTS_DIR, REPORTS_DIR

try:
    import gseapy as gp
except Exception:  # pragma: no cover
    gp = None


DEFAULT_GENE_SET_CANDIDATES = {
    "gobp": [GENE_SET_DIR / "c5.go.bp.v7.5.1.symbols.gmt"],
    "gocc": [GENE_SET_DIR / "c5.go.cc.v7.5.1.symbols.gmt"],
    "reactome": [GENE_SET_DIR / "c2.cp.reactome.v7.5.1.symbols.gmt"],
    "hallmark": [GENE_SET_DIR / "h.all.v7.5.1.symbols.gmt"],
}

ADJUSTMENT_COVARIATE_CANDIDATES = {
    "tumor_cell_content": ["Proteome:Tumorzellgehalt (Mol+Prot)", "Tumorzellgehalt (Mol+Prot)", "Tumorzellgehalt", "tumor content"],
    "missing_fraction": ["Proteome:new_missing_fraction", "new_missing_fraction", "missing_fraction"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--mdata-path", type=Path, default=None)
    parser.add_argument("--cluster-col", default="leiden_cons")
    parser.add_argument("--gene-set", default="reactome", help="One of gobp/gocc/reactome/hallmark or a .gmt path.")
    parser.add_argument("--out-dir", type=Path, default=GSEA_RESULTS_DIR / "reactome")
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--max-size", type=int, default=500)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260507)
    parser.add_argument("--gsea-fdr", type=float, default=0.05)
    parser.add_argument("--redundancy-fdr", type=float, default=0.05)
    parser.add_argument("--gsva-fdr", type=float, default=0.05)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--skip-ssgsea", action="store_true")
    parser.add_argument("--skip-adjusted-gsea", action="store_true", help="Skip tumor-content/missingness-adjusted sensitivity GSEA.")
    return parser.parse_args()


def resolve_gene_set(name_or_path: str) -> Path | str:
    path = Path(name_or_path)
    if path.exists():
        return path
    key = name_or_path.lower()
    if key in DEFAULT_GENE_SET_CANDIDATES:
        for candidate in DEFAULT_GENE_SET_CANDIDATES[key]:
            if candidate.exists():
                return candidate
    if key == "reactome":
        return "Reactome_2022"
    if key == "hallmark":
        return "MSigDB_Hallmark_2020"
    raise FileNotFoundError(f"Could not resolve gene set: {name_or_path}")


def read_gmt(path: Path) -> dict[str, set[str]]:
    gene_sets = {}
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            term = parts[0]
            genes = {g.upper() for g in parts[2:] if g}
            if genes:
                gene_sets[term] = genes
    return gene_sets


def load_gene_expression(results_dir: Path, mdata_path: Path | None) -> tuple[pd.DataFrame, dict]:
    path = mdata_path or resolve_mdata_path(results_dir)
    if path is None:
        raise FileNotFoundError("Could not resolve h5mu path from manifest; pass --mdata-path")
    mdata = mu.read_h5mu(path)
    prot = mdata["Proteome"].to_df().copy()
    prot.index = prot.index.astype(str)
    gene_symbols = mdata["Proteome"].var["gene_symbol"].reindex(prot.columns).astype(str)
    prot.columns = gene_symbols.str.split(";").str[0].str.upper().values
    prot = prot.loc[:, prot.columns.notna()]
    prot = prot.loc[:, prot.columns.astype(str) != "nan"]
    expr = prot.T.groupby(level=0).mean().T
    expr = expr.loc[:, expr.columns.astype(str) != ""]
    arr = expr.to_numpy(dtype=float)
    diagnostics = {
        "source_h5mu": str(path),
        "n_samples": expr.shape[0],
        "n_gene_symbols": expr.shape[1],
        "global_mean": float(np.nanmean(arr)),
        "global_std": float(np.nanstd(arr)),
        "global_min": float(np.nanmin(arr)),
        "global_max": float(np.nanmax(arr)),
        "nan_fraction": float(np.isnan(arr).mean()),
        "note": "Proteome.to_df() used as final normalized proteome matrix; no log1p transform applied.",
    }
    return expr, diagnostics


def differential_stats(expr: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    expr = expr.reindex(clusters.index)
    rows = []
    for cluster in sorted(clusters.dropna().astype(str).unique(), key=lambda x: (len(x), x)):
        in_cluster = clusters.astype(str).eq(cluster)
        if in_cluster.sum() < 3 or (~in_cluster).sum() < 3:
            continue
        x = expr.loc[in_cluster].to_numpy(dtype=float)
        y = expr.loc[~in_cluster].to_numpy(dtype=float)
        nx = np.sum(np.isfinite(x), axis=0).astype(float)
        ny = np.sum(np.isfinite(y), axis=0).astype(float)
        mean_x = np.nanmean(x, axis=0)
        mean_y = np.nanmean(y, axis=0)
        var_x = np.nanvar(x, axis=0, ddof=1)
        var_y = np.nanvar(y, axis=0, ddof=1)
        se2 = (var_x / nx) + (var_y / ny)
        statistic = (mean_x - mean_y) / np.sqrt(se2)
        df_num = se2**2
        df_den = ((var_x / nx) ** 2 / (nx - 1)) + ((var_y / ny) ** 2 / (ny - 1))
        df = df_num / df_den
        p_value = 2 * student_t.sf(np.abs(statistic), df)
        statistic = np.where(np.isfinite(statistic), statistic, 0.0)
        p_value = np.where(np.isfinite(p_value), p_value, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "cluster": cluster,
                    "gene": expr.columns.astype(str),
                    "mean_in": mean_x,
                    "mean_out": mean_y,
                    "mean_shift": mean_x - mean_y,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
        )
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["FDR"] = out.groupby("cluster")["p_value"].transform(lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
        out = out.sort_values(["cluster", "FDR", "p_value", "statistic"], ascending=[True, True, True, False])
    return out


def resolve_adjustment_covariates(meta: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    covariates = pd.DataFrame(index=meta.index)
    resolved = {}
    for label, candidates in ADJUSTMENT_COVARIATE_CANDIDATES.items():
        col = find_col(meta, candidates)
        if col is None:
            continue
        values = pd.to_numeric(meta[col], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) > 2:
            covariates[label] = values
            resolved[label] = col
    return covariates, resolved


def adjusted_differential_stats(expr: pd.DataFrame, clusters: pd.Series, covariates: pd.DataFrame) -> pd.DataFrame:
    """Cluster coefficient t-statistics from protein ~ cluster + covariates."""
    expr = expr.reindex(clusters.index)
    covariates = covariates.reindex(clusters.index)
    rows = []
    for cluster in sorted(clusters.dropna().astype(str).unique(), key=lambda x: (len(x), x)):
        cluster_indicator = clusters.astype(str).eq(cluster).astype(float)
        base = pd.concat([cluster_indicator.rename("cluster_indicator"), covariates], axis=1)
        in_cluster = clusters.astype(str).eq(cluster)
        if in_cluster.sum() < 3 or (~in_cluster).sum() < 3:
            continue
        valid_samples = base.notna().all(axis=1) & clusters.notna()
        x = base.loc[valid_samples].copy()
        covar_cols = [c for c in x.columns if c != "cluster_indicator"]
        if covar_cols:
            x[covar_cols] = (x[covar_cols] - x[covar_cols].mean(axis=0)).div(x[covar_cols].std(axis=0).replace(0, np.nan), axis=1)
            x = x.dropna(axis=1)
        if x.shape[0] <= x.shape[1] + 3 or "cluster_indicator" not in x.columns or x["cluster_indicator"].nunique() < 2:
            continue

        y = expr.loc[valid_samples].apply(pd.to_numeric, errors="coerce")
        genes = y.columns.astype(str)
        mean_in = expr.loc[in_cluster].mean(axis=0, skipna=True).to_numpy(dtype=float)
        mean_out = expr.loc[~in_cluster].mean(axis=0, skipna=True).to_numpy(dtype=float)
        usable = y.notna().sum(axis=0).ge(x.shape[1] + 4)
        y_filled = y.copy()
        y_filled = y_filled.fillna(y_filled.mean(axis=0))

        x_mat = np.column_stack([np.ones(x.shape[0]), x.to_numpy(dtype=float)])
        pinv = np.linalg.pinv(x_mat)
        beta = pinv @ y_filled.to_numpy(dtype=float)
        fitted = x_mat @ beta
        resid = y_filled.to_numpy(dtype=float) - fitted
        df_resid = x_mat.shape[0] - x_mat.shape[1]
        sigma2 = np.sum(resid**2, axis=0) / df_resid
        xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
        cluster_pos = list(x.columns).index("cluster_indicator") + 1
        se = np.sqrt(np.maximum(xtx_inv[cluster_pos, cluster_pos] * sigma2, 0))
        coef = beta[cluster_pos, :]
        statistic = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
        p_value = 2 * student_t.sf(np.abs(statistic), df_resid)
        statistic = np.where(np.isfinite(statistic) & usable.to_numpy(), statistic, 0.0)
        p_value = np.where(np.isfinite(p_value) & usable.to_numpy(), p_value, 1.0)
        coef = np.where(np.isfinite(coef) & usable.to_numpy(), coef, np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "cluster": cluster,
                    "gene": genes,
                    "mean_in": mean_in,
                    "mean_out": mean_out,
                    "mean_shift": mean_in - mean_out,
                    "cluster_coefficient": coef,
                    "statistic": statistic,
                    "p_value": p_value,
                    "n_complete": y.notna().sum(axis=0).to_numpy(dtype=int),
                    "n_samples_in_model": int(x.shape[0]),
                    "covariates": ";".join([c for c in x.columns if c != "cluster_indicator"]),
                }
            )
        )
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out["FDR"] = out.groupby("cluster")["p_value"].transform(lambda p: multipletests(p.fillna(1), method="fdr_bh")[1])
        out = out.sort_values(["cluster", "FDR", "p_value", "statistic"], ascending=[True, True, True, False])
    return out


def run_prerank(stats: pd.DataFrame, gene_sets: Path | str, args: argparse.Namespace) -> pd.DataFrame:
    if gp is None:
        raise RuntimeError("gseapy is unavailable")
    rows = []
    for cluster, sub in stats.groupby("cluster"):
        rnk = sub[["gene", "statistic"]].dropna().groupby("gene", as_index=False)["statistic"].mean()
        rnk = rnk.sort_values("statistic", ascending=False)
        if rnk.shape[0] < args.min_size:
            continue
        res = gp.prerank(
            rnk=rnk,
            gene_sets=str(gene_sets),
            outdir=None,
            min_size=args.min_size,
            max_size=args.max_size,
            permutation_num=args.permutations,
            threads=args.threads,
            seed=args.seed,
            verbose=False,
            no_plot=True,
        ).res2d
        res["cluster"] = cluster
        rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def gsea_columns(gsea: pd.DataFrame) -> tuple[str, str, str]:
    term_col = "Term" if "Term" in gsea.columns else "Name"
    fdr_col = "FDR q-val" if "FDR q-val" in gsea.columns else "FDR"
    nes_col = "NES"
    return term_col, fdr_col, nes_col


def select_top_terms(gsea: pd.DataFrame, fdr_cutoff: float, top_n: int) -> pd.DataFrame:
    if gsea.empty:
        return pd.DataFrame()
    term_col, fdr_col, nes_col = gsea_columns(gsea)
    sig = gsea[pd.to_numeric(gsea[fdr_col], errors="coerce").le(fdr_cutoff)].copy()
    if sig.empty:
        return sig
    sig[nes_col] = pd.to_numeric(sig[nes_col], errors="coerce")
    rows = []
    for cluster, sub in sig.groupby("cluster"):
        up = sub[sub[nes_col] > 0].sort_values([fdr_col, nes_col], ascending=[True, False]).head(top_n)
        down = sub[sub[nes_col] < 0].sort_values([fdr_col, nes_col], ascending=[True, True]).head(top_n)
        rows.extend([up, down])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=sig.columns)


def select_top_positive_terms(gsea: pd.DataFrame, top_n: int) -> list[str]:
    if gsea.empty:
        return []
    term_col, fdr_col, nes_col = gsea_columns(gsea)
    positive = gsea.copy()
    positive[nes_col] = pd.to_numeric(positive[nes_col], errors="coerce")
    positive[fdr_col] = pd.to_numeric(positive[fdr_col], errors="coerce")
    positive = positive[positive[nes_col].gt(0)].copy()
    terms: list[str] = []
    for _, sub in positive.groupby("cluster"):
        top = sub.sort_values([nes_col, fdr_col, term_col], ascending=[False, True, True]).head(top_n)
        terms.extend(top[term_col].astype(str).tolist())
    return list(dict.fromkeys(terms))


def select_ssgsea_terms(nonredundant: pd.DataFrame, gsea: pd.DataFrame, top_n: int) -> list[str]:
    terms: list[str] = []
    if not nonredundant.empty:
        term_col, _, _ = gsea_columns(nonredundant)
        terms.extend(nonredundant[term_col].astype(str).drop_duplicates().tolist())
    terms.extend(select_top_positive_terms(gsea, top_n))
    return list(dict.fromkeys(terms))


def redundancy_prune(gsea: pd.DataFrame, gene_sets: dict[str, set[str]], universe: set[str], fdr_cutoff: float, overlap_fdr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gsea.empty:
        return gsea, pd.DataFrame()
    term_col, fdr_col, nes_col = gsea_columns(gsea)
    sig = gsea[pd.to_numeric(gsea[fdr_col], errors="coerce").le(fdr_cutoff)].copy()
    if sig.empty:
        sig["redundant"] = False
        return sig, pd.DataFrame()
    sig[nes_col] = pd.to_numeric(sig[nes_col], errors="coerce")
    sig["abs_NES"] = sig[nes_col].abs()
    pair_rows = []
    for cluster, sub in sig.groupby("cluster"):
        sub = sub.sort_values("abs_NES", ascending=False)
        terms = sub[term_col].astype(str).tolist()
        for i, high in enumerate(terms):
            genes_high = gene_sets.get(high, set()) & universe
            for low in terms[i + 1 :]:
                genes_low = gene_sets.get(low, set()) & universe
                if not genes_high or not genes_low:
                    continue
                a = len(genes_high & genes_low)
                b = len(genes_low - genes_high)
                c = len(genes_high - genes_low)
                d = len(universe - genes_high - genes_low)
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                pair_rows.append({"cluster": cluster, "higher_effect_term": high, "lower_effect_term": low, "overlap_n": a, "p_value": p})
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs["FDR"] = multipletests(pairs["p_value"].fillna(1), method="fdr_bh")[1]
    keep_rows = []
    for cluster, sub in sig.groupby("cluster"):
        sub = sub.sort_values("abs_NES", ascending=False).copy()
        kept_terms: list[str] = []
        redundant_flags = []
        superseded_by = []
        for _, row in sub.iterrows():
            term = str(row[term_col])
            sign = np.sign(row[nes_col])
            hit = None
            for kept in kept_terms:
                kept_nes = float(sub.loc[sub[term_col].astype(str).eq(kept), nes_col].iloc[0])
                if np.sign(kept_nes) != sign:
                    continue
                pair_hit = pairs[
                    pairs["cluster"].astype(str).eq(str(cluster))
                    & pairs["higher_effect_term"].astype(str).eq(kept)
                    & pairs["lower_effect_term"].astype(str).eq(term)
                    & pairs["FDR"].le(overlap_fdr)
                ]
                if not pair_hit.empty:
                    hit = kept
                    break
            redundant = hit is not None
            redundant_flags.append(redundant)
            superseded_by.append(hit if hit else "")
            if not redundant:
                kept_terms.append(term)
        sub["redundant"] = redundant_flags
        sub["superseded_by"] = superseded_by
        keep_rows.append(sub)
    pruned = pd.concat(keep_rows, ignore_index=True)
    return pruned, pairs


def run_ssgsea_scores(expr: pd.DataFrame, gene_sets: dict[str, set[str]], selected_terms: list[str], args: argparse.Namespace) -> pd.DataFrame:
    if gp is None or not selected_terms:
        return pd.DataFrame()
    selected = {term: sorted(gene_sets[term] & set(expr.columns)) for term in selected_terms if term in gene_sets}
    selected = {k: v for k, v in selected.items() if args.min_size <= len(v) <= args.max_size}
    if not selected:
        return pd.DataFrame()
    data = expr.T
    res = gp.ssgsea(
        data=data,
        gene_sets=selected,
        outdir=None,
        sample_norm_method="rank",
        min_size=args.min_size,
        max_size=args.max_size,
        threads=args.threads,
        seed=args.seed,
        no_plot=True,
        verbose=False,
    ).res2d
    nes_col = "NES" if "NES" in res.columns else "ES"
    scores = res.pivot(index="Name", columns="Term", values=nes_col)
    scores.index = scores.index.astype(str)
    return scores.apply(pd.to_numeric, errors="coerce")


def gsva_like_cluster_shifts(scores: pd.DataFrame, clusters: pd.Series, fdr_cutoff: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame()
    scores = scores.reindex(clusters.index)
    centered = scores.sub(scores.mean(axis=0), axis=1)
    rows = []
    shift_rows = []
    for term in centered.columns:
        groups = [sub[term].dropna() for _, sub in centered.join(clusters.rename("cluster")).groupby("cluster")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
        else:
            stat, p = np.nan, np.nan
        rows.append({"Term": term, "test": "Kruskal-Wallis", "statistic": stat, "p_value": p})
        for cluster, sub in centered.join(clusters.rename("cluster")).groupby("cluster"):
            shift_rows.append({"Term": term, "cluster": cluster, "mean_shift_vs_cohort": sub[term].mean(skipna=True), "n": sub[term].notna().sum()})
    tests = pd.DataFrame(rows)
    if not tests.empty:
        tests["FDR"] = multipletests(tests["p_value"].fillna(1), method="fdr_bh")[1]
        tests["significantly_variable"] = tests["FDR"].le(fdr_cutoff)
    shifts = pd.DataFrame(shift_rows)
    shifts = shifts.merge(tests[["Term", "FDR", "significantly_variable"]], on="Term", how="left")
    return tests, shifts


def pathway_cluster_correlations(scores: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    scores = scores.reindex(clusters.index)
    rows = []
    all_clusters = sorted(clusters.dropna().astype(str).unique(), key=lambda x: (len(x), x))
    for term in scores.columns:
        y = pd.to_numeric(scores[term], errors="coerce")
        for cluster in all_clusters:
            x = clusters.astype(str).eq(cluster).astype(int)
            valid = y.notna() & clusters.notna()
            if valid.sum() < 5 or x.loc[valid].nunique() < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(x.loc[valid], y.loc[valid])
            rows.append({"Term": term, "cluster": cluster, "spearman_rho": rho, "p_value": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = multipletests(out["p_value"].fillna(1), method="fdr_bh")[1]
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(args.results_dir)
    clusters = meta[args.cluster_col].astype(str)
    sample_map = build_sample_id_map(meta)
    expr, diagnostics = load_gene_expression(args.results_dir, args.mdata_path)
    expr = expr.reindex(meta.index)
    gene_set_ref = resolve_gene_set(args.gene_set)
    if isinstance(gene_set_ref, Path):
        gene_sets = read_gmt(gene_set_ref)
    else:
        # gseapy library names can be used for GSEA, but redundancy and ssGSEA
        # require local gene membership. Keep those steps disabled in this case.
        gene_sets = {}
    universe = set(expr.columns.astype(str))
    diagnostics.update(
        {
            "cluster_col": args.cluster_col,
            "gene_set": args.gene_set,
            "gene_set_resolved": str(gene_set_ref),
            "n_gene_sets_loaded_for_overlap": len(gene_sets),
            "n_universe_genes": len(universe),
            "statistic_method": "Welch t-statistic cluster-vs-rest (Python approximation; not limma/eBayes)",
            "single_sample_method": "gseapy.ssgsea NES (Python approximation of GSVA pathway-by-patient representation)",
        }
    )
    (args.out_dir / "run_config.json").write_text(json.dumps(vars(args) | diagnostics, indent=2, default=str) + "\n")

    write_tsv(sample_map.reset_index(drop=True), args.out_dir / "sample_id_map.tsv")
    stats = differential_stats(expr, clusters)
    write_tsv(stats, args.out_dir / "cluster_vs_rest_gene_statistics.tsv")

    gsea = run_prerank(stats, gene_set_ref, args)
    write_tsv(gsea, args.out_dir / "gsea_prerank_all_terms.tsv")
    top_terms = select_top_terms(gsea, args.gsea_fdr, args.top_n)
    write_tsv(top_terms, args.out_dir / "gsea_top5_up_down_significant_terms.tsv")

    if gene_sets:
        pruned, redundancy_pairs = redundancy_prune(gsea, gene_sets, universe, args.gsea_fdr, args.redundancy_fdr)
    else:
        pruned, redundancy_pairs = gsea.copy(), pd.DataFrame()
        if not pruned.empty:
            pruned["redundant"] = False
            pruned["superseded_by"] = ""
    write_tsv(pruned, args.out_dir / "gsea_significant_terms_with_redundancy.tsv")
    write_tsv(redundancy_pairs, args.out_dir / "gsea_redundancy_pair_tests.tsv")
    nonredundant = pruned[~pruned.get("redundant", False)].copy() if not pruned.empty else pd.DataFrame()
    write_tsv(nonredundant, args.out_dir / "gsea_nonredundant_significant_terms.tsv")

    covariates, covariate_sources = resolve_adjustment_covariates(meta)
    if not args.skip_adjusted_gsea and {"tumor_cell_content", "missing_fraction"}.issubset(covariates.columns):
        adjusted_stats = adjusted_differential_stats(expr, clusters, covariates[["tumor_cell_content", "missing_fraction"]])
        write_tsv(adjusted_stats, args.out_dir / "sensitivity_tumor_missingness_adjusted_gene_statistics.tsv")
        adjusted_gsea = run_prerank(adjusted_stats, gene_set_ref, args)
        write_tsv(adjusted_gsea, args.out_dir / "sensitivity_tumor_missingness_adjusted_gsea_prerank_all_terms.tsv")
        save_text(json.dumps(covariate_sources, indent=2) + "\n", args.out_dir / "sensitivity_tumor_missingness_adjustment_covariates.json")
    else:
        reason = "adjusted GSEA skipped by flag" if args.skip_adjusted_gsea else "tumor content and/or missing fraction covariates unavailable"
        write_tsv(pd.DataFrame({"status": ["NA"], "reason": [reason]}), args.out_dir / "sensitivity_tumor_missingness_adjusted_gene_statistics.tsv")
        write_tsv(pd.DataFrame({"status": ["NA"], "reason": [reason]}), args.out_dir / "sensitivity_tumor_missingness_adjusted_gsea_prerank_all_terms.tsv")

    selected_terms = select_ssgsea_terms(nonredundant, gsea, args.top_n)
    if not args.skip_ssgsea and gene_sets and selected_terms:
        scores = run_ssgsea_scores(expr, gene_sets, selected_terms, args)
        scores.to_csv(args.out_dir / "ssgsea_nonredundant_pathway_scores.tsv", sep="\t")
        tests, shifts = gsva_like_cluster_shifts(scores, clusters, args.gsva_fdr)
        correlations = pathway_cluster_correlations(scores, clusters)
        write_tsv(tests, args.out_dir / "ssgsea_pathway_kruskal_tests.tsv")
        write_tsv(shifts, args.out_dir / "ssgsea_cluster_mean_shifts.tsv")
        write_tsv(correlations, args.out_dir / "ssgsea_pathway_cluster_spearman.tsv")
    else:
        write_tsv(pd.DataFrame({"status": ["NA"], "reason": ["ssGSEA skipped or no local non-redundant gene sets"]}), args.out_dir / "ssgsea_pathway_kruskal_tests.tsv")
        write_tsv(pd.DataFrame({"status": ["NA"], "reason": ["ssGSEA skipped or no local non-redundant gene sets"]}), args.out_dir / "ssgsea_cluster_mean_shifts.tsv")
        write_tsv(pd.DataFrame({"status": ["NA"], "reason": ["ssGSEA skipped or no local non-redundant gene sets"]}), args.out_dir / "ssgsea_pathway_cluster_spearman.tsv")

    report = [
        "# GO/Pathway GSEA + ssGSEA",
        "",
        f"Expression matrix: `{diagnostics['source_h5mu']}`",
        f"Gene set: `{gene_set_ref}`",
        f"Samples: {expr.shape[0]}",
        f"Genes: {expr.shape[1]}",
        "",
        "## Method Notes",
        "- Uses final Proteome matrix from the h5mu as-is; no MuVI latent factors and no log1p transform.",
        "- Cluster-vs-rest ranks use Welch t-statistics. This is close in shape but not identical to limma/eBayes.",
        "- Redundancy pruning removes lower-effect pathways when Fisher overlap FDR < threshold and NES signs match.",
        "- Pathway-by-patient scores use gseapy ssGSEA NES as a Python approximation of GSVA.",
        "- The main GSEA outputs are unadjusted. Sensitivity GSEA ranks proteins by the cluster coefficient t-statistic from `protein ~ cluster + tumor_cell_content + missing_fraction`.",
        "",
        "## Key Outputs",
        "- `gsea_prerank_all_terms.tsv`",
        "- `gsea_top5_up_down_significant_terms.tsv`",
        "- `gsea_nonredundant_significant_terms.tsv`",
        "- `ssgsea_nonredundant_pathway_scores.tsv`",
        "- `ssgsea_pathway_kruskal_tests.tsv`",
        "- `ssgsea_cluster_mean_shifts.tsv`",
        "- `ssgsea_pathway_cluster_spearman.tsv`",
        "- `sensitivity_tumor_missingness_adjusted_gene_statistics.tsv`",
        "- `sensitivity_tumor_missingness_adjusted_gsea_prerank_all_terms.tsv`",
        "",
        "Run `python -m cancerscout.plot_gsea --out-dir <this dir>` to render figures.",
    ]
    report_name = f"gsea_{Path(args.out_dir).name}.md"
    save_text("\n".join(report) + "\n", REPORTS_DIR / report_name)
    print(f"Wrote pathway outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
