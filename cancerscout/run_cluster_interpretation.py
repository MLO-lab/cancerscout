"""Interpret locked CancerScout MuVI consensus clusters.

This script is the high-level downstream interpretation layer for the locked
CRC analysis. It assumes the active model folder contains the locked MuVI
handoff and that ``cptac_prediction`` contains the final CPTAC transfer labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2, chi2_contingency, kruskal
from statsmodels.stats.multitest import multipletests

from cancerscout.io import build_sample_id_map, find_col, load_downstream_data, save_text, write_tsv
from cancerscout.survival import normalize_event, survival_summary
from cancerscout.plot_style import CLUSTER_SHORT_NAMES
from cancerscout.paths import ACTIVE_MODEL_DIR, CLUSTER_INTERPRETATION_RESULTS_DIR, GSEA_RESULTS_DIR, REPORTS_DIR


DEFAULT_METADATA = {
    "MSI_status": ["MSI consensus", "MSI Status", "new_MSI_final", "new_MSI_consensus"],
    "CPTAC_prediction": ["cptac_prediction", "cptac_all_pred", "CPTAC"],
    "UICC_stage": ["UICC", "new_UICC", "new_pUICC"],
    "tumor_location": ["Lokalisation", "new_Lokalisation", "new_Lokalisation_grouped"],
    "grade": ["Proteome:new_Grading", "new_Grading", "Proteome:new_G", "new_G", "G"],
    "age": ["Alter bei Diagnose", "new_age"],
    "sex": ["Geschlecht", "new_sex"],
    "tumor_cell_content": ["Tumorzellgehalt (Mol+Prot)"],
    "acquisition_batch": ["Proteome:new_acquisition_batch", "new_acquisition_batch"],
    "plate": ["Proteome:new_Plate", "new_Plate"],
    "missing_fraction": ["Proteome:new_missing_fraction", "new_missing_fraction"],
    "observed_fraction": ["Proteome:new_observed_fraction", "new_observed_fraction"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=CLUSTER_INTERPRETATION_RESULTS_DIR)
    parser.add_argument("--cluster-col", default="leiden_cons")
    parser.add_argument("--pathway-file", type=Path, default=GSEA_RESULTS_DIR / "reactome/gsea_nonredundant_significant_terms.tsv")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-cox-events", type=int, default=20)
    return parser.parse_args()


def clean_category(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    return x.mask(x.str.lower().isin(["nan", "none", "", "na", "x", "xxx"]))


def clean_metadata_value(label: str, s: pd.Series) -> pd.Series:
    x = clean_category(s)
    if label == "grade":
        return x.where(x.str.match(r"^\d+(-\d+)?$", na=False))
    return x


def clean_plot_category(s: pd.Series, missing_label: str = "NA") -> pd.Series:
    x = clean_category(s)
    return x.fillna(missing_label).astype(str)


def cramers_v(tab: pd.DataFrame) -> float:
    if tab.empty or min(tab.shape) < 2:
        return np.nan
    chi2_stat = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    denom = n * (min(tab.shape) - 1)
    return float(np.sqrt(chi2_stat / denom)) if denom else np.nan


def eta_squared(groups: list[pd.Series]) -> float:
    values = pd.concat(groups).dropna()
    if values.empty:
        return np.nan
    grand = values.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups if len(g))
    ss_total = ((values - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total else np.nan


def resolve_metadata_columns(meta: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    for label, candidates in DEFAULT_METADATA.items():
        col = find_col(meta, candidates)
        if col:
            resolved[label] = col
    return resolved


def metadata_decomposition(meta: pd.DataFrame, clusters: pd.Series, columns: dict[str, str], out_dir: Path) -> pd.DataFrame:
    rows = []
    force_numeric = {"age", "tumor_cell_content", "plate", "missing_fraction", "observed_fraction"}
    for label, col in columns.items():
        raw = meta[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_like = label in force_numeric or (numeric.notna().mean() >= 0.8 and raw.nunique(dropna=True) > 8)
        tmp = pd.DataFrame({"cluster": clusters, "value_raw": raw, "value_num": numeric}, index=meta.index)
        if numeric_like:
            for stale in [out_dir / f"metadata_counts_{label}.tsv", out_dir / f"metadata_fraction_{label}.tsv"]:
                if stale.exists():
                    stale.unlink()
            use = tmp.dropna(subset=["cluster", "value_num"])
            groups = [sub["value_num"] for _, sub in use.groupby("cluster")]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                stat, p = kruskal(*groups)
            else:
                stat, p = np.nan, np.nan
            rows.append(
                {
                    "variable": label,
                    "source_column": col,
                    "type": "numeric",
                    "n": len(use),
                    "test": "Kruskal-Wallis",
                    "statistic": stat,
                    "p_value": p,
                    "effect_size": eta_squared(groups),
                    "effect_size_name": "eta_squared",
                }
            )
            summary = use.groupby("cluster")["value_num"].agg(["count", "mean", "median", "std"]).reset_index()
            write_tsv(summary, out_dir / f"metadata_numeric_{label}.tsv")
            # plot-ready per-sample values (cluster, value) so plot_cluster_interpretation can draw box/strip plots
            write_tsv(use[["cluster", "value_num"]].reset_index().rename(columns={"index": "analysis_sample_id"}), out_dir / f"metadata_numeric_{label}_samples.tsv")
        else:
            stale = out_dir / f"metadata_numeric_{label}.tsv"
            if stale.exists():
                stale.unlink()
            cat = clean_metadata_value(label, raw)
            use = pd.DataFrame({"cluster": clusters, "value": cat}, index=meta.index).dropna()
            tab = pd.crosstab(use["cluster"].astype(str), use["value"].astype(str))
            if tab.shape[0] >= 2 and tab.shape[1] >= 2:
                stat, p, _, _ = chi2_contingency(tab)
            else:
                stat, p = np.nan, np.nan
            rows.append(
                {
                    "variable": label,
                    "source_column": col,
                    "type": "categorical",
                    "n": int(tab.to_numpy().sum()) if not tab.empty else 0,
                    "test": "Chi-square",
                    "statistic": stat,
                    "p_value": p,
                    "effect_size": cramers_v(tab),
                    "effect_size_name": "cramers_v",
                }
            )
            write_tsv(tab.reset_index(), out_dir / f"metadata_counts_{label}.tsv")
            frac = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0)
            write_tsv(frac.reset_index(), out_dir / f"metadata_fraction_{label}.tsv")
    out = pd.DataFrame(rows)
    if not out.empty:
        out["FDR"] = multipletests(out["p_value"].fillna(1), method="fdr_bh")[1]
        out["flag_metadata_imbalance"] = (out["FDR"] < 0.05) & (out["effect_size"].fillna(0) >= 0.1)
        out = out.sort_values(["FDR", "effect_size"], ascending=[True, False])
    return out


def prepare_cox_frame(meta: pd.DataFrame, clusters: pd.Series, time_col: str, event_col: str, covariates: dict[str, str]) -> pd.DataFrame:
    df = pd.DataFrame({"time": pd.to_numeric(meta[time_col], errors="coerce"), "event": normalize_event(meta[event_col]), "cluster": clusters.astype(str)}, index=meta.index)
    for label, col in covariates.items():
        if label in {"age", "tumor_cell_content", "missing_fraction", "observed_fraction"}:
            df[label] = pd.to_numeric(meta[col], errors="coerce")
        else:
            df[label] = clean_metadata_value(label, meta[col])
    df = df.dropna(subset=["time", "event", "cluster"])
    df = df[df["time"] > 0].copy()
    return df


def encode_cox(df: pd.DataFrame, use_cluster: bool, covariate_labels: list[str]) -> pd.DataFrame:
    cols = ["time", "event"]
    parts = [df[cols].copy()]
    if use_cluster:
        parts.append(pd.get_dummies(df["cluster"].astype(str), prefix="cluster", drop_first=True, dtype=float))
    for label in covariate_labels:
        if label not in df.columns:
            continue
        x = df[label]
        if pd.api.types.is_numeric_dtype(x):
            z = pd.to_numeric(x, errors="coerce")
            if z.notna().sum() >= 30 and z.std(skipna=True) > 0:
                parts.append(((z - z.mean()) / z.std()).rename(label).to_frame())
        else:
            counts = x.value_counts(dropna=True)
            keep = counts[counts >= 20].index
            y = x.where(x.isin(keep))
            if y.nunique(dropna=True) >= 2:
                parts.append(pd.get_dummies(y.astype(str), prefix=label, drop_first=True, dtype=float))
    out = pd.concat(parts, axis=1).dropna()
    return out.loc[:, out.nunique(dropna=True) > 1]


def fit_adjusted_cox(meta: pd.DataFrame, clusters: pd.Series, columns: dict[str, str], out_dir: Path) -> pd.DataFrame:
    endpoints = [
        ("OS", ["OS Monate", "new_OS time"], ["OS Status", "new_Overall Survival (OS) Status"]),
        ("PFS", ["PFS Monate", "new_PFS time"], ["PFS Status", "new_Progression Free Survival (PFS) Status"]),
    ]
    covariate_priority = ["age", "sex", "UICC_stage", "grade", "MSI_status", "tumor_location", "tumor_cell_content", "acquisition_batch"]
    rows = []
    for endpoint, time_candidates, event_candidates in endpoints:
        time_col = find_col(meta, time_candidates)
        event_col = find_col(meta, event_candidates)
        if not time_col or not event_col:
            rows.append({"endpoint": endpoint, "status": "NA", "reason": "time/event column unavailable"})
            continue
        covs = {k: columns[k] for k in covariate_priority if k in columns}
        base = prepare_cox_frame(meta, clusters, time_col, event_col, covs)
        if base.empty or base["event"].sum() < 20 or base["cluster"].nunique() < 2:
            rows.append({"endpoint": endpoint, "status": "NA", "reason": "insufficient survival data after cleaning", "n": len(base), "events": base["event"].sum() if not base.empty else 0})
            continue
        unadjusted = encode_cox(base, use_cluster=True, covariate_labels=[])
        full = encode_cox(base, use_cluster=True, covariate_labels=covariate_priority)
        reduced = encode_cox(base, use_cluster=False, covariate_labels=covariate_priority)
        try:
            cph_un = CoxPHFitter(penalizer=0.01).fit(unadjusted, "time", "event")
            cph_full = CoxPHFitter(penalizer=0.01).fit(full, "time", "event")
            cph_reduced = CoxPHFitter(penalizer=0.01).fit(reduced, "time", "event") if reduced.shape[1] > 2 else None
            un_p = cph_un.log_likelihood_ratio_test().p_value
            if cph_reduced is not None:
                lr = 2 * (cph_full.log_likelihood_ - cph_reduced.log_likelihood_)
                df_delta = max(cph_full.params_.shape[0] - cph_reduced.params_.shape[0], 1)
                adj_p = chi2.sf(lr, df_delta)
            else:
                lr, df_delta, adj_p = cph_full.log_likelihood_ratio_test().test_statistic, cph_full.params_.shape[0], cph_full.log_likelihood_ratio_test().p_value
            cph_un.summary.to_csv(out_dir / f"cox_{endpoint.lower()}_unadjusted_summary.tsv", sep="\t")
            cph_full.summary.to_csv(out_dir / f"cox_{endpoint.lower()}_adjusted_summary.tsv", sep="\t")
            rows.append(
                {
                    "endpoint": endpoint,
                    "status": "OK",
                    "n": len(full),
                    "events": int(full["event"].sum()),
                    "time_col": time_col,
                    "event_col": event_col,
                    "covariates_considered": ",".join([c for c in covariate_priority if c in columns]),
                    "cox_unadjusted_cluster_p": un_p,
                    "cox_adjusted_cluster_lrt": lr,
                    "cox_adjusted_cluster_df": df_delta,
                    "cox_adjusted_cluster_p": adj_p,
                }
            )
        except Exception as exc:
            rows.append({"endpoint": endpoint, "status": "FAIL", "reason": str(exc), "n": len(base), "events": int(base["event"].sum())})
    return pd.DataFrame(rows)


def pathway_summary(pathway_file: Path, out_dir: Path, top_n: int) -> pd.DataFrame:
    if not pathway_file.exists():
        return pd.DataFrame()
    pw = pd.read_csv(pathway_file, sep="\t")
    if pw.empty or "cluster" not in pw.columns:
        return pd.DataFrame()
    fdr_col = "FDR q-val" if "FDR q-val" in pw.columns else "FDR" if "FDR" in pw.columns else None
    term_col = "Term" if "Term" in pw.columns else "Name" if "Name" in pw.columns else None
    if not fdr_col or not term_col or "NES" not in pw.columns:
        return pd.DataFrame()
    pw = pw.copy()
    pw["cluster"] = pw["cluster"].astype(str)
    pw["abs_NES"] = pd.to_numeric(pw["NES"], errors="coerce").abs()
    top = (
        pw.sort_values(["cluster", "abs_NES", fdr_col], ascending=[True, False, True])
        .groupby("cluster", sort=False)
        .head(top_n)
        .copy()
    )
    write_tsv(top, out_dir / "top_pathways_per_cluster.tsv")
    return top


def fraction(series: pd.Series, value: str) -> float:
    s = clean_category(series).dropna().astype(str)
    if s.empty:
        return np.nan
    return float(s.str.upper().eq(value.upper()).mean())


def cluster_interpretation_table(
    meta: pd.DataFrame,
    clusters: pd.Series,
    columns: dict[str, str],
    pathway_top: pd.DataFrame,
    survival: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    fdr_col = "FDR q-val" if "FDR q-val" in pathway_top.columns else "FDR" if "FDR" in pathway_top.columns else None
    term_col = "Term" if "Term" in pathway_top.columns else "Name" if "Name" in pathway_top.columns else None
    for cluster, idx in clusters.groupby(clusters).groups.items():
        sub = meta.loc[idx]
        row = {"cluster": str(cluster), "n": len(sub)}
        if "MSI_status" in columns:
            row["MSI_fraction"] = fraction(sub[columns["MSI_status"]], "MSI")
        if "CPTAC_prediction" in columns:
            cptac_values = clean_metadata_value("CPTAC_prediction", sub[columns["CPTAC_prediction"]])
            counts = cptac_values.value_counts()
            row["top_CPTAC_prediction"] = counts.index[0] if len(counts) else "NA"
            row["top_CPTAC_fraction"] = counts.iloc[0] / counts.sum() if len(counts) else np.nan
            row["mesenchymal_fraction"] = cptac_values.astype(str).str.contains("Mesenchymal", case=False, na=False).mean()
        for label in ["UICC_stage", "grade", "tumor_location", "sex", "acquisition_batch"]:
            if label in columns:
                counts = clean_metadata_value(label, sub[columns[label]]).value_counts()
                row[f"top_{label}"] = counts.index[0] if len(counts) else "NA"
                row[f"top_{label}_fraction"] = counts.iloc[0] / counts.sum() if len(counts) else np.nan
        if not pathway_top.empty and fdr_col and term_col:
            psub = pathway_top[pathway_top["cluster"].astype(str).eq(str(cluster))].sort_values(fdr_col).head(top_n)
            row["top_pathways"] = "; ".join(psub[term_col].astype(str).tolist())
        os_row = survival[(survival.get("endpoint") == "OS") & (survival.get("group") == str(cluster))] if not survival.empty else pd.DataFrame()
        pfs_row = survival[(survival.get("endpoint") == "PFS") & (survival.get("group") == str(cluster))] if not survival.empty else pd.DataFrame()
        if not os_row.empty:
            row["OS_events"] = int(os_row["events"].iloc[0])
            row["OS_n"] = int(os_row["n"].iloc[0])
        if not pfs_row.empty:
            row["PFS_events"] = int(pfs_row["events"].iloc[0])
            row["PFS_n"] = int(pfs_row["n"].iloc[0])
        row["working_interpretation"] = CLUSTER_SHORT_NAMES.get(str(cluster), "review")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_downstream_data(args.results_dir)
    meta = data.metadata.copy()
    clusters = meta[args.cluster_col].astype(str)
    sample_map = build_sample_id_map(meta)
    columns = resolve_metadata_columns(meta)
    (args.out_dir / "resolved_columns.json").write_text(json.dumps(columns, indent=2) + "\n")

    cluster_sizes = clusters.value_counts().sort_index().rename_axis("cluster").reset_index(name="n")
    write_tsv(cluster_sizes, args.out_dir / "cluster_sizes.tsv")

    # UMAP coordinates with the categorical annotations the plot side will color by.
    if "X_umap" in data.factor_adata.obsm:
        umap = pd.DataFrame(data.factor_adata.obsm["X_umap"], index=data.factor_adata.obs_names.astype(str), columns=["UMAP1", "UMAP2"])
        umap.index.name = "analysis_sample_id"
        umap["cluster"] = clusters.reindex(umap.index).values
        for label, col in [("CPTAC_prediction", columns.get("CPTAC_prediction")), ("MSI_status", columns.get("MSI_status"))]:
            if col:
                umap[label] = clean_plot_category(meta[col]).reindex(umap.index).values
        write_tsv(umap.reset_index(), args.out_dir / "umap_coordinates.tsv")

    metadata_tests = metadata_decomposition(meta, clusters, columns, args.out_dir)
    write_tsv(metadata_tests, args.out_dir / "metadata_association_tests.tsv")

    # Unadjusted survival: write plot-ready KM data + per-cluster summary.
    survival_rows = []
    for endpoint, time_candidates, event_candidates in [
        ("OS", ["OS Monate", "new_OS time"], ["OS Status", "new_Overall Survival (OS) Status"]),
        ("PFS", ["PFS Monate", "new_PFS time"], ["PFS Status", "new_Progression Free Survival (PFS) Status"]),
    ]:
        time_col = find_col(meta, time_candidates)
        event_col = find_col(meta, event_candidates)
        if time_col and event_col:
            df = meta.assign(cluster=clusters)
            survival_rows.append(survival_summary(df, "cluster", time_col, event_col, endpoint))
            km = pd.DataFrame(
                {
                    "cluster": clusters.reindex(meta.index).values,
                    "time": pd.to_numeric(meta[time_col], errors="coerce").values,
                    "event": normalize_event(meta[event_col]).values,
                },
                index=meta.index,
            )
            km.index.name = "analysis_sample_id"
            write_tsv(km.dropna(subset=["time", "event", "cluster"]).reset_index(), args.out_dir / f"km_data_{endpoint}.tsv")
    survival = pd.concat([x for x in survival_rows if x is not None and not x.empty], ignore_index=True) if survival_rows else pd.DataFrame()
    write_tsv(survival, args.out_dir / "survival_unadjusted_summary.tsv")

    cox = fit_adjusted_cox(meta, clusters, columns, args.out_dir)
    write_tsv(cox, args.out_dir / "survival_adjusted_cox_tests.tsv")

    pathway_top = pathway_summary(args.pathway_file, args.out_dir, args.top_n)

    interpretation = cluster_interpretation_table(meta, clusters, columns, pathway_top, survival, args.top_n)
    write_tsv(interpretation, args.out_dir / "cluster_interpretation_summary.tsv")
    interpretation_csv = interpretation.copy()
    interpretation_csv.insert(0, "cluster_size_rank", range(1, len(interpretation_csv) + 1))
    interpretation_csv.to_csv(args.out_dir / "cluster_interpretation_summary.csv", index=False)

    final = pd.DataFrame(index=meta.index)
    final.index.name = "analysis_sample_id"
    final["bid"] = sample_map["bid"]
    final["cluster"] = clusters
    if "CPTAC_prediction" in columns:
        final["cptac_prediction"] = meta[columns["CPTAC_prediction"]]
    if "MSI_status" in columns:
        final["MSI_status"] = meta[columns["MSI_status"]]
    final.to_csv(args.out_dir / "sample_cluster_identity_table.csv")

    report_lines = [
        "# Locked Cluster Interpretation",
        "",
        f"Results directory: `{args.results_dir.resolve()}`",
        f"Cluster column: `{args.cluster_col}`",
        f"Samples: {meta.shape[0]}",
        "",
        "## Cluster Sizes",
        cluster_sizes.to_markdown(index=False),
        "",
        "## Adjusted Survival",
        cox.to_markdown(index=False) if not cox.empty else "No adjusted Cox model available.",
        "",
        "## Strong Metadata Associations",
    ]
    if not metadata_tests.empty:
        strong = metadata_tests[metadata_tests["flag_metadata_imbalance"]].head(12)
        report_lines.append(strong.to_markdown(index=False) if not strong.empty else "No metadata variable crossed the flag threshold.")
    else:
        report_lines.append("No metadata tests available.")
    report_lines.extend(
        [
            "",
            "## Cluster Interpretation Table",
            interpretation.to_markdown(index=False),
            "",
            "## Supplementary Material",
            "- Per-cluster volcano plots (`figures/cluster_interpretation/volcano/`) and the full differential-abundance table (`results/cluster_interpretation/volcano_marker_rankings.tsv`) are supplementary.",
            "- Cell-type signature heatmaps in this directory are supplementary; cell-type deconvolution is not part of the main paper figures.",
            "",
            "## Caveats",
            "- Association tests identify imbalance, not causality.",
            "- Adjusted Cox models are exploratory and depend on complete covariate availability.",
            "- The 18-sample cluster is biologically interesting but underpowered for definitive survival or enrichment conclusions.",
        ]
    )
    save_text("\n".join(report_lines) + "\n", REPORTS_DIR / "cluster_interpretation.md")

    print(f"Wrote cluster interpretation tables to {args.out_dir.resolve()}")
    print(f"Run `python -m cancerscout.plot_cluster_interpretation --out-dir {args.out_dir}` to render figures.")


if __name__ == "__main__":
    main()
