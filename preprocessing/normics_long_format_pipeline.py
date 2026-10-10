#!/usr/bin/env python
"""Stage 1: Biognosys long-format protein report -> Normics VSN -> limma plate correction.

The report has one row per (run, protein) with columns:
  r_file_name, pg_protein_accessions, pg_genes, pg_protein_names,
  pg_protein_descriptions, pg_quantity, pg_qvalue

The report is pivoted to proteins x runs, BID tumours, BatchQC and QC-pool runs
are kept, the best run per cleaned sample is selected (lowest missing fraction,
then original over rerun), clinical metadata and the MSI consensus are joined,
and the Normics inputs (50% missingness filter, top 5% of proteins by the
Normics rank score as the VSN fitting subset) are written. VSN and limma plate
correction run in the `normics-r` environment via r/normics_normalize.R and
r/plate_batch_correct.R. The raw matrices read by the QC figures are written
alongside.

Output: data/raw/normics_r14_goettingen_output_20260428/ (ACTIVE_RAW_NORMICS_DIR).
run_normics_no_outliers then re-fits VSN and limma on this output without the
two outlier tumours.

Usage:
    conda run -n proj python preprocessing/normics_long_format_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cancerscout.paths import (
    ACTIVE_RAW_NORMICS_DIR,
    EXTERNAL_DATA_DIR,
    R_DIR,
    RAW_TSV_GOETTINGEN_R14,
    RSCRIPT_NORMICS_R_BIN,
)


DEFAULT_META = EXTERNAL_DATA_DIR / "meta.csv"
DEFAULT_MSI = EXTERNAL_DATA_DIR / "MSI_consense.csv"
DEFAULT_R_SCRIPT = R_DIR / "normics_normalize.R"
DEFAULT_PLATE_R_SCRIPT = R_DIR / "plate_batch_correct.R"
RESHAPED_NAME = "r14_goettingen_raw_long_reshaped_samples_by_protein"


@dataclass
class NormicsConfig:
    min_data_values: float = 1.0
    handle_zero_as_nan: bool = True
    reverse_log2: bool = False
    reverse_log: bool = False
    tmt_ratio: bool = False
    reduce_correlation_by: int = 1
    subset_fraction: float = 0.10
    subset_min_n: int = 0
    top_var_corr_proteins: int = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normics pipeline for long-format Biognosys reports.")
    parser.add_argument("--report", type=Path, default=RAW_TSV_GOETTINGEN_R14)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--msi", type=Path, default=DEFAULT_MSI)
    parser.add_argument("--outdir", type=Path, default=ACTIVE_RAW_NORMICS_DIR)
    parser.add_argument("--r-script", type=Path, default=DEFAULT_R_SCRIPT)
    parser.add_argument("--plate-r-script", type=Path, default=DEFAULT_PLATE_R_SCRIPT)
    parser.add_argument("--min-data-values", type=float, default=0.5)
    parser.add_argument("--subset-fraction", type=float, default=0.05)
    parser.add_argument("--subset-min-n", type=int, default=0)
    parser.add_argument("--reduce-correlation-by", type=int, default=1)
    return parser.parse_args()


def ensure_dirs(outdir: Path) -> dict[str, Path]:
    paths = {
        "root": outdir,
        "intermediate": outdir / "intermediate",
        "final": outdir / "final",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def strip_leading_index(name: str) -> str:
    return re.sub(r"^\[\d+\]\s*", "", str(name))


def strip_rerun_markers(name: str) -> str:
    name = re.sub(r"_X\d+", "", str(name))
    name = re.sub(r" \(\d+\)", "", name)
    return name


def clean_sample_name(name: str) -> str:
    return strip_rerun_markers(strip_leading_index(name))


def sample_group(name: str) -> str:
    text = str(name)
    if "Blank-" in text:
        return "Blank"
    if "BatchQC-" in text:
        return "BatchQC"
    if "QC-pool-" in text or "QC_pool-" in text:
        return "QC-pool"
    if re.search(r"BID\d{4}", text):
        return "BID"
    return "Other"


def load_long_report(report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a long-format report and return (matrix, protein_annotations).

    matrix: proteins × samples DataFrame of raw intensities (index = protein_id,
            columns = r_file_name values, zeros left as-is for the caller to handle).
    protein_annotations: one row per protein with annotation columns.
    """
    print(f"Loading long-format report: {report_path}")

    # Read quantity columns only for the pivot (saves memory on large files).
    df_quant = pd.read_csv(
        report_path,
        sep="\t",
        usecols=["r_file_name", "pg_protein_accessions", "pg_quantity"],
    )
    matrix = df_quant.pivot(
        index="pg_protein_accessions",
        columns="r_file_name",
        values="pg_quantity",
    )
    matrix.columns.name = None
    matrix.index.name = "protein_id"

    # Annotation table: one row per protein.
    ann_cols = ["pg_protein_accessions", "pg_genes", "pg_protein_names", "pg_protein_descriptions"]
    df_ann = pd.read_csv(report_path, sep="\t", usecols=ann_cols)
    protein_annotations = (
        df_ann.drop_duplicates("pg_protein_accessions")
        .rename(columns={
            "pg_protein_accessions": "protein_id",
            "pg_genes": "PG.Genes",
            "pg_protein_names": "PG.ProteinGroups",
            "pg_protein_descriptions": "PG.ProteinDescriptions",
        })
        .set_index("protein_id")
    )
    return matrix, protein_annotations


def parse_sample_metadata_long(sample_names: list[str]) -> pd.DataFrame:
    """Parse sample metadata from long-format r_file_name values."""
    records: list[dict] = []
    for raw_name in sample_names:
        raw_name = str(raw_name)
        clean_name = clean_sample_name(raw_name)
        raw_core = strip_leading_index(raw_name)
        clean_core = clean_sample_name(raw_core)

        sample_token_match = re.search(
            r"SExp\d{2}-(BID\d{4}|BatchQC-\d+|QC-pool-\d+|QC_pool-\d+|Blank-\d+)",
            clean_name,
        )
        bid_match = re.search(r"(BID\d{4})", clean_name)
        run_match = re.search(r"_R(\d+)(?:_X\d+)?$", clean_name)
        batch_match = re.search(r"I_(D\d{6})_", clean_name)
        rerun_match = re.search(r"_X\d+| \(\d+\)", raw_name)

        records.append({
            "raw_sample": raw_name,
            "clean_sample": clean_name,
            "raw_core": raw_core,
            "sample_label": re.split(r"SExp\d{2}-", clean_core, maxsplit=1)[-1],
            "sample_group": sample_group(clean_name),
            "sample_token": (
                sample_token_match.group(1).replace("QC_pool", "QC-pool")
                if sample_token_match
                else np.nan
            ),
            "bid": bid_match.group(1) if bid_match else np.nan,
            "bid_numeric": int(bid_match.group(1).replace("BID", "")) if bid_match else np.nan,
            "run": int(run_match.group(1)) if run_match else np.nan,
            "acquisition_batch": batch_match.group(1) if batch_match else np.nan,
            "is_rerun_column": bool(rerun_match),
        })
    return pd.DataFrame.from_records(records)


def select_best_reruns(matrix_raw: pd.DataFrame, sample_info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = sample_info.copy()
    audit["missing_fraction"] = matrix_raw.isna().mean(axis=0).reindex(audit["raw_sample"]).to_numpy()
    audit["observed_fraction"] = 1 - audit["missing_fraction"]
    audit = audit.sort_values(
        ["clean_sample", "missing_fraction", "is_rerun_column", "raw_sample"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    audit["selected"] = ~audit.duplicated(subset="clean_sample", keep="first")

    selected = audit.loc[audit["selected"]].copy()
    matrix_selected = matrix_raw.loc[:, selected["raw_sample"]].copy()
    matrix_selected.columns = selected["clean_sample"].tolist()
    matrix_selected.columns.name = None

    selected_meta = selected[
        [
            "clean_sample",
            "sample_group",
            "sample_label",
            "sample_token",
            "bid",
            "bid_numeric",
            "run",
            "acquisition_batch",
            "raw_sample",
            "missing_fraction",
            "observed_fraction",
            "is_rerun_column",
        ]
    ].copy()
    selected_meta = selected_meta.rename(
        columns={
            "clean_sample": "sample",
            "raw_sample": "selected_raw_sample",
            "is_rerun_column": "selected_is_rerun",
        }
    )
    rerun_counts = (
        sample_info.groupby("clean_sample", sort=False)
        .agg(
            n_source_columns=("raw_sample", "size"),
            n_rerun_columns=("is_rerun_column", "sum"),
            source_columns=("raw_sample", lambda values: " | ".join(values)),
        )
        .reset_index()
        .rename(columns={"clean_sample": "sample"})
    )
    selected_meta = selected_meta.merge(rerun_counts, on="sample", how="left")
    return matrix_selected, selected_meta, audit


def merge_external_metadata(sample_meta: pd.DataFrame, meta_path: Path, msi_path: Path) -> pd.DataFrame:
    merged = sample_meta.copy()

    if meta_path.exists():
        meta_df = pd.read_csv(meta_path, index_col=0)
        meta_df.index = meta_df.index.astype(str)
        meta_df.index.name = "bid"
        merged = merged.merge(meta_df, left_on="bid", right_index=True, how="left")

    if msi_path.exists():
        msi_df = pd.read_csv(msi_path, sep=";", encoding="utf-8-sig")
        first_col = msi_df.columns[0]
        msi_df = msi_df.rename(columns={first_col: "bid_numeric", "MSI consensus": "MSI_consensus"})
        msi_df["bid_numeric"] = pd.to_numeric(msi_df["bid_numeric"], errors="coerce")
        merged = merged.merge(msi_df[["bid_numeric", "MSI_consensus"]], on="bid_numeric", how="left")

    msi_candidates = [
        col
        for col in ["MSI_consensus", "MSI consensus", "MSI Status", "MSI.Status"]
        if col in merged.columns
    ]
    if msi_candidates:
        msi_final = merged[msi_candidates[0]].copy()
        for col in msi_candidates[1:]:
            msi_final = msi_final.combine_first(merged[col])
        merged["MSI_final"] = msi_final
        merged["MSI_final"] = merged["MSI_final"].replace({"n/a": np.nan, "NA": np.nan})

    return merged


def preprocess_matrix(df: pd.DataFrame, cfg: NormicsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if cfg.handle_zero_as_nan:
        df = df.replace(0, np.nan)
    else:
        df = df.replace(0, 1e-6)

    max_missing = int((1 - cfg.min_data_values) * df.shape[1])
    keep = df.isna().sum(axis=1) <= max_missing
    df = df.loc[keep].copy()

    if cfg.reverse_log2:
        df = np.power(2.0, df)
    elif cfg.reverse_log:
        df = np.exp(df)

    if (df < 0).any().any():
        raise ValueError("Negative values found after preprocessing.")

    df_corr = df.iloc[:: cfg.reduce_correlation_by, :].copy()
    return df, df_corr


def compute_normics_longlist(
    df_corr: pd.DataFrame,
    protein_ids: pd.Index,
    tmt_ratio: bool = False,
) -> pd.DataFrame:
    log2_df = np.log2(df_corr)

    corr = log2_df.T.corr(method="spearman", min_periods=2)
    mean_corr = corr.mean(axis=1)
    var_corr = corr.var(axis=1)

    std = df_corr.std(axis=1, ddof=1)
    mean = df_corr.mean(axis=1)
    cv = std if tmt_ratio else std / mean.replace(0, np.nan)

    longlist = pd.DataFrame(
        {
            "protein_id": protein_ids.astype(str),
            "mean_correlation": mean_corr.values,
            "cv": cv.values,
            "var_correlation": var_corr.values,
        }
    )
    longlist = longlist.replace([np.inf, -np.inf], np.nan)
    longlist = longlist.dropna(subset=["mean_correlation", "cv"]).reset_index(drop=True)

    longlist = longlist.sort_values("cv", ascending=True).reset_index(drop=True)
    longlist["rank_score"] = longlist.index

    longlist = longlist.sort_values("mean_correlation", ascending=False).reset_index(drop=True)
    longlist["rank_score"] = longlist["rank_score"] + longlist.index

    longlist = longlist.sort_values("rank_score", ascending=True).reset_index(drop=True)
    return longlist


def write_normics_inputs(
    matrix: pd.DataFrame, cfg: NormicsConfig, outdir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_clean, df_corr = preprocess_matrix(matrix, cfg)
    longlist = compute_normics_longlist(
        df_corr=df_corr,
        protein_ids=df_corr.index,
        tmt_ratio=cfg.tmt_ratio,
    )
    top_n = int(round(len(longlist) * cfg.subset_fraction))
    top_n = max(cfg.subset_min_n, top_n, 1)
    top_n = min(top_n, len(longlist))
    subset = longlist.iloc[:top_n].copy()
    subset_ids = subset["protein_id"].astype(str).tolist()
    df_subset = df_clean.loc[df_clean.index.astype(str).isin(subset_ids)].copy()

    df_clean.to_csv(outdir / "data_all.csv")
    df_subset.to_csv(outdir / "data_subset.csv")
    longlist.to_csv(outdir / "normics_longlist.csv", index=False)
    subset.to_csv(outdir / "normics_subset.csv", index=False)
    return df_clean, df_subset, longlist, subset


def run_rscript(script: Path, *args: Path) -> None:
    cmd = [str(RSCRIPT_NORMICS_R_BIN), str(script), *(str(arg) for arg in args)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_normalized_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def write_raw_long_reshape(report_path: Path, outdir: Path) -> None:
    """Write the report verbatim as runs x proteins, as read by run_raw_qc.

    Four annotation rows (genes, names, descriptions, q value) precede the run
    rows; cells keep the report's text. Proteins stay in report order; runs are
    BID tumours by number, then BatchQC, then QC-pool, then anything else.
    """
    df = pd.read_csv(report_path, sep="\t", dtype=str, keep_default_na=False)
    proteins = pd.unique(df["pg_protein_accessions"])
    duplicates = int(df.duplicated(["r_file_name", "pg_protein_accessions"]).sum())
    qvalue_counts = df.groupby("pg_protein_accessions")["pg_qvalue"].nunique()
    qvalue_conflicts = qvalue_counts.index[qvalue_counts > 1].tolist()

    quantity = df.pivot(
        index="r_file_name", columns="pg_protein_accessions", values="pg_quantity"
    )
    info = parse_sample_metadata_long(quantity.index.tolist()).set_index("raw_sample")
    group_rank = {"BID": 0, "BatchQC": 1, "QC-pool": 2}

    def sort_key(name: str) -> tuple:
        group = info.at[name, "sample_group"]
        bid = info.at[name, "bid_numeric"] if group == "BID" else 0
        return (group_rank.get(group, 3), bid, name)

    runs = sorted(quantity.index, key=sort_key)
    annotation_cols = ["pg_genes", "pg_protein_names", "pg_protein_descriptions", "pg_qvalue"]
    annotations = (
        df.drop_duplicates("pg_protein_accessions")
        .set_index("pg_protein_accessions")
        .loc[proteins, annotation_cols]
        .T
    )
    reshaped = pd.concat([annotations, quantity.loc[runs, proteins]])
    reshaped.index.name = "r_file_name"
    reshaped.columns.name = None
    out_csv = outdir / f"{RESHAPED_NAME}.csv"
    # CRLF, as the csv.writer that first produced these QC inputs wrote them.
    reshaped.to_csv(out_csv, lineterminator="\r\n")

    bids = set(info.loc[info["sample_group"].eq("BID"), "bid_numeric"].astype(int))
    manifest = {
        "source_tsv": str(report_path),
        "output_csv": str(out_csv),
        "format": (
            "First five rows are header/metadata rows: r_file_name/protein ids, pg_genes, "
            "pg_protein_names, pg_protein_descriptions, pg_qvalue. Remaining rows are "
            "samples; cells are raw pg_quantity."
        ),
        "n_long_rows_read": int(len(df)),
        "n_samples": int(len(runs)),
        "n_bid_samples": int(info["sample_group"].eq("BID").sum()),
        "missing_bid_numbers_1_to_1000": sorted(set(range(1, 1001)) - bids),
        "n_proteins": int(len(proteins)),
        "duplicate_sample_protein_cells": duplicates,
        "qvalue_conflict_proteins": len(qvalue_conflicts),
        "qvalue_conflict_examples": qvalue_conflicts[:10],
        "sample_sort": (
            "BID samples sorted by BID number, then BatchQC, then QC-pool, then other "
            "samples lexicographically."
        ),
    }
    (outdir / f"{RESHAPED_NAME}_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    cfg = NormicsConfig(
        min_data_values=args.min_data_values,
        subset_fraction=args.subset_fraction,
        subset_min_n=args.subset_min_n,
        reduce_correlation_by=args.reduce_correlation_by,
    )
    paths = ensure_dirs(args.outdir)

    (paths["root"] / "pipeline_config.json").write_text(
        json.dumps(
            {
                "report": str(args.report),
                "meta": str(args.meta),
                "msi": str(args.msi),
                "r_script": str(args.r_script),
                "plate_r_script": str(args.plate_r_script),
                "rscript": str(RSCRIPT_NORMICS_R_BIN),
                "normics_config": asdict(cfg),
            },
            indent=2,
        )
    )

    matrix_raw, protein_annotations = load_long_report(args.report)

    all_sample_names = matrix_raw.columns.tolist()
    sample_info_raw = parse_sample_metadata_long(all_sample_names)
    sample_info_raw.to_csv(paths["intermediate"] / "sample_columns_raw.csv", index=False)
    protein_annotations.to_csv(paths["intermediate"] / "protein_annotations.csv")
    write_raw_long_reshape(args.report, paths["intermediate"])

    keep_info = sample_info_raw.loc[
        sample_info_raw["sample_group"].isin(["BID", "BatchQC", "QC-pool"])
    ].copy()
    matrix_keep = matrix_raw.loc[:, keep_info["raw_sample"]].copy()

    matrix_selected, sample_meta, rerun_audit = select_best_reruns(matrix_keep, keep_info)
    sample_meta = merge_external_metadata(sample_meta, args.meta, args.msi)

    rerun_audit.to_csv(paths["intermediate"] / "best_rerun_audit.csv", index=False)
    sample_meta.to_csv(paths["intermediate"] / "sample_metadata_selected.csv", index=False)
    matrix_selected.to_csv(paths["intermediate"] / "quantity_matrix_nonblank_best_rerun.csv")

    bid_samples = sample_meta.loc[sample_meta["sample_group"] == "BID", "sample"].tolist()
    matrix_bid = matrix_selected.loc[:, bid_samples]
    matrix_bid.to_csv(paths["intermediate"] / "quantity_matrix_bid_only_best_rerun_raw.csv")
    matrix_bid.T.to_csv(
        paths["intermediate"] / "quantity_matrix_bid_only_best_rerun_raw_samples_by_protein.csv",
        index_label="sample",
        lineterminator="\r\n",
    )

    matrix_for_normics, _, longlist, subset = write_normics_inputs(
        matrix_selected, cfg, paths["intermediate"]
    )

    normalized_dir = paths["root"] / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    run_rscript(
        args.r_script,
        paths["intermediate"] / "data_all.csv",
        paths["intermediate"] / "data_subset.csv",
        normalized_dir,
    )

    norm_vsn = load_normalized_matrix(normalized_dir / "normalized_normics_vsn_log2.csv")
    norm_vsn.loc[:, bid_samples].to_csv(
        paths["intermediate"] / "normalized_normics_vsn_log2_bid_only_best_rerun.csv"
    )

    final_dir = paths["root"] / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    run_rscript(
        args.plate_r_script,
        paths["intermediate"] / "normalized_normics_vsn_log2_bid_only_best_rerun.csv",
        paths["intermediate"] / "sample_metadata_selected.csv",
        final_dir,
    )

    summary = {
        "report": str(args.report),
        "n_raw_samples": int(len(all_sample_names)),
        "n_selected_samples": int(matrix_selected.shape[1]),
        "n_bid_samples": int(len(bid_samples)),
        "proteins_before_preprocess": int(matrix_selected.shape[0]),
        "proteins_after_normics_preprocess": int(matrix_for_normics.shape[0]),
        "normics_subset_size": int(subset.shape[0]),
        "selected_rerun_fraction": float(sample_meta["selected_is_rerun"].mean()),
        "samples_with_any_rerun_option": int((sample_meta["n_rerun_columns"] > 0).sum()),
        "final_matrix": str(
            final_dir / "normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"
        ),
    }
    (paths["root"] / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("Done.")
    print(f"Outputs written to: {paths['root']}")
    print(summary["final_matrix"])


if __name__ == "__main__":
    main()
