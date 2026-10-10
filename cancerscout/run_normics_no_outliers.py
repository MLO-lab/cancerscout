#!/usr/bin/env python
"""Re-run Normics VSN + limma plate correction with BID0751 and BID0762 excluded.

Reads existing intermediate files from the current Normics pipeline output,
drops the two outlier samples, re-fits VSN and limma on the clean 997-sample
matrix, then builds a new h5mu.  Nothing in the original output directory is
modified.

Output: ACTIVE_PROCESSED_NORMICS_DIR (data/processed/normics_r14_goettingen_997_no_outliers_20260516/)
unless --out-dir is given.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
from cancerscout.paths import (
    ACTIVE_PROCESSED_NORMICS_DIR,
    ACTIVE_RAW_NORMICS_DIR,
    MDATA_TEMPLATE_H5MU,
    PREPROCESSING_DIR,
    PYTHON_PROJ_BIN,
    R_DIR,
    RSCRIPT_NORMICS_R_BIN,
)

ORIG_DIR   = ACTIVE_RAW_NORMICS_DIR
PYTHON     = PYTHON_PROJ_BIN
RSCRIPT    = RSCRIPT_NORMICS_R_BIN
NORMICS_R  = R_DIR / "normics_normalize.R"
LIMMA_R    = R_DIR / "plate_batch_correct.R"

OUTLIER_BIDS = {751, 762}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ACTIVE_PROCESSED_NORMICS_DIR)
    return parser.parse_args()


def drop_outlier_columns(df: pd.DataFrame, outlier_cols: set[str]) -> pd.DataFrame:
    drop = [c for c in df.columns if c in outlier_cols]
    if drop:
        print(f"  Dropping columns: {drop}")
    return df.drop(columns=drop)


def run(cmd: list[str], log: Path) -> None:
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    with open(log, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}). See {log}")


def main() -> None:
    dst = parse_args().out_dir
    if dst.exists():
        print(f"Output dir already exists: {dst}")
        print("Delete it first if you want to re-run.")
        sys.exit(1)

    inter  = dst / "intermediate"
    normd  = dst / "normalized"
    final  = dst / "final"
    for d in [inter, normd, final]:
        d.mkdir(parents=True)

    # ── 1. Load existing intermediates ────────────────────────────────────────
    print("Loading existing intermediate files ...")
    meta = pd.read_csv(ORIG_DIR / "intermediate/sample_metadata_selected.csv")

    # Identify outlier sample column names from bid_numeric
    outlier_sample_cols = set(
        meta.loc[meta["bid_numeric"].isin(OUTLIER_BIDS), "sample"]
    )
    print(f"Outlier sample columns to drop: {outlier_sample_cols}")

    data_all = pd.read_csv(ORIG_DIR / "intermediate/data_all.csv", index_col=0)
    data_sub = pd.read_csv(ORIG_DIR / "intermediate/data_subset.csv", index_col=0)

    # ── 2. Drop outliers ──────────────────────────────────────────────────────
    print("Dropping outlier samples ...")
    data_all_filt = drop_outlier_columns(data_all, outlier_sample_cols)
    data_sub_filt = drop_outlier_columns(data_sub, outlier_sample_cols)
    meta_filt = meta[~meta["bid_numeric"].isin(OUTLIER_BIDS)].copy()

    print(f"  data_all: {data_all.shape[1]} → {data_all_filt.shape[1]} samples")
    print(f"  data_subset: {data_sub.shape[1]} → {data_sub_filt.shape[1]} samples")
    print(f"  metadata rows: {len(meta)} → {len(meta_filt)}")

    data_all_filt.to_csv(inter / "data_all.csv")
    data_sub_filt.to_csv(inter / "data_subset.csv")
    meta_filt.to_csv(inter / "sample_metadata_selected.csv", index=False)

    # Copy protein annotations unchanged
    shutil.copy2(
        ORIG_DIR / "intermediate/protein_annotations.csv",
        inter / "protein_annotations.csv",
    )

    # ── 3. Normics VSN ────────────────────────────────────────────────────────
    print("\nRunning Normics VSN (normics_normalize.R) ...")
    run(
        [str(RSCRIPT), str(NORMICS_R),
         str(inter / "data_all.csv"),
         str(inter / "data_subset.csv"),
         str(normd)],
        log=dst / "normics_vsn.log",
    )

    # ── 4. Extract BID-only matrix ────────────────────────────────────────────
    print("Extracting BID-only normalized matrix ...")
    norm_vsn = pd.read_csv(normd / "normalized_normics_vsn_log2.csv", index_col=0)
    bid_samples = meta_filt.loc[meta_filt["sample_group"] == "BID", "sample"].tolist()
    bid_norm = norm_vsn.loc[:, bid_samples]
    bid_norm_path = inter / "normalized_normics_vsn_log2_bid_only_best_rerun.csv"
    bid_norm.to_csv(bid_norm_path)
    print(f"  BID-only matrix: {bid_norm.shape}")

    # ── 5. Limma plate correction ─────────────────────────────────────────────
    print("\nRunning limma plate correction (plate_batch_correct.R) ...")
    run(
        [str(RSCRIPT), str(LIMMA_R),
         str(bid_norm_path),
         str(inter / "sample_metadata_selected.csv"),
         str(final)],
        log=dst / "limma.log",
    )

    # ── 6. Build MuData ───────────────────────────────────────────────────────
    print("\nBuilding MuData (build_final_mudata.py) ...")
    h5mu_path = dst / "mdata_r14_goettingen_normics_997.h5mu"
    run(
        [str(PYTHON), str(PREPROCESSING_DIR / "build_final_mudata.py"),
         "--matrix",       str(final / "normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"),
         "--sample-meta",  str(inter / "sample_metadata_selected.csv"),
         "--protein-ann",  str(inter / "protein_annotations.csv"),
         "--template-h5mu", str(MDATA_TEMPLATE_H5MU),
         "--out",           str(h5mu_path)],
        log=dst / "build_mudata.log",
    )

    print("\nOrdering MuData by numeric obs id (order_mudata_by_numeric_obs.py) ...")
    h5mu_ordered = dst / "mdata_r14_goettingen_normics_997_ordered.h5mu"
    run(
        [str(PYTHON), str(PREPROCESSING_DIR / "order_mudata_by_numeric_obs.py"),
         "--input", str(h5mu_path),
         "--out",   str(h5mu_ordered)],
        log=dst / "order_mudata.log",
    )

    # ── 7. Provenance ─────────────────────────────────────────────────────────
    provenance = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": str(ORIG_DIR),
        "outlier_bids_removed": sorted(OUTLIER_BIDS),
        "n_samples": len(bid_samples),
        "h5mu_ordered": str(h5mu_ordered),
        "note": (
            "Normics VSN and limma plate correction re-run after removing BID0751/BID0762. "
            "Protein set and all other settings unchanged from source_dir."
        ),
    }
    (dst / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print(f"\nDone. Output: {dst}")
    print(f"h5mu (ordered): {h5mu_ordered}")


if __name__ == "__main__":
    main()
