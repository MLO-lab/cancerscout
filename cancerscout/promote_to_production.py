#!/usr/bin/env python
"""Promote MuVI seed 2605161932 (997 samples, outliers removed) to production.

Reconstructs ``factor_adata.h5ad`` from the MuVI run's CSVs, recomputes UMAP, sets
``leiden_cons = k=6``, and writes a complete handoff under ``models/``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from cancerscout.paths import ACTIVE_PROCESSED_NORMICS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR

ROOT     = Path(__file__).resolve().parent.parent
SEED_DIR = MODELS_DIR / "muvi_997_seed_2605161932"
PROD_DIR = MODELS_DIR / "production_997_seed_2605161932"
H5MU_997 = ACTIVE_PROCESSED_NORMICS_DIR / "mdata_r14_goettingen_normics_997_ordered.h5mu"

FIGURES_ROOT = FIGURES_DIR
REPORTS_ROOT = REPORTS_DIR

FIGURE_SUBDIRS = [
    "quality_control",
    "muvi",
    "clustering",
    "msi_comparison",
    "cluster_interpretation",
    "gsea",
    "cell_types",
    "pharmacoproteomics",
    "survival",
]

# Files to copy verbatim from seed dir into production dir
COPY_FILES = [
    "config.json",
    "latent.csv",
    "leiden.csv",
    "leiden_single_run.csv",
    "consensus_summary.csv",
    "consensus_scores_by_k.csv",
    "adata_latent_obs.csv",
    "muvi_training_objective.png",
    "Consensus_Heatmap.pdf",
    "SelectPartition.pdf",
    "Consensus_Scores.pdf",
    "Membership_Heatmap.pdf",
    "All_Collect_Plots.pdf",
]
COPY_GLOBS = ["leiden_cons_k_*.csv"]


def build_figure_tree() -> None:
    for sub in FIGURE_SUBDIRS:
        (FIGURES_ROOT / sub).mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Created figures/ tree at {FIGURES_ROOT}")
    print(f"Created reports/ at {REPORTS_ROOT}")


def build_factor_adata(prod_dir: Path) -> sc.AnnData:
    """Reconstruct factor_adata.h5ad from latent.csv + adata_latent_obs.csv + UMAP."""
    print("Loading latent factors and obs metadata ...")
    latent = pd.read_csv(SEED_DIR / "latent.csv", index_col=0)
    latent.index = latent.index.astype(str)

    obs = pd.read_csv(SEED_DIR / "adata_latent_obs.csv", index_col=0)
    obs.index = obs.index.astype(str)
    obs = obs.reindex(latent.index)

    # Override leiden_cons with k=6
    k6 = pd.read_csv(SEED_DIR / "leiden_cons_k_6.csv").set_index("sample_id")
    k6.index = k6.index.astype(str)
    obs["leiden_cons"] = k6.reindex(latent.index)["leiden_cons"].astype("category")

    # Attach all k solutions
    for kpath in sorted(SEED_DIR.glob("leiden_cons_k_*.csv")):
        knum = int(kpath.stem.split("_")[-1])
        kdf = pd.read_csv(kpath).set_index("sample_id")
        kdf.index = kdf.index.astype(str)
        obs[f"leiden_cons_{knum}"] = kdf.reindex(latent.index)["leiden_cons"].astype("category")

    X = latent.values.astype(np.float32)

    print("Building AnnData and recomputing UMAP ...")
    adata = sc.AnnData(X=X)
    adata.obs = obs.copy()
    adata.obs_names = latent.index.tolist()
    adata.var_names = latent.columns.tolist()
    adata.obsm["filtered_scores"] = X.copy()

    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    print(f"  UMAP done. Shape: {adata.shape}")

    out = prod_dir / "factor_adata.h5ad"
    adata.write_h5ad(out)
    print(f"  Saved → {out}")
    return adata


def write_leiden_cons_k6(prod_dir: Path) -> None:
    """Write leiden_cons.csv as the k=6 solution."""
    k6 = pd.read_csv(SEED_DIR / "leiden_cons_k_6.csv")
    k6.to_csv(prod_dir / "leiden_cons.csv", index=False)


def write_handoff_manifest(prod_dir: Path) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": 2605161932,
        "n_samples": 997,
        "outliers_removed": ["BID0751", "BID0762"],
        "cluster_col": "leiden_cons",
        "cluster_source_col": "leiden_cons_6",
        "default_handoff_k": 6,
        "selected_consensus_k": 6,
        "factor_adata": "factor_adata.h5ad",
        "metadata": "metadata.csv",
        "factors": "Z.csv",
        "mdata_path": str(H5MU_997),
        "source_run_dir": str(SEED_DIR),
        "backup_note": "Superseded material is preserved in archive/20260516_pre_layout_refactor/.",
    }
    (prod_dir / "handoff_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_notes(prod_dir: Path) -> None:
    notes = f"""# Production Run Notes — 997 samples, seed 2605161932

## What changed
- BID0751 and BID0762 removed before Normics VSN and limma plate correction.
- MuVI re-trained on the clean 997-sample matrix with seed 2605161932.
- Consensus Leiden clustering: default k=6 (log-rank p=0.0056 at 60 months).

## Backup of old results
| Artefact | Location |
|---|---|
| Superseded material | `archive/20260516_pre_layout_refactor/` |
| Raw MuVI run | `models/muvi_997_seed_2605161932/` |

Created: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    (prod_dir / "NOTES.md").write_text(notes)
    print(f"  Written NOTES.md")


def main() -> None:
    if PROD_DIR.exists():
        print(f"Production dir already exists: {PROD_DIR}")
        print("Delete it first to re-promote.")
        return

    PROD_DIR.mkdir(parents=True)
    print(f"Production dir: {PROD_DIR}\n")

    # 1. Copy verbatim files
    print("Copying seed files ...")
    for name in COPY_FILES:
        src = SEED_DIR / name
        if src.exists():
            shutil.copy2(src, PROD_DIR / name)
    for pattern in COPY_GLOBS:
        for src in SEED_DIR.glob(pattern):
            shutil.copy2(src, PROD_DIR / src.name)

    # Copy muvi model directory
    shutil.copytree(SEED_DIR / "muvi_model", PROD_DIR / "muvi_model")
    print("  Done.")

    # Z.csv = latent.csv (same content, different name expected by downstream)
    shutil.copy2(SEED_DIR / "latent.csv", PROD_DIR / "Z.csv")

    # 2. Override leiden_cons.csv with k=6
    write_leiden_cons_k6(PROD_DIR)

    # 3. Build factor_adata.h5ad
    print("\nBuilding factor_adata.h5ad ...")
    adata = build_factor_adata(PROD_DIR)

    # 4. Write metadata.csv (= factor_adata.obs)
    adata.obs.to_csv(PROD_DIR / "metadata.csv")
    print(f"  Written metadata.csv ({len(adata.obs)} rows)")

    # 5. Handoff manifest + notes
    write_handoff_manifest(PROD_DIR)
    write_notes(PROD_DIR)

    # 6. BID-indexed exports
    print("\nWriting BID-indexed exports ...")
    from cancerscout.io import write_bid_indexed_exports
    write_bid_indexed_exports(PROD_DIR)

    # 7. Create clean figures/ and reports/ tree
    print("\nSetting up figures/ and reports/ directories ...")
    build_figure_tree()

    print(f"\nDone.")
    print(f"  Production handoff: {PROD_DIR}")
    print(f"  Figures root:       {FIGURES_ROOT}")
    print(f"  Reports root:       {REPORTS_ROOT}")


if __name__ == "__main__":
    main()
