"""Canonical repository paths for the active CancerScout analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ACTIVE_RAW_NORMICS_DIR = RAW_DATA_DIR / "normics_r14_goettingen_output_20260428"
ACTIVE_PROCESSED_NORMICS_DIR = (
    PROCESSED_DATA_DIR / "normics_r14_goettingen_997_no_outliers_20260516"
)

MODELS_DIR = REPO_ROOT / "models"
ACTIVE_MODEL_DIR = MODELS_DIR / "production_997_seed_2605161932"

REFERENCES_DIR = REPO_ROOT / "references"
GENE_SET_DIR = REFERENCES_DIR / "gene_sets" / "msigdb"
SIGNATURE_DIR = REFERENCES_DIR / "signatures"

RESULTS_DIR = REPO_ROOT / "results"
QUALITY_CONTROL_RESULTS_DIR = RESULTS_DIR / "quality_control"
CLUSTER_INTERPRETATION_RESULTS_DIR = RESULTS_DIR / "cluster_interpretation"
CPTAC_RESULTS_DIR = RESULTS_DIR / "cptac_classification"
GSEA_RESULTS_DIR = RESULTS_DIR / "gsea"
SURVIVAL_RESULTS_DIR = RESULTS_DIR / "survival"
PHARMACOPROTEOMICS_RESULTS_DIR = RESULTS_DIR / "pharmacoproteomics"
EXTERNAL_TRANSFER_RESULTS_DIR = RESULTS_DIR / "external_cluster_transfer"
CELLLINE_MOA_RESULTS_DIR = RESULTS_DIR / "cellline_moa_sensitivity"
MUTATION_ENRICHMENT_RESULTS_DIR = RESULTS_DIR / "mutation_enrichment"
SUBTYPE_CONCORDANCE_RESULTS_DIR = RESULTS_DIR / "subtype_concordance"
CLUSTER_MARKER_RESULTS_DIR = (
    RESULTS_DIR / "cluster_marker_rfecv_min50_strict10_uncapped_997"
)
MULTICLASS_CLASSIFIER_RESULTS_DIR = RESULTS_DIR / "multiclass_consensus_classifier"
MULTICLASS_CLASSIFIER_CV_RESULTS_DIR = (
    RESULTS_DIR / "multiclass_consensus_classifier_CV"
)
HIERARCHICAL_CLASSIFIER_RESULTS_DIR = RESULTS_DIR / "hierarchical_consensus_classifier"
MODEL_SPLITS_DIR = RESULTS_DIR / "model_splits_997"

FIGURES_DIR = REPO_ROOT / "figures"
REPORTS_DIR = REPO_ROOT / "reports"
R_DIR = REPO_ROOT / "r"
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"

# External data sources (user-supplied, outside the repo tree).
# Override with CANCERSCOUT_DATA_DIR; defaults to a gitignored data/external/ dir.
EXTERNAL_DATA_DIR = Path(os.environ.get("CANCERSCOUT_DATA_DIR", DATA_DIR / "external"))
MUTATION_H5AD = EXTERNAL_DATA_DIR / "mutation.h5ad"
MDATA_TEMPLATE_H5MU = EXTERNAL_DATA_DIR / "mdata.h5mu"
FREJNO_DRUG_ZIP = EXTERNAL_DATA_DIR / "new_samples" / "crc_drug.zip"
LEGACY_CPTAC_DIR = Path(
    os.environ.get("CANCERSCOUT_CPTAC_DIR", EXTERNAL_DATA_DIR / "cptac")
)
CPTAC_PROTEOME_CSV = LEGACY_CPTAC_DIR / "prot.csv"
CPTAC_METADATA_CSV = LEGACY_CPTAC_DIR / "meta.csv"
RAW_TSV_GOETTINGEN_R14 = (
    EXTERNAL_DATA_DIR / "new_samples" / "2026-04-23_Goettingen_R14_Protein_Level.tsv"
)

# Conda env binaries (used by scripts that subprocess R/Python).
# Override with CANCERSCOUT_CONDA_ENVS_DIR; defaults to the envs dir of the
# active interpreter (inside `conda run -n <env>`, sys.prefix is .../envs/<env>).
CONDA_ENVS_DIR = Path(
    os.environ.get("CANCERSCOUT_CONDA_ENVS_DIR", Path(sys.prefix).resolve().parent)
)
PYTHON_PROJ_BIN = CONDA_ENVS_DIR / "proj" / "bin" / "python"
RSCRIPT_CONSENSUS_R_BIN = CONDA_ENVS_DIR / "consensus-r" / "bin" / "Rscript"
RSCRIPT_NORMICS_R_BIN = CONDA_ENVS_DIR / "normics-r" / "bin" / "Rscript"

# Derived data paths referenced by multiple scripts
NORMALIZED_LIMMA_CSV = (
    ACTIVE_PROCESSED_NORMICS_DIR
    / "final"
    / "normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"
)
ORDERED_MDATA_H5MU = (
    ACTIVE_PROCESSED_NORMICS_DIR / "mdata_r14_goettingen_normics_997_ordered.h5mu"
)
RAW_RESHAPED_CSV = (
    ACTIVE_RAW_NORMICS_DIR
    / "intermediate"
    / "r14_goettingen_raw_long_reshaped_samples_by_protein.csv"
)
NORMALIZED_VSN_LOG2_CSV = (
    ACTIVE_RAW_NORMICS_DIR
    / "intermediate"
    / "normalized_normics_vsn_log2_bid_only_best_rerun.csv"
)
