#!/usr/bin/env bash
# Reproduce the CancerScout CRC proteomics figures and supplementary tables.
#
# Usage (inside the activated proj environment, from any directory):
#   ./reproduce.sh              # = downstream
#   ./reproduce.sh downstream   # start from the processed proteome and the locked
#                               # discovery model (models/production_997_seed_2605161932)
#   ./reproduce.sh all          # start from the Biognosys protein report: Normics/VSN,
#                               # limma, MuVI, consensus clustering, then downstream
#
# Inputs and environments are described in README.md. Every flag that differs
# from a script's default is the value the paper's results were produced with.
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:-downstream}"
case "$mode" in
  downstream|all) ;;
  *) echo "usage: $0 [downstream|all]" >&2; exit 2 ;;
esac

# Paths and the pinned consensus-r Rscript come from cancerscout/paths.py.
eval "$(python - <<'EOF'
from cancerscout import paths
for name in ["ACTIVE_RAW_NORMICS_DIR", "ACTIVE_PROCESSED_NORMICS_DIR", "ACTIVE_MODEL_DIR",
             "MODELS_DIR", "RSCRIPT_CONSENSUS_R_BIN"]:
    print(f"{name}='{getattr(paths, name)}'")
EOF
)"
seed=2605161932
seed_dir="$MODELS_DIR/muvi_997_seed_$seed"

step() {
  echo
  echo "=== $1"
  shift
  "$@"
}

if [ "$mode" = all ]; then
  for out in "$ACTIVE_RAW_NORMICS_DIR" "$ACTIVE_PROCESSED_NORMICS_DIR" "$seed_dir" "$ACTIVE_MODEL_DIR"; do
    if [ -e "$out" ]; then
      echo "$out already exists; move it away to rerun from the raw report." >&2
      exit 1
    fi
  done

  step "Stage 1: raw report -> Normics VSN -> limma (all 999 tumours)" \
    python preprocessing/normics_long_format_pipeline.py
  step "Stage 2: drop BID0751/BID0762, refit VSN and limma, build the MuData" \
    python -m cancerscout.run_normics_no_outliers
  # As in the original run, a failure after the repeated Leiden runs is tolerated
  # here and consensus clustering is completed below.
  step "MuVI and 100 Leiden runs (seed $seed)" \
    python -m cancerscout.run_muvi_normics --out-dir "$seed_dir" --seed "$seed" \
      --skip-old-new-check --skip-survival --n-epochs 10000 --learning-rate 0.003 \
      --repeated-leiden-resolution 0.75 --n-leiden-runs 100 || true
  if [ ! -f "$seed_dir/leiden.csv" ]; then
    echo "MuVI training did not finish: $seed_dir/leiden.csv is missing." >&2
    exit 1
  fi
  if [ ! -f "$seed_dir/leiden_cons.csv" ]; then
    step "Consensus clustering" "$RSCRIPT_CONSENSUS_R_BIN" r/consensus_clustering.R "$seed_dir"
  fi
  step "Locked handoff (k = 6)" python -m cancerscout.promote_to_production
fi

step "Plot style spec" python -m cancerscout.export_plot_style
step "CPTAC UMS labels for the discovery cohort" \
  python -m cancerscout.run_cptac_classification --old-label-col cptac_prediction --promote
step "Train/validation/test splits" python -m cancerscout.run_make_downstream_splits

step "Cluster interpretation tables (Fig 2B-D)" python -m cancerscout.run_cluster_interpretation
step "UMAPs (Fig 2B-D)" python -m cancerscout.plot_cluster_interpretation
step "Differential abundance and volcanoes (Supp Fig 7, Supp Table 3)" \
  python -m cancerscout.run_cluster_volcano_plots
step "Top marker heatmap (Fig 4A)" \
  python -m cancerscout.plot_cluster_top_markers --top-n 2 --skip-rfecv --up-down-only

step "Reactome GSEA (Supp Table 4)" \
  python -m cancerscout.run_gsea --gene-set reactome --out-dir results/gsea/reactome
step "Reactome GSEA heatmap (Fig 4B)" \
  python -m cancerscout.plot_gsea --out-dir results/gsea/reactome \
    --figures-dir figures/gsea/reactome --top-n 2 --skip-combined

# Fig 5A depends on this order: RFECV, then the panel refit, then post hoc.
step "Marker RFECV" \
  python -m cancerscout.run_cluster_marker_rfecv --selection-frequency-threshold 1.0 \
    --max-selected-features 0 --reduced-panel-sizes 5,10,20,50 --threads 8
step "Marker panel AUC by size (Fig 5A, Supp Table 5b)" \
  python -m cancerscout.run_biomarker_panel_figure
step "Marker signatures (Supp Table 5a)" \
  python -m cancerscout.run_cluster_marker_posthoc --top-n 10

step "Six-way classifier, 5-fold CV" \
  python -m cancerscout.run_multiclass_consensus_classifier_cv --use-ray
step "Classifier CV figures (Fig 5B, ED Fig 8, Supp Table 5c-e)" \
  python -m cancerscout.plot_multiclass_consensus_classifier \
    --results-dir results/multiclass_consensus_classifier_CV \
    --figures-dir figures/multiclass_consensus_classifier_CV --output-suffix cv \
    --evaluation-label "5-fold cross-validation (out-of-fold)" --top-n 3

step "Transfer to CPTAC and CRC65 (Supp Table 2c)" \
  python -m cancerscout.run_external_cluster_transfer
step "CRC65 predicted drug sensitivity (Fig 3D, Supp Fig 4, Supp Table 2a-b)" \
  python -m cancerscout.run_cellline_moa_sensitivity
step "Subtype concordance (Fig 2E, Fig 5C)" python -m cancerscout.run_subtype_concordance

step "Oncoprint (Supp Fig 5)" python -m cancerscout.run_mutation_complex_oncoprint
step "Mutation enrichment with tuples (ED Fig 6B)" python -m cancerscout.run_mutation_tuples
step "Survival (Fig 3A-B)" python -m cancerscout.run_cluster_survival_analysis

step "Raw data QC" python -m cancerscout.run_raw_qc
step "Preprocessing QC panel (Supp Fig 3)" python -m cancerscout.run_quality_control_figures
step "Consensus k selection (ED Fig 3)" \
  python -m cancerscout.plot_consensus_quality --results-dir "$ACTIVE_MODEL_DIR" --max-k 8 \
    --out figures/muvi/consensus_quality_k_selection.png \
    --out-median figures/muvi/consensus_median_silhouette_k_selection.png
step "Consensus heatmap (Fig 2A)" \
  "$RSCRIPT_CONSENSUS_R_BIN" r/consensus_leiden_cola_heatmap.R "$ACTIVE_MODEL_DIR" \
    figures/muvi/consensus_leiden_cola_heatmap.png

step "Supplementary Tables 2-5 and Supplementary Data 1-2" \
  python -m cancerscout.run_export_supplementary
step "Figure style audit" python -m cancerscout.check_plot_style

echo
echo "Done. Figures are in figures/, tables in results/supplementary/."
