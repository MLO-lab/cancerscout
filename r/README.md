# R scripts

Two scripts run in the `normics-r` environment; every other script runs in `consensus-r`
and is called from the Python pipeline through `cancerscout.plotting.run_r_figure`, which
resolves the pinned `Rscript` from `cancerscout/paths.py`.

**Preprocessing (`normics-r`)**

- `normics_normalize.R`: Normics/VSN normalization. VSN is fitted on the Normics protein
  subset and applied to the full filtered matrix.
- `plate_batch_correct.R`: limma `removeBatchEffect` on plate, protecting MSI status.

**Clustering and figures (`consensus-r`)**

- `consensus_clustering.R`: Hamming-distance consensus over the 100 Leiden runs, and the
  per-k scores.
- `consensus_leiden_cola_heatmap.R`: consensus heatmap (Fig 2A).
- `cluster_cptac_alluvial.R`: CancerScout-to-CPTAC alluvial plot (Fig 2E).
- `km_survival_curves.R`: Kaplan–Meier curves (Fig 3A).
- `cox_forest_plot.R`: adjusted Cox forest plot (Fig 3B).
- `generic_cluster_heatmap.R`: the shared cluster-by-feature heatmap behind
  `cancerscout.plotting.complex_heatmap` (Fig 3D, ED Fig 6B, Supp Fig 4).
- `cluster_top_markers_heatmap.R`: top differentially abundant proteins (Fig 4A).
- `gsea_top_pathways_heatmap.R`: top Reactome pathways (Fig 4B).
- `gsea_combined_heatmap.R`: the combined GSEA heatmap that `plot_gsea` draws unless
  `--skip-combined` is given. It isn't a paper panel.
- `mutation_oncoprint.R`: oncoprint (Supp Fig 5).
- `plot_style.R`: shared type sizes, geometry, colours and device. Every figure script
  sources it. It reads `results/plot_style/style_spec.json`, which
  `python -m cancerscout.export_plot_style` writes, and stops if that file is missing.
