# CancerScout colorectal cancer proteomics

Analysis code for *A clinically annotated proteomic and histological resource of 1,000
colorectal cancers*. It reproduces the proteomic figure panels and Supplementary Tables 2–5
of the paper. The pipeline runs from the Biognosys protein report through Normics/VSN
normalization, limma plate correction, MuVI latent factors and consensus Leiden clustering,
then to the downstream analyses: CPTAC and CRC65 transfer, pathways, markers and classifiers,
mutations, survival, and predicted drug sensitivity.

No data are included. The patient-level inputs are released with the paper (see
[Inputs](#inputs)); third-party references have to be downloaded from their providers.

## What is reproduced

| Panel | Produced by (`cancerscout.` modules unless a path is given) |
|---|---|
| Fig 2A | `r/consensus_leiden_cola_heatmap.R` |
| Fig 2B–D | `run_cluster_interpretation`, `plot_cluster_interpretation` (CPTAC labels from `run_cptac_classification`) |
| Fig 2E, Fig 5C | `run_external_cluster_transfer`, `run_subtype_concordance` |
| Fig 3A–B | `run_cluster_survival_analysis` |
| Fig 3D, Supp Fig 4 | `run_external_cluster_transfer`, `run_cellline_moa_sensitivity` |
| Fig 4A, Supp Fig 7 | `run_cluster_volcano_plots`, `plot_cluster_top_markers` |
| Fig 4B | `run_gsea`, `plot_gsea` |
| Fig 5A | `run_cluster_marker_rfecv`, `run_biomarker_panel_figure` |
| Fig 5B, ED Fig 8 | `run_multiclass_consensus_classifier_cv`, `plot_multiclass_consensus_classifier` |
| ED Fig 3 | `plot_consensus_quality` |
| ED Fig 6B | `run_mutation_tuples` |
| Supp Fig 3 | `run_raw_qc`, `run_quality_control_figures` |
| Supp Fig 5 | `run_mutation_complex_oncoprint` |
| Supp Tables 2–5, Supp Data 1–2 | `run_export_supplementary` |

`reproduce.sh` runs all of these in order with the exact arguments used for the paper.

**Not reproduced by this code:**
- The clinical panels, which were made in GraphPad Prism: Fig 1, Fig 3C, ED Figs 1, 2, 4, 5,
  6A and 7, and Supp Figs 1, 2 and 6.
- The whole-slide-image panels (Fig 5D–F, ED Figs 9–10, Supp Figs 8–9). Their models and
  code are released separately.
- Supplementary Tables 1, 6 and 7.

## Installation

Three conda environments are used, with every version pinned in `envs/`. scikit-learn is pinned
to 1.5.2, under which every step runs and the classifier results reproduce identically.

| Environment | Used for |
|---|---|
| `proj` | all Python code (Python 3.10) |
| `consensus-r` | consensus clustering and every R figure (R 4.5) |
| `normics-r` | Normics/VSN and limma plate correction only (R 4.3) |

```bash
conda env create -f envs/proj.yml
conda env create -f envs/consensus-r.yml
conda env create -f envs/normics-r.yml
conda run -n proj pip install --no-deps -r envs/proj-requirements.txt
conda run -n proj pip install --no-deps -e .
```

`--no-deps` is required. `muvi` 0.1.1 ([PyPI](https://pypi.org/project/muvi/),
[MLO-lab/MuVI](https://github.com/MLO-lab/MuVI)) declares `mudata<0.3` and `seaborn>=0.13`, but the
results were produced with mudata 0.3.2 and seaborn 0.12.2, so a dependency resolver rejects the
pinned set. The consensus
classifier `cclf` is installed from
[MLO-lab/consensus-classifier](https://github.com/MLO-lab/consensus-classifier) at tag
v0.2.0.

The scripts find the R interpreters as `<envs dir>/consensus-r/bin/Rscript` and
`<envs dir>/normics-r/bin/Rscript`. Here `<envs dir>` is the parent of the active `proj`
environment, so keep the three environment names. Set `CANCERSCOUT_CONDA_ENVS_DIR` if the
environments live elsewhere. MuVI training uses a CUDA GPU when one is available.

## Inputs

All paths are relative to the repository root unless they start with `$CANCERSCOUT_DATA_DIR`
(default `data/external/`). CPTAC files can be relocated with `CANCERSCOUT_CPTAC_DIR`
(default `$CANCERSCOUT_DATA_DIR/cptac`).

**Third-party, needed in both modes**

| Path | Content and source |
|---|---|
| `references/gene_sets/msigdb/c2.cp.reactome.v7.5.1.symbols.gmt` | MSigDB v7.5.1 Reactome gene sets. If this file is missing, `run_gsea` falls back to Enrichr `Reactome_2022` and the results differ. |
| `$CANCERSCOUT_DATA_DIR/cptac/prot.csv` | CPTAC-COAD TMT tumour-vs-normal log2 ratios (Vasaikar et al., 2019), from LinkedOmics. This is `Human__CPTAC_COAD__PNNL__Proteome__TMT__03_01_2017__BCM__Gene__Tumor_Normal_log2FC.cct` transposed to 96 tumours × 6,422 genes. |
| `$CANCERSCOUT_DATA_DIR/cptac/meta.csv` | Per-tumour annotation for the same 96 tumours, indexed by the same IDs. A `UMS` column (CIN / MSI / Mesenchymal, blank if unassigned) is required; `CMS`, `Stage`, `Vital.Status` and `MSI_PCR_Result` are shown when present. |
| `$CANCERSCOUT_DATA_DIR/new_samples/crc_drug.zip` | The six expanded-view archives of Frejno et al., 2017 (Mol Syst Biol 13:951), `44320_2017_BFMSB177701_MOESM3_ESM.zip` to `MOESM8_ESM.zip`, zipped together. Members are found by `MOESM<n>` in their name, so how they are packed doesn't matter. |

These are the SHA-256 checksums of the files the results were produced from. The Frejno
workbooks are the `Table_EV*.xlsx` files inside the six archives.

```text
c08ed2adeec0af975e39fd75915301c02d03bc5e157a035796312a585ee1b6ab  c2.cp.reactome.v7.5.1.symbols.gmt
c8e48b0840c3bc4fa4b07840b838e024b1002c9cd5682be01197152976caac41  cptac/prot.csv
3efb8bdf8e833bf960eeb7504cdc7015d0e60a8fc1e1341aa552b7b2352c0504  cptac/meta.csv
8ad2c3687c98ad0cec1f3e4b2456d2e1852e5362692d8d6a1b12bc6712b8a0c5  Table_EV1.xlsx
ae8ab23198be11f6e3c9b022b2e80d6ec5673784dbd757cb351ba64cfc5f71b8  Table_EV2.xlsx
67d3f62d6451588572d523df3994df84338d04a63dd8f18732d245675e75fa34  Table_EV3.xlsx
e289aa010911ed96eaf0dfa7e10b5b8930be1cffc0be5124abd25ca7df83b292  Table_EV4.xlsx
5cf2e8c7e8d34a7eee3cae40f2ad36d1aa68351422c646a074177e69e59a2752  Table_EV5.xlsx
aa7f2c6d64cfcde8d433a0d41908d8d9de3391abdc023e54be12ccaa3b3215d5  Table_EV6.xlsx
```

**Patient-level, needed for `downstream`**

| Path | Content |
|---|---|
| `data/processed/normics_r14_goettingen_997_no_outliers_20260516/` | Processed proteome of the 997 analysable tumours: the ordered MuData `mdata_r14_goettingen_normics_997_ordered.h5mu` and the `final/` and `intermediate/` matrices. Written by stage 2. Its plate-corrected matrix is Supplementary Data 1, and the matrix before plate correction is Supplementary Data 2. |
| `data/raw/normics_r14_goettingen_output_20260428/` | Stage-1 output for all 999 tumours. The QC panels (Supp Fig 3) read it. |
| `models/production_997_seed_2605161932/` | Locked discovery model, as written by `promote_to_production`: latent factors, the 100 Leiden runs, consensus labels for k = 2–6, `metadata.csv` and `factor_adata.h5ad` with the UMAP. The six subgroups are `leiden_cons`. |
| `$CANCERSCOUT_DATA_DIR/mutation.h5ad` | Panel mutation calls (ED Fig 6B). |

**Patient-level, also needed for `all`**

| Path | Content |
|---|---|
| `$CANCERSCOUT_DATA_DIR/new_samples/2026-04-23_Goettingen_R14_Protein_Level.tsv` | Biognosys long-format protein report (one row per run and protein) of the DIA data deposited in PRIDE. |
| `$CANCERSCOUT_DATA_DIR/meta.csv` | Clinical metadata per tumour, indexed by BID, including the proteomics `Plate`. |
| `$CANCERSCOUT_DATA_DIR/MSI_consense.csv` | MSI consensus per tumour (`;`-separated; BID number, `MSI consensus`). |
| `$CANCERSCOUT_DATA_DIR/mdata.h5mu` | MuData template whose clinical observations and panel `Mutation` modality are carried into the new MuData. |

Tumour IDs `BIDnnnn` correspond to the Tumor IDs of Supplementary Table 1.

## Running

```bash
conda activate proj
export CANCERSCOUT_DATA_DIR=/path/to/external/data
./reproduce.sh              # downstream: from the processed proteome and the locked model
./reproduce.sh all          # from the Biognosys report, retraining MuVI first
```

Outputs go to `results/`, `figures/` and `reports/`; the supplementary tables go to
`results/supplementary/`. The last step, `check_plot_style`, fails if a figure doesn't
match the style spec. The longest steps are the marker RFECV and the cross-validated
classifier (hours on a multi-core machine); `all` adds MuVI training. Each script can also
be run on its own with `python -m cancerscout.<module> --help`. Most compute scripts accept
`--plots-only` or have a separate `plot_*` module, so figures can be redrawn without
refitting.

**Reproducibility:**
- Every downstream step uses a fixed seed, recorded in its run config under `results/`.
- The marker RFECV (`run_cluster_marker_rfecv`, behind Fig 5A and Supp Table 5a) originally ran
  with an earlier, unrecorded scikit-learn release. Rerun under the pinned 1.5.2, it selects the
  same genes for every subgroup, with the same rankings and test metrics as the locked run.
- `all` retrains MuVI with seed 2605161932. Its CUDA training isn't guaranteed
  to be bit-for-bit deterministic, so a retrained model can differ slightly from the locked one; `downstream`
  reproduces the paper's numbers from the locked labels.
- `all` refuses to start if any of its output folders already exist, so it can't overwrite
  a locked model.
- `run_cptac_classification` compares its predictions with any earlier `cptac_prediction`
  column. After a fresh retrain there is none, and the comparison tables are made against a
  placeholder.

## Repository layout

| Path | Contents |
|---|---|
| `cancerscout/` | Python package: `run_*` compute entry points, `plot_*` renderers, and the shared modules (`paths`, `io`, `stats`, `survival`, `plotting`, `plot_style`). |
| `r/` | Normalization, batch correction, consensus clustering and the ComplexHeatmap/ggplot figures (see [r/README.md](r/README.md)). |
| `preprocessing/` | Stage 1 and the MuData builders (see [preprocessing/README.md](preprocessing/README.md)). |
| `envs/` | Exact conda and pip environments. |
| `reproduce.sh` | The ordered command sequence for every panel and table. |

All paths come from [cancerscout/paths.py](cancerscout/paths.py). Cluster names and colours
come from [cancerscout/plot_style.py](cancerscout/plot_style.py), which
`export_plot_style` serializes for the R figures.

## License

MIT, see [LICENSE](LICENSE). Third-party data (MSigDB, CPTAC/LinkedOmics, Frejno et al.)
keep their own licenses and are not redistributed.
