# Preprocessing

- `normics_long_format_pipeline.py` (stage 1) starts from the Biognosys protein-level
  report. It selects the best run per tumour, joins the clinical metadata and MSI
  consensus, writes the Normics inputs, and runs VSN and limma for all 999 tumours. It
  also writes the raw matrices read by the QC figures. Output goes to
  `data/raw/normics_r14_goettingen_output_20260428/`.
- `build_final_mudata.py` builds the MuData object (Proteome plus the panel Mutation
  modality) from the plate-corrected matrix.
- `order_mudata_by_numeric_obs.py` orders the MuData observations by numeric ID.

`python -m cancerscout.run_normics_no_outliers` (stage 2) reruns VSN and limma without
the two outlier tumours, then calls the two MuData scripts. The R code lives in `r/`.
