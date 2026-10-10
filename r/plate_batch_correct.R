#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript plate_batch_correct.R normalized_bid_matrix.csv sample_metadata.csv outdir")
}

matrix_path <- args[1]
meta_path <- args[2]
outdir <- args[3]

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

mat_df <- read.csv(matrix_path, header = TRUE, row.names = 1, check.names = FALSE)
meta_df <- read.csv(meta_path, header = TRUE, check.names = FALSE)

if (!"sample" %in% colnames(meta_df)) {
    stop("sample_metadata.csv must contain a 'sample' column.")
}
if (!"Plate" %in% colnames(meta_df)) {
    stop("sample_metadata.csv must contain a 'Plate' column.")
}

meta_df <- meta_df[meta_df$sample %in% colnames(mat_df), , drop = FALSE]
meta_df <- meta_df[match(colnames(mat_df), meta_df$sample), , drop = FALSE]

if (!identical(meta_df$sample, colnames(mat_df))) {
    stop("Sample order mismatch between matrix and metadata.")
}

if (any(is.na(meta_df$Plate))) {
    stop("Plate contains NA values.")
}

mat <- as.matrix(mat_df)
mode(mat) <- "numeric"

batch <- factor(meta_df$Plate)
if ("MSI_final" %in% colnames(meta_df) && all(!is.na(meta_df$MSI_final))) {
    protected_msi <- TRUE
    msi <- factor(meta_df$MSI_final)
    design <- model.matrix(~ msi)
    msi_out <- as.character(msi)
} else {
    protected_msi <- FALSE
    design <- matrix(1, nrow = nrow(meta_df), ncol = 1)
    msi_out <- if ("MSI_final" %in% colnames(meta_df)) as.character(meta_df$MSI_final) else rep(NA_character_, nrow(meta_df))
}

corrected <- removeBatchEffect(mat, batch = batch, design = design)

write.csv(
    corrected,
    file = file.path(outdir, "normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"),
    row.names = TRUE
)

write.csv(
    data.frame(
        sample = meta_df$sample,
        Plate = as.character(batch),
        MSI_final = msi_out,
        msi_protected = protected_msi
    ),
    file = file.path(outdir, "plate_batch_correction_samples.csv"),
    row.names = FALSE
)

cat("Done.\n")
cat("Wrote:\n")
cat("  normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv\n")
cat("  plate_batch_correction_samples.csv\n")
cat("Output directory:", normalizePath(outdir), "\n")
