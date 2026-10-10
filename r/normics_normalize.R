#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(vsn)
    library(Biobase)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("Usage: Rscript r/normics_normalize.R data_all.csv data_subset.csv outdir")
}

data_all_path <- args[1]
subset_path   <- args[2]
outdir        <- args[3]

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

load_numeric_csv <- function(path) {
    df <- read.csv(path, header = TRUE, check.names = FALSE, row.names = 1)
    df[] <- lapply(df, function(x) as.numeric(as.character(x)))
    mat <- as.matrix(df)

    if (!is.numeric(mat)) {
        stop("Matrix contains non-numeric values after conversion.")
    }
    if (any(mat < 0, na.rm = TRUE)) {
        stop("Negative values found. Expected raw positive intensities.")
    }

    return(mat)
}

cat("\n--- Loading CSVs ---\n")
data_all <- load_numeric_csv(data_all_path)
data_sub <- load_numeric_csv(subset_path)

cat("Loaded:\n")
cat("  data_all:", dim(data_all)[1], "proteins x", dim(data_all)[2], "samples\n")
cat("  data_sub:", dim(data_sub)[1], "proteins x", dim(data_sub)[2], "samples\n\n")

if (!identical(colnames(data_all), colnames(data_sub))) {
    stop("Sample columns of data_all and data_subset do not match exactly.")
}

###############################################################################
# 1) NORMICS (subset-trained VSN -> applied to all proteins)
###############################################################################
cat("Running Normics (subset-trained VSN)...\n")

es_sub <- ExpressionSet(assayData = data_sub)

# paper says vsn2 with lts.quantile = 0.5 for robustness
fit_sub <- vsn2(es_sub, lts.quantile = 0.5)

# apply subset-trained fit to full matrix
normics_vsn <- predict(fit_sub, newdata = data_all, log2scale = TRUE)

write.csv(normics_vsn,
          file.path(outdir, "normalized_normics_vsn_log2.csv"),
          row.names = TRUE)

# store subset fit coefficients for reproducibility
coef_sub <- coef(fit_sub)
write.csv(coef_sub,
          file.path(outdir, "normics_vsn_subset_fit_coefficients.csv"),
          row.names = TRUE)

###############################################################################
# 2) NORMICS_median (subset-derived sample factors -> applied to all proteins)
###############################################################################
cat("Running Normics_median (subset-based median normalization)...\n")

# compute per-sample medians on the subset only
subset_medians <- apply(data_sub, 2, median, na.rm = TRUE)

if (any(is.na(subset_medians))) {
    stop("Subset medians contain NA values; subset may be too sparse.")
}
if (any(subset_medians <= 0)) {
    stop("Subset medians must be positive for multiplicative median normalization.")
}

# common reference = median of sample medians
ref_median <- median(subset_medians, na.rm = TRUE)

# multiplicative sample-specific scaling factors
scale_factors <- ref_median / subset_medians

normics_median <- sweep(data_all, 2, scale_factors, FUN = "*")

write.csv(normics_median,
          file.path(outdir, "normalized_normics_median.csv"),
          row.names = TRUE)

write.csv(
    data.frame(
        Sample = names(subset_medians),
        Subset_median = as.numeric(subset_medians),
        Scale_factor = as.numeric(scale_factors)
    ),
    file.path(outdir, "normics_median_scaling_factors.csv"),
    row.names = FALSE
)

###############################################################################
# done
###############################################################################
cat("\nDone.\n")
cat("Wrote:\n")
cat("  normalized_normics_vsn_log2.csv\n")
cat("  normalized_normics_median.csv\n")
cat("  normics_vsn_subset_fit_coefficients.csv\n")
cat("  normics_median_scaling_factors.csv\n")
cat("Output directory:", normalizePath(outdir), "\n\n")