#!/usr/bin/env Rscript
# Locked Leiden consensus block heatmap using cola's native consensus_heatmap().
# This script visualizes the existing repeated-Leiden consensus and final
# leiden_cons assignments; it does not run a new cola partition.
#
# Usage: Rscript r/consensus_leiden_cola_heatmap.R <model_dir> [output_png]

suppressPackageStartupMessages({
  library(cola)
  library(cluster)
  library(circlize)
  library(grid)
})

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repo_root <- if (!is.na(script_file) && nzchar(script_file)) {
  normalizePath(file.path(dirname(normalizePath(script_file, mustWork = FALSE)), ".."), mustWork = FALSE)
} else {
  getwd()
}
source(file.path(repo_root, "r", "plot_style.R"))

args <- commandArgs(trailingOnly = TRUE)
model_dir <- if (length(args) >= 1) args[1] else file.path(repo_root, "models", "production_997_seed_2605161932")
output_png <- if (length(args) >= 2) args[2] else file.path(repo_root, "figures", "muvi", "consensus_leiden_cola_heatmap.png")

leiden_path <- file.path(model_dir, "leiden.csv")
cons_path <- file.path(model_dir, "leiden_cons.csv")
summary_path <- file.path(model_dir, "consensus_summary.csv")
if (!file.exists(leiden_path)) stop("Missing repeated Leiden assignments: ", leiden_path)
if (!file.exists(cons_path)) stop("Missing locked Leiden assignments: ", cons_path)

leiden_df <- read.csv(leiden_path, row.names = 1, check.names = FALSE)
leiden_mat <- as.matrix(leiden_df)
storage.mode(leiden_mat) <- "character"
sample_ids <- rownames(leiden_mat)

cons_df <- read.csv(cons_path, check.names = FALSE)
if (!all(c("sample_id", "leiden_cons") %in% colnames(cons_df))) {
  stop("leiden_cons.csv must contain sample_id and leiden_cons columns")
}
cons_df$sample_id <- as.character(cons_df$sample_id)
class_ids <- setNames(as.character(cons_df$leiden_cons), cons_df$sample_id)[sample_ids]
if (any(is.na(class_ids))) {
  missing_ids <- sample_ids[is.na(class_ids)]
  stop("Missing locked Leiden labels for samples: ", paste(head(missing_ids, 10), collapse = ", "))
}

selected_k <- length(unique(class_ids))
if (file.exists(summary_path)) {
  summary_df <- read.csv(summary_path, check.names = FALSE)
  if ("selected_best_k" %in% colnames(summary_df) && nrow(summary_df) >= 1) {
    selected_k <- as.integer(summary_df$selected_best_k[1])
  }
}
if (selected_k != length(unique(class_ids))) {
  warning("selected_best_k=", selected_k, " but leiden_cons has ", length(unique(class_ids)), " groups; using leiden_cons groups for labels.")
  selected_k <- length(unique(class_ids))
}

compute_consensus <- function(mat) {
  n <- nrow(mat)
  sim <- matrix(0, nrow = n, ncol = n, dimnames = list(rownames(mat), rownames(mat)))
  for (run_idx in seq_len(ncol(mat))) {
    sim <- sim + outer(mat[, run_idx], mat[, run_idx], FUN = "==")
  }
  sim <- sim / ncol(mat)
  diag(sim) <- 1
  sim
}

consensus_mat <- compute_consensus(leiden_mat)
hamming_dist_mat <- 1 - consensus_mat
diag(hamming_dist_mat) <- 0
class_levels <- sort(unique(class_ids))
# Short C1-C6 keys throughout: at full length the C1 entry
# ("C1 Stromal ECM-rich / Mesenchymal") sets the legend and axis-label widths,
# which squeezed the consensus body out of square and clipped both edges.
# Full names belong in the caption.
class_labels <- style_cluster_row_labels(class_levels)
class_factor <- factor(
  unname(class_labels[class_ids]),
  levels = unname(class_labels[class_levels])
)
class_int <- as.integer(class_factor)

sil <- silhouette(class_int, as.dist(hamming_dist_mat))[, "sil_width"]
sil <- pmax(0, pmin(1, sil))

membership_mat <- sapply(class_levels, function(level) {
  rowMeans(consensus_mat[, class_ids == level, drop = FALSE])
})
if (is.null(dim(membership_mat))) {
  membership_mat <- matrix(membership_mat, ncol = 1)
}
colnames(membership_mat) <- unname(class_labels[class_levels])
rownames(membership_mat) <- sample_ids

class_df <- data.frame(
  class = as.character(class_factor),
  silhouette = as.numeric(sil),
  row.names = sample_ids,
  check.names = FALSE
)

cluster_colors <- style_cluster_colors(class_levels)[class_levels]
names(cluster_colors) <- unname(class_labels[class_levels])
cola_opt(color_set_2 = cluster_colors)

cp <- methods::new(
  "ConsensusPartition",
  object_list = list(list(
    consensus = consensus_mat,
    membership = membership_mat,
    class_df = class_df
  )),
  k = as.numeric(selected_k),
  n_partition = as.numeric(ncol(leiden_mat)),
  partition_method = "locked Leiden",
  top_value_method = "locked Leiden",
  top_n = as.numeric(NA),
  top_value_list = numeric(0),
  anno = NULL,
  anno_col = NULL,
  scale_rows = FALSE,
  sample_by = "row",
  column_index = seq_len(nrow(leiden_mat)),
  row_index = seq_len(nrow(leiden_mat)),
  running_time = NA,
  cache = list(),
  others = list(source = "leiden.csv + leiden_cons.csv"),
  hash = "locked_leiden_consensus",
  .env = new.env(parent = emptyenv())
)

# Sized so the consensus block itself reads square: the left-hand membership,
# silhouette and class tracks plus the right-hand legend column sit outside it,
# so the canvas is wider than tall. cola draws internally, so this cannot be
# measured the way style_heatmap_save does.
written <- style_save(output_png, 7.2, 5.4, function() {
  # The title cannot be suppressed here. cola::consensus_heatmap builds it
  # unconditionally from partition_method and n_partition, ignoring any
  # column_title passed through `...`, and its only escape hatch (internal =
  # TRUE) also drops all four legends. Removing it would mean reimplementing
  # cola's layout from the components assembled above.
  consensus_heatmap(
    cp,
    k = selected_k,
    show_row_names = FALSE,
    show_column_names = FALSE
  )
})
cat("Wrote:", paste(written, collapse = " "), "\n")
