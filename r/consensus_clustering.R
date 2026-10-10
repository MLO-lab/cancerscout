#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(circlize)
  library(ComplexHeatmap)
  library(cluster)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript r/consensus_clustering.R <results_path> [predefined_k] [max_k]")
}

results_path <- path.expand(args[1])
predefined_k <- if (length(args) >= 2 && nzchar(args[2])) strtoi(args[2]) else NA_integer_
user_max_k <- if (length(args) >= 3 && nzchar(args[3])) strtoi(args[3]) else NA_integer_

latent_path <- file.path(results_path, "latent.csv")
leiden_path <- file.path(results_path, "leiden.csv")

if (!file.exists(latent_path)) {
  stop(sprintf("Missing latent.csv at %s", latent_path))
}
if (!file.exists(leiden_path)) {
  stop(sprintf("Missing leiden.csv at %s", leiden_path))
}

latent_df <- read.csv(latent_path, row.names = 1, check.names = FALSE)
leiden_df <- read.csv(leiden_path, row.names = 1, check.names = FALSE)

if (!identical(rownames(leiden_df), rownames(latent_df))) {
  warning("latent.csv and leiden.csv rownames differ; continuing with leiden.csv rownames as sample ids.")
}

sample_ids <- rownames(leiden_df)
leiden_mat <- as.matrix(leiden_df)
storage.mode(leiden_mat) <- "character"

n_samples <- nrow(leiden_mat)
n_runs <- ncol(leiden_mat)
if (n_samples < 2 || n_runs < 2) {
  stop("Need at least 2 samples and 2 repeated Leiden runs.")
}

observed_max_k <- suppressWarnings(max(as.numeric(leiden_mat), na.rm = TRUE))
if (is.infinite(observed_max_k) || is.na(observed_max_k)) {
  stop("Could not determine max_k from leiden assignments.")
}

recommended_max_k <- 10L
max_k <- if (!is.na(user_max_k)) user_max_k else min(observed_max_k, recommended_max_k)
max_k <- min(max_k, n_samples - 1L)
if (max_k < 2L) {
  stop("max_k must be at least 2.")
}

message("results_path: ", results_path)
message("samples: ", n_samples)
message("repeated Leiden runs: ", n_runs)
message("max_k: ", max_k)

compute_hamming_dist <- function(mat) {
  n <- nrow(mat)
  d <- matrix(0, nrow = n, ncol = n, dimnames = list(rownames(mat), rownames(mat)))
  for (i in seq_len(n)) {
    for (j in i:n) {
      mismatches <- mean(mat[i, ] != mat[j, ])
      d[i, j] <- mismatches
      d[j, i] <- mismatches
    }
  }
  d
}

hamming_dist_mat <- compute_hamming_dist(leiden_mat)
similarity_mat <- 1 - hamming_dist_mat
diag(similarity_mat) <- 1

hamming_dist <- as.dist(hamming_dist_mat)
hc <- hclust(hamming_dist, method = "average")

evaluate_k <- function(k) {
  cluster_ids <- cutree(hc, k = k)
  widths <- silhouette(cluster_ids, hamming_dist)[, "sil_width"]
  tibble(
    k = k,
    mean_silhouette = mean(widths, na.rm = TRUE),
    median_silhouette = median(widths, na.rm = TRUE),
    min_cluster_size = min(tabulate(cluster_ids))
  )
}

score_df <- map_dfr(2:max_k, evaluate_k)
if (nrow(score_df) == 0) {
  stop("No valid k values evaluated.")
}

suggested_best_k <- score_df %>%
  arrange(desc(mean_silhouette), desc(median_silhouette), desc(min_cluster_size), k) %>%
  slice(1) %>%
  pull(k)

best_k <- if (!is.na(predefined_k)) predefined_k else suggested_best_k
if (best_k < 2L || best_k > max_k) {
  stop(sprintf("selected best_k=%s is outside evaluated range [2, %s].", best_k, max_k))
}

message("suggested_best_k: ", suggested_best_k)
message("selected best_k: ", best_k)

cluster_ids_best <- cutree(hc, k = best_k)
cluster_levels <- sort(unique(cluster_ids_best))
cluster_labels <- paste("Cluster", cluster_levels)
cluster_factor <- factor(cluster_ids_best, levels = cluster_levels, labels = cluster_labels)
fixed_cluster_colors <- c(
  "Cluster 1" = "#452288",
  "Cluster 2" = "#ED7C5A",
  "Cluster 3" = "#B5D33D",
  "Cluster 4" = "#FED23E",
  "Cluster 5" = "#6CA2EA",
  "Cluster 6" = "#A97424"
)
if (all(cluster_labels %in% names(fixed_cluster_colors))) {
  cluster_colors <- fixed_cluster_colors[cluster_labels]
} else {
  cluster_colors <- structure(rand_color(length(cluster_levels)), names = cluster_labels)
}

ordered_ids <- sample_ids[hc$order]
ordered_clusters <- cluster_factor[hc$order]
cluster_ha <- HeatmapAnnotation(
  Cluster = ordered_clusters,
  col = list(Cluster = cluster_colors),
  show_annotation_name = TRUE
)

pdf(file.path(results_path, "Consensus_Heatmap.pdf"), width = 10, height = 8)
draw(
  Heatmap(
    similarity_mat[ordered_ids, ordered_ids],
    name = "1 - Hamming",
    col = colorRamp2(c(0, 0.5, 1), c("#f7fbff", "#6baed6", "#08306b")),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = FALSE,
    show_column_names = FALSE,
    top_annotation = cluster_ha,
    left_annotation = rowAnnotation(
      Cluster = ordered_clusters,
      col = list(Cluster = cluster_colors),
      show_annotation_name = FALSE
    )
  )
)
dev.off()

pdf(file.path(results_path, "SelectPartition.pdf"), width = 8, height = 5)
plot(
  score_df$k,
  score_df$mean_silhouette,
  type = "b",
  pch = 16,
  xlab = "k",
  ylab = "Mean silhouette",
  main = "Hamming Consensus Partition Selection"
)
abline(v = best_k, col = "#d62728", lty = 2)
dev.off()

pdf(file.path(results_path, "Consensus_Scores.pdf"), width = 8, height = 5)
plot(
  score_df$k,
  score_df$mean_silhouette,
  type = "b",
  pch = 16,
  xlab = "k",
  ylab = "Mean silhouette",
  main = "Hamming Consensus Scores"
)
abline(v = best_k, col = "#d62728", lty = 2)
dev.off()

membership_mat <- matrix(
  as.integer(cluster_factor),
  ncol = 1,
  dimnames = list(sample_ids, "Cluster")
)

pdf(file.path(results_path, "Membership_Heatmap.pdf"), width = 4, height = 8)
draw(
  Heatmap(
    membership_mat[ordered_ids, , drop = FALSE],
    name = "Cluster",
    col = structure(unname(cluster_colors), names = seq_along(cluster_colors)),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = FALSE,
    show_column_names = TRUE
  )
)
dev.off()

pdf(file.path(results_path, "All_Collect_Plots.pdf"), width = 12, height = 10)
draw(
  Heatmap(
    similarity_mat[ordered_ids, ordered_ids],
    name = "1 - Hamming",
    col = colorRamp2(c(0, 0.5, 1), c("#f7fbff", "#6baed6", "#08306b")),
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_names = FALSE,
    show_column_names = FALSE,
    top_annotation = cluster_ha
  )
)
dev.off()

summary_df <- tibble(
  method = "hamming_clean",
  suggested_best_k = suggested_best_k,
  selected_best_k = best_k,
  max_k = max_k,
  n_samples = n_samples,
  n_runs = n_runs
) %>%
  bind_cols(
    score_df %>%
      filter(k == best_k) %>%
      select(mean_silhouette, median_silhouette, min_cluster_size)
  )
write.csv(summary_df, file.path(results_path, "consensus_summary.csv"), row.names = FALSE)
write.csv(score_df, file.path(results_path, "consensus_scores_by_k.csv"), row.names = FALSE)

res_class <- tibble(
  sample_id = sample_ids,
  leiden_cons = cluster_ids_best
)
write.csv(res_class, file.path(results_path, "leiden_cons.csv"), row.names = FALSE)

for (k in 2:max_k) {
  res_k <- tibble(
    sample_id = sample_ids,
    leiden_cons = cutree(hc, k = k)
  )
  write.csv(res_k, file.path(results_path, sprintf("leiden_cons_k_%s.csv", k)), row.names = FALSE)
}

cat("Done.\n")
cat("Wrote outputs to:\n")
cat("  ", results_path, "\n", sep = "")
cat("Selected best_k: ", best_k, "\n", sep = "")
