#!/usr/bin/env Rscript
# Top-3 up + top-3 down marker heatmap per cluster (ComplexHeatmap).
#
# For each cluster, selects the top-N markers with the most positive and the
# most negative effect-size value and renders a single heatmap where rows
# (selected genes) are grouped by cluster and ordered up then down when both
# directions are requested. Cell colour is the effect size; significance stars
# come from a second column (either FDR p-values or selection frequencies).
#
# Usage:
#   Rscript r/cluster_top_markers_heatmap.R \
#       --input <path.tsv> \
#       --value-col <name> \
#       --sig-col <name> \
#       --sig-mode <fdr|frequency> \
#       --direction <up|down|both> \
#       --output <path.png> \
#       [--top-n 3] \
#       [--tiebreak <name>]

suppressPackageStartupMessages({
  library(ComplexHeatmap)
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

# ---- arg parsing ------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
opts <- list(
  top_n = 3L,
  tiebreak = NA_character_,
  selection_col = NA_character_,
  filter_col = NA_character_,
  filter_mode = NA_character_,
  legend_title = NA_character_
)
i <- 1L
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
  if (key == "--input") {
    opts$input <- val
  } else if (key == "--value-col") {
    opts$value_col <- val
  } else if (key == "--selection-col") {
    # column used to rank candidates per cluster (defaults to --value-col).
    opts$selection_col <- val
  } else if (key == "--sig-col") {
    opts$sig_col <- val
  } else if (key == "--sig-mode") {
    opts$sig_mode <- val
  } else if (key == "--filter-col") {
    # column used to gate per-cluster eligibility (defaults to --sig-col).
    opts$filter_col <- val
  } else if (key == "--filter-mode") {
    # threshold encoding for the filter column (defaults to --sig-mode).
    opts$filter_mode <- val
  } else if (key == "--legend-title") {
    # human-readable colour-bar / legend title (defaults to --value-col).
    opts$legend_title <- val
  } else if (key == "--direction") {
    opts$direction <- val
  } else if (key == "--output") {
    opts$output <- val
  } else if (key == "--top-n") {
    opts$top_n <- as.integer(val)
  } else if (key == "--tiebreak") {
    opts$tiebreak <- val
  } else {
    stop(sprintf("Unknown argument: %s", key))
  }
  i <- i + 2L
}
for (req in c("input", "value_col", "sig_col", "sig_mode", "direction", "output")) {
  if (is.null(opts[[req]])) stop(sprintf("Missing required arg: --%s", gsub("_", "-", req)))
}
if (is.na(opts$selection_col)) opts$selection_col <- opts$value_col
if (is.na(opts$filter_col)) opts$filter_col <- opts$sig_col
if (is.na(opts$filter_mode)) opts$filter_mode <- opts$sig_mode
if (is.na(opts$legend_title) || !nzchar(opts$legend_title)) opts$legend_title <- opts$value_col
if (!opts$sig_mode %in% c("fdr", "frequency")) {
  stop("--sig-mode must be 'fdr' or 'frequency'")
}
if (!opts$filter_mode %in% c("fdr", "frequency")) {
  stop("--filter-mode must be 'fdr' or 'frequency'")
}
if (!opts$direction %in% c("up", "down", "both")) {
  stop("--direction must be 'up', 'down', or 'both'")
}

# ---- load ------------------------------------------------------------------
df <- read.delim(opts$input, check.names = FALSE, stringsAsFactors = FALSE)
required <- c("cluster", "gene_symbol", opts$value_col, opts$sig_col)
if (!all(required %in% colnames(df))) {
  stop(sprintf("Input is missing columns: %s",
               paste(setdiff(required, colnames(df)), collapse = ", ")))
}
df$cluster <- as.character(df$cluster)
df$gene_symbol <- as.character(df$gene_symbol)
df$value <- as.numeric(df[[opts$value_col]])
df$sig <- as.numeric(df[[opts$sig_col]])
if (opts$selection_col %in% colnames(df)) {
  df$selection_value <- as.numeric(df[[opts$selection_col]])
} else {
  stop(sprintf("--selection-col '%s' not found in input", opts$selection_col))
}
if (opts$filter_col %in% colnames(df)) {
  df$filter_value <- as.numeric(df[[opts$filter_col]])
} else {
  stop(sprintf("--filter-col '%s' not found in input", opts$filter_col))
}
if (!is.na(opts$tiebreak) && opts$tiebreak %in% colnames(df)) {
  df$tiebreak <- as.numeric(df[[opts$tiebreak]])
} else {
  df$tiebreak <- 0
}

# ---- per-cluster top-N up / down -------------------------------------------
cluster_order <- as.character(seq_len(6))
present <- intersect(cluster_order, unique(df$cluster))
if (length(present) == 0L) stop("No known clusters in input")

pass_filter <- function(vals, mode) {
  if (mode == "fdr") {
    is.finite(vals) & vals < 0.05
  } else {
    is.finite(vals) & vals >= 0.3
  }
}

top_n <- opts$top_n
selected_list <- list()
directions <- if (opts$direction == "both") c("up", "down") else opts$direction
for (cluster_id in present) {
  for (direction in directions) {
    sub <- df[df$cluster == cluster_id & is.finite(df$selection_value) & pass_filter(df$filter_value, opts$filter_mode), , drop = FALSE]
    if (nrow(sub) == 0L) next
    if (direction == "up") {
      sub <- sub[sub$selection_value > 0, , drop = FALSE]
      if (nrow(sub) == 0L) next
      sub <- sub[order(-sub$selection_value, -sub$tiebreak, sub$sig), , drop = FALSE]
    } else {
      sub <- sub[sub$selection_value < 0, , drop = FALSE]
      if (nrow(sub) == 0L) next
      sub <- sub[order(sub$selection_value, -sub$tiebreak, sub$sig), , drop = FALSE]
    }
    sub <- head(sub, top_n)
    if (nrow(sub) == 0L) next
    sub$assign_cluster <- cluster_id
    sub$assign_direction <- direction
    selected_list[[paste(cluster_id, direction, sep = "::")]] <- sub
  }
}
if (length(selected_list) == 0L) stop("No markers survived the per-cluster filter")
selected <- do.call(rbind, selected_list)
selected$row_id <- make.unique(
  paste(selected$assign_cluster, selected$assign_direction, selected$gene_symbol, sep = "::"),
  sep = "::"
)

# ---- build value/sig matrices (selected genes × all clusters) ---------------
genes <- as.character(selected$gene_symbol)
row_ids <- selected$row_id
n_rows <- length(row_ids)

value_mat <- matrix(NA_real_, nrow = n_rows, ncol = length(cluster_order),
                    dimnames = list(row_ids, cluster_order))
sig_mat <- matrix(NA_real_, nrow = n_rows, ncol = length(cluster_order),
                  dimnames = list(row_ids, cluster_order))
lookup_key <- paste(df$gene_symbol, df$cluster, sep = "\r")
value_lookup <- setNames(df$value, lookup_key)
sig_lookup <- setNames(df$sig, lookup_key)
for (r in seq_len(n_rows)) {
  for (cluster_id in cluster_order) {
    key <- paste(genes[r], cluster_id, sep = "\r")
    if (key %in% names(value_lookup)) value_mat[r, cluster_id] <- value_lookup[[key]]
    if (key %in% names(sig_lookup)) sig_mat[r, cluster_id] <- sig_lookup[[key]]
  }
}
value_mat[!is.finite(value_mat)] <- 0

# ---- significance stars ----------------------------------------------------
star_fun <- if (opts$sig_mode == "fdr") {
  function(p) ifelse(!is.finite(p), "",
                     ifelse(p <= 0.001, "***",
                            ifelse(p <= 0.01, "**",
                                   ifelse(p <= 0.05, "*", ""))))
} else {
  function(f) ifelse(!is.finite(f), "",
                     ifelse(f >= 0.9, "***",
                            ifelse(f >= 0.5, "**",
                                   ifelse(f >= 0.3, "*", ""))))
}
stars_mat <- matrix(star_fun(as.vector(sig_mat)),
                    nrow = nrow(sig_mat), ncol = ncol(sig_mat),
                    dimnames = dimnames(sig_mat))

# ---- style (shared with other ComplexHeatmap scripts) ----------------------
column_labels <- style_cluster_column_labels(cluster_order)

abs_limit <- style_color_limit(value_mat)
col_fun <- style_diverging_col_fun(value_mat)

# Row split: one block per assign_cluster, with cluster blocks ordered 1..6.
row_split_lookup <- style_cluster_row_labels(cluster_order)
row_split_labels <- row_split_lookup[selected$assign_cluster]
row_split_labels[is.na(row_split_labels)] <- paste0("C", selected$assign_cluster)
row_split <- factor(
  row_split_labels,
  levels = intersect(style_cluster_row_labels(cluster_order), unique(row_split_labels))
)

direction_labels <- ifelse(selected$assign_direction == "up", "↑ ", "↓ ")
row_labels <- paste0(direction_labels, genes)
names(row_labels) <- row_ids

ht <- do.call(Heatmap, c(list(value_mat), modifyList(
  style_heatmap_args(value_mat, legend_title = opts$legend_title, row_split = row_split),
  list(
    col = col_fun,
    row_title_side = "left",
    row_labels = row_labels,
    column_labels = column_labels,
    bottom_annotation = style_cluster_bottom_annotation(cluster_order, show_name = TRUE)
  )
), list(
  cell_fun = function(j, i, x, y, width, height, fill) {
    val <- value_mat[i, j]
    if (!is.finite(val)) return()
    txt_col <- style_cell_text_col(val, abs_limit)
    val_str <- sub("\\.?0+$", "", sprintf("%.2f", val))
    if (val_str == "-0") val_str <- "0"
    grid.text(val_str, x, y, gp = style_gp("annot", col = txt_col))
    star <- stars_mat[i, j]
    style_cell_star(star, x, y, width, height, txt_col)
  }
)))

# ---- save ------------------------------------------------------------------
written <- style_heatmap_save(
  opts$output,
  ht,
  heatmap_legend_side = "right",
  annotation_legend_side = "right",
  merge_legend = TRUE,
  padding = unit(rep(style_mm("pad"), 4), "mm")
)

cat("Wrote:", paste(written, collapse = " "), "\n")
