#!/usr/bin/env Rscript
# Generic CancerScout matrix heatmap renderer using ComplexHeatmap.
#
# Python plot scripts prepare matrix/star/row metadata TSVs; this script owns
# all heatmap rendering so manuscript heatmaps share one visual system.

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

args <- commandArgs(trailingOnly = TRUE)
opts <- list(
  title = "",
  legend_title = "value",
  annotate = "none"
)
i <- 1L
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
  if (key == "--matrix") {
    opts$matrix <- val
  } else if (key == "--stars") {
    opts$stars <- val
  } else if (key == "--row-meta") {
    opts$row_meta <- val
  } else if (key == "--output") {
    opts$output <- val
  } else if (key == "--title") {
    opts$title <- val
  } else if (key == "--legend-title") {
    opts$legend_title <- val
  } else if (key == "--annotate") {
    opts$annotate <- val
  } else if (key == "--color-limit-quantile") {
    opts$color_limit_quantile <- as.numeric(val)
  } else if (key == "--color-limit-floor") {
    opts$color_limit_floor <- as.numeric(val)
  } else if (key == "--row-label-max-mm") {
    opts$row_label_max_mm <- as.numeric(val)
  } else {
    stop(sprintf("Unknown argument: %s", key))
  }
  i <- i + 2L
}
for (req in c("matrix", "output")) {
  if (is.null(opts[[req]])) stop(sprintf("Missing required arg: --%s", req))
}
if (!opts$annotate %in% c("none", "stars", "value_stars", "auto")) {
  stop("--annotate must be one of: none, stars, value_stars, auto")
}
# `auto`: write values + stars when the matrix is small enough to read; for
# bigger matrices keep only the significance stars so the figure stays clean.
canonical_value_threshold_cells <- 80L

read_matrix <- function(path) {
  df <- read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 2L) stop("Matrix TSV must contain row_id plus at least one value column")
  row_id <- as.character(df[[1]])
  mat_df <- df[, -1L, drop = FALSE]
  mat <- as.matrix(data.frame(lapply(mat_df, as.numeric), check.names = FALSE))
  rownames(mat) <- row_id
  colnames(mat) <- colnames(mat_df)
  mat
}

value_mat <- read_matrix(opts$matrix)
value_mat[!is.finite(value_mat)] <- 0

stars_mat <- matrix("", nrow = nrow(value_mat), ncol = ncol(value_mat), dimnames = dimnames(value_mat))
if (!is.null(opts$stars) && file.exists(opts$stars)) {
  raw_stars <- read.delim(opts$stars, check.names = FALSE, stringsAsFactors = FALSE)
  row_id <- as.character(raw_stars[[1]])
  star_df <- raw_stars[, -1L, drop = FALSE]
  stars_tmp <- as.matrix(star_df)
  rownames(stars_tmp) <- row_id
  stars_mat <- stars_tmp[rownames(value_mat), colnames(value_mat), drop = FALSE]
  stars_mat[is.na(stars_mat)] <- ""
}

row_labels <- rownames(value_mat)
row_groups <- NULL
if (!is.null(opts$row_meta) && file.exists(opts$row_meta)) {
  meta <- read.delim(opts$row_meta, check.names = FALSE, stringsAsFactors = FALSE)
  meta$row_id <- as.character(meta$row_id)
  meta <- meta[match(rownames(value_mat), meta$row_id), , drop = FALSE]
  if ("row_label" %in% colnames(meta)) {
    row_labels <- as.character(meta$row_label)
    row_labels[is.na(row_labels) | !nzchar(row_labels)] <- rownames(value_mat)[is.na(row_labels) | !nzchar(row_labels)]
  }
  if ("row_group" %in% colnames(meta)) {
    row_groups <- as.character(meta$row_group)
    row_groups[is.na(row_groups) | !nzchar(row_groups)] <- NA_character_
  }
}
names(row_labels) <- rownames(value_mat)

cluster_id_from_col <- function(x) {
  text <- as.character(x)
  text <- sub("^C([0-9]+).*$", "\\1", text)
  text
}
column_ids <- cluster_id_from_col(colnames(value_mat))
cluster_like_columns <- all(column_ids %in% style_cluster_table$cluster)
column_labels <- if (cluster_like_columns) {
  style_cluster_column_labels(column_ids)
} else {
  colnames(value_mat)
}

row_split <- NULL
if (!is.null(row_groups)) {
  known <- row_groups %in% style_cluster_table$cluster
  row_group_labels <- row_groups
  row_group_labels[known] <- style_cluster_row_labels(row_groups[known])
  if (all(!is.na(row_groups) & row_groups %in% style_cluster_table$cluster)) {
    row_split <- factor(row_group_labels, levels = style_cluster_row_labels(style_cluster_order(unique(row_groups))))
  } else {
    row_split <- factor(row_group_labels, levels = unique(row_group_labels[!is.na(row_group_labels)]))
  }
}

format_value <- function(x) {
  if (!is.finite(x)) return("")
  txt <- sprintf("%.2f", x)
  txt <- sub("\\.?0+$", "", txt)
  if (txt == "-0") txt <- "0"
  txt
}

abs_limit <- style_color_limit(value_mat, opts$color_limit_quantile, opts$color_limit_floor)
col_fun <- style_diverging_col_fun(value_mat, opts$color_limit_quantile, opts$color_limit_floor)

bottom_annotation <- if (cluster_like_columns) {
  style_cluster_bottom_annotation(column_ids, show_name = TRUE)
} else {
  NULL
}

show_title <- nzchar(opts$title)
row_label_max <- if (is.null(opts$row_label_max_mm)) style_mm("row_label_max") else opts$row_label_max_mm
heatmap_args <- modifyList(
  style_heatmap_args(
    value_mat,
    legend_title = opts$legend_title,
    row_split = row_split,
    column_title = if (show_title) opts$title else NULL
  ),
  list(
    col = col_fun,
    row_title_side = "left",
    row_labels = row_labels,
    row_names_max_width = unit(row_label_max, "mm"),
    column_labels = column_labels,
    bottom_annotation = bottom_annotation
  )
)
heatmap_args <- c(list(value_mat), heatmap_args)

resolved_annotate <- opts$annotate
if (resolved_annotate == "auto") {
  resolved_annotate <- if ((nrow(value_mat) * ncol(value_mat)) <= canonical_value_threshold_cells) {
    "value_stars"
  } else {
    "stars"
  }
}
if (resolved_annotate != "none") {
  heatmap_args$cell_fun <- function(j, i, x, y, width, height, fill) {
    star <- stars_mat[i, j]
    value <- value_mat[i, j]
    if (resolved_annotate == "stars") {
      # No value: keep the existing centred-star behaviour.
      if (!is.na(star) && nzchar(star)) {
        txt_col <- style_cell_text_col(value, abs_limit)
        grid.text(star, x, y, gp = style_gp("annot", col = txt_col))
      }
      return()
    }
    # value_stars: value centred, star as small superscript top-right.
    if (!is.finite(value)) return()
    txt_col <- style_cell_text_col(value, abs_limit)
    grid.text(format_value(value), x, y, gp = style_gp("annot", col = txt_col))
    style_cell_star(star, x, y, width, height, txt_col)
  }
}

ht <- do.call(Heatmap, heatmap_args)

written <- style_heatmap_save(
  opts$output,
  ht,
  heatmap_legend_side = "right",
  annotation_legend_side = "right",
  merge_legend = TRUE,
  padding = unit(rep(style_mm("pad"), 4), "mm")
)

cat("Wrote:", paste(written, collapse = " "), "\n")
