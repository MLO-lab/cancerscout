#!/usr/bin/env Rscript
# Manuscript-facing GSEA NES + ssGSEA mean-shift pathway heatmap using ComplexHeatmap.
#
# Usage:
#   Rscript r/gsea_combined_heatmap.R <gsea_results_dir> <output_file> [top_n]

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
if (length(args) < 2) {
  stop("Usage: Rscript r/gsea_combined_heatmap.R <gsea_results_dir> <output_file> [top_n]")
}

results_dir <- args[1]
output_file <- args[2]
top_n <- if (length(args) >= 3) as.integer(args[3]) else 3L
if (is.na(top_n) || top_n < 1L) top_n <- 3L

read_tsv <- function(path) {
  if (!file.exists(path)) return(data.frame())
  read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
}

gsea_pruned <- read_tsv(file.path(results_dir, "gsea_nonredundant_significant_terms.tsv"))
if (nrow(gsea_pruned) == 0L) {
  gsea_pruned <- read_tsv(file.path(results_dir, "gsea_significant_terms_with_redundancy.tsv"))
}
gsea_all <- read_tsv(file.path(results_dir, "gsea_prerank_all_terms.tsv"))
shifts <- read_tsv(file.path(results_dir, "ssgsea_cluster_mean_shifts.tsv"))

if (nrow(gsea_pruned) == 0L) {
  stop("No GSEA significant terms found: ", results_dir)
}

term_col <- if ("Term" %in% colnames(gsea_pruned)) "Term" else "Name"
fdr_col <- if ("FDR q-val" %in% colnames(gsea_pruned)) "FDR q-val" else "FDR"
nes_col <- "NES"

if (!all(c(term_col, fdr_col, nes_col, "cluster") %in% colnames(gsea_pruned))) {
  stop("GSEA significant table is missing required columns")
}

if ("redundant" %in% colnames(gsea_pruned)) {
  redundant <- tolower(as.character(gsea_pruned$redundant)) %in% c("true", "1", "yes")
  nonredundant <- gsea_pruned[!redundant, , drop = FALSE]
} else {
  nonredundant <- gsea_pruned
}
if (nrow(nonredundant) == 0L) {
  stop("No non-redundant GSEA terms found")
}

nonredundant[[nes_col]] <- as.numeric(nonredundant[[nes_col]])
nonredundant[[fdr_col]] <- as.numeric(nonredundant[[fdr_col]])
nonredundant$cluster <- as.character(nonredundant$cluster)
nonredundant <- nonredundant[
  is.finite(nonredundant[[nes_col]]) & !is.na(nonredundant$cluster) & nzchar(nonredundant$cluster),
  ,
  drop = FALSE
]
if (nrow(nonredundant) == 0L) {
  stop("No finite non-redundant GSEA terms found")
}

nes_src <- if (nrow(gsea_all) > 0L) gsea_all else nonredundant
all_term_col <- if ("Term" %in% colnames(nes_src)) "Term" else "Name"
all_nes_col <- "NES"
all_fdr_col <- if ("FDR q-val" %in% colnames(nes_src)) {
  "FDR q-val"
} else if ("FDR" %in% colnames(nes_src)) {
  "FDR"
} else {
  NA_character_
}
if (!all(c(all_term_col, all_nes_col, "cluster") %in% colnames(nes_src))) {
  stop("GSEA all-terms table is missing required columns")
}
nes_src[[all_nes_col]] <- as.numeric(nes_src[[all_nes_col]])
if (!is.na(all_fdr_col)) {
  nes_src[[all_fdr_col]] <- as.numeric(nes_src[[all_fdr_col]])
}
nes_src$cluster <- as.character(nes_src$cluster)

cluster_sort_key <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}
all_clusters <- unique(c(
  as.character(nes_src$cluster),
  as.character(nonredundant$cluster),
  if (nrow(shifts) > 0L && "cluster" %in% colnames(shifts)) as.character(shifts$cluster) else character(0)
))
cluster_order <- as.character(seq_len(6))
if (!all(cluster_order %in% all_clusters)) {
  cluster_order <- sort(unique(all_clusters), na.last = TRUE)
}
numeric_order <- cluster_sort_key(cluster_order)
cluster_order <- cluster_order[order(is.na(numeric_order), numeric_order, cluster_order)]

selected_parts <- lapply(cluster_order, function(cluster_id) {
  rows <- nonredundant[
    nonredundant$cluster == cluster_id &
      is.finite(nonredundant[[nes_col]]),
    ,
    drop = FALSE
  ]
  if (nrow(rows) == 0L) return(NULL)
  positive <- rows[rows[[nes_col]] > 0, , drop = FALSE]
  negative <- rows[rows[[nes_col]] < 0, , drop = FALSE]
  positive <- positive[
    order(-positive[[nes_col]], positive[[fdr_col]], positive[[term_col]], na.last = TRUE),
    ,
    drop = FALSE
  ]
  negative <- negative[
    order(negative[[nes_col]], negative[[fdr_col]], negative[[term_col]], na.last = TRUE),
    ,
    drop = FALSE
  ]
  rbind(head(positive, top_n), head(negative, top_n))
})
selected_parts <- selected_parts[!vapply(selected_parts, is.null, logical(1))]
if (length(selected_parts) == 0L) {
  stop("No selected terms after top_n filtering")
}
selected <- do.call(rbind, selected_parts)
selected$assign_cluster <- as.character(selected$cluster)

# A pathway that is top-N for several clusters was previously emitted once per
# block, with identical values in every copy. Keep one row per pathway, in the
# block where its enrichment is strongest, so the panel shows each pathway once
# and does not read as a rendering error. This drops repeats only; it never
# substitutes a different pathway.
if (anyDuplicated(as.character(selected[[term_col]]))) {
  keep_idx <- unlist(lapply(
    split(seq_len(nrow(selected)), as.character(selected[[term_col]])),
    function(idx) idx[which.max(abs(selected[[nes_col]][idx]))]
  ), use.names = FALSE)
  dropped <- nrow(selected) - length(keep_idx)
  selected <- selected[sort(keep_idx), , drop = FALSE]
  cat(sprintf("Collapsed %d duplicate pathway row(s) across cluster blocks\n", dropped))
}
row_terms <- as.character(selected[[term_col]])
row_ids <- make.unique(paste(selected$assign_cluster, row_terms, sep = "::"), sep = "::")
selected$row_id <- row_ids

make_row_matrix <- function(df, term_column, value_column, row_terms, row_ids, clusters, fill = NA_real_, fun = mean) {
  mat <- matrix(fill, nrow = length(row_ids), ncol = length(clusters), dimnames = list(row_ids, clusters))
  if (nrow(df) == 0L) {
    return(mat)
  }
  value <- as.numeric(df[[value_column]])
  valid <- is.finite(value) & !is.na(df[[term_column]]) & !is.na(df$cluster)
  if (!any(valid)) {
    return(mat)
  }
  agg <- aggregate(
    value[valid],
    by = list(Term = as.character(df[[term_column]][valid]), cluster = as.character(df$cluster[valid])),
    FUN = fun,
    na.rm = TRUE
  )
  lookup <- setNames(agg$x, paste(as.character(agg$Term), as.character(agg$cluster), sep = "\r"))
  for (row_idx in seq_along(row_ids)) {
    for (cluster in clusters) {
      key <- paste(as.character(row_terms[row_idx]), as.character(cluster), sep = "\r")
      if (key %in% names(lookup) && is.finite(lookup[[key]])) {
        mat[row_idx, cluster] <- lookup[[key]]
      }
    }
  }
  mat
}

nes_mat <- make_row_matrix(nes_src, all_term_col, all_nes_col, row_terms, row_ids, cluster_order, fill = NA_real_, fun = mean)
keep_rows <- rowSums(is.finite(nes_mat)) > 0
selected <- selected[keep_rows, , drop = FALSE]
row_terms <- row_terms[keep_rows]
row_ids <- row_ids[keep_rows]
nes_mat <- nes_mat[keep_rows, , drop = FALSE]
if (nrow(nes_mat) == 0L) {
  stop("Selected terms did not overlap the GSEA NES table")
}
nes_mat[!is.finite(nes_mat)] <- 0

if (!is.na(all_fdr_col)) {
  fdr_mat <- make_row_matrix(nes_src, all_term_col, all_fdr_col, row_terms, row_ids, cluster_order, fill = NA_real_, fun = min)
} else {
  fdr_mat <- matrix(NA_real_, nrow = length(row_ids), ncol = length(cluster_order), dimnames = list(row_ids, cluster_order))
}

p_to_stars <- function(p) {
  ifelse(!is.finite(p), "",
    ifelse(p <= 0.001, "***",
      ifelse(p <= 0.01, "**",
        ifelse(p <= 0.05, "*", "")
      )
    )
  )
}
stars_mat <- matrix(p_to_stars(as.vector(fdr_mat)), nrow = nrow(fdr_mat), ncol = ncol(fdr_mat), dimnames = dimnames(fdr_mat))

if (nrow(shifts) > 0L && all(c("Term", "cluster", "mean_shift_vs_cohort") %in% colnames(shifts))) {
  shifts$cluster <- as.character(shifts$cluster)
  shift_mat <- make_row_matrix(shifts, "Term", "mean_shift_vs_cohort", row_terms, row_ids, cluster_order, fill = NA_real_, fun = mean)
  if ("FDR" %in% colnames(shifts)) {
    shift_fdr_mat <- make_row_matrix(shifts, "Term", "FDR", row_terms, row_ids, cluster_order, fill = NA_real_, fun = min)
  } else {
    shift_fdr_mat <- matrix(NA_real_, nrow = length(row_ids), ncol = length(cluster_order), dimnames = list(row_ids, cluster_order))
  }
} else {
  shift_mat <- matrix(NA_real_, nrow = length(row_ids), ncol = length(cluster_order), dimnames = list(row_ids, cluster_order))
  shift_fdr_mat <- matrix(NA_real_, nrow = length(row_ids), ncol = length(cluster_order), dimnames = list(row_ids, cluster_order))
}
shift_stars_mat <- matrix(
  p_to_stars(as.vector(shift_fdr_mat)),
  nrow = nrow(shift_fdr_mat),
  ncol = ncol(shift_fdr_mat),
  dimnames = dimnames(shift_fdr_mat)
)

infer_db_prefix <- function(term) {
  text <- as.character(term)
  if (startsWith(text, "GOBP_") || startsWith(text, "GO_") || grepl("GO:", text)) return("GO")
  if (startsWith(text, "GOCC_")) return("GO")
  if (startsWith(text, "HALLMARK_")) return("H")
  if (startsWith(text, "REACTOME_") || grepl("R-HSA", text)) return("R")
  if (startsWith(text, "KEGG_")) return("K")
  m <- regmatches(text, regexpr("[A-Za-z]+", text))
  if (length(m) && nchar(m)) substr(m, 1, 1) else "?"
}

title_case <- function(x) {
  words <- strsplit(tolower(x), " +")[[1]]
  paste0(toupper(substr(words, 1, 1)), substring(words, 2), collapse = " ")
}

prettify_pathway_name <- function(term, max_len = 58) {
  text <- as.character(term)
  db_prefix <- infer_db_prefix(text)
  name <- if (grepl("__", text, fixed = TRUE)) sub("^.*?__", "", text) else text
  for (prefix in c("GOBP_", "GOCC_", "GOMF_", "HALLMARK_", "REACTOME_", "KEGG_")) {
    if (startsWith(name, prefix)) name <- substring(name, nchar(prefix) + 1L)
  }
  name <- gsub("R[-_]?HSA[-_:]?[0-9]+", "", name)
  name <- gsub("GO[:_][0-9]+", "", name)
  name <- gsub("KEGG[:_][0-9]+", "", name)
  name <- gsub("^[A-Za-z0-9]+__", "", name)
  name <- gsub("^[0-9]+[_[:space:]-]*", "", name)
  name <- gsub("\\(\\s*\\)", "", name)
  name <- gsub("_", " ", name)
  name <- trimws(gsub("\\s{2,}", " ", name), whitespace = " -:;")
  if (!nchar(name)) name <- text
  label <- paste0(title_case(name), " [", db_prefix, "]")
  if (!is.null(max_len) && nchar(label) > max_len) {
    suffix <- paste0(" [", db_prefix, "]")
    keep <- max(1L, max_len - nchar(suffix) - 3L)
    label <- paste0(trimws(substr(title_case(name), 1L, keep)), "...", suffix)
  }
  label
}

column_labels <- style_cluster_column_labels(cluster_order)

nes_limit <- style_color_limit(nes_mat)
nes_col_fun <- style_diverging_col_fun(nes_mat)
shift_limit <- style_color_limit(shift_mat)
shift_col_fun <- style_diverging_col_fun(shift_mat)

row_group_lookup <- style_cluster_row_labels(cluster_order)
row_group_labels <- row_group_lookup[selected$assign_cluster]
row_group_labels[is.na(row_group_labels)] <- paste0("C", selected$assign_cluster)
row_group_levels <- style_cluster_row_labels(cluster_order)
row_split <- factor(row_group_labels, levels = row_group_levels)
row_labels <- vapply(row_terms, prettify_pathway_name, character(1), max_len = 44)
names(row_labels) <- row_ids

cluster_bottom_annotation <- function(show_name = TRUE) {
  style_cluster_bottom_annotation(cluster_order, show_name = show_name)
}

# The paired main-figure asset must fit the double-column width while retaining
# both independent legends. Keep the shared row pitch, but compact only this
# composite's cells and pathway-label allowance.
combined_cell_w <- style_mm("cell_w") * 0.78
combined_row_label_max <- style_mm("row_label_max") * 0.85

nes_ht <- do.call(Heatmap, c(list(nes_mat), modifyList(
  style_heatmap_args(
    nes_mat,
    legend_title = "NES",
    row_split = row_split,
    column_title = "GSEA prerank NES"
  ),
  list(
    col = nes_col_fun,
    width = unit(combined_cell_w * ncol(nes_mat), "mm"),
    row_names_max_width = unit(combined_row_label_max, "mm"),
    row_title_side = "left",
    row_labels = row_labels,
    column_labels = column_labels,
    bottom_annotation = cluster_bottom_annotation(TRUE)
  )
), list(
  cell_fun = function(j, i, x, y, width, height, fill) {
    val <- nes_mat[i, j]
    if (!is.finite(val)) return()
    txt_col <- style_cell_text_col(val, nes_limit)
    val_str <- sub("\\.?0+$", "", sprintf("%.2f", val))
    if (val_str == "-0") val_str <- "0"
    grid.text(val_str, x, y, gp = style_gp("annot", col = txt_col))
    star <- stars_mat[i, j]
    style_cell_star(star, x, y, width, height, txt_col)
  }
)))

shift_ht <- do.call(Heatmap, c(list(shift_mat), modifyList(
  style_heatmap_args(
    shift_mat,
    legend_title = "ssGSEA shift",
    row_split = row_split,
    column_title = "ssGSEA mean shift"
  ),
  list(
    col = shift_col_fun,
    width = unit(combined_cell_w * ncol(shift_mat), "mm"),
    show_row_names = FALSE,
    row_title = NULL,
    row_title_gp = gpar(fontsize = 0),
    column_labels = column_labels,
    bottom_annotation = cluster_bottom_annotation(FALSE),
    cell_fun = function(j, i, x, y, width, height, fill) {
      val <- shift_mat[i, j]
      if (!is.finite(val)) return()
      txt_col <- style_cell_text_col(val, shift_limit)
      val_str <- sub("\\.?0+$", "", sprintf("%.2f", val))
      if (val_str == "-0") val_str <- "0"
      grid.text(val_str, x, y, gp = style_gp("annot", col = txt_col))
      star <- shift_stars_mat[i, j]
      style_cell_star(star, x, y, width, height, txt_col)
    }
  )
)))

combined_ht <- nes_ht + shift_ht
written <- style_heatmap_save(
  output_file,
  combined_ht,
  heatmap_legend_side = "right",
  annotation_legend_side = "right",
  merge_legend = FALSE,
  padding = unit(rep(style_mm("pad"), 4), "mm")
)

cat("Wrote:", paste(written, collapse = " "), "\n")
