#!/usr/bin/env Rscript
# Top-N positive-NES and negative-NES pathway heatmap per cluster.
#
# The renderer uses the non-redundant significant GSEA table for selection and
# the all-terms table for the complete NES/FDR matrix. Repeated pathway names
# are intentionally retained when a pathway is selected for more than one
# cluster.
#
# Cells with NES of exactly +/-1 and nominal p = 1 are drawn plain grey, without a value:
# gseapy returns that value when no permutation has the observed sign (heavily
# skewed rankings), so it is not an enrichment estimate.
#
# Usage:
#   Rscript r/gsea_top_pathways_heatmap.R <gsea_results_dir> <output_file> [top_n]

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
if (length(args) < 2L) {
  stop("Usage: Rscript r/gsea_top_pathways_heatmap.R <gsea_results_dir> <output_file> [top_n]")
}

results_dir <- args[1]
output_file <- args[2]
top_n <- if (length(args) >= 3L) as.integer(args[3]) else 3L
if (is.na(top_n) || top_n < 1L) top_n <- 3L

read_tsv <- function(path) {
  if (!file.exists(path)) return(data.frame())
  read.delim(path, check.names = FALSE, stringsAsFactors = FALSE)
}

significant <- read_tsv(file.path(results_dir, "gsea_nonredundant_significant_terms.tsv"))
all_terms <- read_tsv(file.path(results_dir, "gsea_prerank_all_terms.tsv"))
if (nrow(significant) == 0L) {
  stop("No non-redundant significant GSEA terms found: ", results_dir)
}
if (nrow(all_terms) == 0L) {
  stop("No all-terms GSEA table found: ", results_dir)
}

term_col <- if ("Term" %in% colnames(significant)) "Term" else "Name"
fdr_col <- if ("FDR q-val" %in% colnames(significant)) "FDR q-val" else "FDR"
required <- c(term_col, fdr_col, "NES", "cluster")
if (!all(required %in% colnames(significant))) {
  stop("Significant GSEA table is missing required columns")
}

all_term_col <- if ("Term" %in% colnames(all_terms)) "Term" else "Name"
all_fdr_col <- if ("FDR q-val" %in% colnames(all_terms)) "FDR q-val" else "FDR"
all_required <- c(all_term_col, all_fdr_col, "NES", "cluster")
if (!all(all_required %in% colnames(all_terms))) {
  stop("All-terms GSEA table is missing required columns")
}

significant$cluster <- as.character(significant$cluster)
significant$NES <- as.numeric(significant$NES)
significant$FDR_value <- as.numeric(significant[[fdr_col]])
significant$term_value <- as.character(significant[[term_col]])
significant <- significant[
  significant$cluster %in% as.character(seq_len(6)) &
    is.finite(significant$NES) &
    is.finite(significant$FDR_value) &
    significant$FDR_value <= 0.05,
  ,
  drop = FALSE
]
if (nrow(significant) == 0L) {
  stop("No significant GSEA terms remain after validation")
}

all_terms$cluster <- as.character(all_terms$cluster)
all_terms$NES_value <- as.numeric(all_terms$NES)
all_terms$FDR_value <- as.numeric(all_terms[[all_fdr_col]])
all_terms$term_value <- as.character(all_terms[[all_term_col]])
all_terms$NOM_value <- if ("NOM p-val" %in% colnames(all_terms)) {
  as.numeric(all_terms[["NOM p-val"]])
} else {
  NA_real_
}

# ---- select top positive and negative pathways per cluster -----------------
cluster_order <- style_cluster_order()
selected_list <- list()
for (cluster_id in cluster_order) {
  sub <- significant[significant$cluster == cluster_id, , drop = FALSE]
  if (nrow(sub) == 0L) next
  for (direction in c("up", "down")) {
    if (direction == "up") {
      direction_sub <- sub[sub$NES > 0, , drop = FALSE]
      direction_sub <- direction_sub[
        order(-direction_sub$NES, direction_sub$FDR_value, direction_sub$term_value, na.last = TRUE),
        ,
        drop = FALSE
      ]
    } else {
      direction_sub <- sub[sub$NES < 0, , drop = FALSE]
      direction_sub <- direction_sub[
        order(direction_sub$NES, direction_sub$FDR_value, direction_sub$term_value, na.last = TRUE),
        ,
        drop = FALSE
      ]
    }
    direction_sub <- head(direction_sub, top_n)
    if (nrow(direction_sub) == 0L) next
    direction_sub$assign_cluster <- cluster_id
    direction_sub$assign_direction <- direction
    selected_list[[paste(cluster_id, direction, sep = "::")]] <- direction_sub
  }
}
if (length(selected_list) == 0L) {
  stop("No positive or negative significant pathways were selected")
}
selected <- do.call(rbind, selected_list)
selected$row_id <- make.unique(
  paste(selected$assign_cluster, selected$assign_direction, selected$term_value, sep = "::"),
  sep = "::"
)

# ---- build NES/FDR matrices -------------------------------------------------
make_matrix <- function(source, value_col, row_terms, row_ids, clusters, fun = mean) {
  mat <- matrix(
    NA_real_,
    nrow = length(row_ids),
    ncol = length(clusters),
    dimnames = list(row_ids, clusters)
  )
  valid <- is.finite(source[[value_col]]) & !is.na(source$term_value) & !is.na(source$cluster)
  if (!any(valid)) return(mat)
  agg <- aggregate(
    source[[value_col]][valid],
    by = list(term = source$term_value[valid], cluster = source$cluster[valid]),
    FUN = fun,
    na.rm = TRUE
  )
  lookup <- setNames(agg$x, paste(agg$term, agg$cluster, sep = "\r"))
  for (row_idx in seq_along(row_ids)) {
    for (cluster_id in clusters) {
      key <- paste(row_terms[row_idx], cluster_id, sep = "\r")
      if (key %in% names(lookup) && is.finite(lookup[[key]])) {
        mat[row_idx, cluster_id] <- lookup[[key]]
      }
    }
  }
  mat
}

row_terms <- selected$term_value
row_ids <- selected$row_id
nes_mat <- make_matrix(all_terms, "NES_value", row_terms, row_ids, cluster_order, mean)
fdr_mat <- make_matrix(all_terms, "FDR_value", row_terms, row_ids, cluster_order, min)
keep_rows <- rowSums(is.finite(nes_mat)) > 0
if (!all(keep_rows)) {
  stop("Selected pathways are missing from the all-terms GSEA table")
}
nes_mat[!is.finite(nes_mat)] <- 0
nom_mat <- make_matrix(all_terms, "NOM_value", row_terms, row_ids, cluster_order, max)
not_estimable <- abs(abs(nes_mat) - 1) < 1e-9 & is.finite(nom_mat) & nom_mat >= 1
cat("Not-estimable NES cells (|NES| = 1, NOM p = 1):", sum(not_estimable), "\n")

p_to_stars <- function(p) {
  ifelse(!is.finite(p), "",
    ifelse(p <= 0.001, "***",
      ifelse(p <= 0.01, "**",
        ifelse(p <= 0.05, "*", "")
      )
    )
  )
}
stars_mat <- matrix(
  p_to_stars(as.vector(fdr_mat)),
  nrow = nrow(fdr_mat),
  ncol = ncol(fdr_mat),
  dimnames = dimnames(fdr_mat)
)

# ---- pathway labels and shared house style ---------------------------------
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

prettify_pathway_name <- function(term, max_len = 40) {
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
  suffix <- paste0(" [", db_prefix, "]")
  if (nchar(label) > max_len) {
    keep <- max(1L, max_len - nchar(suffix) - 3L)
    label <- paste0(trimws(substr(title_case(name), 1L, keep)), "...", suffix)
  }
  label
}

row_group_labels <- style_cluster_row_labels(selected$assign_cluster)
row_split <- factor(row_group_labels, levels = style_cluster_row_labels(cluster_order))
direction_labels <- ifelse(selected$assign_direction == "up", "↑ ", "↓ ")
row_labels <- paste0(
  direction_labels,
  vapply(row_terms, prettify_pathway_name, character(1), max_len = 40)
)
names(row_labels) <- row_ids

nes_limit <- style_color_limit(nes_mat)
nes_col_fun <- style_diverging_col_fun(nes_mat)
column_labels <- style_cluster_column_labels(cluster_order)

ht <- do.call(Heatmap, c(list(nes_mat), modifyList(
  style_heatmap_args(nes_mat, legend_title = "NES", row_split = row_split),
  list(
    col = nes_col_fun,
    row_title_side = "left",
    row_labels = row_labels,
    column_labels = column_labels,
    bottom_annotation = style_cluster_bottom_annotation(cluster_order, show_name = TRUE)
  )
), list(
  cell_fun = function(j, i, x, y, width, height, fill) {
    val <- nes_mat[i, j]
    if (!is.finite(val)) return()
    if (not_estimable[i, j]) {
      grid.rect(x, y, width, height, gp = gpar(
        fill = STYLE$palette$missing, col = "white", lwd = style_mm("cell_line_pt")
      ))
      return()
    }
    txt_col <- style_cell_text_col(val, nes_limit)
    val_str <- sub("\\.?0+$", "", sprintf("%.2f", val))
    if (val_str == "-0") val_str <- "0"
    grid.text(val_str, x, y, gp = style_gp("annot", col = txt_col))
    style_cell_star(stars_mat[i, j], x, y, width, height, txt_col)
  }
)))

written <- style_heatmap_save(
  output_file,
  ht,
  heatmap_legend_side = "right",
  annotation_legend_side = "right",
  merge_legend = TRUE,
  padding = unit(rep(style_mm("pad"), 4), "mm")
)

cat("Wrote:", paste(written, collapse = " "), "\n")

