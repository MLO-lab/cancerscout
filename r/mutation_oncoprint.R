#!/usr/bin/env Rscript
# Mutation oncoprint per Leiden consensus cluster — ComplexHeatmap oncoPrint style.
# Samples sorted by total mutation burden within each cluster.
# Columns ending in "_cooccurrence" are treated as mutation tuples and displayed
# in a separate row group below a visual gap (row_split).
#
# Usage: Rscript r/mutation_oncoprint.R <data_csv> <output_png>

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

args       <- commandArgs(trailingOnly = TRUE)
data_csv   <- if (length(args) >= 1) args[1] else file.path(repo_root, "results", "mutation_enrichment", "oncoprint_mutation_single.csv")
output_png <- if (length(args) >= 2) args[2] else "figures/cluster_interpretation/mutation_oncoprint_complex.png"

cluster_colors <- style_cluster_colors(style_cluster_order())
cluster_labels <- style_cluster_labels(style_cluster_order())
# The annotation legend uses short C1-C6 keys: at full length the C1 entry
# ("C1 Stromal ECM-rich / Mesenchymal") sets the legend width and pushes every
# other legend along with it. Full names stay on the column-split titles and in
# the caption.
cluster_keys <- style_cluster_row_labels(style_cluster_order())

df <- read.csv(data_csv, row.names=1, check.names=FALSE)
df$leiden_cons <- as.character(df$leiden_cons)

# All mutation columns = everything except the clinical annotation tracks.
meta_cols  <- c("leiden_cons", "MSI consensus", "CPTAC", "Age", "Sex", "UICC", "OS", "PFS", "Mutation burden")
mut_cols   <- setdiff(colnames(df), meta_cols)

# Split into single genes and tuple co-occurrences.
single_cols <- mut_cols[!grepl("_cooccurrence$", mut_cols)]
tuple_cols  <- mut_cols[grepl("_cooccurrence$",  mut_cols)]

# Pretty labels for tuple rows: TP53_APC_cooccurrence → TP53+APC
pretty_label <- function(x) sub("\\+cooccurrence$", "", gsub("_", "+", x))

all_mut_cols <- c(single_cols, tuple_cols)
row_labels   <- c(single_cols, sapply(tuple_cols, pretty_label))

# Sort: cluster first, then descending total mutation burden within cluster.
df$mut_sum <- suppressWarnings(as.numeric(df[["Mutation burden"]]))
df <- df[order(df$leiden_cons, -df$mut_sum, na.last=TRUE), ]

# ComplexHeatmap oncoPrint matrix
mat    <- t(df[, all_mut_cols, drop=FALSE])
mat_ch <- ifelse(is.na(mat), "MISSING", ifelse(mat == 1, "MUT", ""))
rownames(mat_ch) <- row_labels

col_split <- factor(df$leiden_cons, levels=sort(unique(df$leiden_cons)))

# Row split: "Single" vs "Co-occurrence" (only if there are tuples)
if (length(tuple_cols) > 0) {
  row_split <- factor(
    c(rep("Single mutations", length(single_cols)),
      rep("Co-occurrences",   length(tuple_cols))),
    levels = c("Single mutations", "Co-occurrences")
  )
} else {
  row_split <- NULL
}

cluster_annotation <- factor(
  cluster_keys[df$leiden_cons],
  levels = unname(cluster_keys[names(cluster_keys)])
)
cluster_annotation_colors <- setNames(
  unname(cluster_colors[names(cluster_keys)]),
  unname(cluster_keys[names(cluster_keys)])
)
age <- suppressWarnings(as.numeric(df$Age))
age_limits <- range(age, na.rm = TRUE)
if (!all(is.finite(age_limits)) || diff(age_limits) == 0) age_limits <- c(0, 1)

ha_top <- HeatmapAnnotation(
  `Mut. burden` = anno_barplot(
    df$mut_sum,
    gp            = gpar(fill="#555555", col=NA),
    height        = unit(2, "cm"),
    axis_param    = list(gp=style_gp("tick"))
  ),
  Cluster = cluster_annotation,
  MSI     = df[["MSI consensus"]],
  CPTAC   = df$CPTAC,
  UICC    = df$UICC,
  Age     = age,
  Sex     = df$Sex,
  OS      = df$OS,
  # "PFS Status" codes documented progression only (deaths without progression
  # are 0), so it is neither PFS nor the manuscript's EFS; label what it shows.
  Progression = df$PFS,
  col = list(
    Cluster = cluster_annotation_colors,
    MSI     = c("MSI"=unname(cluster_colors["4"]),"MSS"=unname(cluster_colors["5"])),
    CPTAC   = c(
      "CIN"=unname(cluster_colors["3"]),
      "MSI"=unname(cluster_colors["4"]),
      "Mesenchymal"=unname(cluster_colors["1"]),
      "Other"="#BDBDBD"
    ),
    UICC = c("I"="#D9EAF7", "II"="#A6CEE3", "III"="#4F81BD", "IV"="#17365D"),
    Age = colorRamp2(age_limits, c("#D9EAF7", "#B2182B")),
    Sex = c("female"="#CCCCCC", "male"="#1A1A1A"),
    OS = c("censored"="white", "event"="black"),
    Progression = c("censored"="white", "event"="black")
  ),
  na_col               = "#BDBDBD",
  annotation_name_side = "left",
  annotation_name_gp   = style_gp("label"),
  simple_anno_size     = unit(3.5,"mm"),
  gap                  = unit(0.8,"mm"),
  show_legend          = c(Cluster=TRUE, MSI=TRUE)
)

alter_fun <- list(
  background = alter_graphic("rect", fill="#EFEFEF", col=NA),
  MUT        = alter_graphic("rect", fill="#2C2C2C", col=NA),
  MISSING    = alter_graphic("rect", fill="#BDBDBD", col=NA)
)

fig_height <- if (length(tuple_cols) > 0) 9.0 else 7.6

op <- oncoPrint(
  mat_ch,
  alter_fun         = alter_fun,
  col               = c(MUT="#2C2C2C", MISSING="#BDBDBD"),
  top_annotation    = ha_top,
  column_split      = col_split,
  # 45 degrees keeps the C5 and C6 titles apart (the C6 slice is 17 tumours
  # wide); wrapping the long C1 name stops it setting the height of the band.
  column_title      = sub(" / ", " /\n", cluster_labels[levels(col_split)], fixed=TRUE),
  column_title_gp   = style_gp("title", face="bold"),
  column_title_rot  = 45,
  row_names_gp      = style_gp("tick", face="italic"),
  row_split         = row_split,
  row_title_gp      = style_gp("label", face="bold"),
  row_gap           = unit(3, "mm"),
  show_column_names = FALSE,
  show_pct          = FALSE,
  right_annotation  = rowAnnotation(
    rbar = anno_oncoprint_barplot(
      type       = "MUT",
      # On top, where the space beside the annotation tracks is free; at the
      # bottom the axis runs into the merged legend row.
      axis_param = list(side="top", gp=style_gp("tick")),
      width      = unit(2.5,"cm")
    )
  ),
  row_order         = seq_len(nrow(mat_ch)),
  column_order      = seq_len(ncol(mat_ch)),
  heatmap_legend_param = list(
    title="Call", at=c("MUT", "MISSING"), labels=c("Mutation", "Missing"),
    title_gp=style_gp("label", face="bold"), labels_gp=style_gp("tick")
  )
)

# Constrained to a double column: at 15 in the 7-10pt text would land at
# 3.4-4.8pt once the journal scales the page down.
written <- style_save(output_png, STYLE$widths$double / 25.4, fig_height, function() {
  draw(op, merge_legend=TRUE,
       heatmap_legend_side="bottom",
       annotation_legend_side="bottom")
})
cat("Saved:", paste(written, collapse=" "), "\n")
