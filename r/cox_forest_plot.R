#!/usr/bin/env Rscript
# Forest plot for Cox hazard ratios. Handles three model shapes:
#   * cluster_vs_rest_adjusted: rows are per-cluster HRs from cluster-vs-rest
#     adjusted Cox models. One row per cluster.
#   * global_cluster_deviation: rows from a single mutually-adjusted Cox model
#     using sum-to-zero (deviation) coding. Every cluster gets a real HR and CI
#     against the cohort grand mean; there is no reference class.
#   * metadata_univariable:     rows from per-covariate univariable Cox.
#     One row per covariate term.
#   * clinical_covariates_adjusted: clinical covariates from a single global
#     cluster-adjusted Cox model.
#   * combined_os: six one-vs-rest OS cluster rows plus six global-model
#     clinical rows, sorted by HR for a compact supporting presentation view.
#
# Args:
#   --tsv <input.tsv>
#   --model <cluster_vs_rest_adjusted|global_cluster_deviation|metadata_univariable|clinical_covariates_adjusted|combined_os>
#   --endpoint OS|PFS|BOTH
#   --row-col <cluster|covariate>
#   --output <path.png>
#   [--title "..."]

suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repo_root <- if (!is.na(script_file) && nzchar(script_file)) {
  normalizePath(file.path(dirname(normalizePath(script_file, mustWork = FALSE)), ".."), mustWork = FALSE)
} else {
  getwd()
}
source(file.path(repo_root, "r", "plot_style.R"))

args <- commandArgs(trailingOnly = TRUE)
opts <- list(title = "", row_col = "cluster")
i <- 1L
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
  if (key == "--tsv") opts$tsv <- val
  else if (key == "--model") opts$model <- val
  else if (key == "--endpoint") opts$endpoint <- val
  else if (key == "--row-col") opts$row_col <- val
  else if (key == "--output") opts$output <- val
  else if (key == "--title") opts$title <- val
  else stop(sprintf("Unknown argument: %s", key))
  i <- i + 2L
}
for (req in c("tsv", "model", "endpoint", "output")) {
  if (is.null(opts[[req]])) stop(sprintf("Missing --%s", req))
}
if (!opts$endpoint %in% c("OS", "PFS", "BOTH")) {
  stop("--endpoint must be OS, PFS, or BOTH")
}

df <- read.delim(opts$tsv, check.names = FALSE, stringsAsFactors = FALSE)

# Filter to model + endpoint
if (opts$model != "metadata_univariable" && "model" %in% colnames(df)) {
  df <- df[df$model == opts$model, , drop = FALSE]
}
endpoints <- if (opts$endpoint == "BOTH") c("OS", "PFS") else opts$endpoint
df <- df[df$endpoint %in% endpoints, , drop = FALSE]

# Restrict to rows with usable HR estimates
hr_cols <- c("hazard_ratio", "ci_lower", "ci_upper")
for (col in hr_cols) {
  if (!col %in% colnames(df)) stop(sprintf("Missing required column: %s", col))
}
df <- df[is.finite(df$hazard_ratio) & is.finite(df$ci_lower) & is.finite(df$ci_upper), , drop = FALSE]
if (!"p" %in% colnames(df)) df$p <- NA_real_

write_blank <- function(message) {
  # Through style_save so the degenerate-input path still writes a PDF sibling
  # and passes the figure-style audit.
  style_save(opts$output, 5.5, 3.0, function() {
    grid::grid.newpage(); grid::grid.text(message, gp = style_gp("title"))
  })
  cat("Wrote:", opts$output, "\n")
  quit(save = "no", status = 0)
}

if (nrow(df) == 0L) write_blank("No Cox estimates available")

# ---- Build row identity, ordering, and colours ----
cluster_palette <- style_cluster_colors(as.character(seq_len(6)))
cluster_labels <- style_cluster_labels(as.character(seq_len(6)))

metadata_term_labels <- c(
  age = "Age (per SD)",
  tumor_content = "Tumor content (per SD)",
  sex_male = "Sex: male vs female",
  uicc_major_II = "UICC II vs I",
  uicc_major_III = "UICC III vs I",
  uicc_major_IV = "UICC IV vs I"
)

pretty_metadata_term <- function(term) {
  out <- unname(metadata_term_labels[as.character(term)])
  missing <- is.na(out)
  if (any(missing)) {
    fallback <- gsub("_", " ", as.character(term)[missing])
    out[missing] <- fallback
  }
  out
}

if (opts$model %in% c("cluster_vs_rest_adjusted", "global_cluster_deviation")) {
  # The deviation model emits cluster and covariate terms in one table; keep
  # only the cluster rows. One-vs-rest tables have no term_kind column values.
  if (opts$model == "global_cluster_deviation" && "term_kind" %in% colnames(df)) {
    df <- df[!is.na(df$term_kind) & df$term_kind == "cluster", , drop = FALSE]
  }
  if (nrow(df) == 0L) write_blank("No cluster Cox estimates")
  df$cluster <- as.character(df$cluster)
  df$row_label <- cluster_labels[df$cluster]
  df$row_label[is.na(df$row_label)] <- paste0("C", df$cluster)
  df$row_colour <- cluster_palette[df$cluster]
  df$row_colour[is.na(df$row_colour)] <- "#4C4C4C"
  row_order <- style_cluster_order(unique(df$cluster))
  row_label_levels <- unname(cluster_labels[row_order])
  df$row_label <- factor(df$row_label, levels = rev(row_label_levels))
  use_cluster_colours <- TRUE
} else if (opts$model == "combined_os") {
  if (!"term_kind" %in% colnames(df)) stop("combined_os requires term_kind")
  df$cluster <- as.character(df$cluster)
  is_cluster <- !is.na(df$term_kind) & df$term_kind == "cluster"
  df$row_label <- as.character(df$row_label)
  missing_label <- is.na(df$row_label) | !nzchar(df$row_label)
  df$row_label[missing_label & is_cluster] <- cluster_labels[df$cluster[missing_label & is_cluster]]
  df$row_label[missing_label & !is_cluster] <- pretty_metadata_term(df$term[missing_label & !is_cluster])
  df$row_colour <- rep("#4C4C4C", nrow(df))
  df$row_colour[is_cluster] <- cluster_palette[df$cluster[is_cluster]]
  df$row_colour[is.na(df$row_colour) | !nzchar(df$row_colour)] <- "#4C4C4C"
  ordered_labels <- as.character(df$row_label[order(df$hazard_ratio)])
  df$row_label <- factor(df$row_label, levels = rev(unique(ordered_labels)))
  use_cluster_colours <- TRUE
} else if (opts$model == "clinical_covariates_adjusted") {
  df$row_label <- if ("plot_label" %in% colnames(df)) as.character(df$plot_label) else pretty_metadata_term(df$term)
  missing_label <- is.na(df$row_label) | !nzchar(df$row_label)
  df$row_label[missing_label] <- pretty_metadata_term(df$term[missing_label])
  df$row_colour <- "#4C4C4C"
  term_order <- c("age", "sex_male", "uicc_major_II", "uicc_major_III", "uicc_major_IV", "tumor_content")
  ordered_labels <- df$row_label[match(term_order, df$term, nomatch = 0L)]
  ordered_labels <- ordered_labels[!is.na(ordered_labels) & nzchar(ordered_labels)]
  df$row_label <- factor(df$row_label, levels = rev(unique(ordered_labels)))
  use_cluster_colours <- FALSE
} else {
  # metadata_univariable
  if (!"covariate" %in% colnames(df)) {
    if (opts$row_col %in% colnames(df)) {
      df$covariate <- as.character(df[[opts$row_col]])
    } else {
      df$covariate <- as.character(df$term)
    }
  }
  df$row_label <- df$term
  df$row_colour <- "#4C4C4C"
  ordered_terms <- unique(df$term[order(df$endpoint, df$hazard_ratio)])
  df$row_label <- factor(df$row_label, levels = rev(ordered_terms))
  use_cluster_colours <- FALSE
}

# ---- Right-side annotation: HR (CI), with the declared model-family FDR ----
df$annot <- sprintf("%.2f (%.2f-%.2f)", df$hazard_ratio, df$ci_lower, df$ci_upper)
df$annot[!is.finite(df$ci_lower) | !is.finite(df$ci_upper)] <- "ref"
star_for <- function(p) ifelse(!is.finite(p), "", ifelse(p <= 0.001, "***", ifelse(p <= 0.01, "**", ifelse(p <= 0.05, "*", ""))))
df$annot <- paste0(df$annot, ifelse(nzchar(star_for(df$p)), paste0(" ", star_for(df$p)), ""))
if (opts$model %in% c("cluster_vs_rest_adjusted", "global_cluster_deviation", "clinical_covariates_adjusted") &&
    "q_within_model" %in% colnames(df)) {
  df$annot <- ifelse(
    is.finite(df$q_within_model),
    sprintf("%.2f (%.2f-%.2f); q=%.2g", df$hazard_ratio, df$ci_lower, df$ci_upper, df$q_within_model),
    sprintf("%.2f (%.2f-%.2f)", df$hazard_ratio, df$ci_lower, df$ci_upper)
  )
}

# ---- HR axis range ----
hr_min <- min(c(df$hazard_ratio, df$ci_lower), na.rm = TRUE)
hr_max <- max(c(df$hazard_ratio, df$ci_upper), na.rm = TRUE)
hr_min <- max(min(hr_min, 0.3), 0.1)
hr_max <- min(max(hr_max, 3.0), 30)
x_limits <- if (opts$model == "combined_os") c(hr_min, max(8, hr_max * 1.05)) else NULL

# ---- Build plot ----
forest_tick_pt <- style_pt("tick") * 1.5
forest_label_pt <- style_pt("label") * 1.5

colour_map <- if (use_cluster_colours) {
  setNames(df$row_colour[!duplicated(df$row_label)], df$row_label[!duplicated(df$row_label)])
} else {
  NULL
}

p <- ggplot(df, aes(x = hazard_ratio, y = row_label)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey60", linewidth = 0.5) +
  geom_errorbarh(aes(xmin = ci_lower, xmax = ci_upper, colour = row_label),
                 height = 0.34, linewidth = 0.9, na.rm = TRUE) +
  geom_point(aes(colour = row_label), size = 3.2, na.rm = TRUE) +
  scale_x_log10(limits = x_limits, breaks = c(0.3, 0.5, 1, 2, 4, 8),
                labels = function(x) formatC(x, format = "g")) +
  labs(x = if (opts$model == "global_cluster_deviation") "Hazard ratio vs cohort grand mean (95% CI)" else "Hazard ratio (95% CI)",
       y = NULL,
       title = if (nzchar(opts$title)) opts$title else NULL) +
  style_ggplot_theme() +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_line(colour = "grey92", linewidth = 0.3)
  )

if (!is.null(colour_map)) {
  p <- p + scale_colour_manual(values = colour_map, guide = "none")
} else {
  p <- p + scale_colour_manual(values = setNames(rep("#4C4C4C", length(unique(df$row_label))),
                                                 unique(df$row_label)), guide = "none")
}

# HR (CI) text on the right; the combined supporting view intentionally keeps
# the plotting region clean and matches the recovered presentation asset.
if (opts$model == "combined_os") {
  p <- p +
      coord_cartesian(xlim = x_limits, clip = "off") +
      theme(
      plot.margin = margin(8, 8, 10, 8),
      axis.text.x = element_text(size = forest_tick_pt),
      axis.text.y = element_text(size = forest_tick_pt),
      axis.title.x = element_text(size = forest_label_pt)
    )
} else {
  hr_text_x <- hr_max * 1.04
  p <- p +
    geom_text(aes(x = hr_text_x, label = annot),
              hjust = 0, size = 2.4, colour = "grey25") +
    coord_cartesian(xlim = c(hr_min, hr_max * 2.8), clip = "off") +
    theme(plot.margin = margin(6, 6, 6, 6))
}

if (opts$endpoint == "BOTH") {
  p <- p + facet_wrap(~ endpoint, ncol = 2)
}

# ---- Save ----
n_rows <- length(unique(df$row_label))
plot_height <- if (opts$model == "combined_os") 5.0 else max(2.4, 0.32 * n_rows + 1.2)
plot_width <- if (opts$endpoint == "BOTH") {
  9.0
} else if (opts$model == "clinical_covariates_adjusted") {
  7.2
} else if (opts$model == "combined_os") {
  5.5
} else {
  6.2
}
written <- style_save(opts$output, plot_width, plot_height, function() print(p))
cat("Wrote:", paste(written, collapse = " "), "\n")
