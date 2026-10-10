#!/usr/bin/env Rscript
# Kaplan-Meier survival curves per cluster, rendered from pre-computed step
# coordinates. Curves are coloured by the canonical cluster palette. An
# optional risk table is rendered below the curves.
#
# Inputs:
#   --points <km_curve_points.tsv>   columns: endpoint, cluster, time, survival_prob[, n_at_risk]
#   --risk <km_risk_table.tsv>       (optional) columns: endpoint, cluster, time, n_at_risk
#   --endpoint OS|PFS
#   --logrank-p <float>              (optional) global multivariate log-rank p
#   --output <path.png>

suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
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
opts <- list(risk = NA_character_, logrank_p = NA_real_)
i <- 1L
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1L <= length(args)) args[i + 1L] else NA_character_
  if (key == "--points") opts$points <- val
  else if (key == "--risk") opts$risk <- val
  else if (key == "--endpoint") opts$endpoint <- val
  else if (key == "--logrank-p") opts$logrank_p <- suppressWarnings(as.numeric(val))
  else if (key == "--output") opts$output <- val
  else stop(sprintf("Unknown argument: %s", key))
  i <- i + 2L
}
for (req in c("points", "endpoint", "output")) {
  if (is.null(opts[[req]])) stop(sprintf("Missing --%s", req))
}

points <- read.delim(opts$points, check.names = FALSE, stringsAsFactors = FALSE)
points <- points[points$endpoint == opts$endpoint & is.finite(points$time) & is.finite(points$survival_prob), , drop = FALSE]
points$cluster <- as.character(points$cluster)
present_clusters <- style_cluster_order(unique(points$cluster))
if (length(present_clusters) < 2L) {
  # Through style_save so the degenerate-input path still writes a PDF sibling
  # and passes the figure-style audit.
  style_save(opts$output, 6.2, 5.0, function() {
    grid.text(sprintf("Insufficient %s data", opts$endpoint), gp = style_gp("title"))
  })
  cat("Wrote:", opts$output, "\n")
  quit(save = "no", status = 0)
}

cluster_colors <- style_cluster_colors(present_clusters)
cluster_labels <- style_cluster_labels(present_clusters)
points$cluster <- factor(points$cluster, levels = present_clusters)

# Exact n + events per cluster are repeated on each curve row by the Python
# stage. The fallbacks keep older saved point tables renderable.
n_at_zero <- vapply(present_clusters, function(cl) {
  sub <- points[points$cluster == cl, ]
  if (nrow(sub) == 0L) return(0L)
  if ("n_total" %in% colnames(sub)) return(as.integer(sub$n_total[1L]))
  as.integer(max(sub$n_at_risk[sub$time == min(sub$time)], na.rm = TRUE))
}, integer(1))
event_counts <- vapply(present_clusters, function(cl) {
  sub <- points[points$cluster == cl, ]
  if (nrow(sub) == 0L) return(0L)
  if ("events_total" %in% colnames(sub)) return(as.integer(sub$events_total[1L]))
  sub <- sub[order(sub$time), ]
  sum(diff(c(1.0, sub$survival_prob)) < -1e-9)
}, integer(1))

legend_labels <- sprintf(
  "C%s (n=%d, e=%d)",
  present_clusters,
  n_at_zero,
  event_counts
)
names(legend_labels) <- present_clusters

# ---- Main-panel sizing ---------------------------------------------------
# This pair is placed side by side, so enlarge the visible elements rather
# than enlarging the export canvas. The multipliers keep the house-style
# hierarchy while making this dense, manuscript-facing panel readable after
# reduction.
km_tick_pt <- style_pt("tick") * 1.5
km_label_pt <- style_pt("label") * 1.5

# ---- KM curves ----
# Dynamic y-axis floor: crop just below the minimum survival probability in
# the visible window so curves fill the panel instead of sitting near y=1.
visible_points <- points[points$time <= 60, , drop = FALSE]
min_surv <- if (nrow(visible_points) == 0L) 0 else min(visible_points$survival_prob, na.rm = TRUE)
y_floor <- floor(max(min_surv - 0.05, 0) * 10) / 10
y_breaks <- seq(y_floor, 1, by = 0.1)
if (length(y_breaks) < 4L) y_breaks <- seq(y_floor, 1, by = 0.05)

km_plot <- ggplot(points, aes(x = time, y = survival_prob, colour = cluster, group = cluster)) +
  geom_step(direction = "hv", linewidth = 0.9) +
  scale_colour_manual(values = cluster_colors[present_clusters], labels = legend_labels, name = NULL) +
  scale_x_continuous(limits = c(-1, 61), breaks = seq(0, 60, by = 12), expand = c(0, 0)) +
  scale_y_continuous(limits = c(y_floor, 1.0), breaks = y_breaks, expand = c(0, 0)) +
  labs(x = "Months", y = sprintf("%s probability", opts$endpoint)) +
  style_ggplot_theme() +
  theme(
    legend.position = "bottom",
    legend.background = element_blank(),
    legend.key.height = unit(5, "mm"),
    legend.key.width = unit(8, "mm"),
    legend.margin = margin(t = 2, b = 0),
    legend.spacing.y = unit(1, "mm"),
    axis.text.x = element_text(size = km_tick_pt),
    axis.text.y = element_text(size = km_tick_pt),
    axis.title = element_text(size = km_label_pt),
    legend.text = element_text(size = km_tick_pt),
  ) +
  guides(colour = guide_legend(ncol = 2, byrow = TRUE))

if (is.finite(opts$logrank_p)) {
  p_label <- if (opts$logrank_p < 1e-3) {
    sprintf("Global log-rank p = %.1e", opts$logrank_p)
  } else {
    sprintf("Global log-rank p = %.3f", opts$logrank_p)
  }
  km_plot <- km_plot +
    annotate(
      "text",
      x = 1,
      y = y_floor + (1 - y_floor) * 0.04,
      label = p_label,
      hjust = 0, size = km_label_pt / ggplot2::.pt, colour = "grey25"
    )
}

# ---- Risk table ----
risk_plot <- NULL
if (!is.na(opts$risk) && file.exists(opts$risk)) {
  risk <- read.delim(opts$risk, check.names = FALSE, stringsAsFactors = FALSE)
  risk <- risk[risk$endpoint == opts$endpoint, , drop = FALSE]
  risk$cluster <- as.character(risk$cluster)
  risk <- risk[risk$cluster %in% present_clusters, , drop = FALSE]
  if (nrow(risk) > 0L) {
    risk$cluster <- factor(risk$cluster, levels = present_clusters)
    risk$label <- paste0("C", as.character(risk$cluster))
    risk$label <- factor(risk$label, levels = rev(paste0("C", present_clusters)))
    # The enlarged edge columns otherwise straddle the x-limits and lose a
    # digit; interior counts stay centered on their month.
    risk$label_hjust <- ifelse(
      risk$time == min(risk$time), 0,
      ifelse(risk$time == max(risk$time), 1, 0.5)
    )
    risk_plot <- ggplot(risk, aes(x = time, y = label, colour = cluster)) +
      geom_text(aes(label = n_at_risk, hjust = label_hjust), size = km_tick_pt / ggplot2::.pt, show.legend = FALSE) +
      scale_x_continuous(limits = c(-1, 61), breaks = seq(0, 60, by = 12), expand = c(0, 0)) +
      scale_colour_manual(values = cluster_colors[present_clusters], guide = "none") +
      labs(x = NULL, y = NULL, title = "n at risk") +
      style_ggplot_theme() +
      theme(
        plot.title = element_text(size = km_label_pt, face = "bold", hjust = 0),
        axis.line = element_blank(),
        axis.ticks = element_blank(),
        axis.text.x = element_blank(),
        # Keep the enlarged row labels clear of the first risk-count column.
        axis.text.y = element_text(size = km_tick_pt, margin = margin(r = 8)),
        # Reserve the same left-side furniture as the curve panel so the
        # risk-table columns align with the months axis above.
        plot.margin = margin(2, 4, 2, 20)
      )
  }
}

plot_width <- 5.5
plot_height <- if (is.null(risk_plot)) 4.2 else 5.0
written <- style_save(opts$output, plot_width, plot_height, function() {
  if (is.null(risk_plot)) {
    print(km_plot)
  } else {
    # Stack curves on top, risk-table below using grid layout (no patchwork dep).
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow = 2, ncol = 1, heights = unit(c(4, 1.2), "null"))))
    print(km_plot, vp = viewport(layout.pos.row = 1, layout.pos.col = 1))
    print(risk_plot, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))
    popViewport()
  }
})
cat("Wrote:", paste(written, collapse = " "), "\n")
