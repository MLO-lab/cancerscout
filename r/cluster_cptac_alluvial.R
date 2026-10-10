#!/usr/bin/env Rscript
# Proteomic subgroup -> predicted CPTAC subtype alluvial plot: the alternative
# to the plotly Sankey drawn by run_subtype_concordance.plot_sankey.
#
# Styled after the LymphGen/LymphXpress alluvial: pill-shaped strata in their
# subtype colour with the label inside, and opaque flows that shade from source
# to target colour. ggalluvial cannot draw rounded
# strata or gradient flows, so the layout is computed here and drawn with grid
# in a millimetre coordinate system: one native unit is one printed millimetre,
# so pill radii and label padding need no conversion.
#
# A stratum too thin to hold its label (C6, 15 of 942 tumours) gets a pill
# taller than its flows, as in the reference figure; the flows keep their true
# thickness.
#
# Args: <bilateral_counts.tsv> <output.png> <subtype>=<colour> ...
#   One <subtype>=<colour> pair per CPTAC column, so the CPTAC colour mapping
#   stays in run_subtype_concordance.CPTAC_COLORS.

suppressPackageStartupMessages(library(grid))

script_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
repo_root <- if (!is.na(script_file) && nzchar(script_file)) {
  normalizePath(file.path(dirname(normalizePath(script_file, mustWork = FALSE)), ".."), mustWork = FALSE)
} else {
  getwd()
}
source(file.path(repo_root, "r", "plot_style.R"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L) {
  stop("Usage: cluster_cptac_alluvial.R <bilateral_counts.tsv> <output.png> <subtype>=<colour> ...")
}
pairs <- strsplit(args[-(1:2)], "=", fixed = TRUE)
right_colors <- setNames(vapply(pairs, `[`, "", 2L), vapply(pairs, `[`, "", 1L))

counts <- as.matrix(read.delim(args[1], check.names = FALSE, row.names = 1))
left <- style_cluster_order(rownames(counts))
counts <- counts[left, , drop = FALSE]
# Right axis ordered by the flow-weighted mean rank of its incoming flows, as in
# the Sankey: with C1..C6 fixed top-down this keeps the dominant flows apart.
right <- colnames(counts)[order(colSums(counts * seq_along(left)) / colSums(counts))]
counts <- counts[, right, drop = FALSE]

left_labels <- paste0("C", left)
left_colors <- style_cluster_colors(left)

# ---- geometry (mm) -------------------------------------------------------

width_mm <- STYLE$widths$single
height_mm <- 80
margin_mm <- 2.5
gap_mm <- 2.5
pad_mm <- 1.6
radius_max_mm <- 4

label_gp <- style_gp("title", face = "bold")
text_mm <- function(s, dim = convertWidth, f = grobWidth) {
  dim(f(textGrob(s, gp = label_gp)), "mm", valueOnly = TRUE)
}
pill_w <- max(vapply(c(left_labels, right), text_mm, 0)) + 2 * pad_mm
pill_min_h <- text_mm("C", convertHeight, grobHeight) + 1.2

# Both axes span the same height, so the right axis's fewer gaps are wider.
top <- height_mm - margin_mm
mm_per_n <- (top - margin_mm - gap_mm * (length(left) - 1)) / sum(counts)
right_gap <- gap_mm * (length(left) - 1) / (length(right) - 1)

stack <- function(sizes, gap) {
  heights <- sizes * mm_per_n
  ymax <- top - c(0, cumsum(heights + gap))[seq_along(sizes)]
  data.frame(ymax = ymax, ymin = ymax - heights)
}
left_strata <- stack(rowSums(counts), gap_mm)
right_strata <- stack(colSums(counts), right_gap)

# Within a stratum, flows stack in the order of the opposite axis.
left_offset <- t(apply(counts, 1, function(row) cumsum(row) - row))
right_offset <- apply(counts, 2, function(col) cumsum(col) - col)
flows <- expand.grid(i = seq_along(left), j = seq_along(right))
flows$n <- counts[cbind(flows$i, flows$j)]
flows <- flows[flows$n > 0, ]
flows$l_top <- left_strata$ymax[flows$i] - mm_per_n * left_offset[cbind(flows$i, flows$j)]
flows$r_top <- right_strata$ymax[flows$j] - mm_per_n * right_offset[cbind(flows$i, flows$j)]
# Largest first, so every thin flow stays visible where it crosses a big one.
flows <- flows[order(-flows$n), ]

left_x <- c(0.5, 0.5 + pill_w)
right_x <- c(width_mm - 0.5 - pill_w, width_mm - 0.5)
# Flows start under the pills so no anti-aliasing seam shows at the join.
flow_x <- seq(left_x[2] - 1, right_x[1] + 1, length.out = 80)
flow_s <- (1 - cos(pi * seq(0, 1, length.out = length(flow_x)))) / 2

# Only the outer end is rounded; the inner end is square so the flows leave it
# without a notch at the corners.
pill <- function(x0, x1, ymin, ymax, outer) {
  h <- max(ymax - ymin, pill_min_h)
  yc <- (ymin + ymax) / 2
  y0 <- yc - h / 2
  y1 <- yc + h / 2
  r <- min(radius_max_mm, h / 2)
  # Corners top-right, top-left, bottom-left, bottom-right.
  r <- if (outer == "left") c(0, r, r, 0) else c(r, 0, 0, r)
  cx <- c(x1 - r[1], x0 + r[2], x0 + r[3], x1 - r[4])
  cy <- c(y1 - r[1], y1 - r[2], y0 + r[3], y0 + r[4])
  arc <- seq(0, pi / 2, length.out = 16)
  pts <- do.call(rbind, lapply(1:4, function(q) {
    theta <- arc + (q - 1) * pi / 2
    cbind(cx[q] + r[q] * cos(theta), cy[q] + r[q] * sin(theta))
  }))
  list(x = pts[, 1], y = pts[, 2], yc = yc)
}

contrast_col <- function(fill) {
  rgb <- grDevices::col2rgb(fill)[, 1] / 255
  if (sum(c(0.299, 0.587, 0.114) * rgb) >= 0.62) "black" else "white"
}

draw_axis <- function(labels, colors, strata, x, just) {
  for (k in seq_along(labels)) {
    p <- pill(x[1], x[2], strata$ymin[k], strata$ymax[k], just)
    grid.polygon(p$x, p$y, default.units = "native", gp = gpar(fill = colors[[k]], col = NA))
    grid.text(
      labels[k],
      x = if (just == "left") x[1] + pad_mm else x[2] - pad_mm,
      y = p$yc,
      just = c(just, "centre"),
      default.units = "native",
      gp = style_gp("title", face = "bold", col = contrast_col(colors[[k]]))
    )
  }
}

draw_alluvial <- function() {
  grid.newpage()
  pushViewport(viewport(xscale = c(0, width_mm), yscale = c(0, height_mm)))
  for (k in seq_len(nrow(flows))) {
    f <- flows[k, ]
    src <- left_colors[[f$i]]
    tgt <- right_colors[[right[f$j]]]
    fill <- if (identical(src, tgt)) {
      src
    } else {
      linearGradient(c(src, tgt), stops = c(0.3, 0.7), x1 = 0, y1 = 0.5, x2 = 1, y2 = 0.5)
    }
    y_top <- f$l_top + (f$r_top - f$l_top) * flow_s
    grid.polygon(
      c(flow_x, rev(flow_x)),
      c(y_top, rev(y_top - f$n * mm_per_n)),
      default.units = "native",
      gp = gpar(fill = fill, col = NA)
    )
  }
  draw_axis(left_labels, left_colors, left_strata, left_x, "left")
  draw_axis(right, right_colors[right], right_strata, right_x, "right")
  popViewport()
}

written <- style_save(args[2], width_mm / 25.4, height_mm / 25.4, draw_alluvial)
cat(sprintf("Wrote %s\n", paste(written, collapse = ", ")))
