# Shared CancerScout R plotting style.
#
# There is exactly one definition of the house style and it lives in
# cancerscout/plot_style.py. That module serialises it to
# results/plot_style/style_spec.json, which this file reads. There is
# deliberately NO hardcoded fallback: a fallback that drifts from Python
# produces inconsistent figures with a zero exit code, which is worse than
# stopping. Run `python -m cancerscout.export_plot_style` if this file errors.
#
# Every R figure script sources this and takes its type sizes from style_gp(),
# its geometry from style_mm(), its diverging ramp from style_diverging_col_fun(),
# and its device from style_save(). None of them should name a font size,
# a colour, or a dpi directly.

style_this_file <- tryCatch(
  normalizePath(sys.frame(1)$ofile, mustWork = FALSE),
  error = function(e) NA_character_
)
if (is.na(style_this_file) || !nzchar(style_this_file)) {
  style_this_file <- file.path(getwd(), "r", "plot_style.R")
}
STYLE_REPO_ROOT <- normalizePath(file.path(dirname(style_this_file), ".."), mustWork = FALSE)
STYLE_SPEC_JSON <- file.path(STYLE_REPO_ROOT, "results", "plot_style", "style_spec.json")

if (!file.exists(STYLE_SPEC_JSON)) {
  stop(sprintf(
    "Missing %s\n  Run: conda run -n proj python -m cancerscout.export_plot_style",
    STYLE_SPEC_JSON
  ), call. = FALSE)
}

STYLE <- jsonlite::fromJSON(STYLE_SPEC_JSON, simplifyDataFrame = TRUE)

# Measure text with the font style_save() draws. ComplexHeatmap sizes row
# names, annotations and legends when they are built, on its own pdf(NULL)
# device (ComplexHeatmap:::dev.null). pdf() has no DejaVu Sans metrics and falls
# back to Helvetica, about a quarter narrower, so long labels overflow their
# slot and get clipped. Point that device, and any measurement made before a
# figure device opens, at cairo with the spec family.
style_measure_device <- function() {
  grDevices::cairo_pdf(tempfile(fileext = ".pdf"), family = STYLE$typography$family)
}
if (requireNamespace("ComplexHeatmap", quietly = TRUE)) {
  utils::assignInNamespace("dev.null", function(...) style_measure_device(), ns = "ComplexHeatmap")
}
style_measure_device()

style_cluster_table <- STYLE$clusters
style_cluster_table$cluster <- as.character(style_cluster_table$cluster)

# ---- spec accessors -------------------------------------------------------

style_pt <- function(tier) {
  value <- STYLE$typography[[tier]]
  if (is.null(value)) stop(sprintf("Unknown type tier '%s'", tier), call. = FALSE)
  as.numeric(value)
}

style_mm <- function(name) {
  value <- STYLE$geometry[[name]]
  if (is.null(value)) stop(sprintf("Unknown geometry key '%s'", name), call. = FALSE)
  as.numeric(value)
}

style_gp <- function(tier, face = "plain", col = NULL, lineheight = NULL) {
  args <- list(
    fontsize = style_pt(tier),
    fontface = face,
    fontfamily = STYLE$typography$family
  )
  if (!is.null(col)) args$col <- col
  if (!is.null(lineheight)) args$lineheight <- lineheight
  do.call(gpar, args)
}

# ---- cluster labels and colours -------------------------------------------

style_cluster_order <- function(present = NULL) {
  ids <- as.character(style_cluster_table$cluster)
  if (!is.null(present)) {
    present <- as.character(present)
    ids <- ids[ids %in% present]
    extra <- setdiff(present, ids)
    if (length(extra) > 0L) {
      suppressWarnings(num <- as.numeric(extra))
      extra <- extra[order(is.na(num), num, extra)]
      ids <- c(ids, extra)
    }
  }
  ids
}

style_named <- function(column, ids, fallback_prefix = NULL) {
  lookup <- setNames(style_cluster_table[[column]], style_cluster_table$cluster)
  out <- lookup[as.character(ids)]
  missing <- is.na(out)
  if (any(missing)) {
    out[missing] <- if (is.null(fallback_prefix)) {
      as.character(ids)[missing]
    } else {
      paste0(fallback_prefix, as.character(ids)[missing])
    }
  }
  names(out) <- as.character(ids)
  out
}

style_cluster_labels <- function(ids) style_named("label", ids, "C")
style_cluster_column_labels <- function(ids) style_named("wrapped_label", ids, "C")
style_cluster_row_labels <- function(ids) style_named("row_group_label", ids, "C")
style_cluster_colors <- function(ids) {
  out <- style_named("color", ids, NULL)
  out[is.na(out) | !nzchar(out)] <- STYLE$palette$unknown_cluster
  out
}

# ---- colour scales --------------------------------------------------------

# Robust symmetric limit: a quantile of |x| rather than the maximum, so one
# extreme cell cannot stretch the scale and wash out the rest of the panel.
# Cells beyond the limit still render at the endpoint colour and keep their
# printed value.
style_color_limit <- function(mat, q = NULL, floor_value = NULL) {
  q <- if (is.null(q)) STYLE$color_limit$quantile else q
  floor_value <- if (is.null(floor_value)) STYLE$color_limit$floor else floor_value
  values <- abs(as.numeric(mat))
  values <- values[is.finite(values)]
  if (length(values) == 0L) return(max(floor_value, 1))
  max(as.numeric(stats::quantile(values, probs = q, names = FALSE)), floor_value)
}

# The one diverging ramp. It takes no colour arguments on purpose: red is
# always the high end, so an inverted ramp is not expressible. A panel whose
# "good" direction is negative flips its data and says so in the legend title.
style_diverging_col_fun <- function(mat, q = NULL, floor_value = NULL) {
  if (!identical(STYLE$palette$direction, "red_high")) {
    stop("Diverging direction must be red_high", call. = FALSE)
  }
  lim <- style_color_limit(mat, q, floor_value)
  colorRamp2(
    c(-lim, 0, lim),
    c(STYLE$palette$div_low, STYLE$palette$div_mid, STYLE$palette$div_high)
  )
}

# ---- heatmap furniture ----------------------------------------------------

style_cluster_bottom_annotation <- function(ids, show_name = TRUE) {
  HeatmapAnnotation(
    Cluster = factor(as.character(ids), levels = as.character(ids)),
    col = list(Cluster = style_cluster_colors(ids)),
    annotation_name_side = "left",
    annotation_name_gp = style_gp("label"),
    show_annotation_name = show_name,
    show_legend = FALSE,
    which = "column",
    simple_anno_size = unit(style_mm("anno_size"), "mm")
  )
}

# Superscript significance star, placed in the true top-right corner of the
# cell. Defined once because three renderers draw it: at a smaller offset the
# star lands on the last digit of a 4-5 character value.
style_cell_star <- function(star, x, y, width, height, col) {
  if (is.na(star) || !nzchar(star)) return(invisible(NULL))
  grid.text(
    star,
    x = x + width * 0.40,
    y = y + height * 0.32,
    gp = style_gp("micro", col = col)
  )
}

# In-cell text colour that stays legible on both ends of the diverging ramp.
style_cell_text_col <- function(value, limit) {
  if (is.finite(value) && abs(value) > limit * 0.55) "white" else "black"
}

style_legend_param <- function(title) {
  list(
    title = title,
    title_gp = style_gp("label", face = "bold"),
    labels_gp = style_gp("tick"),
    legend_height = unit(style_mm("legend_h"), "mm"),
    border = NA
  )
}

# Shared Heatmap() arguments. Callers modifyList() their specifics in, so every
# panel gets the same cell size, gaps, name sizes, and legend geometry.
style_heatmap_args <- function(mat, legend_title, row_split = NULL, column_title = NULL) {
  args <- list(
    col = style_diverging_col_fun(mat),
    name = legend_title,
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    row_names_side = "left",
    row_names_max_width = unit(style_mm("row_label_max"), "mm"),
    row_names_gp = style_gp("tick"),
    column_names_side = "bottom",
    column_names_gp = style_gp("tick", lineheight = 0.9),
    rect_gp = gpar(col = "white", lwd = style_mm("cell_line_pt")),
    # Both dimensions are set so the cell, not the device, is the primitive.
    width = unit(style_mm("cell_w") * ncol(mat), "mm"),
    height = unit(style_mm("cell_h") * nrow(mat), "mm"),
    heatmap_legend_param = style_legend_param(legend_title)
  )
  if (!is.null(row_split)) {
    args$row_split <- row_split
    args$row_gap <- unit(style_mm("row_gap"), "mm")
    args$row_title <- "%s"
    args$row_title_gp <- style_gp("label", face = "bold")
    args$row_title_rot <- 0
  }
  if (!is.null(column_title) && nzchar(column_title)) {
    args$column_title <- column_title
    args$column_title_gp <- style_gp("title", face = "bold")
  }
  args
}

# ---- device ---------------------------------------------------------------

# Ask ComplexHeatmap how big the drawn object actually is, on a null device,
# then render at exactly that size. The cell size in mm is the fixed input, so
# every panel shares a row pitch and the canvas is whatever that content needs
# -- no per-script inches-per-row coefficient, and no clipped labels from an
# analytic furniture estimate that under-counts rotated multi-line text.
style_heatmap_device_size <- function(ht, ...) {
  pad_in <- style_mm("pad") / 25.4
  style_measure_device()
  on.exit(if (dev.cur() > 1L) invisible(dev.off()), add = TRUE)
  drawn <- draw(ht, ...)
  size <- list(
    width_in = convertWidth(ComplexHeatmap:::width(drawn), "in", valueOnly = TRUE) + 2 * pad_in,
    height_in = convertHeight(ComplexHeatmap:::height(drawn), "in", valueOnly = TRUE) + 2 * pad_in
  )
  size
}

# Measure, then write every configured format at the measured size.
style_heatmap_save <- function(path, ht, ...) {
  size <- style_heatmap_device_size(ht, ...)
  style_save(path, size$width_in, size$height_in, function() draw(ht, ...))
}

# Writes every configured format from one draw callback, with the font family
# set on both devices. Base pdf() defaults to Helvetica while cairo png()
# resolves to the spec family, which silently split the typeface per format.
style_save <- function(path, width_in, height_in, draw_fun) {
  base <- sub("\\.(png|pdf)$", "", path)
  family <- STYLE$typography$family
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  written <- character(0)
  for (fmt in STYLE$output$formats) {
    target <- paste0(base, ".", fmt)
    if (identical(fmt, "png")) {
      png(target, width = width_in, height = height_in, units = "in",
          res = STYLE$output$dpi, type = "cairo", family = family, bg = "white")
    } else if (identical(fmt, "pdf")) {
      cairo_pdf(target, width = width_in, height = height_in, family = family, bg = "white")
    } else {
      stop(sprintf("Unsupported output format '%s'", fmt), call. = FALSE)
    }
    draw_fun()
    dev.off()
    written <- c(written, target)
  }
  written
}

style_ggplot_theme <- function(base = "tick") {
  ggplot2::theme_classic(
    base_size = style_pt(base),
    base_family = STYLE$typography$family
  ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = style_pt("title"), face = "bold"),
      axis.title = ggplot2::element_text(size = style_pt("label")),
      axis.text = ggplot2::element_text(size = style_pt("tick")),
      legend.title = ggplot2::element_text(size = style_pt("label")),
      legend.text = ggplot2::element_text(size = style_pt("tick")),
      strip.text = ggplot2::element_text(size = style_pt("label"), face = "bold"),
      legend.key.size = ggplot2::unit(3.5, "mm")
    )
}
