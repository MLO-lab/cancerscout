"""Single source of truth for CancerScout figure style, in both Python and R.

Everything a figure needs to look like the rest of the paper lives in ``STYLE``:
the type scale, the diverging/sequential palettes, heatmap cell geometry in mm,
output format and dpi, and the journal column widths. ``export_plot_style``
serialises it to ``results/plot_style/style_spec.json``, which ``r/plot_style.R``
reads, so the R/ComplexHeatmap figures and the matplotlib figures are driven by
one definition rather than two that drift apart.

Rules this module exists to enforce:

- **Red is always the high end** of a diverging scale. ``diverging()`` is the
  only ramp constructor and takes no colour arguments, so an inverted ramp is
  not expressible. A panel whose "good" direction is negative flips its *data*
  and says so in the legend title.
- **The cell is the primitive**, not the canvas. Heatmaps set an explicit cell
  size in mm and derive the figure from it, so a 12-row and a 48-row panel share
  a row pitch instead of stretching to fill whatever the device left over.
- **Nothing outside this module sets a font size, dpi, or colour map.**
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Typography:
    """Five tiers, integer points. Shared by matplotlib rcParams and gpar()."""

    family: str = "DejaVu Sans"
    fallbacks: tuple[str, ...] = ("Helvetica", "Arial", "sans-serif")
    title: float = 9.0  # panel and column titles
    label: float = 8.0  # axis labels, legend titles, row-group titles
    tick: float = 7.0  # tick labels, row/column names, legend labels
    annot: float = 6.0  # in-cell values, centred significance stars
    micro: float = 5.0  # superscript stars, n-at-risk counts


@dataclass(frozen=True)
class Geometry:
    """Heatmap geometry in millimetres."""

    # Sized so a five-character value plus a corner star fits without collision:
    # "-2.36" at 6pt is ~6mm wide, which filled a 7mm cell almost completely.
    cell_w: float = 9.0
    cell_h: float = 5.2
    row_gap: float = 1.2
    row_label_max: float = 68.0
    legend_h: float = 28.0
    anno_size: float = 4.0
    pad: float = 2.0
    cell_line_pt: float = 0.5


@dataclass(frozen=True)
class Palette:
    div_low: str = "#2166AC"
    div_mid: str = "#FFFFFF"
    div_high: str = "#B2182B"
    direction: str = "red_high"  # asserted at every ramp construction
    sequential: str = "crest"
    categorical: str = "colorblind"
    missing: str = "#BDBDBD"
    unknown_cluster: str = "#4C4C4C"


@dataclass(frozen=True)
class ColorLimit:
    """Robust symmetric colour limits, so one outlier cannot wash out a panel."""

    quantile: float = 0.995
    floor: float = 0.01


@dataclass(frozen=True)
class Output:
    dpi: int = 300
    formats: tuple[str, ...] = ("png", "pdf")
    pad_in: float = 0.02
    fonttype: int = 42  # embed TrueType outlines in PDF/PS


@dataclass(frozen=True)
class Widths:
    """Nature column tiers, millimetres."""

    single: float = 89.0
    one_half: float = 120.0
    double: float = 183.0


@dataclass(frozen=True)
class StyleSpec:
    typography: Typography = field(default_factory=Typography)
    geometry: Geometry = field(default_factory=Geometry)
    palette: Palette = field(default_factory=Palette)
    color_limit: ColorLimit = field(default_factory=ColorLimit)
    output: Output = field(default_factory=Output)
    widths: Widths = field(default_factory=Widths)


STYLE = StyleSpec()
SCHEMA_VERSION = 1

CAT_PALETTE = STYLE.palette.categorical
SEQUENTIAL_CMAP = STYLE.palette.sequential
BG_GREY = "0.75"

CLUSTER_SHORT_NAMES = {
    "1": "Stromal ECM-rich / Mesenchymal",
    "2": "Mitochondrial CIN",
    "3": "Cytoskeletal CIN",
    "4": "MSI immune-active",
    "5": "Indolent CIN",
    "6": "Rare adverse state",
}

CLUSTER_COLORS = {
    "1": "#452288",
    "2": "#ED7C5A",
    "3": "#B5D33D",
    "4": "#FED23E",
    "5": "#6CA2EA",
    "6": "#A97424",
}

MSI_STATUS_COLORS = {
    "MSS": CLUSTER_COLORS["5"],
    "pMMR": CLUSTER_COLORS["5"],
    "MSI": CLUSTER_COLORS["4"],
    "MSI-H": CLUSTER_COLORS["4"],
    "dMMR": CLUSTER_COLORS["4"],
    "NA": "#BDBDBD",
}
_CLUSTER_NAME_TO_ID = {name: cluster for cluster, name in CLUSTER_SHORT_NAMES.items()}


def cluster_sort_key(value) -> tuple[int, str]:
    text = str(value)
    if text.startswith("C") and len(text) > 1 and text[1].isdigit():
        text = text[1:].split()[0]
    try:
        return (0, f"{int(float(text)):03d}")
    except Exception:
        return (1, str(value))


def cluster_short_name(value) -> str:
    text = str(value)
    if text.startswith("C") and len(text) > 1 and text[1].isdigit():
        text = text[1:].split()[0]
    return CLUSTER_SHORT_NAMES.get(text, str(value))


def cluster_display_label(value, include_id: bool = False) -> str:
    text = str(value)
    if text.startswith("C") and len(text) > 1 and text[1].isdigit():
        text = text[1:].split()[0]
    name = cluster_short_name(text)
    return f"C{text} {name}" if include_id and text in CLUSTER_SHORT_NAMES else name


def cluster_display_label_wrapped(value, include_id: bool = False) -> str:
    label = cluster_display_label(value, include_id=include_id)
    replacements = {
        "Stromal ECM-rich / Mesenchymal": "Stromal ECM-rich /\nMesenchymal",
        "Mitochondrial CIN": "Mitochondrial\nCIN",
        "Cytoskeletal CIN": "Cytoskeletal\nCIN",
        "MSI immune-active": "MSI immune-\nactive",
        "Indolent CIN": "Indolent\nCIN",
        "Rare adverse state": "Rare adverse\nstate",
    }
    for plain, wrapped in replacements.items():
        label = label.replace(plain, wrapped)
    return label


def cluster_style_records() -> list[dict[str, str]]:
    """Return cluster labels/colours in a TSV-friendly shape."""
    records: list[dict[str, str]] = []
    for cluster in sorted(CLUSTER_SHORT_NAMES, key=cluster_sort_key):
        records.append(
            {
                "cluster": cluster,
                "short_name": CLUSTER_SHORT_NAMES[cluster],
                "label": cluster_display_label(cluster, include_id=True),
                "wrapped_label": cluster_display_label_wrapped(
                    cluster, include_id=True
                ),
                "row_group_label": f"C{cluster}",
                "color": CLUSTER_COLORS[cluster],
            }
        )
    return records


def write_cluster_style_tsv(path: str | Path) -> Path:
    """Export the Python cluster style for R/ComplexHeatmap scripts."""
    import pandas as pd

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cluster_style_records()).to_csv(out, sep="\t", index=False)
    return out


def cluster_palette(values, include_labels: bool = False) -> dict[str, str]:
    out = {}
    for value in values:
        key = str(value)
        cluster_id = key
        if key.startswith("C") and len(key) > 1 and key[1].isdigit():
            cluster_id = key[1:].split()[0]
        elif key in _CLUSTER_NAME_TO_ID:
            cluster_id = _CLUSTER_NAME_TO_ID[key]
        label = (
            cluster_display_label(cluster_id, include_id=True)
            if include_labels
            else key
        )
        out[label] = CLUSTER_COLORS.get(cluster_id, "#4C4C4C")
    return out


def relabel_cluster_axis(
    obj, axis: str = "columns", include_id: bool = False, wrapped: bool = False
):
    """Return a copy with cluster ids on one axis replaced by descriptive names."""
    out = obj.copy()
    label_fn = cluster_display_label_wrapped if wrapped else cluster_display_label
    if axis == "columns":
        name = out.columns.name
        out.columns = [label_fn(c, include_id=include_id) for c in out.columns]
        out.columns.name = name
    elif axis == "index":
        name = out.index.name
        out.index = [label_fn(c, include_id=include_id) for c in out.index]
        out.index.name = name
    else:
        raise ValueError("axis must be 'columns' or 'index'")
    return out


def make_color_map(
    categories, palette: str = CAT_PALETTE
) -> dict[str, tuple[float, float, float]]:
    cats = list(dict.fromkeys([str(c) for c in categories]))
    pal = sns.color_palette(palette, n_colors=len(cats))
    return {c: pal[i] for i, c in enumerate(cats)}


def font_pt(tier: str) -> float:
    """Point size for a named type tier: title/label/tick/annot/micro."""
    try:
        return float(getattr(STYLE.typography, tier))
    except AttributeError as exc:
        raise KeyError(f"Unknown type tier {tier!r}") from exc


def mm(name: str) -> float:
    """Named geometry value, in millimetres."""
    try:
        return float(getattr(STYLE.geometry, name))
    except AttributeError as exc:
        raise KeyError(f"Unknown geometry key {name!r}") from exc


def inches(mm_value: float) -> float:
    return float(mm_value) / MM_PER_INCH


def figure_size(
    tier: str = "single", height_mm: float | None = None, aspect: float = 0.75
) -> tuple[float, float]:
    """Figure size in inches for a journal column tier."""
    width_mm = float(getattr(STYLE.widths, tier))
    height_mm = width_mm * aspect if height_mm is None else height_mm
    return inches(width_mm), inches(height_mm)


def color_limit(values, q: float | None = None, floor: float | None = None) -> float:
    """Symmetric colour limit at a robust quantile of |values|.

    Uses a quantile rather than the maximum so a single extreme cell cannot
    stretch the scale and wash out the rest of the panel. Values beyond the
    limit still render at the endpoint colour and keep their printed number.
    """
    q = STYLE.color_limit.quantile if q is None else q
    floor = STYLE.color_limit.floor if floor is None else floor
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return max(float(floor), 1.0)
    return max(float(np.quantile(np.abs(v), q)), float(floor))


def diverging_cmap() -> LinearSegmentedColormap:
    """The one diverging ramp. Low is blue, high is red, always."""
    palette = STYLE.palette
    if palette.direction != "red_high":
        raise ValueError(
            f"Diverging direction must be red_high, got {palette.direction!r}"
        )
    return LinearSegmentedColormap.from_list(
        "cancerscout_diverging", [palette.div_low, palette.div_mid, palette.div_high]
    )


def diverging(values, q: float | None = None, floor: float | None = None):
    """(cmap, norm) for a zero-centred matrix. The only diverging constructor."""
    limit = color_limit(values, q=q, floor=floor)
    return diverging_cmap(), TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def sequential_cmap():
    return sns.color_palette(STYLE.palette.sequential, as_cmap=True)


def sequential_blue_cmap():
    """White-to-blue ramp for one-directional matrices (e.g. row-% confusion).

    Reuses the diverging ramp's midpoint and low end so a one-directional
    matrix sits in the same colour family as the diverging heatmaps.
    """
    return LinearSegmentedColormap.from_list(
        "cancerscout_sequential_blue",
        [STYLE.palette.div_mid, STYLE.palette.div_low],
    )


CSS_PX_PER_INCH = 96.0


def save_plotly(fig, path: str | Path, width_in: float, height_in: float) -> list[Path]:
    """Write a plotly figure in every configured format at the spec dpi.

    Plotly lays out in CSS pixels and treats ``font.size`` as pixels, so asking
    it for a 300-dpi-sized canvas directly shrinks all type by dpi/96. Instead
    lay out at the true CSS size and let ``scale`` do the resampling, which keeps
    type proportional. Pillow then stamps the dpi into the PNG header, which
    plotly never writes.
    """
    from PIL import Image

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dpi = STYLE.output.dpi
    css_w = round(width_in * CSS_PX_PER_INCH)
    css_h = round(height_in * CSS_PX_PER_INCH)
    written: list[Path] = []
    for ext in STYLE.output.formats:
        target = out.with_suffix(f".{ext}")
        if ext == "png":
            fig.write_image(str(target), width=css_w, height=css_h, scale=dpi / CSS_PX_PER_INCH)
            with Image.open(target) as img:
                img.save(target, dpi=(dpi, dpi))
        else:
            fig.write_image(str(target), width=css_w, height=css_h)
        written.append(target)
    return written


def pt_to_css_px(tier: str) -> float:
    """A style type tier in CSS pixels, for plotly's pixel-based font sizes."""
    return font_pt(tier) * CSS_PX_PER_INCH / 72.0


def savefig(fig, path: str | Path, close: bool = True) -> list[Path]:
    """Write every configured output format and return the paths written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ext in STYLE.output.formats:
        target = out.with_suffix(f".{ext}")
        fig.savefig(
            target,
            dpi=mpl.rcParams["savefig.dpi"],
            bbox_inches="tight",
            pad_inches=STYLE.output.pad_in,
        )
        written.append(target)
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return written


def style_spec_dict() -> dict:
    """The spec plus cluster records, as written to style_spec.json for R."""
    spec = asdict(STYLE)
    spec["typography"]["fallbacks"] = list(STYLE.typography.fallbacks)
    spec["output"]["formats"] = list(STYLE.output.formats)
    return {
        "schema_version": SCHEMA_VERSION,
        **spec,
        "clusters": cluster_style_records(),
    }


def write_style_spec_json(path: str | Path) -> Path:
    """Write the spec, but only when it actually changed.

    Leaving the mtime alone on a no-op re-export keeps it meaningful: a figure
    older than the spec then really was rendered against an older style, rather
    than merely rendered in the same sweep that re-exported an identical spec.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(style_spec_dict(), indent=2) + "\n"
    if out.exists() and out.read_text() == payload:
        return out
    out.write_text(payload)
    return out


def export_style(directory: str | Path) -> tuple[Path, Path]:
    """Write both artifacts the R scripts read. Safe to call repeatedly."""
    directory = Path(directory)
    return (
        write_cluster_style_tsv(directory / "cluster_style.tsv"),
        write_style_spec_json(directory / "style_spec.json"),
    )


def clean_ax(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_zero_lines(ax, alpha: float = 0.6) -> None:
    ax.axhline(0, ls="--", lw=0.8, color="0.6", alpha=alpha, zorder=0)
    ax.axvline(0, ls="--", lw=0.8, color="0.6", alpha=alpha, zorder=0)


def place_legend(
    ax,
    mode: str = "inside",
    title: str | None = None,
    loc: str = "upper right",
    anchor: tuple[float, float] = (0.98, 0.98),
    ncol: int = 1,
    fontsize: int | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if mode == "outside":
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            borderaxespad=0,
            ncol=ncol,
            fontsize=fontsize,
        )
    elif mode == "manual":
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            loc="center",
            bbox_to_anchor=anchor,
            ncol=ncol,
            fontsize=fontsize,
        )
    else:
        ax.legend(
            handles,
            labels,
            title=title,
            frameon=False,
            loc=loc,
            ncol=ncol,
            fontsize=fontsize,
        )


def clean_colorbar(cbar) -> None:
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=font_pt("tick"))


def _house_rcparams(dpi: int, family: str) -> dict:
    return {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": STYLE.output.pad_in,
        "figure.figsize": figure_size("single"),
        # A concrete family name bypasses matplotlib's fallback chain, so set the
        # generic and let the list resolve.
        "font.family": "sans-serif",
        "font.sans-serif": [family, *STYLE.typography.fallbacks],
        "font.size": font_pt("label"),
        "axes.titlesize": font_pt("title"),
        "axes.labelsize": font_pt("label"),
        "axes.labelpad": 2.0,
        "xtick.labelsize": font_pt("tick"),
        "ytick.labelsize": font_pt("tick"),
        "legend.fontsize": font_pt("tick"),
        "legend.title_fontsize": font_pt("label"),
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "pdf.fonttype": STYLE.output.fonttype,
        "ps.fonttype": STYLE.output.fonttype,
    }


def set_house_style(dpi: int | None = None, font: str | None = None) -> None:
    """Apply the house style. Idempotent, so call it again after any library
    that sets global matplotlib state (seaborn, scanpy) to reclaim the spec.

    Order matters: seaborn's ``context="paper"`` rewrites font sizes and line
    widths, so it must run *before* the house parameters, not after.
    """
    dpi = STYLE.output.dpi if dpi is None else dpi
    family = font or STYLE.typography.family
    sns.set_theme(style="white", context="paper")
    sns.set_style({"axes.grid": False})
    mpl.rcParams.update(_house_rcparams(dpi, family))
