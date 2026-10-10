"""Check that figures and figure code still follow the shared style spec.

Two independent halves:

- **Rendered audit** (always on): walks ``figures/`` and asserts every PNG is at
  the spec dpi, has a non-empty PDF sibling, fits the double-column bound, and
  was rendered no earlier than the current ``style_spec.json``. This catches
  drift in the *output*, which a source grep cannot see.
- **Source lint** (``--strict``): fails on inline font sizes, dpi literals,
  colour maps, ``savefig`` bypasses, raw R device calls, and unguarded calls to
  third-party global style setters. Off by default so it cannot block the
  migration it exists to protect; turn it on once every script is converted.

Writes ``reports/_auto/figure_style_audit.md`` and exits non-zero on violations,
so running it last in ``redo_figures`` makes a whole regeneration fail if the
figures drifted.

Usage:
    conda run -n proj python -m cancerscout.check_plot_style [--strict]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image

from cancerscout.paths import FIGURES_DIR, REPO_ROOT, REPORTS_DIR, RESULTS_DIR
from cancerscout.plot_style import STYLE


SPEC_JSON = RESULTS_DIR / "plot_style" / "style_spec.json"
AUDIT_MD = REPORTS_DIR / "_auto" / "figure_style_audit.md"
MM_PER_INCH = 25.4

# Wider than a double column by intent; each needs a reason, not just a pass.
WIDTH_ALLOWLIST: dict[str, str] = {}

# Directories whose generator is deliberately outside redo_figures (it refits
# models), so their figures are stale by construction and are reported in the
# audit table without failing the run.
UNMANAGED_DIRS = ("figures/cluster_marker_rfecv_",)

# Retired assets kept only for provenance (see reports/figure_manifest.md §5).
# They are not manuscript figures and are not regenerated, so they are skipped
# entirely instead of being reported as style problems.
ARCHIVE_DIRS = ("figures/archive/",)

STYLE_SOURCES = {"plot_style.py", "plot_style.R", "check_plot_style.py"}

SOURCE_RULES = [
    (
        "inline type size",
        re.compile(r"(?:fontsize|labelsize|base_size)\s*=\s*[0-9]"),
        ("*.py", "*.R"),
    ),
    ("inline dpi", re.compile(r"\b(?:dpi|res)\s*=\s*[0-9]"), ("*.py", "*.R")),
    (
        "inline colour map",
        re.compile(r"cmap\s*=\s*[\"']|LinearSegmentedColormap|colorRamp2\("),
        ("*.py", "*.R"),
    ),
    ("savefig bypass", re.compile(r"(?:fig|plt)\.savefig\("), ("*.py",)),
    ("raw R device", re.compile(r"(?<![A-Za-z_])(?:png|pdf|cairo_pdf)\s*\("), ("*.R",)),
    (
        "unguarded global style call",
        re.compile(
            r"set_figure_params|sns\.set_theme|sns\.set_style|sns\.set_context|plt\.style\.use"
        ),
        ("*.py",),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on inline style literals in source (see module docstring).",
    )
    return parser.parse_args()


def png_geometry(path: Path) -> tuple[int, int, float]:
    """(width_px, height_px, dpi) from the PNG header."""
    with Image.open(path) as img:
        width, height = img.size
        dpi = img.info.get("dpi", (0, 0))[0]
    return width, height, round(float(dpi), 1)


def audit_figures(figures_dir: Path) -> tuple[list[dict], list[str]]:
    spec_mtime = SPEC_JSON.stat().st_mtime if SPEC_JSON.exists() else 0.0
    max_width_in = STYLE.widths.double / MM_PER_INCH
    rows, problems = [], []

    for png in sorted(figures_dir.rglob("*.png")):
        rel = png.relative_to(REPO_ROOT)
        if str(rel).startswith(ARCHIVE_DIRS):
            continue
        unmanaged = str(rel).startswith(UNMANAGED_DIRS)
        try:
            width_px, height_px, dpi = png_geometry(png)
        except Exception as exc:  # unreadable file is itself a finding
            problems.append(f"{rel}: unreadable PNG ({exc})")
            continue
        width_in = width_px / dpi if dpi else float("nan")
        pdf = png.with_suffix(".pdf")
        has_pdf = pdf.exists() and pdf.stat().st_size > 0
        stale = png.stat().st_mtime < spec_mtime

        if not unmanaged and dpi != STYLE.output.dpi:
            problems.append(f"{rel}: {dpi} dpi, expected {STYLE.output.dpi}")
        if not unmanaged and not has_pdf:
            problems.append(f"{rel}: no PDF sibling")
        if (
            not unmanaged
            and dpi
            and width_in > max_width_in + 0.01
            and str(rel) not in WIDTH_ALLOWLIST
        ):
            problems.append(
                f"{rel}: {width_in:.1f} in wide, over the {max_width_in:.1f} in double-column bound"
            )
        if not unmanaged and stale:
            problems.append(f"{rel}: older than style_spec.json, re-render it")

        rows.append(
            {
                "path": str(rel),
                "px": f"{width_px}x{height_px}",
                "inches": f"{width_in:.2f}" if dpi else "?",
                "dpi": dpi,
                "pdf": "yes" if has_pdf else "NO",
                "fresh": "unmanaged" if unmanaged else ("no" if stale else "yes"),
            }
        )
    return rows, problems


def lint_sources() -> list[str]:
    problems = []
    for directory, patterns in (
        (REPO_ROOT / "cancerscout", ("*.py",)),
        (REPO_ROOT / "r", ("*.R",)),
    ):
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                if path.name in STYLE_SOURCES:
                    continue
                lines = path.read_text().splitlines()
                for rule, regex, applies in SOURCE_RULES:
                    if pattern not in applies:
                        continue
                    for number, line in enumerate(lines, start=1):
                        if line.lstrip().startswith(("#", "//")):
                            continue
                        if regex.search(line):
                            problems.append(
                                f"{path.relative_to(REPO_ROOT)}:{number}: {rule}: {line.strip()[:90]}"
                            )
    return problems


def write_audit(rows: list[dict], problems: list[str], strict: bool) -> Path:
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure style audit",
        "",
        f"Spec: `{SPEC_JSON.relative_to(REPO_ROOT)}` — {STYLE.output.dpi} dpi, "
        f"{', '.join(STYLE.output.formats)}, {STYLE.typography.family}, "
        f"type tiers {STYLE.typography.title}/{STYLE.typography.label}/"
        f"{STYLE.typography.tick}/{STYLE.typography.annot}/{STYLE.typography.micro} pt.",
        "",
        f"{len(rows)} PNG(s) checked; {len(problems)} problem(s)"
        + (" (source lint on)." if strict else " (source lint off; pass --strict)."),
        "",
    ]
    if problems:
        lines += ["## Problems", ""] + [f"- {p}" for p in problems] + [""]
    lines += [
        "## Figures",
        "",
        "| Figure | px | width (in) | dpi | pdf | fresh |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{r['path']}` | {r['px']} | {r['inches']} | {r['dpi']} | {r['pdf']} | {r['fresh']} |"
        for r in rows
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n")
    return AUDIT_MD


def main() -> None:
    args = parse_args()
    rows, problems = audit_figures(args.figures_dir)
    if args.strict:
        problems += lint_sources()
    out = write_audit(rows, problems, args.strict)
    print(f"Checked {len(rows)} figures; wrote {out}")
    if problems:
        for problem in problems[:20]:
            print(f"  {problem}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more (see {out})")
        raise SystemExit(1)
    print("Figure style OK")


if __name__ == "__main__":
    main()
