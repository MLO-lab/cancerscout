"""Export the canonical Python plot style for non-Python renderers.

Writes two files into ``results/plot_style/``:

- ``cluster_style.tsv``  — cluster labels and colours (legacy schema, unchanged)
- ``style_spec.json``    — the full style spec plus the same cluster records

``r/plot_style.R`` reads the JSON, so this is the only place the R figures get
their typography, geometry, palettes, and output settings from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cancerscout.paths import RESULTS_DIR
from cancerscout.plot_style import export_style


DEFAULT_OUT_DIR = RESULTS_DIR / "plot_style"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for cluster_style.tsv and style_spec.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tsv, spec = export_style(args.out_dir)
    print(f"Wrote cluster style TSV: {tsv}")
    print(f"Wrote style spec JSON:   {spec}")


if __name__ == "__main__":
    main()
