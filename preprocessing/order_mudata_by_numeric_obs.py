#!/usr/bin/env python
"""Write a MuData file with observations sorted by numeric obs id.

This is useful for the CancerScout BID-indexed MuData objects whose obs names
are strings like "1", "2", ..., "1000" with one missing BID. It preserves all
modalities and values, changing only observation order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mudata as mu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort MuData observations by numeric obs name.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def numeric_obs_order(obs_names) -> list[str]:
    def key(value: object) -> tuple[int, int | str]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted(map(str, obs_names), key=key)


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    mdata = mu.read_h5mu(args.input)
    order = numeric_obs_order(mdata.obs_names)
    mdata_ordered = mdata[order, :].copy()
    mdata_ordered.write_h5mu(args.out)

    print(f"Wrote {args.out}")
    print(f"n_obs={mdata_ordered.n_obs} n_vars={mdata_ordered.n_vars}")
    print(f"first_obs={list(mdata_ordered.obs_names[:10])}")
    print(f"last_obs={list(mdata_ordered.obs_names[-10:])}")


if __name__ == "__main__":
    main()
