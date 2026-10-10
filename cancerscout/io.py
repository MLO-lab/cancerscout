"""Input/output helpers for downstream CRC proteogenomics analysis."""

from __future__ import annotations

import io as _stdlib_io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cancerscout.paths import ACTIVE_PROCESSED_NORMICS_DIR

try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

try:
    import mudata as mu
except Exception:  # pragma: no cover
    mu = None


@dataclass
class DownstreamData:
    results_dir: Path
    factor_adata: object
    metadata: pd.DataFrame
    factors: pd.DataFrame
    proteome: pd.DataFrame | None
    gene_symbols: pd.Series | None


BID_CANDIDATES = [
    "Proteome:new_bid",
    "Proteome:new_sample_token",
    "new_bid",
    "new_sample_token",
    "BID",
    "bid",
]


def read_metadata(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "metadata.csv"
    if path.exists():
        meta = pd.read_csv(path, index_col=0)
        meta.index = meta.index.astype(str)
        return meta
    if sc is None:
        raise RuntimeError("scanpy is required to read factor_adata.h5ad when metadata.csv is absent")
    adata = sc.read_h5ad(results_dir / "factor_adata.h5ad")
    return adata.obs.copy()


def resolve_mdata_path(results_dir: Path, explicit: str | Path | None = None) -> Path | None:
    """Resolve the path to the active MuData file.

    Checks, in order: an explicit override, ``handoff_manifest.json`` (both
    ``mdata_path`` and ``input_h5mu`` keys), ``config.json[input_h5mu]``, and
    the canonical fallback in ``ACTIVE_PROCESSED_NORMICS_DIR``. Returns the
    first existing path, or ``None``.
    """
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None

    raw: str | None = None
    manifest = results_dir / "handoff_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text())
            raw = data.get("mdata_path") or data.get("input_h5mu")
        except Exception:
            raw = None
    if not raw:
        config = results_dir / "config.json"
        if config.exists():
            try:
                raw = json.loads(config.read_text()).get("input_h5mu")
            except Exception:
                raw = None
    if not raw:
        return None

    raw_path = Path(raw)
    candidates = [
        raw_path,
        results_dir.parent / raw_path,
        Path.cwd() / raw_path,
        ACTIVE_PROCESSED_NORMICS_DIR / "mdata_r14_goettingen_normics_997_ordered.h5mu",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_proteome(results_dir: Path, explicit_mdata_path: str | None = None) -> tuple[pd.DataFrame | None, pd.Series | None]:
    path = resolve_mdata_path(results_dir, explicit_mdata_path)
    if path is None or mu is None:
        return None, None
    mdata = mu.read_h5mu(path)
    proteome = np.log1p(mdata["Proteome"].to_df())
    proteome.index = proteome.index.astype(str)
    gene_symbols = mdata["Proteome"].var["gene_symbol"].reindex(proteome.columns)
    return proteome, gene_symbols


def load_downstream_data(results_dir: str | Path, mdata_path: str | None = None) -> DownstreamData:
    results_dir = Path(results_dir).resolve()
    if sc is None:
        raise RuntimeError("scanpy is required to load factor_adata.h5ad")
    factor_adata = sc.read_h5ad(results_dir / "factor_adata.h5ad")
    metadata = read_metadata(results_dir)
    if "Z.csv" in {p.name for p in results_dir.iterdir()}:
        factors = pd.read_csv(results_dir / "Z.csv", index_col=0)
    else:
        factors = pd.DataFrame(
            factor_adata.obsm["filtered_scores"],
            index=factor_adata.obs_names.astype(str),
            columns=[f"factor_{i + 1}" for i in range(factor_adata.obsm["filtered_scores"].shape[1])],
        )
    factors.index = factors.index.astype(str)
    proteome, gene_symbols = load_proteome(results_dir, mdata_path)
    return DownstreamData(results_dir, factor_adata, metadata, factors, proteome, gene_symbols)


def load_cellline_proteome(frejno_zip: Path) -> pd.DataFrame:
    """CRC65 cell-line full proteome (Frejno EV1A) as cell_lines x gene_symbols (log2 giBAQ)."""
    data = None
    with zipfile.ZipFile(frejno_zip) as outer:
        for name in outer.namelist():
            if (
                name.endswith(".zip")
                and "MOESM3" in name
                and not name.startswith("__MACOSX")
                and "/._" not in name
            ):
                with zipfile.ZipFile(_stdlib_io.BytesIO(outer.read(name))) as inner:
                    xlsx = next(
                        m for m in inner.namelist() if m.endswith(".xlsx") and not m.startswith("__MACOSX")
                    )
                    data = inner.read(xlsx)
                break
    if data is None:
        raise FileNotFoundError(f"Could not find the EV1 workbook (MOESM3) in {frejno_zip}")

    df = pd.read_excel(_stdlib_io.BytesIO(data), sheet_name="Table EV1A", header=2)
    gibaq = [c for c in df.columns if str(c).startswith("giBAQ ")]
    expr = df[gibaq].apply(pd.to_numeric, errors="coerce")
    expr.index = df["Gene names"].astype(str).values
    expr = expr.loc[expr.index.notna() & (expr.index != "nan")]
    expr = expr.loc[~expr.index.duplicated()]
    expr.columns = [c.replace("giBAQ ", "", 1) for c in gibaq]  # cell-line names
    return expr.T  # cell_lines x gene_symbols


def build_sample_id_map(meta: pd.DataFrame) -> pd.DataFrame:
    """Map MuVI/internal sample ids to stable BID-style sample identifiers."""
    bid_col = find_col(meta, BID_CANDIDATES)
    label_col = find_col(meta, ["Proteome:new_sample_label", "new_sample_label"])
    raw_col = find_col(meta, ["Proteome:new_selected_raw_sample", "new_selected_raw_sample", "new_sample"])
    numeric_col = find_col(meta, ["Proteome:new_bid_numeric", "new_bid_numeric"])

    mapping = pd.DataFrame({"analysis_sample_id": meta.index.astype(str)}, index=meta.index.astype(str))
    if bid_col:
        mapping["bid"] = meta[bid_col].astype(str).values
    else:
        mapping["bid"] = mapping["analysis_sample_id"]
    if label_col:
        mapping["sample_label"] = meta[label_col].astype(str).values
    if raw_col:
        mapping["selected_raw_sample"] = meta[raw_col].astype(str).values
    if numeric_col:
        mapping["bid_numeric"] = pd.to_numeric(meta[numeric_col], errors="coerce").values
    else:
        mapping["bid_numeric"] = pd.to_numeric(mapping["bid"].str.extract(r"(\d+)$", expand=False), errors="coerce")

    if mapping["bid"].isna().any() or not mapping["bid"].is_unique:
        raise ValueError("BID identifiers are missing or duplicated; refusing to use them as display ids")
    return mapping


def write_bid_indexed_exports(results_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    """Write human-readable BID-indexed copies without changing MuVI-aligned source files."""
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "bid_indexed"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = read_metadata(results_dir)
    mapping = build_sample_id_map(meta)
    sort_cols = ["bid_numeric", "bid"] if mapping["bid_numeric"].notna().any() else ["bid"]
    ordered = mapping.sort_values(sort_cols, kind="stable")

    mapping.reset_index(drop=True).sort_values(sort_cols, kind="stable").to_csv(results_dir / "sample_id_map.csv", index=False)
    mapping.reset_index(drop=True).sort_values(sort_cols, kind="stable").to_csv(out_dir / "sample_id_map.csv", index=False)

    def convert_table(path: Path, out_name: str) -> None:
        table = pd.read_csv(path, index_col=0)
        table.index = table.index.astype(str)
        shared = ordered.index.intersection(table.index)
        converted = table.reindex(shared).copy()
        converted.insert(0, "analysis_sample_id", shared)
        converted.insert(0, "bid", mapping.loc[shared, "bid"].values)
        converted.to_csv(out_dir / out_name, index=False)

    for path in sorted(results_dir.glob("leiden_cons*.csv")):
        convert_table(path, f"{path.stem}_by_bid.csv")
    for name in ["leiden.csv", "leiden_single_run.csv", "adata_latent_obs.csv", "Z.csv", "latent.csv"]:
        path = results_dir / name
        if path.exists():
            convert_table(path, f"{path.stem}_by_bid.csv")

    metadata_by_bid = meta.reindex(ordered.index).copy()
    metadata_by_bid.insert(0, "analysis_sample_id", ordered["analysis_sample_id"].values)
    metadata_by_bid.insert(0, "bid", ordered["bid"].values)
    metadata_by_bid.to_csv(out_dir / "metadata_by_bid.csv", index=False)
    return out_dir


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]
    for cand in candidates:
        matches = [c for c in df.columns if cand.lower() in c.lower()]
        if matches:
            return matches[0]
    return None


def write_tsv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=index)


def save_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
