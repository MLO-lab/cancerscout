#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import mudata as mu
import numpy as np
import pandas as pd

from cancerscout.paths import MDATA_TEMPLATE_H5MU


NOTEBOOK_DIR = Path(__file__).resolve().parent
DEFAULT_MATRIX = NOTEBOOK_DIR / "final_s2_top382_plate_corrected_20260327" / "proteome_matrix.csv"
DEFAULT_SAMPLE_META = NOTEBOOK_DIR / "final_s2_top382_plate_corrected_20260327" / "sample_metadata.csv"
DEFAULT_PROTEIN_ANN = NOTEBOOK_DIR / "new_data_normics_s2_original_min0p5_top382_20260327" / "intermediate" / "protein_annotations.csv"
DEFAULT_TEMPLATE = MDATA_TEMPLATE_H5MU
DEFAULT_OUT = NOTEBOOK_DIR / "final_s2_top382_plate_corrected_20260327" / "mdata_s2_top382_plate_corrected.h5mu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MuData object from the final S2 top382 corrected proteome matrix.")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--sample-meta", type=Path, default=DEFAULT_SAMPLE_META)
    parser.add_argument("--protein-ann", type=Path, default=DEFAULT_PROTEIN_ANN)
    parser.add_argument("--template-h5mu", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def first_nonempty_token(value: object, sep: str = ";") -> str | None:
    if pd.isna(value):
        return None
    for token in str(value).split(sep):
        token = token.strip()
        if token:
            return token
    return None


def prepare_proteome_var(protein_ids: pd.Index, protein_ann: pd.DataFrame, template_var: pd.DataFrame) -> pd.DataFrame:
    ann = protein_ann.copy()
    ann = ann.drop_duplicates(subset=["protein_id"]).set_index("protein_id")
    prot_name_source = "PG.UniProtIds" if "PG.UniProtIds" in ann.columns else "PG.ProteinGroups"
    fallback = pd.DataFrame(index=protein_ids)
    fallback["gene_symbol"] = (
        ann.reindex(protein_ids)["PG.Genes"].map(first_nonempty_token)
        if "PG.Genes" in ann.columns
        else pd.Series(index=protein_ids, dtype="object")
    )
    fallback["prot_name"] = ann.reindex(protein_ids)[prot_name_source].map(lambda x: x if pd.isna(x) else str(x))
    fallback["prot_desc"] = (
        ann.reindex(protein_ids)["PG.ProteinDescriptions"].map(first_nonempty_token)
        if "PG.ProteinDescriptions" in ann.columns
        else pd.Series(index=protein_ids, dtype="object")
    )

    var = template_var.reindex(protein_ids).copy()
    for col in ["gene_symbol", "prot_name", "prot_desc"]:
        if col not in var.columns:
            var[col] = pd.Series(index=protein_ids, dtype="object")
        else:
            var[col] = var[col].astype("object")
        var[col] = var[col].where(var[col].notna(), fallback[col])
    var = var[["gene_symbol", "prot_name", "prot_desc"]].copy()
    for col in var.columns:
        var[col] = var[col].where(var[col].notna(), "")
        var[col] = var[col].astype(str)
    return var


def stringify_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), "")
            out[col] = out[col].astype(str)
    return out


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix, index_col=0)
    sample_meta = pd.read_csv(args.sample_meta)
    protein_ann = pd.read_csv(args.protein_ann)
    template = mu.read_h5mu(args.template_h5mu)

    sample_meta = sample_meta.loc[sample_meta["sample_group"] == "BID"].copy()
    sample_meta["obs_name"] = sample_meta["bid_numeric"].astype(int).astype(str)
    sample_order = sample_meta["sample"].tolist()
    obs_order = sample_meta["obs_name"].tolist()

    matrix = matrix.loc[:, sample_order]
    matrix.columns = obs_order

    template_obs = stringify_object_columns(template.obs.copy())
    missing = sorted(set(obs_order) - set(template_obs.index.astype(str)))
    if missing:
        raise ValueError(f"Template obs is missing sample ids: {missing[:10]}")
    template_obs = template_obs.loc[obs_order].copy()
    template_mut = template["Mutation"][obs_order, :].copy()
    template_mut.obs = stringify_object_columns(template_mut.obs.copy())
    template_mut.var = stringify_object_columns(template_mut.var.copy())

    proteome_var = prepare_proteome_var(
        protein_ids=matrix.index.astype(str),
        protein_ann=protein_ann,
        template_var=template["Proteome"].var.copy(),
    )

    prot_adata = ad.AnnData(
        X=matrix.T.to_numpy(dtype=np.float32),
        obs=template_obs.copy(),
        var=proteome_var.copy(),
        dtype=np.float32,
    )
    prot_adata.obs_names = obs_order
    prot_adata.var_names = matrix.index.astype(str)

    # Keep the new-run sample bookkeeping available without disturbing the legacy obs schema.
    sample_meta_indexed = sample_meta.set_index("obs_name").loc[obs_order]
    for col in sample_meta_indexed.columns:
        if col in {"obs_name"}:
            continue
        prot_adata.obs[f"new_{col}"] = sample_meta_indexed[col].to_numpy()
    prot_adata.obs = stringify_object_columns(prot_adata.obs.copy())
    prot_adata.var = stringify_object_columns(prot_adata.var.copy())

    mdata = mu.MuData({"Proteome": prot_adata, "Mutation": template_mut})
    mdata.obs = stringify_object_columns(template_obs.copy())
    mdata.var = pd.DataFrame(index=mdata.var_names)
    mdata.var["gene_symbol"] = ""
    mdata.update()
    mdata.obs = stringify_object_columns(mdata.obs.copy())
    mdata.var = stringify_object_columns(mdata.var.copy())
    mdata.write_h5mu(args.out)

    manifest = {
        "out": str(args.out),
        "template_h5mu": str(args.template_h5mu),
        "matrix": str(args.matrix),
        "sample_meta": str(args.sample_meta),
        "protein_ann": str(args.protein_ann),
        "n_obs": int(mdata.n_obs),
        "n_vars_total": int(mdata.n_vars),
        "proteome_shape": [int(x) for x in mdata["Proteome"].shape],
        "mutation_shape": [int(x) for x in mdata["Mutation"].shape],
    }
    (args.out.parent / "mdata_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(mdata)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
