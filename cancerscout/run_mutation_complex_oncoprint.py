"""Prepare locked-cohort mutation tables and run ComplexHeatmap oncoPrint:
  1. Single mutations only  → mutation_oncoprint_complex.png
  2. Single + tuples        → mutation_oncoprint_complex_with_tuples.png

Calls come from the Mutation modality used by the locked MuVI run. This matrix
contains broader mutation calls than the former pathogenic-only mutation.h5ad,
so the outputs are labelled as mutations rather than pathogenic variants.
Missing calls remain missing. Rows are prevalence-ordered within single-gene
and co-occurrence groups.

Usage:
    python -m cancerscout.run_mutation_complex_oncoprint [--plots-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mudata as mu
import numpy as np
import pandas as pd

from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    FIGURES_DIR,
    MUTATION_ENRICHMENT_RESULTS_DIR,
    ORDERED_MDATA_H5MU,
    R_DIR,
)
from cancerscout.plotting import run_r_figure

R_SCRIPT = R_DIR / "mutation_oncoprint.R"
TABLE_SINGLE = MUTATION_ENRICHMENT_RESULTS_DIR / "oncoprint_mutation_single.csv"
TABLE_TUPLES = MUTATION_ENRICHMENT_RESULTS_DIR / "oncoprint_mutation_with_tuples.csv"
SOURCE_AUDIT = MUTATION_ENRICHMENT_RESULTS_DIR / "oncoprint_mutation_source_audit.tsv"

TUPLES = [
    ("TP53_APC_cooccurrence", ["TP53", "APC"]),
    ("APC_KRAS_cooccurrence", ["APC", "KRAS"]),
    ("TP53_APC_KRAS_cooccurrence", ["TP53", "APC", "KRAS"]),
]
SINGLE_GENES = ["TP53", "APC", "KRAS", "BRAF", "PTEN", "CTNNB1", "MLH1"]
ANNOTATION_COLS = [
    "leiden_cons",
    "MSI consensus",
    "CPTAC",
    "Age",
    "Sex",
    "UICC",
    "OS",
    "PFS",
    "Mutation burden",
]

OUT_SINGLE = FIGURES_DIR / "cluster_interpretation" / "mutation_oncoprint_complex.png"
OUT_TUPLES = (
    FIGURES_DIR
    / "cluster_interpretation"
    / "mutation_oncoprint_complex_with_tuples.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Render both oncoprints from the saved sample-aligned CSV tables.",
    )
    parser.add_argument(
        "--mutation-h5mu",
        type=Path,
        default=ORDERED_MDATA_H5MU,
        help="Locked MuData containing the Mutation modality.",
    )
    return parser.parse_args()


def load_mutation_calls(
    mutation_h5mu: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mdata = mu.read_h5mu(mutation_h5mu)
    if "Mutation" not in mdata.mod:
        raise KeyError(f"MuData has no Mutation modality: {mutation_h5mu}")
    mutation = mdata["Mutation"]
    if mutation.var_names.duplicated().any():
        raise ValueError("Mutation modality has duplicate feature identifiers")
    values = (
        mutation.X.toarray()
        if hasattr(mutation.X, "toarray")
        else np.asarray(mutation.X)
    )
    finite_values = set(np.unique(values[np.isfinite(values)]).tolist())
    if not finite_values.issubset({0.0, 1.0}):
        raise ValueError(
            "Mutation modality must contain binary 0/1 calls plus optional missing values"
        )
    calls = pd.DataFrame(
        values,
        index=mutation.obs_names.astype(str),
        columns=mutation.var_names.astype(str),
    )
    audit = pd.DataFrame(
        {
            "gene": calls.columns,
            "n_mutated": calls.eq(1).sum(axis=0).astype(int).values,
            "n_wildtype": calls.eq(0).sum(axis=0).astype(int).values,
            "n_missing": calls.isna().sum(axis=0).astype(int).values,
            "source": str(mutation_h5mu),
            "call_semantics": (
                "locked MuVI Mutation modality; broader mutation calls, "
                "not pathogenic-only"
            ),
        }
    )
    return calls, audit


def build_data(
    model_dir: Path,
    mutation_h5mu: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mutation, audit = load_mutation_calls(mutation_h5mu)
    meta = pd.read_csv(model_dir / "metadata.csv", index_col=0)
    meta.index = meta.index.astype(str)
    if set(meta.index) != set(mutation.index):
        missing = sorted(set(meta.index) - set(mutation.index))
        extra = sorted(set(mutation.index) - set(meta.index))
        raise ValueError(
            "Locked metadata and Mutation modality samples differ: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    mutation = mutation.reindex(meta.index)
    missing_genes = sorted(set(SINGLE_GENES) - set(mutation.columns))
    if missing_genes:
        raise KeyError(f"Mutation modality is missing displayed genes: {missing_genes}")

    for col_name, members in TUPLES:
        member_calls = mutation[members]
        mutation[col_name] = member_calls.eq(1).all(axis=1).astype(float)
        mutation.loc[member_calls.isna().any(axis=1), col_name] = np.nan

    df = meta[["leiden_cons", "MSI consensus"]].copy()
    df["CPTAC"] = meta["cptac_prediction"].astype("string")
    df["Age"] = pd.to_numeric(meta["Alter bei Diagnose"], errors="coerce")
    df["Sex"] = meta["Geschlecht"].astype("string").str.lower()
    df["UICC"] = (
        meta["UICC"]
        .astype("string")
        .str.upper()
        .str.extract(r"^(IV|III|II|I)", expand=False)
    )
    df["OS"] = pd.to_numeric(meta["OS Status"], errors="coerce").map(
        {0.0: "censored", 1.0: "event"}
    )
    df["PFS"] = pd.to_numeric(meta["PFS Status"], errors="coerce").map(
        {0.0: "censored", 1.0: "event"}
    )
    base_mutation = mutation.drop(columns=[name for name, _ in TUPLES])
    df["Mutation burden"] = (
        base_mutation.eq(1).sum(axis=1).where(base_mutation.notna().any(axis=1))
    )
    df = df.join(mutation[SINGLE_GENES + [name for name, _ in TUPLES]])
    return df, audit


def apply_row_order(df: pd.DataFrame, mut_cols: list[str]) -> list[str]:
    """Return mut_cols reordered by prevalence (desc), preserving any not in df."""
    prev = df[mut_cols].mean().sort_values(ascending=False)
    return prev.index.tolist()


def write_csv(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    out = df[ANNOTATION_COLS + cols].copy()
    out.to_csv(path)
    print(f"  CSV written: {path} ({len(out)} rows, {len(cols)} mut cols)")


def main() -> None:
    args = parse_args()
    OUT_SINGLE.parent.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        missing = [path for path in (TABLE_SINGLE, TABLE_TUPLES) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing saved oncoprint table(s): "
                + ", ".join(map(str, missing))
                + ". Run once without --plots-only to build them from the locked MuData."
            )
        run_r_figure(R_SCRIPT, [TABLE_SINGLE, OUT_SINGLE])
        run_r_figure(R_SCRIPT, [TABLE_TUPLES, OUT_TUPLES])
        return

    print("Loading mutation data …")
    df, source_audit = build_data(ACTIVE_MODEL_DIR, args.mutation_h5mu)
    TABLE_SINGLE.parent.mkdir(parents=True, exist_ok=True)
    source_audit.to_csv(SOURCE_AUDIT, sep="\t", index=False)

    # --- Single mutations only ---
    single_cols = apply_row_order(df, SINGLE_GENES)
    write_csv(df, single_cols, TABLE_SINGLE)
    print("\nGenerating single-mutation oncoprint …")
    run_r_figure(R_SCRIPT, [TABLE_SINGLE, OUT_SINGLE])

    # --- Single + selected tuples ---
    tuple_cols = apply_row_order(df, [name for name, _ in TUPLES])
    tuple_order_cols = single_cols + tuple_cols
    write_csv(df, tuple_order_cols, TABLE_TUPLES)
    print("\nGenerating single + tuple oncoprint …")
    run_r_figure(R_SCRIPT, [TABLE_TUPLES, OUT_TUPLES])

    print(f"\nDone.\n  {OUT_SINGLE}\n  {OUT_TUPLES}")


if __name__ == "__main__":
    main()
