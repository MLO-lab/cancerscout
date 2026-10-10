"""Map Frejno CRC pharmacoproteomic drug-response signatures onto patient clusters.

This is an exploratory hypothesis-generation analysis. It uses local Frejno
supplementary tables only and maps elastic-net protein effect sizes onto the
locked six CancerScout proteomic clusters.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mudata as mu
import numpy as np
import pandas as pd

from cancerscout.plot_style import set_house_style
from cancerscout.io import (
    build_sample_id_map,
    read_metadata,
    resolve_mdata_path,
    save_text,
    write_tsv,
)
from cancerscout.plotting import complex_heatmap
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    FIGURES_DIR,
    FREJNO_DRUG_ZIP,
    PHARMACOPROTEOMICS_RESULTS_DIR,
    REPORTS_DIR,
)


EV_ZIP_HINTS = {
    "EV1": "MOESM3",
    "EV2": "MOESM4",
    "EV3": "MOESM5",
    "EV4": "MOESM6",
    "EV5": "MOESM7",
    "EV6": "MOESM8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frejno-zip", type=Path, default=FREJNO_DRUG_ZIP)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--mdata-path", type=Path, default=None)
    parser.add_argument("--cluster-col", default="leiden_cons")
    parser.add_argument("--out-dir", type=Path, default=PHARMACOPROTEOMICS_RESULTS_DIR)
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR / "pharmacoproteomics"
    )
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-N most predicted-sensitive rows per cluster in each panel.",
    )
    parser.add_argument(
        "--min-compounds",
        type=int,
        default=3,
        help="Minimum distinct compounds for a mechanism to be tested.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw pharmacoproteomics figures from saved result tables without rescoring drugs.",
    )
    return parser.parse_args()


def clean_colname(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def open_ev_workbooks(bundle: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    with zipfile.ZipFile(bundle) as outer:
        for name in outer.namelist():
            if not name.endswith(".zip") or name.startswith("__MACOSX"):
                continue
            for ev, hint in EV_ZIP_HINTS.items():
                if hint in name:
                    with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                        xlsx = next(
                            n
                            for n in inner.namelist()
                            if n.endswith(".xlsx") and not n.startswith("__MACOSX")
                        )
                        found[ev] = inner.read(xlsx)
    missing = sorted(set(EV_ZIP_HINTS) - set(found))
    if missing:
        raise FileNotFoundError(f"Could not find EV workbooks in {bundle}: {missing}")
    return found


def read_ev_sheet(
    workbooks: dict[str, bytes], ev: str, sheet: str, header: int = 2
) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(workbooks[ev]), sheet_name=sheet, header=header)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [clean_colname(c) for c in df.columns]
    return df


def schema_summary(workbooks: dict[str, bytes], out_dir: Path) -> pd.DataFrame:
    role = {
        "Table EV1A": "CRC65 full-proteome expression and protein annotations",
        "Table EV1B": "CRC65 Kinobeads kinase/protein expression",
        "Table EV1C": "CPTAC full-proteome expression",
        "Table EV1D": "ComBat-adjusted mRNA expression",
        "Table EV1E": "ComBat-adjusted/imputed full-proteome expression across CRC65/CPTAC",
        "Table EV2A": "FPA/FPB/FPC associations with clinical/molecular labels",
        "Table EV2B": "Full-proteome subtype target/drug enrichment",
        "Table EV2C": "Kinobeads subtype associations",
        "Table EV2D": "MERTK-expression associations",
        "Table EV2E": "CMS prediction confusion/performance",
        "Table EV3A": "Predicted standardized AUC drug sensitivities across CRC65/CPTAC samples",
        "Table EV3B": "Full-proteome elastic-net protein effect sizes per drug model",
        "Table EV3C": "Kinobeads protein effect sizes per drug model",
        "Table EV3D": "Kinase-specific Kinobeads effect sizes per drug model",
        "Table EV4": "Kinases differentially expressed between Kinobeads subtypes",
        "Table EV5": "Cox regression results",
        "Table EV6A": "CRC65/CPTAC sample annotations including Full Proteome subtype",
        "Table EV6B": "Drug annotations, subtype associations, target/activity labels",
    }
    rows = []
    lines = ["# Frejno EV Table Schema Summary", ""]
    for ev, data in workbooks.items():
        xl = pd.ExcelFile(io.BytesIO(data))
        for sheet in xl.sheet_names:
            preview = pd.read_excel(xl, sheet_name=sheet, header=None, nrows=8)
            header_row = (
                2
                if sheet
                not in {
                    "Table EV2A",
                    "Table EV2B",
                    "Table EV2C",
                    "Table EV2D",
                    "Table EV2E",
                    "Table EV4",
                    "Table EV5",
                }
                else 4
            )
            try:
                df = pd.read_excel(
                    io.BytesIO(data), sheet_name=sheet, header=header_row, nrows=5
                )
                cols = [clean_colname(c) for c in df.columns[:15]]
            except Exception:
                cols = [clean_colname(c) for c in preview.iloc[0, :15]]
            rows.append(
                {
                    "ev_table": ev,
                    "sheet_name": sheet,
                    "header_row_used_for_parsing": header_row + 1,
                    "key_columns_preview": "; ".join(cols),
                    "likely_role": role.get(sheet, ""),
                }
            )
            lines.append(f"## {sheet}")
            lines.append(f"- Workbook: {ev}")
            lines.append(f"- Header row used for parsing: {header_row + 1}")
            lines.append(f"- Key columns preview: {', '.join(cols)}")
            lines.append(f"- Likely role: {role.get(sheet, 'supporting table')}")
            lines.append("")
    out = pd.DataFrame(rows)
    write_tsv(out, out_dir / "frejno_ev_schema_summary.tsv")
    save_text(
        "\n".join(lines) + "\n", REPORTS_DIR / "pharmacoproteomics_frejno_schema.md"
    )
    return out


SOURCE_RE = re.compile(r"_(CCLE|CTRP|GDSC|CETUXIMAB)$", flags=re.IGNORECASE)
FIGURE_SUFFIX_RE = re.compile(
    r"(?:_(?:CCLE|CTRP|GDSC|CETUXIMAB)|\[(?:CCLE|CTRP|GDSC|CETUXIMAB)\])$",
    flags=re.IGNORECASE,
)
UNINFORMATIVE_MECHANISMS = {
    "",
    "nan",
    "na",
    "n/a",
    "none",
    "unknown",
    "screening hit",
    "not annotated",
}
# Shared row-label column width so the four drug panels read as one set.
PHARMA_ROW_LABEL_MM = 60.0


def normalize_compound(raw_drug: object) -> tuple[str, str]:
    text = str(raw_drug).strip()
    match = SOURCE_RE.search(text)
    dataset = match.group(1).upper() if match else "unspecified"
    compound = SOURCE_RE.sub("", text).strip()
    return compound, dataset


def normalize_mechanism(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).strip())
    return FIGURE_SUFFIX_RE.sub("", text).strip()


def drug_model_and_mechanism_tables(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = annotations.copy()
    parsed = models["drug"].map(normalize_compound)
    models["compound"] = parsed.map(lambda x: x[0])
    models["source_dataset"] = parsed.map(lambda x: x[1])
    models["compound_key"] = models["compound"].str.casefold()
    # Keep one deterministic display label for case variants of the same
    # compound; the case-insensitive key must also be the grouping key.
    compound_labels = (
        models.groupby("compound_key", as_index=False)["compound"]
        .agg(lambda values: sorted(set(map(str, values)))[0])
        .rename(columns={"compound": "canonical_compound"})
    )
    models = models.merge(compound_labels, on="compound_key", how="left")
    models["compound"] = models["canonical_compound"]
    models = models.drop(columns="canonical_compound")

    exploded = models.assign(
        mechanism=models["target_or_activity"].fillna("").astype(str).str.split(";")
    ).explode("mechanism")
    exploded["mechanism"] = exploded["mechanism"].map(normalize_mechanism)
    exploded = exploded[
        ~exploded["mechanism"].str.casefold().isin(UNINFORMATIVE_MECHANISMS)
    ].copy()
    compound_moa = (
        exploded[["compound_key", "compound", "mechanism", "source_dataset", "drug"]]
        .drop_duplicates()
        .groupby(["compound_key", "compound", "mechanism"], as_index=False)
        .agg(
            source_datasets=(
                "source_dataset",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            drug_models=("drug", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    mechanism_coverage = (
        compound_moa.groupby("mechanism", as_index=False)
        .agg(
            n_unique_compounds=("compound_key", "nunique"),
            annotated_compounds=(
                "compound",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            source_datasets=(
                "source_datasets",
                lambda x: ";".join(
                    sorted({item for values in x for item in str(values).split(";")})
                ),
            ),
        )
        .sort_values(["n_unique_compounds", "mechanism"], ascending=[False, True])
    )
    return models, compound_moa, mechanism_coverage


def collapse_predicted_auc(
    auc: pd.DataFrame,
    models: pd.DataFrame,
    compound_moa: pd.DataFrame,
    included_samples: list[str],
    eligible_mechanisms: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = (
        auc.loc[:, included_samples]
        .rename_axis(index="drug_model", columns="sample_id")
        .reset_index()
        .melt(
            id_vars="drug_model",
            var_name="sample_id",
            value_name="predicted_standardized_auc",
        )
    )
    long["predicted_standardized_auc"] = pd.to_numeric(
        long["predicted_standardized_auc"], errors="coerce"
    )
    long = long.merge(
        models[["drug", "compound", "compound_key", "source_dataset"]],
        left_on="drug_model",
        right_on="drug",
        how="inner",
        validate="many_to_one",
    )
    compound_scores = (
        long.groupby(["sample_id", "compound_key", "compound"], as_index=False)
        .agg(
            predicted_standardized_auc=("predicted_standardized_auc", "median"),
            n_dataset_models=("drug_model", "nunique"),
            source_datasets=(
                "source_dataset",
                lambda x: ";".join(sorted(set(map(str, x)))),
            ),
            drug_models=("drug_model", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(["sample_id", "compound"])
    )

    moa_input = compound_scores.merge(
        compound_moa[["compound_key", "mechanism"]].drop_duplicates(),
        on="compound_key",
        how="inner",
        validate="many_to_many",
    )
    moa_input = moa_input[moa_input["mechanism"].isin(eligible_mechanisms)]
    moa_scores = (
        moa_input.groupby(["sample_id", "mechanism"], as_index=False)
        .agg(
            predicted_standardized_auc=("predicted_standardized_auc", "median"),
            n_compounds_contributing=("compound_key", "nunique"),
        )
        .sort_values(["sample_id", "mechanism"])
    )
    return compound_scores, moa_scores


def load_expression(results_dir: Path, mdata_path: Path | None) -> pd.DataFrame:
    path = mdata_path or resolve_mdata_path(results_dir)
    if path is None:
        raise FileNotFoundError(
            "Could not resolve h5mu path from handoff manifest; pass --mdata-path"
        )
    mdata = mu.read_h5mu(path)
    prot = mdata["Proteome"].to_df().copy()
    prot.index = prot.index.astype(str)
    genes = mdata["Proteome"].var["gene_symbol"].reindex(prot.columns).astype(str)
    prot.columns = genes.str.split(";").str[0].str.upper().values
    prot = prot.loc[:, prot.columns.notna()]
    prot = prot.loc[:, prot.columns.astype(str) != "nan"]
    return prot.T.groupby(level=0).mean().T


def zscore(expr: pd.DataFrame) -> pd.DataFrame:
    mean = expr.mean(axis=0, skipna=True)
    std = expr.std(axis=0, skipna=True).replace(0, np.nan)
    return expr.sub(mean, axis=1).div(std, axis=1)


def load_effect_matrix(workbooks: dict[str, bytes]) -> pd.DataFrame:
    raw = read_ev_sheet(workbooks, "EV3", "Table EV3B", header=2)
    gene_col = raw.columns[0]
    raw = raw.rename(columns={gene_col: "gene_symbol"})
    raw["gene_symbol"] = raw["gene_symbol"].astype(str).str.upper().str.strip()
    raw = raw[raw["gene_symbol"].str.match(r"^[A-Z0-9_.-]+$", na=False)]
    values = (
        raw.set_index("gene_symbol").apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    return values.groupby(level=0).mean()


def load_kinase_effects(workbooks: dict[str, bytes], sheet: str) -> pd.DataFrame:
    raw = read_ev_sheet(workbooks, "EV3", sheet, header=2)
    gene_col = raw.columns[0]
    raw = raw.rename(columns={gene_col: "gene_symbol"})
    raw["gene_symbol"] = raw["gene_symbol"].astype(str).str.upper().str.strip()
    raw = raw[raw["gene_symbol"].str.match(r"^[A-Z0-9_.-]+$", na=False)]
    return (
        raw.set_index("gene_symbol")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .groupby(level=0)
        .mean()
    )


def load_drug_annotations(workbooks: dict[str, bytes]) -> pd.DataFrame:
    ann = read_ev_sheet(workbooks, "EV6", "Table EV6B", header=2)
    ann = ann.rename(
        columns={
            "Drug or Combination": "drug",
            "Alternative name": "alternative_name",
            "Associated with Full Proteome Subtype": "associated_full_proteome_subtype",
            "Target or activity": "target_or_activity",
            "Targets used for Fisher's Exact test": "targets_for_fisher",
        }
    )
    ann["drug"] = ann["drug"].astype(str).str.strip()
    return ann[ann["drug"].ne("nan")].copy()


def load_sample_annotations(workbooks: dict[str, bytes]) -> pd.DataFrame:
    ann = read_ev_sheet(workbooks, "EV6", "Table EV6A", header=2)
    ann = ann.rename(
        columns={
            "Sample name": "sample_name",
            "Full Proteome subtype": "full_proteome_subtype",
        }
    )
    ann["sample_name"] = ann["sample_name"].astype(str)
    return ann


def load_auc_predictions(workbooks: dict[str, bytes]) -> pd.DataFrame:
    raw = read_ev_sheet(workbooks, "EV3", "Table EV3A", header=2)
    drug_col = raw.columns[0]
    raw = raw.rename(columns={drug_col: "drug"})
    raw["drug"] = raw["drug"].astype(str).str.strip()
    auc = raw.set_index("drug").apply(pd.to_numeric, errors="coerce")
    return auc


def score_drugs(
    expr: pd.DataFrame, effects: pd.DataFrame, min_overlap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr_z = zscore(expr)
    common = sorted(set(expr_z.columns) & set(effects.index))
    scores = {}
    cov_rows = []
    for drug in effects.columns:
        weights = pd.to_numeric(effects[drug], errors="coerce").fillna(0)
        model_genes = weights.index[weights.ne(0)].astype(str).tolist()
        overlap = sorted(set(model_genes) & set(common))
        if len(overlap) >= min_overlap:
            w = weights.reindex(overlap).astype(float)
            denom = np.abs(w).sum()
            scores[drug] = (
                expr_z[overlap].mul(w, axis=1).sum(axis=1) / denom if denom else np.nan
            )
        cov_rows.append(
            {
                "drug": drug,
                "n_model_proteins_nonzero": len(model_genes),
                "n_overlapping_proteins": len(overlap),
                "fraction_model_observed": len(overlap) / len(model_genes)
                if model_genes
                else 0,
                "scored": len(overlap) >= min_overlap,
            }
        )
    return pd.DataFrame(scores, index=expr.index), pd.DataFrame(cov_rows)


def cluster_drug_tests(
    scores: pd.DataFrame,
    clusters: pd.Series,
    coverage: pd.DataFrame,
    annotations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    clusters = clusters.reindex(scores.index).astype(str)
    cov = coverage.set_index("drug")
    ann = annotations.set_index("drug")
    for cluster in sorted(
        clusters.dropna().unique(), key=lambda x: (len(str(x)), str(x))
    ):
        in_cluster = clusters.eq(cluster)
        for drug in scores.columns:
            x = pd.to_numeric(scores.loc[in_cluster, drug], errors="coerce").dropna()
            y = pd.to_numeric(scores.loc[~in_cluster, drug], errors="coerce").dropna()
            delta = x.mean() - y.mean() if len(x) >= 3 and len(y) >= 3 else np.nan
            row = {
                "cluster": cluster,
                "drug": drug,
                "n_cluster": len(x),
                "n_rest": len(y),
                "cluster_mean_auc_score": x.mean() if len(x) else np.nan,
                "rest_mean_auc_score": y.mean() if len(y) else np.nan,
                "delta_auc_score_cluster_minus_rest": delta,
                "predicted_direction": "predicted_sensitive"
                if pd.notna(delta) and delta < 0
                else "predicted_resistant"
                if pd.notna(delta) and delta > 0
                else "NA",
            }
            if drug in cov.index:
                row.update(cov.loc[drug].to_dict())
            if drug in ann.index:
                for col in [
                    "alternative_name",
                    "associated_full_proteome_subtype",
                    "target_or_activity",
                    "targets_for_fisher",
                ]:
                    row[col] = ann.loc[drug, col] if col in ann.columns else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Effect size and rank only. These scores are linear functions of the same
    # proteome the clusters were derived from, so a cluster-vs-rest null is not
    # valid here and any p-value or FDR computed from it would be meaningless
    # (it labelled ~70% of drugs "significant"). The ranking is still the
    # quantity of interest: which drug signatures align with each cluster.
    out = out.sort_values(
        ["cluster", "delta_auc_score_cluster_minus_rest"], ascending=[True, True]
    )
    out["sensitivity_rank_in_cluster"] = (
        out.groupby("cluster")["delta_auc_score_cluster_minus_rest"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    out["resistance_rank_in_cluster"] = (
        out.groupby("cluster")["delta_auc_score_cluster_minus_rest"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return out


def cluster_mechanism_tests(
    scores: pd.DataFrame,
    clusters: pd.Series,
    models: pd.DataFrame,
    compound_moa: pd.DataFrame,
    eligible_mechanisms: set[str],
) -> pd.DataFrame:
    """Roll the per-tumour drug scores up to mechanism, then rank per cluster.

    Uses the same compound -> mechanism collapse as the CRC65 analysis, so the
    two are directly comparable. Effect size and rank only, for the reason given
    in cluster_drug_tests.
    """
    _, moa_scores = collapse_predicted_auc(
        scores.T, models, compound_moa, list(scores.index), eligible_mechanisms
    )
    if moa_scores.empty:
        return pd.DataFrame()
    clusters = clusters.reindex(scores.index).astype(str)
    moa_scores = moa_scores.copy()
    moa_scores["cluster"] = moa_scores["sample_id"].map(clusters)
    moa_scores = moa_scores.dropna(subset=["cluster"])

    rows = []
    for cluster in sorted(clusters.dropna().unique(), key=cluster_sort_key):
        in_cluster = moa_scores["cluster"].eq(cluster)
        for mechanism, sub in moa_scores.groupby("mechanism", sort=True):
            x = sub.loc[sub["cluster"].eq(cluster), "predicted_standardized_auc"]
            y = sub.loc[~sub["cluster"].eq(cluster), "predicted_standardized_auc"]
            if len(x) < 3 or len(y) < 3:
                continue
            delta = float(x.mean() - y.mean())
            rows.append(
                {
                    "cluster": cluster,
                    "mechanism": mechanism,
                    "n_cluster": int(len(x)),
                    "n_rest": int(len(y)),
                    "cluster_mean_auc_score": float(x.mean()),
                    "rest_mean_auc_score": float(y.mean()),
                    "delta_auc_score_cluster_minus_rest": delta,
                    "predicted_direction": (
                        "predicted_sensitive" if delta < 0 else "predicted_resistant"
                    ),
                    "n_compounds_contributing": int(
                        sub["n_compounds_contributing"].max()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(
        ["cluster", "delta_auc_score_cluster_minus_rest"], ascending=[True, True]
    )
    out["sensitivity_rank_in_cluster"] = (
        out.groupby("cluster")["delta_auc_score_cluster_minus_rest"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return out


def top_by_effect(
    rankings: pd.DataFrame, index_col: str, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Top-N most predicted-sensitive rows per cluster, as a matrix to plot."""
    if rankings.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    rankings = rankings.copy()
    rankings["cluster"] = rankings["cluster"].astype(str)
    cluster_order = sorted(rankings["cluster"].unique(), key=cluster_sort_key)
    picked = (
        rankings[rankings["delta_auc_score_cluster_minus_rest"] < 0]
        .sort_values(["cluster", "delta_auc_score_cluster_minus_rest"])
        .groupby("cluster", sort=False)
        .head(top_n)
        .drop_duplicates(subset=[index_col])
    )
    matrix = (
        rankings[rankings[index_col].isin(picked[index_col])]
        .pivot_table(
            index=index_col,
            columns="cluster",
            values="delta_auc_score_cluster_minus_rest",
            aggfunc="mean",
        )
        .reindex(index=picked[index_col].tolist(), columns=cluster_order)
    )
    return matrix, picked, cluster_order


def cluster_sort_key(cluster: object) -> tuple[int, object]:
    text = str(cluster)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def prettify_drug_name(drug: object, max_len: int = 48) -> str:
    text = str(drug)
    text = re.sub(r"_(CTRP|GDSC|CCLE)$", r" [\1]", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def plot_outputs(
    drug_rankings: pd.DataFrame,
    mechanism_rankings: pd.DataFrame,
    figures_dir: Path,
    top_n: int,
) -> None:
    """Drug-level and mechanism-level panels, both ranked by effect size.

    Rendered through the same shared renderer as the CRC65 panels so all four
    read as one family. No significance stars: see cluster_drug_tests.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    for rankings, index_col, stem, label in (
        (drug_rankings, "drug", "top_drug_sensitivity_heatmap", "drug"),
        (mechanism_rankings, "mechanism", "top_moa_sensitivity_heatmap", "mechanism"),
    ):
        matrix, picked, _ = top_by_effect(rankings, index_col, top_n)
        if matrix.empty:
            continue
        row_labels = (
            [prettify_drug_name(v) for v in matrix.index]
            if index_col == "drug"
            else [str(v) for v in matrix.index]
        )
        # Plot predicted sensitivity so the shared red-is-high ramp applies:
        # the score is AUC-like, so a lower value means more sensitive.
        complex_heatmap(
            -matrix,
            figures_dir / f"{stem}.png",
            legend_title="Predicted\nsensitivity\n(-\u0394score)",
            row_labels=row_labels,
            row_groups=picked["cluster"].tolist(),
            annotate="value_stars",
            row_label_max_mm=PHARMA_ROW_LABEL_MM,
        )
        print(f"Wrote top-{top_n} {label} panel to {figures_dir / (stem + '.png')}")


def write_readme(
    args: argparse.Namespace,
    schema: pd.DataFrame,
    coverage: pd.DataFrame,
    rankings: pd.DataFrame,
    mechanism_rankings: pd.DataFrame,
) -> None:
    n_scored = int(coverage["scored"].sum()) if not coverage.empty else 0
    n_total = int(coverage.shape[0])

    def top_lines(table: pd.DataFrame, index_col: str) -> list[str]:
        if table.empty:
            return ["- Unavailable."]
        picked = (
            table[table["delta_auc_score_cluster_minus_rest"] < 0]
            .sort_values(["cluster", "delta_auc_score_cluster_minus_rest"])
            .groupby("cluster")
            .head(5)
        )
        out = []
        for cluster, sub in picked.groupby("cluster"):
            hits = "; ".join(
                f"{r[index_col]} ({r.delta_auc_score_cluster_minus_rest:.2f})"
                for r in sub.itertuples(index=False)
            )
            out.append(f"- Cluster {cluster}: {hits}")
        return out

    lines = [
        "# Pharmacoproteomics: drug-signature projection onto the 997 tumours",
        "",
        "Frejno elastic-net drug models (EV3B) are projected onto each tumour's",
        "proteome, giving an AUC-like score per tumour per drug; lower means more",
        "predicted sensitivity. Compounds are then collapsed to mechanisms using the",
        "same EV6B annotation map as the CRC65 analysis.",
        "",
        "**Effect size and rank only, no p-values or FDR.** The projected score is a",
        "linear function of the same proteome the clusters were derived from, so a",
        "cluster-vs-rest null is invalid; an earlier FDR on these tests called ~70% of",
        "drugs significant. Treat this as a ranked candidate list. Independent",
        "significance comes from the CRC65 cell-line analysis, which uses external",
        "samples and a held-out classifier.",
        "",
        f"- Drug models scored: {n_scored} / {n_total}",
        f"- Mechanisms tested: {mechanism_rankings['mechanism'].nunique() if not mechanism_rankings.empty else 0}",
        f"- Clusters covered: {rankings['cluster'].nunique() if not rankings.empty else 0}",
        "",
        "## Top predicted-sensitive drugs per cluster (delta score)",
        "",
        *top_lines(rankings, "drug"),
        "",
        "## Top predicted-sensitive mechanisms per cluster (delta score)",
        "",
        *top_lines(mechanism_rankings, "mechanism"),
        "",
        "## Figures",
        "",
        "- `figures/pharmacoproteomics/top_drug_sensitivity_heatmap.png`",
        "- `figures/pharmacoproteomics/top_moa_sensitivity_heatmap.png`",
    ]
    save_text("\n".join(lines) + "\n", REPORTS_DIR / "pharmacoproteomics_projection.md")


def main() -> None:
    args = parse_args()
    set_house_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        rankings_path = args.out_dir / "cluster_drug_rankings.tsv"
        if not rankings_path.exists():
            raise FileNotFoundError(
                f"Missing saved drug ranking table: {rankings_path}"
            )
        rankings = pd.read_csv(rankings_path, sep="\t")
        mech_path = args.out_dir / "cluster_mechanism_rankings.tsv"
        mechanism_rankings = (
            pd.read_csv(mech_path, sep="\t") if mech_path.exists() else pd.DataFrame()
        )
        plot_outputs(rankings, mechanism_rankings, args.figures_dir, args.top_n)
        print(f"Wrote pharmacoproteomics figures to {args.figures_dir.resolve()}")
        return

    workbooks = open_ev_workbooks(args.frejno_zip)
    schema = schema_summary(workbooks, args.out_dir)
    annotations = load_drug_annotations(workbooks)
    samples = load_sample_annotations(workbooks)
    auc = load_auc_predictions(workbooks)
    effects = load_effect_matrix(workbooks)
    kinase_protein = load_kinase_effects(workbooks, "Table EV3C")
    kinase_specific = load_kinase_effects(workbooks, "Table EV3D")

    write_tsv(annotations, args.out_dir / "frejno_drug_annotations_ev6b.tsv")
    write_tsv(samples, args.out_dir / "frejno_sample_annotations_ev6a.tsv")
    write_tsv(
        pd.DataFrame(
            [
                {
                    "table": "EV3B",
                    "role": "primary full-proteome protein effect sizes",
                    "n_proteins": effects.shape[0],
                    "n_drugs": effects.shape[1],
                },
                {
                    "table": "EV3C",
                    "role": "Kinobeads protein effect sizes, support",
                    "n_proteins": kinase_protein.shape[0],
                    "n_drugs": kinase_protein.shape[1],
                },
                {
                    "table": "EV3D",
                    "role": "Kinobeads kinase-specific effect sizes, support",
                    "n_proteins": kinase_specific.shape[0],
                    "n_drugs": kinase_specific.shape[1],
                },
            ]
        ),
        args.out_dir / "frejno_effect_matrix_summary.tsv",
    )

    expr = load_expression(args.results_dir, args.mdata_path)
    meta = read_metadata(args.results_dir).reindex(expr.index)
    clusters = meta[args.cluster_col].astype(str)
    write_tsv(
        build_sample_id_map(meta).reset_index(drop=True),
        args.out_dir / "sample_id_map.tsv",
    )

    scores, coverage = score_drugs(expr, effects, args.min_overlap)
    scores_out = scores.reset_index().rename(columns={"index": "sample_id"})
    write_tsv(scores_out, args.out_dir / "sample_drug_scores.tsv")
    coverage = coverage.merge(annotations, on="drug", how="left")
    write_tsv(coverage, args.out_dir / "drug_model_coverage.tsv")

    rankings = cluster_drug_tests(scores, clusters, coverage, annotations)
    write_tsv(rankings, args.out_dir / "cluster_drug_rankings.tsv")

    models, compound_moa, mechanism_coverage = drug_model_and_mechanism_tables(
        annotations
    )
    mechanism_coverage["eligible"] = mechanism_coverage["n_unique_compounds"].ge(
        args.min_compounds
    )
    eligible = set(mechanism_coverage.loc[mechanism_coverage["eligible"], "mechanism"])
    mechanism_rankings = cluster_mechanism_tests(
        scores, clusters, models, compound_moa, eligible
    )
    write_tsv(mechanism_coverage, args.out_dir / "mechanism_coverage.tsv")
    write_tsv(mechanism_rankings, args.out_dir / "cluster_mechanism_rankings.tsv")

    plot_outputs(rankings, mechanism_rankings, args.figures_dir, args.top_n)
    write_readme(args, schema, coverage, rankings, mechanism_rankings)
    run_config = {
        "frejno_zip": str(args.frejno_zip),
        "results_dir": str(args.results_dir),
        "mdata_path": str(args.mdata_path) if args.mdata_path else None,
        "cluster_col": args.cluster_col,
        "n_samples": int(expr.shape[0]),
        "n_genes": int(expr.shape[1]),
        "n_drug_models": int(effects.shape[1]),
        "n_scored_drug_models": int(coverage["scored"].sum()),
        "min_overlap": args.min_overlap,
        "n_eligible_mechanisms": len(eligible),
        "minimum_unique_compounds_per_mechanism": args.min_compounds,
        "score_direction": "EV3B models are AUC-like: lower score means predicted sensitivity; higher score means predicted resistance.",
        "statistics": (
            "Effect size and rank only. The projected score is a linear function of "
            "the proteome the clusters were derived from, so a cluster-vs-rest null "
            "is invalid and no p-value or FDR is reported. Independent significance "
            "comes from the CRC65 cell-line analysis."
        ),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    print(f"Wrote pharmacoproteomics outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
