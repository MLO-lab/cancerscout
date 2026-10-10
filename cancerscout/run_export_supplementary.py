"""Export Supplementary Tables 2-5 and Supplementary Data 1-2 for submission.

Reads saved result tables and the locked processed proteome only; nothing is
refit. Every table part becomes one Excel tab laid out like Supplementary
Table 1 (title in row 1, header in row 3, legend as a note under the table), so
it can be moved into the submission workbook as is. The protein matrices are
gzipped CSVs with protein groups as rows and one column per tumor, headed by its
Supplementary Table 1 Tumor ID. README.md and MANIFEST.tsv describe the bundle;
each part's title and legend are defined once, in its builder below.

Usage:
    conda run -n proj python -m cancerscout.run_export_supplementary
    conda run -n proj python -m cancerscout.run_export_supplementary --skip-matrices
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mudata as md
import numpy as np
import pandas as pd
from openpyxl.styles import Font

from cancerscout.io import write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    ACTIVE_PROCESSED_NORMICS_DIR,
    CELLLINE_MOA_RESULTS_DIR,
    CLUSTER_INTERPRETATION_RESULTS_DIR,
    CLUSTER_MARKER_RESULTS_DIR,
    EXTERNAL_TRANSFER_RESULTS_DIR,
    GSEA_RESULTS_DIR,
    MULTICLASS_CLASSIFIER_CV_RESULTS_DIR,
    ORDERED_MDATA_H5MU,
    REPO_ROOT,
    RESULTS_DIR,
)
from cancerscout.plot_style import cluster_display_label

PRE_LIMMA_CSV = (
    ACTIVE_PROCESSED_NORMICS_DIR
    / "intermediate"
    / "normalized_normics_vsn_log2_bid_only_best_rerun.csv"
)
GSEA_DIR = GSEA_RESULTS_DIR / "reactome"
MARKER_POSTHOC_DIR = CLUSTER_MARKER_RESULTS_DIR / "posthoc"
CV_DIR = MULTICLASS_CLASSIFIER_CV_RESULTS_DIR
DA_FDR = 0.05
DA_LOG2FC = 0.5
GSEA_FDR = 0.05
DIRECTION = {
    "predicted_sensitivity": "predicted more sensitive",
    "predicted_resistance": "predicted more resistant",
    "no_difference": "no difference",
}


@dataclass
class Part:
    """One Excel tab or one data file, with the text that describes it."""

    name: str
    title: str
    legend: str
    sources: list[Path]
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "supplementary")
    parser.add_argument(
        "--skip-matrices",
        action="store_true",
        help="Rebuild the tables only; keep the existing Supplementary Data files.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def subgroup(values: pd.Series) -> pd.Series:
    return values.astype(str).map(lambda v: cluster_display_label(v, include_id=True))


def drug_table(df: pd.DataFrame, lead: dict, fdr_col: str, tail: dict) -> pd.DataFrame:
    """Shared ΔAUC columns of the per-target and per-compound tables."""
    tested = df["status"].eq("ok")
    untested = "no (" + df["n_cluster"].astype(str) + " cell lines)"
    stats = {
        "Cell lines in subgroup (n)": df["n_cluster"],
        "Other assigned cell lines (n)": df["n_rest"],
        "Median predicted AUC, subgroup": df["median_auc_cluster"],
        "Median predicted AUC, other lines": df["median_auc_rest"],
        "ΔAUC (subgroup − other lines)": df["delta_auc_cluster_minus_rest"],
        "Direction": df["effect_interpretation"].map(DIRECTION),
        "Mann–Whitney U": df["mann_whitney_u"],
        "p value": df["p_value"],
        "q value (BH)": df[fdr_col],
        "Tested": untested.where(~tested, "yes"),
    }
    return pd.DataFrame({**lead, **stats, **tail})


def table2() -> list[Part]:
    moa_path = CELLLINE_MOA_RESULTS_DIR / "cluster_moa_rankings.tsv"
    compound_path = CELLLINE_MOA_RESULTS_DIR / "cluster_compound_rankings.tsv"
    map_path = CELLLINE_MOA_RESULTS_DIR / "compound_mechanism_map.tsv"
    auc_path = CELLLINE_MOA_RESULTS_DIR / "cellline_moa_auc.tsv"
    assign_path = EXTERNAL_TRANSFER_RESULTS_DIR / "crc65_assignments.tsv"
    order = ["cluster", "q", "p_value", "delta_auc_cluster_minus_rest"]

    moa = read_tsv(moa_path).assign(q=lambda d: d["fdr_bh_global"])
    moa = moa.sort_values(order, na_position="last")
    per_target = drug_table(
        moa,
        {"Subgroup": subgroup(moa["cluster"]), "Target": moa["mechanism"]},
        "fdr_bh_global",
        {
            "Compounds (n)": moa["n_unique_compounds"],
            "Compounds": moa["annotated_compounds"],
            "Source datasets": moa["source_datasets"],
        },
    )
    n_moa_tests = int(moa["p_value"].notna().sum())

    compounds = read_tsv(compound_path).assign(q=lambda d: d["fdr_bh_compound"])
    compounds = compounds.sort_values(order, na_position="last")
    targets = (
        read_tsv(map_path)
        .groupby("compound_key")["mechanism"]
        .agg(lambda s: ";".join(sorted(set(s))))
    )
    per_compound = drug_table(
        compounds,
        {
            "Subgroup": subgroup(compounds["cluster"]),
            "Compound": compounds["compound"],
            "Target(s)": compounds["compound_key"].map(targets),
        },
        "fdr_bh_compound",
        {},
    )
    n_compound_tests = int(compounds["p_value"].notna().sum())

    assign = read_tsv(assign_path)
    with_auc = set(read_tsv(auc_path)["sample_id"].astype(str))
    has_data = assign["sample_id"].astype(str).isin(with_auc)
    known = assign["assignment_status"].eq("known")
    lines = pd.DataFrame(
        {
            "Cell line": assign["sample_id"],
            "Assigned subgroup": subgroup(assign["predicted_cluster_thresholded"]),
            "Assignment": np.where(known, "assigned", "Unknown (below threshold)"),
            "Maximum class probability": assign["max_probability"],
            "Rejection threshold": assign["rejection_threshold"],
            "Nearest subgroup (before rejection)": subgroup(
                assign["predicted_cluster_nearest"]
            ),
            **{f"Probability C{k}": assign[f"prob_C{k}"] for k in range(1, 7)},
            "Predicted drug response available": np.where(has_data, "yes", "no"),
            "Used in subgroup comparisons": np.where(known & has_data, "yes", "no"),
        }
    ).sort_values(["Assigned subgroup", "Cell line"])

    return [
        Part(
            "Table S2a",
            "Supplementary Table 2a. Predicted drug sensitivity of CRC65 cell lines "
            "by proteomic subgroup, per drug target (Fig. 3D)",
            "Predicted standardized AUC values for the CRC65 cell lines (Frejno et al., "
            "Table EV3A; lower values mean greater predicted sensitivity) were "
            "summarized per compound as the median over its dataset-specific drug "
            "models, and per target as the median over its annotated compounds "
            "(Table EV6B); targets with at least three compounds were tested. The "
            "lines assigned to each subgroup were compared with all other assigned "
            "lines by a two-sided Mann–Whitney U test. ΔAUC is the subgroup median "
            "minus the median of the other lines, so negative values mean greater "
            "predicted sensitivity (Fig. 3D colors −ΔAUC). q values are "
            f"Benjamini–Hochberg-adjusted across all {n_moa_tests:,} subgroup–target "
            "tests. C2 has two lines with drug-response data and was not tested; its "
            "ΔAUC is reported without a test. No cell line was assigned to C5 or C6.",
            [moa_path],
            per_target,
        ),
        Part(
            "Table S2b",
            "Supplementary Table 2b. Predicted drug sensitivity of CRC65 cell lines "
            "by proteomic subgroup, per compound (Supplementary Fig. 4)",
            "As Supplementary Table 2a, per compound. Each compound's score is the "
            "median predicted standardized AUC over its dataset-specific drug models; "
            "targets are its Table EV6B annotations, blank where it has none. q values "
            f"are Benjamini–Hochberg-adjusted across all {n_compound_tests:,} "
            "subgroup–compound tests, a family separate from Supplementary Table 2a.",
            [compound_path, map_path],
            per_compound,
        ),
        Part(
            "Table S2c",
            "Supplementary Table 2c. Proteomic subgroup assignment of the CRC65 cell "
            "lines",
            "Cell lines were assigned with the consensus classifier trained on all 997 "
            "discovery tumors. Lines whose maximum class probability fell below the "
            "calibrated rejection threshold are labelled Unknown and were excluded "
            "from the drug comparisons; the nearest subgroup is the highest-probability "
            "class before rejection. Assigned lines with predicted drug-response data "
            "entered Supplementary Tables 2a and 2b.",
            [assign_path, auc_path],
            lines,
        ),
    ]


def table3(annotation: pd.DataFrame) -> list[Part]:
    path = CLUSTER_INTERPRETATION_RESULTS_DIR / "volcano_marker_rankings.tsv"
    da = read_tsv(path)
    significant = da["FDR"] < DA_FDR
    call = np.select(
        [
            significant & (da["mean_diff"] > DA_LOG2FC),
            significant & (da["mean_diff"] < -DA_LOG2FC),
        ],
        ["up", "down"],
        "not called",
    )
    frame = pd.DataFrame(
        {
            "Subgroup": subgroup(da["cluster"]),
            "UniProt protein group": da["protein"],
            "Gene": da["protein"].map(annotation["gene_name"]),
            "Protein description": da["protein"].map(annotation["protein_description"]),
            "Mean log2 abundance, subgroup": da["mean_in"],
            "Mean log2 abundance, other tumors": da["mean_out"],
            "log2 fold change": da["mean_diff"],
            "Welch t": da["t_stat"],
            "p value": da["p_value"],
            "q value (BH, within subgroup)": da["FDR"],
            "Call (q < 0.05, |log2 FC| > 0.5)": call,
        }
    )
    n_proteins = da["protein"].nunique()
    return [
        Part(
            "Table S3",
            "Supplementary Table 3. Differentially abundant proteins, each proteomic "
            "subgroup versus all other tumors (Fig. 4A, Supplementary Fig. 7)",
            f"For each subgroup, every protein group (n = {n_proteins:,}) was compared "
            "with the remaining tumors by a two-sided Welch t-test on the Normics/VSN "
            "log2, limma plate-corrected abundances, omitting missing values. The log2 "
            "fold change is the difference of the two group means, and q values are "
            "Benjamini–Hochberg-adjusted within each subgroup. Protein groups with "
            "q < 0.05 and |log2 fold change| > 0.5 are called up or down.",
            [path, ORDERED_MDATA_H5MU],
            frame,
        )
    ]


def table4() -> list[Part]:
    all_path = GSEA_DIR / "gsea_prerank_all_terms.tsv"
    redundancy_path = GSEA_DIR / "gsea_significant_terms_with_redundancy.tsv"
    redundancy = read_tsv(redundancy_path)[
        ["cluster", "Term", "redundant", "superseded_by"]
    ]
    gsea = read_tsv(all_path).merge(
        redundancy, on=["cluster", "Term"], how="left", validate="one_to_one"
    )
    gsea = gsea.assign(abs_nes=gsea["NES"].abs()).sort_values(
        ["cluster", "FDR q-val", "abs_nes"], ascending=[True, True, False]
    )
    significant = gsea["FDR q-val"] <= GSEA_FDR
    redundant = gsea["redundant"].eq(True)
    frame = pd.DataFrame(
        {
            "Subgroup": subgroup(gsea["cluster"]),
            "Reactome pathway": gsea["Term"],
            "ES": gsea["ES"],
            "NES": gsea["NES"],
            "Nominal p": gsea["NOM p-val"],
            "FDR q": gsea["FDR q-val"],
            "FWER p": gsea["FWER p-val"],
            "Significant (FDR q ≤ 0.05)": np.where(significant, "yes", "no"),
            "Non-redundant": np.where(
                significant, np.where(redundant, "no", "yes"), ""
            ),
            "Redundant with": gsea["superseded_by"].fillna(""),
            "No permutation with observed sign (NES = ±1, p = 1)": np.where(
                (gsea["abs_nes"] == 1) & (gsea["NOM p-val"] == 1), "yes", "no"
            ),
            "Leading edge (genes in edge / set size)": gsea["Tag %"],
            "Leading-edge genes": gsea["Lead_genes"],
        }
    )
    n_terms = int(gsea.groupby("cluster").size().max())
    return [
        Part(
            "Table S4",
            "Supplementary Table 4. Reactome gene-set enrichment, each proteomic "
            "subgroup versus all other tumors (Fig. 4B)",
            "Preranked GSEA (gseapy 1.0.5; Reactome, MSigDB v7.5.1; gene sets of "
            "15–500 measured genes; 1,000 permutations) on the Welch t statistic of "
            f"each subgroup versus all other tumors, for all {n_terms:,} tested "
            "pathways per subgroup. Pathways with FDR q ≤ 0.05 are significant. Within "
            "a subgroup, a significant pathway whose genes overlap a same-direction "
            "pathway of larger |NES| (one-sided Fisher test, Benjamini–Hochberg FDR ≤ "
            "0.05) is marked redundant with that pathway. Fig. 4B shows, per subgroup, "
            "the two non-redundant significant pathways with the largest positive and "
            "negative NES. Where NES is exactly ±1 with nominal p = 1, no permutation "
            "had the observed sign; Fig. 4B draws these cells grey.",
            [all_path, redundancy_path],
            frame,
        )
    ]


def table5() -> list[Part]:
    signature_path = MARKER_POSTHOC_DIR / "supplement_cluster_marker_signatures.tsv"
    order_path = MARKER_POSTHOC_DIR / "reduced_panel_protein_order.tsv"
    panel_path = MARKER_POSTHOC_DIR / "reduced_panel_summary.tsv"
    per_class_path = CV_DIR / "per_class_metrics_cv.tsv"
    ovr_path = CV_DIR / "ovr_curve_metrics_cv.tsv"
    ece_path = CV_DIR / "calibration_per_class_ece_uniform_cv.tsv"
    overall_path = CV_DIR / "metrics.tsv"
    confusion_path = CV_DIR / "confusion_matrix_cv.tsv"

    markers = (
        read_tsv(signature_path)
        .drop(columns="rank")
        .merge(
            read_tsv(order_path),
            on=["cluster", "gene_symbol"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values(["cluster", "rank"])
    )
    rankings = pd.DataFrame(
        {
            "Subgroup": subgroup(markers["cluster"]),
            "Rank (Fig. 5A order)": markers["rank"],
            "Gene": markers["gene_symbol"],
            "UniProt protein group(s)": markers["source_protein_ids"],
            "Direction": markers["marker_direction"].map(
                {
                    "higher_in_cluster": "higher in subgroup",
                    "lower_in_cluster": "lower in subgroup",
                }
            ),
            "Mean |SHAP|, training + validation": markers["mean_abs_shap_train"],
            "RFECV selection frequency": markers["selection_frequency"],
            "Mean RFECV rank": markers["mean_rfecv_rank"],
            "Mean outer-fold selection fraction": markers[
                "mean_outer_fold_selection_fraction"
            ],
            "Single-protein AUC, test split": markers["single_feature_auc_test"],
            "Mean z, subgroup − rest (training)": markers["train_cluster_minus_rest_z"],
        }
    )

    panels = read_tsv(panel_path).sort_values(["cluster", "panel_size_used"])
    panel_table = pd.DataFrame(
        {
            "Subgroup": subgroup(panels["cluster"]),
            "Panel size (proteins)": panels["panel_size_used"],
            "Test ROC AUC": panels["roc_auc"],
            "Test average precision": panels["average_precision"],
            "Test balanced accuracy": panels["balanced_accuracy"],
            "Test F1": panels["f1"],
            "Full signature size (proteins)": panels["full_signature_size"],
            "Full-signature test ROC AUC": panels["full_roc_auc"],
        }
    )

    ovr = read_tsv(ovr_path).astype({"cluster": str})
    per_class = (
        read_tsv(per_class_path)
        .astype({"cluster": str})
        .merge(ovr.drop(columns="support"), on="cluster", validate="one_to_one")
        .merge(
            read_tsv(ece_path).astype({"cluster": str}).drop(columns="support"),
            on="cluster",
            validate="one_to_one",
        )
    )
    per_class_table = pd.DataFrame(
        {
            "Subgroup": subgroup(per_class["cluster"]),
            "Tumors (n)": per_class["support"],
            "Precision": per_class["precision"],
            "Recall": per_class["recall"],
            "F1": per_class["f1"],
            "Balanced accuracy (one-vs-rest)": per_class["balanced_accuracy_ovr"],
            "AUROC": per_class["auroc"],
            "AUROC 95% CI lower": per_class["auroc_ci_low"],
            "AUROC 95% CI upper": per_class["auroc_ci_high"],
            "Average precision": per_class["aupr"],
            "Average precision 95% CI lower": per_class["aupr_ci_low"],
            "Average precision 95% CI upper": per_class["aupr_ci_high"],
            "Prevalence": per_class["prevalence"],
            "Expected calibration error (10 equal-width bins)": per_class["ece"],
        }
    )

    overall = read_tsv(overall_path).iloc[0]
    macro = ovr.set_index("cluster").loc["macro"]
    overall_table = pd.DataFrame(
        {
            "Evaluation": ["Stratified 5-fold cross-validation, out-of-fold"],
            "Tumors (n)": [overall["n"]],
            "Accuracy": [overall["accuracy"]],
            "Balanced accuracy": [overall["balanced_accuracy"]],
            "Macro F1": [overall["macro_f1"]],
            "Weighted F1": [overall["weighted_f1"]],
            "Macro AUROC": [macro["auroc"]],
            "Macro AUROC 95% CI lower": [macro["auroc_ci_low"]],
            "Macro AUROC 95% CI upper": [macro["auroc_ci_high"]],
            "Macro average precision": [macro["aupr"]],
            "Macro average precision 95% CI lower": [macro["aupr_ci_low"]],
            "Macro average precision 95% CI upper": [macro["aupr_ci_high"]],
        }
    )

    confusion = pd.read_csv(confusion_path, sep="\t", index_col=0)
    relabel = lambda key: cluster_display_label(key.split("_")[1], include_id=True)
    confusion.index = [relabel(k) for k in confusion.index]
    confusion.columns = [relabel(k) for k in confusion.columns]
    confusion = confusion.rename_axis("True subgroup \\ predicted subgroup")

    cv_legend = (
        "Stratified five-fold cross-validation over all 997 tumors; each tumor "
        "was predicted by the consensus ensemble of 100 XGBoost models fitted "
        "without it."
    )
    return [
        Part(
            "Table S5a",
            "Supplementary Table 5a. Ranked subgroup-discriminating proteins from "
            "recursive feature elimination (Fig. 5A)",
            "For each subgroup versus all other tumors, recursive feature elimination "
            "with cross-validation (XGBoost; 10 repeats of 5 outer folds on the "
            "597-tumor training split, starting from the 1,000 protein features left "
            "after missingness and variance filtering) selected the signature listed "
            "here. Proteins are ranked as in Fig. 5A, by mean |SHAP| of the "
            "full-signature model on the training and validation tumors, so the test "
            "split plays no part in the order. Selection frequency is the fraction of "
            "the 10 repeats that selected the protein. The single-protein AUC is that "
            "of the protein's own abundance on the 200-tumor test split, not oriented "
            "by direction, so values below 0.5 mean lower abundance in the subgroup; "
            "the refitted single-protein panels of Fig. 5A are in Supplementary "
            "Table 5b.",
            [signature_path, order_path],
            rankings,
        ),
        Part(
            "Table S5b",
            "Supplementary Table 5b. One-versus-rest test performance by marker-panel "
            "size (Fig. 5A)",
            "Nested panels of the top-ranked proteins from Supplementary Table 5a were "
            "refitted on the training and validation tumors (n = 797) and scored on "
            "the held-out test split (n = 200); the largest panel of each subgroup is "
            "its full signature.",
            [panel_path],
            panel_table,
        ),
        Part(
            "Table S5c",
            "Supplementary Table 5c. Six-way subgroup classification, per-subgroup "
            "out-of-fold performance (Fig. 5B, Extended Data Fig. 8)",
            f"{cv_legend} AUROC and average precision are one-versus-rest, with "
            "class-stratified bootstrap 95% confidence intervals (2,000 replicates). "
            "Expected calibration error uses ten equal-width probability bins "
            "(Extended Data Fig. 8C).",
            [per_class_path, ovr_path, ece_path],
            per_class_table,
        ),
        Part(
            "Table S5d",
            "Supplementary Table 5d. Six-way subgroup classification, overall "
            "out-of-fold performance (Fig. 5B)",
            f"{cv_legend} Macro AUROC and macro average precision are the means of the "
            "six one-versus-rest values, with class-stratified bootstrap 95% "
            "confidence intervals.",
            [overall_path, ovr_path],
            overall_table,
        ),
        Part(
            "Table S5e",
            "Supplementary Table 5e. Six-way subgroup classification, out-of-fold "
            "confusion matrix (Fig. 5B)",
            f"{cv_legend} Rows are the consensus subgroup labels and columns the "
            "out-of-fold predicted subgroups; cells count tumors.",
            [confusion_path],
            confusion.reset_index(),
        ),
    ]


def load_proteome() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locked limma-corrected matrix (protein groups × Tumor ID) and its annotation.

    Supplementary Table 1's Tumor ID n is BIDn, so the BID's number is the ID.
    """
    prot = md.read_h5mu(ORDERED_MDATA_H5MU)["Proteome"]
    values = pd.DataFrame(
        np.asarray(prot.X, dtype=float).T,
        index=prot.var_names,
        columns=prot.obs["new_bid"].astype(str).str[3:].astype(int).values,
    ).sort_index(axis=1)
    annotation = pd.DataFrame(
        {
            "gene_name": prot.var["gene_symbol"].astype(str),
            "protein_description": prot.var["prot_desc"].astype(str),
        },
        index=prot.var_names,
    )
    return values, annotation


def load_pre_limma(proteins: pd.Index, tumors: pd.Index) -> pd.DataFrame:
    """The Normics/VSN matrix that limma corrected, relabelled by Tumor ID."""
    raw = pd.read_csv(PRE_LIMMA_CSV, index_col=0)
    raw.columns = [int(re.search(r"BID(\d+)", c).group(1)) for c in raw.columns]
    if (
        raw.columns.duplicated().any()
        or set(raw.columns) != set(tumors)
        or set(raw.index) != set(proteins)
    ):
        raise ValueError(f"{PRE_LIMMA_CSV} doesn't cover the locked tumors/proteins")
    return raw.loc[proteins, tumors]


def matrix_table(values: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat([annotation, values], axis=1)
        .rename_axis("uniprot_protein_group")
        .reset_index()
    )


def data_parts(values, pre_limma, annotation) -> list[Part]:
    layout = (
        "Rows are the {n:,} protein groups (UniProt accessions; groups with several "
        "accessions are separated by semicolons) with gene name and protein "
        "description; columns are tumors, headed by their Tumor ID in Supplementary "
        "Table 1. Values are rounded to four decimals; empty fields are missing "
        "values."
    ).format(n=len(values))
    return [
        Part(
            "Supplementary_Data_1_proteome_log2_limma_corrected.csv.gz",
            "Supplementary Data 1. Normalized, plate-corrected log2 protein "
            f"abundances of the {values.shape[1]:,} discovery tumors",
            "Normics/VSN log2 abundances after limma correction for acquisition "
            "plate, as used for every analysis in this study. " + layout,
            [ORDERED_MDATA_H5MU],
            matrix_table(values, annotation),
        ),
        Part(
            "Supplementary_Data_2_proteome_log2_normics_vsn_pre_limma.csv.gz",
            "Supplementary Data 2. Normics/VSN log2 protein abundances before plate "
            "correction",
            "The matrix of Supplementary Data 1 before limma plate correction: "
            "Normics/VSN normalization fitted on all acquired runs, restricted to the "
            "same tumors. " + layout,
            [PRE_LIMMA_CSV],
            matrix_table(pre_limma, annotation),
        ),
    ]


def write_workbook(path: Path, parts: list[Part]) -> None:
    """One tab per part in the Supplementary Table 1 layout: bold title in row 1,
    header in row 3, and the legend as a note one blank row under the table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for part in parts:
            part.frame.to_excel(writer, sheet_name=part.name, index=False, startrow=2)
            sheet = writer.sheets[part.name]
            sheet["A1"] = f"{part.title}."
            sheet["A1"].font = Font(bold=True)
            sheet.cell(row=len(part.frame) + 5, column=1, value=part.legend)
            sheet.freeze_panes = "A4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    # relpath, not resolve(): data/ may be a symlink to storage outside the repo.
    return Path(os.path.relpath(path, REPO_ROOT)).as_posix()


def write_readme(out_dir: Path, written: list[tuple[Path, Part]]) -> None:
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # A release archive is not a git checkout.
    commit = git.stdout.strip() if git.returncode == 0 else "unknown"
    lines = [
        "# CancerScout Supplementary Tables 2–5 and Supplementary Data",
        "",
        f"Generated {date.today().isoformat()} from commit `{commit}` by "
        "`python -m cancerscout.run_export_supplementary`. Every table is read from "
        "saved pipeline results and nothing was refit; subgroups are the locked "
        f"six-subgroup solution of `{repo_path(ACTIVE_MODEL_DIR)}`. Supplementary "
        "Table 1 (clinical data) is assembled separately.",
        "",
        "Each Excel tab follows the Supplementary Table 1 layout (title in row 1, "
        "column headers in row 3, legend as a note under the table), so it can be "
        "moved into the submission workbook as is. `MANIFEST.tsv` lists every file "
        "with its size and SHA-256.",
        "",
    ]
    section = None
    for path, part in written:
        heading = (
            "Supplementary Tables" if path.suffix == ".xlsx" else "Supplementary Data"
        )
        if heading != section:
            lines += [f"## {heading}", ""]
            section = heading
        location = f"`{path.relative_to(out_dir)}`"
        if path.suffix == ".xlsx":
            location += f", tab `{part.name}`"
        rows, cols = part.frame.shape
        lines += [
            f"### {part.title}",
            "",
            part.legend,
            "",
            f"- File: {location} ({rows:,} rows × {cols:,} columns)",
            "- Source: " + ", ".join(f"`{repo_path(s)}`" for s in part.sources),
            "",
        ]
    (out_dir / "README.md").write_text("\n".join(lines))


def write_manifest(out_dir: Path, written: list[tuple[Path, Part]]) -> None:
    digests = {path: sha256(path) for path in {path for path, _ in written}}
    write_tsv(
        pd.DataFrame(
            [
                {
                    "file": str(path.relative_to(out_dir)),
                    "tab": part.name if path.suffix == ".xlsx" else "",
                    "rows": part.frame.shape[0],
                    "columns": part.frame.shape[1],
                    "bytes": path.stat().st_size,
                    "sha256": digests[path],
                    "sources": ";".join(repo_path(s) for s in part.sources),
                }
                for path, part in written
            ]
        ),
        out_dir / "MANIFEST.tsv",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    values, annotation = load_proteome()

    workbooks = {
        "Supplementary_Table_2_drug_response.xlsx": table2(),
        "Supplementary_Table_3_differential_abundance.xlsx": table3(annotation),
        "Supplementary_Table_4_gsea_reactome.xlsx": table4(),
        "Supplementary_Table_5_markers_classifier.xlsx": table5(),
    }
    written: list[tuple[Path, Part]] = []
    for name, parts in workbooks.items():
        path = out_dir / "tables" / name
        write_workbook(path, parts)
        written += [(path, part) for part in parts]
        print(f"Wrote {path}")

    pre_limma = load_pre_limma(values.index, values.columns)
    for part in data_parts(values, pre_limma, annotation):
        path = out_dir / "data" / part.name
        if not (args.skip_matrices and path.exists() and path.suffix == ".gz"):
            path.parent.mkdir(parents=True, exist_ok=True)
            part.frame.to_csv(path, index=False, float_format="%.4f")
            print(f"Wrote {path}")
        written.append((path, part))

    write_readme(out_dir, written)
    write_manifest(out_dir, written)
    print(f"Wrote {out_dir / 'README.md'} and {out_dir / 'MANIFEST.tsv'}")


if __name__ == "__main__":
    main()
