"""Apply one fitted CancerScout classifier to CPTAC and CRC65.

The discovery labels and MuVI model remain locked. This stage loads the
997-sample proteome, fixes one feature list shared with both external cohorts,
trains one ConsensusClassifier, and predicts a concatenated CPTAC/CRC65 test
matrix. Both cohorts therefore use the same fitted classifier rather than
separate target-specific fits.

External labels retain their raw nearest CancerScout state and probability
vector. A discovery-only calibration split supplies one common maximum-
probability rejection threshold for CPTAC and CRC65; rejected samples receive
an explicit ``Unknown`` thresholded assignment.

Usage:
    conda run -n proj python -m cancerscout.run_external_cluster_transfer
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from cclf import ConsensusClassifier
from sklearn.metrics import adjusted_rand_score

from cancerscout.io import load_cellline_proteome, write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CPTAC_METADATA_CSV,
    CPTAC_PROTEOME_CSV,
    EXTERNAL_TRANSFER_RESULTS_DIR,
    FIGURES_DIR,
    FREJNO_DRUG_ZIP,
    MODEL_SPLITS_DIR,
)
from cancerscout.plot_style import (
    font_pt,
    CLUSTER_COLORS,
    CLUSTER_SHORT_NAMES,
    savefig,
    set_house_style,
)
from cancerscout.run_cptac_classification import (
    feature_zscore,
    load_current_proteome,
    matrix_diagnostics,
    resolve_mdata_path,
)
from cancerscout.run_cluster_marker_rfecv import load_splits


CLUSTER_IDS = [str(i) for i in range(1, 7)]
UNKNOWN_CLUSTER = "Unknown"
OUTPUT_CLUSTER_IDS = CLUSTER_IDS + [UNKNOWN_CLUSTER]
CMS_ORDER = ["CMS1", "CMS2", "CMS3", "CMS4"]
UMS_ORDER = ["CIN", "MSI", "Mesenchymal"]
DEFAULT_FIGURES_DIR = FIGURES_DIR / "external_cluster_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--mdata-path", type=Path, default=None)
    parser.add_argument("--cptac-prot", type=Path, default=CPTAC_PROTEOME_CSV)
    parser.add_argument("--cptac-meta", type=Path, default=CPTAC_METADATA_CSV)
    parser.add_argument("--frejno-zip", type=Path, default=FREJNO_DRUG_ZIP)
    parser.add_argument("--cluster-column", default="leiden_cons")
    parser.add_argument("--out-dir", type=Path, default=EXTERNAL_TRANSFER_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument("--n-features-per-model", default="auto")
    parser.add_argument("--seed", type=int, default=2605161932)
    parser.add_argument(
        "--use-ray", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Redraw the CPTAC summary from saved unified assignments.",
    )
    return parser.parse_args()


def cluster_label(cluster: str) -> str:
    if str(cluster) == UNKNOWN_CLUSTER:
        return UNKNOWN_CLUSTER
    return f"C{cluster} {CLUSTER_SHORT_NAMES.get(str(cluster), '')}"


def ordered_shared_features(*matrices: pd.DataFrame) -> list[str]:
    """Return a deterministic feature intersection for all supplied matrices."""
    if not matrices:
        return []
    shared = set(matrices[0].columns.astype(str))
    for matrix in matrices[1:]:
        shared &= set(matrix.columns.astype(str))
    return sorted(shared)


def unique_feature_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    out.columns = out.columns.astype(str)
    return out.loc[:, ~out.columns.duplicated()].copy()


def feature_manifest(
    features: list[str],
    train: pd.DataFrame,
    cptac: pd.DataFrame,
    crc65: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for feature in features:
        rows.append(
            {
                "gene_symbol": feature,
                "discovery_finite_fraction": float(
                    np.isfinite(pd.to_numeric(train[feature], errors="coerce")).mean()
                ),
                "cptac_finite_fraction": float(
                    np.isfinite(pd.to_numeric(cptac[feature], errors="coerce")).mean()
                ),
                "crc65_finite_fraction": float(
                    np.isfinite(pd.to_numeric(crc65[feature], errors="coerce")).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def sample_audit(
    cptac_prot_ids: pd.Index,
    cptac_meta_ids: pd.Index,
    crc65_ids: pd.Index,
) -> pd.DataFrame:
    rows = []
    prot = set(cptac_prot_ids.astype(str))
    meta = set(cptac_meta_ids.astype(str))
    for sample_id in sorted(prot | meta):
        in_prot = sample_id in prot
        in_meta = sample_id in meta
        rows.append(
            {
                "cohort": "CPTAC",
                "sample_id": sample_id,
                "in_proteome": in_prot,
                "in_metadata": in_meta,
                "included": in_prot and in_meta,
                "reason": "included"
                if in_prot and in_meta
                else "missing_metadata"
                if in_prot
                else "missing_proteome",
            }
        )
    rows.extend(
        {
            "cohort": "CRC65",
            "sample_id": sample_id,
            "in_proteome": True,
            "in_metadata": False,
            "included": True,
            "reason": "included",
        }
        for sample_id in sorted(crc65_ids.astype(str))
    )
    return pd.DataFrame(rows)


def normalized_probabilities(
    probabilities: pd.DataFrame, expected_index: pd.Index
) -> pd.DataFrame:
    out = probabilities.copy()
    out.index = expected_index.astype(str)
    out.columns = out.columns.astype(str)
    return out.reindex(columns=CLUSTER_IDS, fill_value=0.0)


def calibrate_rejection_threshold(
    probabilities: pd.DataFrame,
    truth: pd.Series,
    target_accuracy: float = 0.90,
    minimum_coverage: float = 0.50,
) -> tuple[float, pd.DataFrame]:
    """Choose the lowest max-probability cutoff meeting both policy targets."""
    truth = truth.reindex(probabilities.index).astype(str)
    nearest = probabilities.idxmax(axis=1).astype(str)
    maximum = probabilities.max(axis=1).astype(float)
    correct = nearest.eq(truth)
    candidates = sorted({0.0, 1.0, *maximum.dropna().tolist()})
    rows = []
    for threshold in candidates:
        retained = maximum.ge(threshold)
        n_retained = int(retained.sum())
        coverage = n_retained / len(maximum) if len(maximum) else np.nan
        retained_accuracy = (
            float(correct.loc[retained].mean()) if n_retained else np.nan
        )
        rows.append(
            {
                "threshold": float(threshold),
                "n_calibration": len(maximum),
                "n_retained": n_retained,
                "n_rejected": int(len(maximum) - n_retained),
                "coverage": coverage,
                "retained_accuracy": retained_accuracy,
                "meets_accuracy": bool(
                    np.isfinite(retained_accuracy)
                    and retained_accuracy >= target_accuracy
                ),
                "meets_coverage": bool(
                    np.isfinite(coverage) and coverage >= minimum_coverage
                ),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["meets_accuracy"] & table["meets_coverage"]]
    if eligible.empty:
        raise ValueError(
            "No external-transfer rejection threshold achieves "
            f"retained accuracy >= {target_accuracy:.2f} and coverage >= "
            f"{minimum_coverage:.2f} on the calibration split"
        )
    return float(eligible.iloc[0]["threshold"]), table


def apply_rejection(assignments: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Add thresholded assignments without modifying the raw nearest state."""
    out = assignments.copy()
    accepted = out["max_probability"].astype(float).ge(threshold)
    out["predicted_cluster_nearest"] = out["predicted_cluster"].astype(str)
    out["predicted_cluster_thresholded"] = np.where(
        accepted, out["predicted_cluster"].astype(str), UNKNOWN_CLUSTER
    )
    out["predicted_cluster_thresholded_label"] = out[
        "predicted_cluster_thresholded"
    ].map(cluster_label)
    out["assignment_status"] = np.where(accepted, "known", "rejected")
    out["rejection_threshold"] = float(threshold)
    return out


def assignments_for_cohort(
    predictions: pd.Series,
    probabilities: pd.DataFrame,
    cohort: str,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    assignments = pd.DataFrame(
        {
            "cohort": cohort,
            "sample_id": predictions.index.astype(str),
            "predicted_cluster": predictions.astype(str).values,
        }
    )
    assignments["predicted_cluster_label"] = assignments["predicted_cluster"].map(
        cluster_label
    )
    assignments["max_probability"] = probabilities.max(axis=1).to_numpy()
    for cluster in CLUSTER_IDS:
        assignments[f"prob_C{cluster}"] = probabilities[cluster].to_numpy()
    if metadata is not None:
        metadata = metadata.copy()
        metadata.index = metadata.index.astype(str)
        for column in [
            "CMS",
            "UMS",
            "Stage",
            "Vital.Status",
            "MSI_PCR_Result",
        ]:
            if column in metadata:
                assignments[column] = (
                    metadata[column].reindex(assignments["sample_id"]).to_numpy()
                )
    return assignments


def distribution_table(
    assignments: pd.DataFrame,
    cluster_column: str = "predicted_cluster_thresholded",
) -> pd.DataFrame:
    rows = []
    for cohort, cohort_df in assignments.groupby("cohort", sort=False):
        counts = cohort_df[cluster_column].astype(str).value_counts()
        for cluster in OUTPUT_CLUSTER_IDS:
            n = int(counts.get(cluster, 0))
            rows.append(
                {
                    "cohort": cohort,
                    "cluster": cluster,
                    "cluster_label": cluster_label(cluster),
                    "n": n,
                    "fraction": n / len(cohort_df) if len(cohort_df) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def concordance_tables(
    assignments: pd.DataFrame,
    label_column: str,
    label_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment_column = "predicted_cluster_thresholded"
    valid = assignments[[assignment_column, label_column]].dropna().copy()
    valid[label_column] = valid[label_column].astype(str)
    valid = valid[valid[label_column].isin(label_order)]
    valid[assignment_column] = valid[assignment_column].astype(str)
    counts = pd.crosstab(valid[assignment_column], valid[label_column])
    counts = counts.reindex(index=OUTPUT_CLUSTER_IDS, columns=label_order, fill_value=0)
    proportions = counts.div(counts.sum(axis=0).replace(0, np.nan), axis=1).fillna(0)
    known = valid[assignment_column].isin(CLUSTER_IDS)
    known_valid = valid.loc[known]
    summary = pd.DataFrame(
        [
            {
                "label": label_column,
                "n": len(valid),
                "n_total": len(valid),
                "n_retained": int(known.sum()),
                "n_rejected": int((~known).sum()),
                "coverage": float(known.mean()) if len(valid) else np.nan,
                "adjusted_rand_index": adjusted_rand_score(
                    known_valid[label_column], known_valid[assignment_column]
                )
                if len(known_valid)
                else np.nan,
            }
        ]
    )
    return counts, proportions, summary


def write_concordance_outputs(
    assignments: pd.DataFrame, out_dir: Path
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    rendered: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for label, order in (("UMS", UMS_ORDER), ("CMS", CMS_ORDER)):
        if label not in assignments:
            continue
        counts, proportions, summary = concordance_tables(assignments, label, order)
        tag = label.lower()
        write_tsv(counts.reset_index(), out_dir / f"cptac_{tag}_counts.tsv")
        write_tsv(proportions.reset_index(), out_dir / f"cptac_{tag}_proportions.tsv")
        write_tsv(summary, out_dir / f"cptac_{tag}_summary.tsv")
        rendered[label] = (proportions, summary)
    return rendered


def stacked_concordance(
    ax: plt.Axes,
    proportions: pd.DataFrame,
    summary: pd.DataFrame,
    label: str,
) -> None:
    categories = list(proportions.columns)
    y = np.arange(len(categories))
    left = np.zeros(len(categories), dtype=float)
    for cluster in OUTPUT_CLUSTER_IDS:
        values = proportions.loc[cluster, categories].to_numpy(dtype=float)
        ax.barh(
            y,
            values,
            left=left,
            height=0.62,
            color=CLUSTER_COLORS.get(cluster, "#BDBDBD"),
            edgecolor="white",
            linewidth=0.4,
        )
        for row, value in enumerate(values):
            if value >= 0.09:
                ax.text(
                    left[row] + value / 2,
                    row,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=font_pt("tick"),
                    color="black",
                )
        left += values
    n = int(summary.iloc[0]["n_total"])
    n_retained = int(summary.iloc[0]["n_retained"])
    n_rejected = int(summary.iloc[0]["n_rejected"])
    coverage = float(summary.iloc[0]["coverage"])
    ari = float(summary.iloc[0]["adjusted_rand_index"])
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of reference class")
    ari_text = f"{ari:.2f}" if np.isfinite(ari) else "NA"
    ax.set_title(f"{label} concordance", fontsize=font_pt("tick"), pad=12)
    ax.text(
        0.5,
        1.02,
        f"n={n} | known={n_retained} | Unknown={n_rejected} | "
        f"cov={coverage:.0%} | ARI={ari_text}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=font_pt("micro"),
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))


def confidence_panel(ax: plt.Axes, assignments: pd.DataFrame, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for index, cluster in enumerate(OUTPUT_CLUSTER_IDS, start=1):
        values = assignments.loc[
            assignments["predicted_cluster_thresholded"].astype(str).eq(cluster),
            "max_probability",
        ].dropna()
        if len(values):
            color = CLUSTER_COLORS.get(cluster, "#BDBDBD")
            ax.boxplot(
                [values.to_numpy()],
                positions=[index],
                widths=0.5,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": color, "alpha": 0.55},
                medianprops={"color": "black"},
            )
            jitter = rng.normal(0, 0.04, len(values))
            ax.scatter(
                index + jitter,
                values,
                s=13,
                color=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.8,
            )
    n_known = assignments["predicted_cluster_thresholded"].isin(CLUSTER_IDS).sum()
    n_rejected = len(assignments) - n_known
    coverage = n_known / len(assignments) if len(assignments) else np.nan
    ax.set_xlim(0.4, len(OUTPUT_CLUSTER_IDS) + 0.6)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(1, len(OUTPUT_CLUSTER_IDS) + 1))
    ax.set_xticklabels(
        [f"C{cluster}" for cluster in CLUSTER_IDS] + ["Unknown"],
        fontsize=font_pt("micro"),
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Maximum predicted probability")
    ax.set_title("Transfer confidence", fontsize=font_pt("tick"), pad=12)
    ax.text(
        0.5,
        1.02,
        f"n={len(assignments)} | known={n_known} | Unknown={n_rejected} | "
        f"cov={coverage:.0%}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=font_pt("micro"),
    )


def plot_cptac_summary(assignments: pd.DataFrame, out_path: Path, seed: int) -> None:
    concordance = {}
    for label, order in (("UMS", UMS_ORDER), ("CMS", CMS_ORDER)):
        if label in assignments:
            _, proportions, summary = concordance_tables(assignments, label, order)
            concordance[label] = (proportions, summary)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.7))
    for ax, label in zip(axes[:2], ("UMS", "CMS")):
        if label in concordance:
            stacked_concordance(ax, *concordance[label], label)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{label} unavailable", ha="center", va="center")
    confidence_panel(axes[2], assignments, seed)
    fig.legend(
        handles=[
            Patch(facecolor=CLUSTER_COLORS.get(c, "#BDBDBD"), label=cluster_label(c))
            for c in OUTPUT_CLUSTER_IDS
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=font_pt("label"),
        title="Thresholded assignment",
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.82))
    savefig(fig, out_path)


def render_saved(out_dir: Path, figures_dir: Path, seed: int) -> None:
    path = out_dir / "cptac_assignments.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Unified CPTAC assignments not found: {path}")
    assignments = pd.read_csv(path, sep="\t")
    if "predicted_cluster_thresholded" not in assignments:
        raise ValueError(
            f"Assignments lack thresholded transfer labels: {path}; rerun the full transfer stage"
        )
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_cptac_summary(assignments, figures_dir / "cptac_transfer_summary.png", seed)


def main() -> None:
    args = parse_args()
    set_house_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        render_saved(args.out_dir, args.figures_dir, args.seed)
        print(f"Redrew CPTAC transfer summary -> {args.figures_dir.resolve()}")
        return

    missing = [
        path
        for path in (args.cptac_prot, args.cptac_meta, args.frejno_zip)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing external transfer input(s): " + ", ".join(map(str, missing))
        )

    mdata_path = resolve_mdata_path(args.results_dir, args.mdata_path)
    metadata = pd.read_csv(args.results_dir / "metadata.csv", index_col=0)
    metadata.index = metadata.index.astype(str)
    discovery = unique_feature_columns(load_current_proteome(mdata_path))
    discovery = discovery.reindex(metadata.index)
    y_train = metadata[args.cluster_column].dropna().astype(str)
    discovery = discovery.loc[y_train.index]

    cptac_all = unique_feature_columns(pd.read_csv(args.cptac_prot, index_col=0))
    cptac_all.index = cptac_all.index.astype(str)
    cptac_meta = pd.read_csv(args.cptac_meta, index_col=0)
    cptac_meta.index = cptac_meta.index.astype(str)
    common_samples = cptac_all.index.intersection(cptac_meta.index)
    cptac = cptac_all.loc[common_samples].copy()
    crc65 = unique_feature_columns(load_cellline_proteome(args.frejno_zip))
    crc65.index = crc65.index.astype(str)

    audit = sample_audit(cptac_all.index, cptac_meta.index, crc65.index)
    write_tsv(audit, args.out_dir / "external_sample_audit.tsv")

    features = ordered_shared_features(discovery, cptac, crc65)
    if not features:
        raise ValueError("No gene-symbol features are shared by all three cohorts")
    write_tsv(
        feature_manifest(features, discovery, cptac, crc65),
        args.out_dir / "feature_manifest.tsv",
    )

    discovery_raw = discovery.loc[:, features]
    cptac_raw = cptac.loc[:, features]
    crc65_raw = crc65.loc[:, features]
    discovery_z = feature_zscore(discovery_raw)
    cptac_z = feature_zscore(cptac_raw)
    crc65_z = feature_zscore(crc65_raw)

    cptac_test = cptac_z.copy()
    cptac_test.index = "cptac::" + cptac_test.index.astype(str)
    crc65_test = crc65_z.copy()
    crc65_test.index = "crc65::" + crc65_test.index.astype(str)
    external_test = pd.concat([cptac_test, crc65_test], axis=0)

    diagnostics = pd.DataFrame(
        [
            matrix_diagnostics(discovery_raw, "discovery_raw_shared"),
            matrix_diagnostics(cptac_raw, "cptac_raw_shared"),
            matrix_diagnostics(crc65_raw, "crc65_raw_shared"),
            matrix_diagnostics(discovery_z, "discovery_feature_zscore"),
            matrix_diagnostics(cptac_z, "cptac_feature_zscore"),
            matrix_diagnostics(crc65_z, "crc65_feature_zscore"),
        ]
    )
    write_tsv(diagnostics, args.out_dir / "matrix_diagnostics.tsv")

    # Calibrate abstention on the fixed discovery validation split only. This
    # model is a policy-calibration object; it is not the final external model
    # used for CPTAC or CRC65 assignments.
    splits = load_splits(MODEL_SPLITS_DIR, set(discovery_z.index.astype(str)))
    calibration_train = discovery_z.loc[splits.train]
    calibration_val = discovery_z.loc[splits.val]
    calibration_truth = y_train.loc[splits.val]
    calibration_classifier = ConsensusClassifier(
        n_models=args.n_models,
        n_samples_per_model=None,
        n_features_per_model=args.n_features_per_model,
        scaler=None,
        verbose=True,
        use_ray=args.use_ray,
    )
    calibration_classifier.add_train_data(
        calibration_train, y_train.loc[splits.train].astype("category")
    )
    calibration_classifier.add_test_data(calibration_val)
    calibration_seed = args.seed + 1
    calibration_classifier.fit(calibration_seed)
    calibration_probabilities = normalized_probabilities(
        calibration_classifier.predict_proba(), calibration_val.index
    )
    rejection_threshold, calibration_table = calibrate_rejection_threshold(
        calibration_probabilities,
        calibration_truth,
    )
    calibration_table["target_retained_accuracy"] = 0.90
    calibration_table["minimum_coverage"] = 0.50
    calibration_table["selected"] = calibration_table["threshold"].eq(
        rejection_threshold
    )
    write_tsv(calibration_table, args.out_dir / "rejection_calibration.tsv")
    selected_calibration = calibration_table.loc[calibration_table["selected"]].copy()
    selected_calibration["calibration_seed"] = calibration_seed
    selected_calibration["calibration_train_n"] = len(calibration_train)
    selected_calibration["calibration_validation_n"] = len(calibration_val)
    write_tsv(selected_calibration, args.out_dir / "rejection_policy.tsv")
    selected_row = selected_calibration.iloc[0]
    print(
        f"Calibrated rejection threshold={rejection_threshold:.6f}; "
        f"retained accuracy={selected_row['retained_accuracy']:.3f}, "
        f"coverage={selected_row['coverage']:.3f}"
    )

    print(
        f"Training one ConsensusClassifier on {len(discovery_z)} tumours x "
        f"{len(features)} shared proteins; predicting {len(external_test)} external samples"
    )
    # Deliberately NOT the internal six-way recipe from
    # run_multiclass_consensus_classifier.py. That one bags samples (0.8), pins the
    # RFECV markers into every feature bag, and applies OOB-tuned per-class
    # thresholds; all three are tuned for held-out accuracy on our own cohort.
    # The transfer instead fits every discovery sample over the fixed three-way
    # feature manifest shared with CPTAC and CRC65, and the resulting assignments
    # (and their SHA-verified feature order) are what the manuscript quotes. Keep
    # them frozen: changing this recipe invalidates crc65_assignments.tsv and with
    # it the CRC65 mechanism and compound panels.
    classifier = ConsensusClassifier(
        n_models=args.n_models,
        n_samples_per_model=None,
        n_features_per_model=args.n_features_per_model,
        scaler=None,
        verbose=True,
        use_ray=args.use_ray,
    )
    classifier.add_train_data(discovery_z, y_train.astype("category"))
    classifier.add_test_data(external_test)
    classifier.fit(args.seed)

    predictions = classifier.predict().astype(str)
    predictions.index = external_test.index.astype(str)
    probabilities = normalized_probabilities(
        classifier.predict_proba(), external_test.index
    )

    cptac_mask = predictions.index.str.startswith("cptac::")
    crc65_mask = predictions.index.str.startswith("crc65::")
    cptac_pred = predictions[cptac_mask].copy()
    cptac_pred.index = cptac_pred.index.str.removeprefix("cptac::")
    cptac_prob = probabilities.loc[cptac_mask].copy()
    cptac_prob.index = cptac_prob.index.str.removeprefix("cptac::")
    crc65_pred = predictions[crc65_mask].copy()
    crc65_pred.index = crc65_pred.index.str.removeprefix("crc65::")
    crc65_prob = probabilities.loc[crc65_mask].copy()
    crc65_prob.index = crc65_prob.index.str.removeprefix("crc65::")

    cptac_assignments = assignments_for_cohort(
        cptac_pred, cptac_prob, "CPTAC", cptac_meta
    )
    crc65_assignments = assignments_for_cohort(crc65_pred, crc65_prob, "CRC65")
    cptac_assignments = apply_rejection(cptac_assignments, rejection_threshold)
    crc65_assignments = apply_rejection(crc65_assignments, rejection_threshold)
    combined = pd.concat(
        [cptac_assignments, crc65_assignments], ignore_index=True, sort=False
    )
    write_tsv(cptac_assignments, args.out_dir / "cptac_assignments.tsv")
    write_tsv(crc65_assignments, args.out_dir / "crc65_assignments.tsv")
    write_tsv(combined, args.out_dir / "external_assignments.tsv")
    write_tsv(
        distribution_table(combined, "predicted_cluster"),
        args.out_dir / "cluster_distribution_nearest.tsv",
    )
    write_tsv(distribution_table(combined), args.out_dir / "cluster_distribution.tsv")
    write_concordance_outputs(cptac_assignments, args.out_dir)

    feature_hash = hashlib.sha256(
        ("\n".join(features) + "\n").encode("utf-8")
    ).hexdigest()
    manifest = {
        "discovery_model_dir": str(args.results_dir),
        "mdata_path": str(mdata_path),
        "cluster_column": args.cluster_column,
        "n_discovery_samples": len(discovery_z),
        "n_cptac_samples": len(cptac_z),
        "n_crc65_samples": len(crc65_z),
        "n_shared_features": len(features),
        "feature_manifest_sha256": feature_hash,
        "normalization": "feature_zscore_independently_per_cohort",
        "classifier": "cclf.ConsensusClassifier",
        "classifier_fit_calls": 1,
        "final_classifier_fit_calls": 1,
        "calibration_classifier_fit_calls": 1,
        "calibration_split": {
            "train_n": len(calibration_train),
            "validation_n": len(calibration_val),
            "test_n_unused": len(splits.test),
            "seed": calibration_seed,
        },
        "n_models": args.n_models,
        "n_samples_per_model": None,
        "n_features_per_model": args.n_features_per_model,
        "seed": args.seed,
        "use_ray": bool(args.use_ray),
        "cptac_proteome": str(args.cptac_prot),
        "cptac_metadata": str(args.cptac_meta),
        "frejno_zip": str(args.frejno_zip),
        "rejection_policy": {
            "threshold": rejection_threshold,
            "target_retained_accuracy": 0.90,
            "minimum_coverage": 0.50,
            "selection": "lowest maximum-probability cutoff meeting both targets",
            "unknown_value": UNKNOWN_CLUSTER,
        },
        "assignment_semantics": (
            "predicted_cluster is the raw nearest CancerScout state; "
            "predicted_cluster_thresholded applies the common calibrated cutoff"
        ),
    }
    (args.out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    plot_cptac_summary(
        cptac_assignments,
        args.figures_dir / "cptac_transfer_summary.png",
        args.seed,
    )
    print(f"Wrote unified external transfer -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
