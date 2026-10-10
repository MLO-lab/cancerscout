"""Classify current CancerScout samples into CPTAC/UMS classes and compare labels.

This is the script version of the core workflow in ``notebooks/clf_cptac.ipynb``
for the current locked downstream handoff. It trains a consensus classifier on
the CPTAC reference proteome and predicts CPTAC UMS labels for the current
CancerScout proteome, then compares those predictions to an existing CPTAC
assignment column such as ``cptac_prediction``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mudata as mu
import muvi
import numpy as np
import pandas as pd
import seaborn as sns
from cclf import ConsensusClassifier
from cancerscout.io import resolve_mdata_path as _resolve_mdata_path
from cancerscout.plot_style import savefig, set_house_style
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    matthews_corrcoef,
)
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CPTAC_METADATA_CSV,
    CPTAC_PROTEOME_CSV,
    CPTAC_RESULTS_DIR,
    FIGURES_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument(
        "--mdata-path",
        type=Path,
        default=None,
        help="Current h5mu. Default reads mdata_path from handoff_manifest.json.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="MuVI model directory. Default: RESULTS_DIR/muvi_model.",
    )
    parser.add_argument(
        "--current-source",
        choices=["muvi_observations", "raw_mdata"],
        default="raw_mdata",
        help=(
            "Feature matrix for current samples. raw_mdata is the locked primary CPTAC-transfer input; "
            "muvi_observations is useful as a sensitivity analysis."
        ),
    )
    parser.add_argument("--cptac-prot", type=Path, default=CPTAC_PROTEOME_CSV)
    parser.add_argument("--cptac-meta", type=Path, default=CPTAC_METADATA_CSV)
    parser.add_argument("--cptac-label-col", default="UMS")
    parser.add_argument("--old-label-col", default="cptac_prediction_previous")
    parser.add_argument("--out-dir", type=Path, default=CPTAC_RESULTS_DIR)
    parser.add_argument(
        "--figures-dir", type=Path, default=FIGURES_DIR / "cptac_classification"
    )
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument("--n-features-per-model", default="auto")
    parser.add_argument(
        "--scaler",
        default=None,
        choices=[None, "standard", "robust"],
        help="ConsensusClassifier scaler.",
    )
    parser.add_argument(
        "--use-ray", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--seed", type=int, default=2605071350, help="Classifier random seed."
    )
    parser.add_argument(
        "--normalization-mode",
        choices=["notebook", "feature_zscore_both", "global_both", "none"],
        default="feature_zscore_both",
        help=(
            "How to normalize CPTAC train and current test matrices after feature alignment. "
            "feature_zscore_both is the safest cross-cohort default; notebook reproduces clf_cptac.ipynb."
        ),
    )
    parser.add_argument(
        "--normalize-cptac",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility switch. Prefer --normalization-mode.",
    )
    parser.add_argument(
        "--normalize-current",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated compatibility switch. Prefer --normalization-mode.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the new labels into model metadata and refresh collaborator/downstream CPTAC handoff tables.",
    )
    return parser.parse_args()


def resolve_mdata_path(results_dir: Path, explicit: Path | None) -> Path:
    path = _resolve_mdata_path(results_dir, explicit)
    if path is None:
        raise FileNotFoundError(
            f"Could not resolve mdata_path: no --mdata-path, no manifest, and no config.json under {results_dir}"
        )
    return path


def global_normalize(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    return pd.DataFrame(
        (arr - np.nanmean(arr)) / np.nanstd(arr), index=df.index, columns=df.columns
    )


def feature_zscore(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=float)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    return pd.DataFrame((arr - mean) / std, index=df.index, columns=df.columns)


def apply_normalization(
    cptac_x: pd.DataFrame,
    current_x: pd.DataFrame,
    mode: str,
    normalize_cptac: bool | None = None,
    normalize_current: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "notebook":
        x_train = global_normalize(cptac_x)
        x_test = current_x.copy()
    elif mode == "feature_zscore_both":
        x_train = feature_zscore(cptac_x)
        x_test = feature_zscore(current_x)
    elif mode == "global_both":
        x_train = global_normalize(cptac_x)
        x_test = global_normalize(current_x)
    elif mode == "none":
        x_train = cptac_x.copy()
        x_test = current_x.copy()
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    if normalize_cptac is not None:
        x_train = global_normalize(cptac_x) if normalize_cptac else cptac_x.copy()
    if normalize_current is not None:
        x_test = global_normalize(current_x) if normalize_current else current_x.copy()
    return x_train, x_test


def load_current_proteome(mdata_path: Path) -> pd.DataFrame:
    mdata = mu.read_h5mu(mdata_path)
    prot = mdata["Proteome"].to_df().copy()
    prot.index = prot.index.astype(str)
    gene_symbols = mdata["Proteome"].var["gene_symbol"].reindex(prot.columns)
    prot.columns = gene_symbols.astype(str).values
    prot = prot.loc[:, prot.columns.notna()]
    prot = prot.loc[:, prot.columns.astype(str) != "nan"]
    prot = prot.loc[:, ~prot.columns.duplicated()].copy()
    return prot


def load_current_muvi_observations(model_dir: Path, mdata_path: Path) -> pd.DataFrame:
    model = muvi.load(model_dir, map_location="cpu")
    prot = model.get_observations(as_df=True)["Proteome"].copy()
    prot.index = prot.index.astype(str)
    mdata = mu.read_h5mu(mdata_path)
    gene_symbols = mdata["Proteome"].var["gene_symbol"].reindex(prot.columns)
    prot.columns = gene_symbols.astype(str).values
    prot = prot.loc[:, prot.columns.notna()]
    prot = prot.loc[:, prot.columns.astype(str) != "nan"]
    prot = prot.loc[:, ~prot.columns.duplicated()].copy()
    return prot


def align_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    common = train.columns.intersection(test.columns)
    if len(common) == 0:
        raise ValueError(
            "No overlapping gene-symbol features between CPTAC and current proteome"
        )
    return train.loc[:, common].copy(), test.loc[:, common].copy(), common


def matrix_diagnostics(df: pd.DataFrame, name: str) -> dict:
    arr = df.to_numpy(dtype=float)
    sample_means = np.nanmean(arr, axis=1)
    feature_means = np.nanmean(arr, axis=0)
    return {
        "matrix": name,
        "n_samples": df.shape[0],
        "n_features": df.shape[1],
        "finite_fraction": float(np.isfinite(arr).mean()),
        "nan_fraction": float(np.isnan(arr).mean()),
        "global_mean": float(np.nanmean(arr)),
        "global_std": float(np.nanstd(arr)),
        "global_min": float(np.nanmin(arr)),
        "global_max": float(np.nanmax(arr)),
        "sample_mean_median": float(np.nanmedian(sample_means)),
        "sample_mean_min": float(np.nanmin(sample_means)),
        "sample_mean_max": float(np.nanmax(sample_means)),
        "feature_mean_median": float(np.nanmedian(feature_means)),
        "feature_mean_min": float(np.nanmin(feature_means)),
        "feature_mean_max": float(np.nanmax(feature_means)),
    }


def matched_accuracy(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[float, dict[str, str], pd.DataFrame]:
    tab = pd.crosstab(y_true.astype(str), y_pred.astype(str))
    rows, cols = linear_sum_assignment(-tab.to_numpy())
    matched = tab.to_numpy()[rows, cols].sum() if len(rows) else 0
    mapping = {str(tab.columns[c]): str(tab.index[r]) for r, c in zip(rows, cols)}
    return matched / len(y_true), mapping, tab


def save_confusion_plot(
    tab: pd.DataFrame, path: Path, title: str, normalize: bool = False
) -> None:
    mat = tab.copy()
    fmt = "d"
    if normalize:
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0)
        fmt = ".2f"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(mat, cmap="Blues", ax=ax)
    mesh = ax.collections[0]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            if pd.isna(value):
                continue
            rgba = mesh.cmap(mesh.norm(float(value)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "black" if luminance > 0.58 else "white"
            ax.text(
                j + 0.5,
                i + 0.5,
                format(value, fmt),
                ha="center",
                va="center",
                color=color,
            )
    ax.set_title(title)
    ax.set_xlabel("new predicted CPTAC")
    ax.set_ylabel("existing CPTAC")
    fig.tight_layout()
    savefig(fig, path)
    plt.close(fig)


def clean_msi(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.upper()
    return values.where(values.isin(["MSI", "MSS"]))


def msi_binary_summary(
    truth: pd.Series, labels: pd.Series, assignment: str
) -> dict[str, object]:
    valid = truth.notna() & labels.notna()
    y = truth.loc[valid].eq("MSI")
    pred = labels.loc[valid].astype(str).eq("MSI")
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    tn = int((~y & ~pred).sum())
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity
        else np.nan
    )
    return {
        "assignment": assignment,
        "n": int(valid.sum()),
        "true_msi_n": int(y.sum()),
        "pred_msi_n": int(pred.sum()),
        "accuracy": accuracy_score(y, pred) if valid.any() else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred)
        if valid.any()
        else np.nan,
        "msi_precision": precision,
        "msi_recall_sensitivity": sensitivity,
        "msi_f1": f1,
        "specificity_mss": specificity,
        "mcc": matthews_corrcoef(y, pred) if valid.any() else np.nan,
        "tp_msi": tp,
        "fp_pred_msi_but_true_mss": fp,
        "fn_pred_nonmsi_but_true_msi": fn,
        "tn_mss": tn,
    }


def refresh_handoff_tables(
    meta: pd.DataFrame,
    pred: pd.Series,
    proba: pd.DataFrame,
    out_dir: Path,
    results_dir: Path,
) -> None:
    previous = meta["cptac_prediction"].astype(str).copy()
    meta["cptac_prediction_previous"] = previous
    meta["cptac_prediction"] = pred.reindex(meta.index).astype(str)
    meta.to_csv(results_dir / "metadata.csv")

    bid_meta = results_dir / "bid_indexed/metadata_by_bid.csv"
    if bid_meta.exists():
        by_bid = pd.read_csv(bid_meta)
        by_bid["cptac_prediction_previous"] = (
            by_bid["analysis_sample_id"].astype(str).map(previous)
        )
        by_bid["cptac_prediction"] = (
            by_bid["analysis_sample_id"].astype(str).map(meta["cptac_prediction"])
        )
        by_bid.to_csv(bid_meta, index=False)

    bid_col = next(
        (
            c
            for c in ["Proteome:new_bid", "Proteome:new_sample_token"]
            if c in meta.columns
        ),
        None,
    )
    handoff = pd.DataFrame(
        {
            "sample_id": meta.index,
            "bid": meta[bid_col].values if bid_col else meta.index,
            "leiden_cons": meta["leiden_cons"].values,
            "leiden_cons_current": meta["leiden_cons_current"].values
            if "leiden_cons_current" in meta
            else np.nan,
            "previous_cptac_prediction": previous.values,
            "cptac_prediction": meta["cptac_prediction"].values,
        }
    )
    handoff["cptac_prediction_changed_from_previous"] = (
        handoff["previous_cptac_prediction"] != handoff["cptac_prediction"]
    )
    for col in proba.columns:
        handoff[f"prob_{col}"] = proba[col].reindex(meta.index).values
    handoff.to_csv(out_dir / "sample_leiden_cons_cptac_prediction.csv", index=False)
    handoff[["bid", "leiden_cons", "cptac_prediction"]].rename(
        columns={
            "leiden_cons": "final_leiden_cons",
            "cptac_prediction": "final_cptac_prediction",
        }
    ).to_csv(out_dir / "final_leiden_cons_cptac_prediction.csv", index=False)

    for label, values in [("previous", previous), ("new", meta["cptac_prediction"])]:
        counts = pd.crosstab(meta["leiden_cons"], values).reindex(
            columns=["CIN", "MSI", "Mesenchymal", "Other"], fill_value=0
        )
        counts.index.name = "leiden_cons"
        counts.reset_index().to_csv(
            out_dir / f"locked_leiden_cons_{label}_cptac_distribution_counts.tsv",
            sep="\t",
            index=False,
        )
        frac = counts.div(counts.sum(axis=1), axis=0)
        frac.reset_index().to_csv(
            out_dir / f"locked_leiden_cons_{label}_cptac_distribution_fraction.tsv",
            sep="\t",
            index=False,
        )

    msi_col = next(
        (
            c
            for c in [
                "MSI consensus",
                "Proteome:new_MSI_consensus",
                "Proteome:new_MSI_final",
            ]
            if c in meta.columns
        ),
        None,
    )
    if msi_col:
        truth = clean_msi(meta[msi_col])
        pd.DataFrame(
            [
                msi_binary_summary(truth, previous, "previous_cptac_prediction"),
                msi_binary_summary(
                    truth, meta["cptac_prediction"], "new_cptac_prediction"
                ),
            ]
        ).to_csv(
            out_dir / "cptac_msi_concordance_with_truth.tsv", sep="\t", index=False
        )
        pd.crosstab(truth, previous).to_csv(
            out_dir / "cptac_previous_prediction_by_msi_consensus.tsv", sep="\t"
        )
        pd.crosstab(truth, meta["cptac_prediction"]).to_csv(
            out_dir / "cptac_new_prediction_by_msi_consensus.tsv", sep="\t"
        )


def main() -> None:
    args = parse_args()
    seed = args.seed
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_house_style()

    mdata_path = resolve_mdata_path(args.results_dir, args.mdata_path)
    meta = pd.read_csv(args.results_dir / "metadata.csv", index_col=0)
    meta.index = meta.index.astype(str)
    if args.old_label_col not in meta.columns:
        # A freshly trained handoff has no earlier assignment, so the agreement
        # tables compare against the placeholder label "None".
        print(
            f"Existing CPTAC label column not found in metadata: {args.old_label_col}; "
            "comparing against 'None'."
        )
        meta[args.old_label_col] = "None"

    cptac_x = pd.read_csv(args.cptac_prot, index_col=0)
    cptac_y = (
        pd.read_csv(args.cptac_meta, index_col=0)[args.cptac_label_col]
        .fillna("Other")
        .astype(str)
    )
    cptac_x.index = cptac_x.index.astype(str)
    cptac_y.index = cptac_y.index.astype(str)
    cptac_x = cptac_x.loc[cptac_y.index]
    cptac_x_raw = cptac_x.copy()

    model_dir = args.model_dir or args.results_dir / "muvi_model"
    if args.current_source == "muvi_observations":
        current_x = load_current_muvi_observations(model_dir, mdata_path)
    else:
        current_x = load_current_proteome(mdata_path)
    current_x = current_x.reindex(meta.index)
    current_x_raw = current_x.copy()

    x_train_raw, x_test_raw, common_features = align_features(
        cptac_x_raw, current_x_raw
    )
    x_train, x_test = apply_normalization(
        x_train_raw,
        x_test_raw,
        args.normalization_mode,
        normalize_cptac=args.normalize_cptac,
        normalize_current=args.normalize_current,
    )
    pd.DataFrame(
        [
            matrix_diagnostics(x_train_raw, "cptac_raw_common_features"),
            matrix_diagnostics(x_test_raw, "current_raw_common_features"),
            matrix_diagnostics(x_train, "cptac_classifier_input_common_features"),
            matrix_diagnostics(x_test, "current_classifier_input_common_features"),
        ]
    ).to_csv(args.out_dir / "feature_matrix_diagnostics.tsv", sep="\t", index=False)
    y_train = cptac_y.loc[x_train.index].astype("category")

    cclf = ConsensusClassifier(
        n_models=args.n_models,
        n_samples_per_model=None,
        n_features_per_model=args.n_features_per_model,
        scaler=args.scaler,
        verbose=True,
        use_ray=args.use_ray,
    )
    cclf.add_train_data(x_train, y_train)
    cclf.add_test_data(x_test)
    _ = cclf.fit(seed)

    pred = cclf.predict().astype(str)
    proba = cclf.predict_proba()
    pred.index = x_test.index.astype(str)
    proba.index = x_test.index.astype(str)

    old = meta[args.old_label_col].astype(str).reindex(pred.index)
    valid = old.notna() & pred.notna() & old.ne("nan") & pred.ne("nan")
    old_valid = old.loc[valid]
    pred_valid = pred.loc[valid]

    acc = accuracy_score(old_valid, pred_valid)
    bal_acc = balanced_accuracy_score(old_valid, pred_valid)
    macro_f1 = f1_score(old_valid, pred_valid, average="macro")
    weighted_f1 = f1_score(old_valid, pred_valid, average="weighted")
    ari = adjusted_rand_score(old_valid, pred_valid)
    nmi = normalized_mutual_info_score(old_valid, pred_valid)
    match_acc, match_map, tab = matched_accuracy(old_valid, pred_valid)
    old_counts = old_valid.value_counts(normalize=False)
    pred_counts = pred_valid.value_counts(normalize=False)
    old_fractions = old_valid.value_counts(normalize=True)
    pred_fractions = pred_valid.value_counts(normalize=True)
    max_new_label_fraction = (
        float(pred_fractions.max()) if len(pred_fractions) else np.nan
    )

    out = pd.DataFrame(
        {
            "sample_id": pred.index,
            "old_cptac_assignment": old.reindex(pred.index).values,
            "new_cptac_prediction": pred.reindex(pred.index).values,
            "changed": (
                old.reindex(pred.index).astype(str).values
                != pred.reindex(pred.index).astype(str).values
            ),
        },
        index=pred.index,
    )
    bid_col = next(
        (
            c
            for c in ["Proteome:new_bid", "Proteome:new_sample_token"]
            if c in meta.columns
        ),
        None,
    )
    if bid_col:
        out.insert(1, "bid", meta[bid_col].reindex(pred.index).values)
    for col in proba.columns:
        out[f"prob_{col}"] = proba[col].reindex(pred.index).values
    out.to_csv(
        args.out_dir / "cptac_predictions_current_vs_existing.tsv",
        sep="\t",
        index=False,
    )
    proba.to_csv(args.out_dir / "cptac_prediction_probabilities.tsv", sep="\t")
    out.loc[out["changed"]].to_csv(
        args.out_dir / "cptac_changed_samples.tsv", sep="\t", index=False
    )

    metrics = pd.DataFrame(
        [
            {"metric": "n_train_cptac", "value": len(x_train)},
            {"metric": "n_current_samples", "value": len(x_test)},
            {"metric": "n_common_features", "value": len(common_features)},
            {"metric": "accuracy", "value": acc},
            {"metric": "balanced_accuracy", "value": bal_acc},
            {"metric": "macro_f1", "value": macro_f1},
            {"metric": "weighted_f1", "value": weighted_f1},
            {"metric": "adjusted_rand_index", "value": ari},
            {"metric": "normalized_mutual_info", "value": nmi},
            {"metric": "hungarian_matched_accuracy", "value": match_acc},
            {"metric": "n_changed", "value": int((old_valid != pred_valid).sum())},
            {
                "metric": "fraction_changed",
                "value": float((old_valid != pred_valid).mean()),
            },
            {"metric": "max_new_label_fraction", "value": max_new_label_fraction},
        ]
    )
    metrics.to_csv(args.out_dir / "cptac_agreement_metrics.tsv", sep="\t", index=False)

    report = pd.DataFrame(
        classification_report(old_valid, pred_valid, output_dict=True, zero_division=0)
    ).T
    report.to_csv(args.out_dir / "cptac_per_class_report.tsv", sep="\t")
    tab.to_csv(args.out_dir / "cptac_confusion_existing_vs_new.tsv", sep="\t")
    tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).to_csv(
        args.out_dir / "cptac_confusion_existing_vs_new_row_normalized.tsv", sep="\t"
    )

    pd.DataFrame(
        {
            "new_prediction": list(match_map.keys()),
            "matched_existing_label": list(match_map.values()),
        }
    ).to_csv(args.out_dir / "cptac_hungarian_label_matching.tsv", sep="\t", index=False)

    dist = pd.concat(
        [
            old_counts.rename("old_count"),
            pred_counts.rename("new_count"),
            old_fractions.rename("old_fraction"),
            pred_fractions.rename("new_fraction"),
        ],
        axis=1,
    ).fillna(0)
    dist.to_csv(args.out_dir / "cptac_label_distribution_existing_vs_new.tsv", sep="\t")

    warnings = []
    if max_new_label_fraction > 0.9:
        top_label = str(pred_fractions.idxmax())
        warnings.append(
            {
                "warning": "collapsed_prediction_distribution",
                "details": f"{top_label} accounts for {max_new_label_fraction:.3f} of current predictions; inspect normalization and feature alignment.",
            }
        )
    pd.DataFrame(warnings, columns=["warning", "details"]).to_csv(
        args.out_dir / "cptac_warnings.tsv", sep="\t", index=False
    )

    save_confusion_plot(
        tab,
        fig_dir / "cptac_confusion_existing_vs_new.png",
        "Existing vs new CPTAC predictions",
        normalize=False,
    )
    save_confusion_plot(
        tab,
        fig_dir / "cptac_confusion_existing_vs_new_row_normalized.png",
        "Existing vs new CPTAC predictions, row-normalized",
        normalize=True,
    )

    dist_plot = dist.reset_index(names="label").melt(
        id_vars="label",
        value_vars=["old_fraction", "new_fraction"],
        var_name="assignment",
        value_name="fraction",
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(dist_plot, x="label", y="fraction", hue="assignment", ax=ax)
    ax.set_title("CPTAC label distribution: existing vs new")
    fig.tight_layout()
    savefig(fig, fig_dir / "cptac_distribution_existing_vs_new.png")
    plt.close(fig)

    run_config = {
        "seed": seed,
        "results_dir": str(args.results_dir),
        "mdata_path": str(mdata_path),
        "model_dir": str(model_dir),
        "current_source": args.current_source,
        "cptac_prot": str(args.cptac_prot),
        "cptac_meta": str(args.cptac_meta),
        "old_label_col": args.old_label_col,
        "n_models": args.n_models,
        "n_features_per_model": args.n_features_per_model,
        "use_ray": args.use_ray,
        "normalization_mode": args.normalization_mode,
        "normalize_cptac": args.normalize_cptac,
        "normalize_current": args.normalize_current,
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )
    if args.promote:
        refresh_handoff_tables(meta, pred, proba, args.out_dir, args.results_dir)
    print(f"Wrote CPTAC classification outputs to {args.out_dir.resolve()}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
