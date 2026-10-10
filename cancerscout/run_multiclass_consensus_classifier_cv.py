"""Five-fold out-of-fold evaluation for the locked six-way classifier.

This is a separate evaluation from ``run_multiclass_consensus_classifier``.
It uses every labelled sample exactly once as a held-out fold, fitting the
missingness filter and preprocessing pipeline inside each training fold.

Outputs land in ``results/multiclass_consensus_classifier_CV/`` by default.

Usage:
    conda run -n proj python -m cancerscout.run_multiclass_consensus_classifier_cv --use-ray
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from cancerscout.io import read_metadata, write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    MULTICLASS_CLASSIFIER_CV_RESULTS_DIR,
    RESULTS_DIR,
)
from cancerscout.plot_style import cluster_sort_key
from cancerscout.run_cluster_marker_rfecv import (
    fit_preprocessor,
    load_proteome_by_gene,
    transform_df,
)
from cancerscout.run_multiclass_consensus_classifier import (
    build_shap_top_markers,
    build_vote_table,
    calibration_report,
    fit_classifier,
    load_keep_features,
    per_class_metrics,
    shap_long_table,
    summarize_shap_per_class,
)


DEFAULT_OUT_DIR = MULTICLASS_CLASSIFIER_CV_RESULTS_DIR
DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR
DEFAULT_VOLCANO_TSV = (
    RESULTS_DIR / "cluster_interpretation" / "volcano_marker_rankings.tsv"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--mdata-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cluster-column", default="leiden_cons")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260520)
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument(
        "--n-samples-per-model",
        type=float,
        default=0.8,
        help="Fraction (0–1] or absolute count of training samples per submodel.",
    )
    parser.add_argument(
        "--n-features-per-model",
        default="auto",
        help="'auto' for the 95%%-coverage formula, or an int/float fraction.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.5,
        help="Drop proteins with > this fraction missing on training rows.",
    )
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument(
        "--keep-features-from",
        type=Path,
        default=None,
        help="Optional TSV of features forced into every ensemble feature bag.",
    )
    parser.add_argument(
        "--keep-minority-clusters",
        default=None,
        help="Comma-separated cluster ids whose samples appear in every sample bag.",
    )
    parser.add_argument(
        "--cluster-features",
        action="store_true",
        help="Use co-expression-aware feature bagging instead of uniform sampling.",
    )
    parser.add_argument(
        "--no-aggregate-logits",
        dest="aggregate_logits",
        action="store_false",
        help="Average submodel probabilities arithmetically instead of in logit space.",
    )
    parser.add_argument(
        "--ece-bins", type=int, default=10, help="Number of bins for calibration ECE."
    )
    parser.add_argument(
        "--ece-binning",
        choices=("uniform", "quantile"),
        default="quantile",
        help="Binning strategy for ECE.",
    )
    parser.add_argument(
        "--top-n", type=int, default=3, help="Top-N SHAP markers per direction."
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP aggregation."
    )
    return parser.parse_args()


def sample_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def ordered_ids(values: object) -> list[str]:
    return sorted((str(value) for value in values), key=sample_sort_key)


def metric_row(
    split: str,
    fold: int | str,
    y_true: pd.Series,
    y_pred: pd.Series,
    classes: list[str],
) -> dict[str, object]:
    y_true = y_true.astype(str)
    y_pred = y_pred.astype(str)
    return {
        "split": split,
        "fold": fold,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=classes,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                labels=classes,
                zero_division=0,
            )
        ),
    }


def aggregate_fold_shap(
    fold_shap: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.Series]],
    feature_order: list[str],
    classes: list[str],
    n_models_total: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Combine held-out SHAP summaries without treating absent features as zero."""
    signed_num: dict[str, dict[str, float]] = {cluster: {} for cluster in classes}
    signed_den: dict[str, dict[str, int]] = {cluster: {} for cluster in classes}
    abs_num: dict[str, dict[str, float]] = {cluster: {} for cluster in classes}
    abs_den: dict[str, dict[str, int]] = {cluster: {} for cluster in classes}
    exposure_total: Counter[str] = Counter()

    for n_samples, signed, abs_summary, exposure in fold_shap:
        exposure_total.update(
            {str(feature): int(count) for feature, count in exposure.items()}
        )
        for cluster in classes:
            for feature, value in signed[cluster].items():
                if np.isfinite(value):
                    feature = str(feature)
                    signed_num[cluster][feature] = (
                        signed_num[cluster].get(feature, 0.0) + float(value) * n_samples
                    )
                    signed_den[cluster][feature] = (
                        signed_den[cluster].get(feature, 0) + n_samples
                    )
            for feature, value in abs_summary[cluster].items():
                if np.isfinite(value):
                    feature = str(feature)
                    abs_num[cluster][feature] = (
                        abs_num[cluster].get(feature, 0.0) + float(value) * n_samples
                    )
                    abs_den[cluster][feature] = (
                        abs_den[cluster].get(feature, 0) + n_samples
                    )

    features = [feature for feature in feature_order if feature in exposure_total]
    signed_summary = pd.DataFrame(index=features, columns=classes, dtype=float)
    abs_summary = pd.DataFrame(index=features, columns=classes, dtype=float)
    for cluster in classes:
        signed_summary[cluster] = [
            signed_num[cluster].get(feature, np.nan) / signed_den[cluster][feature]
            if feature in signed_den[cluster]
            else np.nan
            for feature in features
        ]
        abs_summary[cluster] = [
            abs_num[cluster].get(feature, np.nan) / abs_den[cluster][feature]
            if feature in abs_den[cluster]
            else np.nan
            for feature in features
        ]

    exposure = pd.Series(
        [exposure_total.get(feature, 0) for feature in features],
        index=features,
        name="exposure_count",
        dtype=int,
    )
    long_df = shap_long_table(
        signed_summary,
        abs_summary,
        exposure,
        n_models=n_models_total,
    )
    return signed_summary, abs_summary, exposure, long_df


def main() -> None:
    args = parse_args()
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2")
    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading proteome and metadata...")
    x_gene, _feature_map = load_proteome_by_gene(args.results_dir, args.mdata_path)
    meta = read_metadata(args.results_dir)
    meta.index = meta.index.astype(str)
    if args.cluster_column not in meta.columns:
        raise KeyError(f"Missing cluster column: {args.cluster_column}")

    labels = meta[args.cluster_column].dropna().astype(str)
    sample_ids = ordered_ids(set(x_gene.index) & set(labels.index))
    labels = labels.reindex(sample_ids)
    log(f"Valid samples with cluster label: {len(sample_ids)}")
    if len(sample_ids) != 997:
        raise ValueError(f"Expected 997 locked samples, found {len(sample_ids)}")

    classes = sorted(labels.unique().tolist(), key=cluster_sort_key)
    expected_classes = [str(index) for index in range(1, 7)]
    if classes != expected_classes:
        raise ValueError(
            f"Expected six locked classes {expected_classes}, found {classes}"
        )

    splitter = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.seed,
    )
    y_values = labels.to_numpy()
    fold_assignments = pd.Series(index=sample_ids, dtype="Int64")
    prediction_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, object]] = []
    fold_per_class: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, object]] = []
    fold_shap: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.Series]] = []
    feature_order = list(x_gene.columns.astype(str))

    for fold_number, (train_pos, val_pos) in enumerate(
        splitter.split(np.zeros(len(sample_ids)), y_values), start=1
    ):
        train_ids = [sample_ids[position] for position in train_pos]
        val_ids = [sample_ids[position] for position in val_pos]
        y_train = labels.loc[train_ids]
        y_val = labels.loc[val_ids]
        if set(y_train.unique()) != set(classes):
            raise RuntimeError(
                f"Fold {fold_number} training data does not contain all classes"
            )
        fold_assignments.loc[val_ids] = fold_number
        log(
            f"Fold {fold_number}/{args.n_folds}: train={len(train_ids)}, held_out={len(val_ids)}"
        )

        x_train_raw = x_gene.loc[train_ids]
        missing_fraction = x_train_raw.isna().mean(axis=0)
        keep_columns = missing_fraction[
            missing_fraction <= args.missing_threshold
        ].index.tolist()
        x_train = x_train_raw.loc[:, keep_columns]
        x_val = x_gene.loc[val_ids, keep_columns]
        pipe = fit_preprocessor(x_train)
        x_train_t = transform_df(pipe, x_train)
        x_val_t = transform_df(pipe, x_val)

        keep_features = (
            load_keep_features(args.keep_features_from, list(x_train_t.columns))
            if args.keep_features_from
            else None
        )
        keep_samples = None
        if args.keep_minority_clusters:
            minority = {
                value.strip() for value in str(args.keep_minority_clusters).split(",")
            }
            keep_samples = y_train[y_train.isin(minority)].index.tolist()

        fold_classifier = fit_classifier(
            args,
            x_train_t,
            y_train,
            x_val_t,
            keep_features,
            keep_samples,
            seed=args.seed + fold_number,
        )
        probabilities = fold_classifier.predict_proba().reindex(columns=classes)
        predictions = fold_classifier.predict().astype(str)
        votes = fold_classifier.all_votes.reindex(columns=classes)
        if probabilities.isna().any().any():
            raise RuntimeError(f"Fold {fold_number} produced missing probabilities")
        if not np.allclose(probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-6):
            raise RuntimeError(f"Fold {fold_number} probabilities do not sum to one")

        y_val_aligned = y_val.reindex(probabilities.index).astype(str)
        vote_table = build_vote_table(votes, predictions, classes, y_val_aligned)
        fold_predictions = vote_table[
            [
                "sample_id",
                "predicted_cluster",
                "n_vote_models",
                "max_votes",
                "vote_share",
            ]
        ].copy()
        fold_predictions.insert(1, "fold", fold_number)
        fold_predictions.insert(3, "true_cluster", y_val_aligned.to_numpy())
        fold_predictions.insert(4, "split", "out_of_fold")
        for cluster in classes:
            fold_predictions[f"prob_{cluster}"] = probabilities[cluster].to_numpy()
            fold_predictions[f"votes_{cluster}"] = votes[cluster].to_numpy()
        prediction_parts.append(fold_predictions)

        fold_metrics.append(
            metric_row("fold", fold_number, y_val_aligned, predictions, classes)
        )
        per_class = per_class_metrics(y_val_aligned, predictions, classes)
        per_class.insert(0, "fold", fold_number)
        fold_per_class.append(per_class)
        fold_summary = {
            "fold": fold_number,
            "n_train": len(train_ids),
            "n_held_out": len(val_ids),
            "n_features_after_missing_filter": len(keep_columns),
            "n_models": args.n_models,
            "n_samples_per_model": int(fold_classifier.n_samples_per_model),
            "n_features_per_model_resolved": int(fold_classifier.n_features_per_model),
            "n_keep_features": len(keep_features) if keep_features else 0,
        }
        fold_summary.update(
            {
                f"n_train_C{cluster}": int((y_train == cluster).sum())
                for cluster in classes
            }
        )
        fold_summary.update(
            {
                f"n_held_out_C{cluster}": int((y_val == cluster).sum())
                for cluster in classes
            }
        )
        fold_summaries.append(fold_summary)

        if not args.skip_shap:
            log(f"Fold {fold_number}: aggregating held-out SHAP values...")
            signed, absolute, exposure = summarize_shap_per_class(
                fold_classifier,
                x_val_t,
                classes,
            )
            fold_shap.append((len(val_ids), signed, absolute, exposure))

    assignment_table = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "fold": fold_assignments.loc[sample_ids].astype(int).to_numpy(),
            "true_cluster": labels.loc[sample_ids].to_numpy(),
        }
    )
    if (
        assignment_table["fold"].isna().any()
        or assignment_table["sample_id"].duplicated().any()
    ):
        raise RuntimeError("CV fold assignments do not cover each sample exactly once")
    write_tsv(assignment_table, args.out_dir / "fold_assignments.tsv")

    predictions_df = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions_df) != len(sample_ids):
        raise RuntimeError("Out-of-fold predictions do not cover all samples")
    if (
        set(predictions_df["sample_id"]) != set(sample_ids)
        or predictions_df["sample_id"].duplicated().any()
    ):
        raise RuntimeError(
            "Out-of-fold predictions contain duplicate or missing samples"
        )
    predictions_df = predictions_df.set_index("sample_id").loc[sample_ids].reset_index()
    write_tsv(predictions_df, args.out_dir / "predictions.tsv")

    y_oof = predictions_df.set_index("sample_id")["true_cluster"].astype(str)
    pred_oof = predictions_df.set_index("sample_id")["predicted_cluster"].astype(str)
    metrics = metric_row("out_of_fold", "all", y_oof, pred_oof, classes)
    write_tsv(pd.DataFrame([metrics]), args.out_dir / "metrics.tsv")
    write_tsv(pd.DataFrame(fold_metrics), args.out_dir / "metrics_by_fold.tsv")

    per_class_oof = per_class_metrics(y_oof, pred_oof, classes)
    write_tsv(per_class_oof, args.out_dir / "per_class_metrics_cv.tsv")
    write_tsv(
        pd.concat(fold_per_class, ignore_index=True),
        args.out_dir / "per_class_metrics_by_fold.tsv",
    )
    write_tsv(pd.DataFrame(fold_summaries), args.out_dir / "fold_model_summary.tsv")

    cm = confusion_matrix(y_oof, pred_oof, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{cluster}" for cluster in classes],
        columns=[f"pred_{cluster}" for cluster in classes],
    )
    cm_df.to_csv(args.out_dir / "confusion_matrix_cv.tsv", sep="\t")

    probability_columns = [f"prob_{cluster}" for cluster in classes]
    probability_df = predictions_df.set_index("sample_id")[probability_columns].copy()
    probability_df.columns = classes
    log("Computing out-of-fold calibration...")
    per_class_ece, per_class_bins, top_label_bins = calibration_report(
        y_oof,
        probability_df,
        classes,
        args.ece_bins,
        args.ece_binning,
    )
    write_tsv(per_class_ece, args.out_dir / "calibration_per_class_ece.tsv")
    write_tsv(per_class_bins, args.out_dir / "calibration_per_class_bins.tsv")
    write_tsv(top_label_bins, args.out_dir / "calibration_top_label_bins.tsv")

    if not args.skip_shap:
        log("Writing aggregated held-out SHAP summaries...")
        signed_summary, absolute_summary, exposure, shap_long = aggregate_fold_shap(
            fold_shap,
            feature_order,
            classes,
            n_models_total=args.n_models * args.n_folds,
        )
        signed_summary.index.name = "gene_symbol"
        absolute_summary.index.name = "gene_symbol"
        write_tsv(
            signed_summary.reset_index(),
            args.out_dir / "shap_class_signed_summary.tsv",
        )
        write_tsv(
            absolute_summary.reset_index(),
            args.out_dir / "shap_class_abs_summary.tsv",
        )
        write_tsv(
            exposure.rename_axis("gene_symbol").reset_index(),
            args.out_dir / "shap_feature_exposure.tsv",
        )
        write_tsv(shap_long, args.out_dir / "shap_class_long.tsv")
        write_tsv(
            build_shap_top_markers(shap_long, args.top_n),
            args.out_dir / "shap_top_markers_per_cluster.tsv",
        )

    config = {
        "schema": "multiclass_consensus_classifier_cv_v1",
        "results_dir": str(args.results_dir),
        "cluster_column": args.cluster_column,
        "evaluation": "stratified_out_of_fold",
        "cv_method": "StratifiedKFold",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_samples": len(sample_ids),
        "classes": classes,
        "n_models": args.n_models,
        "n_models_total_for_shap": args.n_models * args.n_folds,
        "n_samples_per_model": args.n_samples_per_model,
        "n_features_per_model_arg": args.n_features_per_model,
        "use_ray": bool(args.use_ray),
        "ece_bins": args.ece_bins,
        "ece_binning": args.ece_binning,
        "shap_computed": not args.skip_shap,
        "top_n_shap_markers": args.top_n,
        "keep_features_from": str(args.keep_features_from)
        if args.keep_features_from
        else None,
        "keep_minority_clusters": args.keep_minority_clusters,
        "cluster_features": bool(args.cluster_features),
        "aggregate_logits": bool(args.aggregate_logits),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    log(f"Wrote five-fold CV outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
