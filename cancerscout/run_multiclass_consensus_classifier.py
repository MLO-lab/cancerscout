"""Multi-class proteome classifier using `cclf.ConsensusClassifier`.

Trains a bagged XGBoost consensus ensemble (100 submodels, stratified
sample bag, random feature bag with 95%-coverage `auto` k) on the locked
997-sample proteome to predict all six consensus clusters jointly. Held-
out test split is the shared `results/model_splits_997/` split.

In addition to standard classification metrics, computes:

- One-vs-rest **expected calibration error** + reliability bins per class
  and a top-label calibration table.
- Class-vs-rest **SHAP** values aggregated across the 100 submodels (one
  pass per class). Per-feature mean signed SHAP becomes the SHAP-based
  signature; per-cluster top-3 up / top-3 down are written for the
  manuscript-facing heatmaps.

Outputs land in `results/multiclass_consensus_classifier/`.

Usage:
    conda run -n proj python -m cancerscout.run_multiclass_consensus_classifier --use-ray
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
    classification_report,
    confusion_matrix,
    f1_score,
)

from cclf import ConsensusClassifier

from cancerscout.io import read_metadata, write_tsv
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    MODEL_SPLITS_DIR,
    MULTICLASS_CLASSIFIER_RESULTS_DIR,
    RESULTS_DIR,
)
from cancerscout.run_cluster_marker_rfecv import (
    fit_preprocessor,
    load_proteome_by_gene,
    load_splits,
    transform_df,
)


DEFAULT_OUT_DIR = MULTICLASS_CLASSIFIER_RESULTS_DIR
DEFAULT_SPLIT_DIR = MODEL_SPLITS_DIR


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--mdata-path", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--cluster-column", default="leiden_cons")
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
        help=(
            "TSV of RFECV-selected markers (cluster_marker_selected_features.tsv). "
            "Forces every submodel's feature bag to contain them, which counters the "
            "dilution of sampling a few hundred of ~8400 proteins per model. Those "
            "markers were selected on the train split only, so this leaks no test data."
        ),
    )
    parser.add_argument(
        "--keep-minority-clusters",
        default=None,
        help=(
            "Comma-separated cluster ids whose training samples appear in every "
            "submodel bag (e.g. '1,6'). The default proportional draw leaves the "
            "smallest classes with very few cases per bag."
        ),
    )
    parser.add_argument(
        "--cluster-features",
        action="store_true",
        help="Co-expression-aware feature bagging instead of uniform sampling.",
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
        help="Binning strategy for ECE (quantile is more robust on imbalanced classes).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Top-N SHAP markers per direction per cluster.",
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP aggregation."
    )
    return parser.parse_args()


def parse_features_per_model(arg: str) -> str | int | float:
    if arg == "auto":
        return "auto"
    try:
        value = float(arg)
    except ValueError as exc:
        raise ValueError(
            f"--n-features-per-model must be 'auto' or numeric, got {arg!r}"
        ) from exc
    return int(value) if value > 1 else value


def labels_aligned(
    meta: pd.DataFrame, sample_ids: list[str], cluster_col: str
) -> pd.Series:
    s = meta[cluster_col].astype(str).reindex(sample_ids)
    s.index = sample_ids
    return s


def expected_calibration_error_binary(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int, binning: str
) -> tuple[float, pd.DataFrame]:
    """One-vs-rest ECE for a single binary task, with bin-level diagnostics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if binning == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    else:
        qbins = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")
        bin_ids = np.asarray(qbins)
        n_bins = int(np.nanmax(bin_ids)) + 1 if np.isfinite(np.nanmax(bin_ids)) else 0

    ece = 0.0
    rows = []
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append((bin_idx, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        frac = n_bin / len(y_true)
        ece += frac * abs(acc - conf)
        rows.append(
            (
                bin_idx,
                n_bin,
                conf,
                acc,
                float(y_prob[mask].min()),
                float(y_prob[mask].max()),
            )
        )
    table = pd.DataFrame(
        rows,
        columns=[
            "bin",
            "count",
            "mean_confidence",
            "accuracy",
            "bin_left",
            "bin_right",
        ],
    )
    return ece, table


def calibration_report(
    y_true: pd.Series,
    prob_df: pd.DataFrame,
    classes: list[str],
    n_bins: int,
    binning: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-class one-vs-rest ECE + top-label ECE. Returns (per_class_ece, per_class_bins, top_label_bins)."""
    y_true = y_true.astype(str)
    per_class_rows = []
    per_class_bin_frames = []
    for cls in classes:
        y_bin = (y_true.values == cls).astype(int)
        ece, table = expected_calibration_error_binary(
            y_bin, prob_df[cls].values, n_bins, binning
        )
        per_class_rows.append({"cluster": cls, "ece": ece, "support": int(y_bin.sum())})
        bin_table = table.copy()
        bin_table.insert(0, "cluster", cls)
        per_class_bin_frames.append(bin_table)
    per_class_ece = pd.DataFrame(per_class_rows)
    per_class_bins = pd.concat(per_class_bin_frames, ignore_index=True)

    pred_label = prob_df.idxmax(axis=1)
    pred_conf = prob_df.max(axis=1).values
    correct = (pred_label.values == y_true.values).astype(int)
    top_ece, top_bin_table = expected_calibration_error_binary(
        correct, pred_conf, n_bins, binning
    )
    top_bin_table.insert(0, "kind", "top_label")
    top_bin_table["ece"] = top_ece
    return per_class_ece, per_class_bins, top_bin_table


def aggregate_shap_class_vs_rest(
    classifier_list: list,
    feature_subset_list: list[list[str]],
    x_test: pd.DataFrame,
    target_idx: int,
) -> pd.DataFrame:
    """Mean signed SHAP for one target class, aggregated across submodels.

    For each submodel: compute TreeExplainer SHAP values on x_test restricted
    to that submodel's feature subset, take the target-class slice, then
    average across submodels (only where a feature was in the submodel's bag,
    i.e. exposure-weighted).
    """
    import shap

    all_features = list(x_test.columns)
    n_samples = len(x_test)
    n_features = len(all_features)
    feat_index = {feat: idx for idx, feat in enumerate(all_features)}

    aggregated = np.zeros((n_samples, n_features), dtype=float)
    counts = np.zeros((n_samples, n_features), dtype=float)

    for clf, features in zip(classifier_list, feature_subset_list):
        x_sub = x_test.loc[:, features]
        explainer = shap.TreeExplainer(
            clf, feature_perturbation="interventional", model_output="raw"
        )
        shap_vals = explainer.shap_values(x_sub)
        if isinstance(shap_vals, list):
            shap_target = shap_vals[target_idx]
        else:
            # xgboost >=1.5 returns ndarray (n_samples, n_features, n_classes)
            shap_target = shap_vals[:, :, target_idx]
        for col_idx, feat in enumerate(features):
            out_idx = feat_index[feat]
            aggregated[:, out_idx] += shap_target[:, col_idx]
            counts[:, out_idx] += 1.0

    valid = counts > 0
    out = np.full_like(aggregated, np.nan)
    out[valid] = aggregated[valid] / counts[valid]
    return pd.DataFrame(out, index=x_test.index, columns=all_features)


def summarize_shap_per_class(
    consensus,
    x_test: pd.DataFrame,
    classes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run SHAP for each class and return:
    - signed_summary: DataFrame indexed by gene_symbol, columns=class labels, values=mean signed SHAP
    - abs_summary:    same shape, values=mean |SHAP|
    - exposure: Series gene_symbol -> count of submodels that contained the feature
    """
    class_to_idx = {label: idx for idx, label in consensus.le_mapping.items()}
    signed_cols, abs_cols = {}, {}
    for target_class in classes:
        if target_class not in class_to_idx:
            raise ValueError(
                f"target class {target_class!r} not in classifier's label mapping"
            )
        log(f"  SHAP aggregation for class {target_class}...")
        shap_df = aggregate_shap_class_vs_rest(
            consensus.classifier_list,
            consensus.feature_subset_list,
            x_test,
            target_idx=class_to_idx[target_class],
        )
        signed_cols[target_class] = shap_df.mean(axis=0)
        abs_cols[target_class] = shap_df.abs().mean(axis=0)
    signed_summary = pd.DataFrame(signed_cols)
    abs_summary = pd.DataFrame(abs_cols)
    counter: Counter[str] = Counter()
    for features in consensus.feature_subset_list:
        counter.update(features)
    exposure = pd.Series(
        [counter.get(g, 0) for g in signed_summary.index],
        index=signed_summary.index,
        name="exposure_count",
        dtype=int,
    )
    return signed_summary, abs_summary, exposure


def shap_long_table(
    signed: pd.DataFrame,
    abs_: pd.DataFrame,
    exposure: pd.Series,
    n_models: int,
) -> pd.DataFrame:
    """Long-form table for downstream R rendering."""
    abs_max = abs_.values.max() if abs_.size else 1.0
    abs_max = abs_max if np.isfinite(abs_max) and abs_max > 0 else 1.0
    rows = []
    for cluster in signed.columns:
        for gene in signed.index:
            shap_signed = float(signed.loc[gene, cluster])
            shap_abs = float(abs_.loc[gene, cluster])
            rows.append(
                {
                    "cluster": str(cluster),
                    "gene_symbol": gene,
                    "shap_signed": shap_signed,
                    "shap_abs": shap_abs,
                    "shap_abs_norm": shap_abs / abs_max
                    if np.isfinite(shap_abs)
                    else np.nan,
                    "exposure_count": int(exposure.get(gene, 0)),
                    "selection_frequency": float(exposure.get(gene, 0)) / n_models,
                    "always_pass": 1.0,
                }
            )
    return pd.DataFrame(rows)


def build_shap_top_markers(long_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    for cluster, sub in long_df.groupby("cluster"):
        sub = sub.dropna(subset=["shap_signed"])
        for direction, ranked in (
            ("up", sub.nlargest(top_n, "shap_signed")),
            ("down", sub.nsmallest(top_n, "shap_signed")),
        ):
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "cluster": cluster,
                        "direction": direction,
                        "rank": rank,
                        "gene_symbol": row.gene_symbol,
                        "shap_signed": row.shap_signed,
                        "shap_abs": row.shap_abs,
                        "exposure_count": row.exposure_count,
                    }
                )
    return pd.DataFrame(rows)


def per_class_metrics(
    y_true: pd.Series, y_pred: pd.Series, classes: list[str]
) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    rows = []
    for cls in classes:
        stats = report.get(cls, {})
        y_true_binary = (y_true.astype(str) == cls).astype(int)
        y_pred_binary = (y_pred.astype(str) == cls).astype(int)
        rows.append(
            {
                "cluster": cls,
                "precision": stats.get("precision", np.nan),
                "recall": stats.get("recall", np.nan),
                "f1": stats.get("f1-score", np.nan),
                "balanced_accuracy_ovr": balanced_accuracy_score(
                    y_true_binary, y_pred_binary
                ),
                "support": int(stats.get("support", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def load_keep_features(path: Path, available: list[str]) -> list[str]:
    """RFECV-selected marker gene symbols, restricted to the columns we actually have."""
    table = pd.read_csv(path, sep="\t")
    col = next(
        (
            c
            for c in ("gene_symbol", "feature", "protein", "name")
            if c in table.columns
        ),
        None,
    )
    if col is None:
        raise KeyError(f"No feature-name column found in {path}: {list(table.columns)}")
    wanted = {str(v) for v in table[col].dropna()}
    keep = [f for f in available if f in wanted]
    log(f"keep_features: {len(keep)} of {len(wanted)} selected markers present")
    return keep


def build_vote_table(
    votes: pd.DataFrame,
    predicted: pd.Series,
    classes: list[str],
    y_true: pd.Series | None = None,
) -> pd.DataFrame:
    """Keep vote counts, vote fraction and optional truth in one audit table."""
    votes = votes[classes].copy()
    n_models = votes.sum(axis=1).astype(int)
    max_votes = votes.max(axis=1).astype(int)
    table = pd.DataFrame(
        {
            "sample_id": votes.index.astype(str),
            "predicted_cluster": pd.Series(predicted, index=votes.index)
            .astype(str)
            .values,
            "n_vote_models": n_models.values,
            "max_votes": max_votes.values,
            "vote_share": (max_votes / n_models).values,
        },
        index=votes.index,
    )
    for cls in classes:
        table[f"votes_{cls}"] = votes[cls].values
    if y_true is not None:
        table["true_cluster"] = pd.Series(y_true, index=votes.index).astype(str).values
        table["correct"] = table["true_cluster"].eq(table["predicted_cluster"])
    return table


def make_classifier(
    args: argparse.Namespace,
    keep_features: list[str] | None,
    keep_samples: list[str] | None,
) -> ConsensusClassifier:
    return ConsensusClassifier(
        n_models=args.n_models,
        n_samples_per_model=args.n_samples_per_model,
        n_features_per_model=parse_features_per_model(args.n_features_per_model),
        scaler=None,
        use_ray=args.use_ray,
        verbose=True,
        keep_features=keep_features,
        keep_samples=keep_samples,
        cluster_features=args.cluster_features,
        aggregate_logits=args.aggregate_logits,
    )


def fit_classifier(
    args: argparse.Namespace,
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    x_eval: pd.DataFrame,
    keep_features: list[str] | None,
    keep_samples: list[str] | None,
    seed: int,
) -> ConsensusClassifier:
    clf = make_classifier(args, keep_features, keep_samples)
    clf.add_train_data(x_fit, y_fit)
    clf.add_test_data(x_eval)
    clf.fit(seed=seed)
    return clf


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log("Loading proteome and metadata...")
    x_gene, _feature_map = load_proteome_by_gene(args.results_dir, args.mdata_path)
    meta = read_metadata(args.results_dir)
    meta.index = meta.index.astype(str)

    valid_ids = set(x_gene.index) & set(meta.dropna(subset=[args.cluster_column]).index)
    log(f"Valid samples with cluster label: {len(valid_ids)}")

    splits = load_splits(args.split_dir, valid_ids)
    train_ids, val_ids, test_ids = splits.train, splits.val, splits.test
    log(f"Splits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # cclf has no separate validation set, so train it on train + val.
    trainval_ids = train_ids + val_ids
    y_trainval = labels_aligned(meta, trainval_ids, args.cluster_column)
    y_test = labels_aligned(meta, test_ids, args.cluster_column)
    classes = sorted(
        y_trainval.unique(), key=lambda v: int(v) if str(v).isdigit() else v
    )
    log(f"Classes: {classes}")

    # Missing-fraction filter on training rows only, then median-impute + standardize.
    x_trainval = x_gene.loc[trainval_ids]
    miss = x_trainval.isna().mean(axis=0)
    keep_cols = miss[miss <= args.missing_threshold].index.tolist()
    log(
        f"Proteins after missing-threshold filter: {len(keep_cols)} / {x_gene.shape[1]}"
    )

    x_trainval = x_trainval.loc[:, keep_cols]
    x_test = x_gene.loc[test_ids, keep_cols]
    pipe = fit_preprocessor(x_trainval)
    x_trainval_t = transform_df(pipe, x_trainval)
    x_test_t = transform_df(pipe, x_test)

    keep_features = (
        load_keep_features(args.keep_features_from, list(x_trainval_t.columns))
        if args.keep_features_from
        else None
    )
    keep_samples = None
    if args.keep_minority_clusters:
        minority = {c.strip() for c in str(args.keep_minority_clusters).split(",")}
        keep_samples = y_trainval[y_trainval.astype(str).isin(minority)].index.tolist()
        log(
            f"keep_samples: {len(keep_samples)} training samples from clusters "
            f"{sorted(minority)} forced into every submodel bag"
        )

    log(
        f"Training final ConsensusClassifier (n_models={args.n_models}, "
        f"n_samples_per_model={args.n_samples_per_model}, "
        f"n_features_per_model={args.n_features_per_model})..."
    )
    clf = fit_classifier(
        args,
        x_trainval_t,
        y_trainval,
        x_test_t,
        keep_features,
        keep_samples,
        args.seed,
    )

    log("Predicting on test split...")
    probs = clf.predict_proba()  # DataFrame, columns = class labels
    preds = clf.predict()  # Series of predicted labels
    votes = clf.all_votes  # DataFrame, columns = class labels

    # Align class label ordering across both views.
    probs = probs[classes]
    votes = votes[classes]

    # ---- metrics ----
    test_pred_str = preds.astype(str)
    test_true_str = y_test.astype(str)
    test_vote_table = build_vote_table(votes, test_pred_str, classes, test_true_str)
    metrics_row = {
        "split": "test",
        "n": int(len(y_test)),
        "accuracy": float(accuracy_score(test_true_str, test_pred_str)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_true_str, test_pred_str)
        ),
        "macro_f1": float(
            f1_score(
                test_true_str,
                test_pred_str,
                average="macro",
                labels=classes,
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                test_true_str,
                test_pred_str,
                average="weighted",
                labels=classes,
                zero_division=0,
            )
        ),
    }
    write_tsv(pd.DataFrame([metrics_row]), args.out_dir / "metrics.tsv")

    pc_df = per_class_metrics(test_true_str, test_pred_str, classes)
    write_tsv(pc_df, args.out_dir / "per_class_metrics_test.tsv")

    cm = confusion_matrix(test_true_str, test_pred_str, labels=classes)
    cm_df = pd.DataFrame(
        cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]
    )
    cm_df.to_csv(args.out_dir / "confusion_matrix_test.tsv", sep="\t")

    preds_df = pd.DataFrame(
        {
            "sample_id": preds.index.astype(str),
            "split": "test",
            "true_cluster": test_true_str.values,
            "predicted_cluster": test_pred_str.values,
            "n_vote_models": test_vote_table["n_vote_models"].values,
            "max_votes": test_vote_table["max_votes"].values,
            "vote_share": test_vote_table["vote_share"].values,
        }
    )
    for cls in classes:
        preds_df[f"prob_{cls}"] = probs[cls].values
        preds_df[f"votes_{cls}"] = votes[cls].values

    write_tsv(preds_df, args.out_dir / "predictions.tsv")

    # ---- calibration ----
    log("Computing calibration (top-label + per-class ECE)...")
    per_class_ece, per_class_bins, top_label_bins = calibration_report(
        y_test, probs, classes, args.ece_bins, args.ece_binning
    )
    write_tsv(per_class_ece, args.out_dir / "calibration_per_class_ece.tsv")
    write_tsv(per_class_bins, args.out_dir / "calibration_per_class_bins.tsv")
    write_tsv(top_label_bins, args.out_dir / "calibration_top_label_bins.tsv")

    # ---- SHAP class-vs-rest ----
    shap_long: pd.DataFrame | None = None
    if not args.skip_shap:
        log("Aggregating SHAP values across submodels (one pass per class)...")
        signed_summary, abs_summary, exposure = summarize_shap_per_class(
            clf, x_test_t, classes
        )
        signed_summary.index.name = "gene_symbol"
        abs_summary.index.name = "gene_symbol"
        write_tsv(
            signed_summary.reset_index(), args.out_dir / "shap_class_signed_summary.tsv"
        )
        write_tsv(
            abs_summary.reset_index(), args.out_dir / "shap_class_abs_summary.tsv"
        )
        write_tsv(
            exposure.rename_axis("gene_symbol").reset_index(),
            args.out_dir / "shap_feature_exposure.tsv",
        )
        shap_long = shap_long_table(
            signed_summary, abs_summary, exposure, args.n_models
        )
        write_tsv(shap_long, args.out_dir / "shap_class_long.tsv")
        top_markers = build_shap_top_markers(shap_long, args.top_n)
        write_tsv(top_markers, args.out_dir / "shap_top_markers_per_cluster.tsv")

    config = {
        "results_dir": str(args.results_dir),
        "cluster_column": args.cluster_column,
        "seed": args.seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_trainval": len(trainval_ids),
        "n_test": len(test_ids),
        "n_features_after_missing_filter": len(keep_cols),
        "classes": classes,
        "n_models": args.n_models,
        "n_samples_per_model": args.n_samples_per_model,
        "n_features_per_model_arg": args.n_features_per_model,
        "n_features_per_model_resolved": int(clf.n_features_per_model),
        "use_ray": bool(args.use_ray),
        "ece_bins": args.ece_bins,
        "ece_binning": args.ece_binning,
        "shap_computed": not args.skip_shap,
        "top_n_shap_markers": args.top_n,
        "keep_features_from": str(args.keep_features_from)
        if args.keep_features_from
        else None,
        "n_keep_features": len(keep_features) if keep_features else 0,
        "keep_minority_clusters": args.keep_minority_clusters,
        "n_keep_samples": len(keep_samples) if keep_samples else 0,
        "cluster_features": bool(args.cluster_features),
        "aggregate_logits": bool(args.aggregate_logits),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )

    log(f"Wrote multi-class cclf outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
