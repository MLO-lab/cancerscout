"""RFECV-based proteomic marker discovery for consensus clusters.

This script intentionally keeps the modeling problem simple and auditable:
for each consensus Leiden cluster it fits a one-vs-rest marker model using
the locked normalized proteome matrix, performs recursive feature elimination
inside the training split, and evaluates the selected marker panel on held-out
validation and test samples.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from scipy import sparse
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cancerscout.io import (
    build_sample_id_map,
    load_downstream_data,
    read_metadata,
    write_tsv,
)
from cancerscout.plot_style import (
    diverging_cmap,
    clean_ax,
    clean_colorbar,
    cluster_display_label,
    cluster_palette,
    relabel_cluster_axis,
    savefig,
    set_house_style,
)
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    CLUSTER_MARKER_RESULTS_DIR,
    FIGURES_DIR,
    MODEL_SPLITS_DIR,
    RESULTS_DIR,
)

try:
    import shap
except Exception:  # pragma: no cover
    shap = None


DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR
DEFAULT_OUT_DIR = CLUSTER_MARKER_RESULTS_DIR
DEFAULT_FIGURES_DIR = FIGURES_DIR / CLUSTER_MARKER_RESULTS_DIR.name
DEFAULT_SPLIT_DIR = MODEL_SPLITS_DIR


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass
class SplitIds:
    train: list[str]
    val: list[str]
    test: list[str]
    dropped: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-vs-rest RFECV marker discovery for locked consensus clusters.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Active model handoff directory.",
    )
    parser.add_argument(
        "--mdata-path",
        default=None,
        help="Optional explicit h5mu path; otherwise read handoff_manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for TSV/markdown outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(DEFAULT_FIGURES_DIR),
        help="Directory for PNG figures.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "rfecv", "evaluate"],
        help="Run the full workflow, only RFECV, or only downstream evaluation from saved RFECV rankings.",
    )
    parser.add_argument(
        "--rfecv-rankings-path",
        default=None,
        help="Saved RFECV rankings TSV for --stage evaluate. Defaults to out-dir/cluster_marker_rfecv_rankings_all_prefiltered.tsv.",
    )
    parser.add_argument(
        "--split-dir",
        default=str(DEFAULT_SPLIT_DIR),
        help="Directory containing global_train/val/test_idx.csv.",
    )
    parser.add_argument(
        "--cluster-column",
        default="leiden_cons",
        help="Cluster label column in metadata.csv.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        default=None,
        help="Optional cluster labels to run, e.g. --clusters 1 2 3.",
    )
    parser.add_argument("--seed", type=int, default=260510, help="Random seed.")
    parser.add_argument(
        "--n-rfecv-repeats",
        type=int,
        default=10,
        help="Number of RFECV repeats with shuffled CV folds.",
    )
    parser.add_argument(
        "--rfecv-outer-folds",
        type=int,
        default=5,
        help="Outer folds per consensus RFECV model.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Inner CV folds for each RFEFCV selector.",
    )
    parser.add_argument(
        "--rfecv-outer-min-count",
        type=int,
        default=2,
        help="Within each model repeat, keep proteins selected in at least this many outer folds.",
    )
    parser.add_argument(
        "--rfecv-scoring",
        default="balanced_accuracy",
        help="Inner RFECV scoring metric.",
    )
    parser.add_argument(
        "--min-features-to-select",
        type=int,
        default=50,
        help="Minimum markers selected by each RFEFCV selector.",
    )
    parser.add_argument(
        "--rfecv-step",
        type=float,
        default=0.1,
        help="Fraction of features removed at each RFECV step.",
    )
    parser.add_argument(
        "--max-features-for-rfecv",
        type=int,
        default=1000,
        help="Variance-prefiltered proteins entering RFECV.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.5,
        help="Drop proteins with missing fraction above this value.",
    )
    parser.add_argument(
        "--selection-frequency-threshold",
        type=float,
        default=0.8,
        help="Consensus threshold across RFECV repeats.",
    )
    parser.add_argument(
        "--fallback-top-n",
        type=int,
        default=50,
        help="Fallback selected markers if consensus threshold selects none.",
    )
    parser.add_argument(
        "--max-selected-features",
        type=int,
        default=200,
        help="Maximum markers retained for the final one-vs-rest model.",
    )
    parser.add_argument(
        "--reduced-panel-sizes",
        default="1,2,3,4,5,10,20,50,100,200,500",
        help="Comma-separated top-N panels to benchmark after final ranking.",
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Threads for RFECV and XGBoost."
    )
    parser.add_argument(
        "--hyperopt-metric",
        default="average_precision",
        choices=["average_precision", "roc_auc"],
        help="Validation metric used to choose XGBoost hyperparameters.",
    )
    parser.add_argument(
        "--hyperopt-method",
        default="optuna",
        choices=["optuna", "grid"],
        help="Hyperparameter optimizer.",
    )
    parser.add_argument(
        "--n-hyperopt-trials",
        type=int,
        default=100,
        help="Optuna trial budget per cluster.",
    )
    parser.add_argument(
        "--skip-shap", action="store_true", help="Skip SHAP computation and plots."
    )
    parser.add_argument(
        "--top-n-plot",
        type=int,
        default=25,
        help="Top markers to show per cluster in figures.",
    )
    parser.add_argument(
        "--top-markers-per-cluster-heatmap",
        type=int,
        default=10,
        help="Top markers per cluster for the combined marker heatmap.",
    )
    return parser.parse_args()


def make_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def clean_gene_symbol(value: object, fallback: str) -> str:
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return fallback
    return text.split(";")[0].split(",")[0].strip() or fallback


def load_proteome_by_gene(
    results_dir: Path, mdata_path: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_downstream_data(results_dir, mdata_path)
    if data.proteome is None:
        raise RuntimeError(
            "Could not load the Proteome modality from the locked h5mu file."
        )
    x = data.proteome.copy()
    if sparse.issparse(x):
        x = pd.DataFrame(
            x.toarray(), index=data.proteome.index, columns=data.proteome.columns
        )
    x.index = x.index.astype(str)

    if data.gene_symbols is None:
        genes = pd.Series(
            x.columns.astype(str), index=x.columns.astype(str), name="gene_symbol"
        )
    else:
        genes = data.gene_symbols.reindex(x.columns).copy()
        genes.index = genes.index.astype(str)
    cleaned = pd.Series(
        [clean_gene_symbol(genes.loc[col], str(col)) for col in x.columns],
        index=x.columns.astype(str),
        name="gene_symbol",
    )
    feature_map = (
        pd.DataFrame(
            {"protein_id": x.columns.astype(str), "gene_symbol": cleaned.values}
        )
        .groupby("gene_symbol", as_index=False)
        .agg(
            source_protein_ids=(
                "protein_id",
                lambda s: ";".join(sorted(set(map(str, s)))),
            ),
            n_source_proteins=("protein_id", "nunique"),
        )
    )

    x_gene = x.copy()
    x_gene.columns = cleaned.values
    x_gene = x_gene.T.groupby(level=0).median().T
    x_gene = x_gene.loc[:, ~x_gene.columns.duplicated()].copy()
    return x_gene, feature_map


def read_split_file(path: Path) -> list[str]:
    raw = pd.read_csv(path, index_col=0).iloc[:, 0]
    ids: list[str] = []
    for value in raw.dropna():
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].isdigit():
            text = text[:-2]
        ids.append(text)
    return ids


def load_splits(split_dir: Path, valid_ids: set[str]) -> SplitIds:
    splits: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    for name in ["train", "val", "test"]:
        path = split_dir / f"global_{name}_idx.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        ids = read_split_file(path)
        splits[name] = [sid for sid in ids if sid in valid_ids]
        dropped[name] = [sid for sid in ids if sid not in valid_ids]
    overlap = (
        (set(splits["train"]) & set(splits["val"]))
        | (set(splits["train"]) & set(splits["test"]))
        | (set(splits["val"]) & set(splits["test"]))
    )
    if overlap:
        raise ValueError(f"Split files overlap after filtering: {sorted(overlap)[:10]}")
    return SplitIds(splits["train"], splits["val"], splits["test"], dropped)


def prefilter_features(
    x_train: pd.DataFrame,
    x_all: pd.DataFrame,
    missing_threshold: float,
    max_features: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    missing = x_train.isna().mean(axis=0)
    keep = missing <= missing_threshold
    x_train_kept = x_train.loc[:, keep]
    variances = x_train_kept.var(axis=0, skipna=True).sort_values(ascending=False)
    selected = variances.head(max_features).index.astype(str).tolist()
    stats = pd.DataFrame(
        {
            "gene_symbol": missing.index.astype(str),
            "missing_fraction_train": missing.values,
            "variance_train": x_train.var(axis=0, skipna=True)
            .reindex(missing.index)
            .values,
            "passed_missing_filter": keep.values,
            "entered_rfecv_prefilter": missing.index.astype(str).isin(selected),
        }
    )
    return x_all.loc[:, selected].copy(), selected, stats


def fit_preprocessor(x_train: pd.DataFrame) -> Pipeline:
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pipe.fit(x_train)
    return pipe


def transform_df(pipe: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    values = pipe.transform(x)
    return pd.DataFrame(values, index=x.index, columns=x.columns)


def rfecv_xgb_estimator(
    seed: int, y: pd.Series, args: argparse.Namespace
) -> XGBClassifier:
    pos = max(int(y.sum()), 1)
    neg = max(int((1 - y).sum()), 1)
    return XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.05,
        gamma=0.0,
        reg_alpha=5.0,
        reg_lambda=10.0,
        subsample=0.9,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        random_state=seed,
        n_jobs=1,
    )


def run_consensus_rfecv(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
    cluster: str,
) -> pd.DataFrame:
    records = []
    log(
        f"Cluster {cluster}: starting XGB consensus RFECV with {args.n_rfecv_repeats} repeats"
    )
    for repeat in range(args.n_rfecv_repeats):
        seed = args.seed + repeat * 101
        positives = int(y_train.sum())
        negatives = int((1 - y_train).sum())
        outer_folds = max(2, min(args.rfecv_outer_folds, positives, negatives))
        outer_cv = StratifiedKFold(
            n_splits=outer_folds, shuffle=True, random_state=seed
        )
        fold_supports = []
        fold_ranks = []
        fold_scores = []
        log(
            f"Cluster {cluster}: RFECV repeat {repeat + 1}/{args.n_rfecv_repeats}, outer_folds={outer_folds}"
        )
        for outer_fold, (fold_train_idx, _) in enumerate(
            outer_cv.split(x_train, y_train), start=1
        ):
            log(
                f"Cluster {cluster}: repeat {repeat + 1}/{args.n_rfecv_repeats}, outer fold {outer_fold}/{outer_folds} started"
            )
            y_fold = y_train.iloc[fold_train_idx]
            inner_folds = max(
                2, min(args.cv_folds, int(y_fold.sum()), int((1 - y_fold).sum()))
            )
            cv = StratifiedKFold(
                n_splits=inner_folds, shuffle=True, random_state=seed + outer_fold
            )
            selector = RFECV(
                estimator=rfecv_xgb_estimator(seed + outer_fold, y_fold, args),
                step=args.rfecv_step,
                min_features_to_select=min(
                    args.min_features_to_select, x_train.shape[1]
                ),
                cv=cv,
                scoring=args.rfecv_scoring,
                importance_getter="feature_importances_",
                n_jobs=max(1, int(args.threads)),
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                selector.fit(x_train.iloc[fold_train_idx], y_fold)
            fold_supports.append(selector.support_)
            fold_ranks.append(selector.ranking_)
            fold_scores.append(
                float(np.nanmax(selector.cv_results_["mean_test_score"]))
            )
            log(
                f"Cluster {cluster}: repeat {repeat + 1}/{args.n_rfecv_repeats}, outer fold {outer_fold}/{outer_folds} "
                f"done; selected={int(selector.support_.sum())}; best_inner_{args.rfecv_scoring}={fold_scores[-1]:.3f}"
            )
        fold_supports_arr = np.vstack(fold_supports)
        fold_ranks_arr = np.vstack(fold_ranks)
        model_support = fold_supports_arr.sum(axis=0) >= min(
            args.rfecv_outer_min_count, fold_supports_arr.shape[0]
        )
        for idx, gene in enumerate(x_train.columns):
            records.append(
                {
                    "cluster": cluster,
                    "gene_symbol": str(gene),
                    "rfecv_repeat": repeat + 1,
                    "selected": bool(model_support[idx]),
                    "outer_fold_selection_count": int(fold_supports_arr[:, idx].sum()),
                    "outer_fold_selection_fraction": float(
                        fold_supports_arr[:, idx].mean()
                    ),
                    "rfecv_rank": float(fold_ranks_arr[:, idx].mean()),
                    "n_features_selected_in_repeat": int(model_support.sum()),
                    "mean_best_inner_cv_score": float(np.nanmean(fold_scores)),
                    "rfecv_scoring": args.rfecv_scoring,
                }
            )
        log(
            f"Cluster {cluster}: RFECV repeat {repeat + 1}/{args.n_rfecv_repeats} consensus selected {int(model_support.sum())} proteins"
        )
    repeat_df = pd.DataFrame(records)
    summary = (
        repeat_df.groupby(["cluster", "gene_symbol"], as_index=False)
        .agg(
            selection_frequency=("selected", "mean"),
            mean_outer_fold_selection_count=("outer_fold_selection_count", "mean"),
            mean_outer_fold_selection_fraction=(
                "outer_fold_selection_fraction",
                "mean",
            ),
            mean_rfecv_rank=("rfecv_rank", "mean"),
            median_rfecv_rank=("rfecv_rank", "median"),
            n_repeats_selected=("selected", "sum"),
            mean_best_inner_cv_score=("mean_best_inner_cv_score", "mean"),
            mean_n_features_selected=("n_features_selected_in_repeat", "mean"),
        )
        .sort_values(
            [
                "selection_frequency",
                "mean_rfecv_rank",
                "mean_outer_fold_selection_fraction",
                "median_rfecv_rank",
                "gene_symbol",
            ],
            ascending=[False, True, False, True, True],
        )
    )
    log(
        f"Cluster {cluster}: finished RFECV; {int((summary['selection_frequency'] >= args.selection_frequency_threshold).sum())} proteins pass selection frequency >= {args.selection_frequency_threshold}"
    )
    return summary


def train_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
    params: dict[str, float | int] | None = None,
) -> XGBClassifier:
    pos = max(int(y_train.sum()), 1)
    neg = max(int((1 - y_train).sum()), 1)
    base_params = {
        "n_estimators": 350,
        "max_depth": 2,
        "learning_rate": 0.035,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_weight": 2.0,
    }
    if params:
        base_params.update(params)
    model = XGBClassifier(
        **base_params,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        random_state=args.seed,
        n_jobs=args.threads,
    )
    model.fit(x_train, y_train)
    return model


def xgb_hyperparameter_candidates() -> list[dict[str, float | int]]:
    return [
        {
            "n_estimators": 250,
            "max_depth": 1,
            "learning_rate": 0.04,
            "min_child_weight": 2.0,
            "reg_alpha": 1.0,
            "reg_lambda": 8.0,
        },
        {
            "n_estimators": 350,
            "max_depth": 2,
            "learning_rate": 0.035,
            "min_child_weight": 2.0,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        },
        {
            "n_estimators": 450,
            "max_depth": 2,
            "learning_rate": 0.025,
            "min_child_weight": 3.0,
            "reg_alpha": 2.0,
            "reg_lambda": 8.0,
        },
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 4.0,
            "reg_alpha": 2.0,
            "reg_lambda": 10.0,
        },
    ]


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 1, 4),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 8.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
    }


def tune_xgb_on_validation(
    cluster: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int], XGBClassifier, pd.DataFrame]:
    records = []
    log(
        f"Cluster {cluster}: tuning final XGBoost on validation using {args.hyperopt_metric} ({args.hyperopt_method})"
    )
    best_score = -np.inf
    best_params: dict[str, float | int] = {}
    best_model: XGBClassifier | None = None
    if args.hyperopt_method == "optuna":
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_params, best_model
            params = suggest_xgb_params(trial)
            model = train_xgb(x_train, y_train, args, params=params)
            val_score = model.predict_proba(x_val)[:, 1]
            metrics = binary_metrics(y_val, val_score)
            score = metrics[args.hyperopt_metric]
            records.append(
                {
                    "cluster": cluster,
                    "candidate": trial.number + 1,
                    "selection_metric": args.hyperopt_metric,
                    "selection_score": score,
                    "optimizer": "optuna",
                    **params,
                    **{f"val_{k}": v for k, v in metrics.items()},
                }
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_params = params
                best_model = model
                log(
                    f"Cluster {cluster}: optuna trial {trial.number + 1}/{args.n_hyperopt_trials} new best {args.hyperopt_metric}={score:.3f}"
                )
            elif (trial.number + 1) % 10 == 0 or trial.number == 0:
                log(
                    f"Cluster {cluster}: optuna trial {trial.number + 1}/{args.n_hyperopt_trials} {args.hyperopt_metric}={score:.3f}; best={best_score:.3f}"
                )
            return float(score) if np.isfinite(score) else -1.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            study_name=f"cluster_{cluster}_xgb_validation",
        )
        study.optimize(
            objective, n_trials=args.n_hyperopt_trials, show_progress_bar=False
        )
        if best_model is None:
            best_params = dict(study.best_params)
            best_model = train_xgb(x_train, y_train, args, params=best_params)
        log(f"Cluster {cluster}: optuna best {args.hyperopt_metric}={best_score:.3f}")
        log(
            f"Cluster {cluster}: selected XGBoost params {json.dumps(best_params, sort_keys=True)}"
        )
        return best_params, best_model, pd.DataFrame(records)

    for idx, params in enumerate(xgb_hyperparameter_candidates(), start=1):
        log(
            f"Cluster {cluster}: hyperopt candidate {idx}/{len(xgb_hyperparameter_candidates())} started"
        )
        model = train_xgb(x_train, y_train, args, params=params)
        val_score = model.predict_proba(x_val)[:, 1]
        metrics = binary_metrics(y_val, val_score)
        score = metrics[args.hyperopt_metric]
        records.append(
            {
                "cluster": cluster,
                "candidate": idx,
                "selection_metric": args.hyperopt_metric,
                "selection_score": score,
                "optimizer": "grid",
                **params,
                **{f"val_{k}": v for k, v in metrics.items()},
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_params = params
            best_model = model
        log(
            f"Cluster {cluster}: hyperopt candidate {idx} {args.hyperopt_metric}={score:.3f}"
        )
    if best_model is None:
        best_params = xgb_hyperparameter_candidates()[0]
        best_model = train_xgb(x_train, y_train, args, params=best_params)
    log(
        f"Cluster {cluster}: selected XGBoost params {json.dumps(best_params, sort_keys=True)}"
    )
    return best_params, best_model, pd.DataFrame(records)


def binary_metrics(
    y_true: pd.Series, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_score))
        if len(np.unique(y_true)) == 2
        else np.nan,
        "average_precision": float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) == 2
        else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_pred)) > 1
        else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_single_feature_auc(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    out = {}
    for col in x.columns:
        values = x[col]
        if values.nunique(dropna=True) <= 1 or len(np.unique(y)) < 2:
            out[col] = np.nan
            continue
        ranks = rankdata(values.fillna(values.median()))
        out[col] = roc_auc_score(y, ranks)
    return pd.Series(out)


def compute_shap_summary(
    model: XGBClassifier, x_test: pd.DataFrame, cluster: str
) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    if shap is None:
        return None, None
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_test)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        summary = pd.DataFrame(
            {
                "cluster": cluster,
                "gene_symbol": x_test.columns.astype(str),
                "mean_abs_shap_test": np.abs(values).mean(axis=0),
                "mean_shap_test": values.mean(axis=0),
            }
        ).sort_values("mean_abs_shap_test", ascending=False)
        return summary, values
    except Exception as exc:
        warnings.warn(f"SHAP failed for cluster {cluster}: {exc}", RuntimeWarning)
        return None, None


def add_feature_statistics(
    ranking: pd.DataFrame,
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_map: pd.DataFrame,
) -> pd.DataFrame:
    selected = ranking["gene_symbol"].tolist()
    result = ranking.copy()
    result["xgb_gain_importance"] = (
        pd.Series(model.feature_importances_, index=selected)
        .reindex(result["gene_symbol"])
        .values
    )
    try:
        perm = permutation_importance(
            model,
            x_test[selected],
            y_test,
            n_repeats=15,
            random_state=17,
            scoring="roc_auc",
            n_jobs=1,
        )
        perm_df = pd.DataFrame(
            {
                "gene_symbol": selected,
                "permutation_importance_test_mean": perm.importances_mean,
                "permutation_importance_test_sd": perm.importances_std,
            }
        )
        result = result.merge(perm_df, on="gene_symbol", how="left")
    except Exception:
        result["permutation_importance_test_mean"] = np.nan
        result["permutation_importance_test_sd"] = np.nan

    auc_train = compute_single_feature_auc(x_train[selected], y_train)
    auc_val = compute_single_feature_auc(x_val[selected], y_val)
    auc_test = compute_single_feature_auc(x_test[selected], y_test)
    result["single_feature_auc_train"] = result["gene_symbol"].map(auc_train)
    result["single_feature_auc_val"] = result["gene_symbol"].map(auc_val)
    result["single_feature_auc_test"] = result["gene_symbol"].map(auc_test)

    pos_mean = x_train.loc[y_train.astype(bool), selected].mean(axis=0)
    rest_mean = x_train.loc[~y_train.astype(bool), selected].mean(axis=0)
    result["train_cluster_mean_z"] = result["gene_symbol"].map(pos_mean)
    result["train_rest_mean_z"] = result["gene_symbol"].map(rest_mean)
    result["train_cluster_minus_rest_z"] = (
        result["train_cluster_mean_z"] - result["train_rest_mean_z"]
    )
    result["marker_direction"] = np.where(
        result["train_cluster_minus_rest_z"] >= 0,
        "higher_in_cluster",
        "lower_in_cluster",
    )
    result = result.merge(feature_map, on="gene_symbol", how="left")
    return result.sort_values(
        [
            "selection_frequency",
            "mean_abs_shap_test",
            "xgb_gain_importance",
            "mean_rfecv_rank",
            "mean_outer_fold_selection_fraction",
            "median_rfecv_rank",
            "gene_symbol",
        ],
        ascending=[False, False, False, True, False, True, True],
        na_position="last",
    )


def selected_features_from_ranking(
    ranking: pd.DataFrame, args: argparse.Namespace
) -> list[str]:
    sort_cols = [
        c
        for c in [
            "selection_frequency",
            "mean_rfecv_rank",
            "mean_outer_fold_selection_fraction",
            "median_rfecv_rank",
            "gene_symbol",
        ]
        if c in ranking.columns
    ]
    ascending = [
        False
        if c in {"selection_frequency", "mean_outer_fold_selection_fraction"}
        else True
        for c in sort_cols
    ]
    ranking = ranking.sort_values(
        sort_cols, ascending=ascending, na_position="last"
    ).copy()
    selected = ranking.loc[
        ranking["selection_frequency"] >= args.selection_frequency_threshold,
        "gene_symbol",
    ].tolist()
    if not selected:
        selected = ranking.head(args.fallback_top_n)["gene_symbol"].tolist()
    if args.max_selected_features and len(selected) > args.max_selected_features:
        selected = selected[: args.max_selected_features]
    return selected


def checkpoint_rfecv(
    ranking: pd.DataFrame,
    selected: list[str],
    out_dir: Path,
    cluster: str,
    all_rankings: list[pd.DataFrame],
    all_selected: list[dict[str, object]],
) -> None:
    checkpoint_dir = out_dir / "rfecv_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(ranking, checkpoint_dir / f"cluster_{cluster}_rfecv_rankings.tsv")
    write_tsv(
        pd.DataFrame(
            {
                "cluster": cluster,
                "rank": range(1, len(selected) + 1),
                "gene_symbol": selected,
            }
        ),
        checkpoint_dir / f"cluster_{cluster}_selected_features.tsv",
    )
    cumulative_rankings = (
        pd.concat(all_rankings, ignore_index=True) if all_rankings else pd.DataFrame()
    )
    write_tsv(
        cumulative_rankings,
        out_dir / "cluster_marker_rfecv_rankings_all_prefiltered.tsv",
    )
    write_tsv(
        pd.DataFrame(all_selected), out_dir / "cluster_marker_selected_features.tsv"
    )


def parse_panel_sizes(text: str) -> list[int]:
    sizes = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        sizes.append(int(item))
    return sorted(set(s for s in sizes if s > 0))


def plot_performance_summary(metrics: pd.DataFrame, fig_dir: Path) -> None:
    if metrics.empty or "split" not in metrics.columns:
        return
    test = metrics.loc[metrics["split"] == "test"].copy()
    if test.empty:
        return
    long = test.melt(
        id_vars=["cluster"],
        value_vars=["roc_auc", "average_precision", "balanced_accuracy", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    long["cluster_label"] = long["cluster"].astype(str).map(cluster_display_label)
    order = [
        cluster_display_label(c)
        for c in sorted(
            test["cluster"].astype(str).unique(),
            key=lambda x: int(float(x))
            if str(x).replace(".", "", 1).isdigit()
            else str(x),
        )
    ]
    sns.barplot(
        data=long,
        x="cluster_label",
        y="value",
        hue="metric",
        order=order,
        ax=ax,
        palette="colorblind",
    )
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Held-out test score")
    clean_ax(ax)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )
    savefig(fig, fig_dir / "cluster_marker_performance_summary.png")


def plot_roc_pr_curves(cluster: str, predictions: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
    for split, frame in predictions.groupby("split"):
        if frame["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(frame["y_true"], frame["y_score"])
        precision, recall, _ = precision_recall_curve(frame["y_true"], frame["y_score"])
        axes[0].plot(
            fpr,
            tpr,
            label=f"{split} AUC={roc_auc_score(frame['y_true'], frame['y_score']):.2f}",
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{split} AP={average_precision_score(frame['y_true'], frame['y_score']):.2f}",
        )
    axes[0].plot([0, 1], [0, 1], ls="--", color="0.7", lw=0.9)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        clean_ax(ax)
        ax.legend(frameon=False, fontsize=7)
    savefig(fig, fig_dir / f"cluster_{cluster}_roc_pr_curves.png")


def plot_top_importance(
    cluster: str, feature_stats: pd.DataFrame, fig_dir: Path, top_n: int
) -> None:
    score_col = (
        "mean_abs_shap_test"
        if "mean_abs_shap_test" in feature_stats.columns
        and feature_stats["mean_abs_shap_test"].notna().any()
        else "xgb_gain_importance"
    )
    top = feature_stats.sort_values(score_col, ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, max(2.8, 0.18 * len(top) + 1.0)))
    palette = sns.color_palette("colorblind")
    colors = [
        palette[0] if delta >= 0 else palette[3]
        for delta in top["train_cluster_minus_rest_z"]
    ]
    ax.barh(top["gene_symbol"], top[score_col], color=colors)
    ax.set_xlabel(score_col.replace("_", " "))
    ax.set_ylabel("")
    clean_ax(ax)
    savefig(fig, fig_dir / f"cluster_{cluster}_top_marker_importance.png")


def plot_marker_heatmap(
    cluster: str,
    feature_stats: pd.DataFrame,
    x_z: pd.DataFrame,
    labels: pd.Series,
    fig_dir: Path,
    top_n: int,
) -> None:
    top = (
        feature_stats.sort_values(
            ["selection_frequency", "mean_abs_shap_test", "xgb_gain_importance"],
            ascending=[False, False, False],
            na_position="last",
        )
        .head(top_n)["gene_symbol"]
        .tolist()
    )
    if not top:
        return
    mean_by_cluster = (
        x_z[top].join(labels.rename("cluster")).groupby("cluster").mean().T
    )
    ordered_cols = sorted(
        mean_by_cluster.columns.astype(str),
        key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x),
    )
    mean_by_cluster = mean_by_cluster.loc[:, ordered_cols]
    vmax = max(float(np.nanquantile(np.abs(mean_by_cluster.values), 0.98)), 1e-6)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.18 * len(top) + 1.0)))
    sns.heatmap(
        relabel_cluster_axis(mean_by_cluster, axis="columns", wrapped=True),
        ax=ax,
        cmap=diverging_cmap(),
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Mean protein z-score"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    clean_colorbar(ax.collections[0].colorbar)
    savefig(fig, fig_dir / f"cluster_{cluster}_marker_expression_heatmap.png")


def plot_calibration(cluster: str, predictions: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    for split, frame in predictions.groupby("split"):
        if frame["y_true"].nunique() < 2:
            continue
        frac_pos, mean_pred = calibration_curve(
            frame["y_true"], frame["y_score"], n_bins=8, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=split)
    ax.plot([0, 1], [0, 1], ls="--", color="0.7", lw=0.9)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    clean_ax(ax)
    ax.legend(frameon=False)
    savefig(fig, fig_dir / f"cluster_{cluster}_calibration.png")


def plot_shap_beeswarm(
    cluster: str,
    shap_values: np.ndarray | None,
    x_test: pd.DataFrame,
    fig_dir: Path,
    top_n: int,
) -> None:
    if shap_values is None or shap is None:
        return
    try:
        plt.figure(figsize=(5.4, 4.2))
        shap.summary_plot(
            shap_values, x_test, max_display=top_n, show=False, plot_size=None
        )
        fig = plt.gcf()
        savefig(fig, fig_dir / f"cluster_{cluster}_shap_beeswarm.png")
    except Exception as exc:
        warnings.warn(
            f"Could not plot SHAP beeswarm for cluster {cluster}: {exc}", RuntimeWarning
        )


def plot_overlap_heatmap(
    selected_by_cluster: dict[str, list[str]], fig_dir: Path
) -> None:
    clusters = list(selected_by_cluster)
    if len(clusters) < 2:
        return
    mat = pd.DataFrame(index=clusters, columns=clusters, dtype=float)
    for a in clusters:
        for b in clusters:
            sa, sb = set(selected_by_cluster[a]), set(selected_by_cluster[b])
            mat.loc[a, b] = len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    plot_mat = relabel_cluster_axis(
        relabel_cluster_axis(mat.astype(float), axis="columns", wrapped=True),
        axis="index",
        wrapped=True,
    )
    sns.heatmap(
        plot_mat,
        ax=ax,
        cmap="crest",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Jaccard overlap"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    clean_colorbar(ax.collections[0].colorbar)
    savefig(fig, fig_dir / "cluster_marker_selected_feature_overlap.png")


def benchmark_reduced_panels(
    cluster: str,
    ranked_features: pd.DataFrame,
    x_z: pd.DataFrame,
    labels: pd.Series,
    splits: SplitIds,
    args: argparse.Namespace,
    best_params: dict[str, float | int],
) -> list[dict[str, float]]:
    ordered = (
        ranked_features.sort_values(
            [
                "mean_abs_shap_test",
                "xgb_gain_importance",
                "selection_frequency",
                "mean_rfecv_rank",
            ],
            ascending=[False, False, False, True],
            na_position="last",
        )["gene_symbol"]
        .drop_duplicates()
        .tolist()
    )
    out = []
    for panel_size in parse_panel_sizes(args.reduced_panel_sizes):
        genes = ordered[: min(panel_size, len(ordered))]
        if not genes:
            continue
        final_train_ids = list(splits.train) + list(splits.val)
        final_train_ids = [sid for sid in final_train_ids if sid in x_z.index]
        test_ids = [sid for sid in splits.test if sid in x_z.index]
        y_train = (labels.loc[final_train_ids] == cluster).astype(int)
        y_test = (labels.loc[test_ids] == cluster).astype(int)
        if y_train.sum() < 2 or y_test.nunique() < 2:
            continue
        model = train_xgb(
            x_z.loc[final_train_ids, genes], y_train, args, params=best_params
        )
        score = model.predict_proba(x_z.loc[test_ids, genes])[:, 1]
        metrics = binary_metrics(y_test, score)
        metrics.update(
            {
                "cluster": cluster,
                "panel_size_requested": panel_size,
                "panel_size_used": len(genes),
            }
        )
        out.append(metrics)
    return out


def plot_reduced_panel_metrics(panel_metrics: pd.DataFrame, fig_dir: Path) -> None:
    if panel_metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    plot_metrics = panel_metrics.copy()
    plot_metrics["cluster_label"] = (
        plot_metrics["cluster"].astype(str).map(cluster_display_label)
    )
    sns.lineplot(
        data=plot_metrics,
        x="panel_size_used",
        y="roc_auc",
        hue="cluster_label",
        marker="o",
        ax=ax,
        palette=cluster_palette(plot_metrics["cluster_label"]),
    )
    ax.set_xlabel("Top-N markers")
    ax.set_ylabel("Held-out test ROC AUC")
    ax.set_ylim(0, 1.03)
    clean_ax(ax)
    ax.legend(frameon=False, title="", ncol=2)
    savefig(fig, fig_dir / "cluster_marker_reduced_panel_roc_auc.png")


def plot_combined_marker_heatmap(
    feature_stats: pd.DataFrame,
    x_z: pd.DataFrame,
    labels: pd.Series,
    fig_dir: Path,
    top_n: int,
) -> None:
    required = {"cluster", "gene_symbol", "train_cluster_minus_rest_z"}
    if feature_stats.empty or not required.issubset(feature_stats.columns):
        return
    score_cols = [
        c
        for c in ["mean_abs_shap_test", "xgb_gain_importance", "selection_frequency"]
        if c in feature_stats.columns
    ]
    if not score_cols:
        return
    rows = []
    for cluster, frame in feature_stats.groupby("cluster", sort=False):
        frame = frame.copy()
        sort_cols = [
            c
            for c in [
                "mean_abs_shap_test",
                "xgb_gain_importance",
                "selection_frequency",
                "mean_rfecv_rank",
            ]
            if c in frame.columns
        ]
        ascending = [False if c != "mean_rfecv_rank" else True for c in sort_cols]
        top = frame.sort_values(
            sort_cols, ascending=ascending, na_position="last"
        ).head(top_n)
        top = top.assign(marker_group=str(cluster))
        rows.append(top)
    if not rows:
        return
    top_markers = pd.concat(rows, ignore_index=True)
    top_markers["row_id"] = (
        top_markers["marker_group"].astype(str)
        + " | "
        + top_markers["gene_symbol"].astype(str)
    )
    genes = top_markers["gene_symbol"].drop_duplicates().tolist()
    mean_by_cluster = (
        x_z[genes].join(labels.rename("cluster")).groupby("cluster").mean().T
    )
    ordered_cols = sorted(
        mean_by_cluster.columns.astype(str),
        key=lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else str(x),
    )
    mean_by_cluster = mean_by_cluster.loc[:, ordered_cols]
    row_table = top_markers.drop_duplicates(["marker_group", "gene_symbol"]).copy()
    row_table = row_table.set_index("row_id")
    heat = pd.DataFrame(index=row_table.index, columns=ordered_cols, dtype=float)
    for row_id, row in row_table.iterrows():
        heat.loc[row_id] = mean_by_cluster.loc[row["gene_symbol"]].values
    row_labels = [
        f"{row['gene_symbol']} {'+' if row['train_cluster_minus_rest_z'] >= 0 else '-'}"
        for _, row in row_table.iterrows()
    ]
    heat.index = row_labels
    vmax = max(float(np.nanquantile(np.abs(heat.values.astype(float)), 0.98)), 1e-6)
    fig_h = max(4.0, 0.16 * heat.shape[0] + 1.2)
    fig_w = max(8.5, 0.65 * heat.shape[1] + 4.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        relabel_cluster_axis(heat.astype(float), axis="columns", wrapped=True),
        ax=ax,
        cmap=diverging_cmap(),
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Mean protein z-score"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=7)
    clean_colorbar(ax.collections[0].colorbar)
    # Add separators between marker groups.
    group_sizes = row_table["marker_group"].value_counts(sort=False).tolist()
    pos = 0
    for size in group_sizes[:-1]:
        pos += size
        ax.axhline(pos, color="black", lw=0.8)
    savefig(fig, fig_dir / "cluster_marker_top10_combined_heatmap.png")


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    metrics: pd.DataFrame,
    splits: SplitIds,
    selected_by_cluster: dict[str, list[str]],
    n_features_start: int,
    n_features_prefiltered: int,
) -> None:
    if metrics.empty or "split" not in metrics.columns:
        test_metrics = pd.DataFrame()
    else:
        test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    rows = []
    if not test_metrics.empty and "cluster" in test_metrics.columns:
        test_metrics = test_metrics.sort_values("cluster")
    for _, row in test_metrics.iterrows():
        warn = ""
        if row["n_positive"] < 10:
            warn = " Underpowered positive count; interpret as exploratory."
        rows.append(
            f"| {row['cluster']} | {int(row['n_positive'])} | {int(row['n_features_selected'])} | "
            f"{row['roc_auc']:.3f} | {row['average_precision']:.3f} | {row['balanced_accuracy']:.3f} | {row['f1']:.3f} |{warn} |"
        )
    report = f"""# Cluster Marker RFECV Report

## Purpose

This analysis searches for compact protein marker panels that distinguish each locked consensus Leiden cluster from all remaining samples. It is meant as manuscript support for cluster interpretability, not as a clinical diagnostic model.

## Inputs

- Results directory: `{Path(args.results_dir)}`
- Cluster column: `{args.cluster_column}`
- Split directory: `{Path(args.split_dir)}`
- Starting gene-level proteome features: {n_features_start:,}
- Features entering RFECV after missingness and variance filtering: {n_features_prefiltered:,}
- RFECV repeats: {args.n_rfecv_repeats}
- RFECV outer folds per repeat: {args.rfecv_outer_folds}
- RFECV inner folds per selector: {args.cv_folds}
- RFECV estimator: XGBoost, feature importance from `feature_importances_`
- RFECV scoring: `{args.rfecv_scoring}`
- RFECV repeat-level support rule: selected in at least {args.rfecv_outer_min_count} outer folds
- Hyperparameter optimization: `{args.hyperopt_method}` with {args.n_hyperopt_trials if args.hyperopt_method == "optuna" else len(xgb_hyperparameter_candidates())} validation candidates/trials per cluster
- Stage: `{args.stage}`
- Existing split handling: train={len(splits.train)}, validation={len(splits.val)}, test={len(splits.test)}
- Dropped split IDs absent from current handoff: `{json.dumps(splits.dropped, sort_keys=True)}`

## What Changed Relative To The Old Notebooks

The old `rfecv.ipynb` / `rfecv_downstream.ipynb` workflow already had the right idea: XGBoost-based recursive feature elimination, held-out validation, feature importance, SHAP, and downstream transfer checks. The main risks were that parts of the code pointed to older `current_results/model_downstream` objects, used MuVI model observations rather than the final normalized proteome matrix, and mixed a lot of model selection, interpretation, and external transfer code in notebooks.

This script uses a single current cluster-stratified split: 60% training, 20% validation, and 20% locked test. Equivalently, 80% of samples are development data and 20% are final held-out test data. RFECV feature selection is done on the training split only, XGBoost hyperparameters are chosen on validation only, and the final test predictions come from a model refit on train + validation after tuning. The long RFECV stage can be run separately from the shorter evaluation and hyperparameter stage.

## Held-Out Test Summary

| Cluster | Test positives | Selected markers | ROC AUC | Average precision | Balanced accuracy | F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows) if rows else '| NA | NA | NA | NA | NA | NA | NA | No test metrics available |'}

## Critical Interpretation Notes

- Cluster 6 is small in this cohort, so marker panels for it are expected to be less stable even when ROC AUC looks high.
- XGBoost feature gain and SHAP are model-specific importance summaries; they should be cross-checked against expression direction and biological plausibility.
- Proteins selected for one-vs-rest discrimination may reflect tumor purity, stromal/immune composition, or technical missingness. The `cluster_marker_feature_rankings.tsv` table includes missingness-filter provenance and direction of effect to help catch this.
- Strong single-feature AUC is useful, but a manuscript marker should ideally be stable across RFECV repeats, directionally coherent, biologically interpretable, and visible in the cluster-level expression heatmap.
- When more proteins pass the consensus threshold than the final cap allows, ties are resolved deterministically by selection frequency, mean RFECV rank, outer-fold selection fraction, median RFECV rank, and gene symbol.

## Main Outputs

- `cluster_marker_model_metrics.tsv`: validation/test model metrics per cluster.
- `cluster_marker_hyperopt_validation.tsv`: validation-set hyperparameter candidates and selected metric.
- `cluster_marker_rfecv_rankings_all_prefiltered.tsv`: saved RFECV rankings, usable as input to `--stage evaluate`.
- `rfecv_checkpoints/cluster_<cluster>_rfecv_rankings.tsv`: per-cluster RFECV checkpoints written immediately after each cluster finishes RFECV.
- `cluster_marker_feature_rankings.tsv`: marker ranking with RFECV stability, XGBoost importance, SHAP, permutation importance, single-protein AUC, and expression direction.
- `cluster_marker_selected_features.tsv`: compact selected marker list per cluster.
- `cluster_marker_reduced_panel_metrics.tsv`: held-out test performance for top-N reduced panels.
- `cluster_marker_predictions.tsv`: per-sample probabilities for train/validation/test.
- `feature_prefilter_stats.tsv`: missingness and variance filtering audit.
- Figures are written to `{Path(args.figures_dir)}`.
"""
    (out_dir / "README_cluster_marker_rfecv.md").write_text(report)


def main() -> None:
    args = parse_args()
    set_house_style()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figures_dir)
    make_dirs(out_dir, fig_dir)

    log("Starting cluster marker RFECV pipeline")
    log(f"Outputs: tables={out_dir}, figures={fig_dir}")
    results_dir = Path(args.results_dir)
    log(f"Loading metadata from {results_dir}")
    metadata = read_metadata(results_dir)
    metadata.index = metadata.index.astype(str)
    if args.cluster_column not in metadata.columns:
        raise KeyError(f"Cluster column {args.cluster_column!r} not found in metadata.")
    labels = metadata[args.cluster_column].astype(str)

    log("Loading current MuData Proteome matrix")
    x_gene, feature_map = load_proteome_by_gene(results_dir, args.mdata_path)
    common = labels.index.intersection(x_gene.index.astype(str))
    labels = labels.loc[common]
    x_gene = x_gene.loc[common]
    id_map = build_sample_id_map(metadata.loc[common])

    log(f"Loading current split files from {Path(args.split_dir)}")
    splits = load_splits(Path(args.split_dir), set(common))
    split_names = {sid: "train" for sid in splits.train}
    split_names.update({sid: "val" for sid in splits.val})
    split_names.update({sid: "test" for sid in splits.test})
    used_ids = [sid for sid in common if sid in split_names]
    x_gene = x_gene.loc[used_ids]
    labels = labels.loc[used_ids]

    clusters = (
        args.clusters
        if args.clusters
        else sorted(
            labels.dropna().unique(),
            key=lambda x: int(float(x))
            if str(x).replace(".", "", 1).isdigit()
            else str(x),
        )
    )
    split_table = pd.DataFrame(
        {
            "analysis_sample_id": used_ids,
            "split": [split_names[sid] for sid in used_ids],
            "cluster": labels.loc[used_ids].values,
        }
    ).merge(id_map.reset_index(drop=True), on="analysis_sample_id", how="left")
    split_table.to_csv(out_dir / "sample_splits.tsv", sep="\t", index=False)
    feature_map.to_csv(out_dir / "feature_gene_symbol_map.tsv", sep="\t", index=False)

    log("Prefiltering proteins by train-set missingness and variance")
    x_pref, selected_prefilter, feature_filter_stats = prefilter_features(
        x_gene.loc[splits.train],
        x_gene,
        args.missing_threshold,
        args.max_features_for_rfecv,
    )
    write_tsv(feature_filter_stats, out_dir / "feature_prefilter_stats.tsv")

    log(
        f"Fitting train-only imputer/scaler for {len(selected_prefilter)} RFECV candidate proteins"
    )
    pipe = fit_preprocessor(x_pref.loc[splits.train])
    x_z = transform_df(pipe, x_pref)
    x_train = x_z.loc[[sid for sid in splits.train if sid in x_z.index]]
    x_val = x_z.loc[[sid for sid in splits.val if sid in x_z.index]]
    x_test = x_z.loc[[sid for sid in splits.test if sid in x_z.index]]
    y_labels_train = labels.loc[x_train.index]
    y_labels_val = labels.loc[x_val.index]
    y_labels_test = labels.loc[x_test.index]

    all_rfecv_rankings = []
    all_feature_stats = []
    all_selected = []
    all_metrics = []
    all_predictions = []
    all_shap = []
    all_panel_metrics = []
    all_hyperopt = []
    selected_by_cluster: dict[str, list[str]] = {}
    saved_rankings = pd.DataFrame()
    if args.stage == "evaluate":
        rankings_path = (
            Path(args.rfecv_rankings_path)
            if args.rfecv_rankings_path
            else out_dir / "cluster_marker_rfecv_rankings_all_prefiltered.tsv"
        )
        if not rankings_path.exists():
            raise FileNotFoundError(
                f"Cannot run --stage evaluate; missing RFECV rankings: {rankings_path}"
            )
        log(f"Loading saved RFECV rankings from {rankings_path}")
        saved_rankings = pd.read_csv(rankings_path, sep="\t")
        saved_rankings["cluster"] = saved_rankings["cluster"].astype(str)

    for cluster in clusters:
        cluster = str(cluster)
        log("=" * 80)
        log(f"Cluster {cluster}: one-vs-rest analysis started ({args.stage})")
        if cluster not in set(labels):
            warnings.warn(f"Skipping missing cluster label: {cluster}", RuntimeWarning)
            continue
        y_train = (y_labels_train == cluster).astype(int)
        y_val = (y_labels_val == cluster).astype(int)
        y_test = (y_labels_test == cluster).astype(int)
        if y_train.sum() < 5 or (1 - y_train).sum() < 5:
            warnings.warn(
                f"Cluster {cluster} has too few training positives/negatives for RFECV; skipping.",
                RuntimeWarning,
            )
            continue

        if args.stage == "evaluate":
            ranking = saved_rankings.loc[
                saved_rankings["cluster"].astype(str).eq(cluster)
            ].copy()
            if ranking.empty:
                warnings.warn(
                    f"No saved RFECV rankings found for cluster {cluster}; skipping.",
                    RuntimeWarning,
                )
                continue
            log(f"Cluster {cluster}: loaded {len(ranking)} saved RFECV-ranked proteins")
        else:
            ranking = run_consensus_rfecv(x_train, y_train, args, cluster)
            all_rfecv_rankings.append(ranking)
        selected = selected_features_from_ranking(ranking, args)
        log(
            f"Cluster {cluster}: final selected feature count after consensus/cap = {len(selected)}"
        )
        selected_by_cluster[cluster] = selected
        for rank, gene in enumerate(selected, start=1):
            all_selected.append({"cluster": cluster, "rank": rank, "gene_symbol": gene})
        if args.stage in {"all", "rfecv"}:
            checkpoint_rfecv(
                ranking, selected, out_dir, cluster, all_rfecv_rankings, all_selected
            )
            log(f"Cluster {cluster}: RFECV checkpoint written")
        if args.stage == "rfecv":
            continue

        best_params, tuning_model, hyperopt_df = tune_xgb_on_validation(
            cluster,
            x_train[selected],
            y_train,
            x_val[selected],
            y_val,
            args,
        )
        all_hyperopt.append(hyperopt_df)

        final_train_ids = list(x_train.index) + list(x_val.index)
        x_final_train = x_z.loc[final_train_ids, selected]
        y_final_train = (labels.loc[final_train_ids] == cluster).astype(int)
        final_model = train_xgb(x_final_train, y_final_train, args, params=best_params)
        log(f"Cluster {cluster}: final model refit on train+validation")

        split_frames = {
            "train": (x_train[selected], y_train),
            "val": (x_val[selected], y_val),
            "test": (x_test[selected], y_test),
        }
        cluster_predictions = []
        for split, (x_split, y_split) in split_frames.items():
            model_for_split = final_model if split == "test" else tuning_model
            score = model_for_split.predict_proba(x_split)[:, 1]
            metrics = binary_metrics(y_split, score)
            metrics.update(
                {
                    "cluster": cluster,
                    "split": split,
                    "n_features_selected": len(selected),
                    "n_rfecv_prefilter_features": len(selected_prefilter),
                    "n_train_positive": int(y_train.sum()),
                    "n_val_positive": int(y_val.sum()),
                    "n_test_positive": int(y_test.sum()),
                    "prediction_model_stage": "final_train_plus_val"
                    if split == "test"
                    else "tuning_train_only",
                    "hyperopt_metric": args.hyperopt_metric,
                    "best_xgb_params": json.dumps(best_params, sort_keys=True),
                }
            )
            all_metrics.append(metrics)
            pred = pd.DataFrame(
                {
                    "analysis_sample_id": x_split.index,
                    "cluster": cluster,
                    "split": split,
                    "y_true": y_split.values,
                    "y_score": score,
                    "y_pred_0_5": (score >= 0.5).astype(int),
                    "prediction_model_stage": "final_train_plus_val"
                    if split == "test"
                    else "tuning_train_only",
                }
            )
            cluster_predictions.append(pred)
            all_predictions.append(pred)

        pred_df = pd.concat(cluster_predictions, ignore_index=True)
        plot_roc_pr_curves(cluster, pred_df, fig_dir)
        plot_calibration(cluster, pred_df, fig_dir)

        shap_summary, shap_values = (None, None)
        if not args.skip_shap:
            shap_summary, shap_values = compute_shap_summary(
                final_model, x_test[selected], cluster
            )
            if shap_summary is not None:
                all_shap.append(shap_summary)
                plot_shap_beeswarm(
                    cluster, shap_values, x_test[selected], fig_dir, args.top_n_plot
                )

        stat_input = ranking.loc[ranking["gene_symbol"].isin(selected)].copy()
        if shap_summary is not None:
            stat_input = stat_input.merge(
                shap_summary, on=["cluster", "gene_symbol"], how="left"
            )
        else:
            stat_input["mean_abs_shap_test"] = np.nan
            stat_input["mean_shap_test"] = np.nan
        feature_stats = add_feature_statistics(
            stat_input,
            final_model,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            feature_map,
        )
        all_feature_stats.append(feature_stats)
        all_panel_metrics.extend(
            benchmark_reduced_panels(
                cluster,
                feature_stats,
                x_z,
                labels.loc[x_z.index],
                splits,
                args,
                best_params,
            )
        )
        plot_top_importance(cluster, feature_stats, fig_dir, args.top_n_plot)
        plot_marker_heatmap(
            cluster, feature_stats, x_z, labels.loc[x_z.index], fig_dir, args.top_n_plot
        )
        log(f"Cluster {cluster}: outputs and figures written")

    if args.stage == "evaluate":
        rfecv_rankings = saved_rankings.loc[
            saved_rankings["cluster"].astype(str).isin([str(c) for c in clusters])
        ].copy()
    else:
        rfecv_rankings = (
            pd.concat(all_rfecv_rankings, ignore_index=True)
            if all_rfecv_rankings
            else pd.DataFrame()
        )
    feature_stats = (
        pd.concat(all_feature_stats, ignore_index=True)
        if all_feature_stats
        else pd.DataFrame()
    )
    selected_df = pd.DataFrame(all_selected)
    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame()
    )
    shap_df = pd.concat(all_shap, ignore_index=True) if all_shap else pd.DataFrame()
    panel_metrics_df = pd.DataFrame(all_panel_metrics)
    hyperopt_df = (
        pd.concat(all_hyperopt, ignore_index=True) if all_hyperopt else pd.DataFrame()
    )

    write_tsv(
        rfecv_rankings, out_dir / "cluster_marker_rfecv_rankings_all_prefiltered.tsv"
    )
    write_tsv(feature_stats, out_dir / "cluster_marker_feature_rankings.tsv")
    write_tsv(selected_df, out_dir / "cluster_marker_selected_features.tsv")
    write_tsv(metrics_df, out_dir / "cluster_marker_model_metrics.tsv")
    write_tsv(predictions_df, out_dir / "cluster_marker_predictions.tsv")
    write_tsv(shap_df, out_dir / "cluster_marker_shap_summary.tsv")
    write_tsv(panel_metrics_df, out_dir / "cluster_marker_reduced_panel_metrics.tsv")
    write_tsv(hyperopt_df, out_dir / "cluster_marker_hyperopt_validation.tsv")

    plot_performance_summary(metrics_df, fig_dir)
    plot_overlap_heatmap(selected_by_cluster, fig_dir)
    plot_reduced_panel_metrics(panel_metrics_df, fig_dir)
    plot_combined_marker_heatmap(
        feature_stats,
        x_z,
        labels.loc[x_z.index],
        fig_dir,
        args.top_markers_per_cluster_heatmap,
    )

    config = vars(args).copy()
    config.update(
        {
            "n_samples_used": len(used_ids),
            "n_features_gene_level_start": int(x_gene.shape[1]),
            "n_features_entered_rfecv": int(len(selected_prefilter)),
            "clusters_run": clusters,
            "dropped_split_ids": splits.dropped,
        }
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True)
    )
    write_report(
        out_dir,
        args,
        metrics_df,
        splits,
        selected_by_cluster,
        x_gene.shape[1],
        len(selected_prefilter),
    )
    log("Cluster marker RFECV pipeline complete")


if __name__ == "__main__":
    main()
