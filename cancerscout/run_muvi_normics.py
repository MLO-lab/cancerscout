"""Run the MuVI Normics workflow without a notebook.

Trains MuVI on the current Normics h5mu, runs repeated Leiden clustering on
the filtered latent factors, and derives the Hamming-distance consensus
clusters with r/consensus_clustering.R. promote_to_production turns the output
into the locked downstream handoff.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mudata as mu
import muvi
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from cancerscout.io import write_bid_indexed_exports
from cancerscout.paths import (
    ACTIVE_PROCESSED_NORMICS_DIR,
    EXTERNAL_DATA_DIR,
    MDATA_TEMPLATE_H5MU,
    MODELS_DIR,
    R_DIR,
)
from cancerscout.plotting import run_r_figure
from cancerscout.plot_style import savefig, set_house_style
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from tqdm.auto import tqdm


LOGGER = logging.getLogger("cancerscout.run_muvi_normics")


def parse_args() -> argparse.Namespace:
    data = EXTERNAL_DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5mu",
        type=Path,
        default=ACTIVE_PROCESSED_NORMICS_DIR
        / "mdata_r14_goettingen_normics_997_ordered.h5mu",
    )
    parser.add_argument("--old-h5mu", type=Path, default=MDATA_TEMPLATE_H5MU)
    parser.add_argument(
        "--metadata-path", type=Path, default=data / "current_results/metadata.csv"
    )
    parser.add_argument(
        "--msi-consensus-path", type=Path, default=data / "MSI_consense.csv"
    )
    parser.add_argument(
        "--current-leiden-path",
        type=Path,
        default=data / "current_results/leiden_cons.csv",
    )
    parser.add_argument(
        "--consensus-script", type=Path, default=R_DIR / "consensus_clustering.R"
    )
    parser.add_argument("--out-root", type=Path, default=MODELS_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Exact output directory. Default creates timestamped results_muvi_normics_*.",
    )
    parser.add_argument(
        "--handoff-dir", type=Path, default=MODELS_DIR / "current_handoff"
    )
    parser.add_argument(
        "--write-handoff", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--overwrite-handoff", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--force-overwrite-locked-handoff",
        action="store_true",
        help="Allow overwriting a handoff manifest marked as locked. Use only when intentionally replacing the current handoff.",
    )
    parser.add_argument("--skip-old-new-check", action="store_true")
    parser.add_argument("--skip-survival", action="store_true")
    parser.add_argument("--n-factors", type=int, default=50)
    parser.add_argument("--n-dense", type=int, default=3)
    parser.add_argument("--scale-confidence", type=float, default=0.995)
    parser.add_argument("--informed", action="store_true")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--repeated-leiden-resolution", type=float, default=0.75)
    parser.add_argument("--n-leiden-runs", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--optimizer", default="clipped")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Default uses the run timestamp as YYMMDDHHMM.",
    )
    parser.add_argument("--r2-thresh", type=float, default=0.95)
    parser.add_argument("--consensus-env", default="consensus-r")
    parser.add_argument("--survival-duration-col", default="OS Monate")
    parser.add_argument("--survival-event-col", default="OS Status")
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument(
        "--small-cluster-threshold",
        type=int,
        default=10,
        help="If the selected consensus k has any cluster at or below this size, use k-1 as the default handoff clustering.",
    )
    parser.add_argument("--exclude-uicc-iv", action="store_true")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def setup_logging(out_dir: Path, level: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(out_dir / "run_muvi_normics.log"),
    ]
    logging.basicConfig(
        level=getattr(logging, level), format=fmt, handlers=handlers, force=True
    )


def build_config(args: argparse.Namespace, out_dir: Path, seed_timestamp: str) -> dict:
    seed = args.seed if args.seed is not None else int(seed_timestamp)
    return {
        "input_h5mu": str(args.input_h5mu),
        "old_h5mu": str(args.old_h5mu),
        "out_dir": str(out_dir),
        "center_features": True,
        "n_factors": args.n_factors,
        "n_dense": args.n_dense,
        "scale_confidence": args.scale_confidence,
        "informed": args.informed,
        "resolution": args.resolution,
        "repeated_leiden_resolution": args.repeated_leiden_resolution,
        "n_leiden_runs": args.n_leiden_runs,
        "n_epochs": args.n_epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "seed": seed,
        "r2_thresh": args.r2_thresh,
        "consensus_env": args.consensus_env,
        "survival_duration_col": args.survival_duration_col,
        "survival_event_col": args.survival_event_col,
        "min_cluster_size": args.min_cluster_size,
        "small_cluster_threshold": args.small_cluster_threshold,
        "exclude_uicc_iv": args.exclude_uicc_iv,
    }


def first_token(x: object) -> str | None:
    if pd.isna(x):
        return None
    token = str(x).split(";")[0].strip()
    return token or None


def compare_old_new_by_accession(
    mdata_old,
    mdata_new,
    prot_label: str = "Proteome",
    log1p_old: bool = True,
    corr_method: str = "pearson",
    n_subsample: int = 120,
    random_state: int = 42,
) -> dict:
    old_df = mdata_old[prot_label].to_df().copy()
    new_df = mdata_new[prot_label].to_df().copy()
    old_df.index = old_df.index.astype(str)
    new_df.index = new_df.index.astype(str)
    old_df.columns = old_df.columns.astype(str)

    if log1p_old:
        old_df = np.log1p(old_df)

    old_df = old_df.loc[:, ~old_df.columns.duplicated()].copy()
    new_df = new_df.loc[:, new_df.columns.notna()].copy()
    new_df = new_df.loc[:, ~new_df.columns.duplicated()].copy()

    common_samples = old_df.index.intersection(new_df.index)
    common_features = old_df.columns.intersection(new_df.columns)
    old_sub = old_df.loc[common_samples, common_features]
    new_sub = new_df.loc[common_samples, common_features]

    diag_corr = pd.Series(
        {
            s: old_sub.loc[s].corr(new_sub.loc[s], method=corr_method)
            for s in common_samples
        },
        name="diag_corr",
    )
    diag_spearman = pd.Series(
        {
            s: old_sub.loc[s].corr(new_sub.loc[s], method="spearman")
            for s in common_samples
        },
        name="diag_spearman",
    )

    rng = np.random.default_rng(random_state)
    n_take = min(n_subsample, len(common_samples))
    sampled_ids = pd.Index(
        rng.choice(common_samples.to_numpy(), size=n_take, replace=False)
    )
    corr_sub = pd.DataFrame(index=sampled_ids, columns=sampled_ids, dtype=float)
    for s_old in sampled_ids:
        x = old_sub.loc[s_old]
        for s_new in sampled_ids:
            corr_sub.loc[s_old, s_new] = x.corr(new_sub.loc[s_new], method=corr_method)

    stats = {
        "n_common_samples": int(len(common_samples)),
        "n_common_features": int(len(common_features)),
        "diag_corr_mean": float(diag_corr.mean()),
        "diag_corr_median": float(diag_corr.median()),
        "diag_corr_min": float(diag_corr.min()),
        "diag_corr_max": float(diag_corr.max()),
        "diag_spearman_mean": float(diag_spearman.mean()),
        "diag_spearman_median": float(diag_spearman.median()),
        "feature_mean_corr": float(
            old_sub.mean(axis=0).corr(new_sub.mean(axis=0), method="pearson")
        ),
        "sample_mean_corr": float(
            old_sub.mean(axis=1).corr(new_sub.mean(axis=1), method="pearson")
        ),
        "n_subsample": int(n_take),
    }
    return {
        "stats": stats,
        "diag_corr": diag_corr.sort_values(),
        "diag_spearman": diag_spearman.sort_values(),
        "corr_sub": corr_sub,
    }


def plot_old_new_check(comparison: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(comparison["diag_corr"], bins=40, ax=axes[0])
    axes[0].axvline(comparison["diag_corr"].median(), color="red", linestyle="--")
    axes[0].set_title("Diagonal Pearson correlations")
    sns.histplot(comparison["diag_spearman"], bins=40, ax=axes[1])
    axes[1].axvline(comparison["diag_spearman"].median(), color="red", linestyle="--")
    axes[1].set_title("Diagonal Spearman correlations")
    sns.heatmap(comparison["corr_sub"], cmap="vlag", center=0, ax=axes[2])
    axes[2].set_title("Subsample old-vs-new sample correlations")
    fig.tight_layout()
    savefig(fig, out_dir / "old_new_proteome_sanity.png")
    plt.close(fig)


def attach_external_metadata(mdata, args: argparse.Namespace) -> None:
    LOGGER.info("Attaching external metadata")
    # The previous run's metadata and consensus labels only add comparison
    # columns; without them cptac_prediction is set by run_cptac_classification.
    has_current_meta = args.metadata_path.exists()
    has_current_leiden = args.current_leiden_path.exists()
    for path, present in [
        (args.metadata_path, has_current_meta),
        (args.current_leiden_path, has_current_leiden),
    ]:
        if not present:
            LOGGER.warning("Not found, skipping comparison columns: %s", path)

    msi_external = pd.read_csv(args.msi_consensus_path, index_col=0, sep=";")
    msi_external.index = msi_external.index.astype(str)
    msi_external = msi_external.loc[mdata.obs_names, :].copy()
    msi_external.index = mdata.obs_names.copy()

    if has_current_meta:
        current_meta = pd.read_csv(args.metadata_path, index_col=0)
        current_meta.index = current_meta.index.astype(str)
        cptac_col = (
            "cptac_prediction"
            if "cptac_prediction" in current_meta.columns
            else "cptac_all_pred"
        )
        mdata.obs["cptac_prediction"] = (
            current_meta[cptac_col].reindex(mdata.obs_names).astype("category")
        )
    mdata.obs["MSI consensus"] = msi_external.loc[
        mdata.obs_names, "MSI consensus"
    ].astype("category")
    if not has_current_leiden:
        return

    current_leiden = pd.read_csv(args.current_leiden_path, index_col=0)
    if "Prot_id" in current_leiden.columns:
        current_leiden = current_leiden.set_index("Prot_id")
    current_leiden.index = current_leiden.index.astype(str)
    current_leiden_map = {
        "5": "4",
        "7": "3",
        "6": "5",
        "2": "1",
        "4": "6",
        "3": "2",
        "1": "7",
    }
    mdata.obs["leiden_cons_current"] = (
        current_leiden.loc[mdata.obs_names, "class"]
        .astype(str)
        .map(current_leiden_map)
        .astype("category")
    )
    mdata.obs["leiden_cons_entropy_current"] = pd.to_numeric(
        current_leiden.loc[mdata.obs_names, "entropy"], errors="coerce"
    )
    mdata.obs["leiden_cons_silhouette_current"] = pd.to_numeric(
        current_leiden.loc[mdata.obs_names, "silhouette"], errors="coerce"
    )


def center_proteome(mdata, prot_label: str) -> float:
    LOGGER.info("Centering proteome and scaling by one global standard deviation")
    x = np.asarray(mdata[prot_label].X, dtype=np.float32)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    global_std = float(np.nanstd(x))
    x = x / global_std
    mdata[prot_label].X = x
    mdata.update()
    LOGGER.info("Proteome shape=%s global_std=%.6g", x.shape, global_std)
    return global_std


def train_muvi(
    mdata,
    config: dict,
    out_dir: Path,
    prot_label: str,
    mut_label: str,
    view_names: list[str],
):
    LOGGER.info("Building MuVI model")
    kwargs = {
        "n_factors": config["n_factors"],
        "prior_mask_key": "gene_set_mask",
        "prior_confidence": config["scale_confidence"],
        "view_names": view_names,
        "nmf": {
            prot_label: not config["center_features"],
            mut_label: not config["center_features"],
        },
    }
    if not config["informed"]:
        kwargs.pop("prior_mask_key")

    if len(view_names) == 1:
        model = muvi.tl.from_adata(mdata[view_names[0]], **kwargs)
    else:
        model = muvi.tl.from_mdata(mdata, **kwargs)

    LOGGER.info("Fitting MuVI for %s epochs", config["n_epochs"])
    model.fit(
        n_epochs=config["n_epochs"],
        n_particles=1,
        learning_rate=config["learning_rate"],
        optimizer=config["optimizer"],
        seed=config["seed"],
        callbacks=[
            muvi.EarlyStoppingCallback(
                config["n_epochs"], min_epochs=config["n_epochs"] // 3
            )
        ],
    )
    model_dir = out_dir / "muvi_model"
    muvi.save(model, str(model_dir))
    LOGGER.info("Saved MuVI model to %s", model_dir)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(model._training_log["history"])
    ax.set_title("MuVI training objective")
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("loss")
    fig.tight_layout()
    savefig(fig, out_dir / "muvi_training_objective.png")
    plt.close(fig)
    return model


def attach_model_metadata(model, mdata, mut_label: str) -> None:
    LOGGER.info("Attaching model metadata")
    model._cache.factor_adata.obs = mdata.obs.copy()

    if mut_label in mdata.mod:
        mut_df = mdata[mut_label].to_df()
        for col in mut_df.columns:
            muvi.tl.add_metadata(
                model,
                col,
                (mut_df.loc[model.sample_names, col] == 1.0).astype("category"),
            )

    for col in ["cptac_prediction", "MSI consensus", "leiden_cons_current"]:
        if col in mdata.obs.columns:
            muvi.tl.add_metadata(
                model,
                col,
                mdata.obs.loc[model.sample_names, col].astype("category"),
                overwrite=True,
            )

    for col in ["leiden_cons_entropy_current", "leiden_cons_silhouette_current"]:
        if col in mdata.obs.columns:
            muvi.tl.add_metadata(
                model,
                col,
                pd.to_numeric(mdata.obs.loc[model.sample_names, col], errors="coerce"),
                overwrite=True,
            )


def run_base_latent_analysis(model, config: dict, out_dir: Path) -> sc.AnnData:
    LOGGER.info("Running MuVI factor filtering, neighbors, Leiden, and UMAP")
    muvi.tl.filter_factors(model, r2_thresh=config["r2_thresh"])
    muvi.tl.neighbors(model)
    muvi.tl.leiden(model, resolution=config["resolution"])
    muvi.tl.umap(model)
    model._cache.factor_adata.obs["leiden"].to_csv(out_dir / "leiden_single_run.csv")

    LOGGER.info("Building latent AnnData")
    adata_latent = sc.AnnData(
        np.asarray(model._cache.factor_adata.obsm["filtered_scores"], dtype=np.float32)
    )
    adata_latent.obs = model._cache.factor_adata.obs.copy()
    adata_latent.obs_names = adata_latent.obs_names.astype(str)
    adata_latent.var_names = [f"LF{i + 1}" for i in range(adata_latent.n_vars)]
    sc.pp.neighbors(adata_latent, use_rep="X")
    sc.tl.umap(adata_latent)
    return adata_latent


def run_repeated_leiden(
    adata_latent: sc.AnnData, config: dict, out_dir: Path
) -> pd.DataFrame:
    LOGGER.info("Running %s repeated Leiden clusterings", config["n_leiden_runs"])
    leiden_assignments = []
    start = time.time()
    for seed in tqdm(range(config["n_leiden_runs"]), desc="Repeated Leiden"):
        sc.tl.leiden(
            adata_latent,
            resolution=config["repeated_leiden_resolution"],
            random_state=seed,
            key_added="leiden_tmp",
        )
        leiden_assignments.append(
            adata_latent.obs["leiden_tmp"].rename(str(seed)).copy()
        )

    leiden_df = pd.concat(leiden_assignments, axis=1)
    latent_path = out_dir / "latent.csv"
    leiden_path = out_dir / "leiden.csv"
    adata_latent.to_df().to_csv(latent_path)
    leiden_df.to_csv(leiden_path)
    adata_latent.obs.drop(columns=["leiden_tmp"], errors="ignore", inplace=True)
    LOGGER.info(
        "Repeated Leiden completed in %.1fs; wrote %s and %s",
        time.time() - start,
        latent_path,
        leiden_path,
    )
    return leiden_df


def run_consensus(
    args: argparse.Namespace, config: dict, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Running consensus clustering: %s %s", args.consensus_script, out_dir)
    start = time.time()
    run_r_figure(args.consensus_script, [str(out_dir)])
    LOGGER.info("Consensus clustering completed in %.1fs", time.time() - start)

    expected = [
        out_dir / "consensus_summary.csv",
        out_dir / "consensus_scores_by_k.csv",
        out_dir / "leiden_cons.csv",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Consensus command finished but expected outputs are missing: {missing}"
        )
    return pd.read_csv(out_dir / "consensus_summary.csv"), pd.read_csv(
        out_dir / "consensus_scores_by_k.csv"
    )


def attach_consensus(
    model,
    adata_latent: sc.AnnData,
    consensus_summary: pd.DataFrame,
    out_dir: Path,
    config: dict,
) -> str:
    leiden_ks = sorted(
        int(path.stem.split("_")[-1]) for path in out_dir.glob("leiden_cons_k_*.csv")
    )
    if not leiden_ks:
        raise FileNotFoundError(f"No leiden_cons_k_*.csv files found in {out_dir}")

    for k in leiden_ks:
        df = pd.read_csv(out_dir / f"leiden_cons_k_{k}.csv", index_col="sample_id")
        df.index = df.index.astype(str)
        labels = df.loc[adata_latent.obs_names, "leiden_cons"].astype("category")
        adata_latent.obs[f"leiden_cons_{k}"] = labels
        model._cache.factor_adata.obs[f"leiden_cons_{k}"] = labels

    best_k = int(consensus_summary["selected_best_k"].iloc[0])
    default_k = best_k
    selected_sizes = (
        adata_latent.obs[f"leiden_cons_{best_k}"].astype(str).value_counts()
    )
    threshold = int(config.get("small_cluster_threshold", 10))
    if selected_sizes.min() <= threshold and (best_k - 1) in leiden_ks:
        default_k = best_k - 1
        LOGGER.info(
            "Selected k=%s has minimum cluster size %s <= %s; using k=%s as default handoff clustering",
            best_k,
            int(selected_sizes.min()),
            threshold,
            default_k,
        )
    cluster_key = f"leiden_cons_{default_k}"
    adata_latent.obs["leiden_cons"] = adata_latent.obs[cluster_key].astype("category")
    model._cache.factor_adata.obs["leiden_cons"] = adata_latent.obs["leiden_cons"]
    LOGGER.info(
        "Consensus k values=%s best_k=%s default_k=%s cluster_key=%s",
        leiden_ks,
        best_k,
        default_k,
        cluster_key,
    )
    LOGGER.info(
        "Cluster sizes:\n%s", adata_latent.obs[cluster_key].value_counts().sort_index()
    )
    adata_latent.obs.to_csv(out_dir / "adata_latent_obs.csv")
    return cluster_key


def coerce_event(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "1.0": 1,
        "true": 1,
        "yes": 1,
        "dead": 1,
        "deceased": 1,
        "event": 1,
        "progressed": 1,
        "0": 0,
        "0.0": 0,
        "false": 0,
        "no": 0,
        "alive": 0,
        "censored": 0,
    }
    return values.map(mapping)


def prepare_cluster_survival_df(
    adata: sc.AnnData, cluster_key: str, config: dict
) -> pd.DataFrame:
    duration_col = config["survival_duration_col"]
    event_col = config["survival_event_col"]
    keep_cols = [cluster_key, duration_col, event_col]
    if "UICC" in adata.obs.columns:
        keep_cols.append("UICC")

    df = adata.obs[keep_cols].copy()
    df[duration_col] = pd.to_numeric(df[duration_col], errors="coerce")
    df[event_col] = coerce_event(df[event_col])

    if config["exclude_uicc_iv"] and "UICC" in df.columns:
        df = df.loc[df["UICC"].astype(str) != "IV"].copy()

    df = df.dropna(subset=[cluster_key, duration_col, event_col]).copy()
    counts = df[cluster_key].astype(str).value_counts()
    keep_clusters = counts[counts >= config["min_cluster_size"]].index
    df = df.loc[df[cluster_key].astype(str).isin(keep_clusters)].copy()
    df[cluster_key] = df[cluster_key].astype(str).astype("category")
    return df


def run_survival(
    adata: sc.AnnData, cluster_key: str, config: dict, out_dir: Path
) -> None:
    LOGGER.info("Running survival analysis for %s", cluster_key)
    duration_col = config["survival_duration_col"]
    event_col = config["survival_event_col"]
    surv_df = prepare_cluster_survival_df(adata, cluster_key, config)
    if surv_df.empty or surv_df[cluster_key].nunique() < 2:
        LOGGER.warning("Skipping survival: insufficient complete samples/groups")
        pd.DataFrame(
            {"status": ["NA"], "reason": ["insufficient complete samples/groups"]}
        ).to_csv(out_dir / "survival_summary.csv", index=False)
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    kmfs = []
    for name, grouped_df in surv_df.groupby(cluster_key, observed=False):
        kmf = KaplanMeierFitter(alpha=0.5)
        kmf.fit(grouped_df[duration_col], grouped_df[event_col], label=str(name))
        kmf.plot_survival_function(ax=ax)
        kmfs.append(kmf)
    add_at_risk_counts(*kmfs, ax=ax)
    ax.set_title(f"Survival by {cluster_key}")
    fig.tight_layout()
    savefig(fig, out_dir / "km_survival_by_consensus_cluster.png")
    plt.close(fig)

    logrank = multivariate_logrank_test(
        surv_df[duration_col], surv_df[cluster_key], surv_df[event_col]
    )
    cox_df = pd.get_dummies(surv_df[[cluster_key]], prefix=cluster_key, drop_first=True)
    cox_df[duration_col] = surv_df[duration_col].values
    cox_df[event_col] = surv_df[event_col].values
    cph = CoxPHFitter().fit(cox_df, duration_col=duration_col, event_col=event_col)

    surv_df.to_csv(out_dir / "survival_input.csv")
    logrank.summary.to_csv(out_dir / "survival_logrank_summary.csv")
    cph.summary.to_csv(out_dir / "survival_cox_summary.csv")
    pd.DataFrame(
        [
            {
                "multivariate_logrank_p": logrank.p_value,
                "n": len(surv_df),
                "n_clusters": surv_df[cluster_key].nunique(),
            }
        ]
    ).to_csv(out_dir / "survival_summary.csv", index=False)
    LOGGER.info("Survival logrank p-value %.4g", logrank.p_value)


def save_umap_plots(adata: sc.AnnData, cluster_key: str, out_dir: Path) -> None:
    color = [
        c
        for c in [
            cluster_key,
            "leiden_cons_current",
            "cptac_prediction",
            "MSI consensus",
        ]
        if c in adata.obs.columns
    ]
    if not color or "X_umap" not in adata.obsm:
        return
    sc.pl.umap(adata, color=color, wspace=0.35, show=False)
    savefig(plt.gcf(), out_dir / "latent_umap_annotations.png")
    plt.close("all")


def write_handoff(
    args: argparse.Namespace,
    config: dict,
    model,
    adata_latent: sc.AnnData,
    out_dir: Path,
) -> None:
    handoff = args.handoff_dir
    LOGGER.info("Writing downstream handoff to %s", handoff)
    manifest_path = handoff / "handoff_manifest.json"
    if handoff.exists() and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        is_locked = bool(
            existing_manifest.get("locked_default_muvi_run")
            or existing_manifest.get("default_handoff_k")
        )
        if is_locked and not args.force_overwrite_locked_handoff:
            raise RuntimeError(
                f"Refusing to overwrite locked handoff {handoff}. "
                "Write only the timestamped result directory, or pass --force-overwrite-locked-handoff "
                "together with --overwrite-handoff if you intentionally want to replace the current handoff."
            )
        if not args.overwrite_handoff:
            raise RuntimeError(
                f"Handoff directory already exists: {handoff}. "
                "Use --overwrite-handoff for an unlocked handoff, or --no-write-handoff to keep only timestamped results."
            )
    if handoff.exists() and args.overwrite_handoff:
        backup = (
            handoff.parent / f"{handoff.name}_backup_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copytree(handoff, backup)
        LOGGER.info("Backed up existing handoff to %s", backup)
    handoff.mkdir(parents=True, exist_ok=True)

    factor_adata = model._cache.factor_adata.copy()
    factor_adata.obs = adata_latent.obs.copy()
    factor_adata.obsm["X_umap"] = adata_latent.obsm["X_umap"].copy()
    factor_adata.write_h5ad(handoff / "factor_adata.h5ad")
    factor_adata.obs.to_csv(handoff / "metadata.csv")
    adata_latent.to_df().to_csv(handoff / "Z.csv")
    adata_latent.obs.to_csv(handoff / "adata_latent_obs.csv")

    for pattern in [
        "config.json",
        "latent.csv",
        "leiden.csv",
        "leiden_single_run.csv",
        "consensus_summary.csv",
        "consensus_scores_by_k.csv",
        "leiden_cons.csv",
        "leiden_cons_k_*.csv",
        "Consensus_Heatmap.pdf",
        "SelectPartition.pdf",
        "Consensus_Scores.pdf",
        "Membership_Heatmap.pdf",
        "All_Collect_Plots.pdf",
    ]:
        for src in out_dir.glob(pattern):
            shutil.copy2(src, handoff / src.name)

    model_dst = handoff / "muvi_model"
    if model_dst.exists():
        shutil.rmtree(model_dst)
    if (out_dir / "muvi_model").exists():
        shutil.copytree(out_dir / "muvi_model", model_dst)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_results_dir": str(out_dir.resolve()),
        "input_h5mu": config["input_h5mu"],
        "cluster_col": "leiden_cons",
        "factor_adata": "factor_adata.h5ad",
        "metadata": "metadata.csv",
        "factors": "Z.csv",
    }
    (handoff / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    try:
        write_bid_indexed_exports(handoff)
    except Exception as exc:  # pragma: no cover - non-critical convenience export
        LOGGER.warning("Could not write BID-indexed handoff exports: %s", exc)


def main() -> None:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    seed_timestamp = time.strftime("%y%m%d%H%M")
    out_dir = args.out_dir or args.out_root / f"results_muvi_normics_{timestamp}"
    setup_logging(out_dir, args.log_level)
    set_house_style()
    # scanpy's set_figure_params rewrites figure.figsize, font.size and
    # savefig.dpi as well as dpi, so reclaim the house style after it.
    sc.settings.set_figure_params(facecolor="white")
    set_house_style()

    config = build_config(args, out_dir, seed_timestamp)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    LOGGER.info("Output directory: %s", out_dir)
    LOGGER.info("Input h5mu: %s", args.input_h5mu)

    prot_label = "Proteome"
    mut_label = "Mutation"
    view_names = [prot_label]

    LOGGER.info("Loading input MuData")
    mdata = mu.read_h5mu(args.input_h5mu)

    if not args.skip_old_new_check and args.old_h5mu.exists():
        LOGGER.info("Running old-vs-new proteome sanity check")
        mdata_old = mu.read_h5mu(args.old_h5mu)
        comparison = compare_old_new_by_accession(
            mdata_old, mdata, prot_label=prot_label
        )
        pd.Series(comparison["stats"]).to_csv(
            out_dir / "old_new_proteome_sanity_stats.csv", header=["value"]
        )
        comparison["diag_corr"].to_csv(out_dir / "old_new_diag_corr.csv")
        comparison["diag_spearman"].to_csv(out_dir / "old_new_diag_spearman.csv")
        plot_old_new_check(comparison, out_dir)

    attach_external_metadata(mdata, args)
    global_std = center_proteome(mdata, prot_label)
    config["global_std"] = global_std
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    model = train_muvi(mdata, config, out_dir, prot_label, mut_label, view_names)
    muvi.tl.variance_explained(model)
    attach_model_metadata(model, mdata, mut_label)
    adata_latent = run_base_latent_analysis(model, config, out_dir)
    run_repeated_leiden(adata_latent, config, out_dir)
    consensus_summary, consensus_scores = run_consensus(args, config, out_dir)
    consensus_summary.to_csv(out_dir / "consensus_summary.csv", index=False)
    consensus_scores.to_csv(out_dir / "consensus_scores_by_k.csv", index=False)
    cluster_key = attach_consensus(
        model, adata_latent, consensus_summary, out_dir, config
    )
    save_umap_plots(adata_latent, cluster_key, out_dir)

    if not args.skip_survival:
        run_survival(adata_latent, cluster_key, config, out_dir)

    if args.write_handoff:
        write_handoff(args, config, model, adata_latent, out_dir)

    LOGGER.info("Done. Results: %s", out_dir.resolve())


if __name__ == "__main__":
    main()
