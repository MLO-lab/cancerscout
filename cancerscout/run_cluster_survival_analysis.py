"""Cluster-level survival analysis for the locked CancerScout solution.

This script runs post-hoc Kaplan-Meier and Cox analyses for the final six
proteomic clusters. It reads existing downstream metadata only; it does not
rerun MuVI, clustering, or marker selection.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from cancerscout.plot_style import (
    CLUSTER_SHORT_NAMES,
    set_house_style,
    export_style,
)
from cancerscout.plot_survival_combined_forest import render_combined_os_forest
from cancerscout.paths import (
    ACTIVE_MODEL_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SURVIVAL_RESULTS_DIR,
)
from cancerscout.io import save_text
from cancerscout.plotting import run_r_figure
from cancerscout.survival import normalize_event


DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR
DEFAULT_OUT_DIR = SURVIVAL_RESULTS_DIR
DEFAULT_FIGURES_DIR = FIGURES_DIR / "survival"


ENDPOINTS = {
    "OS": ("OS Monate", "OS Status"),
    "PFS": ("PFS Monate", "PFS Status"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KM and Cox survival analyses for locked proteomic clusters."
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--cluster-column", default="leiden_cons")
    parser.add_argument("--min-cluster-events", type=int, default=3)
    parser.add_argument("--cox-penalizer", type=float, default=0.05)
    return parser.parse_args()


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def uicc_major(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if not text:
        return np.nan
    if text.startswith("IV"):
        return "IV"
    if text.startswith("III"):
        return "III"
    if text.startswith("II"):
        return "II"
    if text.startswith("I"):
        return "I"
    return text


def make_sample_table(results_dir: Path, cluster_col: str) -> pd.DataFrame:
    meta_path = results_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = pd.read_csv(meta_path)
    if cluster_col not in meta.columns:
        raise KeyError(f"Cluster column not found in metadata: {cluster_col}")

    bid_col = first_existing(meta, ["Proteome:new_bid", "new_bid", "bid"])
    if bid_col:
        sample_id = meta[bid_col].astype(str)
    else:
        idx_col = first_existing(meta, ["Unnamed: 0"])
        sample_id = "BID" + pd.to_numeric(meta[idx_col], errors="coerce").fillna(
            -1
        ).astype(int).astype(str).str.zfill(4)

    cols = {
        "sample_id": sample_id,
        "cluster": meta[cluster_col].astype("Int64").astype(str),
        "age": pd.to_numeric(
            meta[
                first_existing(
                    meta,
                    [
                        "Alter bei Diagnose",
                        "Proteome:Alter bei Diagnose",
                        "Proteome:new_age",
                    ],
                )
            ],
            errors="coerce",
        ),
        "sex": meta[
            first_existing(
                meta, ["Geschlecht", "Proteome:Geschlecht", "Proteome:new_sex"]
            )
        ]
        .astype(str)
        .str.strip(),
        "uicc": meta[
            first_existing(meta, ["UICC", "Proteome:UICC", "Proteome:new_UICC"])
        ],
        "grade": meta[first_existing(meta, ["G", "Proteome:G", "Proteome:new_G"])],
        "localization": meta[
            first_existing(
                meta,
                ["Lokalisation", "Proteome:Lokalisation", "Proteome:new_Lokalisation"],
            )
        ],
        "tumor_content": pd.to_numeric(
            meta[
                first_existing(
                    meta,
                    [
                        "Tumorzellgehalt (Mol+Prot)",
                        "Proteome:Tumorzellgehalt (Mol+Prot)",
                    ],
                )
            ],
            errors="coerce",
        ),
        "missing_fraction": pd.to_numeric(
            meta[
                first_existing(
                    meta, ["Proteome:new_missing_fraction", "new_missing_fraction"]
                )
            ],
            errors="coerce",
        ),
        "msi_consensus": meta[
            first_existing(
                meta,
                [
                    "MSI consensus",
                    "Proteome:new_MSI_consensus",
                    "MSI Status",
                    "Proteome:MSI Status",
                ],
            )
        ],
        "cptac_prediction": meta[
            first_existing(meta, ["cptac_prediction", "final_cptac_prediction"])
        ],
    }
    for endpoint, (time_col, event_col) in ENDPOINTS.items():
        cols[f"{endpoint}_time"] = (
            pd.to_numeric(meta[time_col], errors="coerce")
            if time_col in meta
            else np.nan
        )
        cols[f"{endpoint}_event"] = (
            normalize_event(meta[event_col]) if event_col in meta else np.nan
        )
    out = pd.DataFrame(cols)
    out["uicc_major"] = out["uicc"].map(uicc_major)
    out["cluster"] = out["cluster"].replace({"<NA>": np.nan, "nan": np.nan})
    out["short_name"] = out["cluster"].map(
        lambda c: CLUSTER_SHORT_NAMES.get(str(c), np.nan)
    )
    out["cluster_label"] = (
        "C" + out["cluster"].astype(str) + " " + out["short_name"].fillna("")
    )
    return out


def metadata_audit(raw_meta: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("OS", "time", "OS Monate"),
        ("OS", "event", "OS Status"),
        ("PFS", "time", "PFS Monate"),
        ("PFS", "event", "PFS Status"),
        ("clinical", "age", "Alter bei Diagnose"),
        ("clinical", "sex", "Geschlecht"),
        ("clinical", "stage", "UICC"),
        ("clinical", "grade", "G"),
        ("clinical", "localization", "Lokalisation"),
        ("clinical", "tumor_content", "Tumorzellgehalt (Mol+Prot)"),
        ("technical", "missing_fraction", "Proteome:new_missing_fraction"),
        ("biology", "MSI", "MSI consensus"),
        ("biology", "CPTAC", "cptac_prediction"),
        ("cluster", "final_cluster", "leiden_cons"),
    ]
    rows = []
    for domain, role, col in candidates:
        if col in raw_meta.columns:
            s = raw_meta[col]
            rows.append(
                {
                    "domain": domain,
                    "role": role,
                    "source_column": col,
                    "available": True,
                    "n_nonmissing": int(s.notna().sum()),
                    "n_unique": int(s.nunique(dropna=True)),
                    "example_values": "; ".join(
                        map(str, s.dropna().astype(str).unique()[:8])
                    ),
                }
            )
        else:
            rows.append(
                {
                    "domain": domain,
                    "role": role,
                    "source_column": col,
                    "available": False,
                    "n_nonmissing": 0,
                    "n_unique": 0,
                    "example_values": "",
                }
            )
    for endpoint in ENDPOINTS:
        use = endpoint_data(sample, endpoint)
        rows.append(
            {
                "domain": "derived",
                "role": f"{endpoint}_analysis_set",
                "source_column": f"{endpoint}_time/{endpoint}_event",
                "available": not use.empty,
                "n_nonmissing": len(use),
                "n_unique": int(use["cluster"].nunique()) if not use.empty else 0,
                "example_values": f"events={int(use['event'].sum()) if not use.empty else 0}",
            }
        )
    return pd.DataFrame(rows)


def endpoint_data(
    df: pd.DataFrame,
    endpoint: str,
    covariates: list[str] | None = None,
    require_positive_time: bool = False,
) -> pd.DataFrame:
    cols = [
        "sample_id",
        "cluster",
        "cluster_label",
        "short_name",
        f"{endpoint}_time",
        f"{endpoint}_event",
    ]
    if covariates:
        cols += covariates
    cols = list(dict.fromkeys(cols))
    use = df[cols].copy()
    use = use.rename(columns={f"{endpoint}_time": "time", f"{endpoint}_event": "event"})
    use["time"] = pd.to_numeric(use["time"], errors="coerce")
    use["event"] = pd.to_numeric(use["event"], errors="coerce")
    use = use.dropna(subset=["cluster", "time", "event"])
    use = use.loc[use["time"] >= 0].copy()
    if require_positive_time:
        use = use.loc[use["time"] > 0].copy()
    return use


def km_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        use = endpoint_data(df, endpoint)
        if use.empty or use["cluster_label"].nunique() < 2:
            continue
        try:
            logrank_p = multivariate_logrank_test(
                use["time"], use["cluster_label"], use["event"]
            ).p_value
        except Exception:
            logrank_p = np.nan
        kmf = KaplanMeierFitter()
        for label, sub in use.groupby("cluster_label", sort=True):
            try:
                kmf.fit(sub["time"], sub["event"])
                median_survival = kmf.median_survival_time_
            except Exception:
                median_survival = np.nan
            rows.append(
                {
                    "endpoint": endpoint,
                    "cluster_label": label,
                    "cluster": sub["cluster"].iloc[0],
                    "n": len(sub),
                    "events": int(sub["event"].sum()),
                    "event_fraction": sub["event"].mean(),
                    "median_followup_or_event_time": sub["time"].median(),
                    "km_median_survival": median_survival,
                    "global_logrank_p": logrank_p,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        pvals = out[["endpoint", "global_logrank_p"]].drop_duplicates()[
            "global_logrank_p"
        ]
        if pvals.notna().any():
            endpoint_q = dict(
                zip(
                    out[["endpoint", "global_logrank_p"]].drop_duplicates()["endpoint"],
                    multipletests(pvals.fillna(1), method="fdr_bh")[1],
                )
            )
            out["global_logrank_q_across_endpoints"] = out["endpoint"].map(endpoint_q)
    return out


def build_km_curve_points(
    df: pd.DataFrame, endpoints: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-cluster KM step coordinates for each endpoint.

    Returns a long-format frame (endpoint, cluster, time, survival_prob,
    n_at_risk, n_total, events_total) plus the global multivariate log-rank p
    per endpoint.
    """
    rows: list[dict[str, object]] = []
    logrank: dict[str, float] = {}
    for endpoint in endpoints:
        use = endpoint_data(df, endpoint)
        if use.empty or use["cluster"].nunique() < 2:
            continue
        try:
            logrank[endpoint] = float(
                multivariate_logrank_test(
                    use["time"], use["cluster_label"], use["event"]
                ).p_value
            )
        except Exception:
            logrank[endpoint] = float("nan")
        for cluster, sub in use.groupby("cluster", sort=True):
            sub = sub.sort_values("time")
            kmf = KaplanMeierFitter()
            try:
                kmf.fit(sub["time"], sub["event"])
            except Exception:
                continue
            survival = kmf.survival_function_.reset_index()
            survival.columns = ["time", "survival_prob"]
            n_at_risk_series = sub["time"].sort_values().reset_index(drop=True)
            n_total = int(len(sub))
            events_total = int(sub["event"].sum())
            for t, p in zip(survival["time"], survival["survival_prob"]):
                n_at_risk = int((n_at_risk_series >= t).sum())
                rows.append(
                    {
                        "endpoint": endpoint,
                        "cluster": str(cluster),
                        "time": float(t),
                        "survival_prob": float(p),
                        "n_at_risk": n_at_risk,
                        "n_total": n_total,
                        "events_total": events_total,
                    }
                )
    return pd.DataFrame(rows), logrank


def build_km_risk_table(
    df: pd.DataFrame,
    endpoints: list[str],
    time_points: tuple[int, ...] = (0, 12, 24, 36, 48, 60),
) -> pd.DataFrame:
    """n_at_risk at discrete time points per (endpoint, cluster) for the risk table panel."""
    rows: list[dict[str, object]] = []
    for endpoint in endpoints:
        use = endpoint_data(df, endpoint)
        if use.empty:
            continue
        for cluster, sub in use.groupby("cluster", sort=True):
            times = sub["time"].to_numpy()
            for t in time_points:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "cluster": str(cluster),
                        "time": int(t),
                        "n_at_risk": int(np.sum(times >= t)),
                    }
                )
    return pd.DataFrame(rows)


def cox_fit(
    data: pd.DataFrame,
    duration_col: str = "time",
    event_col: str = "event",
    penalizer: float = 0.05,
) -> CoxPHFitter | None:
    if data.empty or data[event_col].sum() < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CoxPHFitter(penalizer=penalizer).fit(
                data, duration_col=duration_col, event_col=event_col
            )
    except Exception:
        return None


def prep_cox_covariates(use: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    cox = use[["time", "event"] + covariates].copy()
    for col in covariates:
        if col in {"age", "tumor_content", "missing_fraction"}:
            cox[col] = pd.to_numeric(cox[col], errors="coerce")
            sd = cox[col].std(skipna=True)
            if pd.notna(sd) and sd > 0:
                cox[col] = (cox[col] - cox[col].mean()) / sd
        else:
            cox[col] = cox[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    cox = cox.dropna()
    cat_cols = [
        c for c in covariates if c not in {"age", "tumor_content", "missing_fraction"}
    ]
    cox = pd.get_dummies(cox, columns=cat_cols, drop_first=True, dtype=float)
    return cox


def extract_term(
    fit: CoxPHFitter, term: str, extra: dict[str, object]
) -> dict[str, object]:
    summary = fit.summary.loc[term]
    return {
        **extra,
        "term": term,
        "hazard_ratio": float(summary["exp(coef)"]),
        "ci_lower": float(summary["exp(coef) lower 95%"]),
        "ci_upper": float(summary["exp(coef) upper 95%"]),
        "p": float(summary["p"]),
        "coef": float(summary["coef"]),
    }


def one_vs_rest_cox(
    df: pd.DataFrame,
    endpoint: str,
    covariates: list[str],
    penalizer: float,
    min_events: int,
) -> pd.DataFrame:
    rows = []
    use = endpoint_data(df, endpoint, covariates=covariates, require_positive_time=True)
    for cluster, sub_all in use.groupby("cluster", sort=True):
        cluster_label = sub_all["cluster_label"].iloc[0]
        cluster_short_name = (
            sub_all["short_name"].iloc[0] if "short_name" in sub_all else cluster_label
        )
        tmp = use.copy()
        tmp["cluster_indicator"] = (tmp["cluster"].astype(str) == str(cluster)).astype(
            int
        )
        n_cluster = int(tmp["cluster_indicator"].sum())
        events_cluster = int(tmp.loc[tmp["cluster_indicator"].eq(1), "event"].sum())
        if events_cluster < min_events:
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": "cluster_vs_rest_adjusted",
                    "cluster": cluster,
                    "cluster_label": cluster_label,
                    "cluster_short_name": cluster_short_name,
                    "n": len(tmp),
                    "n_cluster": n_cluster,
                    "events_cluster": events_cluster,
                    "status": "skipped_few_cluster_events",
                }
            )
            continue
        cox = prep_cox_covariates(tmp, covariates)
        cox["cluster_indicator"] = tmp.loc[cox.index, "cluster_indicator"].astype(float)
        fit = cox_fit(cox, penalizer=penalizer)
        if fit is None or "cluster_indicator" not in fit.summary.index:
            rows.append(
                {
                    "endpoint": endpoint,
                    "model": "cluster_vs_rest_adjusted",
                    "cluster": cluster,
                    "cluster_label": cluster_label,
                    "cluster_short_name": cluster_short_name,
                    "n": len(cox),
                    "n_cluster": int(cox["cluster_indicator"].sum())
                    if "cluster_indicator" in cox
                    else n_cluster,
                    "events_cluster": events_cluster,
                    "status": "fit_failed",
                }
            )
            continue
        rows.append(
            extract_term(
                fit,
                "cluster_indicator",
                {
                    "endpoint": endpoint,
                    "model": "cluster_vs_rest_adjusted",
                    "cluster": cluster,
                    "cluster_label": cluster_label,
                    "cluster_short_name": cluster_short_name,
                    "n": len(cox),
                    "events": int(cox["event"].sum()),
                    "n_cluster": int(cox["cluster_indicator"].sum()),
                    "events_cluster": events_cluster,
                    "status": "ok",
                },
            )
        )
    out = pd.DataFrame(rows)
    ok = (
        out["status"].eq("ok") if "status" in out else pd.Series(False, index=out.index)
    )
    out["q_within_model"] = np.nan
    if ok.any():
        out.loc[ok, "q_within_model"] = multipletests(
            out.loc[ok, "p"], method="fdr_bh"
        )[1]
    return out


def global_cluster_deviation_cox(
    df: pd.DataFrame, endpoint: str, covariates: list[str], penalizer: float
) -> pd.DataFrame:
    """One mutually-adjusted Cox model reporting every cluster, with no reference class.

    Uses sum-to-zero (deviation) coding rather than treatment dummies. For the
    k-1 explicitly coded clusters the design column is +1 in that cluster, -1 in
    the omitted baseline cluster, and 0 elsewhere, so each coefficient is the log
    hazard ratio against the unweighted mean of the six cluster log-hazards. The
    baseline cluster's effect is recovered as the negative sum of the others,
    with its standard error from the full covariance submatrix (delta method),
    so it gets a real confidence interval instead of being pinned at HR = 1.
    """
    use = endpoint_data(df, endpoint, covariates=covariates, require_positive_time=True)
    if use.empty:
        return pd.DataFrame()
    cox = prep_cox_covariates(use, covariates)
    cluster_series = use.loc[cox.index, "cluster"].astype(str)
    levels = sorted(cluster_series.unique())
    if len(levels) < 2:
        return pd.DataFrame()
    baseline, coded = levels[0], levels[1:]

    design = {}
    for level in coded:
        design[f"cluster_dev_{level}"] = np.where(
            cluster_series.eq(level),
            1.0,
            np.where(cluster_series.eq(baseline), -1.0, 0.0),
        )
    cox = pd.concat([cox, pd.DataFrame(design, index=cox.index)], axis=1)

    fit = cox_fit(cox, penalizer=penalizer)
    if fit is None:
        return pd.DataFrame()

    terms = [f"cluster_dev_{level}" for level in coded]
    if not set(terms).issubset(fit.summary.index):
        return pd.DataFrame()

    label_lookup = (
        use.loc[cox.index, ["cluster", "cluster_label"]].astype(str).drop_duplicates()
    )
    labels = dict(zip(label_lookup["cluster"], label_lookup["cluster_label"]))
    short_lookup = (
        use.loc[cox.index, ["cluster", "short_name"]].astype(str).drop_duplicates()
        if "short_name" in use.columns
        else pd.DataFrame(columns=["cluster", "short_name"])
    )
    shorts = dict(zip(short_lookup["cluster"], short_lookup.get("short_name", [])))

    common = {
        "endpoint": endpoint,
        "model": "global_cluster_deviation",
        "coding": "deviation_sum_to_zero",
        "contrast": "vs cohort grand mean",
        "penalizer": penalizer,
        "n": len(cox),
        "events": int(cox["event"].sum()),
    }
    cluster_counts = cluster_series.value_counts()
    events_by_cluster = cox["event"].groupby(cluster_series).sum()

    def cluster_row(level: str, coef: float, se: float) -> dict[str, object]:
        z = coef / se if se > 0 else np.nan
        return {
            **common,
            "term_kind": "cluster",
            "term": f"cluster_dev_{level}",
            "cluster": level,
            "cluster_label": labels.get(level, level),
            "cluster_short_name": shorts.get(level, labels.get(level, level)),
            "n_cluster": int(cluster_counts.get(level, 0)),
            "events_cluster": int(events_by_cluster.get(level, 0)),
            "hazard_ratio": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - 1.96 * se)),
            "ci_upper": float(np.exp(coef + 1.96 * se)),
            "p": float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "coef": float(coef),
            "se": float(se),
            "status": "ok",
        }

    variance = fit.variance_matrix_.loc[terms, terms].to_numpy(dtype=float)
    coefs = fit.params_[terms].to_numpy(dtype=float)
    rows = [
        cluster_row(level, coefs[i], float(np.sqrt(variance[i, i])))
        for i, level in enumerate(coded)
    ]
    # Baseline cluster: beta = -sum(beta_coded), Var = 1' V 1.
    ones = np.ones(len(coded))
    rows.insert(
        0,
        cluster_row(
            baseline,
            float(-coefs.sum()),
            float(np.sqrt(ones @ variance @ ones)),
        ),
    )

    for term in fit.summary.index:
        if term in terms:
            continue
        rows.append(
            extract_term(
                fit, term, {**common, "term_kind": "covariate", "status": "ok"}
            )
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["term_kind", "cluster"], na_position="last").reset_index(
        drop=True
    )
    # One BH family over the cluster terms only, matching the one-vs-rest convention.
    out["q_within_model"] = np.nan
    is_cluster = out["term_kind"].eq("cluster")
    if is_cluster.any():
        out.loc[is_cluster, "q_within_model"] = multipletests(
            out.loc[is_cluster, "p"].fillna(1), method="fdr_bh"
        )[1]
    return out


def adjusted_covariate_label_map(
    df: pd.DataFrame, endpoint: str, covariates: list[str]
) -> dict[str, str]:
    """Human-readable labels for standardized continuous Cox covariates."""
    use = endpoint_data(df, endpoint, covariates=covariates, require_positive_time=True)
    if use.empty:
        return {}
    fit_cols = ["time", "event"] + covariates
    raw = use[fit_cols].copy()
    for col in covariates:
        if col in {"age", "tumor_content", "missing_fraction"}:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
        else:
            raw[col] = raw[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    raw = raw.dropna()

    labels = {
        "sex_male": "Sex: male vs female",
        "uicc_major_II": "UICC II vs I",
        "uicc_major_III": "UICC III vs I",
        "uicc_major_IV": "UICC IV vs I",
    }
    if "age" in raw:
        sd = raw["age"].std(skipna=True)
        if pd.notna(sd) and sd > 0:
            labels["age"] = f"Age (+{sd:.1f} years)"
    if "tumor_content" in raw:
        sd = raw["tumor_content"].std(skipna=True)
        if pd.notna(sd) and sd > 0:
            if raw["tumor_content"].max(skipna=True) <= 1.5:
                labels["tumor_content"] = f"Tumor content (+{sd * 100:.1f} pct points)"
            else:
                labels["tumor_content"] = f"Tumor content (+{sd:.1f} points)"
    return labels


def clinical_covariate_forest_rows(
    cox: pd.DataFrame, endpoint: str, label_map: dict[str, str] | None = None
) -> pd.DataFrame:
    """Clinical-covariate rows from one global cluster-adjusted Cox model."""
    if cox.empty:
        return pd.DataFrame()
    label_map = label_map or {}
    rows = cox.loc[
        cox["endpoint"].eq(endpoint)
        & cox["model"].eq("global_cluster_deviation")
        & cox["term_kind"].eq("covariate")
    ].copy()
    if rows.empty:
        return rows
    rows["source_model"] = rows["model"]
    rows["model"] = "clinical_covariates_adjusted"
    rows["plot_label"] = (
        rows["term"].astype(str).map(label_map).fillna(rows["term"].astype(str))
    )
    return rows


def metadata_univariable_cox(
    df: pd.DataFrame, endpoint: str, penalizer: float
) -> pd.DataFrame:
    covariates = ["cluster_label", "age", "sex", "uicc_major", "tumor_content"]
    use = endpoint_data(df, endpoint, covariates=covariates, require_positive_time=True)
    rows = []
    for cov in covariates:
        sub = prep_cox_covariates(use, [cov])
        if len(sub) < 20 or sub["event"].sum() < 5:
            continue
        fit = cox_fit(sub, penalizer=penalizer)
        if fit is None:
            continue
        for term in fit.summary.index:
            rows.append(
                extract_term(
                    fit,
                    term,
                    {
                        "endpoint": endpoint,
                        "model": "metadata_univariable",
                        "covariate": cov,
                        "n": len(sub),
                        "events": int(sub["event"].sum()),
                    },
                )
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_within_endpoint"] = multipletests(out["p"].fillna(1), method="fdr_bh")[1]
    return out


# Figure rendering moved to r/km_survival_curves.R and r/cox_forest_plot.R.
# Compute helpers above prepare flat TSVs that those scripts consume.


def write_report(
    out_dir: Path,
    km: pd.DataFrame,
    cox: pd.DataFrame,
    meta: pd.DataFrame,
    sample: pd.DataFrame,
    covariates: list[str],
) -> None:
    lines = [
        "# Cluster Survival Analysis",
        "",
        "This analysis uses the locked six-cluster solution and existing clinical metadata. It is exploratory and intended to assess whether proteomic clusters show survival differences before and after adjustment for major clinical/sample covariates.",
        "",
        "## Inputs",
        "",
        "- Clusters: `leiden_cons` from the active model `metadata.csv`.",
        "- Endpoints: OS = `OS Monate` / `OS Status`; PFS = `PFS Monate` / `PFS Status`.",
        f"- Base adjusted Cox covariates: {', '.join(covariates)}.",
        "- Tumor content is included as a sample-composition covariate.",
        "",
        "## Endpoint Availability",
        "",
    ]
    for endpoint in ENDPOINTS:
        use = endpoint_data(sample, endpoint)
        lines.append(
            f"- {endpoint}: n={len(use)}, events={int(use['event'].sum()) if not use.empty else 0}."
        )
    lines += ["", "## Kaplan-Meier / Log-Rank", ""]
    if km.empty:
        lines.append("No Kaplan-Meier summaries were available.")
    else:
        for endpoint, sub in km.groupby("endpoint"):
            p = (
                sub["global_logrank_p"].dropna().iloc[0]
                if sub["global_logrank_p"].notna().any()
                else np.nan
            )
            q = (
                sub["global_logrank_q_across_endpoints"].dropna().iloc[0]
                if "global_logrank_q_across_endpoints" in sub
                and sub["global_logrank_q_across_endpoints"].notna().any()
                else np.nan
            )
            lines.append(
                f"- {endpoint}: global log-rank p={p:.3g}, endpoint-level q={q:.3g}."
            )
    lines += ["", "## Adjusted Cluster Cox Models", ""]
    if not cox.empty:
        ok_mask = (
            cox["status"].fillna("ok").eq("ok")
            if "status" in cox
            else np.ones(len(cox), dtype=bool)
        )
        ok = cox.loc[ok_mask].copy()
    else:
        ok = pd.DataFrame()
    cluster_ok = (
        ok.loc[ok["model"].eq("cluster_vs_rest_adjusted")]
        if not ok.empty
        else pd.DataFrame()
    )
    if cluster_ok.empty:
        lines.append("No adjusted cluster-vs-rest Cox estimates were available.")
    else:
        sig = cluster_ok.loc[cluster_ok["q_within_model"].le(0.10)]
        if sig.empty:
            lines.append(
                "No cluster-vs-rest adjusted Cox association passed FDR q<=0.10."
            )
        else:
            for _, r in sig.sort_values(["endpoint", "q_within_model"]).iterrows():
                label = r.get("cluster_label", f"C{r['cluster']}")
                lines.append(
                    f"- {r['endpoint']} {label} vs rest: HR={r['hazard_ratio']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}], p={r['p']:.3g}, q={r['q_within_model']:.3g}."
                )
    lines += ["", "## Global Deviation-Coded Cox Model (no reference class)", ""]
    deviation = (
        ok.loc[
            ok["model"].eq("global_cluster_deviation") & ok["term_kind"].eq("cluster")
        ]
        if not ok.empty and "term_kind" in ok
        else pd.DataFrame()
    )
    if deviation.empty:
        lines.append("No deviation-coded global Cox estimates were available.")
    else:
        lines.append(
            "One mutually-adjusted model per endpoint using sum-to-zero coding, so every "
            "cluster is reported against the unweighted cohort grand mean rather than "
            "against a reference cluster. BH correction runs over the six cluster terms."
        )
        for endpoint in sorted(deviation["endpoint"].unique()):
            sub = deviation.loc[deviation["endpoint"].eq(endpoint)].sort_values(
                "cluster"
            )
            lines += ["", f"### {endpoint}", ""]
            for _, r in sub.iterrows():
                label = r.get("cluster_label", f"C{r['cluster']}")
                lines.append(
                    f"- {label}: HR={r['hazard_ratio']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}], "
                    f"p={r['p']:.3g}, q={r['q_within_model']:.3g} (n={int(r['n_cluster'])}, events={int(r['events_cluster'])})."
                )
    skipped = (
        cox.loc[
            cox.get("status", pd.Series(dtype=str)).eq("skipped_few_cluster_events")
        ]
        if not cox.empty and "status" in cox
        else pd.DataFrame()
    )
    if not skipped.empty:
        lines += ["", "Models skipped for too few within-cluster events:"]
        for _, r in skipped.iterrows():
            lines.append(
                f"- {r['endpoint']} C{r['cluster']}: events in cluster={int(r['events_cluster'])}."
            )
    lines += ["", "## Metadata Univariable Cox Screen", ""]
    if meta.empty:
        lines.append("No metadata Cox screen was available.")
    else:
        sig = meta.loc[meta["q_within_endpoint"].le(0.10)]
        if sig.empty:
            lines.append(
                "No metadata term passed FDR q<=0.10 in the univariable screen."
            )
        else:
            for _, r in (
                sig.sort_values(["endpoint", "q_within_endpoint"]).head(20).iterrows()
            ):
                lines.append(
                    f"- {r['endpoint']} {r['term']}: HR={r['hazard_ratio']:.2f}, p={r['p']:.3g}, q={r['q_within_endpoint']:.3g}."
                )
    lines += [
        "",
        "## Caveats",
        "",
        "- Cluster 6 remains a low-prevalence exploratory subgroup; survival estimates for it are unstable.",
        "- Adjusted Cox models are association models, not causal models.",
        "- MSI/CPTAC labels are reported in other analyses but are not part of the default survival forest plots to keep the focus on the six proteomic clusters.",
    ]
    save_text("\n".join(lines) + "\n", REPORTS_DIR / "survival_analysis.md")


def main() -> None:
    args = parse_args()
    set_house_style()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw_meta = pd.read_csv(results_dir / "metadata.csv")
    sample = make_sample_table(results_dir, args.cluster_column)
    sample.to_csv(out_dir / "survival_sample_table.tsv", sep="\t", index=False)
    metadata_audit(raw_meta, sample).to_csv(
        out_dir / "survival_metadata_audit.tsv", sep="\t", index=False
    )

    km = km_summary(sample)
    km.to_csv(out_dir / "cluster_survival_km_summary.tsv", sep="\t", index=False)

    # KM step coordinates and risk-table for the R renderer.
    km_points, logrank_p = build_km_curve_points(sample, list(ENDPOINTS))
    km_points_path = out_dir / "km_curve_points.tsv"
    km_points.to_csv(km_points_path, sep="\t", index=False)
    risk_path = out_dir / "km_risk_table.tsv"
    build_km_risk_table(sample, list(ENDPOINTS)).to_csv(
        risk_path, sep="\t", index=False
    )

    covariates = ["age", "sex", "uicc_major", "tumor_content"]
    cox_parts = []
    for endpoint in ENDPOINTS:
        cox_parts.append(
            one_vs_rest_cox(
                sample,
                endpoint,
                covariates,
                args.cox_penalizer,
                args.min_cluster_events,
            )
        )
        global_fit = global_cluster_deviation_cox(
            sample, endpoint, covariates, args.cox_penalizer
        )
        if not global_fit.empty:
            cox_parts.append(global_fit)
    cox = (
        pd.concat(
            [x for x in cox_parts if x is not None and not x.empty], ignore_index=True
        )
        if cox_parts
        else pd.DataFrame()
    )
    cox_path = out_dir / "cluster_survival_cox_models.tsv"
    cox.to_csv(cox_path, sep="\t", index=False)
    clinical_covariates_path = out_dir / "clinical_covariates_adjusted_cox_forest.tsv"
    label_map = adjusted_covariate_label_map(sample, "OS", covariates)
    clinical_covariate_forest_rows(cox, "OS", label_map).to_csv(
        clinical_covariates_path, sep="\t", index=False
    )

    meta_parts = [
        metadata_univariable_cox(sample, endpoint, args.cox_penalizer)
        for endpoint in ENDPOINTS
    ]
    meta = (
        pd.concat(
            [x for x in meta_parts if x is not None and not x.empty], ignore_index=True
        )
        if meta_parts
        else pd.DataFrame()
    )
    meta_path = out_dir / "metadata_univariable_survival_cox.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)

    # Sync the cluster-style TSV the R scripts read for palette/names.
    export_style(RESULTS_DIR / "plot_style")

    render_combined_os_forest(
        cox_path,
        clinical_covariates_path,
        out_dir / "combined_os_cox_forest.tsv",
        figures_dir / "cluster_survival_adjusted_cox_forest.png",
    )

    # ----- R figure dispatch -----
    for endpoint in ENDPOINTS:
        args_km = [
            "--points",
            str(km_points_path),
            "--risk",
            str(risk_path),
            "--endpoint",
            endpoint,
            "--output",
            str(figures_dir / f"km_{endpoint.lower()}_by_cluster.png"),
        ]
        p_value = logrank_p.get(endpoint)
        if p_value is not None and np.isfinite(p_value):
            args_km += ["--logrank-p", f"{p_value:.6g}"]
        run_r_figure("km_survival_curves.R", args_km)

    for endpoint in ENDPOINTS:
        run_r_figure(
            "cox_forest_plot.R",
            [
                "--tsv",
                str(cox_path),
                "--model",
                "cluster_vs_rest_adjusted",
                "--endpoint",
                endpoint,
                "--row-col",
                "cluster",
                "--output",
                str(
                    figures_dir
                    / f"cluster_survival_adjusted_cox_forest_{endpoint.lower()}.png"
                ),
            ],
        )
    run_r_figure(
        "cox_forest_plot.R",
        [
            "--tsv",
            str(cox_path),
            "--model",
            "global_cluster_deviation",
            "--endpoint",
            "OS",
            "--row-col",
            "cluster",
            "--output",
            str(figures_dir / "cluster_survival_global_deviation_cox_forest_os.png"),
            "--title",
            "All states vs cohort grand mean (mutually adjusted)",
        ],
    )
    run_r_figure(
        "cox_forest_plot.R",
        [
            "--tsv",
            str(clinical_covariates_path),
            "--model",
            "clinical_covariates_adjusted",
            "--endpoint",
            "OS",
            "--row-col",
            "term",
            "--output",
            str(figures_dir / "clinical_covariates_adjusted_cox_forest_os.png"),
            "--title",
            "Clinical covariates adjusted for CancerScout state",
        ],
    )

    write_report(out_dir, km, cox, meta, sample, covariates)
    print(f"Wrote survival analysis tables to {out_dir}")
    print(f"Wrote survival figures to {figures_dir}")


if __name__ == "__main__":
    main()
