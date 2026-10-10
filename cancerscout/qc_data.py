"""Shared data loading and helpers for QC panel scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from cancerscout.paths import ACTIVE_PROCESSED_NORMICS_DIR, ACTIVE_RAW_NORMICS_DIR, FIGURES_DIR

ROOT = Path(__file__).resolve().parent.parent

# 997-sample run (BID0751/BID0762 removed before Normics+limma)
NORM_DIR = ACTIVE_PROCESSED_NORMICS_DIR

# Raw matrix comes from original run (outliers present in raw, removed after)
_ORIG_DIR    = ACTIVE_RAW_NORMICS_DIR
RAW_BID_FILE = _ORIG_DIR / "intermediate/quantity_matrix_bid_only_best_rerun_raw_samples_by_protein.csv"

NORM_FILE  = NORM_DIR / "intermediate/normalized_normics_vsn_log2_bid_only_best_rerun.csv"
FINAL_FILE = NORM_DIR / "final/normalized_normics_vsn_log2_bid_only_plate_corrected_limma.csv"
META_FILE  = NORM_DIR / "intermediate/sample_metadata_selected.csv"

OUT_DIR = FIGURES_DIR / "quality_control"

_STAGE_COLORS = sns.color_palette("Set1", 3)
STAGE_COLOR = {
    "Raw":           _STAGE_COLORS[0],
    "Normics/VSN":   _STAGE_COLORS[1],
    "Normics+limma": _STAGE_COLORS[2],
}
STAGES = list(STAGE_COLOR)

_P20 = sns.color_palette("tab20", n_colors=20)
PLATE_COLOR = {str(i): _P20[i - 1] for i in range(1, 12)}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw_l2, norm, final, meta_bid) — all samples × proteins, log2."""
    raw = pd.read_csv(RAW_BID_FILE, index_col=0).astype(float)
    raw_l2 = np.log2(raw.replace(0.0, np.nan))

    norm  = pd.read_csv(NORM_FILE,  index_col=0).T
    final = pd.read_csv(FINAL_FILE, index_col=0).T

    meta = pd.read_csv(META_FILE, index_col=0)
    meta_bid = meta[meta["sample_group"] == "BID"].copy()
    meta_bid["Plate"] = meta_bid["Plate"].astype(int).astype(str)

    return raw_l2, norm, final, meta_bid


def aligned(df: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    shared = df.index.intersection(meta.index)
    return df.loc[shared], meta.loc[shared, "Plate"]


def sorted_plates(plates: pd.Series) -> list[str]:
    return sorted(plates.unique(), key=int)


def pca2(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-protein median impute → scale → PCA(2). df: samples × proteins."""
    x = df.fillna(df.median(axis=0, skipna=True))
    x = x.loc[:, x.std(axis=0) > 0]
    xs = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(xs), pca.explained_variance_ratio_


def kde_line(ax: plt.Axes, values: np.ndarray, color, label: str) -> None:
    from scipy.stats import gaussian_kde
    v = values[~np.isnan(values)]
    kde = gaussian_kde(v, bw_method=0.15)
    x = np.linspace(0, np.percentile(v, 99.5), 500)
    ax.plot(x, kde(x), color=color, lw=2, label=label)
    ax.axvline(float(np.median(v)), color=color, lw=1, ls="--", alpha=0.55)


def label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")
