"""Survival helpers for downstream analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lifelines.statistics import multivariate_logrank_test
except Exception:  # pragma: no cover
    multivariate_logrank_test = None


def normalize_event(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    x = s.astype(str).str.lower().str.strip()
    return x.map(
        {
            "1": 1,
            "0": 0,
            "event": 1,
            "dead": 1,
            "deceased": 1,
            "progression": 1,
            "progressed": 1,
            "alive": 0,
            "censored": 0,
            "no event": 0,
            "none": 0,
        }
    )


def survival_summary(
    df: pd.DataFrame, group_col: str, time_col: str, event_col: str, endpoint: str
) -> pd.DataFrame:
    use = df[[group_col, time_col, event_col]].dropna().copy()
    use[time_col] = pd.to_numeric(use[time_col], errors="coerce")
    use[event_col] = normalize_event(use[event_col])
    use = use.dropna()
    if use.empty or use[group_col].nunique() < 2:
        return pd.DataFrame()
    try:
        p = (
            multivariate_logrank_test(
                use[time_col], use[group_col], use[event_col]
            ).p_value
            if multivariate_logrank_test
            else np.nan
        )
    except Exception:
        p = np.nan
    rows = []
    for group, sub in use.groupby(group_col):
        rows.append(
            {
                "endpoint": endpoint,
                "grouping": group_col,
                "group": group,
                "n": len(sub),
                "events": int(sub[event_col].sum()),
                "median_time": sub[time_col].median(),
                "logrank_p_global": p,
            }
        )
    return pd.DataFrame(rows)
