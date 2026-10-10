"""Create reproducible train/validation/test splits for downstream modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from cancerscout.io import build_sample_id_map, read_metadata, write_tsv
from cancerscout.paths import ACTIVE_MODEL_DIR, MODEL_SPLITS_DIR


DEFAULT_RESULTS_DIR = ACTIVE_MODEL_DIR
DEFAULT_OUT_DIR = MODEL_SPLITS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build current cluster-stratified 60/20/20 downstream splits.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory containing current metadata.csv.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output split directory.")
    parser.add_argument("--cluster-column", default="leiden_cons", help="Cluster column used for stratification.")
    parser.add_argument("--seed", type=int, default=260510, help="Random seed.")
    parser.add_argument("--train-size", type=float, default=0.60, help="Fraction assigned to training.")
    parser.add_argument("--val-size", type=float, default=0.20, help="Fraction assigned to validation.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction assigned to test.")
    return parser.parse_args()


def write_split(ids: list[str], path: Path) -> None:
    pd.DataFrame({"analysis_sample_id": ids}).to_csv(path)


def summarize_splits(meta: pd.DataFrame, split_assignments: pd.Series, cluster_col: str) -> pd.DataFrame:
    rows = []
    labels = meta[cluster_col].astype(str)
    for split in ["train", "val", "test"]:
        ids = split_assignments.index[split_assignments.eq(split)]
        counts = labels.loc[ids].value_counts().sort_index()
        for cluster, n in counts.items():
            rows.append({"split": split, "cluster": cluster, "n": int(n), "fraction_within_split": n / len(ids)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = args.train_size + args.val_size + args.test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError("--train-size + --val-size + --test-size must equal 1.")

    meta = read_metadata(Path(args.results_dir))
    meta.index = meta.index.astype(str)
    if args.cluster_column not in meta.columns:
        raise KeyError(f"Missing cluster column: {args.cluster_column}")

    labels = meta[args.cluster_column].astype(str)
    valid = labels.notna() & ~labels.str.lower().isin(["nan", "none", ""])
    meta = meta.loc[valid].copy()
    labels = labels.loc[valid]

    train_val_ids, test_ids = train_test_split(
        labels.index.to_numpy(),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    remaining_labels = labels.loc[train_val_ids]
    val_fraction_of_remaining = args.val_size / (args.train_size + args.val_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction_of_remaining,
        random_state=args.seed + 1,
        stratify=remaining_labels,
    )

    train_ids = sorted(map(str, train_ids), key=lambda x: int(x) if x.isdigit() else x)
    val_ids = sorted(map(str, val_ids), key=lambda x: int(x) if x.isdigit() else x)
    test_ids = sorted(map(str, test_ids), key=lambda x: int(x) if x.isdigit() else x)

    write_split(train_ids, out_dir / "global_train_idx.csv")
    write_split(val_ids, out_dir / "global_val_idx.csv")
    write_split(test_ids, out_dir / "global_test_idx.csv")

    split_assignments = pd.Series(index=meta.index, dtype="object")
    split_assignments.loc[train_ids] = "train"
    split_assignments.loc[val_ids] = "val"
    split_assignments.loc[test_ids] = "test"

    sample_table = pd.DataFrame(
        {
            "analysis_sample_id": split_assignments.index,
            "split": split_assignments.values,
            "cluster": labels.loc[split_assignments.index].values,
        }
    )
    try:
        id_map = build_sample_id_map(meta)
        sample_table = sample_table.merge(id_map.reset_index(drop=True), on="analysis_sample_id", how="left")
    except Exception:
        pass
    write_tsv(sample_table, out_dir / "sample_split_assignments.tsv")

    summary = summarize_splits(meta, split_assignments, args.cluster_column)
    write_tsv(summary, out_dir / "split_cluster_counts.tsv")
    contingency = pd.crosstab(sample_table["split"], sample_table["cluster"])
    contingency.to_csv(out_dir / "split_cluster_contingency.tsv", sep="\t")
    chi2, p_value, dof, _ = chi2_contingency(contingency)

    config = {
        "results_dir": str(Path(args.results_dir)),
        "out_dir": str(out_dir),
        "cluster_column": args.cluster_column,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "n_samples": int(len(meta)),
        "n_train": int(len(train_ids)),
        "n_val": int(len(val_ids)),
        "n_test": int(len(test_ids)),
        "cluster_counts": labels.value_counts().sort_index().astype(int).to_dict(),
        "chi_square_split_cluster_balance": {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)},
    }
    (out_dir / "split_config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    print(f"Wrote current downstream splits to {out_dir}")
    print(contingency)


if __name__ == "__main__":
    main()
