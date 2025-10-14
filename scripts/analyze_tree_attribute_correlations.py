#!/usr/bin/env python3
"""Correlate random forest tree attributes with IRT-derived signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ATTR_COLUMNS = ["depth", "n_leaves", "oob_accuracy", "oob_count"]
TARGET_LABELS = {
    "theta": "Tree Ability (θ)",
    "tree_accuracy": "Tree Accuracy",
    "mean_disc_correct": "Mean Discrimination (Correct)",
    "mean_disc_error": "Mean Discrimination (Missed)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-attributes",
        type=Path,
        required=True,
        help="Path to JSON file produced by train_random_forest with per-tree attributes.",
    )
    parser.add_argument(
        "--irt-params",
        type=Path,
        required=True,
        help="Path to IRT parameter archive (e.g., data/irt_parameters_2pl.npz).",
    )
    parser.add_argument(
        "--response-matrix",
        type=Path,
        required=True,
        help="Path to response_matrix.npz used for the IRT fit (to compute per-tree accuracies).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/tree_attribute_correlations.json"),
        help="Where to save correlation summaries (JSON).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Where to save scatter plots.",
    )
    parser.add_argument(
        "--table-output",
        type=Path,
        default=None,
        help="Optional path to save the merged attribute table as CSV.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="tree",
        help="Prefix for generated figure filenames.",
    )
    return parser.parse_args()


def load_tree_attributes(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    frame = pd.DataFrame(data)
    if "tree_index" not in frame.columns:
        frame["tree_index"] = np.arange(len(frame))
    frame = frame.sort_values("tree_index").reset_index(drop=True)
    return frame


def load_irt_params(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    if "ability_loc" not in arrays:
        raise KeyError("ability_loc missing from IRT parameter archive")
    return arrays


def load_response_matrix(path: Path) -> np.ndarray:
    with np.load(path) as data:
        matrix = data["R"].astype(np.float32)
    return matrix


def per_tree_discrimination_stats(
    responses: np.ndarray,
    discrimination: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if discrimination is None:
        nan_array = np.full(responses.shape[0], np.nan, dtype=np.float32)
        return nan_array, nan_array

    correct = responses
    incorrect = 1.0 - responses
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_correct = (correct @ discrimination) / correct.sum(axis=1)
        mean_error = (incorrect @ discrimination) / incorrect.sum(axis=1)
    mean_correct = np.where(np.isfinite(mean_correct), mean_correct, np.nan)
    mean_error = np.where(np.isfinite(mean_error), mean_error, np.nan)
    return mean_correct.astype(np.float32), mean_error.astype(np.float32)


def correlation(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}
    xv = x_arr[valid]
    yv = y_arr[valid]
    pearson = float(np.corrcoef(xv, yv)[0, 1])
    rank_x = pd.Series(xv).rank(method="average").to_numpy()
    rank_y = pd.Series(yv).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4))
    plt.scatter(x[mask], y[mask], s=18, alpha=0.45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    tree_df = load_tree_attributes(args.tree_attributes)
    params = load_irt_params(args.irt_params)
    responses = load_response_matrix(args.response_matrix)

    ability = params["ability_loc"].astype(np.float32)
    if ability.shape[0] != tree_df.shape[0]:
        raise ValueError(
            "Mismatch between tree attribute count and ability count: "
            f"{tree_df.shape[0]} vs {ability.shape[0]}"
        )

    tree_accuracy = responses.mean(axis=1)
    discrimination = params.get("slope_loc")
    mean_disc_correct, mean_disc_error = per_tree_discrimination_stats(responses, discrimination)

    tree_df = tree_df.copy()
    tree_df["theta"] = ability
    tree_df["tree_accuracy"] = tree_accuracy
    tree_df["mean_disc_correct"] = mean_disc_correct
    tree_df["mean_disc_error"] = mean_disc_error

    if args.table_output is not None:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        tree_df.to_csv(args.table_output, index=False)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for attr in ATTR_COLUMNS:
        if attr not in tree_df:
            continue
        summary[attr] = {}
        for target, label in TARGET_LABELS.items():
            if target not in tree_df:
                continue
            stats = correlation(tree_df[attr], tree_df[target])
            summary[attr][target] = stats
            figure_name = f"{args.prefix}_{attr}_vs_{target}.png"
            scatter_plot(
                tree_df[attr].to_numpy(dtype=np.float32),
                tree_df[target].to_numpy(dtype=np.float32),
                xlabel=attr.replace("_", " ").title(),
                ylabel=label,
                title=f"{label} vs {attr.replace('_', ' ').title()}",
                output=args.figures_dir / figure_name,
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
