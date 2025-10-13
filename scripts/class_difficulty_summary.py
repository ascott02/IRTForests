#!/usr/bin/env python3
"""Summarize IRT item difficulty by CIFAR-10 class and compare with RF error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-matrix",
        type=Path,
        default=Path("data/response_matrix.npz"),
        help="NPZ file containing 'y_test' and 'classes'.",
    )
    parser.add_argument(
        "--irt-params",
        type=Path,
        default=Path("data/irt_parameters.npz"),
        help="NPZ file providing diff_loc array.",
    )
    parser.add_argument(
        "--rf-metrics",
        type=Path,
        default=Path("data/rf_metrics.json"),
        help="JSON with per-class accuracy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/class_difficulty_summary.json"),
        help="Output JSON file for class-level statistics.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/class_difficulty_vs_error.png"),
        help="Output figure comparing difficulty and RF error per class.",
    )
    return parser.parse_args()


def load_inputs(
    response_path: Path, irt_path: Path, rf_metrics_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    with np.load(response_path) as data:
        y_test = data["y_test"]
        classes = data["classes"]
    with np.load(irt_path) as data:
        difficulty = data["diff_loc"]
    with rf_metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    per_class_accuracy = metrics["per_class_accuracy"]
    return y_test, classes, difficulty, per_class_accuracy


def summarise_by_class(
    y_test: np.ndarray,
    classes: np.ndarray,
    difficulty: np.ndarray,
    per_class_accuracy: list,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "label": y_test,
            "difficulty": difficulty,
        }
    )
    grouped = df.groupby("label").agg(
        difficulty_mean=("difficulty", "mean"),
        difficulty_std=("difficulty", "std"),
        difficulty_median=("difficulty", "median"),
    )
    grouped["rf_accuracy"] = per_class_accuracy
    grouped["rf_error"] = 1.0 - grouped["rf_accuracy"]
    grouped["class_name"] = classes[grouped.index]
    grouped = grouped.reset_index(drop=True)
    grouped = grouped[
        ["class_name", "difficulty_mean", "difficulty_std", "difficulty_median", "rf_accuracy", "rf_error"]
    ]
    grouped = grouped.sort_values("difficulty_mean", ascending=False).reset_index(drop=True)
    return grouped


def save_summary(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)


def plot_summary(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    idx = np.arange(len(df))
    width = 0.35

    ax1.bar(idx - width / 2, df["difficulty_mean"], width, label="IRT difficulty (mean)", color="#e41a1c")
    ax1.set_ylabel("Difficulty (δ)")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(df["class_name"], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(idx + width / 2, df["rf_error"], width, label="RF error", color="#377eb8")
    ax2.set_ylabel("RF error rate")

    ax1.set_title("Class-wise IRT Difficulty vs RF Error")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    y_test, classes, difficulty, per_class_accuracy = load_inputs(
        args.response_matrix, args.irt_params, args.rf_metrics
    )
    df = summarise_by_class(y_test, classes, difficulty, per_class_accuracy)
    save_summary(df, args.output)
    plot_summary(df, args.figure)
    print(df)


if __name__ == "__main__":
    main()
