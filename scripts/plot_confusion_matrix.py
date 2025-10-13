#!/usr/bin/env python3
"""Plot the Random Forest confusion matrix for CIFAR-10 test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confusion",
        type=Path,
        default=Path("data/rf_confusion.npy"),
        help="Path to confusion matrix saved as numpy array.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/rf_metrics.json"),
        help="JSON containing class labels.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize rows to sum to 1 (true class proportions).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/rf_confusion_matrix.png"),
        help="Output figure path.",
    )
    return parser.parse_args()


def load_confusion(confusion_path: Path, metrics_path: Path) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(confusion_path)
    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    classes = metrics["classes"]
    return matrix, classes


def plot_matrix(matrix: np.ndarray, classes: list[str], normalize: bool, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = matrix.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        data = data / row_sums

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
    )
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    title = "Random Forest Confusion Matrix"
    if normalize:
        title += " (normalized)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    matrix, classes = load_confusion(args.confusion, args.metrics)
    plot_matrix(matrix, classes, args.normalize, args.output)
    print(f"Confusion matrix figure written to {args.output}")


if __name__ == "__main__":
    main()
