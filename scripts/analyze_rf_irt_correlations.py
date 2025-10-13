#!/usr/bin/env python3
"""Compare Random Forest per-item signals with IRT difficulty estimates."""

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
        "--margin",
        type=Path,
        default=Path("data/rf_margins.npy"),
        help="Path to RF margin array (n_items,).",
    )
    parser.add_argument(
        "--entropy",
        type=Path,
        default=Path("data/rf_entropy.npy"),
        help="Path to RF entropy array (n_items,).",
    )
    parser.add_argument(
        "--irt-params",
        type=Path,
        default=Path("data/irt_parameters.npz"),
        help="Path to IRT parameter archive containing diff_loc (difficulty).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save correlation statistics.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save scatter plots.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="difficulty",
        help="Prefix to use when naming output figures.",
    )
    return parser.parse_args()


def load_arrays(args: argparse.Namespace) -> Dict[str, np.ndarray]:
    margin = np.load(args.margin)
    entropy = np.load(args.entropy)
    with np.load(args.irt_params) as data:
        difficulty = data["diff_loc"]
    if not (margin.shape == entropy.shape == difficulty.shape):
        raise ValueError(
            "Margin, entropy, and difficulty arrays must share the same length."
        )
    return {"margin": margin, "entropy": entropy, "difficulty": difficulty}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson(xr, yr)


def summarise_correlations(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    difficulty = data["difficulty"]
    summary = {}
    for key in ("margin", "entropy"):
        summary[key] = {
            "pearson": pearson(difficulty, data[key]),
            "spearman": spearman(difficulty, data[key]),
        }
    return summary


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=12, alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    arrays = load_arrays(args)
    summary = summarise_correlations(arrays)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rf_irt_correlations.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    scatter_plot(
        arrays["difficulty"],
        arrays["margin"],
        xlabel="Item Difficulty (δ)",
        ylabel="RF Margin",
        title="Item Difficulty vs. RF Margin",
        output=args.figures_dir / f"{args.prefix}_vs_margin.png",
    )
    scatter_plot(
        arrays["difficulty"],
        arrays["entropy"],
        xlabel="Item Difficulty (δ)",
        ylabel="RF Entropy",
        title="Item Difficulty vs. RF Entropy",
        output=args.figures_dir / f"{args.prefix}_vs_entropy.png",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
