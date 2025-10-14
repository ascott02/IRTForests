#!/usr/bin/env python3
"""Generate ability-vs-accuracy and difficulty-vs-error plots for a run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_run_arrays(
    data_dir: Path,
    suffix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load ability, difficulty, tree accuracy, item error, and optional discrimination arrays."""
    params = np.load(data_dir / f"irt_parameters{suffix}.npz")
    ability = params["ability_loc"]
    difficulty = params["diff_loc"]
    slope = params["slope_loc"] if "slope_loc" in params.files else None

    resp = np.load(data_dir / "response_matrix.npz")
    R = resp["R"].astype(np.float32)

    tree_accuracy = R.mean(axis=1)
    item_error = 1.0 - R.mean(axis=0)
    return ability, difficulty, tree_accuracy, item_error, slope


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    r = np.corrcoef(x, y)[0, 1]
    coeffs = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=24, alpha=0.65, edgecolor="none")
    ax.plot(x_line, y_line, color="#555555", linewidth=1.5, linestyle="--", label="Linear fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.text(
        0.02,
        0.96,
        f"Pearson r = {r:.3f}\n{subtitle}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#dddddd"),
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def hist_plot(values: np.ndarray, xlabel: str, title: str, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=30, alpha=0.75, color="#4c72b0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to run-specific data directory")
    parser.add_argument(
        "--figures-dir", type=Path, required=True, help="Directory where figures should be written"
    )
    parser.add_argument("--label", type=str, required=True, help="Label used in figure subtitles")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix appended to IRT parameter filenames (e.g., _2pl).",
    )
    args = parser.parse_args()

    ability, difficulty, tree_accuracy, item_error, slope = load_run_arrays(args.data_dir, args.suffix)

    ability_path = args.figures_dir / "ability_vs_accuracy.png"
    scatter_plot(
        ability,
        tree_accuracy,
        xlabel="Ability (θ)",
        ylabel="Per-tree accuracy",
        title="Ability vs Tree Accuracy",
        subtitle=args.label,
        output_path=ability_path,
    )

    difficulty_path = args.figures_dir / "difficulty_vs_error.png"
    scatter_plot(
        difficulty,
        item_error,
        xlabel="Difficulty (δ)",
        ylabel="Mean tree error",
        title="Difficulty vs Tree Error",
        subtitle=args.label,
        output_path=difficulty_path,
    )

    if slope is not None:
        slope_path = args.figures_dir / "discrimination_hist.png"
        hist_plot(
            slope,
            xlabel="Discrimination (a)",
            title="Item Discrimination Distribution",
            output_path=slope_path,
        )


if __name__ == "__main__":
    main()
