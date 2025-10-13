#!/usr/bin/env python3
"""Generate a Wright map-style visualization for tree abilities and item difficulties."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--irt-params",
        type=Path,
        default=Path("data/irt_parameters.npz"),
        help="Path to IRT parameter archive containing ability_loc and diff_loc arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/wright_map.png"),
        help="Output path for the generated figure.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for the histograms.",
    )
    return parser.parse_args()


def load_params(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        ability = data["ability_loc"]
        difficulty = data["diff_loc"]
    return ability, difficulty


def plot_wright_map(
    ability: np.ndarray,
    difficulty: np.ndarray,
    bins: int,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1, figsize=(7.5, 5.5), sharex=True, height_ratios=[1, 1.2]
    )

    ax_top.hist(
        ability,
        bins=bins,
        color="#377eb8",
        alpha=0.8,
        edgecolor="white",
    )
    ax_top.set_ylabel("Tree Count")
    ax_top.set_title("Wright Map: Tree Ability (θ) vs Item Difficulty (δ)")
    ax_top.axvline(ability.mean(), color="#1f78b4", linestyle="--", linewidth=1.2)
    ax_top.text(
        ability.mean(),
        ax_top.get_ylim()[1] * 0.9,
        f"mean θ = {ability.mean():.2f}",
        color="#1f78b4",
        ha="left",
    )

    ax_bottom.hist(
        difficulty,
        bins=bins,
        color="#e41a1c",
        alpha=0.8,
        edgecolor="white",
    )
    ax_bottom.set_ylabel("Item Count")
    ax_bottom.set_xlabel("Latent Scale")
    ax_bottom.axvline(difficulty.mean(), color="#b22222", linestyle="--", linewidth=1.2)
    ax_bottom.text(
        difficulty.mean(),
        ax_bottom.get_ylim()[1] * 0.9,
        f"mean δ = {difficulty.mean():.2f}",
        color="#b22222",
        ha="left",
    )
    ax_bottom.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ability, difficulty = load_params(args.irt_params)
    plot_wright_map(ability, difficulty, args.bins, args.output)


if __name__ == "__main__":
    main()
