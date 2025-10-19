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
    # Anchor both distributions so the mean tree ability lands at zero.
    ability_mean = ability.mean()
    ability_centered = ability - ability_mean
    difficulty_centered = difficulty - ability_mean
    ability_centered_mean = ability_centered.mean()
    difficulty_centered_mean = difficulty_centered.mean()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2, ncols=1, figsize=(7.5, 5.5), sharex=True, height_ratios=[1, 1.2]
    )

    ax_top.hist(
        ability_centered,
        bins=bins,
        color="#377eb8",
        alpha=0.8,
        edgecolor="white",
    )
    ax_top.set_ylabel("Tree Count")
    ax_top.set_title("Wright Map: Tree Ability (θ) vs Item Difficulty (δ)")
    ax_top.axvline(0.0, color="#1f78b4", linestyle="--", linewidth=1.2)
    ax_top.text(
        0.0,
        ax_top.get_ylim()[1] * 0.9,
        f"mean θ (anchored) = {ability_centered_mean:.2f}",
        color="#1f78b4",
        ha="left",
    )

    ax_bottom.hist(
        difficulty_centered,
        bins=bins,
        color="#e41a1c",
        alpha=0.8,
        edgecolor="white",
    )
    ax_bottom.set_ylabel("Item Count")
    ax_bottom.set_xlabel("Latent Scale (θ anchored at 0)")
    ax_bottom.axvline(
        difficulty_centered_mean, color="#b22222", linestyle="--", linewidth=1.2
    )
    ax_bottom.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
    ax_bottom.text(
        difficulty_centered_mean,
        ax_bottom.get_ylim()[1] * 0.9,
        f"mean δ (anchored) = {difficulty_centered_mean:.2f}",
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
