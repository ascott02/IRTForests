#!/usr/bin/env python3
"""Compare Random Forest per-item signals with IRT difficulty estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAM_KEY_MAP = {
    "difficulty": "diff_loc",
    "discrimination": "slope_loc",
}

PARAM_LABELS = {
    "difficulty": "Item Difficulty (δ)",
    "discrimination": "Item Discrimination (a)",
}

SIGNAL_LABELS = {
    "margin": "RF Margin",
    "entropy": "RF Entropy",
}


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
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=sorted(PARAM_KEY_MAP.keys()),
        default=["difficulty"],
        help="IRT parameters to correlate against RF signals.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="rf_irt_correlations.json",
        help="Filename for the correlations JSON summary.",
    )
    return parser.parse_args()


def load_arrays(args: argparse.Namespace) -> Dict[str, Dict[str, np.ndarray]]:
    margin = np.load(args.margin)
    entropy = np.load(args.entropy)
    with np.load(args.irt_params) as data:
        param_arrays = {}
        for param_name, key in PARAM_KEY_MAP.items():
            if key in data:
                param_arrays[param_name] = data[key]
    if "difficulty" not in param_arrays:
        raise KeyError(
            f"IRT parameter archive {args.irt_params} does not contain 'diff_loc'."
        )

    if not (margin.shape == entropy.shape == next(iter(param_arrays.values())).shape):
        raise ValueError(
            "Margin, entropy, and difficulty arrays must share the same length."
        )
    return {"signals": {"margin": margin, "entropy": entropy}, "parameters": param_arrays}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return pearson(xr, yr)


def summarise_correlations(
    parameter_names: List[str],
    item_params: Dict[str, np.ndarray],
    signals: Dict[str, np.ndarray],
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for param in parameter_names:
        values = item_params[param]
        param_stats: Dict[str, Dict[str, float]] = {}
        for signal_name, signal_values in signals.items():
            stats = {
                "pearson": pearson(values, signal_values),
                "spearman": spearman(values, signal_values),
            }
            if param == "difficulty":
                summary[signal_name] = stats
            param_stats[signal_name] = stats
        summary[param] = param_stats
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
    signals = arrays["signals"]
    item_params = arrays["parameters"]

    missing = [p for p in args.parameters if p not in item_params]
    if missing:
        raise KeyError(
            "Requested parameters not found in IRT archive: " + ", ".join(missing)
        )

    summary = summarise_correlations(args.parameters, item_params, signals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for param in args.parameters:
        xlabel = PARAM_LABELS[param]
        parameter_values = item_params[param]
        for signal_name, signal_values in signals.items():
            ylabel = SIGNAL_LABELS[signal_name]
            title = f"{xlabel} vs {ylabel}"
            if args.prefix:
                figure_name = f"{args.prefix}_{param}_vs_{signal_name}.png"
            else:
                figure_name = f"{param}_vs_{signal_name}.png"
            scatter_plot(
                parameter_values,
                signal_values,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                output=args.figures_dir / figure_name,
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
