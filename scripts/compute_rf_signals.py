#!/usr/bin/env python3
"""Compute Random Forest per-example signals (margin, entropy) for downstream analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probabilities",
        type=Path,
        default=Path("data/rf_test_proba.npy"),
        help="Path to numpy array with shape (n_items, n_classes) of RF test probabilities.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/response_matrix.npz"),
        help=(
            "Source file containing true labels. If .npz, it should include a 'y_test' array; "
            "otherwise this path should point to a .npy with label integers."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where computed signals and summaries will be saved.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Decimal precision for printed summary statistics.",
    )
    return parser.parse_args()


def load_labels(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Label source not found: {path}")
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "y_test" not in data:
                keys = ", ".join(data.files)
                raise KeyError(f"Expected 'y_test' in {path}, found {keys}")
            labels = data["y_test"]
    else:
        labels = np.load(path)
    return labels.astype(np.int64)


def compute_margin_and_entropy(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if probabilities.ndim != 2:
        raise ValueError("Probabilities array must be 2D (n_items, n_classes)")
    n_items, n_classes = probabilities.shape
    if labels.shape[0] != n_items:
        raise ValueError(
            f"Label count ({labels.shape[0]}) does not match probabilities ({n_items})"
        )

    # Clip probabilities for numerical stability before entropy.
    probs = np.clip(probabilities, 1e-12, 1.0)
    true_prob = probs[np.arange(n_items), labels]

    # For max other class probability, temporarily zero-out the true class.
    mask = np.ones_like(probs, dtype=bool)
    mask[np.arange(n_items), labels] = False
    max_other = np.max(np.where(mask, probs, -np.inf), axis=1)

    margin = true_prob - max_other
    entropy = -np.sum(probs * np.log(probs), axis=1)
    return margin.astype(np.float32), entropy.astype(np.float32)


def summarise_signals(
    margin: np.ndarray, entropy: np.ndarray, precision: int
) -> Dict[str, Dict[str, float]]:
    def stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p05": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary: Dict[str, Dict[str, float]] = {
        "margin": stats(margin),
        "entropy": stats(entropy),
    }
    return summary


def save_outputs(
    output_dir: Path,
    margin: np.ndarray,
    entropy: np.ndarray,
    summary: Dict[str, Dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rf_margins.npy", margin)
    np.save(output_dir / "rf_entropy.npy", entropy)
    with (output_dir / "rf_signal_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    probs = np.load(args.probabilities)
    labels = load_labels(args.labels)
    margin, entropy = compute_margin_and_entropy(probs, labels)
    summary = summarise_signals(margin, entropy, args.precision)
    save_outputs(args.output_dir, margin, entropy, summary)

    np.set_printoptions(precision=args.precision)
    print("Margin stats:", summary["margin"])
    print("Entropy stats:", summary["entropy"])


if __name__ == "__main__":
    main()
