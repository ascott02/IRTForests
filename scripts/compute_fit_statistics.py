#!/usr/bin/env python3
"""Compute item and person fit statistics from saved IRT estimates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class FitInputs:
    response_matrix: np.ndarray
    ability: np.ndarray
    difficulty: np.ndarray
    slope: Optional[np.ndarray]
    guess: Optional[np.ndarray]

    @property
    def num_persons(self) -> int:
        return int(self.response_matrix.shape[0])

    @property
    def num_items(self) -> int:
        return int(self.response_matrix.shape[1])


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expected_probabilities(inputs: FitInputs) -> np.ndarray:
    theta = inputs.ability.reshape(-1, 1)
    diff = inputs.difficulty.reshape(1, -1)
    logits = theta - diff
    if inputs.slope is not None:
        logits = logits * inputs.slope.reshape(1, -1)
    probs = logistic(logits)
    if inputs.guess is not None:
        guess = inputs.guess.reshape(1, -1)
        probs = guess + (1.0 - guess) * probs
    return np.clip(probs, 1e-6, 1.0 - 1e-6)


def mean_square(residuals: np.ndarray, weights: Optional[np.ndarray], axis: int) -> np.ndarray:
    if weights is None:
        return np.mean(residuals**2, axis=axis)
    numerator = np.sum(weights * residuals**2, axis=axis)
    denominator = np.sum(weights, axis=axis)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return result


def zstd_transform(mnsq: np.ndarray, df: np.ndarray) -> np.ndarray:
    """Approximate Z-standardized fit statistics using a common cubic-root transform."""
    safe_df = np.maximum(df, 1)
    return (np.power(mnsq, 1.0 / 3.0) - 1.0) * (3.0 / np.sqrt(safe_df))


def compute_fit_statistics(inputs: FitInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    responses = inputs.response_matrix
    expected = expected_probabilities(inputs)
    residuals = responses - expected
    variances = expected * (1.0 - expected)

    # Person-level statistics (trees)
    person_infit = mean_square(residuals, variances, axis=1)
    person_outfit = mean_square(residuals, None, axis=1)
    person_df = pd.DataFrame(
        {
            "tree_index": np.arange(inputs.num_persons),
            "ability": inputs.ability,
            "total_score": responses.sum(axis=1),
            "expected_score": expected.sum(axis=1),
            "infit_msq": person_infit,
            "outfit_msq": person_outfit,
            "infit_zstd": zstd_transform(person_infit, np.full(inputs.num_persons, inputs.num_items)),
            "outfit_zstd": zstd_transform(person_outfit, np.full(inputs.num_persons, inputs.num_items)),
        }
    )

    # Item-level statistics (images)
    item_infit = mean_square(residuals, variances, axis=0)
    item_outfit = mean_square(residuals, None, axis=0)
    item_df = {
        "item_index": np.arange(inputs.num_items),
        "difficulty": inputs.difficulty,
        "total_score": responses.sum(axis=0),
        "expected_score": expected.sum(axis=0),
        "infit_msq": item_infit,
        "outfit_msq": item_outfit,
        "infit_zstd": zstd_transform(item_infit, np.full(inputs.num_items, inputs.num_persons)),
        "outfit_zstd": zstd_transform(item_outfit, np.full(inputs.num_items, inputs.num_persons)),
    }
    if inputs.slope is not None:
        item_df["discrimination"] = inputs.slope
    if inputs.guess is not None:
        item_df["guess"] = inputs.guess
    item_df = pd.DataFrame(item_df)

    return person_df, item_df


def load_fit_inputs(root: Path, suffix: str) -> FitInputs:
    response_path = root / "response_matrix.npz"
    if not response_path.exists():
        raise FileNotFoundError(f"Missing response matrix: {response_path}")
    with np.load(response_path) as data:
        matrix = data["R"].astype(np.float32)

    params_path = root / f"irt_parameters{suffix}.npz"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {params_path}")
    params = np.load(params_path)

    ability = params["ability_loc"].astype(np.float32)
    difficulty = params["diff_loc"].astype(np.float32)
    slope = params["slope_loc"].astype(np.float32) if "slope_loc" in params else None
    guess = params["guess_mean"].astype(np.float32) if "guess_mean" in params else None

    if matrix.shape[0] != ability.shape[0]:
        raise ValueError(
            f"Ability vector length ({ability.shape[0]}) does not match number of respondents ({matrix.shape[0]})."
        )
    if matrix.shape[1] != difficulty.shape[0]:
        raise ValueError(
            f"Difficulty vector length ({difficulty.shape[0]}) does not match number of items ({matrix.shape[1]})."
        )

    if slope is not None and slope.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Discrimination vector length ({slope.shape[0]}) does not match number of items ({matrix.shape[1]})."
        )
    if guess is not None and guess.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Guess parameter length ({guess.shape[0]}) does not match number of items ({matrix.shape[1]})."
        )

    return FitInputs(matrix, ability, difficulty, slope, guess)


def discover_models(root: Path) -> Iterable[str]:
    for suffix in ("", "_2pl", "_3pl"):
        if (root / f"irt_parameters{suffix}.npz").exists():
            yield suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path("data"), Path("data/mnist"), Path("data/mobilenet")],
        help="Directories that contain response_matrix.npz and irt_parameters*.npz",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fit statistic CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root in args.roots:
        if not root.exists():
            continue
        for suffix in discover_models(root):
            try:
                inputs = load_fit_inputs(root, suffix)
            except (FileNotFoundError, ValueError) as exc:
                print(f"Skipping {root} (suffix '{suffix}'): {exc}")
                continue

            person_df, item_df = compute_fit_statistics(inputs)

            person_path = root / f"person_fit_stats{suffix}.csv"
            item_path = root / f"item_fit_stats{suffix}.csv"
            if not args.overwrite and (person_path.exists() or item_path.exists()):
                print(f"Skipping {root} (suffix '{suffix}') — outputs already exist. Use --overwrite to recompute.")
                continue

            person_df.to_csv(person_path, index=False)
            item_df.to_csv(item_path, index=False)
            print(
                f"Saved fit statistics for {root} (suffix '{suffix}') -> "
                f"{person_path.name}, {item_path.name}"
            )


if __name__ == "__main__":
    main()
