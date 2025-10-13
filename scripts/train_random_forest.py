#!/usr/bin/env python3
"""Train a RandomForestClassifier on cached embeddings and persist artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/cifar10_embeddings.npz"),
        help="Path to embeddings archive containing train/val/test splits and labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where RF artifacts (metrics, predictions, response matrix) will be saved.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth of each tree (None for unrestricted).",
    )
    parser.add_argument(
        "--max-features",
        type=str,
        default="sqrt",
        help="Number of features to consider when looking for the best split.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples required to be at a leaf node.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run for training and inference.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Number of repeats for permutation importance on the validation split.",
    )
    return parser.parse_args()


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    with np.load(path) as data:
        required = {
            "train_embeddings",
            "val_embeddings",
            "test_embeddings",
            "y_train",
            "y_val",
            "y_test",
            "classes",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Embeddings archive missing keys: {sorted(missing)}")
        arrays = {name: data[name] for name in required}
    return arrays


def train_random_forest(
    features: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=args.max_features,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        oob_score=True,
    )
    rf.fit(features["train_embeddings"], features["y_train"])
    return rf


def compute_metrics(
    rf: RandomForestClassifier,
    features: Dict[str, np.ndarray],
) -> Dict[str, float | list | int]:
    X_train, X_val, X_test = (
        features["train_embeddings"],
        features["val_embeddings"],
        features["test_embeddings"],
    )
    y_train, y_val, y_test = (
        features["y_train"],
        features["y_val"],
        features["y_test"],
    )

    y_pred_test = rf.predict(X_test)
    y_pred_val = rf.predict(X_val)
    conf = confusion_matrix(y_test, y_pred_test, labels=np.arange(len(features["classes"])))
    per_class_accuracy = np.divide(
        conf.diagonal(),
        conf.sum(axis=1),
        out=np.zeros_like(conf.diagonal(), dtype=float),
        where=conf.sum(axis=1) != 0,
    )

    metrics: Dict[str, float | list | int] = {
        "overall_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "val_accuracy": float(accuracy_score(y_val, y_pred_val)),
        "oob_accuracy": float(rf.oob_score_) if hasattr(rf, "oob_score_") else float("nan"),
        "per_class_accuracy": per_class_accuracy.tolist(),
        "classes": features["classes"].tolist(),
        "n_features": int(X_train.shape[1]),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    return metrics


def build_response_matrix(
    rf: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    predictions = []
    for tree in rf.estimators_:
        pred = tree.predict(X_test)
        predictions.append(pred == y_test)
    matrix = np.vstack(predictions).astype(np.uint8)
    return matrix


def save_artifacts(
    rf: RandomForestClassifier,
    features: Dict[str, np.ndarray],
    metrics: Dict[str, float | list | int],
    args: argparse.Namespace,
) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test = features["test_embeddings"]
    y_test = features["y_test"]

    y_pred = rf.predict(X_test)
    probas = rf.predict_proba(X_test)
    confusion = confusion_matrix(y_test, y_pred, labels=np.arange(len(features["classes"])))

    np.save(output_dir / "rf_test_pred.npy", y_pred)
    np.save(output_dir / "rf_test_proba.npy", probas.astype(np.float32))
    np.save(output_dir / "rf_confusion.npy", confusion)
    np.save(output_dir / "rf_feature_importances.npy", rf.feature_importances_.astype(np.float32))

    # Response matrix for IRT fitting.
    R = build_response_matrix(rf, X_test, y_test)
    np.savez_compressed(
        output_dir / "response_matrix.npz",
        R=R,
        y_test=y_test,
        classes=features["classes"],
    )

    with (output_dir / "rf_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # Permutation importance on validation data for interpretability.
    perm = permutation_importance(
        rf,
        features["val_embeddings"],
        features["y_val"],
        n_repeats=args.permutation_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    importances_mean = perm["importances_mean"]
    importances_std = perm["importances_std"]
    permutation_df = pd.DataFrame(
        {
            "feature": np.arange(importances_mean.size),
            "importances_mean": importances_mean,
            "importances_std": importances_std,
        }
    )
    permutation_df.to_csv(output_dir / "rf_permutation_importance.csv", index=False)


def main() -> None:
    args = parse_args()
    features = load_embeddings(args.embeddings)
    rf = train_random_forest(features, args)
    metrics = compute_metrics(rf, features)
    save_artifacts(rf, features, metrics, args)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
