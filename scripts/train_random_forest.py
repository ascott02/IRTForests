#!/usr/bin/env python3
"""Train a RandomForestClassifier on cached embeddings and persist artifacts."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib


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
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path to persist the trained RandomForestClassifier (joblib).",
    )
    parser.add_argument(
        "--tree-attributes",
        type=Path,
        default=None,
        help="Optional path to export per-tree attributes (depth, leaves, OOB accuracy).",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("loky", "threading"),
        default="loky",
        help="Joblib backend for parallelism; 'threading' avoids extra data copies.",
    )
    parser.add_argument(
        "--no-oob",
        action="store_true",
        help="Disable OOB scoring to save memory (metrics will report NaN).",
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
        arrays: Dict[str, np.ndarray] = {}
        for name in required:
            arr = data[name]
            if "embeddings" in name:
                arrays[name] = np.asarray(arr, dtype=np.float32, order="C")
            elif name.startswith("y_"):
                arrays[name] = np.asarray(arr, dtype=np.int32, order="C")
            else:
                arrays[name] = arr
    return arrays


def _joblib_backend(args: argparse.Namespace):
    if args.n_jobs == 1 or args.parallel_backend == "loky":
        return nullcontext()
    return joblib.parallel_backend(args.parallel_backend, n_jobs=args.n_jobs)


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
        oob_score=not args.no_oob,
    )
    with _joblib_backend(args):
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


def compute_tree_attributes(
    rf: RandomForestClassifier,
    features: Dict[str, np.ndarray],
) -> List[Dict[str, float | int]]:
    X_train = features["train_embeddings"]
    y_train = features["y_train"]
    attributes: List[Dict[str, float | int]] = []

    has_samples = hasattr(rf, "estimators_samples_")
    for idx, tree in enumerate(rf.estimators_):
        depth = int(tree.tree_.max_depth)  # type: ignore[attr-defined]
        n_leaves = int(tree.tree_.n_leaves)  # type: ignore[attr-defined]

        oob_accuracy = float("nan")
        oob_count = 0
        if has_samples:
            samples = rf.estimators_samples_[idx]  # type: ignore[attr-defined]
            if samples is not None:
                mask = np.ones(X_train.shape[0], dtype=bool)
                mask[samples] = False
                oob_indices = np.flatnonzero(mask)
                oob_count = int(oob_indices.size)
                if oob_indices.size > 0:
                    preds = tree.predict(X_train[oob_indices])
                    oob_accuracy = float(np.mean(preds == y_train[oob_indices]))

        attributes.append(
            {
                "tree_index": idx,
                "depth": depth,
                "n_leaves": n_leaves,
                "oob_accuracy": oob_accuracy,
                "oob_count": oob_count,
            }
        )

    return attributes


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
    with _joblib_backend(args):
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

    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, args.save_model)

    if args.tree_attributes is not None:
        tree_attrs = compute_tree_attributes(rf, features)
        args.tree_attributes.parent.mkdir(parents=True, exist_ok=True)
        with args.tree_attributes.open("w", encoding="utf-8") as fh:
            json.dump(tree_attrs, fh, indent=2)


def main() -> None:
    args = parse_args()
    features = load_embeddings(args.embeddings)
    rf = train_random_forest(features, args)
    metrics = compute_metrics(rf, features)
    save_artifacts(rf, features, metrics, args)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
