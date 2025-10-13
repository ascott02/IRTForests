#!/usr/bin/env python3
"""Train a RandomForest on a classic scikit-learn tabular dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

DATASETS: Dict[str, Callable] = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS.keys(), default="breast_cancer")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("reports/rf_tabular_summary.json"))
    return parser.parse_args()


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    loader = DATASETS[name]
    data = loader()
    feature_names = getattr(data, "feature_names", None)
    if feature_names is not None:
        feature_names = list(feature_names)
    return data.data, data.target, {"feature_names": feature_names}


def train_rf(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
    n_estimators: int,
) -> Tuple[RandomForestClassifier, Dict[str, object], Dict[str, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    metrics: Dict[str, object] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "oob_accuracy": clf.oob_score_,
        "classification_report": report,
        "confusion_matrix": conf_mat.tolist(),
    }

    if len(np.unique(y)) == 2:
        y_prob = clf.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    feature_importances = clf.feature_importances_.tolist()
    return clf, metrics, {"feature_importances": feature_importances, "confusion_matrix": conf_mat}


def main() -> None:
    args = parse_args()
    X, y, meta = load_dataset(args.dataset)
    clf, metrics, arrays = train_rf(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )

    feature_names = meta.get("feature_names")
    if isinstance(feature_names, (list, tuple, np.ndarray)):
        feature_names = list(feature_names)
    else:
        feature_names = [f"feature_{i}" for i in range(len(arrays["feature_importances"]))]

    summary = {
        "dataset": args.dataset,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "metrics": metrics,
        "top_features": sorted(
            zip(feature_names, arrays["feature_importances"]),
            key=lambda item: item[1],
            reverse=True,
        )[:10],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
