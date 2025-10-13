"""Utilities to prepare CIFAR-10 subsets and feature embeddings.

This module focuses on the reproducible data preparation steps used by the
Random Forest × Item Response Theory study.  It handles downloading CIFAR-10,
performing stratified subsampling, and persisting cached tensors for the rest
of the pipeline.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True)
class SubsetConfig:
    """Configuration for the CIFAR-10 subset."""

    train_per_class: int = 1000
    val_per_class: int = 200
    test_per_class: int = 200
    resize: int = 64
    seed: int = 42
    data_root: pathlib.Path = pathlib.Path("data")

    @property
    def train_size(self) -> int:
        return 10 * self.train_per_class

    @property
    def val_size(self) -> int:
        return 10 * self.val_per_class

    @property
    def test_size(self) -> int:
        return 10 * self.test_per_class


def _build_transform(resize: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _stratified_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = []
    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < per_class:
            raise ValueError(f"Requested {per_class} samples for class {cls}, only {len(cls_idx)} available")
        indices.append(rng.choice(cls_idx, size=per_class, replace=False))
    return np.concatenate(indices)


def _collect_subset(dataset: datasets.CIFAR10, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tensors = []
    labels = []
    for idx in indices:
        tensor, label = dataset[idx]
        tensors.append(tensor.numpy())
        labels.append(label)
    return np.stack(tensors), np.array(labels, dtype=np.int64)


def build_cifar10_subset(config: SubsetConfig) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Download CIFAR-10, create stratified train/val/test subsets, and return arrays."""

    transform = _build_transform(config.resize)
    root = config.data_root / "torchvision"
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    train_labels = np.array(train_dataset.targets)
    train_idx = _stratified_indices(train_labels, config.train_per_class + config.val_per_class, config.seed)

    # shuffle once and split into train / val
    rng = np.random.default_rng(config.seed)
    rng.shuffle(train_idx)
    train_split = train_idx[: config.train_size]
    val_split = train_idx[config.train_size : config.train_size + config.val_size]

    test_labels = np.array(test_dataset.targets)
    test_idx = _stratified_indices(test_labels, config.test_per_class, config.seed)

    x_train, y_train = _collect_subset(train_dataset, train_split)
    x_val, y_val = _collect_subset(train_dataset, val_split)
    x_test, y_test = _collect_subset(test_dataset, test_idx)

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
        "classes": np.array(train_dataset.classes),
    }


def save_cifar10_subset(config: SubsetConfig, out_path: pathlib.Path | None = None) -> pathlib.Path:
    """Persist the CIFAR-10 subset to disk as an .npz archive."""

    data = build_cifar10_subset(config)
    out_dir = config.data_root
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = out_dir / "cifar10_subset.npz"

    np.savez(
        out_path,
        x_train=data["train"][0],
        y_train=data["train"][1],
        x_val=data["val"][0],
        y_val=data["val"][1],
        x_test=data["test"][0],
        y_test=data["test"][1],
        classes=data["classes"],
        mean=np.array(CIFAR10_MEAN),
        std=np.array(CIFAR10_STD),
    )
    return out_path


def compute_pca_embeddings(
    subset_path: pathlib.Path,
    n_components: int = 128,
    random_state: int = 42,
    out_path: pathlib.Path | None = None,
) -> Tuple[pathlib.Path, Dict[str, Tuple[int, ...]]]:
    """Compute PCA embeddings from a saved CIFAR-10 subset archive.

    Returns
    -------
    embeddings_path:
        Location of the saved `.npz` archive containing PCA features.
    summary:
        Dictionary of shape metadata useful for logging.
    """

    subset = np.load(subset_path)
    x_train = subset["x_train"].reshape(subset["x_train"].shape[0], -1)
    x_val = subset["x_val"].reshape(subset["x_val"].shape[0], -1)
    x_test = subset["x_test"].reshape(subset["x_test"].shape[0], -1)

    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    train_embeddings = pca.fit_transform(x_train)
    val_embeddings = pca.transform(x_val)
    test_embeddings = pca.transform(x_test)

    if out_path is None:
        out_path = subset_path.parent / "cifar10_embeddings.npz"

    np.savez(
        out_path,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        y_train=subset["y_train"],
        y_val=subset["y_val"],
        y_test=subset["y_test"],
        classes=subset["classes"],
        mean=subset["mean"],
        std=subset["std"],
        n_components=np.array([n_components]),
        pca_components=pca.components_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )
    summary = {
        "train_embeddings": train_embeddings.shape,
        "val_embeddings": val_embeddings.shape,
        "test_embeddings": test_embeddings.shape,
        "explained_variance_ratio": pca.explained_variance_ratio_.shape,
    }
    return out_path, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 subset for RF × IRT study")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=200)
    parser.add_argument("--test-per-class", type=int, default=200)
    parser.add_argument("--resize", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--embeddings", action="store_true", help="Also compute PCA embeddings for cached subset")
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--no-subset", action="store_true", help="Skip subset generation and reuse existing archive")

    args = parser.parse_args()
    config = SubsetConfig(
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        resize=args.resize,
        seed=args.seed,
        data_root=args.data_root,
    )
    subset_path: pathlib.Path
    if args.no_subset:
        if args.out is None:
            subset_path = config.data_root / "cifar10_subset.npz"
        else:
            subset_path = args.out
        if not subset_path.exists():
            raise FileNotFoundError(f"Requested to skip subset generation but {subset_path} not found")
    else:
        subset_path = save_cifar10_subset(config, args.out)
        print(f"Saved CIFAR-10 subset to {subset_path.resolve()}")

    if args.embeddings:
        embeddings_path, summary = compute_pca_embeddings(subset_path, args.n_components)
        print(f"Saved PCA embeddings to {embeddings_path.resolve()}")
        print(
            "Shapes → train:{train} val:{val} test:{test}".format(
                train=summary["train_embeddings"],
                val=summary["val_embeddings"],
                test=summary["test_embeddings"],
            )
        )


if __name__ == "__main__":
    main()
