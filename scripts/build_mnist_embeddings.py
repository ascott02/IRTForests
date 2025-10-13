#!/usr/bin/env python3
"""Build a stratified MNIST subset and persist flattened embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from torchvision import datasets, transforms

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory for torchvision downloads and saved embeddings.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination .npz file; defaults to data/mnist_embeddings.npz",
    )
    return parser.parse_args()


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)]
    )


def _stratified_indices(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = []
    for cls in range(10):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) < per_class:
            raise ValueError(
                f"Requested {per_class} samples for class {cls}, only {len(cls_idx)} available"
            )
        indices.append(rng.choice(cls_idx, size=per_class, replace=False))
    return np.concatenate(indices)


def _collect(dataset: datasets.MNIST, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img.numpy())
        labels.append(label)
    stacked = np.stack(images)
    flat = stacked.reshape(stacked.shape[0], -1)
    return flat, np.array(labels, dtype=np.int64)


def build_mnist_subset(args: argparse.Namespace) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    transform = _build_transform()
    root = args.data_root / "torchvision"
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_labels = np.array(train_dataset.targets)
    train_idx = _stratified_indices(
        train_labels, args.train_per_class + args.val_per_class, args.seed
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    train_split = train_idx[: 10 * args.train_per_class]
    val_split = train_idx[10 * args.train_per_class :]

    test_labels = np.array(test_dataset.targets)
    test_idx = _stratified_indices(test_labels, args.test_per_class, args.seed)

    x_train, y_train = _collect(train_dataset, train_split)
    x_val, y_val = _collect(train_dataset, val_split)
    x_test, y_test = _collect(test_dataset, test_idx)

    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }


def save_embeddings(data: Dict[str, Tuple[np.ndarray, np.ndarray]], args: argparse.Namespace) -> Path:
    out_dir = args.data_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (out_dir / "mnist_embeddings.npz")

    np.savez(
        out_path,
        train_embeddings=data["train"][0],
        val_embeddings=data["val"][0],
        test_embeddings=data["test"][0],
        y_train=data["train"][1],
        y_val=data["val"][1],
        y_test=data["test"][1],
        classes=np.arange(10, dtype=np.int64),
        mean=np.array(MNIST_MEAN),
        std=np.array(MNIST_STD),
        image_shape=np.array([1, 28, 28], dtype=np.int64),
    )
    return out_path


def main() -> None:
    args = parse_args()
    subset = build_mnist_subset(args)
    out_path = save_embeddings(subset, args)
    print(
        f"Saved MNIST embeddings to {out_path.resolve()} \n"
        f"Train/Val/Test shapes: {subset['train'][0].shape}, {subset['val'][0].shape}, {subset['test'][0].shape}"
    )


if __name__ == "__main__":
    main()
