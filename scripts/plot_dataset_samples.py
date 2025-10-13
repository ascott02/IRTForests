#!/usr/bin/env python3
"""Create sample grids from cached dataset bundles."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray | None, np.ndarray | None]:
    """Return images, labels, class names, and optional normalization stats."""
    with np.load(npz_path) as data:
        if "x_train" in data:
            images = data["x_train"].astype(np.float32)
            labels = data["y_train"].astype(np.int64)
        elif "train_embeddings" in data and "image_shape" in data:
            shape = tuple(int(x) for x in data["image_shape"].tolist())
            flat = data["train_embeddings"].astype(np.float32)
            images = flat.reshape((-1, *shape))
            labels = data["y_train"].astype(np.int64)
        else:
            raise ValueError(f"Unsupported dataset format in {npz_path}")

        classes_raw = data.get("classes")
        if classes_raw is None:
            raise ValueError(f"Missing class names in {npz_path}")
        classes = [str(cls) for cls in classes_raw.tolist()]

        mean = data.get("mean")
        std = data.get("std")

    return images, labels, classes, mean, std


def _unnormalize(image: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return image

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    if image.ndim == 3:
        # Broadcast channel stats for both channel-first and channel-last layouts.
        if image.shape[0] == mean_arr.shape[0]:  # (C, H, W)
            mean_arr = mean_arr[:, None, None]
            std_arr = std_arr[:, None, None]
        elif image.shape[-1] == mean_arr.shape[0]:  # (H, W, C)
            mean_arr = mean_arr[None, None, :]
            std_arr = std_arr[None, None, :]
        else:
            mean_arr = mean_arr.reshape((1,) * (image.ndim - 1) + (-1,))
            std_arr = std_arr.reshape((1,) * (image.ndim - 1) + (-1,))
    else:
        mean_arr = np.reshape(mean_arr, (1,) * image.ndim)
        std_arr = np.reshape(std_arr, (1,) * image.ndim)

    return image * std_arr + mean_arr


def _to_display(image: np.ndarray) -> Tuple[np.ndarray, str | None]:
    """Convert array to display-ready format and return (image, cmap)."""
    if image.ndim == 3 and image.shape[0] in {1, 3}:  # channel-first
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.shape[2] == 1:
        return np.squeeze(image, axis=2), "gray"
    if image.ndim == 2:
        return image, "gray"
    return image, None


def plot_samples(
    npz_path: Path,
    output_path: Path,
    count: int = 16,
    cols: int = 4,
    seed: int = 7,
    title: str | None = None,
) -> None:
    images, labels, classes, mean, std = _load_dataset(npz_path)
    total = images.shape[0]
    if count > total:
        raise ValueError(f"Requested {count} samples but only {total} available in {npz_path}")

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False)

    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, indices):
        img = images[idx]
        img = _unnormalize(img, mean, std)
        img = np.clip(img, 0.0, 1.0)
        display_img, cmap = _to_display(img)
        ax.imshow(display_img, cmap=cmap)
        class_id = int(labels[idx])
        label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        ax.set_title(label, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset .npz bundle")
    parser.add_argument("--output", type=Path, required=True, help="Destination image path")
    parser.add_argument("--count", type=int, default=16, help="Number of samples to show (default: 16)")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns in the grid (default: 4)")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    args = parser.parse_args()

    plot_samples(args.data, args.output, count=args.count, cols=args.cols, seed=args.seed, title=args.title)


if __name__ == "__main__":
    main()
