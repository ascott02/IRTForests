#!/usr/bin/env python3
"""Render thumbnails of the hardest and easiest CIFAR-10 items based on IRT difficulty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extremes",
        type=Path,
        default=Path("data/irt_extremes.json"),
        help="JSON with item_id and difficulty stats (as generated earlier).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/torchvision"),
        help="Root directory containing cached CIFAR-10 data (train/test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "test"),
        default="test",
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of items to show for hardest/easiest examples.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures"),
        help="Directory to write output montage images.",
    )
    return parser.parse_args()


def load_extreme_ids(path: Path, count: int) -> Tuple[List[int], List[int]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    hardest = [item["item_id"] for item in data["hardest_items"][:count]]
    easiest = [item["item_id"] for item in data["easiest_items"][:count]]
    return hardest, easiest


def load_dataset(root: Path, split: str) -> torchvision.datasets.CIFAR10:
    transform = T.ToTensor()
    train = split == "train"
    return torchvision.datasets.CIFAR10(root=str(root), train=train, download=False, transform=transform)


def build_montage(
    dataset: torchvision.datasets.CIFAR10,
    indices: List[int],
    title: str,
    output_path: Path,
    cols: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(indices)
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(dataset.classes[label], fontsize=10)
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hardest_ids, easiest_ids = load_extreme_ids(args.extremes, args.count)
    dataset = load_dataset(args.dataset, args.split)

    build_montage(
        dataset,
        hardest_ids,
        title=f"Top {args.count} Hardest Items (δ high)",
        output_path=args.output_prefix / f"hardest_items_{args.split}.png",
    )
    build_montage(
        dataset,
        easiest_ids,
        title=f"Top {args.count} Easiest Items (δ low)",
        output_path=args.output_prefix / f"easiest_items_{args.split}.png",
    )
    print(
        f"Wrote montages: {args.output_prefix / f'hardest_items_{args.split}.png'} and "
        f"{args.output_prefix / f'easiest_items_{args.split}.png'}"
    )


if __name__ == "__main__":
    main()
