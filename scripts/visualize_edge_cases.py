#!/usr/bin/env python3
"""Render targeted edge-case thumbnails with optional uncertainty annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=int, nargs="+", required=True, help="Item indices to visualize.")
    parser.add_argument(
        "--dataset-type",
        choices=("cifar10", "mnist"),
        default="cifar10",
        help="Dataset family used to render thumbnails.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/torchvision"),
        help="Root directory containing cached torchvision data.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Dataset split to use for visualization.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the composed figure will be written.",
    )
    parser.add_argument("--title", type=str, default="Edge Cases", help="Figure title displayed at the top.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the output grid.")
    parser.add_argument(
        "--extremes-json",
        type=Path,
        help="Optional JSON containing difficulty/accuracy stats (e.g., data/irt_extremes.json).",
    )
    parser.add_argument("--margin-npy", type=Path, help="Optional .npy array with margins indexed by item id.")
    parser.add_argument("--entropy-npy", type=Path, help="Optional .npy array with entropies indexed by item id.")
    parser.add_argument(
        "--pred-npy",
        type=Path,
        help="Optional .npy array with ensemble majority predictions indexed by item id.",
    )
    parser.add_argument(
        "--proba-npy",
        type=Path,
        help="Optional .npy array with class probabilities (same indexing as ids).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of highest-probability classes to display when probabilities are provided.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=2.8,
        help="Scaling factor (inches) applied per column to control figure width.",
    )
    return parser.parse_args()


def load_dataset(dataset_type: str, root: Path, split: str) -> tv_datasets.VisionDataset:
    transform = T.ToTensor()
    train = split == "train"
    if dataset_type == "cifar10":
        return tv_datasets.CIFAR10(root=str(root), train=train, download=False, transform=transform)
    if dataset_type == "mnist":
        return tv_datasets.MNIST(root=str(root), train=train, download=False, transform=transform)
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_extremes(path: Optional[Path]) -> Dict[int, Dict[str, float]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    summary: Dict[int, Dict[str, float]] = {}
    for key in ("hardest_items", "easiest_items"):
        for item in data.get(key, []):
            summary[item["item_id"]] = {
                "difficulty": item.get("difficulty"),
                "tree_accuracy": item.get("accuracy"),
            }
    return summary


def topk_probs(row: np.ndarray, k: int) -> Iterable[tuple[int, float]]:
    if row.ndim != 1:
        raise ValueError("Probability row should be 1D")
    k = min(k, row.shape[0])
    top_indices = np.argsort(row)[::-1][:k]
    return [(int(idx), float(row[idx])) for idx in top_indices]


def render_edges(
    dataset: Any,
    ids: List[int],
    output: Path,
    classes: List[str],
    difficulty_lookup: Dict[int, Dict[str, float]],
    margin: Optional[np.ndarray],
    entropy: Optional[np.ndarray],
    preds: Optional[np.ndarray],
    probas: Optional[np.ndarray],
    topk: int,
    title: str,
    cols: int,
    figscale: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = int(np.ceil(len(ids) / cols))
    fig_width = max(4.0, cols * figscale)
    fig_height = max(3.5, rows * figscale)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = np.atleast_1d(axes).flatten()

    for ax in axes[len(ids) :]:
        ax.axis("off")

    for ax, idx in zip(axes, ids):
        img, label = dataset[idx]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            array = img.numpy()
            if array.shape[0] == 1:
                ax.imshow(array.squeeze(0), cmap="gray")
            else:
                ax.imshow(np.transpose(array, (1, 2, 0)))
        else:
            ax.imshow(img)
        true_label = classes[label] if 0 <= label < len(classes) else str(label)
        pred_label = None
        if preds is not None:
            pred_idx = int(preds[idx])
            if 0 <= pred_idx < len(classes):
                pred_label = classes[pred_idx]
            else:
                pred_label = str(pred_idx)
        title_line = f"#{idx} {true_label}"
        if pred_label is not None:
            title_line += f" → {pred_label}"
        ax.set_title(title_line, fontsize=11)
        ax.axis("off")

        annotations: List[str] = []
        extras = difficulty_lookup.get(idx)
        if extras and extras.get("difficulty") is not None:
            annotations.append(f"δ={extras['difficulty']:.2f}")
        if extras and extras.get("tree_accuracy") is not None:
            annotations.append(f"acc={extras['tree_accuracy']:.2f}")
        if margin is not None:
            annotations.append(f"margin={float(margin[idx]):.2f}")
        if entropy is not None:
            annotations.append(f"H={float(entropy[idx]):.2f}")
        if probas is not None:
            top_pairs = topk_probs(probas[idx], topk)
            readable = ", ".join(f"{classes[c]}={p:.2f}" for c, p in top_pairs)
            annotations.append(f"top: {readable}")
        if annotations:
            ax.text(
                0.5,
                -0.12,
                "\n".join(annotations),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                wrap=True,
            )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.12)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote edge-case panel to {output}")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_type, args.dataset_root, args.split)
    classes = list(dataset.classes) if hasattr(dataset, "classes") else [str(i) for i in range(10)]  # type: ignore[attr-defined]

    difficulty_lookup = load_extremes(args.extremes_json)
    margin = np.load(args.margin_npy) if args.margin_npy else None
    entropy = np.load(args.entropy_npy) if args.entropy_npy else None
    preds = np.load(args.pred_npy) if args.pred_npy else None
    probas = np.load(args.proba_npy) if args.proba_npy else None

    render_edges(
        dataset=dataset,
        ids=args.ids,
        output=args.output,
        classes=[str(c) for c in classes],
        difficulty_lookup=difficulty_lookup,
        margin=margin,
        entropy=entropy,
        preds=preds,
        probas=probas,
        topk=args.topk,
        title=args.title,
        cols=args.cols,
        figscale=args.figscale,
    )


if __name__ == "__main__":
    main()
