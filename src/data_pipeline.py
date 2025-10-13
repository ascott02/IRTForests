"""Data preparation utilities for the RF × IRT study.

The module is intentionally lightweight so it can be imported both by
command-line entry points and notebooks without executing heavy work on import.
It provides three primary facilities:

1. :class:`SubsetConfig` – structured configuration for consistent CIFAR-10
    sampling.
2. :func:`save_cifar10_subset` – idempotent download + stratified sampling that
    materializes cached tensors on disk.
3. :func:`compute_pca_embeddings` – embeddings from cached tensors via PCA,
    returning both the saved path and useful shape metadata.
4. :func:`compute_mobilenet_embeddings` – feature extraction via an ImageNet
    pre-trained MobileNet backbone for richer representations.

None of these utilities perform work at import time, keeping dry-run scenarios
cheap until the CLI or notebook drives execution.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
IMAGENET_MEAN = (0.4850, 0.4560, 0.4060)
IMAGENET_STD = (0.2290, 0.2240, 0.2250)


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
            raise ValueError(
                f"Requested {per_class} samples for class {cls}, only {len(cls_idx)} available"
            )
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


def compute_mobilenet_embeddings(
    subset_path: pathlib.Path,
    *,
    out_path: pathlib.Path | None = None,
    batch_size: int = 64,
    image_size: int = 224,
    device: str | torch.device | None = None,
) -> Tuple[pathlib.Path, Dict[str, Tuple[int, ...]]]:
    """Compute MobileNet embeddings from a saved CIFAR-10 subset archive.

    The cached tensors are first restored to their un-normalised 0–1 pixel
    domain before being resized and normalised to the ImageNet statistics
    expected by the MobileNet backbone. The resulting penultimate feature
    vectors are persisted alongside the original labels for downstream use.
    """

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.to(resolved_device)
    model.eval()

    subset = np.load(subset_path)
    cifar_mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    cifar_std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
    imagenet_mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(resolved_device)
    imagenet_std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(resolved_device)

    def _extract(split: str) -> np.ndarray:
        images = torch.from_numpy(subset[f"x_{split}"].astype(np.float32))
        images = images * cifar_std + cifar_mean
        dataset = TensorDataset(images)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(resolved_device)
                batch = F.interpolate(batch, size=(image_size, image_size), mode="bilinear", align_corners=False)
                batch = batch.clamp(0.0, 1.0)
                batch = (batch - imagenet_mean) / imagenet_std
                features = model.features(batch)
                pooled = model.avgpool(features)
                flattened = torch.flatten(pooled, 1)
                embeddings.append(flattened.cpu())
        return torch.cat(embeddings, dim=0).numpy()

    train_embeddings = _extract("train")
    val_embeddings = _extract("val")
    test_embeddings = _extract("test")

    if out_path is None:
        out_path = subset_path.parent / "cifar10_mobilenet_embeddings.npz"

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
        feature_dim=np.array([train_embeddings.shape[1]], dtype=np.int64),
        image_size=np.array([image_size], dtype=np.int64),
        backbone=np.array(["mobilenet_v3_large"]),
        device=np.array([str(resolved_device)]),
    )
    summary = {
        "train_embeddings": train_embeddings.shape,
        "val_embeddings": val_embeddings.shape,
        "test_embeddings": test_embeddings.shape,
        "feature_dim": train_embeddings.shape[1],
        "device": str(resolved_device),
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
    parser.add_argument("--embeddings", action="store_true", help="Also compute embeddings for cached subset")
    parser.add_argument(
        "--embedding-type",
        choices=["pca", "mobilenet"],
        default="pca",
        help="Embedding backend to compute when --embeddings is supplied",
    )
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--embeddings-out", type=pathlib.Path, default=None, help="Destination for embeddings archive")
    parser.add_argument("--mobilenet-batch-size", type=int, default=64)
    parser.add_argument("--mobilenet-image-size", type=int, default=224)
    parser.add_argument(
        "--mobilenet-device",
        type=str,
        default=None,
        help="Force the device used for MobileNet feature extraction (e.g. cuda:0)",
    )
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
        if args.embedding_type == "pca":
            embeddings_path, summary = compute_pca_embeddings(
                subset_path,
                n_components=args.n_components,
                out_path=args.embeddings_out,
            )
        else:
            embeddings_path, summary = compute_mobilenet_embeddings(
                subset_path,
                out_path=args.embeddings_out,
                batch_size=args.mobilenet_batch_size,
                image_size=args.mobilenet_image_size,
                device=args.mobilenet_device,
            )

        print(f"Saved {args.embedding_type.upper()} embeddings to {embeddings_path.resolve()}")
        print(
            "Shapes → train:{train} val:{val} test:{test}".format(
                train=summary["train_embeddings"],
                val=summary["val_embeddings"],
                test=summary["test_embeddings"],
            )
        )


if __name__ == "__main__":
    main()
