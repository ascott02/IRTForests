#!/usr/bin/env python3
"""End-to-end pipeline to regenerate Random Forest + IRT artifacts for all studies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class IRTConfig:
    model: str
    epochs: int
    learning_rate: float
    log_every: int
    suffix: str | None = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    embeddings: Path
    data_dir: Path
    model_path: Path
    tree_attrs: Path
    figure_dir: Path
    label: str
    prefix_1pl: str
    prefix_2pl: str
    tree_prefix: str
    tree_corr_json: Path
    tree_table_csv: Path
    class_summary_json: Path
    class_summary_fig: Path
    dataset_type: str
    noun: str
    dataset_root: Path
    irt: Sequence[IRTConfig]

    @property
    def response_matrix(self) -> Path:
        return self.data_dir / "response_matrix.npz"

    @property
    def probs_path(self) -> Path:
        return self.data_dir / "rf_test_proba.npy"

    @property
    def labels_source(self) -> Path:
        return self.data_dir / "response_matrix.npz"

    @property
    def margins_path(self) -> Path:
        return self.data_dir / "rf_margins.npy"

    @property
    def entropy_path(self) -> Path:
        return self.data_dir / "rf_entropy.npy"

    @property
    def irt_params_1pl(self) -> Path:
        return self.data_dir / "irt_parameters.npz"

    @property
    def irt_params_2pl(self) -> Path:
        return self.data_dir / "irt_parameters_2pl.npz"

    @property
    def irt_params_3pl(self) -> Path:
        return self.data_dir / "irt_parameters_3pl.npz"

    @property
    def extremes_path(self) -> Path:
        return self.data_dir / "irt_extremes.json"

    @property
    def metrics_json(self) -> Path:
        return self.data_dir / "rf_metrics.json"

    @property
    def confusion_path(self) -> Path:
        return self.data_dir / "rf_confusion.npy"


RUNS: list[RunConfig] = [
    RunConfig(
        name="pca",
        embeddings=ROOT / "data/cifar10_embeddings.npz",
        data_dir=ROOT / "data",
        model_path=ROOT / "models/random_forest.joblib",
        tree_attrs=ROOT / "data/tree_attributes.json",
        figure_dir=ROOT / "figures",
        label="CIFAR-10 · PCA · 2000 trees",
        prefix_1pl="",
        prefix_2pl="pca_2pl",
        tree_prefix="pca_tree",
        tree_corr_json=ROOT / "data/tree_attribute_correlations_pca.json",
        tree_table_csv=ROOT / "data/tree_attributes_with_signals.csv",
        class_summary_json=ROOT / "data/class_difficulty_summary.json",
        class_summary_fig=ROOT / "figures/class_difficulty_vs_error.png",
        dataset_type="cifar10",
        noun="items",
        dataset_root=ROOT / "data/torchvision",
        irt=(
            IRTConfig(model="1pl", epochs=600, learning_rate=0.05, log_every=100, suffix=None),
            IRTConfig(model="2pl", epochs=800, learning_rate=0.02, log_every=100, suffix="_2pl"),
        ),
    ),
    RunConfig(
        name="mobilenet",
        embeddings=ROOT / "data/cifar10_mobilenet_embeddings.npz",
        data_dir=ROOT / "data/mobilenet",
        model_path=ROOT / "models/random_forest_mobilenet.joblib",
        tree_attrs=ROOT / "data/mobilenet/tree_attributes.json",
        figure_dir=ROOT / "figures/mobilenet",
        label="CIFAR-10 · MobileNet · 2000 trees",
        prefix_1pl="mobilenet",
        prefix_2pl="mobilenet_2pl",
        tree_prefix="mobilenet_tree",
        tree_corr_json=ROOT / "data/mobilenet/tree_attribute_correlations.json",
        tree_table_csv=ROOT / "data/mobilenet/tree_attributes_with_signals.csv",
        class_summary_json=ROOT / "data/mobilenet/class_difficulty_summary.json",
        class_summary_fig=ROOT / "figures/mobilenet/class_difficulty_vs_error.png",
        dataset_type="cifar10",
        noun="items",
        dataset_root=ROOT / "data/torchvision",
        irt=(
            IRTConfig(model="1pl", epochs=600, learning_rate=0.05, log_every=100, suffix=None),
            IRTConfig(model="2pl", epochs=800, learning_rate=0.02, log_every=100, suffix="_2pl"),
            IRTConfig(model="3pl", epochs=1000, learning_rate=0.01, log_every=100, suffix="_3pl"),
        ),
    ),
    RunConfig(
        name="mnist",
        embeddings=ROOT / "data/mnist_embeddings.npz",
        data_dir=ROOT / "data/mnist",
        model_path=ROOT / "models/random_forest_mnist.joblib",
        tree_attrs=ROOT / "data/mnist/tree_attributes.json",
        figure_dir=ROOT / "figures/mnist",
        label="MNIST · 2000 trees",
        prefix_1pl="mnist",
        prefix_2pl="mnist_2pl",
        tree_prefix="mnist_tree",
        tree_corr_json=ROOT / "data/mnist/tree_attribute_correlations.json",
        tree_table_csv=ROOT / "data/mnist/tree_attributes_with_signals.csv",
        class_summary_json=ROOT / "data/mnist/class_difficulty_summary.json",
        class_summary_fig=ROOT / "figures/mnist/class_difficulty_vs_error.png",
        dataset_type="mnist",
        noun="digits",
        dataset_root=ROOT / "data/torchvision",
        irt=(
            IRTConfig(model="1pl", epochs=600, learning_rate=0.05, log_every=100, suffix=None),
            IRTConfig(model="2pl", epochs=800, learning_rate=0.02, log_every=100, suffix="_2pl"),
        ),
    ),
]


def invoke(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"→ {printable}")
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def train_random_forest(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_training:
        return
    invoke(
        [
            PYTHON,
            "scripts/train_random_forest.py",
            "--n-estimators",
            str(args.n_estimators),
            "--parallel-backend",
            args.parallel_backend,
            "--embeddings",
            str(run.embeddings.relative_to(ROOT)),
            "--output-dir",
            str(run.data_dir.relative_to(ROOT)),
            "--tree-attributes",
            str(run.tree_attrs.relative_to(ROOT)),
            "--save-model",
            str(run.model_path.relative_to(ROOT)),
        ]
    )


def compute_rf_signals(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_signals:
        return
    invoke(
        [
            PYTHON,
            "scripts/compute_rf_signals.py",
            "--probabilities",
            str(run.probs_path.relative_to(ROOT)),
            "--labels",
            str(run.labels_source.relative_to(ROOT)),
            "--output-dir",
            str(run.data_dir.relative_to(ROOT)),
        ]
    )


def fit_irt_models(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_irt:
        return
    for config in run.irt:
        cmd = [
            PYTHON,
            "scripts/fit_irt.py",
            "--response-matrix",
            str(run.response_matrix.relative_to(ROOT)),
            "--output-dir",
            str(run.data_dir.relative_to(ROOT)),
            "--epochs",
            str(config.epochs),
            "--learning-rate",
            str(config.learning_rate),
            "--model",
            config.model,
            "--log-every",
            str(config.log_every),
        ]
        if config.suffix is not None:
            cmd.extend(["--suffix", config.suffix])
        invoke(cmd)


def write_extremes(run: RunConfig, count: int = 10) -> None:
    if not run.irt_params_1pl.exists():
        return
    response = np.load(run.response_matrix)
    R = response["R"].astype(np.float32)
    tree_acc = R.mean(axis=1)
    item_acc = R.mean(axis=0)

    params = np.load(run.irt_params_1pl)
    ability = params["ability_loc"].astype(np.float32)
    difficulty = params["diff_loc"].astype(np.float32)

    def serialise(indices: Iterable[int], values: np.ndarray, acc: np.ndarray, key: str) -> list[dict[str, float | int]]:
        records: list[dict[str, float | int]] = []
        for idx in indices:
            record = {
                key: int(idx),
                "ability" if key == "tree_id" else "difficulty": float(values[idx]),
                "accuracy": float(acc[idx]),
            }
            records.append(record)
        return records

    top_idx = ability.argsort()[::-1][:count]
    bottom_idx = ability.argsort()[:count]
    hardest_idx = difficulty.argsort()[::-1][:count]
    easiest_idx = difficulty.argsort()[:count]

    payload = {
        "top_trees": serialise(top_idx, ability, tree_acc, "tree_id"),
        "bottom_trees": serialise(bottom_idx, ability, tree_acc, "tree_id"),
        "hardest_items": serialise(hardest_idx, difficulty, item_acc, "item_id"),
        "easiest_items": serialise(easiest_idx, difficulty, item_acc, "item_id"),
    }
    run.extremes_path.parent.mkdir(parents=True, exist_ok=True)
    with run.extremes_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def visualize_extremes(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots:
        return
    if not run.extremes_path.exists():
        return
    run.figure_dir.mkdir(parents=True, exist_ok=True)
    invoke(
        [
            PYTHON,
            "scripts/visualize_difficulty_extremes.py",
            "--extremes",
            str(run.extremes_path.relative_to(ROOT)),
            "--dataset",
            str(run.dataset_root.relative_to(ROOT)),
            "--split",
            args.extreme_split,
            "--count",
            str(args.extreme_count),
            "--output-prefix",
            str(run.figure_dir.relative_to(ROOT)),
            "--dataset-type",
            run.dataset_type,
            "--noun",
            run.noun,
        ]
    )


def run_correlations(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots:
        return
    run.figure_dir.mkdir(parents=True, exist_ok=True)
    # 1PL difficulty correlations
    invoke(
        [
            PYTHON,
            "scripts/analyze_rf_irt_correlations.py",
            "--margin",
            str(run.margins_path.relative_to(ROOT)),
            "--entropy",
            str(run.entropy_path.relative_to(ROOT)),
            "--irt-params",
            str(run.irt_params_1pl.relative_to(ROOT)),
            "--output-dir",
            str(run.data_dir.relative_to(ROOT)),
            "--figures-dir",
            str(run.figure_dir.relative_to(ROOT)),
            "--prefix",
            run.prefix_1pl,
        ]
    )
    # 2PL correlations with discrimination
    invoke(
        [
            PYTHON,
            "scripts/analyze_rf_irt_correlations.py",
            "--margin",
            str(run.margins_path.relative_to(ROOT)),
            "--entropy",
            str(run.entropy_path.relative_to(ROOT)),
            "--irt-params",
            str(run.irt_params_2pl.relative_to(ROOT)),
            "--output-dir",
            str(run.data_dir.relative_to(ROOT)),
            "--figures-dir",
            str((ROOT / "figures").relative_to(ROOT)),
            "--prefix",
            run.prefix_2pl,
            "--parameters",
            "difficulty",
            "discrimination",
            "--output-name",
            "rf_irt_correlations_2pl.json",
        ]
    )


def run_tree_analysis(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_tree:
        return
    run.figure_dir.mkdir(parents=True, exist_ok=True)
    invoke(
        [
            PYTHON,
            "scripts/analyze_tree_attribute_correlations.py",
            "--tree-attributes",
            str(run.tree_attrs.relative_to(ROOT)),
            "--irt-params",
            str(run.irt_params_2pl.relative_to(ROOT)),
            "--response-matrix",
            str(run.response_matrix.relative_to(ROOT)),
            "--output-json",
            str(run.tree_corr_json.relative_to(ROOT)),
            "--figures-dir",
            str((ROOT / "figures").relative_to(ROOT)),
            "--table-output",
            str(run.tree_table_csv.relative_to(ROOT)),
            "--prefix",
            run.tree_prefix,
        ]
    )


def run_class_summary(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots:
        return
    invoke(
        [
            PYTHON,
            "scripts/class_difficulty_summary.py",
            "--response-matrix",
            str(run.response_matrix.relative_to(ROOT)),
            "--irt-params",
            str(run.irt_params_1pl.relative_to(ROOT)),
            "--rf-metrics",
            str(run.metrics_json.relative_to(ROOT)),
            "--output",
            str(run.class_summary_json.relative_to(ROOT)),
            "--figure",
            str(run.class_summary_fig.relative_to(ROOT)),
        ]
    )


def run_wright_map(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots:
        return
    run.figure_dir.mkdir(parents=True, exist_ok=True)
    output = run.figure_dir / "wright_map.png"
    invoke(
        [
            PYTHON,
            "scripts/plot_wright_map.py",
            "--irt-params",
            str(run.irt_params_1pl.relative_to(ROOT)),
            "--output",
            str(output.relative_to(ROOT)),
        ]
    )


def run_additional_diagnostics(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots:
        return
    run.figure_dir.mkdir(parents=True, exist_ok=True)
    invoke(
        [
            PYTHON,
            "scripts/plot_additional_diagnostics.py",
            "--data-dir",
            str(run.data_dir.relative_to(ROOT)),
            "--figures-dir",
            str(run.figure_dir.relative_to(ROOT)),
            "--label",
            run.label,
            "--suffix",
            "_2pl",
        ]
    )


def run_confusion_matrix(run: RunConfig, args: argparse.Namespace) -> None:
    if args.skip_plots or run.name != "pca":
        return
    invoke(
        [
            PYTHON,
            "scripts/plot_confusion_matrix.py",
            "--confusion",
            str(run.confusion_path.relative_to(ROOT)),
            "--metrics",
            str(run.metrics_json.relative_to(ROOT)),
            "--normalize",
        ]
    )


def compute_fit_statistics(args: argparse.Namespace) -> None:
    if args.skip_fitstats:
        return
    roots = [str(run.data_dir.relative_to(ROOT)) for run in RUNS]
    invoke(
        [
            PYTHON,
            "scripts/compute_fit_statistics.py",
            "--overwrite",
            *roots,
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-training", action="store_true", help="Skip Random Forest retraining step.")
    parser.add_argument("--skip-signals", action="store_true", help="Skip recomputing RF margin/entropy signals.")
    parser.add_argument("--skip-irt", action="store_true", help="Skip IRT fitting (assumes parameters already exist).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation (correlations, diagnostics, extremes, class summaries, wright maps).")
    parser.add_argument("--skip-tree", action="store_true", help="Skip tree attribute correlation analysis.")
    parser.add_argument("--skip-fitstats", action="store_true", help="Skip item/person fit statistic recomputation.")
    parser.add_argument("--n-estimators", type=int, default=2000, help="Number of trees to train per Random Forest run (default: 2000).")
    parser.add_argument("--parallel-backend", choices=("threading", "loky"), default="threading", help="Parallel backend forwarded to train_random_forest.")
    parser.add_argument("--extreme-count", type=int, default=10, help="Number of hardest/easiest examples to include in montages.")
    parser.add_argument("--extreme-split", choices=("train", "test"), default="test", help="Dataset split used when rendering extremes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run in RUNS:
        train_random_forest(run, args)
        compute_rf_signals(run, args)
        fit_irt_models(run, args)
        write_extremes(run)
        visualize_extremes(run, args)
        run_correlations(run, args)
        run_tree_analysis(run, args)
        run_class_summary(run, args)
        run_wright_map(run, args)
        run_additional_diagnostics(run, args)
        run_confusion_matrix(run, args)
    compute_fit_statistics(args)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
