#!/usr/bin/env python3
"""Compare RF × IRT statistics across embedding backbones."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

EMBED_KEYS = {
    "metrics": "rf_metrics.json",
    "signals": "rf_signal_summary.json",
    "irt": "irt_summary.json",
    "correlations": "rf_irt_correlations.json",
}


@dataclass
class RunStats:
    name: str
    overall_accuracy: float
    val_accuracy: float
    oob_accuracy: float
    margin_mean: float
    entropy_mean: float
    corr_margin: float
    corr_entropy: float
    ability_std: float
    difficulty_std: float


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_run(name: str, root: Path) -> RunStats:
    metrics = load_json(root / EMBED_KEYS["metrics"])
    signals = load_json(root / EMBED_KEYS["signals"])
    irt = load_json(root / EMBED_KEYS["irt"])
    corr = load_json(root / EMBED_KEYS["correlations"])
    return RunStats(
        name=name,
        overall_accuracy=float(metrics["overall_accuracy"]),
        val_accuracy=float(metrics["val_accuracy"]),
        oob_accuracy=float(metrics.get("oob_accuracy", np.nan)),
        margin_mean=float(signals["margin"]["mean"]),
        entropy_mean=float(signals["entropy"]["mean"]),
        corr_margin=float(corr["margin"]["pearson"]),
        corr_entropy=float(corr["entropy"]["pearson"]),
        ability_std=float(irt["ability_std"]),
        difficulty_std=float(irt["diff_std"]),
    )


def make_table(rows: Iterable[Tuple[str, float, float]]) -> str:
    header = "| Metric | PCA | MobileNet |\n|---|---|---|"
    body_lines = []
    for metric, pca_value, mobile_value in rows:
        body_lines.append(f"| {metric} | {pca_value:.4f} | {mobile_value:.4f} |")
    return "\n".join([header, *body_lines])


def generate_report(pca: RunStats, mobile: RunStats) -> str:
    metrics_rows = [
        ("Overall accuracy", pca.overall_accuracy, mobile.overall_accuracy),
        ("Validation accuracy", pca.val_accuracy, mobile.val_accuracy),
        ("OOB accuracy", pca.oob_accuracy, mobile.oob_accuracy),
        ("Margin mean", pca.margin_mean, mobile.margin_mean),
        ("Entropy mean", pca.entropy_mean, mobile.entropy_mean),
        ("Pearson δ↔margin", pca.corr_margin, mobile.corr_margin),
        ("Pearson δ↔entropy", pca.corr_entropy, mobile.corr_entropy),
        ("Ability σ", pca.ability_std, mobile.ability_std),
        ("Difficulty σ", pca.difficulty_std, mobile.difficulty_std),
    ]

    doc = [
        "# Embedding Comparison",
        "",
        "PCA reflects the original 128-D projection, MobileNet uses a pretrained MobileNetV3-Large backbone (960-D).",
        "",
        make_table(metrics_rows),
    ]
    return "\n".join(doc) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pca-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing baseline PCA artifacts.",
    )
    parser.add_argument(
        "--mobilenet-dir",
        type=Path,
        default=Path("data/mobilenet"),
        help="Directory containing MobileNet artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/embedding_comparison.md"),
        help="Path to write the markdown summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pca = load_run("PCA", args.pca_dir)
    mobilenet = load_run("MobileNet", args.mobilenet_dir)
    report = generate_report(pca, mobilenet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
