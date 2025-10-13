#!/usr/bin/env python3
"""Fit a 1-PL IRT model on the saved random forest response matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from py_irt.models.one_param_logistic import OneParamLog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--response-matrix",
        type=Path,
        default=Path("data/response_matrix.npz"),
        help="Path to the .npz file containing the binary response matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where IRT parameters and summaries will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of SVI epochs to run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device to use for tensors; 'cuda' requires an available GPU.",
    )
    parser.add_argument(
        "--priors",
        choices=("vague", "hierarchical"),
        default="vague",
        help="Which prior family to use in the IRT model.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print training loss every N epochs when verbose output is enabled.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print loss updates during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for Pyro and PyTorch RNGs.",
    )
    return parser.parse_args()


def load_response_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Response matrix file not found: {path}")
    with np.load(path) as data:
        matrix = data["R"].astype(np.float32)
        y_true = data.get("y_test")  # optional, may be used downstream
        classes = data.get("classes")
    return matrix, y_true, classes


def prepare_tensors(matrix: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, ...]:
    num_models, num_items = matrix.shape
    models_idx, items_idx = np.indices((num_models, num_items))
    models = torch.tensor(models_idx.reshape(-1), dtype=torch.long, device=device)
    items = torch.tensor(items_idx.reshape(-1), dtype=torch.long, device=device)
    responses = torch.tensor(matrix.reshape(-1), dtype=torch.float32, device=device)
    return models, items, responses


def configure_model(
    priors: str, irt_device: str, num_items: int, num_models: int, verbose: bool
) -> OneParamLog:
    return OneParamLog(
        priors=priors,
        device=irt_device,
        num_items=num_items,
        num_models=num_models,
        verbose=verbose,
    )


def fit_irt(
    matrix: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    torch_device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")

    num_models, num_items = matrix.shape
    irt_device = "gpu" if args.device == "cuda" else "cpu"

    irt_model = configure_model(
        priors=args.priors,
        irt_device=irt_device,
        num_items=num_items,
        num_models=num_models,
        verbose=args.verbose,
    )

    models, items, responses = prepare_tensors(matrix, torch_device)

    pyro.util.set_rng_seed(args.seed)
    pyro.clear_param_store()

    optimizer = Adam({"lr": args.learning_rate})
    if args.priors == "vague":
        svi = SVI(irt_model.model_vague, irt_model.guide_vague, optimizer, Trace_ELBO())
    else:
        svi = SVI(
            irt_model.model_hierarchical,
            irt_model.guide_hierarchical,
            optimizer,
            Trace_ELBO(),
        )

    losses = []
    for epoch in range(args.epochs):
        loss = svi.step(models, items, responses)
        losses.append(loss)
        if args.verbose and (epoch + 1) % args.log_every == 0:
            print(f"[epoch {epoch + 1:04d}] loss: {loss:.4f}")

    store = pyro.get_param_store()
    ability_loc = store["loc_ability"].detach().cpu().numpy()
    ability_scale = store["scale_ability"].detach().cpu().numpy()
    diff_loc = store["loc_diff"].detach().cpu().numpy()
    diff_scale = store["scale_diff"].detach().cpu().numpy()

    return {
        "losses": np.array(losses, dtype=np.float32),
        "ability_loc": ability_loc,
        "ability_scale": ability_scale,
        "diff_loc": diff_loc,
        "diff_scale": diff_scale,
    }


def summarise_results(
    matrix: np.ndarray,
    params: Dict[str, np.ndarray],
    priors: str,
    epochs: int,
    learning_rate: float,
) -> Dict[str, float]:
    tree_accuracy = matrix.mean(axis=1)
    item_accuracy = matrix.mean(axis=0)

    def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    ability_corr = safe_corr(params["ability_loc"], tree_accuracy)
    difficulty_corr = safe_corr(params["diff_loc"], 1.0 - item_accuracy)

    summary = {
        "priors": priors,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "num_models": int(matrix.shape[0]),
        "num_items": int(matrix.shape[1]),
        "final_loss": float(params["losses"][-1]),
        "loss_min": float(params["losses"].min()),
        "loss_max": float(params["losses"].max()),
        "ability_mean": float(params["ability_loc"].mean()),
        "ability_std": float(params["ability_loc"].std()),
        "ability_min": float(params["ability_loc"].min()),
        "ability_max": float(params["ability_loc"].max()),
        "diff_mean": float(params["diff_loc"].mean()),
        "diff_std": float(params["diff_loc"].std()),
        "diff_min": float(params["diff_loc"].min()),
        "diff_max": float(params["diff_loc"].max()),
        "tree_accuracy_mean": float(tree_accuracy.mean()),
        "tree_accuracy_std": float(tree_accuracy.std()),
        "item_accuracy_mean": float(item_accuracy.mean()),
        "item_accuracy_std": float(item_accuracy.std()),
        "ability_tree_accuracy_corr": ability_corr,
        "difficulty_item_error_corr": difficulty_corr,
    }
    return summary


def save_outputs(
    output_dir: Path,
    params: Dict[str, np.ndarray],
    summary: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "irt_parameters.npz",
        ability_loc=params["ability_loc"],
        ability_scale=params["ability_scale"],
        diff_loc=params["diff_loc"],
        diff_scale=params["diff_scale"],
    )
    np.save(output_dir / "irt_losses.npy", params["losses"])
    with (output_dir / "irt_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    matrix, _, _ = load_response_matrix(args.response_matrix)
    params = fit_irt(matrix, args)
    summary = summarise_results(
        matrix,
        params,
        priors=args.priors,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    save_outputs(args.output_dir, params, summary)
    print(
        "Training complete. Final loss: "
        f"{summary['final_loss']:.4f}. Summary saved to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
