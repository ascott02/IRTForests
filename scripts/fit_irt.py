#!/usr/bin/env python3
"""Fit an IRT model (1PL/2PL/3PL) on a saved random forest response matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam  # type: ignore[attr-defined]

from py_irt.models.one_param_logistic import OneParamLog
from py_irt.models.two_param_logistic import TwoParamLog

from torch.distributions import constraints


class ThreeParamLog:
    """Minimal 3PL IRT model with vague priors."""

    def __init__(self, priors: str, device: str, num_items: int, num_models: int, verbose: bool = False):
        if priors != "vague":
            raise ValueError("The 3PL implementation currently supports only vague priors.")
        if device not in {"cpu", "gpu"}:
            raise ValueError("Options for device are cpu and gpu")
        if num_items <= 0:
            raise ValueError("Number of items must be greater than 0")
        if num_models <= 0:
            raise ValueError("Number of subjects must be greater than 0")
        self.priors = priors
        self.device = device
        self.num_items = num_items
        self.num_models = num_models
        self.verbose = verbose

    def model_vague(self, models: torch.Tensor, items: torch.Tensor, obs: torch.Tensor) -> None:
        with pyro.plate("thetas", self.num_models, device=self.device):
            ability = pyro.sample(
                "theta",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(1.0, device=self.device),
                ),
            )
        with pyro.plate("items", self.num_items, device=self.device):
            difficulty = pyro.sample(
                "b",
                dist.Normal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.5, device=self.device),
                ),
            )
            slope = pyro.sample(
                "a",
                dist.LogNormal(
                    torch.tensor(0.0, device=self.device),
                    torch.tensor(0.3, device=self.device),
                ),
            )
            guess = pyro.sample(
                "c",
                dist.Beta(
                    torch.tensor(2.0, device=self.device),
                    torch.tensor(8.0, device=self.device),
                ),
            )

        logits = slope[items] * (ability[models] - difficulty[items])
        probs = guess[items] + (1.0 - guess[items]) * torch.sigmoid(logits)
        probs = probs.clamp(1e-5, 1.0 - 1e-5)
        with pyro.plate("observe_data", obs.size(0), device=self.device):
            pyro.sample("obs", dist.Bernoulli(probs=probs), obs=obs)

    def guide_vague(self, models: torch.Tensor, items: torch.Tensor, obs: torch.Tensor) -> None:
        loc_theta = pyro.param(
            "loc_ability",
            torch.zeros(self.num_models, device=self.device),
        )
        scale_theta = pyro.param(
            "scale_ability",
            torch.ones(self.num_models, device=self.device),
            constraint=constraints.positive,
        )
        loc_diff = pyro.param(
            "loc_diff",
            torch.zeros(self.num_items, device=self.device),
        )
        scale_diff = pyro.param(
            "scale_diff",
            torch.ones(self.num_items, device=self.device) * 0.5,
            constraint=constraints.positive,
        )
        loc_slope = pyro.param(
            "loc_slope",
            torch.ones(self.num_items, device=self.device),
            constraint=constraints.positive,
        )
        scale_slope = pyro.param(
            "scale_slope",
            torch.ones(self.num_items, device=self.device) * 0.1,
            constraint=constraints.positive,
        )
        alpha_guess = pyro.param(
            "alpha_guess",
            torch.ones(self.num_items, device=self.device) * 2.0,
            constraint=constraints.positive,
        )
        beta_guess = pyro.param(
            "beta_guess",
            torch.ones(self.num_items, device=self.device) * 8.0,
            constraint=constraints.positive,
        )

        with pyro.plate("thetas", self.num_models, device=self.device):
            pyro.sample("theta", dist.Normal(loc_theta, scale_theta))
        with pyro.plate("items", self.num_items, device=self.device):
            pyro.sample("b", dist.Normal(loc_diff, scale_diff))
            pyro.sample("a", dist.LogNormal(loc_slope, scale_slope))
            pyro.sample("c", dist.Beta(alpha_guess, beta_guess))


MODEL_REGISTRY = {
    "1pl": OneParamLog,
    "2pl": TwoParamLog,
    "3pl": ThreeParamLog,
}


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
        "--model",
        choices=("1pl", "2pl", "3pl"),
        default="1pl",
        help="IRT model variant to fit.",
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
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Optional suffix for output artifact names (defaults to model-specific suffix).",
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
    model_name: str,
    priors: str,
    irt_device: str,
    num_items: int,
    num_models: int,
    verbose: bool,
):
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(
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
        model_name=args.model,
        priors=args.priors,
        irt_device=irt_device,
        num_items=num_items,
        num_models=num_models,
        verbose=args.verbose,
    )

    models, items, responses = prepare_tensors(matrix, torch_device)

    pyro.util.set_rng_seed(args.seed)  # type: ignore[attr-defined]
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

    def extract_param(name: str) -> np.ndarray | None:
        if name not in store:
            return None
        value = store[name]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        raise TypeError(f"Unsupported parameter type for {name}: {type(value)!r}")

    name_map = {
        "loc_ability": "ability_loc",
        "scale_ability": "ability_scale",
        "loc_diff": "diff_loc",
        "scale_diff": "diff_scale",
        "loc_slope": "slope_loc",
        "scale_slope": "slope_scale",
        "alpha_guess": "guess_alpha",
        "beta_guess": "guess_beta",
    }

    params: Dict[str, np.ndarray] = {"losses": np.array(losses, dtype=np.float32)}
    for pyro_name, output_name in name_map.items():
        value = extract_param(pyro_name)
        if value is not None:
            params[output_name] = value

    if "guess_alpha" in params and "guess_beta" in params:
        alpha = params["guess_alpha"]
        beta = params["guess_beta"]
        params["guess_mean"] = alpha / (alpha + beta)

    if "ability_loc" not in params or "diff_loc" not in params:
        available = list(pyro.get_param_store().keys())
        raise KeyError(
            "ability_loc or diff_loc missing from extracted parameters. Available Pyro keys: "
            + ", ".join(available)
        )

    return params


def summarise_results(
    matrix: np.ndarray,
    params: Dict[str, np.ndarray],
    priors: str,
    epochs: int,
    learning_rate: float,
    model: str,
) -> Dict[str, float | int | str]:
    tree_accuracy = matrix.mean(axis=1)
    item_accuracy = matrix.mean(axis=0)

    def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    ability_corr = safe_corr(params["ability_loc"], tree_accuracy)
    difficulty_corr = safe_corr(params["diff_loc"], 1.0 - item_accuracy)

    summary: Dict[str, float | int | str] = {
        "model": model,
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

    if "slope_loc" in params:
        slope = params["slope_loc"]
        summary.update(
            {
                "slope_mean": float(slope.mean()),
                "slope_std": float(slope.std()),
                "slope_min": float(slope.min()),
                "slope_max": float(slope.max()),
            }
        )

    if "guess_mean" in params:
        guess = params["guess_mean"]
        summary.update(
            {
                "guess_mean_mean": float(guess.mean()),
                "guess_mean_std": float(guess.std()),
                "guess_mean_min": float(guess.min()),
                "guess_mean_max": float(guess.max()),
            }
        )
    return summary


def save_outputs(
    output_dir: Path,
    params: Dict[str, np.ndarray],
    summary: Dict[str, float | int | str],
    suffix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in params.items() if k != "losses"}
    np.savez_compressed(output_dir / f"irt_parameters{suffix}.npz", **arrays)  # type: ignore[arg-type]
    np.save(output_dir / f"irt_losses{suffix}.npy", params["losses"])
    with (output_dir / f"irt_summary{suffix}.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    args = parse_args()
    matrix, _, _ = load_response_matrix(args.response_matrix)
    params = fit_irt(matrix, args)
    suffix = args.suffix
    if suffix is None:
        suffix = "" if args.model == "1pl" else f"_{args.model}"
    summary = summarise_results(
        matrix,
        params,
        priors=args.priors,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=args.model,
    )
    save_outputs(args.output_dir, params, summary, suffix)
    print(
        "Training complete. Final loss: "
        f"{summary['final_loss']:.4f}. Summary saved to {args.output_dir} (suffix '{suffix}')."
    )


if __name__ == "__main__":
    main()
