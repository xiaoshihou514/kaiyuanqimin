from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .dataset import DatasetBundle
from .model import LSTMAttentionRegressor


@dataclass(frozen=True)
class TrainParams:
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    model_output_path: Path
    metrics_output_path: Path


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_epoch(
    model: LSTMAttentionRegressor,
    loader: torch.utils.data.DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
    desc: str,
) -> float:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    batch_count = 0
    iterator = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train_mode):
            pred, _ = model(x)
            loss = loss_fn(pred, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        batch_count += 1
        iterator.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")

    if batch_count == 0:
        raise ValueError("Empty dataloader encountered during training/evaluation.")
    return total_loss / batch_count


def _collect_predictions(
    model: LSTMAttentionRegressor,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred, _ = model(x)
            preds.append(pred.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(targets)


def train_and_evaluate(
    bundle: DatasetBundle, params: TrainParams
) -> dict[str, float | int | str]:
    _seed_everything(params.seed)
    device = _resolve_device(params.device)
    model = LSTMAttentionRegressor(
        input_dim=params.input_dim,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        dropout=params.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    epochs_done = 0

    for epoch in tqdm(
        range(1, params.epochs + 1),
        desc="Training epochs",
        unit="epoch",
        dynamic_ncols=True,
    ):
        epochs_done = epoch
        train_loss = _run_epoch(
            model,
            bundle.train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            desc=f"train e{epoch}",
        )
        val_loss = _run_epoch(
            model,
            bundle.val_loader,
            optimizer=None,
            loss_fn=loss_fn,
            device=device,
            desc=f"val e{epoch}",
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= params.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.to(device)

    test_pred_scaled, test_target_scaled = _collect_predictions(
        model, bundle.test_loader, device=device
    )
    test_pred = bundle.target_standardizer.inverse_transform_column(test_pred_scaled, 0)
    test_target = bundle.target_standardizer.inverse_transform_column(
        test_target_scaled, 0
    )

    mae = float(np.mean(np.abs(test_pred - test_target)))
    rmse = float(np.sqrt(np.mean((test_pred - test_target) ** 2)))

    params.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_columns": bundle.feature_columns,
            "target_mean": bundle.target_standardizer.mean.tolist(),
            "target_std": bundle.target_standardizer.std.tolist(),
            "model_config": {
                "input_dim": params.input_dim,
                "hidden_dim": params.hidden_dim,
                "num_layers": params.num_layers,
                "dropout": params.dropout,
            },
        },
        params.model_output_path,
    )

    metrics: dict[str, float | int | str] = {
        "device": str(device),
        "epochs_trained": epochs_done,
        "best_val_mse": float(best_val_loss),
        "test_mae": mae,
        "test_rmse": rmse,
    }
    params.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    params.metrics_output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics
