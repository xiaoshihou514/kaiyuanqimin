from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .dataset import build_sequence_datasets
from .metrics import mae, mape, prediction_preview, rmse, smape
from .model import LSTMAttentionRegressor


@dataclass(frozen=True)
class LstmResult:
    metrics: dict[str, float | int | str]
    prediction_path: Path


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


def _standardize(
    train_values: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    return (values - mean) / std, mean, std


def _make_sequences(
    features: np.ndarray, target: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= seq_len:
        raise ValueError(
            f"Not enough rows ({len(features)}) for sequence length {seq_len}."
        )
    xs, ys = [], []
    for idx in range(seq_len, len(features)):
        xs.append(features[idx - seq_len : idx])
        ys.append(target[idx])
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)


def _to_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _run_epoch(
    model: LSTMAttentionRegressor,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    grad_clip_norm: float,
    device: torch.device,
    desc: str,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    loss_fn = nn.HuberLoss(delta=1.0)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred, _ = model(x)
            loss = loss_fn(pred[:, 0], y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        iterator.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")
    if total_batches == 0:
        raise ValueError("Empty dataloader for LSTM training.")
    return total_loss / total_batches


def train_lstm_model(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    patience: int,
    weight_decay: float,
    grad_clip_norm: float,
    model_output_path: Path,
    metrics_output_path: Path,
    prediction_output_dir: Path,
    seed: int,
    device: str,
    train_end: str,
    val_end: str,
    test_end: str,
) -> LstmResult:
    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    bundle = build_sequence_datasets(
        frame,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )

    model = LSTMAttentionRegressor(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    for epoch in tqdm(
        range(1, epochs + 1), desc="LSTM epochs", unit="epoch", dynamic_ncols=True
    ):
        _run_epoch(
            model,
            bundle.train_loader,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
            device=resolved_device,
            desc=f"lstm train e{epoch}",
        )
        val_loss = _run_epoch(
            model,
            bundle.val_loader,
            optimizer=None,
            grad_clip_norm=grad_clip_norm,
            device=resolved_device,
            desc=f"lstm val e{epoch}",
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    preds = []
    attn_rows = []
    with torch.no_grad():
        for x, _ in bundle.test_loader:
            x = x.to(resolved_device)
            pred, attn = model(x)
            preds.append(pred.cpu().numpy())
            attn_rows.append(attn.cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0)
    y_true = bundle.test_target_values.astype(float)
    y_pred = pred_arr[:, 0] * bundle.target_std + bundle.target_mean

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_columns": feature_columns,
            "feature_mean": bundle.feature_mean.tolist(),
            "feature_std": bundle.feature_std.tolist(),
            "target_mean": bundle.target_mean,
            "target_std": bundle.target_std,
            "sequence_length": sequence_length,
            "model_config": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
        model_output_path,
    )
    attn_output_path = model_output_path.with_name("attention_weights.npy")
    np.save(attn_output_path, np.concatenate(attn_rows, axis=0))

    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_output_dir / "lstm_predictions.csv"
    pd.DataFrame(
        {
            "date": bundle.test_target_dates.dt.strftime("%Y-%m-%d"),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": "lstm_attention",
        "test_mae": mae(y_true, y_pred),
        "test_rmse": rmse(y_true, y_pred),
        "test_mape": mape(y_true, y_pred),
        "test_smape": smape(y_true, y_pred),
        "best_val_loss": float(best_val),
        "attention_path": str(attn_output_path),
        "prediction_preview": prediction_preview(
            bundle.test_target_dates.dt.strftime("%Y-%m-%d"), y_true, y_pred
        ),
    }
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return LstmResult(metrics=metrics, prediction_path=prediction_path)
