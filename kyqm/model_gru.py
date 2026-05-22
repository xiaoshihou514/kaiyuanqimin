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

from .feature_engineering import TARGET_COLUMN
from .metrics import interval_mean_width, mae, mape, picp, rmse, smape


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, outputs: torch.Tensor, last_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.energy(
            torch.tanh(self.query(last_hidden).unsqueeze(1) + self.key(outputs))
        ).squeeze(-1)
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)
        return context, weights


class GruAttentionRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.attention = AdditiveAttention(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.gru(x)
        last_hidden = hidden[-1]
        context, weights = self.attention(outputs, last_hidden)
        pred = self.head(torch.cat([context, last_hidden], dim=1))
        return pred, weights


@dataclass(frozen=True)
class GruResult:
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


def _pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantile: float
) -> torch.Tensor:
    diff = target - pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1.0) * diff))


def _run_epoch(
    model: GruAttentionRegressor,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    quantiles_enabled: bool,
    grad_clip_norm: float,
    device: torch.device,
    desc: str,
) -> float:
    model.train(optimizer is not None)
    total_loss = 0.0
    total_batches = 0
    iterator = tqdm(loader, desc=desc, leave=False, unit="batch", dynamic_ncols=True)
    huber = nn.HuberLoss(delta=1.0)
    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            pred, _ = model(x)
            if quantiles_enabled:
                p10, p50, p90 = pred[:, 0], pred[:, 1], pred[:, 2]
                loss = (
                    _pinball_loss(p10, y, 0.1)
                    + _pinball_loss(p50, y, 0.5)
                    + _pinball_loss(p90, y, 0.9)
                ) / 3.0
            else:
                loss = huber(pred[:, 0], y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        total_batches += 1
        iterator.set_postfix(loss=f"{float(loss.detach().cpu().item()):.4f}")
    if total_batches == 0:
        raise ValueError("Empty dataloader for GRU training.")
    return total_loss / total_batches


def train_gru_model(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
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
    quantiles_enabled: bool,
    model_output_path: Path,
    metrics_output_path: Path,
    prediction_output_dir: Path,
    seed: int,
    device: str,
) -> GruResult:
    _seed_everything(seed)
    resolved_device = _resolve_device(device)

    train_values = train_df[feature_columns].to_numpy(dtype=np.float32)
    val_values = val_df[feature_columns].to_numpy(dtype=np.float32)
    test_values = test_df[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    x_train_scaled, mean, std = _standardize(train_values, train_values)
    x_val_scaled = (val_values - mean) / std
    x_test_scaled = (test_values - mean) / std
    target_mean = float(y_train.mean())
    target_std = float(y_train.std() if y_train.std() > 0 else 1.0)
    y_train_scaled = (y_train - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std
    y_test_scaled = (y_test - target_mean) / target_std

    train_x, train_y = _make_sequences(x_train_scaled, y_train_scaled, sequence_length)
    val_x, val_y = _make_sequences(x_val_scaled, y_val_scaled, sequence_length)
    test_x, test_y = _make_sequences(x_test_scaled, y_test_scaled, sequence_length)

    train_loader = _to_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
    val_loader = _to_loader(val_x, val_y, batch_size=batch_size, shuffle=False)
    test_loader = _to_loader(test_x, test_y, batch_size=batch_size, shuffle=False)

    output_dim = 3 if quantiles_enabled else 1
    model = GruAttentionRegressor(
        input_dim=len(feature_columns),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        output_dim=output_dim,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    for epoch in tqdm(
        range(1, epochs + 1), desc="GRU epochs", unit="epoch", dynamic_ncols=True
    ):
        _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            quantiles_enabled=quantiles_enabled,
            grad_clip_norm=grad_clip_norm,
            device=resolved_device,
            desc=f"gru train e{epoch}",
        )
        val_loss = _run_epoch(
            model,
            val_loader,
            optimizer=None,
            quantiles_enabled=quantiles_enabled,
            grad_clip_norm=grad_clip_norm,
            device=resolved_device,
            desc=f"gru val e{epoch}",
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
        raise RuntimeError("GRU training failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()

    preds = []
    attn_rows = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(resolved_device)
            pred, attn = model(x)
            preds.append(pred.cpu().numpy())
            attn_rows.append(attn.cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0)
    y_true = (test_y * target_std) + target_mean

    if quantiles_enabled:
        p10 = pred_arr[:, 0] * target_std + target_mean
        p50 = pred_arr[:, 1] * target_std + target_mean
        p90 = pred_arr[:, 2] * target_std + target_mean
        y_pred = p50
    else:
        y_pred = pred_arr[:, 0] * target_std + target_mean
        p10 = np.full_like(y_pred, np.nan)
        p90 = np.full_like(y_pred, np.nan)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_columns": feature_columns,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "target_mean": target_mean,
            "target_std": target_std,
            "quantiles_enabled": quantiles_enabled,
        },
        model_output_path,
    )
    attn_output_path = model_output_path.with_name("attention_weights.npy")
    np.save(attn_output_path, np.concatenate(attn_rows, axis=0))

    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_output_dir / "gru_predictions.csv"
    pd.DataFrame(
        {
            "date": test_df["date"]
            .iloc[sequence_length:]
            .dt.strftime("%Y-%m-%d")
            .reset_index(drop=True),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_p10": p10,
            "y_pred_p90": p90,
        }
    ).to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": "gru_attention",
        "test_mae": mae(y_true, y_pred),
        "test_rmse": rmse(y_true, y_pred),
        "test_mape": mape(y_true, y_pred),
        "test_smape": smape(y_true, y_pred),
        "best_val_loss": float(best_val),
        "attention_path": str(attn_output_path),
    }
    if quantiles_enabled:
        metrics["test_picp"] = picp(y_true, p10, p90)
        metrics["test_interval_width"] = interval_mean_width(p10, p90)

    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return GruResult(metrics=metrics, prediction_path=prediction_path)
