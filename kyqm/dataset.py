from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .prepare import FEATURE_COLUMNS


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform_column(self, data: np.ndarray, index: int) -> np.ndarray:
        return data * self.std[index] + self.mean[index]


@dataclass(frozen=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_standardizer: Standardizer
    target_standardizer: Standardizer
    feature_columns: list[str]


def _make_windows(
    features: np.ndarray, targets: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= sequence_length:
        raise ValueError(
            f"Not enough rows ({len(features)}) for sequence_length={sequence_length}. "
            "Collect more data or reduce sequence length."
        )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for idx in range(sequence_length, len(features)):
        xs.append(features[idx - sequence_length : idx])
        ys.append(np.array([targets[idx]], dtype=np.float32))
    return np.stack(xs), np.stack(ys)


def _build_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_datasets(
    frame: pd.DataFrame,
    *,
    sequence_length: int,
    batch_size: int,
    train_end: str,
    val_end: str,
    test_end: str,
) -> DatasetBundle:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            raise ValueError(f"Missing required feature column: {column}")

    train = data[data["date"] <= pd.Timestamp(train_end)].copy()
    val = data[
        (data["date"] > pd.Timestamp(train_end))
        & (data["date"] <= pd.Timestamp(val_end))
    ].copy()
    test = data[
        (data["date"] > pd.Timestamp(val_end))
        & (data["date"] <= pd.Timestamp(test_end))
    ].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Time split produced empty train/val/test partition.")

    train_features = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    feature_standardizer = Standardizer(mean=feature_mean, std=feature_std)

    target_mean = np.array([feature_mean[0]], dtype=np.float32)
    target_std = np.array([feature_std[0]], dtype=np.float32)
    target_standardizer = Standardizer(mean=target_mean, std=target_std)

    def transform_split(split: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        features = split[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        scaled_features = feature_standardizer.transform(features)
        targets = scaled_features[:, 0]
        return _make_windows(scaled_features, targets, sequence_length=sequence_length)

    train_x, train_y = transform_split(train)
    val_x, val_y = transform_split(val)
    test_x, test_y = transform_split(test)

    return DatasetBundle(
        train_loader=_build_loader(
            train_x, train_y, batch_size=batch_size, shuffle=True
        ),
        val_loader=_build_loader(val_x, val_y, batch_size=batch_size, shuffle=False),
        test_loader=_build_loader(test_x, test_y, batch_size=batch_size, shuffle=False),
        feature_standardizer=feature_standardizer,
        target_standardizer=target_standardizer,
        feature_columns=list(FEATURE_COLUMNS),
    )
