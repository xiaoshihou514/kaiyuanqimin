from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .feature_engineering import TARGET_COLUMN, TARGET_DATE_COLUMN


@dataclass(frozen=True)
class SequenceDatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float
    test_target_dates: pd.Series
    test_target_values: np.ndarray


def _build_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_sequence_datasets(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    sequence_length: int,
    batch_size: int,
    train_end: str,
    val_end: str,
    test_end: str,
) -> SequenceDatasetBundle:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[TARGET_DATE_COLUMN] = pd.to_datetime(
        data[TARGET_DATE_COLUMN], errors="coerce"
    )
    data = data.dropna(subset=["date", TARGET_DATE_COLUMN]).sort_values("date").reset_index(
        drop=True
    )

    train_rows = data[data[TARGET_DATE_COLUMN] <= pd.Timestamp(train_end)].copy()
    if train_rows.empty:
        raise ValueError("Training split is empty after applying target-date boundaries.")

    feature_mean = train_rows[feature_columns].to_numpy(dtype=np.float32).mean(axis=0)
    feature_std = train_rows[feature_columns].to_numpy(dtype=np.float32).std(axis=0)
    feature_std[feature_std == 0] = 1.0

    target_train = train_rows[TARGET_COLUMN].to_numpy(dtype=np.float32)
    target_mean = float(target_train.mean())
    target_std = float(target_train.std() if target_train.std() > 0 else 1.0)

    features = data[feature_columns].to_numpy(dtype=np.float32)
    features = (features - feature_mean) / feature_std
    targets = data[TARGET_COLUMN].to_numpy(dtype=np.float32)
    targets = (targets - target_mean) / target_std
    target_dates = data[TARGET_DATE_COLUMN].reset_index(drop=True)

    xs: list[np.ndarray] = []
    ys: list[float] = []
    sample_target_dates: list[pd.Timestamp] = []
    sample_target_values: list[float] = []
    for idx in range(sequence_length - 1, len(data)):
        xs.append(features[idx - sequence_length + 1 : idx + 1])
        ys.append(float(targets[idx]))
        sample_target_dates.append(pd.Timestamp(target_dates.iloc[idx]))
        sample_target_values.append(float(data[TARGET_COLUMN].iloc[idx]))

    if not xs:
        raise ValueError(
            f"Not enough rows ({len(data)}) for sequence_length={sequence_length}."
        )

    x_all = np.stack(xs).astype(np.float32)
    y_all = np.array(ys, dtype=np.float32)
    sample_dates = pd.Series(sample_target_dates, name=TARGET_DATE_COLUMN)
    sample_values = np.array(sample_target_values, dtype=np.float32)

    train_mask = sample_dates <= pd.Timestamp(train_end)
    val_mask = (sample_dates > pd.Timestamp(train_end)) & (
        sample_dates <= pd.Timestamp(val_end)
    )
    test_mask = (sample_dates > pd.Timestamp(val_end)) & (
        sample_dates <= pd.Timestamp(test_end)
    )

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Sequence split produced empty train/val/test partition.")

    return SequenceDatasetBundle(
        train_loader=_build_loader(x_all[train_mask], y_all[train_mask], batch_size, True),
        val_loader=_build_loader(x_all[val_mask], y_all[val_mask], batch_size, False),
        test_loader=_build_loader(x_all[test_mask], y_all[test_mask], batch_size, False),
        feature_mean=feature_mean,
        feature_std=feature_std,
        target_mean=target_mean,
        target_std=target_std,
        test_target_dates=sample_dates[test_mask].reset_index(drop=True),
        test_target_values=sample_values[test_mask],
    )
