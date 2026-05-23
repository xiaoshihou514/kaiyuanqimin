from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(
        np.abs(y_true) + np.abs(y_pred) < 1e-8, 1e-8, np.abs(y_true) + np.abs(y_pred)
    )
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def interval_mean_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


def prediction_preview(
    dates: pd.Series | np.ndarray | list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    limit: int = 5,
) -> list[dict[str, float | str]]:
    date_values = list(dates)[:limit]
    rows = []
    for idx, date_value in enumerate(date_values):
        rows.append(
            {
                "date": str(date_value),
                "y_true": float(y_true[idx]),
                "y_pred": float(y_pred[idx]),
            }
        )
    return rows


def baseline_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "test_mae": mae(y_true, y_pred),
        "test_rmse": rmse(y_true, y_pred),
        "test_mape": mape(y_true, y_pred),
        "test_smape": smape(y_true, y_pred),
    }
