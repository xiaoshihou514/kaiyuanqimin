from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "target"
TARGET_DATE_COLUMN = "target_date"
BASE_COLUMNS = ["local_price", "temp_avg", "precip", "sentiment_score"]


@dataclass(frozen=True)
class FeatureSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_feature_table(cleaned: pd.DataFrame, *, forecast_horizon: int = 1) -> pd.DataFrame:
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1.")

    frame = cleaned.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    frame[TARGET_COLUMN] = frame["local_price"].shift(-forecast_horizon).astype(float)
    frame[TARGET_DATE_COLUMN] = frame["date"].shift(-forecast_horizon)

    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"lag_{lag}"] = frame["local_price"].shift(lag)

    for window in (7, 14):
        rolling = frame["local_price"].rolling(window=window)
        frame[f"roll_mean_{window}"] = rolling.mean()
        frame[f"roll_std_{window}"] = rolling.std(ddof=0)
        frame[f"roll_max_{window}"] = rolling.max()
        frame[f"roll_min_{window}"] = rolling.min()

    frame["price_diff_1"] = frame["local_price"] - frame["lag_1"]
    frame["price_diff_7"] = frame["local_price"] - frame["lag_7"]

    for lag in (1, 3, 7):
        frame[f"temp_lag_{lag}"] = frame["temp_avg"].shift(lag)
        frame[f"precip_lag_{lag}"] = frame["precip"].shift(lag)
    frame["precip_sum_3"] = frame["precip"].rolling(window=3).sum()
    frame["precip_sum_7"] = frame["precip"].rolling(window=7).sum()
    frame["temp_mean_7"] = frame["temp_avg"].rolling(window=7).mean()

    frame["weekday"] = frame["date"].dt.weekday
    frame["month"] = frame["date"].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    day_of_year = frame["date"].dt.dayofyear.astype(float)
    frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.0)
    frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.0)

    numeric_cols = frame.select_dtypes(include=["number"]).columns
    frame[numeric_cols] = frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna().reset_index(drop=True)
    return frame


def split_by_time(
    frame: pd.DataFrame, *, train_end: str, val_end: str, test_end: str
) -> FeatureSplits:
    data = frame.copy()
    data[TARGET_DATE_COLUMN] = pd.to_datetime(data[TARGET_DATE_COLUMN], errors="coerce")
    train = data[data[TARGET_DATE_COLUMN] <= pd.Timestamp(train_end)].copy()
    val = data[
        (data[TARGET_DATE_COLUMN] > pd.Timestamp(train_end))
        & (data[TARGET_DATE_COLUMN] <= pd.Timestamp(val_end))
    ].copy()
    test = data[
        (data[TARGET_DATE_COLUMN] > pd.Timestamp(val_end))
        & (data[TARGET_DATE_COLUMN] <= pd.Timestamp(test_end))
    ].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Feature split produced empty train/val/test partition.")
    return FeatureSplits(train=train, val=val, test=test)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in {"date", TARGET_COLUMN, TARGET_DATE_COLUMN}
    ]
