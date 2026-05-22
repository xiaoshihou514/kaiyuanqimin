from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "target"
BASE_COLUMNS = ["local_price", "temp_avg", "precip", "sentiment_score"]


@dataclass(frozen=True)
class FeatureSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_feature_table(cleaned: pd.DataFrame) -> pd.DataFrame:
    frame = cleaned.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    frame[TARGET_COLUMN] = frame["local_price"].astype(float)

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


def split_by_time(frame: pd.DataFrame, *, train_end: str, val_end: str, test_end: str) -> FeatureSplits:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    train = data[data["date"] <= pd.Timestamp(train_end)].copy()
    val = data[(data["date"] > pd.Timestamp(train_end)) & (data["date"] <= pd.Timestamp(val_end))].copy()
    test = data[(data["date"] > pd.Timestamp(val_end)) & (data["date"] <= pd.Timestamp(test_end))].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Feature split produced empty train/val/test partition.")
    return FeatureSplits(train=train, val=val, test=test)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {"date", TARGET_COLUMN}]
