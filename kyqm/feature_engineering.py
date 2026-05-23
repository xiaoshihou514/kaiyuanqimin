from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TARGET_COLUMN = "target"
TARGET_DATE_COLUMN = "target_date"
LOCAL_PRICE_COLUMN = "local_price"
CORE_RECURRENT_COLUMNS = ["local_price", "temp_avg", "precip"]


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

    frame[TARGET_COLUMN] = frame[LOCAL_PRICE_COLUMN].shift(-forecast_horizon).astype(float)
    frame[TARGET_DATE_COLUMN] = frame["date"].shift(-forecast_horizon)

    for lag in (1, 2, 3, 7, 14, 30):
        frame[f"lag_{lag}"] = frame[LOCAL_PRICE_COLUMN].shift(lag)

    for window in (7, 14):
        rolling = frame[LOCAL_PRICE_COLUMN].rolling(window=window, min_periods=window)
        frame[f"roll_mean_{window}"] = rolling.mean()
        frame[f"roll_std_{window}"] = rolling.std(ddof=0)
        frame[f"roll_max_{window}"] = rolling.max()
        frame[f"roll_min_{window}"] = rolling.min()

    frame["price_diff_1"] = frame[LOCAL_PRICE_COLUMN] - frame["lag_1"]
    frame["price_diff_7"] = frame[LOCAL_PRICE_COLUMN] - frame["lag_7"]

    for column in frame.columns:
        if column.endswith("_price") and column != LOCAL_PRICE_COLUMN:
            frame[f"{column}_lag1"] = frame[column].shift(1)
            frame[f"{column}_lag7"] = frame[column].shift(7)

    for column in frame.columns:
        if column.endswith("_cucumber_price"):
            frame[f"{column}_lag1"] = frame[column].shift(1)
            if f"{column}_lag7" in frame.columns:
                frame = frame.drop(columns=[f"{column}_lag7"])

    frame["temp_lag_1"] = frame["temp_avg"].shift(1)
    frame["temp_lag_3"] = frame["temp_avg"].shift(3)
    frame["precip_lag_1"] = frame["precip"].shift(1)
    frame["precip_lag_3"] = frame["precip"].shift(3)

    frame["weekday"] = frame[TARGET_DATE_COLUMN].dt.weekday
    frame["month"] = frame[TARGET_DATE_COLUMN].dt.month
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    day_of_year = frame[TARGET_DATE_COLUMN].dt.dayofyear.astype(float)
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
    excluded_raw_cross_series = {
        col
        for col in frame.columns
        if col.endswith("_price") and col != LOCAL_PRICE_COLUMN
    }
    return [
        col
        for col in frame.columns
        if col
        not in {"date", TARGET_COLUMN, TARGET_DATE_COLUMN, *excluded_raw_cross_series}
    ]


def lgbm_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in feature_columns(frame)
        if col.startswith("lag_")
        or col.startswith("roll_")
        or col.startswith("price_diff_")
    ]
