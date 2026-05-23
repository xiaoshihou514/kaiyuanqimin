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

    if "beijing_cucumber_price" in frame.columns:
        frame["beijing_cucumber_price_lag2"] = frame["beijing_cucumber_price"].shift(2)
        frame["sd_beijing_spread"] = (
            frame[LOCAL_PRICE_COLUMN] - frame["beijing_cucumber_price"]
        )
        frame["sd_beijing_spread_lag1"] = frame["sd_beijing_spread"].shift(1)
        frame["sd_beijing_spread_lag2"] = frame["sd_beijing_spread"].shift(2)

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


def lgbm_feature_columns(
    frame: pd.DataFrame,
    *,
    include_beijing: bool = False,
    include_refined_weather: bool = False,
) -> list[str]:
    beijing_features = {
        "beijing_cucumber_price",
        "beijing_cucumber_price_lag1",
        "beijing_cucumber_price_lag2",
        "sd_beijing_spread",
        "sd_beijing_spread_lag1",
        "sd_beijing_spread_lag2",
    }
    refined_weather_features = {
        "temp_avg",
        "precip",
        "temp_lag_1",
        "temp_lag_3",
        "precip_lag_1",
        "precip_lag_3",
        "precip_sum_3d",
        "precip_sum_7d",
        "temp_mean_7d",
        "temp_change_1d",
        "precip_change_1d",
        "extreme_precip_flag",
        "heatwave_3d_flag",
    }
    selected_named_features = set()
    if include_beijing:
        selected_named_features.update(beijing_features)
    if include_refined_weather:
        selected_named_features.update(refined_weather_features)
    return [
        col
        for col in feature_columns(frame)
        if col.startswith("lag_")
        or col.startswith("roll_")
        or col.startswith("price_diff_")
        or col in selected_named_features
    ]


def _climatology_mean(
    frame: pd.DataFrame,
    *,
    anchor_date: pd.Timestamp,
    target_month: int,
    value_column: str,
    lookback_years: int = 5,
) -> float:
    start_date = anchor_date - pd.DateOffset(years=lookback_years)
    mask = (
        (frame["date"] < anchor_date)
        & (frame["date"] >= start_date)
        & (frame["date"].dt.month == target_month)
    )
    values = frame.loc[mask, value_column]
    if values.empty:
        return float("nan")
    return float(values.mean())


def build_long_horizon_feature_table(
    cleaned: pd.DataFrame,
    *,
    horizons: list[int],
    history_window_days: int,
    anchor_step_days: int,
    train_end: str,
    augmentation_enabled: bool,
    augmentation_jitter_days: int,
) -> pd.DataFrame:
    frame = cleaned.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    numeric_columns = [
        column
        for column in frame.columns
        if column != "date" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    frame["month"] = frame["date"].dt.month
    date_to_index = {date_value: idx for idx, date_value in enumerate(frame["date"])}
    train_end_ts = pd.Timestamp(train_end)
    max_horizon = max(horizons)

    base_anchor_indices = list(
        range(history_window_days, len(frame) - max_horizon, anchor_step_days)
    )
    rows: list[dict[str, float | int | str]] = []

    def add_sample(anchor_idx: int, horizon_days: int, *, is_augmented: bool) -> None:
        if anchor_idx < history_window_days:
            return
        target_idx = anchor_idx + horizon_days
        if target_idx >= len(frame):
            return

        anchor_date = pd.Timestamp(frame.at[anchor_idx, "date"])
        target_date = pd.Timestamp(frame.at[target_idx, "date"])
        if is_augmented and target_date > train_end_ts:
            return

        history = frame.iloc[anchor_idx - history_window_days + 1 : anchor_idx + 1]
        if len(history) < history_window_days:
            return

        seasonal_lookup_date = target_date - pd.Timedelta(days=365)
        seasonal_idx = date_to_index.get(seasonal_lookup_date)
        seasonal_naive_price = (
            float(frame.at[seasonal_idx, LOCAL_PRICE_COLUMN])
            if seasonal_idx is not None
            else float(frame.at[anchor_idx, LOCAL_PRICE_COLUMN])
        )

        target_month = int(target_date.month)
        target_week = int(target_date.isocalendar().week)
        quarter = int(target_date.quarter)
        recent_returns_7 = history[LOCAL_PRICE_COLUMN].pct_change().replace(
            [np.inf, -np.inf], np.nan
        )
        recent_returns_7 = recent_returns_7.tail(7).dropna()
        recent_window_7 = history[LOCAL_PRICE_COLUMN].tail(7)
        recent_range_7 = float(recent_window_7.max() - recent_window_7.min())
        recent_return_std_7 = float(recent_returns_7.std(ddof=0)) if not recent_returns_7.empty else 0.0
        recent_trend_7 = float(
            frame.at[anchor_idx, LOCAL_PRICE_COLUMN]
            - frame.at[anchor_idx - 7, LOCAL_PRICE_COLUMN]
        )
        recent_vol_anchor = max(float(history[LOCAL_PRICE_COLUMN].tail(30).std(ddof=0)), 1e-6)

        row: dict[str, float | int | str] = {
            "anchor_date": anchor_date.strftime("%Y-%m-%d"),
            TARGET_DATE_COLUMN: target_date.strftime("%Y-%m-%d"),
            TARGET_COLUMN: float(frame.at[target_idx, LOCAL_PRICE_COLUMN]),
            "horizon_days": int(horizon_days),
            "is_augmented": int(is_augmented),
            "local_price": float(frame.at[anchor_idx, LOCAL_PRICE_COLUMN]),
            "naive_current_price": float(frame.at[anchor_idx, LOCAL_PRICE_COLUMN]),
            "naive_seasonal_price": seasonal_naive_price,
            "prior_year_same_day_price": seasonal_naive_price,
            "selected_baseline": (
                seasonal_naive_price if horizon_days in {30, 90} else float(frame.at[anchor_idx, LOCAL_PRICE_COLUMN])
            ),
            "selected_baseline_name": (
                "seasonal_last_year" if horizon_days in {30, 90} else "current_price"
            ),
            "roll_mean_30": float(history[LOCAL_PRICE_COLUMN].tail(30).mean()),
            "roll_mean_60": float(history[LOCAL_PRICE_COLUMN].tail(60).mean()),
            "roll_mean_90": float(history[LOCAL_PRICE_COLUMN].tail(90).mean()),
            "roll_std_30": float(history[LOCAL_PRICE_COLUMN].tail(30).std(ddof=0)),
            "roll_std_60": float(history[LOCAL_PRICE_COLUMN].tail(60).std(ddof=0)),
            "roll_std_90": float(history[LOCAL_PRICE_COLUMN].tail(90).std(ddof=0)),
            "price_change_7": float(
                frame.at[anchor_idx, LOCAL_PRICE_COLUMN]
                - frame.at[anchor_idx - 7, LOCAL_PRICE_COLUMN]
            ),
            "price_change_30": float(
                frame.at[anchor_idx, LOCAL_PRICE_COLUMN]
                - frame.at[anchor_idx - 30, LOCAL_PRICE_COLUMN]
            ),
            "price_change_90": float(
                frame.at[anchor_idx, LOCAL_PRICE_COLUMN]
                - frame.at[anchor_idx - 90, LOCAL_PRICE_COLUMN]
            ),
            "recent_return_std_7": recent_return_std_7,
            "recent_price_range_7": recent_range_7,
            "rapid_rise_flag_7": float(recent_trend_7 > recent_vol_anchor),
            "rapid_drop_flag_7": float(recent_trend_7 < -recent_vol_anchor),
            "week_sin": float(np.sin(2.0 * np.pi * target_week / 52.0)),
            "week_cos": float(np.cos(2.0 * np.pi * target_week / 52.0)),
            "month_sin": float(np.sin(2.0 * np.pi * target_month / 12.0)),
            "month_cos": float(np.cos(2.0 * np.pi * target_month / 12.0)),
            "quarter": quarter,
            "days_to_next_spring_festival": float(
                frame.at[target_idx, "days_to_next_spring_festival"]
            ),
            "days_to_next_mid_autumn": float(
                frame.at[target_idx, "days_to_next_mid_autumn"]
            ),
            "days_to_next_qingming": float(frame.at[target_idx, "days_to_next_qingming"]),
            "temp_mean_30": float(history["temp_avg"].tail(30).mean()),
            "temp_mean_60": float(history["temp_avg"].tail(60).mean()),
            "temp_mean_90": float(history["temp_avg"].tail(90).mean()),
            "precip_sum_30": float(history["precip"].tail(30).sum()),
            "precip_sum_60": float(history["precip"].tail(60).sum()),
            "precip_sum_90": float(history["precip"].tail(90).sum()),
            "temp_climatology_month_5y": _climatology_mean(
                frame,
                anchor_date=anchor_date,
                target_month=target_month,
                value_column="temp_avg",
            ),
            "precip_climatology_month_5y": _climatology_mean(
                frame,
                anchor_date=anchor_date,
                target_month=target_month,
                value_column="precip",
            ),
        }

        if "beijing_cucumber_price" in frame.columns:
            row["beijing_current_price"] = float(frame.at[anchor_idx, "beijing_cucumber_price"])
            row["beijing_roll_mean_30"] = float(
                history["beijing_cucumber_price"].tail(30).mean()
            )
            row["beijing_change_30"] = float(
                frame.at[anchor_idx, "beijing_cucumber_price"]
                - frame.at[anchor_idx - 30, "beijing_cucumber_price"]
            )
            row["sd_beijing_spread"] = float(
                frame.at[anchor_idx, LOCAL_PRICE_COLUMN]
                - frame.at[anchor_idx, "beijing_cucumber_price"]
            )

        rows.append(row)

    for horizon_days in horizons:
        for anchor_idx in base_anchor_indices:
            add_sample(anchor_idx, horizon_days, is_augmented=False)
            if not augmentation_enabled:
                continue
            for jitter in range(-augmentation_jitter_days, augmentation_jitter_days + 1):
                if jitter == 0:
                    continue
                augmented_idx = anchor_idx + jitter
                if augmented_idx < history_window_days:
                    continue
                if augmented_idx + horizon_days >= len(frame):
                    continue
                if pd.Timestamp(frame.at[augmented_idx + horizon_days, "date"]) > train_end_ts:
                    continue
                add_sample(augmented_idx, horizon_days, is_augmented=True)

    long_frame = pd.DataFrame(rows)
    long_frame[TARGET_DATE_COLUMN] = pd.to_datetime(
        long_frame[TARGET_DATE_COLUMN], errors="coerce"
    )
    long_frame["anchor_date"] = pd.to_datetime(long_frame["anchor_date"], errors="coerce")
    numeric_cols = long_frame.select_dtypes(include=["number"]).columns
    long_frame[numeric_cols] = long_frame[numeric_cols].replace([np.inf, -np.inf], np.nan)
    long_frame = long_frame.dropna().sort_values([TARGET_DATE_COLUMN, "horizon_days"]).reset_index(drop=True)
    return long_frame


def long_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "anchor_date",
        TARGET_DATE_COLUMN,
        TARGET_COLUMN,
        "horizon_days",
        "selected_baseline",
        "selected_baseline_name",
        "naive_current_price",
        "naive_seasonal_price",
        "is_augmented",
    }
    return [column for column in frame.columns if column not in excluded]
