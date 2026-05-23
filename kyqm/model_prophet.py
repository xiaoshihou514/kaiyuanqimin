from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from prophet import Prophet

from .feature_engineering import LOCAL_PRICE_COLUMN, TARGET_COLUMN, TARGET_DATE_COLUMN
from .metrics import mae, mape, prediction_preview, rmse, smape


@dataclass(frozen=True)
class ProphetResult:
    metrics: dict[str, float | int | str]
    prediction_path: Path


def add_prophet_seasonal_features(
    frame: pd.DataFrame,
    *,
    train_end: str,
    weekly_seasonality: bool,
    yearly_seasonality: bool,
) -> pd.DataFrame:
    prophet_train = (
        frame.loc[frame[TARGET_DATE_COLUMN] <= pd.Timestamp(train_end), ["date", LOCAL_PRICE_COLUMN]]
        .rename(columns={"date": "ds", LOCAL_PRICE_COLUMN: "y"})
        .copy()
    )
    prophet_train["ds"] = pd.to_datetime(prophet_train["ds"])
    if prophet_train.empty:
        raise ValueError("Cannot fit Prophet seasonal features on an empty training frame.")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    model.fit(prophet_train)

    seasonal_df = frame[[TARGET_DATE_COLUMN]].rename(columns={TARGET_DATE_COLUMN: "ds"}).copy()
    seasonal_df["ds"] = pd.to_datetime(seasonal_df["ds"])
    forecast = model.predict(seasonal_df)

    augmented = frame.copy()
    augmented["prophet_weekly"] = forecast["weekly"].to_numpy(dtype=float)
    augmented["prophet_yearly"] = forecast["yearly"].to_numpy(dtype=float)
    return augmented


def train_prophet_model(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_output_path: Path,
    prediction_output_dir: Path,
    daily_seasonality: bool,
    weekly_seasonality: bool,
    yearly_seasonality: bool,
) -> ProphetResult:
    fit_df = train_df[["date", TARGET_COLUMN]].copy()
    fit_df = fit_df.rename(columns={"date": "ds", TARGET_COLUMN: "y"})
    fit_df["ds"] = pd.to_datetime(fit_df["ds"])

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    model.fit(fit_df)

    future_df = test_df[["date"]].copy()
    future_df = future_df.rename(columns={"date": "ds"})
    future_df["ds"] = pd.to_datetime(future_df["ds"])
    forecast = model.predict(future_df)

    y_true = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = forecast["yhat"].to_numpy(dtype=float)
    prediction_dates = pd.to_datetime(test_df[TARGET_DATE_COLUMN]).dt.strftime("%Y-%m-%d")

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_output_path.open("wb") as f:
        pickle.dump(model, f)

    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_output_dir / "prophet_predictions.csv"
    pd.DataFrame(
        {
            "date": prediction_dates,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": "prophet",
        "test_mae": mae(y_true, y_pred),
        "test_rmse": rmse(y_true, y_pred),
        "test_mape": mape(y_true, y_pred),
        "test_smape": smape(y_true, y_pred),
        "prediction_preview": prediction_preview(prediction_dates, y_true, y_pred),
    }
    metrics_path = model_output_path.with_name("metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ProphetResult(metrics=metrics, prediction_path=prediction_path)
