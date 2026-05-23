from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from prophet import Prophet

from .feature_engineering import TARGET_COLUMN, TARGET_DATE_COLUMN
from .metrics import mae, mape, prediction_preview, rmse, smape


@dataclass(frozen=True)
class ProphetResult:
    metrics: dict[str, float | int | str]
    prediction_path: Path


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
    regressors = ["temp_avg", "precip", "sentiment_score"]
    fit_df = train_df[[TARGET_DATE_COLUMN, TARGET_COLUMN, *regressors]].copy()
    fit_df = fit_df.rename(columns={TARGET_DATE_COLUMN: "ds", TARGET_COLUMN: "y"})
    fit_df["ds"] = pd.to_datetime(fit_df["ds"])

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
    )
    for reg in regressors:
        model.add_regressor(reg)
    model.fit(fit_df)

    future_df = test_df[[TARGET_DATE_COLUMN, *regressors]].copy()
    future_df = future_df.rename(columns={TARGET_DATE_COLUMN: "ds"})
    future_df["ds"] = pd.to_datetime(future_df["ds"])
    forecast = model.predict(future_df)

    y_true = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = forecast["yhat"].to_numpy(dtype=float)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    with model_output_path.open("wb") as f:
        pickle.dump(model, f)

    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_output_dir / "prophet_predictions.csv"
    pd.DataFrame(
        {
            "date": future_df["ds"].dt.strftime("%Y-%m-%d"),
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
        "prediction_preview": prediction_preview(
            future_df["ds"].dt.strftime("%Y-%m-%d"), y_true, y_pred
        ),
    }
    metrics_path = model_output_path.with_name("metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ProphetResult(metrics=metrics, prediction_path=prediction_path)
