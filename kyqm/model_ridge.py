from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import TARGET_COLUMN, TARGET_DATE_COLUMN
from .metrics import mae, mape, prediction_preview, rmse, smape


@dataclass(frozen=True)
class RidgeResult:
    metrics: dict[str, float | int | str]
    prediction_path: Path


def train_ridge_model(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_output_dir: Path,
    prediction_output_dir: Path,
    alpha: float = 10.0,
    baseline_column: str | None = None,
    model_name: str = "ridge",
    prediction_filename: str = "ridge_predictions.csv",
) -> RidgeResult:
    model_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    x_train = train_df[feature_columns].to_numpy(dtype=float)
    x_val = val_df[feature_columns].to_numpy(dtype=float)
    x_test = test_df[feature_columns].to_numpy(dtype=float)

    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    train_baseline = (
        train_df[baseline_column].to_numpy(dtype=float)
        if baseline_column is not None
        else np.zeros(len(train_df), dtype=float)
    )
    val_baseline = (
        val_df[baseline_column].to_numpy(dtype=float)
        if baseline_column is not None
        else np.zeros(len(val_df), dtype=float)
    )
    test_baseline = (
        test_df[baseline_column].to_numpy(dtype=float)
        if baseline_column is not None
        else np.zeros(len(test_df), dtype=float)
    )

    fit_target = y_train - train_baseline if baseline_column is not None else y_train

    model = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )
    model.fit(x_train, fit_target)

    pred_val = model.predict(x_val) + val_baseline
    pred_test = model.predict(x_test) + test_baseline

    with (model_output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)

    val_prediction_dates = val_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d")
    test_prediction_dates = test_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d")
    prediction_path = prediction_output_dir / prediction_filename
    pd.concat(
        [
            pd.DataFrame(
                {
                    "date": val_prediction_dates,
                    "split": "val",
                    "y_true": y_val,
                    "y_pred": pred_val,
                }
            ),
            pd.DataFrame(
                {
                    "date": test_prediction_dates,
                    "split": "test",
                    "y_true": y_test,
                    "y_pred": pred_test,
                }
            ),
        ],
        ignore_index=True,
    ).to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "alpha": alpha,
        "val_mae": mae(y_val, pred_val),
        "test_mae": mae(y_test, pred_test),
        "test_rmse": rmse(y_test, pred_test),
        "test_mape": mape(y_test, pred_test),
        "test_smape": smape(y_test, pred_test),
        "prediction_preview": prediction_preview(
            test_prediction_dates, y_test, pred_test
        ),
    }
    (model_output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return RidgeResult(metrics=metrics, prediction_path=prediction_path)
