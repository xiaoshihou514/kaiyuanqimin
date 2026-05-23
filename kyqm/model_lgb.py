from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .feature_engineering import TARGET_COLUMN, TARGET_DATE_COLUMN
from .metrics import interval_mean_width, mae, mape, picp, prediction_preview, rmse, smape


@dataclass(frozen=True)
class LgbmResult:
    metrics: dict[str, float | int | str]
    prediction_path: Path


def _build_params(
    *,
    objective: str,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
    min_data_in_leaf: int,
    lambda_l1: float,
    lambda_l2: float,
    alpha: float | None = None,
) -> dict[str, float | int | str]:
    params: dict[str, float | int | str] = {
        "objective": objective,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    return params


def train_lightgbm_models(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_output_dir: Path,
    prediction_output_dir: Path,
    quantiles_enabled: bool,
    lower_alpha: float,
    upper_alpha: float,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
    min_data_in_leaf: int,
    lambda_l1: float,
    lambda_l2: float,
    early_stopping_rounds: int,
) -> LgbmResult:
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model_output_dir.mkdir(parents=True, exist_ok=True)
    prediction_output_dir.mkdir(parents=True, exist_ok=True)

    point_model = lgb.LGBMRegressor(
        **_build_params(
            objective="regression",
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
        )
    )
    point_model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    pred_point = point_model.predict(x_test)

    pred_lower = np.full_like(pred_point, np.nan, dtype=float)
    pred_upper = np.full_like(pred_point, np.nan, dtype=float)
    lower_model = None
    upper_model = None

    if quantiles_enabled:
        lower_model = lgb.LGBMRegressor(
            **_build_params(
                objective="quantile",
                alpha=lower_alpha,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_data_in_leaf=min_data_in_leaf,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
            )
        )
        upper_model = lgb.LGBMRegressor(
            **_build_params(
                objective="quantile",
                alpha=upper_alpha,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_data_in_leaf=min_data_in_leaf,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
            )
        )
        lower_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        upper_model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        pred_lower = lower_model.predict(x_test)
        pred_upper = upper_model.predict(x_test)

    with (model_output_dir / "point_model.pkl").open("wb") as f:
        pickle.dump(point_model, f)
    if lower_model is not None:
        with (model_output_dir / "quantile_p10.pkl").open("wb") as f:
            pickle.dump(lower_model, f)
    if upper_model is not None:
        with (model_output_dir / "quantile_p90.pkl").open("wb") as f:
            pickle.dump(upper_model, f)

    pred_frame = pd.DataFrame(
        {
            "date": test_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d"),
            "y_true": y_test,
            "y_pred": pred_point,
            "y_pred_p10": pred_lower,
            "y_pred_p90": pred_upper,
        }
    )
    prediction_path = prediction_output_dir / "lgbm_predictions.csv"
    pred_frame.to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": "lightgbm",
        "test_mae": mae(y_test, pred_point),
        "test_rmse": rmse(y_test, pred_point),
        "test_mape": mape(y_test, pred_point),
        "test_smape": smape(y_test, pred_point),
        "prediction_preview": prediction_preview(
            test_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d"), y_test, pred_point
        ),
    }
    if quantiles_enabled:
        metrics["test_picp"] = picp(y_test, pred_lower, pred_upper)
        metrics["test_interval_width"] = interval_mean_width(pred_lower, pred_upper)

    (model_output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return LgbmResult(metrics=metrics, prediction_path=prediction_path)
