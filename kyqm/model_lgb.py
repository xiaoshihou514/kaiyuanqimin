from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

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


def _time_series_cv_mae(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    baseline_column: str | None,
    learning_rate: float,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
    min_data_in_leaf: int,
    lambda_l1: float,
    lambda_l2: float,
    cv_splits: int,
) -> tuple[list[float], float, float]:
    ordered = frame.sort_values(TARGET_DATE_COLUMN).reset_index(drop=True)
    split_count = min(cv_splits, len(ordered) - 1)
    if split_count < 2:
        return [], float("nan"), float("nan")

    splitter = TimeSeriesSplit(n_splits=split_count)
    fold_mae: list[float] = []
    x_all = ordered[feature_columns]
    y_all = ordered[TARGET_COLUMN].to_numpy(dtype=float)
    baseline_all = (
        ordered[baseline_column].to_numpy(dtype=float)
        if baseline_column is not None
        else np.zeros(len(ordered), dtype=float)
    )
    train_target = y_all - baseline_all if baseline_column is not None else y_all
    for train_idx, val_idx in splitter.split(x_all):
        model = lgb.LGBMRegressor(
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
        model.fit(x_all.iloc[train_idx], train_target[train_idx])
        fold_pred = model.predict(x_all.iloc[val_idx]) + baseline_all[val_idx]
        fold_mae.append(mae(y_all[val_idx], fold_pred))
    return fold_mae, float(np.mean(fold_mae)), float(np.std(fold_mae))


def _write_feature_importance(
    model: lgb.LGBMRegressor, feature_columns: list[str], output_dir: Path
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": model.booster_.feature_importance(importance_type="gain"),
            "split_importance": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain_importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)


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
    cv_splits: int,
    baseline_column: str | None = "local_price",
    model_name: str = "lightgbm",
    prediction_filename: str = "lgbm_predictions.csv",
) -> LgbmResult:
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    x_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    x_test = test_df[feature_columns]
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
    y_train_fit = y_train - train_baseline if baseline_column is not None else y_train
    y_val_fit = y_val - val_baseline if baseline_column is not None else y_val

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
        y_train_fit,
        eval_set=[(x_val, y_val_fit)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    pred_point = point_model.predict(x_test) + test_baseline

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
            y_train_fit,
            eval_set=[(x_val, y_val_fit)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        upper_model.fit(
            x_train,
            y_train_fit,
            eval_set=[(x_val, y_val_fit)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        pred_lower = lower_model.predict(x_test) + test_baseline
        pred_upper = upper_model.predict(x_test) + test_baseline

    with (model_output_dir / "point_model.pkl").open("wb") as f:
        pickle.dump(point_model, f)
    if lower_model is not None:
        with (model_output_dir / "quantile_p10.pkl").open("wb") as f:
            pickle.dump(lower_model, f)
    if upper_model is not None:
        with (model_output_dir / "quantile_p90.pkl").open("wb") as f:
            pickle.dump(upper_model, f)

    _write_feature_importance(point_model, feature_columns, model_output_dir)
    cv_frame = pd.concat([train_df, val_df], ignore_index=True)
    fold_mae, cv_mae_mean, cv_mae_std = _time_series_cv_mae(
        cv_frame,
        feature_columns=feature_columns,
        baseline_column=baseline_column,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        cv_splits=cv_splits,
    )

    prediction_dates = test_df[TARGET_DATE_COLUMN].dt.strftime("%Y-%m-%d")
    pred_frame = pd.DataFrame(
        {
            "date": prediction_dates,
            "y_true": y_test,
            "y_pred": pred_point,
            "y_pred_p10": pred_lower,
            "y_pred_p90": pred_upper,
        }
    )
    prediction_path = prediction_output_dir / prediction_filename
    pred_frame.to_csv(prediction_path, index=False)

    metrics: dict[str, float | int | str] = {
        "model": model_name,
        "test_mae": mae(y_test, pred_point),
        "test_rmse": rmse(y_test, pred_point),
        "test_mape": mape(y_test, pred_point),
        "test_smape": smape(y_test, pred_point),
        "cv_mae_mean": cv_mae_mean,
        "cv_mae_std": cv_mae_std,
        "cv_fold_mae": fold_mae,
        "prediction_preview": prediction_preview(prediction_dates, y_test, pred_point),
    }
    if quantiles_enabled:
        metrics["test_picp"] = picp(y_test, pred_lower, pred_upper)
        metrics["test_interval_width"] = interval_mean_width(pred_lower, pred_upper)

    (model_output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return LgbmResult(metrics=metrics, prediction_path=prediction_path)
