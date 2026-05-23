from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .config import KyqmConfig, load_config
from .feature_engineering import (
    CORE_RECURRENT_COLUMNS,
    build_feature_table,
    build_long_horizon_feature_table,
    feature_columns,
    lgbm_feature_columns,
    long_feature_columns,
    split_by_time,
)
from .metrics import (
    baseline_metrics,
    interval_mean_width,
    mae,
    mape,
    picp,
    prediction_preview,
    rmse,
    smape,
)
from .model_lgb import (
    _build_params,
    lgbm_oof_and_eval_predictions,
    train_lightgbm_models,
)
from .model_ridge import ridge_oof_and_eval_predictions, train_ridge_model
from .prepare import PrepareParams, prepare_training_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multi-model forecasting pipeline for agricultural prices."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("kyqm/config.toml"),
        help="Path to kyqm config TOML.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["short", "long"],
        default=None,
        help="Run the existing short-horizon pipeline or the long-horizon comparison path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "lgbm", "gru", "lstm", "prophet", "ridge"],
        default=None,
        help="Run specific model path.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override recurrent-model epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override recurrent-model batch size."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override recurrent-model learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override recurrent-model device (cpu/cuda/auto).",
    )
    return parser.parse_args()


def _apply_overrides(
    cfg: KyqmConfig,
    *,
    pipeline: str | None,
    model: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    device: str | None,
) -> KyqmConfig:
    return KyqmConfig(
        data=cfg.data,
        lgbm=cfg.lgbm,
        prophet=cfg.prophet,
        long=cfg.long,
        lstm=cfg.lstm.__class__(
            enabled=cfg.lstm.enabled,
            sequence_length=cfg.lstm.sequence_length,
            hidden_dim=cfg.lstm.hidden_dim,
            num_layers=cfg.lstm.num_layers,
            dropout=cfg.lstm.dropout,
            batch_size=batch_size or cfg.lstm.batch_size,
            epochs=epochs or cfg.lstm.epochs,
            learning_rate=learning_rate or cfg.lstm.learning_rate,
            patience=cfg.lstm.patience,
            weight_decay=cfg.lstm.weight_decay,
            grad_clip_norm=cfg.lstm.grad_clip_norm,
            model_output_path=cfg.lstm.model_output_path,
            metrics_output_path=cfg.lstm.metrics_output_path,
            device=device or cfg.lstm.device,
        ),
        run=cfg.run.__class__(
            model=model or cfg.run.model,
            pipeline=pipeline or cfg.run.pipeline,
            seed=cfg.run.seed,
            summary_output_path=cfg.run.summary_output_path,
            prediction_output_dir=cfg.run.prediction_output_dir,
        ),
        gru=cfg.gru.__class__(
            enabled=cfg.gru.enabled,
            sequence_length=cfg.gru.sequence_length,
            hidden_dim=cfg.gru.hidden_dim,
            num_layers=cfg.gru.num_layers,
            dropout=cfg.gru.dropout,
            batch_size=batch_size or cfg.gru.batch_size,
            epochs=epochs or cfg.gru.epochs,
            learning_rate=learning_rate or cfg.gru.learning_rate,
            patience=cfg.gru.patience,
            weight_decay=cfg.gru.weight_decay,
            grad_clip_norm=cfg.gru.grad_clip_norm,
            quantiles_enabled=cfg.gru.quantiles_enabled,
            model_output_path=cfg.gru.model_output_path,
            metrics_output_path=cfg.gru.metrics_output_path,
            device=device or cfg.gru.device,
        ),
    )


def _prepare_cleaned(cfg: KyqmConfig):
    return prepare_training_frame(
        PrepareParams(
            market_prices_path=cfg.data.market_prices_path,
            weather_path=cfg.data.weather_path,
            output_path=cfg.data.cleaned_output_path,
            province_name=cfg.data.province_name,
            city_name=cfg.data.city_name,
            product_name=cfg.data.product_name,
            companion_products=cfg.data.companion_products,
            nearby_cucumber_provinces=cfg.data.nearby_cucumber_provinces,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            max_forward_fill_days=cfg.data.max_forward_fill_days,
            outlier_sigma=cfg.data.outlier_sigma,
        )
    )


def _run_short_pipeline(
    cfg: KyqmConfig,
    cleaned,
    *,
    run_model_override: str | None = None,
) -> dict[str, dict[str, float | int | str]]:
    feature_df = build_feature_table(cleaned, forecast_horizon=cfg.data.forecast_horizon)
    if cfg.lgbm.use_prophet_components:
        from .model_prophet import add_prophet_seasonal_features

        feature_df = add_prophet_seasonal_features(
            feature_df,
            train_end=cfg.data.train_end,
            weekly_seasonality=cfg.prophet.weekly_seasonality,
            yearly_seasonality=cfg.prophet.yearly_seasonality,
        )
    cfg.data.feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cfg.data.feature_output_path, index=False)

    splits = split_by_time(
        feature_df,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.test_end,
    )
    lightgbm_feature_cols = lgbm_feature_columns(
        feature_df,
        include_beijing=cfg.lgbm.use_beijing_lead_features,
        include_refined_weather=cfg.lgbm.use_refined_weather_features,
    )
    recurrent_feature_cols = [
        column for column in CORE_RECURRENT_COLUMNS if column in feature_df.columns
    ]

    summary: dict[str, dict[str, float | int | str]] = {}
    run_model = run_model_override or cfg.run.model
    prediction_dir = cfg.run.prediction_output_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)

    baseline_true = splits.test["target"].to_numpy(dtype=float)
    baseline_pred = splits.test["local_price"].to_numpy(dtype=float)
    summary["naive_last_price"] = {
        "model": "naive_last_price",
        **baseline_metrics(baseline_true, baseline_pred),
        "prediction_preview": prediction_preview(
            splits.test["target_date"].dt.strftime("%Y-%m-%d"),
            baseline_true,
            baseline_pred,
        ),
    }

    if run_model in {"all", "lgbm"} and cfg.lgbm.enabled:
        lgbm_result = train_lightgbm_models(
            train_df=splits.train,
            val_df=splits.val,
            test_df=splits.test,
            feature_columns=lightgbm_feature_cols,
            model_output_dir=cfg.lgbm.model_output_dir,
            prediction_output_dir=prediction_dir,
            quantiles_enabled=cfg.lgbm.quantiles_enabled,
            lower_alpha=cfg.lgbm.lower_alpha,
            upper_alpha=cfg.lgbm.upper_alpha,
            learning_rate=cfg.lgbm.learning_rate,
            n_estimators=cfg.lgbm.n_estimators,
            max_depth=cfg.lgbm.max_depth,
            num_leaves=cfg.lgbm.num_leaves,
            min_data_in_leaf=cfg.lgbm.min_data_in_leaf,
            lambda_l1=cfg.lgbm.lambda_l1,
            lambda_l2=cfg.lgbm.lambda_l2,
            early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
            cv_splits=cfg.lgbm.cv_splits,
        )
        summary["lightgbm"] = lgbm_result.metrics

    if run_model in {"all", "gru"} and cfg.gru.enabled:
        from .model_gru import train_gru_model

        gru_result = train_gru_model(
            frame=feature_df,
            feature_columns=recurrent_feature_cols,
            sequence_length=cfg.gru.sequence_length,
            hidden_dim=cfg.gru.hidden_dim,
            num_layers=cfg.gru.num_layers,
            dropout=cfg.gru.dropout,
            batch_size=cfg.gru.batch_size,
            epochs=cfg.gru.epochs,
            learning_rate=cfg.gru.learning_rate,
            patience=cfg.gru.patience,
            weight_decay=cfg.gru.weight_decay,
            grad_clip_norm=cfg.gru.grad_clip_norm,
            quantiles_enabled=cfg.gru.quantiles_enabled,
            model_output_path=cfg.gru.model_output_path,
            metrics_output_path=cfg.gru.metrics_output_path,
            prediction_output_dir=prediction_dir,
            seed=cfg.run.seed,
            device=cfg.gru.device,
            train_end=cfg.data.train_end,
            val_end=cfg.data.val_end,
            test_end=cfg.data.test_end,
        )
        summary["gru_attention"] = gru_result.metrics

    if run_model in {"all", "lstm"} and cfg.lstm.enabled:
        from .model_lstm import train_lstm_model

        lstm_result = train_lstm_model(
            frame=feature_df,
            feature_columns=recurrent_feature_cols,
            sequence_length=cfg.lstm.sequence_length,
            hidden_dim=cfg.lstm.hidden_dim,
            num_layers=cfg.lstm.num_layers,
            dropout=cfg.lstm.dropout,
            batch_size=cfg.lstm.batch_size,
            epochs=cfg.lstm.epochs,
            learning_rate=cfg.lstm.learning_rate,
            patience=cfg.lstm.patience,
            weight_decay=cfg.lstm.weight_decay,
            grad_clip_norm=cfg.lstm.grad_clip_norm,
            model_output_path=cfg.lstm.model_output_path,
            metrics_output_path=cfg.lstm.metrics_output_path,
            prediction_output_dir=prediction_dir,
            seed=cfg.run.seed,
            device=cfg.lstm.device,
            train_end=cfg.data.train_end,
            val_end=cfg.data.val_end,
            test_end=cfg.data.test_end,
        )
        summary["lstm_attention"] = lstm_result.metrics

    if run_model in {"all", "prophet"} and cfg.prophet.enabled:
        from .model_prophet import train_prophet_model

        prophet_result = train_prophet_model(
            train_df=splits.train,
            test_df=splits.test,
            model_output_path=cfg.prophet.model_output_path,
            prediction_output_dir=prediction_dir,
            daily_seasonality=cfg.prophet.daily_seasonality,
            weekly_seasonality=cfg.prophet.weekly_seasonality,
            yearly_seasonality=cfg.prophet.yearly_seasonality,
        )
        summary["prophet"] = prophet_result.metrics

    cfg.run.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.run.summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved feature table to: {cfg.data.feature_output_path}")
    print(f"Saved model summary to: {cfg.run.summary_output_path}")
    for name, metrics in summary.items():
        print(
            f"{name}: MAE={metrics.get('test_mae', float('nan')):.4f}, RMSE={metrics.get('test_rmse', float('nan')):.4f}"
        )
    return summary


def _load_prediction_frame(prediction_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(prediction_path, parse_dates=["date"]).sort_values("date")
    return frame.reset_index(drop=True)


def _eval_split_metrics(frame: pd.DataFrame, *, split: str) -> dict[str, float]:
    subset = frame[frame["split"] == split].copy()
    metrics = {
        f"{split}_mae": mae(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        ),
        f"{split}_rmse": rmse(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        ),
        f"{split}_mape": mape(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        ),
        f"{split}_smape": smape(
            subset["y_true"].to_numpy(dtype=float),
            subset["y_pred"].to_numpy(dtype=float),
        ),
    }
    interval_columns = None
    if {"y_pred_p05", "y_pred_p95"}.issubset(subset.columns):
        interval_columns = ("y_pred_p05", "y_pred_p95")
    elif {"y_pred_p10", "y_pred_p90"}.issubset(subset.columns):
        interval_columns = ("y_pred_p10", "y_pred_p90")
    if interval_columns is not None:
        lower = subset[interval_columns[0]].to_numpy(dtype=float)
        upper = subset[interval_columns[1]].to_numpy(dtype=float)
        if not np.isnan(lower).all() and not np.isnan(upper).all():
            metrics[f"{split}_picp"] = picp(
                subset["y_true"].to_numpy(dtype=float),
                lower,
                upper,
            )
            metrics[f"{split}_interval_width"] = interval_mean_width(lower, upper)
    return metrics


def _candidate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        **_eval_split_metrics(frame, split="val"),
        **_eval_split_metrics(frame, split="test"),
    }


def _write_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _blend_prediction_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    weight_left: float,
) -> pd.DataFrame:
    merged = left.merge(
        right,
        on=["date", "split"],
        suffixes=("_left", "_right"),
    ).sort_values("date")
    blended = pd.DataFrame(
        {
            "date": merged["date"],
            "split": merged["split"],
            "y_true": merged["y_true_left"],
            "y_pred": weight_left * merged["y_pred_left"]
            + (1.0 - weight_left) * merged["y_pred_right"],
        }
    )
    if {"y_pred_p10_left", "y_pred_p90_left", "y_pred_p10_right", "y_pred_p90_right"}.issubset(
        merged.columns
    ):
        blended["y_pred_p10"] = weight_left * merged["y_pred_p10_left"] + (
            1.0 - weight_left
        ) * merged["y_pred_p10_right"]
        blended["y_pred_p90"] = weight_left * merged["y_pred_p90_left"] + (
            1.0 - weight_left
        ) * merged["y_pred_p90_right"]
    if {"y_pred_p05_left", "y_pred_p95_left", "y_pred_p05_right", "y_pred_p95_right"}.issubset(
        merged.columns
    ):
        blended["y_pred_p05"] = weight_left * merged["y_pred_p05_left"] + (
            1.0 - weight_left
        ) * merged["y_pred_p05_right"]
        blended["y_pred_p95"] = weight_left * merged["y_pred_p95_left"] + (
            1.0 - weight_left
        ) * merged["y_pred_p95_right"]
    return blended


def _ridge_bias_calibrated_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    val_frame = frame[frame["split"] == "val"]
    bias = float((val_frame["y_true"] - val_frame["y_pred"]).mean())
    calibrated = frame.copy()
    calibrated["y_pred"] = calibrated["y_pred"] + bias
    return calibrated, bias


def _val_shift_scan_frame(
    frame: pd.DataFrame,
    cleaned: pd.DataFrame,
    *,
    max_shift_days: int = 5,
) -> tuple[pd.DataFrame, int]:
    actual_lookup = cleaned.copy()
    actual_lookup["date"] = pd.to_datetime(actual_lookup["date"])
    actual_lookup = actual_lookup.set_index("date")["local_price"]
    val_frame = frame[frame["split"] == "val"].copy()
    rows: list[dict[str, float | int]] = []
    best_shift = 0
    best_mae = float("inf")
    for shift_days in range(-max_shift_days, max_shift_days + 1):
        shifted_dates = val_frame["date"] + pd.to_timedelta(shift_days, unit="D")
        shifted_actual = shifted_dates.map(actual_lookup)
        mask = shifted_actual.notna()
        if not mask.any():
            continue
        candidate_mae = mae(
            shifted_actual.loc[mask].to_numpy(dtype=float),
            val_frame.loc[mask, "y_pred"].to_numpy(dtype=float),
        )
        rows.append({"shift_days": shift_days, "val_mae": candidate_mae})
        if candidate_mae < best_mae:
            best_mae = candidate_mae
            best_shift = shift_days
    return pd.DataFrame(rows), best_shift


def _lgb_oof_residual_quantiles(
    *,
    train_df: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    baseline_column: str | None,
    params: dict[str, float | int],
    cv_splits: int,
) -> tuple[float, float, dict[str, float]]:
    ordered = train_df.sort_values("target_date").reset_index(drop=True)
    split_count = min(cv_splits, len(ordered) - 1)
    if split_count < 2:
        return 0.0, 0.0, {}

    x_all = ordered[feature_columns]
    y_all = ordered["target"].to_numpy(dtype=float)
    baseline_all = (
        ordered[baseline_column].to_numpy(dtype=float)
        if baseline_column is not None
        else np.zeros(len(ordered), dtype=float)
    )
    fit_target = y_all - baseline_all if baseline_column is not None else y_all

    splitter = TimeSeriesSplit(n_splits=split_count)
    oof_pred = np.full(len(ordered), np.nan, dtype=float)
    for fit_idx, holdout_idx in splitter.split(x_all):
        model = lgb.LGBMRegressor(
            **_build_params(
                objective="regression",
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                num_leaves=int(params["num_leaves"]),
                min_data_in_leaf=int(params["min_data_in_leaf"]),
                lambda_l1=float(params["lambda_l1"]),
                lambda_l2=float(params["lambda_l2"]),
            )
        )
        model.fit(x_all.iloc[fit_idx], fit_target[fit_idx])
        oof_pred[holdout_idx] = model.predict(x_all.iloc[holdout_idx]) + baseline_all[holdout_idx]

    residuals = y_all[~np.isnan(oof_pred)] - oof_pred[~np.isnan(oof_pred)]
    q10 = float(np.quantile(residuals, 0.1))
    q90 = float(np.quantile(residuals, 0.9))

    val_subset = val_frame[val_frame["split"] == "val"].copy()
    calibrated_val_lower = val_subset["y_pred"].to_numpy(dtype=float) + q10
    calibrated_val_upper = val_subset["y_pred"].to_numpy(dtype=float) + q90
    val_metrics = {
        "val_picp": picp(
            val_subset["y_true"].to_numpy(dtype=float),
            calibrated_val_lower,
            calibrated_val_upper,
        ),
        "val_interval_width": interval_mean_width(
            calibrated_val_lower, calibrated_val_upper
        ),
    }
    return q10, q90, val_metrics


def _choose_best_candidate(
    candidate_rows: list[dict[str, float | int | str]],
    *,
    preferred_order: list[str],
) -> dict[str, float | int | str]:
    order_map = {name: idx for idx, name in enumerate(preferred_order)}

    def _score(row: dict[str, float | int | str]) -> tuple[float, float, int]:
        base_score = float(row["val_mae"])
        if "val_picp" in row:
            picp_penalty = abs(float(row["val_picp"]) - 0.85)
            width_penalty = float(row.get("val_interval_width", 0.0)) * 0.02
            return (
                base_score + picp_penalty + width_penalty,
                base_score,
                order_map.get(str(row["candidate"]), len(order_map)),
            )
        return (
            base_score,
            base_score,
            order_map.get(str(row["candidate"]), len(order_map)),
        )

    ranked = sorted(
        candidate_rows,
        key=_score,
    )
    return ranked[0]


def _volatility_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    raw = frame["roll_std_30"].to_numpy(dtype=float)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    mean_value = float(np.nanmean(raw)) if not np.isnan(raw).all() else 1.0
    mean_value = max(mean_value, 1e-6)
    filled = np.nan_to_num(raw, nan=mean_value)
    normalized = filled / mean_value
    return np.clip(normalized, 0.5, 3.0)


def _train_multi_quantile_candidate(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    prediction_path: Path,
    params: dict[str, float | int],
    sample_weight: np.ndarray | None,
    baseline_column: str | None,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    x_train = train_df[feature_columns]
    x_val = val_df[feature_columns]
    x_test = test_df[feature_columns]
    y_train = train_df["target"].to_numpy(dtype=float)
    y_val = val_df["target"].to_numpy(dtype=float)
    y_test = test_df["target"].to_numpy(dtype=float)
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

    val_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    for alpha in quantiles:
        model = lgb.LGBMRegressor(
            **_build_params(
                objective="quantile",
                alpha=alpha,
                learning_rate=float(params["learning_rate"]),
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                num_leaves=int(params["num_leaves"]),
                min_data_in_leaf=int(params["min_data_in_leaf"]),
                lambda_l1=float(params["lambda_l1"]),
                lambda_l2=float(params["lambda_l2"]),
            )
        )
        model.fit(
            x_train,
            y_train_fit,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val_fit)],
            eval_metric="quantile",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        val_predictions.append(model.predict(x_val) + val_baseline)
        test_predictions.append(model.predict(x_test) + test_baseline)

    val_quantiles = np.sort(np.column_stack(val_predictions), axis=1)
    test_quantiles = np.sort(np.column_stack(test_predictions), axis=1)
    val_dates = val_df["target_date"].dt.strftime("%Y-%m-%d")
    test_dates = test_df["target_date"].dt.strftime("%Y-%m-%d")
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": val_dates,
                    "split": "val",
                    "y_true": y_val,
                    "y_pred": val_quantiles[:, 2],
                    "y_pred_p05": val_quantiles[:, 0],
                    "y_pred_p25": val_quantiles[:, 1],
                    "y_pred_p50": val_quantiles[:, 2],
                    "y_pred_p75": val_quantiles[:, 3],
                    "y_pred_p95": val_quantiles[:, 4],
                    "y_pred_p10": val_quantiles[:, 0],
                    "y_pred_p90": val_quantiles[:, 4],
                }
            ),
            pd.DataFrame(
                {
                    "date": test_dates,
                    "split": "test",
                    "y_true": y_test,
                    "y_pred": test_quantiles[:, 2],
                    "y_pred_p05": test_quantiles[:, 0],
                    "y_pred_p25": test_quantiles[:, 1],
                    "y_pred_p50": test_quantiles[:, 2],
                    "y_pred_p75": test_quantiles[:, 3],
                    "y_pred_p95": test_quantiles[:, 4],
                    "y_pred_p10": test_quantiles[:, 0],
                    "y_pred_p90": test_quantiles[:, 4],
                }
            ),
        ],
        ignore_index=True,
    )
    _write_prediction_frame(frame, prediction_path)
    return frame


def _lgb_oof_residual_frame(
    *,
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, float | int],
    baseline_column: str | None,
    cv_splits: int,
    sample_weight: np.ndarray | None = None,
    point_objective: str = "regression",
    point_alpha: float | None = None,
) -> pd.DataFrame:
    oof = lgbm_oof_and_eval_predictions(
        train_df=train_df,
        val_df=train_df.iloc[:1].copy(),
        test_df=train_df.iloc[:1].copy(),
        feature_columns=feature_columns,
        learning_rate=float(params["learning_rate"]),
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        num_leaves=int(params["num_leaves"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        lambda_l1=float(params["lambda_l1"]),
        lambda_l2=float(params["lambda_l2"]),
        cv_splits=cv_splits,
        baseline_column=baseline_column,
        point_objective=point_objective,
        point_alpha=point_alpha,
        sample_weight=sample_weight,
    )
    ordered = train_df.sort_values("target_date").reset_index(drop=True).copy()
    ordered["oof_pred"] = oof["train_oof_pred"]
    ordered["residual"] = ordered["target"] - ordered["oof_pred"]
    return ordered[["target_date", "residual", "roll_std_30", "recent_return_std_7"]].rename(
        columns={"target_date": "date"}
    )


def _bucket_thresholds(series: pd.Series) -> tuple[float, float]:
    q1, q2 = series.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    return float(q1), float(q2)


def _assign_bucket(values: pd.Series | np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), bins=np.array(thresholds, dtype=float))


def _adaptive_offsets(
    calibration_frame: pd.DataFrame,
    *,
    use_split: bool,
) -> tuple[tuple[float, float], dict[int, tuple[float, float]]]:
    ordered = calibration_frame.sort_values("date").reset_index(drop=True)
    if use_split and len(ordered) >= 20:
        split_idx = len(ordered) // 2
        threshold_frame = ordered.iloc[:split_idx]
        residual_frame = ordered.iloc[split_idx:]
    else:
        threshold_frame = ordered
        residual_frame = ordered
    thresholds = _bucket_thresholds(threshold_frame["roll_std_30"])
    residual_frame = residual_frame.copy()
    residual_frame["bucket"] = _assign_bucket(residual_frame["roll_std_30"], thresholds)
    global_offsets = (
        float(residual_frame["residual"].quantile(0.1)),
        float(residual_frame["residual"].quantile(0.9)),
    )
    offsets: dict[int, tuple[float, float]] = {}
    for bucket in (0, 1, 2):
        bucket_values = residual_frame.loc[residual_frame["bucket"] == bucket, "residual"]
        if len(bucket_values) < 20:
            offsets[bucket] = global_offsets
            continue
        offsets[bucket] = (
            float(bucket_values.quantile(0.1)),
            float(bucket_values.quantile(0.9)),
        )
    return thresholds, offsets


def _apply_adaptive_offsets(
    base_frame: pd.DataFrame,
    eval_meta: pd.DataFrame,
    *,
    thresholds: tuple[float, float],
    offsets: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    merged = base_frame.merge(eval_meta, on=["date", "split"], how="left").sort_values("date")
    buckets = _assign_bucket(merged["roll_std_30"], thresholds)
    lower = []
    upper = []
    for bucket in buckets:
        bucket_lower, bucket_upper = offsets.get(int(bucket), offsets[1])
        lower.append(bucket_lower)
        upper.append(bucket_upper)
    calibrated = base_frame.copy()
    calibrated["y_pred_p10"] = calibrated["y_pred"] + np.asarray(lower, dtype=float)
    calibrated["y_pred_p90"] = calibrated["y_pred"] + np.asarray(upper, dtype=float)
    return calibrated


def _fit_evt_tail_quantiles(
    calibration_frame: pd.DataFrame,
    *,
    min_tail_size: int = 50,
) -> tuple[float, float] | None:
    residuals = calibration_frame["residual"].to_numpy(dtype=float)
    upper_threshold = float(np.quantile(residuals, 0.9))
    lower_threshold = float(np.quantile(residuals, 0.1))
    upper_excess = residuals[residuals > upper_threshold] - upper_threshold
    lower_excess = -residuals[residuals < lower_threshold] + lower_threshold
    if len(upper_excess) < min_tail_size or len(lower_excess) < min_tail_size:
        return None
    upper_shape, _, upper_scale = genpareto.fit(upper_excess, floc=0)
    lower_shape, _, lower_scale = genpareto.fit(lower_excess, floc=0)
    upper_q = upper_threshold + float(genpareto.ppf(0.95, upper_shape, loc=0, scale=upper_scale))
    lower_q = lower_threshold - float(genpareto.ppf(0.95, lower_shape, loc=0, scale=lower_scale))
    return lower_q, upper_q


def _apply_evt_tail_adjustment(
    adaptive_frame: pd.DataFrame,
    eval_meta: pd.DataFrame,
    *,
    thresholds: tuple[float, float],
    evt_tail: tuple[float, float],
) -> pd.DataFrame:
    merged = adaptive_frame.merge(eval_meta, on=["date", "split"], how="left").sort_values("date")
    buckets = _assign_bucket(merged["roll_std_30"], thresholds)
    evt_frame = adaptive_frame.copy()
    high_bucket = buckets == 2
    evt_frame.loc[high_bucket, "y_pred_p10"] = np.minimum(
        evt_frame.loc[high_bucket, "y_pred_p10"],
        evt_frame.loc[high_bucket, "y_pred"] + evt_tail[0],
    )
    evt_frame.loc[high_bucket, "y_pred_p90"] = np.maximum(
        evt_frame.loc[high_bucket, "y_pred_p90"],
        evt_frame.loc[high_bucket, "y_pred"] + evt_tail[1],
    )
    return evt_frame


def _stacking_candidate_frame(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_meta_features: dict[str, np.ndarray],
    val_meta_features: dict[str, np.ndarray],
    test_meta_features: dict[str, np.ndarray],
) -> pd.DataFrame:
    meta_columns = list(train_meta_features)
    x_train = np.column_stack([train_meta_features[col] for col in meta_columns])
    x_val = np.column_stack([val_meta_features[col] for col in meta_columns])
    x_test = np.column_stack([test_meta_features[col] for col in meta_columns])
    y_train = train_df["target"].to_numpy(dtype=float)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(x_train, y_train)
    val_dates = val_df["target_date"].dt.strftime("%Y-%m-%d")
    test_dates = test_df["target_date"].dt.strftime("%Y-%m-%d")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": val_dates,
                    "split": "val",
                    "y_true": val_df["target"].to_numpy(dtype=float),
                    "y_pred": meta_model.predict(x_val),
                }
            ),
            pd.DataFrame(
                {
                    "date": test_dates,
                    "split": "test",
                    "y_true": test_df["target"].to_numpy(dtype=float),
                    "y_pred": meta_model.predict(x_test),
                }
            ),
        ],
        ignore_index=True,
    )


def _dynamic_blend_frame(
    *,
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    train_volatility: pd.Series,
    eval_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, float]]:
    left_frame = left_frame.copy()
    right_frame = right_frame.copy()
    eval_meta = eval_meta.copy()
    left_frame["date"] = pd.to_datetime(left_frame["date"])
    right_frame["date"] = pd.to_datetime(right_frame["date"])
    eval_meta["date"] = pd.to_datetime(eval_meta["date"])
    thresholds = _bucket_thresholds(train_volatility)
    val_meta = eval_meta[eval_meta["split"] == "val"].copy()
    val_merged = (
        left_frame[left_frame["split"] == "val"]
        .merge(
            right_frame[right_frame["split"] == "val"],
            on=["date", "split"],
            suffixes=("_left", "_right"),
        )
        .merge(val_meta, on=["date", "split"], how="left")
    )
    val_merged["bucket"] = _assign_bucket(val_merged["roll_std_30"], thresholds)
    weights: dict[int, float] = {}
    grid = np.linspace(0.0, 1.0, 11)
    global_best = 0.5
    global_mae = float("inf")
    for weight in grid:
        pred = weight * val_merged["y_pred_left"] + (1.0 - weight) * val_merged["y_pred_right"]
        current = mae(val_merged["y_true_left"].to_numpy(dtype=float), pred.to_numpy(dtype=float))
        if current < global_mae:
            global_mae = current
            global_best = float(weight)
    for bucket in (0, 1, 2):
        bucket_rows = val_merged[val_merged["bucket"] == bucket]
        if bucket_rows.empty:
            weights[bucket] = global_best
            continue
        best_weight = global_best
        best_mae = float("inf")
        for weight in grid:
            pred = weight * bucket_rows["y_pred_left"] + (1.0 - weight) * bucket_rows["y_pred_right"]
            current = mae(bucket_rows["y_true_left"].to_numpy(dtype=float), pred.to_numpy(dtype=float))
            if current < best_mae:
                best_mae = current
                best_weight = float(weight)
        weights[bucket] = best_weight

    merged = left_frame.merge(
        right_frame,
        on=["date", "split"],
        suffixes=("_left", "_right"),
    ).merge(eval_meta, on=["date", "split"], how="left")
    merged["bucket"] = _assign_bucket(merged["roll_std_30"], thresholds)
    dynamic_weights = merged["bucket"].map(weights).astype(float)
    blended = pd.DataFrame(
        {
            "date": merged["date"],
            "split": merged["split"],
            "y_true": merged["y_true_left"],
            "y_pred": dynamic_weights.to_numpy(dtype=float) * merged["y_pred_left"]
            + (1.0 - dynamic_weights.to_numpy(dtype=float)) * merged["y_pred_right"],
        }
    )
    if {"y_pred_p10_left", "y_pred_p90_left", "y_pred_p10_right", "y_pred_p90_right"}.issubset(merged.columns):
        blended["y_pred_p10"] = dynamic_weights.to_numpy(dtype=float) * merged["y_pred_p10_left"] + (
            1.0 - dynamic_weights.to_numpy(dtype=float)
        ) * merged["y_pred_p10_right"]
        blended["y_pred_p90"] = dynamic_weights.to_numpy(dtype=float) * merged["y_pred_p90_left"] + (
            1.0 - dynamic_weights.to_numpy(dtype=float)
        ) * merged["y_pred_p90_right"]
    blended["date"] = pd.to_datetime(blended["date"]).dt.strftime("%Y-%m-%d")
    return blended, weights


def _run_long_pipeline(cfg: KyqmConfig, cleaned) -> dict[str, dict[str, float | int | str]]:
    long_df = build_long_horizon_feature_table(
        cleaned,
        horizons=cfg.long.horizons,
        history_window_days=cfg.long.history_window_days,
        anchor_step_days=cfg.long.anchor_step_days,
        train_end=cfg.data.train_end,
        augmentation_enabled=cfg.long.augmentation_enabled,
        augmentation_jitter_days=cfg.long.augmentation_jitter_days,
    )
    cfg.long.feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(cfg.long.feature_output_path, index=False)

    summary: dict[str, dict[str, float | int | str]] = {}
    comparison_rows: list[dict[str, float | int | str]] = []
    candidate_rows: list[dict[str, float | int | str]] = []
    selection_metadata: dict[str, dict[str, float | int | str]] = {}
    run_model = cfg.run.model
    prediction_dir = cfg.long.prediction_output_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    candidate_output_path = cfg.long.comparison_output_path.with_name(
        "model_candidates_long.csv"
    )
    actual_lookup = (
        cleaned.assign(date=pd.to_datetime(cleaned["date"]))
        .set_index("date")["local_price"]
        .astype(float)
    )

    short_summary = _run_short_pipeline(cfg, cleaned, run_model_override="lgbm")
    if "naive_last_price" in short_summary:
        baseline_mae_1d = float(short_summary["naive_last_price"]["test_mae"])
        comparison_rows.append(
            {
                "horizon_days": 1,
                "model": "naive_last_price",
                "baseline_name": "current_price",
                "test_mae": baseline_mae_1d,
                "baseline_mae": baseline_mae_1d,
                "mae_ratio": 1.0,
            }
        )
        summary["naive_1d"] = short_summary["naive_last_price"]
    if "lightgbm" in short_summary:
        ratio_1d = float(short_summary["lightgbm"]["test_mae"]) / baseline_mae_1d
        comparison_rows.append(
            {
                "horizon_days": 1,
                "model": "lightgbm_1d",
                "baseline_name": "current_price",
                "test_mae": float(short_summary["lightgbm"]["test_mae"]),
                "baseline_mae": baseline_mae_1d,
                "mae_ratio": ratio_1d,
            }
        )
        summary["lightgbm_1d"] = short_summary["lightgbm"]

    for horizon_days in cfg.long.horizons:
        horizon_df = long_df[long_df["horizon_days"] == horizon_days].copy()
        splits = split_by_time(
            horizon_df,
            train_end=cfg.data.train_end,
            val_end=cfg.data.val_end,
            test_end=cfg.data.test_end,
        )
        features = long_feature_columns(horizon_df)
        eval_meta = pd.concat(
            [
                splits.val[["target_date", "roll_std_30", "recent_return_std_7", "price_change_30"]]
                .rename(columns={"target_date": "date"})
                .assign(split="val"),
                splits.test[["target_date", "roll_std_30", "recent_return_std_7", "price_change_30"]]
                .rename(columns={"target_date": "date"})
                .assign(split="test"),
            ],
            ignore_index=True,
        )
        train_sample_weight = _volatility_sample_weight(splits.train)
        baseline_true = splits.test["target"].to_numpy(dtype=float)
        baseline_pred = splits.test["selected_baseline"].to_numpy(dtype=float)
        baseline_name = str(splits.test["selected_baseline_name"].mode().iloc[0])
        horizon_prediction_frames: dict[str, pd.DataFrame] = {}
        horizon_lgb_meta: dict[str, dict[str, object]] = {}
        horizon_oof_meta: dict[str, dict[str, np.ndarray]] = {}
        baseline_key = f"naive_{horizon_days}d"
        baseline_metrics_row = {
            "model": baseline_key,
            "baseline_name": baseline_name,
            **baseline_metrics(baseline_true, baseline_pred),
            "prediction_preview": prediction_preview(
                splits.test["target_date"].dt.strftime("%Y-%m-%d"),
                baseline_true,
                baseline_pred,
            ),
        }
        summary[baseline_key] = baseline_metrics_row
        baseline_mae = float(baseline_metrics_row["test_mae"])
        comparison_rows.append(
            {
                "horizon_days": horizon_days,
                "model": baseline_key,
                "baseline_name": baseline_name,
                "test_mae": baseline_mae,
                "baseline_mae": baseline_mae,
                "mae_ratio": 1.0,
            }
        )

        ridge_frame: pd.DataFrame | None = None
        if run_model in {"all", "ridge"}:
            ridge_oof_bundle = ridge_oof_and_eval_predictions(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                alpha=10.0,
                baseline_column=None,
                cv_splits=cfg.lgbm.cv_splits,
            )
            horizon_oof_meta["ridge"] = {
                "train_oof_pred": ridge_oof_bundle["train_oof_pred"],  # type: ignore[assignment]
                "val_pred": ridge_oof_bundle["val_pred"],  # type: ignore[assignment]
                "test_pred": ridge_oof_bundle["test_pred"],  # type: ignore[assignment]
            }
            ridge_result = train_ridge_model(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                model_output_dir=cfg.long.ridge_model_output_dir / f"h{horizon_days}",
                prediction_output_dir=prediction_dir / f"h{horizon_days}",
                baseline_column=None,
                model_name=f"ridge_{horizon_days}d",
                prediction_filename=f"ridge_{horizon_days}d_predictions.csv",
            )
            ridge_frame = _load_prediction_frame(ridge_result.prediction_path)
            horizon_prediction_frames["ridge"] = ridge_frame
            ridge_metrics = _candidate_metrics(ridge_frame)
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "ridge",
                    "baseline_name": baseline_name,
                    **ridge_metrics,
                }
            )
            summary[f"ridge_{horizon_days}d"] = ridge_result.metrics

            if horizon_days in {30, 90}:
                bias_frame, bias_value = _ridge_bias_calibrated_frame(ridge_frame)
                bias_path = prediction_dir / f"h{horizon_days}" / f"ridge_bias_{horizon_days}d_predictions.csv"
                _write_prediction_frame(bias_frame, bias_path)
                horizon_prediction_frames["ridge_bias_calibrated"] = bias_frame
                candidate_rows.append(
                    {
                        "horizon_days": horizon_days,
                        "candidate": "ridge_bias_calibrated",
                        "baseline_name": baseline_name,
                        "bias": bias_value,
                        **_candidate_metrics(bias_frame),
                    }
                )

        if run_model in {"all", "lgbm"} and cfg.lgbm.enabled:
            base_lgb_params = {
                "learning_rate": cfg.lgbm.learning_rate,
                "n_estimators": cfg.lgbm.n_estimators,
                "max_depth": cfg.lgbm.max_depth,
                "num_leaves": cfg.lgbm.num_leaves,
                "min_data_in_leaf": cfg.lgbm.min_data_in_leaf,
                "lambda_l1": cfg.lgbm.lambda_l1,
                "lambda_l2": cfg.lgbm.lambda_l2,
            }
            lgbm_result = train_lightgbm_models(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                model_output_dir=cfg.long.lgbm_model_output_dir / f"h{horizon_days}",
                prediction_output_dir=prediction_dir / f"h{horizon_days}",
                baseline_column=None,
                model_name=f"lightgbm_{horizon_days}d",
                prediction_filename=f"lgbm_{horizon_days}d_predictions.csv",
                quantiles_enabled=cfg.lgbm.quantiles_enabled,
                lower_alpha=cfg.lgbm.lower_alpha,
                upper_alpha=cfg.lgbm.upper_alpha,
                learning_rate=cfg.lgbm.learning_rate,
                n_estimators=cfg.lgbm.n_estimators,
                max_depth=cfg.lgbm.max_depth,
                num_leaves=cfg.lgbm.num_leaves,
                min_data_in_leaf=cfg.lgbm.min_data_in_leaf,
                lambda_l1=cfg.lgbm.lambda_l1,
                lambda_l2=cfg.lgbm.lambda_l2,
                early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
                cv_splits=cfg.lgbm.cv_splits,
            )
            horizon_oof_meta["lightgbm"] = lgbm_oof_and_eval_predictions(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                learning_rate=cfg.lgbm.learning_rate,
                n_estimators=cfg.lgbm.n_estimators,
                max_depth=cfg.lgbm.max_depth,
                num_leaves=cfg.lgbm.num_leaves,
                min_data_in_leaf=cfg.lgbm.min_data_in_leaf,
                lambda_l1=cfg.lgbm.lambda_l1,
                lambda_l2=cfg.lgbm.lambda_l2,
                cv_splits=cfg.lgbm.cv_splits,
                baseline_column=None,
            )
            base_lgb_frame = _load_prediction_frame(lgbm_result.prediction_path)
            horizon_prediction_frames["lightgbm"] = base_lgb_frame
            horizon_lgb_meta["lightgbm"] = {
                "feature_columns": features,
                "baseline_column": None,
                "params": base_lgb_params,
            }
            summary[f"lightgbm_{horizon_days}d"] = lgbm_result.metrics
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "lightgbm",
                    "baseline_name": baseline_name,
                    **_candidate_metrics(base_lgb_frame),
                }
            )

            weighted_huber_params = {
                "learning_rate": cfg.lgbm.learning_rate,
                "n_estimators": int(cfg.lgbm.n_estimators * 1.2),
                "max_depth": max(cfg.lgbm.max_depth + 1, 4),
                "num_leaves": max(cfg.lgbm.num_leaves * 2, 15),
                "min_data_in_leaf": max(cfg.lgbm.min_data_in_leaf // 2, 12),
                "lambda_l1": max(cfg.lgbm.lambda_l1 * 0.5, 0.0),
                "lambda_l2": max(cfg.lgbm.lambda_l2 * 0.5, 0.0),
            }
            weighted_huber_result = train_lightgbm_models(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                model_output_dir=cfg.long.lgbm_model_output_dir / f"h{horizon_days}_weighted_huber",
                prediction_output_dir=prediction_dir / f"h{horizon_days}",
                baseline_column=None,
                model_name=f"lightgbm_weighted_huber_{horizon_days}d",
                prediction_filename=f"lgbm_weighted_huber_{horizon_days}d_predictions.csv",
                quantiles_enabled=cfg.lgbm.quantiles_enabled,
                lower_alpha=cfg.lgbm.lower_alpha,
                upper_alpha=cfg.lgbm.upper_alpha,
                learning_rate=float(weighted_huber_params["learning_rate"]),
                n_estimators=int(weighted_huber_params["n_estimators"]),
                max_depth=int(weighted_huber_params["max_depth"]),
                num_leaves=int(weighted_huber_params["num_leaves"]),
                min_data_in_leaf=int(weighted_huber_params["min_data_in_leaf"]),
                lambda_l1=float(weighted_huber_params["lambda_l1"]),
                lambda_l2=float(weighted_huber_params["lambda_l2"]),
                early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
                cv_splits=cfg.lgbm.cv_splits,
                point_objective="huber",
                sample_weight=train_sample_weight,
            )
            horizon_oof_meta["lightgbm_weighted_huber"] = lgbm_oof_and_eval_predictions(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                learning_rate=float(weighted_huber_params["learning_rate"]),
                n_estimators=int(weighted_huber_params["n_estimators"]),
                max_depth=int(weighted_huber_params["max_depth"]),
                num_leaves=int(weighted_huber_params["num_leaves"]),
                min_data_in_leaf=int(weighted_huber_params["min_data_in_leaf"]),
                lambda_l1=float(weighted_huber_params["lambda_l1"]),
                lambda_l2=float(weighted_huber_params["lambda_l2"]),
                cv_splits=cfg.lgbm.cv_splits,
                baseline_column=None,
                point_objective="huber",
                sample_weight=train_sample_weight,
            )
            weighted_huber_frame = _load_prediction_frame(weighted_huber_result.prediction_path)
            horizon_prediction_frames["lightgbm_weighted_huber"] = weighted_huber_frame
            horizon_lgb_meta["lightgbm_weighted_huber"] = {
                "feature_columns": features,
                "baseline_column": None,
                "params": weighted_huber_params,
                "point_objective": "huber",
                "sample_weight": train_sample_weight,
            }
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "lightgbm_weighted_huber",
                    "baseline_name": baseline_name,
                    **_candidate_metrics(weighted_huber_frame),
                }
            )

            multiq_path = prediction_dir / f"h{horizon_days}" / f"lgbm_multiq_{horizon_days}d_predictions.csv"
            multiq_frame = _train_multi_quantile_candidate(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                prediction_path=multiq_path,
                params=weighted_huber_params,
                sample_weight=train_sample_weight,
                baseline_column=None,
            )
            horizon_oof_meta["multi_quantile_p50"] = lgbm_oof_and_eval_predictions(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                learning_rate=float(weighted_huber_params["learning_rate"]),
                n_estimators=int(weighted_huber_params["n_estimators"]),
                max_depth=int(weighted_huber_params["max_depth"]),
                num_leaves=int(weighted_huber_params["num_leaves"]),
                min_data_in_leaf=int(weighted_huber_params["min_data_in_leaf"]),
                lambda_l1=float(weighted_huber_params["lambda_l1"]),
                lambda_l2=float(weighted_huber_params["lambda_l2"]),
                cv_splits=cfg.lgbm.cv_splits,
                baseline_column=None,
                point_objective="quantile",
                point_alpha=0.5,
                sample_weight=train_sample_weight,
            )
            horizon_prediction_frames["multi_quantile_p50"] = multiq_frame
            horizon_lgb_meta["multi_quantile_p50"] = {
                "feature_columns": features,
                "baseline_column": None,
                "params": weighted_huber_params,
                "point_objective": "quantile",
                "point_alpha": 0.5,
                "sample_weight": train_sample_weight,
            }
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "multi_quantile_p50",
                    "baseline_name": baseline_name,
                    **_candidate_metrics(multiq_frame),
                }
            )

            lowreg_params = {
                "learning_rate": cfg.lgbm.learning_rate,
                "n_estimators": int(cfg.lgbm.n_estimators * 1.25),
                "max_depth": max(cfg.lgbm.max_depth + 1, 4),
                "num_leaves": max(cfg.lgbm.num_leaves * 2, 15),
                "min_data_in_leaf": max(cfg.lgbm.min_data_in_leaf // 2, 12),
                "lambda_l1": max(cfg.lgbm.lambda_l1 * 0.25, 0.0),
                "lambda_l2": max(cfg.lgbm.lambda_l2 * 0.25, 0.0),
            }
            lowreg_result = train_lightgbm_models(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                model_output_dir=cfg.long.lgbm_model_output_dir / f"h{horizon_days}_lowreg",
                prediction_output_dir=prediction_dir / f"h{horizon_days}",
                baseline_column=None,
                model_name=f"lightgbm_lowreg_{horizon_days}d",
                prediction_filename=f"lgbm_lowreg_{horizon_days}d_predictions.csv",
                quantiles_enabled=cfg.lgbm.quantiles_enabled,
                lower_alpha=cfg.lgbm.lower_alpha,
                upper_alpha=cfg.lgbm.upper_alpha,
                learning_rate=float(lowreg_params["learning_rate"]),
                n_estimators=int(lowreg_params["n_estimators"]),
                max_depth=int(lowreg_params["max_depth"]),
                num_leaves=int(lowreg_params["num_leaves"]),
                min_data_in_leaf=int(lowreg_params["min_data_in_leaf"]),
                lambda_l1=float(lowreg_params["lambda_l1"]),
                lambda_l2=float(lowreg_params["lambda_l2"]),
                early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
                cv_splits=cfg.lgbm.cv_splits,
            )
            horizon_oof_meta["lightgbm_lowreg"] = lgbm_oof_and_eval_predictions(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                feature_columns=features,
                learning_rate=float(lowreg_params["learning_rate"]),
                n_estimators=int(lowreg_params["n_estimators"]),
                max_depth=int(lowreg_params["max_depth"]),
                num_leaves=int(lowreg_params["num_leaves"]),
                min_data_in_leaf=int(lowreg_params["min_data_in_leaf"]),
                lambda_l1=float(lowreg_params["lambda_l1"]),
                lambda_l2=float(lowreg_params["lambda_l2"]),
                cv_splits=cfg.lgbm.cv_splits,
                baseline_column=None,
            )
            lowreg_frame = _load_prediction_frame(lowreg_result.prediction_path)
            horizon_prediction_frames["lightgbm_lowreg"] = lowreg_frame
            horizon_lgb_meta["lightgbm_lowreg"] = {
                "feature_columns": features,
                "baseline_column": None,
                "params": lowreg_params,
            }
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "lightgbm_lowreg",
                    "baseline_name": baseline_name,
                    **_candidate_metrics(lowreg_frame),
                }
            )

            if horizon_days == 7:
                diff_result = train_lightgbm_models(
                    train_df=splits.train,
                    val_df=splits.val,
                    test_df=splits.test,
                    feature_columns=features,
                    model_output_dir=cfg.long.lgbm_model_output_dir / "h7_diff",
                    prediction_output_dir=prediction_dir / "h7",
                    baseline_column="naive_current_price",
                    model_name="lightgbm_diff_7d",
                    prediction_filename="lgbm_diff_7d_predictions.csv",
                    quantiles_enabled=cfg.lgbm.quantiles_enabled,
                    lower_alpha=cfg.lgbm.lower_alpha,
                    upper_alpha=cfg.lgbm.upper_alpha,
                    learning_rate=cfg.lgbm.learning_rate,
                    n_estimators=cfg.lgbm.n_estimators,
                    max_depth=cfg.lgbm.max_depth,
                    num_leaves=cfg.lgbm.num_leaves,
                    min_data_in_leaf=cfg.lgbm.min_data_in_leaf,
                    lambda_l1=cfg.lgbm.lambda_l1,
                    lambda_l2=cfg.lgbm.lambda_l2,
                    early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
                    cv_splits=cfg.lgbm.cv_splits,
                )
                diff_frame = _load_prediction_frame(diff_result.prediction_path)
                horizon_prediction_frames["lightgbm_diff"] = diff_frame
                horizon_lgb_meta["lightgbm_diff"] = {
                    "feature_columns": features,
                    "baseline_column": "naive_current_price",
                    "params": base_lgb_params,
                }
                candidate_rows.append(
                    {
                        "horizon_days": 7,
                        "candidate": "lightgbm_diff",
                        "baseline_name": baseline_name,
                        **_candidate_metrics(diff_frame),
                    }
                )

                shift_scan, best_shift = _val_shift_scan_frame(
                    base_lgb_frame,
                    cleaned,
                    max_shift_days=5,
                )
                if not shift_scan.empty:
                    shifted_frame = base_lgb_frame.copy()
                    shifted_frame["date"] = shifted_frame["date"] + pd.to_timedelta(best_shift, unit="D")
                    shifted_frame["y_true"] = shifted_frame["date"].map(actual_lookup)
                    shifted_frame = shifted_frame.dropna(subset=["y_true"]).reset_index(drop=True)
                    shift_path = prediction_dir / "h7" / "lgbm_shifted_7d_predictions.csv"
                    _write_prediction_frame(shifted_frame, shift_path)
                    horizon_prediction_frames["lightgbm_shifted"] = shifted_frame
                    candidate_rows.append(
                        {
                            "horizon_days": 7,
                            "candidate": "lightgbm_shifted",
                            "baseline_name": baseline_name,
                            "best_shift_days": int(best_shift),
                            **_candidate_metrics(shifted_frame),
                        }
                    )

            if horizon_days in {30, 90} and ridge_frame is not None:
                ridge_stack = ridge_oof_and_eval_predictions(
                    train_df=splits.train,
                    val_df=splits.val,
                    test_df=splits.test,
                    feature_columns=features,
                    alpha=10.0,
                    baseline_column=None,
                    cv_splits=cfg.lgbm.cv_splits,
                )
                stack_train = splits.train.copy()
                stack_val = splits.val.copy()
                stack_test = splits.test.copy()
                stack_train["ridge_trend_pred"] = ridge_stack["train_oof_pred"]
                stack_val["ridge_trend_pred"] = ridge_stack["val_pred"]
                stack_test["ridge_trend_pred"] = ridge_stack["test_pred"]
                stack_features = features + ["ridge_trend_pred"]
                stack_result = train_lightgbm_models(
                    train_df=stack_train,
                    val_df=stack_val,
                    test_df=stack_test,
                    feature_columns=stack_features,
                    model_output_dir=cfg.long.lgbm_model_output_dir / f"h{horizon_days}_stack",
                    prediction_output_dir=prediction_dir / f"h{horizon_days}",
                    baseline_column="ridge_trend_pred",
                    model_name=f"ridge_stack_{horizon_days}d",
                    prediction_filename=f"ridge_stack_{horizon_days}d_predictions.csv",
                    quantiles_enabled=cfg.lgbm.quantiles_enabled,
                    lower_alpha=cfg.lgbm.lower_alpha,
                    upper_alpha=cfg.lgbm.upper_alpha,
                    learning_rate=cfg.lgbm.learning_rate,
                    n_estimators=cfg.lgbm.n_estimators,
                    max_depth=cfg.lgbm.max_depth,
                    num_leaves=cfg.lgbm.num_leaves,
                    min_data_in_leaf=cfg.lgbm.min_data_in_leaf,
                    lambda_l1=cfg.lgbm.lambda_l1,
                    lambda_l2=cfg.lgbm.lambda_l2,
                    early_stopping_rounds=cfg.lgbm.early_stopping_rounds,
                    cv_splits=cfg.lgbm.cv_splits,
                )
                stack_frame = _load_prediction_frame(stack_result.prediction_path)
                horizon_prediction_frames["ridge_residual_stack"] = stack_frame
                horizon_lgb_meta["ridge_residual_stack"] = {
                    "feature_columns": stack_features,
                    "baseline_column": "ridge_trend_pred",
                    "params": base_lgb_params,
                }
                candidate_rows.append(
                    {
                        "horizon_days": horizon_days,
                        "candidate": "ridge_residual_stack",
                        "baseline_name": baseline_name,
                        **_candidate_metrics(stack_frame),
                    }
                )

        if "lightgbm" in horizon_prediction_frames and "ridge" in horizon_prediction_frames:
            blend_weight = 0.5 if horizon_days == 7 else 0.7
            blend_frame = _blend_prediction_frames(
                horizon_prediction_frames["lightgbm"],
                horizon_prediction_frames["ridge"],
                weight_left=blend_weight,
            )
            blend_name = "mean_blend" if horizon_days == 7 else "weighted_blend"
            blend_path = prediction_dir / f"h{horizon_days}" / f"{blend_name}_{horizon_days}d_predictions.csv"
            _write_prediction_frame(blend_frame, blend_path)
            horizon_prediction_frames[blend_name] = blend_frame
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": blend_name,
                    "baseline_name": baseline_name,
                    "lightgbm_weight": blend_weight,
                    **_candidate_metrics(blend_frame),
                }
            )

        if horizon_days in {30, 90} and {"ridge", "lightgbm", "lightgbm_weighted_huber", "multi_quantile_p50"}.issubset(
            horizon_oof_meta
        ):
            train_meta_features = {
                "ridge_pred": horizon_oof_meta["ridge"]["train_oof_pred"],
                "lightgbm_pred": horizon_oof_meta["lightgbm"]["train_oof_pred"],
                "weighted_huber_pred": horizon_oof_meta["lightgbm_weighted_huber"]["train_oof_pred"],
                "p50_pred": horizon_oof_meta["multi_quantile_p50"]["train_oof_pred"],
                "recent_return_std_7": splits.train["recent_return_std_7"].to_numpy(dtype=float),
                "price_change_30": splits.train["price_change_30"].to_numpy(dtype=float),
            }
            val_meta_features = {
                "ridge_pred": horizon_oof_meta["ridge"]["val_pred"],
                "lightgbm_pred": horizon_oof_meta["lightgbm"]["val_pred"],
                "weighted_huber_pred": horizon_oof_meta["lightgbm_weighted_huber"]["val_pred"],
                "p50_pred": horizon_oof_meta["multi_quantile_p50"]["val_pred"],
                "recent_return_std_7": splits.val["recent_return_std_7"].to_numpy(dtype=float),
                "price_change_30": splits.val["price_change_30"].to_numpy(dtype=float),
            }
            test_meta_features = {
                "ridge_pred": horizon_oof_meta["ridge"]["test_pred"],
                "lightgbm_pred": horizon_oof_meta["lightgbm"]["test_pred"],
                "weighted_huber_pred": horizon_oof_meta["lightgbm_weighted_huber"]["test_pred"],
                "p50_pred": horizon_oof_meta["multi_quantile_p50"]["test_pred"],
                "recent_return_std_7": splits.test["recent_return_std_7"].to_numpy(dtype=float),
                "price_change_30": splits.test["price_change_30"].to_numpy(dtype=float),
            }
            stacking_frame = _stacking_candidate_frame(
                train_df=splits.train,
                val_df=splits.val,
                test_df=splits.test,
                train_meta_features=train_meta_features,
                val_meta_features=val_meta_features,
                test_meta_features=test_meta_features,
            )
            stacking_path = prediction_dir / f"h{horizon_days}" / f"stacking_{horizon_days}d_predictions.csv"
            _write_prediction_frame(stacking_frame, stacking_path)
            horizon_prediction_frames["stacking_ridge_meta"] = stacking_frame
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": "stacking_ridge_meta",
                    "baseline_name": baseline_name,
                    **_candidate_metrics(stacking_frame),
                }
            )

        if horizon_days in {30, 90} and "ridge" in horizon_prediction_frames:
            lgb_like_candidates = [
                row
                for row in candidate_rows
                if int(row["horizon_days"]) == horizon_days
                and str(row["candidate"]) in {
                    "lightgbm",
                    "lightgbm_lowreg",
                    "lightgbm_weighted_huber",
                    "multi_quantile_p50",
                }
            ]
            if lgb_like_candidates:
                best_lgb_like = min(lgb_like_candidates, key=lambda row: float(row["val_mae"]))
                lgb_name = str(best_lgb_like["candidate"])
                dynamic_frame, bucket_weights = _dynamic_blend_frame(
                    left_frame=horizon_prediction_frames[lgb_name],
                    right_frame=horizon_prediction_frames["ridge"],
                    train_volatility=splits.train["roll_std_30"],
                    eval_meta=eval_meta,
                )
                dynamic_path = prediction_dir / f"h{horizon_days}" / f"dynamic_blend_{horizon_days}d_predictions.csv"
                _write_prediction_frame(dynamic_frame, dynamic_path)
                horizon_prediction_frames["dynamic_blend"] = dynamic_frame
                candidate_rows.append(
                    {
                        "horizon_days": horizon_days,
                        "candidate": "dynamic_blend",
                        "baseline_name": baseline_name,
                        "low_vol_weight": bucket_weights.get(0, 0.5),
                        "mid_vol_weight": bucket_weights.get(1, 0.5),
                        "high_vol_weight": bucket_weights.get(2, 0.5),
                        "blend_source": lgb_name,
                        **_candidate_metrics(dynamic_frame),
                    }
                )

        horizon_candidates = [
            row for row in candidate_rows if int(row["horizon_days"]) == horizon_days
        ]
        preferred_order = [
            "stacking_ridge_meta",
            "dynamic_blend",
            "multi_quantile_p50",
            "lightgbm_weighted_huber",
            "lightgbm_diff",
            "ridge_residual_stack",
            "lightgbm_lowreg",
            "lightgbm_shifted",
            "weighted_blend",
            "mean_blend",
            "lightgbm",
            "ridge_bias_calibrated",
            "ridge",
        ]
        selected_candidate = _choose_best_candidate(
            horizon_candidates,
            preferred_order=preferred_order,
        )
        selected_name = str(selected_candidate["candidate"])
        selected_frame = horizon_prediction_frames.get(selected_name)
        if selected_name in horizon_lgb_meta and horizon_days in {30, 90} and selected_frame is not None:
            meta = horizon_lgb_meta[selected_name]
            oof_residual_frame = _lgb_oof_residual_frame(
                train_df=splits.train,
                feature_columns=list(meta["feature_columns"]),  # type: ignore[arg-type]
                baseline_column=meta["baseline_column"],  # type: ignore[arg-type]
                params=meta["params"],  # type: ignore[arg-type]
                cv_splits=cfg.lgbm.cv_splits,
                sample_weight=meta.get("sample_weight"),  # type: ignore[arg-type]
                point_objective=str(meta.get("point_objective", "regression")),
                point_alpha=meta.get("point_alpha"),  # type: ignore[arg-type]
            )
            thresholds, adaptive_offsets = _adaptive_offsets(
                oof_residual_frame,
                use_split=False,
            )
            adaptive_frame = _apply_adaptive_offsets(
                selected_frame,
                eval_meta,
                thresholds=thresholds,
                offsets=adaptive_offsets,
            )
            adaptive_metrics = {
                **_eval_split_metrics(adaptive_frame, split="val"),
                **_eval_split_metrics(adaptive_frame, split="test"),
            }
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": f"{selected_name}_adaptive_conformal",
                    "baseline_name": baseline_name,
                    **adaptive_metrics,
                }
            )
            split_thresholds, split_offsets = _adaptive_offsets(
                oof_residual_frame,
                use_split=True,
            )
            split_frame = _apply_adaptive_offsets(
                selected_frame,
                eval_meta,
                thresholds=split_thresholds,
                offsets=split_offsets,
            )
            split_metrics = {
                **_eval_split_metrics(split_frame, split="val"),
                **_eval_split_metrics(split_frame, split="test"),
            }
            candidate_rows.append(
                {
                    "horizon_days": horizon_days,
                    "candidate": f"{selected_name}_split_conformal",
                    "baseline_name": baseline_name,
                    **split_metrics,
                }
            )

            interval_candidates = [
                row
                for row in candidate_rows
                if int(row["horizon_days"]) == horizon_days
                and str(row["candidate"]).startswith(f"{selected_name}_")
            ]
            interval_frames = {
                f"{selected_name}_adaptive_conformal": adaptive_frame,
                f"{selected_name}_split_conformal": split_frame,
            }
            if horizon_days == 90:
                evt_tail = _fit_evt_tail_quantiles(oof_residual_frame)
                if evt_tail is not None:
                    evt_frame = _apply_evt_tail_adjustment(
                        adaptive_frame,
                        eval_meta,
                        thresholds=thresholds,
                        evt_tail=evt_tail,
                    )
                    evt_metrics = {
                        **_eval_split_metrics(evt_frame, split="val"),
                        **_eval_split_metrics(evt_frame, split="test"),
                    }
                    candidate_rows.append(
                        {
                            "horizon_days": 90,
                            "candidate": f"{selected_name}_evt_conformal",
                            "baseline_name": baseline_name,
                            "evt_lower_q": evt_tail[0],
                            "evt_upper_q": evt_tail[1],
                            **evt_metrics,
                        }
                    )
                    interval_candidates.append(candidate_rows[-1])
                    interval_frames[f"{selected_name}_evt_conformal"] = evt_frame

            best_interval_candidate = _choose_best_candidate(
                interval_candidates,
                preferred_order=[
                    f"{selected_name}_evt_conformal",
                    f"{selected_name}_adaptive_conformal",
                    f"{selected_name}_split_conformal",
                ],
            )
            selected_name = str(best_interval_candidate["candidate"])
            selected_candidate = best_interval_candidate
            selected_frame = interval_frames[selected_name]

        for row in candidate_rows:
            if int(row["horizon_days"]) == horizon_days:
                row["selected"] = int(str(row["candidate"]) == selected_name)

        if selected_frame is not None:
            selected_path = prediction_dir / f"h{horizon_days}" / f"selected_{horizon_days}d_predictions.csv"
            _write_prediction_frame(selected_frame, selected_path)

        selection_metadata[f"{horizon_days}d"] = {
            "selected_candidate": selected_name,
            "val_mae": float(selected_candidate["val_mae"]),
            "test_mae": float(selected_candidate["test_mae"]),
        }
        summary[f"selected_{horizon_days}d"] = {
            "model": selected_name,
            "baseline_name": baseline_name,
            "val_mae": float(selected_candidate["val_mae"]),
            "val_rmse": float(selected_candidate.get("val_rmse", float("nan"))),
            "test_mae": float(selected_candidate["test_mae"]),
            "test_rmse": float(selected_candidate.get("test_rmse", float("nan"))),
        }
        if "test_picp" in selected_candidate:
            summary[f"selected_{horizon_days}d"]["test_picp"] = float(
                selected_candidate["test_picp"]
            )
        if "test_interval_width" in selected_candidate:
            summary[f"selected_{horizon_days}d"]["test_interval_width"] = float(
                selected_candidate["test_interval_width"]
            )
        comparison_rows.append(
            {
                "horizon_days": horizon_days,
                "model": f"selected_{horizon_days}d",
                "baseline_name": baseline_name,
                "test_mae": float(selected_candidate["test_mae"]),
                "baseline_mae": baseline_mae,
                "mae_ratio": float(selected_candidate["test_mae"]) / baseline_mae,
            }
        )

    summary["long_selection"] = selection_metadata
    cfg.long.summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.long.summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison_frame = pd.DataFrame(comparison_rows).sort_values(
        ["horizon_days", "model"]
    )
    comparison_frame.to_csv(cfg.long.comparison_output_path, index=False)
    pd.DataFrame(candidate_rows).sort_values(["horizon_days", "candidate"]).to_csv(
        candidate_output_path,
        index=False,
    )

    print(f"Saved long feature table to: {cfg.long.feature_output_path}")
    print(f"Saved long model summary to: {cfg.long.summary_output_path}")
    print(f"Saved long comparison table to: {cfg.long.comparison_output_path}")
    print(f"Saved long candidate table to: {candidate_output_path}")
    for name, metrics in summary.items():
        if "test_mae" in metrics:
            print(
                f"{name}: MAE={metrics.get('test_mae', float('nan')):.4f}, RMSE={metrics.get('test_rmse', float('nan')):.4f}"
            )
    return summary


def run(
    config_path: Path,
    *,
    pipeline: str | None,
    model: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    device: str | None,
) -> dict[str, dict[str, float | int | str]]:
    cfg = _apply_overrides(
        load_config(config_path),
        pipeline=pipeline,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    cleaned = _prepare_cleaned(cfg)
    if cfg.run.pipeline == "long":
        return _run_long_pipeline(cfg, cleaned)
    return _run_short_pipeline(cfg, cleaned)


def main() -> None:
    args = parse_args()
    run(
        args.config,
        pipeline=args.pipeline,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
