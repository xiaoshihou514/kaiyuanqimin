from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import KyqmConfig, load_config
from .feature_engineering import build_feature_table, feature_columns, split_by_time
from .model_gru import train_gru_model
from .model_lgb import train_lightgbm_models
from .model_prophet import train_prophet_model
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
        "--model",
        type=str,
        choices=["all", "lgbm", "gru", "prophet"],
        default=None,
        help="Run specific model path.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override GRU epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override GRU batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override GRU learning rate."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override GRU device (cpu/cuda/auto)."
    )
    return parser.parse_args()


def _apply_overrides(
    cfg: KyqmConfig,
    *,
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
        run=cfg.run.__class__(
            model=model or cfg.run.model,
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


def run(
    config_path: Path,
    *,
    model: str | None,
    epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    device: str | None,
) -> dict[str, dict[str, float | int | str]]:
    cfg = _apply_overrides(
        load_config(config_path),
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    cleaned = prepare_training_frame(
        PrepareParams(
            market_prices_path=cfg.data.market_prices_path,
            weather_path=cfg.data.weather_path,
            output_path=cfg.data.cleaned_output_path,
            province_name=cfg.data.province_name,
            city_name=cfg.data.city_name,
            product_name=cfg.data.product_name,
            start_date=cfg.data.start_date,
            end_date=cfg.data.end_date,
            max_forward_fill_days=cfg.data.max_forward_fill_days,
            outlier_sigma=cfg.data.outlier_sigma,
        )
    )
    feature_df = build_feature_table(cleaned)
    cfg.data.feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(cfg.data.feature_output_path, index=False)

    splits = split_by_time(
        feature_df,
        train_end=cfg.data.train_end,
        val_end=cfg.data.val_end,
        test_end=cfg.data.test_end,
    )
    feature_cols = feature_columns(feature_df)

    summary: dict[str, dict[str, float | int | str]] = {}
    run_model = cfg.run.model
    prediction_dir = cfg.run.prediction_output_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if run_model in {"all", "lgbm"} and cfg.lgbm.enabled:
        lgbm_result = train_lightgbm_models(
            train_df=splits.train,
            val_df=splits.val,
            test_df=splits.test,
            feature_columns=feature_cols,
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
        )
        summary["lightgbm"] = lgbm_result.metrics

    if run_model in {"all", "gru"} and cfg.gru.enabled:
        gru_result = train_gru_model(
            train_df=splits.train,
            val_df=splits.val,
            test_df=splits.test,
            feature_columns=feature_cols,
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
        )
        summary["gru_attention"] = gru_result.metrics

    if run_model in {"all", "prophet"} and cfg.prophet.enabled:
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


def main() -> None:
    args = parse_args()
    run(
        args.config,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
