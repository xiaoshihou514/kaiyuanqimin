from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class DataConfig:
    market_prices_path: Path
    weather_path: Path
    cleaned_output_path: Path
    feature_output_path: Path
    province_name: str
    city_name: str | None
    product_name: str
    start_date: str
    end_date: str
    train_end: str
    val_end: str
    test_end: str
    max_forward_fill_days: int
    outlier_sigma: float


@dataclass(frozen=True)
class LgbmConfig:
    enabled: bool
    learning_rate: float
    n_estimators: int
    max_depth: int
    num_leaves: int
    min_data_in_leaf: int
    lambda_l1: float
    lambda_l2: float
    early_stopping_rounds: int
    quantiles_enabled: bool
    lower_alpha: float
    upper_alpha: float
    model_output_dir: Path


@dataclass(frozen=True)
class GruConfig:
    enabled: bool
    sequence_length: int
    hidden_dim: int
    num_layers: int
    dropout: float
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    weight_decay: float
    grad_clip_norm: float
    quantiles_enabled: bool
    model_output_path: Path
    metrics_output_path: Path
    device: str


@dataclass(frozen=True)
class ProphetConfig:
    enabled: bool
    daily_seasonality: bool
    weekly_seasonality: bool
    yearly_seasonality: bool
    model_output_path: Path


@dataclass(frozen=True)
class RunConfig:
    model: str
    seed: int
    summary_output_path: Path
    prediction_output_dir: Path


@dataclass(frozen=True)
class KyqmConfig:
    data: DataConfig
    lgbm: LgbmConfig
    gru: GruConfig
    prophet: ProphetConfig
    run: RunConfig


def load_config(config_path: Path) -> KyqmConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    data = raw.get("data", {})
    lgbm = raw.get("lgbm", {})
    gru = raw.get("gru", {})
    prophet = raw.get("prophet", {})
    run = raw.get("run", {})

    return KyqmConfig(
        data=DataConfig(
            market_prices_path=Path(
                str(data.get("market_prices_path", "data/market_prices.csv"))
            ),
            weather_path=Path(str(data.get("weather_path", "data/weather.csv"))),
            cleaned_output_path=Path(
                str(data.get("cleaned_output_path", "data/cleaned_data.csv"))
            ),
            feature_output_path=Path(
                str(data.get("feature_output_path", "data/feature_data.csv"))
            ),
            province_name=str(data.get("province_name", "山东省")),
            city_name=_optional_text(data.get("city_name")),
            product_name=str(data.get("product_name", "黄瓜")),
            start_date=str(data.get("start_date", "2020-01-01")),
            end_date=str(data.get("end_date", "2026-05-21")),
            train_end=str(data.get("train_end", "2024-12-31")),
            val_end=str(data.get("val_end", "2025-09-30")),
            test_end=str(data.get("test_end", "2026-05-21")),
            max_forward_fill_days=int(data.get("max_forward_fill_days", 3)),
            outlier_sigma=float(data.get("outlier_sigma", 3.0)),
        ),
        lgbm=LgbmConfig(
            enabled=bool(lgbm.get("enabled", True)),
            learning_rate=float(lgbm.get("learning_rate", 0.03)),
            n_estimators=int(lgbm.get("n_estimators", 300)),
            max_depth=int(lgbm.get("max_depth", 4)),
            num_leaves=int(lgbm.get("num_leaves", 15)),
            min_data_in_leaf=int(lgbm.get("min_data_in_leaf", 20)),
            lambda_l1=float(lgbm.get("lambda_l1", 0.1)),
            lambda_l2=float(lgbm.get("lambda_l2", 0.5)),
            early_stopping_rounds=int(lgbm.get("early_stopping_rounds", 50)),
            quantiles_enabled=bool(lgbm.get("quantiles_enabled", True)),
            lower_alpha=float(lgbm.get("lower_alpha", 0.1)),
            upper_alpha=float(lgbm.get("upper_alpha", 0.9)),
            model_output_dir=Path(
                str(lgbm.get("model_output_dir", "data/models/lgbm"))
            ),
        ),
        gru=GruConfig(
            enabled=bool(gru.get("enabled", True)),
            sequence_length=int(gru.get("sequence_length", 30)),
            hidden_dim=int(gru.get("hidden_dim", 64)),
            num_layers=int(gru.get("num_layers", 2)),
            dropout=float(gru.get("dropout", 0.3)),
            batch_size=int(gru.get("batch_size", 32)),
            epochs=int(gru.get("epochs", 100)),
            learning_rate=float(gru.get("learning_rate", 1e-3)),
            patience=int(gru.get("patience", 15)),
            weight_decay=float(gru.get("weight_decay", 1e-4)),
            grad_clip_norm=float(gru.get("grad_clip_norm", 1.0)),
            quantiles_enabled=bool(gru.get("quantiles_enabled", True)),
            model_output_path=Path(
                str(gru.get("model_output_path", "data/models/gru/model.pth"))
            ),
            metrics_output_path=Path(
                str(gru.get("metrics_output_path", "data/models/gru/metrics.json"))
            ),
            device=str(gru.get("device", "auto")),
        ),
        prophet=ProphetConfig(
            enabled=bool(prophet.get("enabled", True)),
            daily_seasonality=bool(prophet.get("daily_seasonality", True)),
            weekly_seasonality=bool(prophet.get("weekly_seasonality", True)),
            yearly_seasonality=bool(prophet.get("yearly_seasonality", True)),
            model_output_path=Path(
                str(prophet.get("model_output_path", "data/models/prophet/model.pkl"))
            ),
        ),
        run=RunConfig(
            model=str(run.get("model", "all")),
            seed=int(run.get("seed", 42)),
            summary_output_path=Path(
                str(run.get("summary_output_path", "data/model_summary.json"))
            ),
            prediction_output_dir=Path(
                str(run.get("prediction_output_dir", "data/predictions"))
            ),
        ),
    )
