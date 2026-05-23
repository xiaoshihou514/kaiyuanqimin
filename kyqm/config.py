from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: object, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


@dataclass(frozen=True)
class DataConfig:
    market_prices_path: Path
    weather_path: Path
    cleaned_output_path: Path
    feature_output_path: Path
    province_name: str
    city_name: str | None
    product_name: str
    companion_products: list[str]
    nearby_cucumber_provinces: list[str]
    forecast_horizon: int
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
    cv_splits: int
    use_prophet_components: bool
    use_beijing_lead_features: bool
    use_refined_weather_features: bool
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
class LstmConfig:
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
class LongConfig:
    enabled: bool
    horizons: list[int]
    history_window_days: int
    anchor_step_days: int
    augmentation_enabled: bool
    augmentation_jitter_days: int
    feature_output_path: Path
    summary_output_path: Path
    comparison_output_path: Path
    prediction_output_dir: Path
    ridge_model_output_dir: Path
    lgbm_model_output_dir: Path


@dataclass(frozen=True)
class RunConfig:
    model: str
    pipeline: str
    seed: int
    summary_output_path: Path
    prediction_output_dir: Path


@dataclass(frozen=True)
class KyqmConfig:
    data: DataConfig
    lgbm: LgbmConfig
    gru: GruConfig
    lstm: LstmConfig
    prophet: ProphetConfig
    long: LongConfig
    run: RunConfig


def load_config(config_path: Path) -> KyqmConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    data = raw.get("data", {})
    lgbm = raw.get("lgbm", {})
    gru = raw.get("gru", {})
    lstm = raw.get("lstm", {})
    prophet = raw.get("prophet", {})
    long = raw.get("long", {})
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
            companion_products=_string_list(
                data.get("companion_products"),
                ["西红柿", "茄子", "大白菜", "青椒"],
            ),
            nearby_cucumber_provinces=_string_list(
                data.get("nearby_cucumber_provinces"),
                ["河北省", "河南省"],
            ),
            forecast_horizon=int(data.get("forecast_horizon", 1)),
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
            learning_rate=float(lgbm.get("learning_rate", 0.01)),
            n_estimators=int(lgbm.get("n_estimators", 200)),
            max_depth=int(lgbm.get("max_depth", 3)),
            num_leaves=int(lgbm.get("num_leaves", 7)),
            min_data_in_leaf=int(lgbm.get("min_data_in_leaf", 40)),
            lambda_l1=float(lgbm.get("lambda_l1", 0.2)),
            lambda_l2=float(lgbm.get("lambda_l2", 1.0)),
            early_stopping_rounds=int(lgbm.get("early_stopping_rounds", 50)),
            quantiles_enabled=bool(lgbm.get("quantiles_enabled", True)),
            lower_alpha=float(lgbm.get("lower_alpha", 0.1)),
            upper_alpha=float(lgbm.get("upper_alpha", 0.9)),
            cv_splits=int(lgbm.get("cv_splits", 5)),
            use_prophet_components=bool(lgbm.get("use_prophet_components", False)),
            use_beijing_lead_features=bool(
                lgbm.get("use_beijing_lead_features", False)
            ),
            use_refined_weather_features=bool(
                lgbm.get("use_refined_weather_features", False)
            ),
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
        lstm=LstmConfig(
            enabled=bool(lstm.get("enabled", True)),
            sequence_length=int(lstm.get("sequence_length", 30)),
            hidden_dim=int(lstm.get("hidden_dim", 64)),
            num_layers=int(lstm.get("num_layers", 2)),
            dropout=float(lstm.get("dropout", 0.3)),
            batch_size=int(lstm.get("batch_size", 32)),
            epochs=int(lstm.get("epochs", 100)),
            learning_rate=float(lstm.get("learning_rate", 1e-3)),
            patience=int(lstm.get("patience", 15)),
            weight_decay=float(lstm.get("weight_decay", 1e-4)),
            grad_clip_norm=float(lstm.get("grad_clip_norm", 1.0)),
            model_output_path=Path(
                str(lstm.get("model_output_path", "data/models/lstm/model.pth"))
            ),
            metrics_output_path=Path(
                str(lstm.get("metrics_output_path", "data/models/lstm/metrics.json"))
            ),
            device=str(lstm.get("device", "auto")),
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
        long=LongConfig(
            enabled=bool(long.get("enabled", True)),
            horizons=[int(value) for value in long.get("horizons", [7, 30, 90])],
            history_window_days=int(long.get("history_window_days", 90)),
            anchor_step_days=int(long.get("anchor_step_days", 7)),
            augmentation_enabled=bool(long.get("augmentation_enabled", True)),
            augmentation_jitter_days=int(long.get("augmentation_jitter_days", 3)),
            feature_output_path=Path(
                str(long.get("feature_output_path", "data/feature_data_long.csv"))
            ),
            summary_output_path=Path(
                str(long.get("summary_output_path", "data/model_summary_long.json"))
            ),
            comparison_output_path=Path(
                str(long.get("comparison_output_path", "data/model_comparison_long.csv"))
            ),
            prediction_output_dir=Path(
                str(long.get("prediction_output_dir", "data/predictions/long"))
            ),
            ridge_model_output_dir=Path(
                str(long.get("ridge_model_output_dir", "data/models/long/ridge"))
            ),
            lgbm_model_output_dir=Path(
                str(long.get("lgbm_model_output_dir", "data/models/long/lgbm"))
            ),
        ),
        run=RunConfig(
            model=str(run.get("model", "all")),
            pipeline=str(run.get("pipeline", "short")),
            seed=int(run.get("seed", 42)),
            summary_output_path=Path(
                str(run.get("summary_output_path", "data/model_summary.json"))
            ),
            prediction_output_dir=Path(
                str(run.get("prediction_output_dir", "data/predictions"))
            ),
        ),
    )
