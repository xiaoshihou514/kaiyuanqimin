from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cpca
import pandas as pd

FEATURE_COLUMNS = ["local_price", "temp_avg", "precip", "sentiment_score"]


@dataclass(frozen=True)
class PrepareParams:
    market_prices_path: Path
    weather_path: Path
    output_path: Path
    province_name: str
    product_name: str
    start_date: str
    end_date: str
    max_forward_fill_days: int
    outlier_sigma: float


def _province_mask(frame: pd.DataFrame, province_name: str) -> pd.Series:
    county_series = frame["county_name"].fillna("").astype(str)
    parsed = cpca.transform(county_series.tolist(), pos_sensitive=False)
    parsed_province = parsed["省"].fillna("").astype(str).map(_normalize_province_name)
    target = _normalize_province_name(province_name)
    cpca_match = parsed_province == target
    county_fallback = county_series.str.contains(province_name, regex=False)
    return cpca_match | county_fallback


def _normalize_province_name(name: str) -> str:
    cleaned = str(name).strip()
    for suffix in ("省", "市", "自治区", "维吾尔自治区", "回族自治区", "壮族自治区", "特别行政区"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return cleaned


def _clean_local_price(
    series: pd.Series, max_forward_fill_days: int, outlier_sigma: float
) -> pd.Series:
    cleaned = series.copy()
    mean_value = cleaned.mean()
    std_value = cleaned.std(ddof=0)
    if pd.notna(std_value) and std_value > 0:
        outlier_mask = (cleaned - mean_value).abs() > (outlier_sigma * std_value)
        cleaned.loc[outlier_mask] = pd.NA

    cleaned = cleaned.interpolate(limit_direction="both")
    cleaned = cleaned.ffill(limit=max_forward_fill_days)
    return cleaned


def prepare_training_frame(params: PrepareParams) -> pd.DataFrame:
    if not params.market_prices_path.exists():
        raise FileNotFoundError(
            f"{params.market_prices_path} not found. Run `uv run python -m swyb` before training."
        )
    if not params.weather_path.exists():
        raise FileNotFoundError(
            f"{params.weather_path} not found. Run `uv run python -m qixiang` before training."
        )

    market = pd.read_csv(params.market_prices_path)
    weather = pd.read_csv(params.weather_path)

    for required_col in ("date", "product", "price", "area_code", "county_name"):
        if required_col not in market.columns:
            raise ValueError(
                f"market_prices.csv missing required column: {required_col}"
            )
    for required_col in ("date", "temp_avg", "precip"):
        if required_col not in weather.columns:
            raise ValueError(f"weather.csv missing required column: {required_col}")

    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    market = market.dropna(subset=["date"]).copy()
    weather = weather.dropna(subset=["date"]).copy()

    product_mask = market["product"].astype(str) == params.product_name
    province_mask = _province_mask(market, params.province_name)
    filtered = market.loc[product_mask & province_mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No records found for province={params.province_name}, product={params.product_name}."
        )

    filtered["price"] = pd.to_numeric(filtered["price"], errors="coerce")
    filtered = filtered.dropna(subset=["price"])
    if filtered.empty:
        raise ValueError("Filtered records exist but all price values are invalid.")

    local_daily = (
        filtered.groupby("date", as_index=False)
        .agg(local_price=("price", "median"), market_count=("market", "nunique"))
        .sort_values("date")
    )

    weather_daily = (
        weather[["date", "temp_avg", "precip"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )
    merged = pd.merge(local_daily, weather_daily, on="date", how="left")

    date_index = pd.DataFrame(
        {"date": pd.date_range(start=params.start_date, end=params.end_date, freq="D")}
    )
    merged = date_index.merge(merged, on="date", how="left").sort_values("date")

    merged["local_price"] = _clean_local_price(
        merged["local_price"],
        max_forward_fill_days=params.max_forward_fill_days,
        outlier_sigma=params.outlier_sigma,
    )
    merged["temp_avg"] = pd.to_numeric(merged["temp_avg"], errors="coerce").interpolate(
        limit_direction="both"
    )
    merged["precip"] = pd.to_numeric(merged["precip"], errors="coerce").fillna(0.0)
    merged["market_count"] = (
        pd.to_numeric(merged["market_count"], errors="coerce").fillna(0).astype(int)
    )
    merged["sentiment_score"] = 0.0

    cleaned = merged.dropna(subset=["local_price", "temp_avg"]).copy()
    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")
    cleaned = cleaned[
        ["date", "local_price", "temp_avg", "precip", "sentiment_score", "market_count"]
    ]

    params.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(params.output_path, index=False)
    return cleaned
