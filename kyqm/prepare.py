from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cpca
import pandas as pd

FEATURE_COLUMNS = ["local_price", "temp_avg", "precip", "sentiment_score"]
DIRECT_ADMIN_CITIES = {"北京", "天津", "上海", "重庆", "香港", "澳门"}


@dataclass(frozen=True)
class PrepareParams:
    market_prices_path: Path
    weather_path: Path
    output_path: Path
    province_name: str
    city_name: str | None
    product_name: str
    start_date: str
    end_date: str
    max_forward_fill_days: int
    outlier_sigma: float


def _parse_cpca_components(series: pd.Series) -> pd.DataFrame:
    parsed = cpca.transform(
        series.fillna("").astype(str).tolist(),
        pos_sensitive=False,
    )
    province = parsed["省"].fillna("").astype(str).map(_normalize_province_name)
    city = parsed["市"].fillna("").astype(str).map(_normalize_city_name)
    city = city.where(city != "", province.where(province.isin(DIRECT_ADMIN_CITIES), ""))
    return pd.DataFrame({"province": province, "city": city}, index=series.index)


def _resolve_market_regions(frame: pd.DataFrame) -> pd.DataFrame:
    county_parsed = _parse_cpca_components(frame["county_name"])
    market_parsed = _parse_cpca_components(frame["market"])

    county_province = county_parsed["province"]
    county_city = county_parsed["city"]
    market_province = market_parsed["province"]
    market_city = market_parsed["city"]

    resolved_province = pd.Series("", index=frame.index, dtype="object")
    province_source = pd.Series("unresolved", index=frame.index, dtype="object")

    province_agree = (
        county_province.ne("")
        & market_province.ne("")
        & county_province.eq(market_province)
    )
    resolved_province.loc[province_agree] = county_province.loc[province_agree]
    province_source.loc[province_agree] = "county+market_cpca"

    county_only = resolved_province.eq("") & county_province.ne("")
    resolved_province.loc[county_only] = county_province.loc[county_only]
    province_source.loc[county_only] = "county_cpca"

    market_only = resolved_province.eq("") & market_province.ne("")
    resolved_province.loc[market_only] = market_province.loc[market_only]
    province_source.loc[market_only] = "market_cpca"

    resolved_city = pd.Series("", index=frame.index, dtype="object")
    city_source = pd.Series("unresolved", index=frame.index, dtype="object")

    market_city_valid = market_city.ne("") & (
        market_province.eq("") | market_province.eq(resolved_province)
    )
    resolved_city.loc[market_city_valid] = market_city.loc[market_city_valid]
    city_source.loc[market_city_valid] = "market_cpca"

    county_city_valid = (
        resolved_city.eq("")
        & county_city.ne("")
        & (county_province.eq("") | county_province.eq(resolved_province))
    )
    resolved_city.loc[county_city_valid] = county_city.loc[county_city_valid]
    city_source.loc[county_city_valid] = "county_cpca"

    direct_admin_city = resolved_city.eq("") & resolved_province.isin(DIRECT_ADMIN_CITIES)
    resolved_city.loc[direct_admin_city] = resolved_province.loc[direct_admin_city]
    city_source.loc[direct_admin_city] = "direct_admin_province"

    return pd.DataFrame(
        {
            "province_cpca_county": county_province,
            "city_cpca_county": county_city,
            "province_cpca_market": market_province,
            "city_cpca_market": market_city,
            "resolved_province": resolved_province,
            "resolved_city": resolved_city,
            "province_resolution_source": province_source,
            "city_resolution_source": city_source,
        },
        index=frame.index,
    )


def _normalize_province_name(name: str) -> str:
    cleaned = str(name).strip()
    alias_map = {
        "新疆生产建设兵团": "新疆",
        "新疆兵团": "新疆",
        "内蒙古": "内蒙古",
        "广西": "广西",
        "宁夏": "宁夏",
        "新疆": "新疆",
        "西藏": "西藏",
        "香港": "香港",
        "澳门": "澳门",
    }
    if cleaned in alias_map:
        return alias_map[cleaned]
    for suffix in (
        "维吾尔自治区",
        "回族自治区",
        "壮族自治区",
        "特别行政区",
        "自治区",
        "省",
        "市",
    ):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    return alias_map.get(cleaned, cleaned)


def _normalize_city_name(name: str) -> str:
    cleaned = str(name).strip()
    if not cleaned:
        return ""
    for suffix in ("特别行政区", "自治州", "地区", "盟", "市"):
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

    for required_col in ("date", "market", "product", "price", "county_name"):
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
    product_market = market.loc[product_mask].copy()
    region_resolution = _resolve_market_regions(product_market)
    product_market = product_market.join(region_resolution)

    target_province = _normalize_province_name(params.province_name)
    final_mask = product_market["resolved_province"] == target_province
    target_city = None
    if params.city_name:
        target_city = _normalize_city_name(params.city_name)
        final_mask &= product_market["resolved_city"] == target_city

    filtered = product_market.loc[final_mask].copy()
    if filtered.empty:
        location_bits = [f"province={params.province_name}"]
        if params.city_name:
            location_bits.append(f"city={params.city_name}")
        raise ValueError(
            f"No records found for {', '.join(location_bits)}, product={params.product_name}."
        )

    province_resolution_counts = (
        product_market["province_resolution_source"].value_counts().sort_index()
    )
    city_resolution_counts = (
        product_market["city_resolution_source"].value_counts().sort_index()
    )
    province_summary = ", ".join(
        f"{key}={value}" for key, value in province_resolution_counts.items()
    )
    city_summary = ", ".join(
        f"{key}={value}" for key, value in city_resolution_counts.items()
    )
    matched_rows = int(final_mask.sum())
    unresolved_rows = int(product_market["resolved_province"].eq("").sum())
    unresolved_city_rows = int(product_market["resolved_city"].eq("").sum())
    city_part = (
        f", city={params.city_name}, city_resolved_rows={len(product_market) - unresolved_city_rows}"
        if params.city_name
        else f", city_resolved_rows={len(product_market) - unresolved_city_rows}"
    )
    print(
        "Province resolution summary "
        f"(product={params.product_name}, target={params.province_name}): "
        f"matched_rows={matched_rows}, unresolved_rows={unresolved_rows}{city_part}, "
        f"province_sources=[{province_summary}], city_sources=[{city_summary}]"
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
