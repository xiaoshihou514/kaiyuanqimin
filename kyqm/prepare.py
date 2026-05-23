from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import cpca

FEATURE_COLUMNS = ["local_price", "temp_avg", "precip", "sentiment_score"]
DIRECT_ADMIN_CITIES = {"北京", "天津", "上海", "重庆", "香港", "澳门"}
PRODUCT_ALIASES = {
    "黄瓜": "cucumber",
    "西红柿": "tomato",
    "茄子": "eggplant",
    "大白菜": "cabbage",
    "青椒": "green_pepper",
}
PROVINCE_ALIASES = {
    "北京市": "beijing",
    "北京": "beijing",
    "山东省": "shandong",
    "河北省": "hebei",
    "河南省": "henan",
}
HOLIDAY_DATES = {
    "spring_festival": [
        "2020-01-25",
        "2021-02-12",
        "2022-02-01",
        "2023-01-22",
        "2024-02-10",
        "2025-01-29",
        "2026-02-17",
        "2027-02-06",
    ],
    "mid_autumn": [
        "2020-10-01",
        "2021-09-21",
        "2022-09-10",
        "2023-09-29",
        "2024-09-17",
        "2025-10-06",
        "2026-09-25",
        "2027-09-15",
    ],
    "qingming": [
        "2020-04-04",
        "2021-04-04",
        "2022-04-05",
        "2023-04-05",
        "2024-04-04",
        "2025-04-04",
        "2026-04-05",
        "2027-04-05",
    ],
}


@dataclass(frozen=True)
class PrepareParams:
    market_prices_path: Path
    weather_path: Path
    output_path: Path
    province_name: str
    city_name: str | None
    product_name: str
    companion_products: list[str]
    nearby_cucumber_provinces: list[str]
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


def _slug_product_name(product_name: str) -> str:
    if product_name in PRODUCT_ALIASES:
        return PRODUCT_ALIASES[product_name]
    return (
        product_name.strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .lower()
    )


def _slug_province_name(province_name: str) -> str:
    normalized = _normalize_province_name(province_name)
    if province_name in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[province_name]
    if normalized in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[normalized]
    return normalized.replace(" ", "_").lower()


def _clean_price_series(
    series: pd.Series, max_forward_fill_days: int, outlier_sigma: float
) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce").astype(float)
    mean_value = cleaned.mean()
    std_value = cleaned.std(ddof=0)
    if pd.notna(std_value) and std_value > 0:
        outlier_mask = (cleaned - mean_value).abs() > (outlier_sigma * std_value)
        cleaned.loc[outlier_mask] = pd.NA

    cleaned = cleaned.interpolate(limit_direction="both")
    cleaned = cleaned.ffill(limit=max_forward_fill_days)
    cleaned = cleaned.bfill(limit=max_forward_fill_days)
    return cleaned


def _load_market_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    frame = pd.read_csv(path)
    for required_col in ("date", "market", "product", "price", "county_name"):
        if required_col not in frame.columns:
            raise ValueError(f"{path.name} missing required column: {required_col}")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).copy()


def _find_product_market_file(base_dir: Path, product_name: str) -> Path:
    matches = sorted(base_dir.glob(f"market_prices__*__{product_name}__*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No market price file found for product '{product_name}' in {base_dir}."
        )
    return matches[0]


def _filter_market_rows(
    frame: pd.DataFrame,
    *,
    province_name: str,
    product_name: str,
    city_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    product_mask = frame["product"].astype(str) == product_name
    product_market = frame.loc[product_mask].copy()
    region_resolution = _resolve_market_regions(product_market)
    product_market = product_market.join(region_resolution)

    final_mask = product_market["resolved_province"] == _normalize_province_name(
        province_name
    )
    if city_name:
        final_mask &= product_market["resolved_city"] == _normalize_city_name(city_name)

    return product_market.loc[final_mask].copy(), product_market


def _daily_price_series(
    frame: pd.DataFrame,
    *,
    province_name: str,
    product_name: str,
    city_name: str | None = None,
    value_column: str,
) -> pd.DataFrame:
    filtered, product_market = _filter_market_rows(
        frame,
        province_name=province_name,
        city_name=city_name,
        product_name=product_name,
    )
    if filtered.empty:
        raise ValueError(
            f"No records found for province={province_name}, city={city_name}, product={product_name}."
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
    matched_rows = len(filtered)
    unresolved_rows = int(product_market["resolved_province"].eq("").sum())
    unresolved_city_rows = int(product_market["resolved_city"].eq("").sum())
    city_part = (
        f", city={city_name}, city_resolved_rows={len(product_market) - unresolved_city_rows}"
        if city_name
        else f", city_resolved_rows={len(product_market) - unresolved_city_rows}"
    )
    print(
        "Province resolution summary "
        f"(product={product_name}, target={province_name}): "
        f"matched_rows={matched_rows}, unresolved_rows={unresolved_rows}{city_part}, "
        f"province_sources=[{province_summary}], city_sources=[{city_summary}]"
    )

    filtered["price"] = pd.to_numeric(filtered["price"], errors="coerce")
    filtered = filtered.dropna(subset=["price"])
    if filtered.empty:
        raise ValueError("Filtered records exist but all price values are invalid.")

    series = (
        filtered.groupby("date", as_index=False)
        .agg(
            **{
                value_column: ("price", "median"),
                f"{value_column}_market_count": ("market", "nunique"),
            }
        )
        .sort_values("date")
    )
    return series


def _days_to_next_holiday(series: pd.Series, holiday_name: str) -> pd.Series:
    holiday_dates = pd.to_datetime(HOLIDAY_DATES[holiday_name])
    values: list[int] = []
    for date_value in pd.to_datetime(series):
        future = holiday_dates[holiday_dates >= date_value]
        if len(future) == 0:
            values.append(int((holiday_dates[-1] - date_value).days))
        else:
            values.append(int((future[0] - date_value).days))
    return pd.Series(values, index=series.index, dtype="int64")


def prepare_training_frame(params: PrepareParams) -> pd.DataFrame:
    market = _load_market_frame(params.market_prices_path)
    weather_path = params.weather_path
    if not weather_path.exists():
        raise FileNotFoundError(
            f"{weather_path} not found. Run `uv run python -m qixiang` before training."
        )
    weather = pd.read_csv(weather_path)
    for required_col in ("date", "temp_avg", "precip"):
        if required_col not in weather.columns:
            raise ValueError(f"{weather_path.name} missing required column: {required_col}")
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather = weather.dropna(subset=["date"]).copy()

    local_daily = _daily_price_series(
        market,
        province_name=params.province_name,
        city_name=params.city_name,
        product_name=params.product_name,
        value_column="local_price",
    )
    local_daily = local_daily.rename(
        columns={"local_price_market_count": "market_count"}
    )

    merged = local_daily.copy()
    base_dir = params.market_prices_path.parent
    for product_name in params.companion_products:
        if product_name == params.product_name:
            continue
        product_path = _find_product_market_file(base_dir, product_name)
        product_frame = _load_market_frame(product_path)
        alias = _slug_product_name(product_name)
        product_daily = _daily_price_series(
            product_frame,
            province_name=params.province_name,
            city_name=None,
            product_name=product_name,
            value_column=f"{alias}_price",
        )[["date", f"{alias}_price"]]
        merged = merged.merge(product_daily, on="date", how="left")

    for province_name in params.nearby_cucumber_provinces:
        alias = _slug_province_name(province_name)
        nearby_daily = _daily_price_series(
            market,
            province_name=province_name,
            city_name=None,
            product_name=params.product_name,
            value_column=f"{alias}_cucumber_price",
        )[["date", f"{alias}_cucumber_price"]]
        merged = merged.merge(nearby_daily, on="date", how="left")

    beijing_daily = _daily_price_series(
        market,
        province_name="北京市",
        city_name=None,
        product_name=params.product_name,
        value_column="beijing_cucumber_price",
    )[["date", "beijing_cucumber_price"]]
    merged = merged.merge(beijing_daily, on="date", how="left")

    weather_daily = (
        weather[["date", "temp_avg", "precip"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
    )
    merged = pd.merge(merged, weather_daily, on="date", how="left")

    date_index = pd.DataFrame(
        {"date": pd.date_range(start=params.start_date, end=params.end_date, freq="D")}
    )
    merged = date_index.merge(merged, on="date", how="left").sort_values("date")

    price_columns = [
        "local_price",
        *[
            f"{_slug_product_name(product_name)}_price"
            for product_name in params.companion_products
            if product_name != params.product_name
        ],
        *[
            f"{_slug_province_name(province_name)}_cucumber_price"
            for province_name in params.nearby_cucumber_provinces
        ],
        "beijing_cucumber_price",
    ]
    for column in price_columns:
        if column in merged.columns:
            merged[column] = _clean_price_series(
                merged[column],
                max_forward_fill_days=params.max_forward_fill_days,
                outlier_sigma=params.outlier_sigma,
            )

    merged["temp_avg"] = pd.to_numeric(merged["temp_avg"], errors="coerce").interpolate(
        limit_direction="both"
    )
    merged["precip"] = pd.to_numeric(merged["precip"], errors="coerce").fillna(0.0)
    merged["precip_sum_3d"] = (
        merged["precip"].rolling(window=3, min_periods=1).sum().astype(float)
    )
    merged["precip_sum_7d"] = (
        merged["precip"].rolling(window=7, min_periods=1).sum().astype(float)
    )
    merged["temp_mean_7d"] = (
        merged["temp_avg"].rolling(window=7, min_periods=1).mean().astype(float)
    )
    merged["temp_change_1d"] = merged["temp_avg"].diff().fillna(0.0).astype(float)
    merged["precip_change_1d"] = merged["precip"].diff().fillna(0.0).astype(float)
    merged["extreme_precip_flag"] = (merged["precip"] >= 30.0).astype(int)
    merged["heatwave_3d_flag"] = (
        merged["temp_avg"]
        .rolling(window=3, min_periods=3)
        .apply(lambda values: 1.0 if (values > 32.0).all() else 0.0, raw=True)
        .fillna(0.0)
        .astype(int)
    )
    merged["days_to_next_spring_festival"] = _days_to_next_holiday(
        merged["date"], "spring_festival"
    )
    merged["days_to_next_mid_autumn"] = _days_to_next_holiday(
        merged["date"], "mid_autumn"
    )
    merged["days_to_next_qingming"] = _days_to_next_holiday(
        merged["date"], "qingming"
    )
    merged["market_count"] = (
        pd.to_numeric(merged["market_count"], errors="coerce").fillna(0).astype(int)
    )
    merged["sentiment_score"] = 0.0

    required = ["local_price", "temp_avg"]
    cleaned = merged.dropna(subset=required).copy()
    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")

    ordered_columns = [
        "date",
        "local_price",
        *[
            f"{_slug_product_name(product_name)}_price"
            for product_name in params.companion_products
            if product_name != params.product_name
        ],
        *[
            f"{_slug_province_name(province_name)}_cucumber_price"
            for province_name in params.nearby_cucumber_provinces
        ],
        "beijing_cucumber_price",
        "temp_avg",
        "precip",
        "precip_sum_3d",
        "precip_sum_7d",
        "temp_mean_7d",
        "temp_change_1d",
        "precip_change_1d",
        "extreme_precip_flag",
        "heatwave_3d_flag",
        "days_to_next_spring_festival",
        "days_to_next_mid_autumn",
        "days_to_next_qingming",
        "sentiment_score",
        "market_count",
    ]
    cleaned = cleaned[[column for column in ordered_columns if column in cleaned.columns]]

    params.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(params.output_path, index=False)
    return cleaned
