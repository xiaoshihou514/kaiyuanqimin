from __future__ import annotations

import pandas as pd


def hourly_to_city_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["date"] = frame["timestamp"].dt.date
    daily = (
        frame.groupby(["city", "date"], as_index=False)
        .agg(
            temp_avg=("temperature_2m", "mean"),
            precip=("rain", "sum"),
        )
        .sort_values(["date", "city"])
    )
    return daily


def province_plus_city(city_daily: pd.DataFrame) -> pd.DataFrame:
    province_daily = (
        city_daily.groupby("date", as_index=False)
        .agg(temp_avg=("temp_avg", "mean"), precip=("precip", "mean"))
        .sort_values("date")
    )

    temp_wide = city_daily.pivot(
        index="date", columns="city", values="temp_avg"
    ).add_prefix("temp_avg_")
    precip_wide = city_daily.pivot(
        index="date", columns="city", values="precip"
    ).add_prefix("precip_")
    city_wide = temp_wide.join(precip_wide)

    merged = province_daily.set_index("date").join(city_wide, how="left").reset_index()
    merged["date"] = merged["date"].astype(str)
    return merged
