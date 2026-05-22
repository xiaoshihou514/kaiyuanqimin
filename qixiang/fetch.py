from __future__ import annotations

from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from .config import City

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def build_client(cache_dir: Path) -> openmeteo_requests.Client:
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests_cache.CachedSession(
        str(cache_dir / "weather_cache"), expire_after=-1
    )
    retry_session = retry(session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_city_hourly(
    client: openmeteo_requests.Client,
    city: City,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "rain"],
    }
    responses = client.weather_api(OPEN_METEO_ARCHIVE_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()

    temperature = hourly.Variables(0).ValuesAsNumpy()
    rain = hourly.Variables(1).ValuesAsNumpy()

    date_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    return pd.DataFrame(
        {
            "timestamp": date_index,
            "temperature_2m": temperature,
            "rain": rain,
            "city": city.name,
        }
    )
