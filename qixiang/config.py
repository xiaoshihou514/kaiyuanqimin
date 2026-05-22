from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class WeatherConfig:
    start_date: str
    end_date: str
    output_path: Path
    cache_dir: Path
    cities: list[City]


def load_config(config_path: Path) -> WeatherConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    weather = raw.get("weather", {})
    cities_raw = raw.get("cities", [])
    if not cities_raw:
        raise ValueError("No cities configured in TOML.")

    cities = [
        City(
            name=str(city["name"]),
            latitude=float(city["latitude"]),
            longitude=float(city["longitude"]),
        )
        for city in cities_raw
    ]

    return WeatherConfig(
        start_date=str(weather.get("start_date", "2020-01-01")),
        end_date=str(weather.get("end_date", "2026-05-21")),
        output_path=Path(str(weather.get("output_path", "data/weather.csv"))),
        cache_dir=Path(str(weather.get("cache_dir", ".cache/openmeteo"))),
        cities=cities,
    )
