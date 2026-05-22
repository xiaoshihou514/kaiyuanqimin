from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import load_config
from .fetch import build_client, fetch_city_hourly
from .transform import hourly_to_city_daily, province_plus_city


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and aggregate weather data for Shandong.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qixiang/config.toml"),
        help="Path to weather config TOML.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Override start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date YYYY-MM-DD.")
    parser.add_argument("--output-path", type=Path, default=None, help="Override output CSV path.")
    return parser.parse_args()


def run(config_path: Path, start_date: str | None, end_date: str | None, output_path: Path | None) -> Path:
    cfg = load_config(config_path)
    effective_start = start_date or cfg.start_date
    effective_end = end_date or cfg.end_date
    effective_output = output_path or cfg.output_path

    client = build_client(cfg.cache_dir)
    city_daily_frames: list[pd.DataFrame] = []
    for city in cfg.cities:
        hourly = fetch_city_hourly(client, city, effective_start, effective_end)
        city_daily_frames.append(hourly_to_city_daily(hourly))

    city_daily = pd.concat(city_daily_frames, ignore_index=True)
    output_df = province_plus_city(city_daily)

    effective_output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(effective_output, index=False)
    return effective_output


def main() -> None:
    args = parse_args()
    output_path = run(args.config, args.start_date, args.end_date, args.output_path)
    print(f"Saved weather dataset to: {output_path}")


if __name__ == "__main__":
    main()
