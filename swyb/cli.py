from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import time

import pandas as pd

from .config import SwybConfig, load_config
from .fetch import fetch_for_date, iter_dates
from .transform import OUTPUT_COLUMNS, response_to_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch SWYB cucumber market prices.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("swyb/config.toml"),
        help="Path to SWYB config TOML.",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Override start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="Override end date YYYY-MM-DD.")
    parser.add_argument("--cate-id", type=int, default=None, help="Override commodity cateId.")
    parser.add_argument("--output-path", type=Path, default=None, help="Override output CSV path.")
    parser.add_argument(
        "--failure-policy",
        type=str,
        choices=["fail_fast", "partial_with_report"],
        default=None,
        help="Failure behavior when some dates cannot be fetched.",
    )
    return parser.parse_args()


def _effective_config(
    cfg: SwybConfig,
    start_date: str | None,
    end_date: str | None,
    cate_id: int | None,
    output_path: Path | None,
    failure_policy: str | None,
) -> SwybConfig:
    return SwybConfig(
        start_date=start_date or cfg.start_date,
        end_date=end_date or cfg.end_date,
        cate_id=cate_id or cfg.cate_id,
        output_path=output_path or cfg.output_path,
        timeout_seconds=cfg.timeout_seconds,
        max_attempts=cfg.max_attempts,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
        request_interval_seconds=cfg.request_interval_seconds,
        failure_policy=failure_policy or cfg.failure_policy,
        endpoint_url=cfg.endpoint_url,
        referer_url=cfg.referer_url,
        user_agent=cfg.user_agent,
    )


def run(
    config_path: Path,
    start_date: str | None,
    end_date: str | None,
    cate_id: int | None,
    output_path: Path | None,
    failure_policy: str | None,
) -> Path:
    base_cfg = load_config(config_path)
    cfg = _effective_config(base_cfg, start_date, end_date, cate_id, output_path, failure_policy)

    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    all_dates = iter_dates(start, end)

    frames: list[pd.DataFrame] = []
    failed_dates: list[str] = []
    empty_dates: list[str] = []

    for idx, day in enumerate(all_dates):
        date_str = day.isoformat()
        try:
            response = fetch_for_date(
                endpoint_url=cfg.endpoint_url,
                referer_url=cfg.referer_url,
                user_agent=cfg.user_agent,
                cate_id=cfg.cate_id,
                search_date=day,
                timeout_seconds=cfg.timeout_seconds,
                max_attempts=cfg.max_attempts,
                retry_backoff_seconds=cfg.retry_backoff_seconds,
            )
            frame = response_to_frame(response, fallback_date=date_str)
            if frame.empty:
                empty_dates.append(date_str)
            else:
                frames.append(frame)
        except RuntimeError as exc:
            if cfg.failure_policy == "partial_with_report":
                failed_dates.append(date_str)
                print(f"[WARN] {date_str}: {exc}")
            else:
                raise

        if idx < len(all_dates) - 1:
            time.sleep(cfg.request_interval_seconds)

    if frames:
        output_df = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "market", "report_date"], kind="stable"
        )
    else:
        output_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(cfg.output_path, index=False)

    print(
        f"Saved SWYB dataset to: {cfg.output_path} "
        f"(rows={len(output_df)}, failed_dates={len(failed_dates)}, empty_dates={len(empty_dates)})"
    )
    if failed_dates:
        print("Failed dates:", ",".join(failed_dates))
    if empty_dates:
        print("Empty dates:", ",".join(empty_dates))

    return cfg.output_path


def main() -> None:
    args = parse_args()
    run(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        cate_id=args.cate_id,
        output_path=args.output_path,
        failure_policy=args.failure_policy,
    )


if __name__ == "__main__":
    main()
