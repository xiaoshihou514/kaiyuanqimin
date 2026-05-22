from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import random
import re
import time

import pandas as pd
from tqdm import tqdm

from .config import SwybConfig, load_config
from .fetch import (
    fetch_daily_category_map,
    fetch_for_date,
    iter_dates,
    resolve_category_to_cate_id,
)
from .transform import OUTPUT_COLUMNS, response_to_frame

MIN_BASE_INTERVAL_SECONDS = 0.2
MAX_JITTER_SECONDS = 0.7
BLOCK_COOLDOWN_STEP_SECONDS = 2.0
MAX_BLOCK_COOLDOWN_SECONDS = 12.0
ALL_MODE_MAX_WORKERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SWYB market prices by category."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("swyb/config.toml"),
        help="Path to SWYB config TOML.",
    )
    parser.add_argument(
        "--start-date", type=str, default=None, help="Override start date YYYY-MM-DD."
    )
    parser.add_argument(
        "--end-date", type=str, default=None, help="Override end date YYYY-MM-DD."
    )
    parser.add_argument(
        "--cate-id", type=int, default=None, help="Use explicit commodity cateId."
    )
    parser.add_argument(
        "--category-path",
        type=str,
        default=None,
        help="Category selector path, e.g. '蔬菜-黄瓜' or '粮油-面粉'.",
    )
    parser.add_argument(
        "--category-group",
        type=str,
        default=None,
        help="Category group, e.g. 蔬菜 or 粮油.",
    )
    parser.add_argument(
        "--category-name",
        type=str,
        default=None,
        help="Category name, e.g. 黄瓜 or 面粉.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all daily categories to separate CSV files.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=None, help="Override output CSV path."
    )
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
    category_group: str | None,
    category_name: str | None,
    category_path: str | None,
    output_path: Path | None,
    failure_policy: str | None,
) -> SwybConfig:
    selector_provided = any([category_path, category_group, category_name])
    effective_cate_id = (
        cate_id if cate_id is not None else (None if selector_provided else cfg.cate_id)
    )
    return SwybConfig(
        start_date=start_date or cfg.start_date,
        end_date=end_date or cfg.end_date,
        cate_id=effective_cate_id,
        category_group=category_group or cfg.category_group,
        category_name=category_name or cfg.category_name,
        category_path=category_path or cfg.category_path,
        output_path=output_path or cfg.output_path,
        timeout_seconds=cfg.timeout_seconds,
        max_attempts=cfg.max_attempts,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
        request_interval_seconds=cfg.request_interval_seconds,
        failure_policy=failure_policy or cfg.failure_policy,
        endpoint_url=cfg.endpoint_url,
        category_data_url=cfg.category_data_url,
        referer_url=cfg.referer_url,
        user_agent=cfg.user_agent,
    )


def _resolve_cate_id(cfg: SwybConfig) -> int:
    if cfg.cate_id is not None:
        return cfg.cate_id
    category_map = fetch_daily_category_map(
        category_data_url=cfg.category_data_url,
        referer_url=cfg.referer_url,
        user_agent=cfg.user_agent,
        timeout_seconds=cfg.timeout_seconds,
        max_attempts=cfg.max_attempts,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
    )
    return resolve_category_to_cate_id(
        category_map=category_map,
        category_group=cfg.category_group,
        category_name=cfg.category_name,
        category_path=cfg.category_path,
    )


def _fetch_single_category(
    cfg: SwybConfig, *, cate_id: int, all_dates: list[date]
) -> tuple[pd.DataFrame, int, int]:
    frames: list[pd.DataFrame] = []
    failed_dates = 0
    empty_dates = 0
    block_cooldown_seconds = 0.0

    progress = tqdm(
        all_dates, desc=f"SWYB {cate_id}", unit="day", dynamic_ncols=True, leave=False
    )
    for idx, day in enumerate(progress):
        date_str = day.isoformat()
        try:
            response = fetch_for_date(
                endpoint_url=cfg.endpoint_url,
                referer_url=cfg.referer_url,
                user_agent=cfg.user_agent,
                cate_id=cate_id,
                search_date=day,
                timeout_seconds=cfg.timeout_seconds,
                max_attempts=cfg.max_attempts,
                retry_backoff_seconds=cfg.retry_backoff_seconds,
            )
            frame = response_to_frame(response, fallback_date=date_str)
            if frame.empty:
                empty_dates += 1
            else:
                frames.append(frame)
            block_cooldown_seconds = max(0.0, block_cooldown_seconds - 0.8)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "anti-scrape" in message or "block response signature" in message:
                block_cooldown_seconds = min(
                    MAX_BLOCK_COOLDOWN_SECONDS,
                    block_cooldown_seconds + BLOCK_COOLDOWN_STEP_SECONDS,
                )
            if cfg.failure_policy == "partial_with_report":
                failed_dates += 1
                print(f"[WARN] cate_id={cate_id} date={date_str}: {exc}")
            else:
                raise

        progress.set_postfix(
            failed=failed_dates,
            empty=empty_dates,
            cooldown=f"{block_cooldown_seconds:.1f}s",
        )
        if idx < len(all_dates) - 1:
            base_sleep = max(cfg.request_interval_seconds, MIN_BASE_INTERVAL_SECONDS)
            jitter_sleep = random.uniform(0.0, MAX_JITTER_SECONDS)
            total_sleep = base_sleep + jitter_sleep + block_cooldown_seconds
            time.sleep(total_sleep)

    if frames:
        output_df = pd.concat(frames, ignore_index=True).sort_values(
            ["date", "market", "report_date"], kind="stable"
        )
    else:
        output_df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    return output_df, failed_dates, empty_dates


def _safe_filename(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]", "_", normalized)


def _resolve_all_output_path(
    output_path: Path, group: str, category_name: str, cate_id: int
) -> Path:
    if output_path.suffix.lower() == ".csv":
        out_dir = output_path.parent
        base_name = output_path.stem
    else:
        out_dir = output_path
        base_name = "market_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    return (
        out_dir
        / f"{base_name}__{_safe_filename(group)}__{_safe_filename(category_name)}__{cate_id}.csv"
    )


def _run_all_categories(cfg: SwybConfig, all_dates: list[date]) -> Path:
    category_map = fetch_daily_category_map(
        category_data_url=cfg.category_data_url,
        referer_url=cfg.referer_url,
        user_agent=cfg.user_agent,
        timeout_seconds=cfg.timeout_seconds,
        max_attempts=cfg.max_attempts,
        retry_backoff_seconds=cfg.retry_backoff_seconds,
    )
    jobs: list[tuple[str, str, int]] = []
    for group_key, items in category_map.items():
        for category_name, cate_id in items.items():
            jobs.append((group_key, category_name, cate_id))

    print(
        f"Starting --all crawl for {len(jobs)} categories with {ALL_MODE_MAX_WORKERS} threads."
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=ALL_MODE_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                _fetch_single_category, cfg, cate_id=cate_id, all_dates=all_dates
            ): (
                group,
                name,
                cate_id,
            )
            for group, name, cate_id in jobs
        }
        overall_progress = tqdm(
            total=len(jobs), desc="SWYB --all", unit="category", dynamic_ncols=True
        )
        for future in as_completed(future_map):
            group, name, cate_id = future_map[future]
            frame, failed_dates, empty_dates = future.result()
            output_file = _resolve_all_output_path(
                cfg.output_path, group=group, category_name=name, cate_id=cate_id
            )
            frame.to_csv(output_file, index=False)
            completed += 1
            overall_progress.update(1)
            overall_progress.set_postfix(done=completed, rows=len(frame))
            print(
                f"Saved {output_file} (cate_id={cate_id}, rows={len(frame)}, "
                f"failed_dates={failed_dates}, empty_dates={empty_dates})"
            )
        overall_progress.close()

    return cfg.output_path


def run(
    config_path: Path,
    start_date: str | None,
    end_date: str | None,
    cate_id: int | None,
    category_group: str | None,
    category_name: str | None,
    category_path: str | None,
    fetch_all: bool,
    output_path: Path | None,
    failure_policy: str | None,
) -> Path:
    base_cfg = load_config(config_path)
    cfg = _effective_config(
        base_cfg,
        start_date,
        end_date,
        cate_id,
        category_group,
        category_name,
        category_path,
        output_path,
        failure_policy,
    )
    start = date.fromisoformat(cfg.start_date)
    end = date.fromisoformat(cfg.end_date)
    all_dates = iter_dates(start, end)
    if fetch_all:
        return _run_all_categories(cfg, all_dates)

    resolved_cate_id = _resolve_cate_id(cfg)

    output_df, failed_count, empty_count = _fetch_single_category(
        cfg, cate_id=resolved_cate_id, all_dates=all_dates
    )
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(cfg.output_path, index=False)

    print(
        f"Saved SWYB dataset to: {cfg.output_path} (cate_id={resolved_cate_id}) "
        f"(rows={len(output_df)}, failed_dates={failed_count}, empty_dates={empty_count})"
    )

    return cfg.output_path


def main() -> None:
    args = parse_args()
    run(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        cate_id=args.cate_id,
        category_group=args.category_group,
        category_name=args.category_name,
        category_path=args.category_path,
        fetch_all=args.all,
        output_path=args.output_path,
        failure_policy=args.failure_policy,
    )


if __name__ == "__main__":
    main()
