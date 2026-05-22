from __future__ import annotations

from datetime import date, timedelta
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import requests_cache
from retry_requests import retry
from urllib.parse import urlencode

BLOCK_MARKERS = (
    "captcha",
    "verify",
    "forbidden",
    "access denied",
    "blocked",
    "waf",
    "robot",
    "anti-spider",
    "anti crawl",
    "anti-crawl",
    "验证码",
    "安全验证",
    "访问过于频繁",
    "拒绝访问",
)
USER_AGENT_POOL = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)
ACCEPT_LANGUAGE_POOL = (
    "zh-CN,zh;q=0.9,en;q=0.8",
    "zh-CN,en-US;q=0.9,en;q=0.8",
    "en-US,en;q=0.8,zh-CN;q=0.6",
)
GROUP_LABELS = {
    "liangyou": "粮油",
    "roulei": "肉类",
    "qindan": "禽蛋",
    "shucai": "蔬菜",
}


class AntiScrapeBlockedError(RuntimeError):
    """Raised when anti-scrape or bot wall behavior is detected."""


def build_client(cache_dir: Path) -> requests_cache.session.CachedSession:
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests_cache.CachedSession(
        str(cache_dir / "swyb_cache"),
        expire_after=-1,
        allowable_methods=("GET", "POST"),
    )
    return retry(session, retries=5, backoff_factor=0.2)


def _looks_like_block(content_type: str, raw_text: str) -> bool:
    lowered = raw_text.lower()
    marker_hit = any(marker in lowered for marker in BLOCK_MARKERS)
    html_like = raw_text.lstrip().startswith("<") and (
        "<html" in lowered or "<!doctype html" in lowered
    )
    non_json_type = "application/json" not in content_type.lower()
    return marker_hit or (html_like and non_json_type)


def iter_dates(start_date: date, end_date: date) -> list[date]:
    total_days = (end_date - start_date).days
    if total_days < 0:
        raise ValueError("start_date cannot be later than end_date.")

    return [start_date + timedelta(days=offset) for offset in range(total_days + 1)]


def _request_headers(user_agent: str, referer_url: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": random.choice(ACCEPT_LANGUAGE_POOL),
        "Origin": "https://cif.mofcom.gov.cn",
        "Referer": referer_url,
        "User-Agent": random.choice((user_agent, *USER_AGENT_POOL)),
        "X-Requested-With": "XMLHttpRequest",
    }


def fetch_daily_category_map(
    *,
    client: requests_cache.session.CachedSession,
    category_data_url: str,
    referer_url: str,
    user_agent: str,
    timeout_seconds: float,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> dict[str, dict[str, int]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    for attempt in range(1, max_attempts + 1):
        headers = {**_request_headers(user_agent, referer_url), "Accept": "*/*"}
        try:
            resp = client.get(category_data_url, headers=headers, timeout=timeout_seconds)
            resp.raise_for_status()
            content_type = str(resp.headers.get("Content-Type", ""))
            raw = resp.text
            if _looks_like_block(content_type, raw):
                raise AntiScrapeBlockedError(
                    "Detected anti-scrape/block response signature."
                )
            parsed = parse_daily_category_js(raw)
            if not parsed:
                raise ValueError("No category mapping found in riduData.js response.")
            return parsed
        except Exception as exc:  # requests-cache / requests can raise multiple exception types
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed fetching category map after {max_attempts} attempt(s)."
                ) from exc
            retry_sleep = (
                retry_backoff_seconds * attempt + random.uniform(0.2, 0.9) * attempt
            )
            time.sleep(retry_sleep)

    raise RuntimeError("Unreachable retry branch.")


def parse_daily_category_js(script_text: str) -> dict[str, dict[str, int]]:
    block_pattern = re.compile(r"([a-zA-Z0-9_]+)\s*:\s*\[(.*?)\]", re.DOTALL)
    item_pattern = re.compile(r"\{[^{}]*\}")
    id_pattern = re.compile(r"id\s*:\s*['\"](\d+)['\"]")
    name_pattern = re.compile(r"name\s*:\s*['\"]([^'\"]+)['\"]")

    parsed: dict[str, dict[str, int]] = {}
    for group_key, raw_items in block_pattern.findall(script_text):
        items: dict[str, int] = {}
        for item_block in item_pattern.findall(raw_items):
            id_match = id_pattern.search(item_block)
            name_match = name_pattern.search(item_block)
            if not id_match or not name_match:
                continue
            items[name_match.group(1)] = int(id_match.group(1))
        if items:
            parsed[group_key] = items
    return parsed


def resolve_category_to_cate_id(
    *,
    category_map: dict[str, dict[str, int]],
    category_group: str | None,
    category_name: str | None,
    category_path: str | None,
) -> int:
    group_key, item_name = normalize_category_selector(
        category_group=category_group,
        category_name=category_name,
        category_path=category_path,
    )
    if item_name is None:
        raise ValueError("Category name is required when cate_id is not provided.")

    if group_key is not None:
        group_items = category_map.get(group_key)
        if not group_items:
            available_groups = ", ".join(
                sorted(_display_group_name(k) for k in category_map)
            )
            raise ValueError(
                f"Unknown category group: {group_key}. Available groups: {available_groups}"
            )
        if item_name not in group_items:
            available_items = ", ".join(sorted(group_items.keys()))
            raise ValueError(
                f"Category '{item_name}' not found in group '{_display_group_name(group_key)}'. "
                f"Available: {available_items}"
            )
        return group_items[item_name]

    matches: list[tuple[str, int]] = []
    for key, items in category_map.items():
        if item_name in items:
            matches.append((key, items[item_name]))
    if not matches:
        raise ValueError(f"Category '{item_name}' not found in daily category map.")
    if len(matches) > 1:
        display = ", ".join(
            f"{_display_group_name(key)}-{item_name}={cate_id}"
            for key, cate_id in matches
        )
        raise ValueError(
            f"Ambiguous category name '{item_name}'. Please specify group/category-path. Matches: {display}"
        )
    return matches[0][1]


def normalize_category_selector(
    *,
    category_group: str | None,
    category_name: str | None,
    category_path: str | None,
) -> tuple[str | None, str | None]:
    if category_path:
        text = category_path.strip()
        for sep in ("-", "—", "－", "/", "｜", "|"):
            if sep in text:
                left, right = text.split(sep, 1)
                return _normalize_group(left), right.strip() or None
        return None, text
    return _normalize_group(category_group), (
        category_name.strip() if category_name else None
    )


def _normalize_group(group: str | None) -> str | None:
    if not group:
        return None
    g = group.strip().lower()
    aliases = {
        "粮油": "liangyou",
        "liangyou": "liangyou",
        "肉类": "roulei",
        "roulei": "roulei",
        "禽蛋": "qindan",
        "qindan": "qindan",
        "蔬菜": "shucai",
        "shucai": "shucai",
    }
    return aliases.get(g, g)


def _display_group_name(group_key: str) -> str:
    return GROUP_LABELS.get(group_key, group_key)


def fetch_for_date(
    *,
    client: requests_cache.session.CachedSession,
    endpoint_url: str,
    referer_url: str,
    user_agent: str,
    cate_id: int,
    search_date: date,
    timeout_seconds: float,
    max_attempts: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1.")

    payload = urlencode({"cateId": str(cate_id), "searchDate": search_date.isoformat()})

    for attempt in range(1, max_attempts + 1):
        headers = {
            **_request_headers(user_agent, referer_url),
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        }
        try:
            resp = client.post(
                endpoint_url,
                data=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            resp.raise_for_status()
            content_type = str(resp.headers.get("Content-Type", ""))
            raw = resp.text

            if _looks_like_block(content_type, raw):
                raise AntiScrapeBlockedError(
                    "Detected anti-scrape/block response signature."
                )

            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Response JSON must be an object.")
            if "datas" not in parsed:
                raise ValueError("Response JSON missing 'datas' key.")
            return parsed
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed fetching {search_date.isoformat()} after {max_attempts} attempt(s)."
                ) from exc
            retry_sleep = (
                retry_backoff_seconds * attempt + random.uniform(0.2, 0.9) * attempt
            )
            time.sleep(retry_sleep)

    raise RuntimeError("Unreachable retry branch.")
