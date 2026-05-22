from __future__ import annotations

from datetime import date, timedelta
import json
import time
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError


def iter_dates(start_date: date, end_date: date) -> list[date]:
    total_days = (end_date - start_date).days
    if total_days < 0:
        raise ValueError("start_date cannot be later than end_date.")

    return [start_date + timedelta(days=offset) for offset in range(total_days + 1)]


def fetch_for_date(
    *,
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

    payload = parse.urlencode(
        {"cateId": str(cate_id), "searchDate": search_date.isoformat()}
    ).encode("utf-8")

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://cif.mofcom.gov.cn",
        "Referer": referer_url,
        "User-Agent": user_agent,
        "X-Requested-With": "XMLHttpRequest",
    }

    for attempt in range(1, max_attempts + 1):
        req = request.Request(url=endpoint_url, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Response JSON must be an object.")
            return parsed
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed fetching {search_date.isoformat()} after {max_attempts} attempt(s)."
                ) from exc
            time.sleep(retry_backoff_seconds * attempt)

    raise RuntimeError("Unreachable retry branch.")
