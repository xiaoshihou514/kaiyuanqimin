from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SwybConfig:
    start_date: str
    end_date: str
    cate_id: int | None
    category_group: str | None
    category_name: str | None
    category_path: str | None
    output_path: Path
    cache_dir: Path
    timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: float
    request_interval_seconds: float
    failure_policy: str
    endpoint_url: str
    category_data_url: str
    referer_url: str
    user_agent: str


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    text = _optional_text(value)
    if text is None:
        return None
    return int(text)


def load_config(config_path: Path) -> SwybConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    swyb = raw.get("swyb", {})

    return SwybConfig(
        start_date=str(swyb.get("start_date", "2020-01-01")),
        end_date=str(swyb.get("end_date", "2026-05-21")),
        cate_id=_optional_int(swyb.get("cate_id", 170130)),
        category_group=_optional_text(swyb.get("category_group")),
        category_name=_optional_text(swyb.get("category_name")),
        category_path=_optional_text(swyb.get("category_path")),
        output_path=Path(str(swyb.get("output_path", "data/market_prices.csv"))),
        cache_dir=Path(str(swyb.get("cache_dir", ".cache/swyb"))),
        timeout_seconds=float(swyb.get("timeout_seconds", 10.0)),
        max_attempts=int(swyb.get("max_attempts", 3)),
        retry_backoff_seconds=float(swyb.get("retry_backoff_seconds", 0.5)),
        request_interval_seconds=float(swyb.get("request_interval_seconds", 0.15)),
        failure_policy=str(swyb.get("failure_policy", "partial_with_report")),
        endpoint_url=str(
            swyb.get(
                "endpoint_url",
                "https://cif.mofcom.gov.cn/cif/getEnterpriseListForDate.fhtml",
            )
        ),
        category_data_url=str(
            swyb.get(
                "category_data_url",
                "https://cif.mofcom.gov.cn/cif/resDataIndex/js/riduData.js",
            )
        ),
        referer_url=str(
            swyb.get(
                "referer_url",
                "https://cif.mofcom.gov.cn/cif/html/dataCenter2021/index.html?jgnfcprd",
            )
        ),
        user_agent=str(
            swyb.get(
                "user_agent",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            )
        ),
    )
