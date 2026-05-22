from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class SwybConfig:
    start_date: str
    end_date: str
    cate_id: int
    output_path: Path
    timeout_seconds: float
    max_attempts: int
    retry_backoff_seconds: float
    request_interval_seconds: float
    failure_policy: str
    endpoint_url: str
    referer_url: str
    user_agent: str


def load_config(config_path: Path) -> SwybConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    swyb = raw.get("swyb", {})

    return SwybConfig(
        start_date=str(swyb.get("start_date", "2022-01-01")),
        end_date=str(swyb.get("end_date", "2022-04-30")),
        cate_id=int(swyb.get("cate_id", 170130)),
        output_path=Path(str(swyb.get("output_path", "data/market_prices.csv"))),
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
