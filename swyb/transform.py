from __future__ import annotations

from typing import Any

import pandas as pd

OUTPUT_COLUMNS = [
    "date",
    "report_date",
    "market",
    "product",
    "price",
    "price_1",
    "price_2",
    "price_3",
    "area_code",
    "county_name",
    "enterprise_id",
    "commodity_id",
]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def response_to_frame(response: dict[str, Any], fallback_date: str) -> pd.DataFrame:
    query_date = str(response.get("date") or fallback_date)
    rows = []
    for item in response.get("datas", []):
        rows.append(
            {
                "date": query_date,
                "report_date": str(item.get("RPT_DATE") or ""),
                "market": str(item.get("NAME") or ""),
                "product": str(item.get("COMMDITYNAME") or ""),
                "price": _to_float(item.get("PRICE2")),
                "price_1": _to_float(item.get("PRICE1")),
                "price_2": _to_float(item.get("PRICE2")),
                "price_3": _to_float(item.get("PRICE3")),
                "area_code": str(item.get("AREA") or ""),
                "county_name": str(item.get("COUNTY_NAME") or ""),
                "enterprise_id": int(item["ENTERID"]) if item.get("ENTERID") is not None else None,
                "commodity_id": int(item["COMMDITY_ID"])
                if item.get("COMMDITY_ID") is not None
                else None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame = frame.sort_values(["date", "market", "report_date"], kind="stable").reset_index(drop=True)
    return frame
