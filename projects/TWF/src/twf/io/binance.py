from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import httpx

BASE_URL = "https://api.binance.com"
KLINES = "/api/v3/klines"

def _ms(ts: datetime) -> int:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() * 1000)

def get_klines(symbol: str, interval: str, start: Optional[datetime]=None, end: Optional[datetime]=None, limit: int=1000) -> list[list]:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    if start: params["startTime"] = _ms(start)
    if end:   params["endTime"]   = _ms(end)
    with httpx.Client(timeout=30) as client:
        r = client.get(BASE_URL + KLINES, params=params)
        r.raise_for_status()
        return r.json()
