"""Bar-by-bar backtest engine: institutional grade.

Supports: long/short, SL/TP/trailing, cooldown, gate hysteresis,
Mon-Fri flatten, risk-based sizing, costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

# FTMO resets daily PnL limits at midnight CET (UTC+1 winter / UTC+2 summer).
# Using Europe/Paris which observes CET/CEST with correct DST transitions.
_TZ_CET = ZoneInfo("Europe/Paris")


def _to_cet_date(t: Any) -> Any:
    """Convert UTC timestamp to CET/CEST date for FTMO daily reset."""
    if hasattr(t, "astimezone"):
        return t.astimezone(_TZ_CET).date()
    if hasattr(t, "date") and callable(getattr(t, "date", None)):
        return t.date()
    return t

from .config import CostsConfig, EngineConfig, RiskConfig
from .costs import roundtrip_cost_dec
from .risk import DailyRiskTracker, compute_position_size


@dataclass
class Trade:
    """A single completed trade."""
    symbol: str
    fold_id: str
    segment: str
    side: str
    signal_time_utc: Any
    entry_time_utc: Any
    exit_time_utc: Any
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    hold_bars: int
    exit_reason: str
    pos_size: float = 1.0


def _is_finite(x: Any) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _is_weekend(dow: int) -> bool:
    """Polars weekday: 1=Mon..7=Sun. Weekend = 6 or 7."""
    return dow >= 6


def run_engine(
    df: pl.DataFrame,
    fold_id: str,
    is_start: Any,
    is_end: Any,
    oos_start: Any,
    oos_end: Any,
    signal_long: list[bool],
    signal_short: list[bool],
    engine_cfg: EngineConfig,
    costs_cfg: CostsConfig,
    risk_cfg: RiskConfig,
    symbol: str = "UNKNOWN",
) -> list[Trade]:
    """Run bar-by-bar simulation on prepared data.

    Args:
        df: DataFrame with columns [time_utc, open, high, low, close, atr_price].
            Must be sorted by time_utc, single symbol.
        signal_long/signal_short: boolean lists aligned to df rows.
            These are the RAW gate signals (before confirmation).
        engine_cfg/costs_cfg/risk_cfg: configuration objects.

    Returns:
        List of completed Trade objects.
    """
    n = df.height
    if n < 10:
        return []

    t_list = df.get_column("time_utc").to_list()
    o_list = df.get_column("open").to_list()
    h_list = df.get_column("high").to_list()
    l_list = df.get_column("low").to_list()
    c_list = df.get_column("close").to_list()

    # ATR (in price units)
    if "atr_price" in df.columns:
        atr_list = df.get_column("atr_price").to_list()
    elif "atr_bps_96" in df.columns:
        atr_list = [(float(a) / 10_000 * float(c)) if _is_finite(a) and _is_finite(c) else None
                    for a, c in zip(df.get_column("atr_bps_96").to_list(), c_list)]
    else:
        atr_list = [float(c) * 0.005 if _is_finite(c) else None for c in c_list]

    # Weekday
    dow_list = df.get_column("time_utc").dt.weekday().to_list()

    # CET/CEST dates para FTMO daily reset (FTMO cuenta pérdida desde medianoche CET, no UTC)
    cet_date_list = [_to_cet_date(t) for t in t_list]

    # Confirmation rolling
    confirm = engine_cfg.entry_confirm_bars
    conf_long = [False] * n
    conf_short = [False] * n
    run_l = 0
    run_s = 0
    for i in range(n):
        run_l = run_l + 1 if signal_long[i] else 0
        run_s = run_s + 1 if signal_short[i] else 0
        conf_long[i] = run_l >= confirm
        conf_short[i] = run_s >= confirm

    cost_rt = roundtrip_cost_dec(costs_cfg)
    tracker = DailyRiskTracker(risk_cfg)
    trades: list[Trade] = []

    # Position state
    pos = 0
    entry_idx: int | None = None
    entry_price = 0.0
    stop = 0.0
    tp_price = 0.0
    trail_stop: float | None = None
    trail_dist: float | None = None
    best_price = 0.0
    pos_size = 1.0
    gate_off_streak = 0
    cooldown_cnt = 0

    def _segment(idx: int) -> str | None:
        ti = t_list[idx]
        if is_start <= ti <= is_end:
            return "IS"
        if oos_start <= ti <= oos_end:
            return "OOS"
        return None

    for idx in range(n):
        tracker.new_bar(cet_date_list[idx])

        # ---- EXIT LOGIC ----
        if pos != 0 and entry_idx is not None:
            bars_held = idx - entry_idx
            gate_now = signal_long[idx] if pos == 1 else signal_short[idx]
            gate_off_streak = 0 if gate_now else gate_off_streak + 1

            hi = float(h_list[idx]) if _is_finite(h_list[idx]) else float(c_list[idx])
            lo = float(l_list[idx]) if _is_finite(l_list[idx]) else float(c_list[idx])

            # Update floating PnL so tracker reflects FTMO-style equity
            # (open losses count against daily and total drawdown limits)
            _close_now = float(c_list[idx]) if _is_finite(c_list[idx]) else entry_price
            _sign_f = 1.0 if pos == 1 else -1.0
            tracker.update_floating_pnl(_sign_f * (_close_now / entry_price - 1.0) - cost_rt)

            exit_reason: str | None = None
            exit_price_val = 0.0

            if pos == 1:
                best_price = max(best_price, hi)
                if trail_dist is not None:
                    ts = best_price - trail_dist
                    trail_stop = ts if trail_stop is None else max(trail_stop, ts)
                if lo <= stop:
                    exit_reason, exit_price_val = "SL", stop
                elif trail_stop is not None and lo <= trail_stop:
                    exit_reason, exit_price_val = "TRAIL", trail_stop
                elif hi >= tp_price:
                    exit_reason, exit_price_val = "TP", tp_price
            else:  # pos == -1
                best_price = min(best_price, lo)
                if trail_dist is not None:
                    ts = best_price + trail_dist
                    trail_stop = ts if trail_stop is None else min(trail_stop, ts)
                if hi >= stop:
                    exit_reason, exit_price_val = "SL", stop
                elif trail_stop is not None and hi >= trail_stop:
                    exit_reason, exit_price_val = "TRAIL", trail_stop
                elif lo <= tp_price:
                    exit_reason, exit_price_val = "TP", tp_price

            if exit_reason is None and bars_held >= engine_cfg.time_stop_bars:
                exit_reason, exit_price_val = "TIME", float(c_list[idx])
            if (exit_reason is None and bars_held >= engine_cfg.min_hold_bars
                    and gate_off_streak >= engine_cfg.exit_gate_off_bars):
                exit_reason, exit_price_val = "REGIME_OFF", float(c_list[idx])
            if exit_reason is None and engine_cfg.mon_fri_only and _is_weekend(dow_list[idx]):
                exit_reason, exit_price_val = "WEEKEND", float(c_list[idx])

            # DAILY_CAP: floating + closed daily PnL breaches daily loss limit
            if tracker.is_daily_capped():
                exit_reason, exit_price_val = "DAILY_CAP", float(c_list[idx])
            # DD_KILL overrides all — drawdown now includes floating PnL
            if tracker.is_killed():
                exit_reason, exit_price_val = "DD_KILL", float(c_list[idx])

            if exit_reason is not None:
                sign = 1.0 if pos == 1 else -1.0
                gross = sign * (exit_price_val / entry_price - 1.0)
                net = gross - cost_rt
                seg = _segment(entry_idx)
                trades.append(Trade(
                    symbol=symbol, fold_id=fold_id, segment=seg or "UNKNOWN",
                    side="LONG" if pos == 1 else "SHORT",
                    signal_time_utc=t_list[entry_idx],
                    entry_time_utc=t_list[min(entry_idx + 1, n - 1)],
                    exit_time_utc=t_list[idx],
                    entry_price=entry_price, exit_price=exit_price_val,
                    gross_pnl=gross, net_pnl=net,
                    hold_bars=bars_held, exit_reason=exit_reason,
                    pos_size=pos_size,
                ))
                tracker.record_trade(net)
                tracker.update_floating_pnl(0.0)  # position closed
                pos = 0
                entry_idx = None
                cooldown_cnt = engine_cfg.cooldown_bars
                gate_off_streak = 0
                trail_stop = None
                continue

        # ---- COOLDOWN ----
        if cooldown_cnt > 0:
            cooldown_cnt -= 1
            continue

        # ---- ENTRY LOGIC ----
        if pos == 0 and idx < n - 2:
            if engine_cfg.mon_fri_only and _is_weekend(dow_list[idx]):
                continue
            if not tracker.can_trade():
                continue
            if tracker.is_killed():
                continue

            atr_val = float(atr_list[idx]) if _is_finite(atr_list[idx]) else float(c_list[idx]) * 0.005
            if atr_val <= 0:
                continue

            entered = False
            if conf_long[idx]:
                entry_price = float(o_list[idx + 1]) if _is_finite(o_list[idx + 1]) else float(c_list[idx])
                sl_dist = engine_cfg.sl_atr * atr_val
                stop = entry_price - sl_dist
                tp_price = entry_price + engine_cfg.tp_atr * atr_val
                trail_dist = engine_cfg.trail_atr * atr_val if engine_cfg.trail_atr else None
                trail_stop = None
                best_price = entry_price
                pos_size = compute_position_size(risk_cfg, entry_price, sl_dist)
                pos = 1
                entered = True
            elif conf_short[idx]:
                entry_price = float(o_list[idx + 1]) if _is_finite(o_list[idx + 1]) else float(c_list[idx])
                sl_dist = engine_cfg.sl_atr * atr_val
                stop = entry_price + sl_dist
                tp_price = entry_price - engine_cfg.tp_atr * atr_val
                trail_dist = engine_cfg.trail_atr * atr_val if engine_cfg.trail_atr else None
                trail_stop = None
                best_price = entry_price
                pos_size = compute_position_size(risk_cfg, entry_price, sl_dist)
                pos = -1
                entered = True

            if entered:
                entry_idx = idx
                gate_off_streak = 0

    return trades


def trades_to_dataframe(trades: list[Trade]) -> pl.DataFrame:
    """Convert trade list to polars DataFrame."""
    if not trades:
        return pl.DataFrame()
    from dataclasses import asdict  # noqa: PLC0415
    return pl.DataFrame([asdict(t) for t in trades])
