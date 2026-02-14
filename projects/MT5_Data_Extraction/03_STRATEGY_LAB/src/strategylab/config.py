"""Configuration: dataclasses + YAML/JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CostsConfig:
    spread_bps: float = 3.0
    commission_bps: float = 0.0
    slippage_bps: float = 1.0
    borrow_bps_annual: float = 0.0

    @property
    def total_one_way_bps(self) -> float:
        return self.spread_bps / 2 + self.commission_bps + self.slippage_bps

    @property
    def total_roundtrip_dec(self) -> float:
        return 2.0 * self.total_one_way_bps / 10_000


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_pos_size: float = 3.0
    min_pos_size: float = 0.25
    daily_loss_cap: float = -0.02
    daily_profit_cap: float = 0.03
    max_drawdown_cap: float = -0.15
    max_trades_per_day: int = 3
    max_positions: int = 1


@dataclass
class EngineConfig:
    sl_atr: float = 2.0
    tp_atr: float = 5.0
    trail_atr: float | None = 3.0
    time_stop_bars: int = 288
    entry_confirm_bars: int = 12
    exit_gate_off_bars: int = 12
    min_hold_bars: int = 6
    cooldown_bars: int = 24
    mon_fri_only: bool = True
    ema_filter: bool = True
    ema_fast: int = 48
    ema_slow: int = 288


@dataclass
class WFOConfig:
    is_months: int = 18
    oos_months: int = 3
    step_months: int = 3
    embargo_days: int = 5
    min_folds: int = 6
    max_combos_per_symbol: int = 100
    param_grid: dict[str, list[Any]] = field(default_factory=lambda: {
        "sl_atr": [1.5, 2.0, 2.5],
        "tp_atr": [3.0, 5.0, 7.0],
        "trail_atr": [3.0, 4.0, 5.0],
    })
    score_metric: str = "calmar"
    seed: int = 42


@dataclass
class MCConfig:
    n_sims: int = 1000
    block_size: int = 20
    stress_cost_factor: float = 2.0
    stress_slippage_factor: float = 3.0
    seed: int = 42
    confidence_levels: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.50, 0.75, 0.95])


@dataclass
class StrategyLabConfig:
    strategy: str = "trend_v2"
    symbols: list[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "XAUUSD", "LVMHP"])
    bar_seconds: int = 300
    costs: CostsConfig = field(default_factory=CostsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    wfo: WFOConfig = field(default_factory=WFOConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    outputs_root: str = ""
    data_root: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategyLabConfig:
        """Load config from YAML file."""
        try:
            import yaml  # noqa: PLC0415
            with open(path, encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
        except ImportError:
            # Fallback: try JSON
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        return cls._from_dict(d)

    @classmethod
    def from_json(cls, path: str | Path) -> StrategyLabConfig:
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> StrategyLabConfig:
        costs = CostsConfig(**d.get("costs", {}))
        risk = RiskConfig(**d.get("risk", {}))
        engine = EngineConfig(**d.get("engine", {}))

        wfo_raw = d.get("wfo", {})
        wfo_kw = {k: v for k, v in wfo_raw.items() if k not in ("param_grid",)}
        if "param_grid" in wfo_raw:
            wfo_kw["param_grid"] = wfo_raw["param_grid"]
        wfo = WFOConfig(**wfo_kw)

        mc_raw = d.get("mc", {})
        mc_kw = {k: v for k, v in mc_raw.items() if k not in ("confidence_levels",)}
        if "confidence_levels" in mc_raw:
            mc_kw["confidence_levels"] = mc_raw["confidence_levels"]
        mc = MCConfig(**mc_kw)

        return cls(
            strategy=d.get("strategy", "trend_v2"),
            symbols=d.get("symbols", ["EURUSD", "GBPUSD", "XAUUSD", "LVMHP"]),
            bar_seconds=d.get("bar_seconds", 300),
            costs=costs, risk=risk, engine=engine, wfo=wfo, mc=mc,
            outputs_root=d.get("outputs_root", ""),
            data_root=d.get("data_root", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for snapshot."""
        from dataclasses import asdict  # noqa: PLC0415
        return asdict(self)

    def snapshot_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)
