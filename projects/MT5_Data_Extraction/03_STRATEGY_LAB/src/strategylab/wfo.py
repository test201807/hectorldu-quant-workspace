"""Walk-Forward Optimization: splits, grid search, OOS evaluation."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from .backtest_engine import run_engine, trades_to_dataframe
from .config import CostsConfig, EngineConfig, RiskConfig
from .metrics import compute_kpis

# ---------------------------------------------------------------------------
# WFO Splits
# ---------------------------------------------------------------------------

@dataclass
class WFOFold:
    """A single walk-forward fold."""
    fold_id: str
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime


def build_wfo_folds(
    start: datetime,
    end: datetime,
    is_months: int = 18,
    oos_months: int = 3,
    step_months: int = 3,
    embargo_days: int = 0,
) -> list[WFOFold]:
    """Build expanding or rolling WFO folds.

    Returns folds where IS period is `is_months` long and OOS is
    `oos_months` long, stepping forward by `step_months`.
    An optional embargo gap (in days) separates IS end from OOS start
    to prevent information leakage.
    """
    folds: list[WFOFold] = []
    fold_num = 0
    cursor = start

    while True:
        is_start = cursor
        is_end = _add_months(is_start, is_months)
        oos_start = is_end + timedelta(days=embargo_days)
        oos_end = _add_months(oos_start, oos_months)

        if oos_end > end:
            break

        folds.append(WFOFold(
            fold_id=f"F{fold_num:02d}",
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
        ))
        fold_num += 1
        cursor = _add_months(cursor, step_months)

    return folds


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime (approximate: 30.44 days per month)."""
    return dt + timedelta(days=int(months * 30.44))


# ---------------------------------------------------------------------------
# Grid Search
# ---------------------------------------------------------------------------

@dataclass
class GridResult:
    """Result of a single grid search combination."""
    params: dict[str, Any]
    kpis_is: dict
    kpis_oos: dict
    trades_is: int
    trades_oos: int
    score: float


def grid_search(
    df: pl.DataFrame,
    fold: WFOFold,
    signal_long: list[bool],
    signal_short: list[bool],
    param_grid: dict[str, list[Any]],
    costs_cfg: CostsConfig,
    risk_cfg: RiskConfig,
    symbol: str = "UNKNOWN",
    max_combos: int = 100,
    min_trades_is: int = 30,
) -> list[GridResult]:
    """Run grid search over engine parameters on IS data.

    Only IS trades are scored. OOS trades are computed but not used
    for selection (no leakage).

    param_grid keys must match EngineConfig field names.
    """
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    # Limit combinations
    if len(combos) > max_combos:
        combos = combos[:max_combos]

    results: list[GridResult] = []

    for combo in combos:
        params = dict(zip(keys, combo))

        # Enforce trail > sl constraint
        sl = params.get("sl_atr", 2.0)
        trail = params.get("trail_atr", 3.0)
        if trail and trail <= sl:
            continue  # skip invalid: trail must be > SL

        engine_cfg = EngineConfig(**params)

        trades = run_engine(
            df=df,
            fold_id=fold.fold_id,
            is_start=fold.is_start,
            is_end=fold.is_end,
            oos_start=fold.oos_start,
            oos_end=fold.oos_end,
            signal_long=signal_long,
            signal_short=signal_short,
            engine_cfg=engine_cfg,
            costs_cfg=costs_cfg,
            risk_cfg=risk_cfg,
            symbol=symbol,
        )

        trades_df = trades_to_dataframe(trades)
        if trades_df.is_empty():
            continue

        # Split IS / OOS
        is_trades = trades_df.filter(pl.col("segment") == "IS")
        oos_trades = trades_df.filter(pl.col("segment") == "OOS")

        if is_trades.height < min_trades_is:
            continue

        kpis_is = compute_kpis(is_trades)
        kpis_oos = compute_kpis(oos_trades)

        # Score: risk-adjusted return on IS
        std_is = 1.0
        pnls = is_trades.get_column("net_pnl").to_list()
        if len(pnls) > 1:
            mean_p = sum(pnls) / len(pnls)
            var_p = sum((p - mean_p) ** 2 for p in pnls) / (len(pnls) - 1)
            std_is = max(var_p ** 0.5, 1e-10)
        score = sum(pnls) / std_is

        results.append(GridResult(
            params=params,
            kpis_is=kpis_is,
            kpis_oos=kpis_oos,
            trades_is=is_trades.height,
            trades_oos=oos_trades.height,
            score=score,
        ))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Full WFO Pipeline
# ---------------------------------------------------------------------------

@dataclass
class WFOResult:
    """Complete WFO result across all folds."""
    folds: list[WFOFold]
    best_per_fold: dict[str, GridResult]
    all_combos_per_fold: dict[str, list[GridResult]]  # Todos los combos OOS por fold — habilita PBO real (Bailey 2014)
    oos_trades: pl.DataFrame
    oos_kpis: dict
    # Holdout final — últimos N meses reservados sin tocar durante WFO
    holdout_trades: pl.DataFrame = field(default_factory=pl.DataFrame)
    holdout_kpis: dict = field(default_factory=dict)


def _modal_params(best_per_fold: dict[str, GridResult]) -> dict[str, Any]:
    """Parámetros seleccionados con mayor frecuencia entre los folds WFO.

    Usados para aplicar al holdout: el combo elegido más veces en IS
    es el 'mejor representante' de la configuración para el periodo futuro.
    """
    if not best_per_fold:
        return {}
    counter: Counter = Counter()
    params_by_key: dict = {}
    for gr in best_per_fold.values():
        key = frozenset(gr.params.items())
        counter[key] += 1
        params_by_key[key] = gr.params
    most_common_key = counter.most_common(1)[0][0]
    return params_by_key[most_common_key]


def run_wfo(
    df: pl.DataFrame,
    signal_long: list[bool],
    signal_short: list[bool],
    param_grid: dict[str, list[Any]],
    costs_cfg: CostsConfig,
    risk_cfg: RiskConfig,
    symbol: str = "UNKNOWN",
    is_months: int = 18,
    oos_months: int = 3,
    step_months: int = 3,
    embargo_days: int = 0,
    min_folds: int = 0,
    max_combos: int = 100,
    min_trades_is: int = 30,
    holdout_months: int = 0,
) -> WFOResult:
    """Run complete walk-forward optimization.

    For each fold: grid search on IS, apply best params to OOS.
    Concatenate all OOS trades for final evaluation.
    Embargo gap between IS and OOS prevents leakage.

    When holdout_months > 0, the last N months are reserved as a final
    holdout set never touched during fold building or grid search. After
    all folds complete, the most-selected params are applied to the
    holdout to produce an unbiased final validation estimate.
    """
    t_col = df.get_column("time_utc").to_list()
    start = min(t_col)
    end = max(t_col)

    # Reservar holdout si se solicita — los últimos holdout_months meses
    holdout_start = end  # default: no holdout
    if holdout_months > 0:
        holdout_start = end - timedelta(days=int(holdout_months * 30.44))

    folds = build_wfo_folds(start, holdout_start, is_months, oos_months,
                            step_months, embargo_days=embargo_days)

    if min_folds > 0 and len(folds) < min_folds:
        return WFOResult(folds=[], best_per_fold={}, all_combos_per_fold={}, oos_trades=pl.DataFrame(), oos_kpis={})

    best_per_fold: dict[str, GridResult] = {}
    all_combos_per_fold: dict[str, list[GridResult]] = {}
    all_oos_trades: list[pl.DataFrame] = []

    for fold in folds:
        results = grid_search(
            df=df,
            fold=fold,
            signal_long=signal_long,
            signal_short=signal_short,
            param_grid=param_grid,
            costs_cfg=costs_cfg,
            risk_cfg=risk_cfg,
            symbol=symbol,
            max_combos=max_combos,
            min_trades_is=min_trades_is,
        )

        if not results:
            continue

        # Guardar TODOS los combos OOS de este fold (para PBO posterior)
        all_combos_per_fold[fold.fold_id] = results

        best = results[0]
        best_per_fold[fold.fold_id] = best

        # Re-run with best params to get OOS trades
        engine_cfg = EngineConfig(**best.params)
        trades = run_engine(
            df=df,
            fold_id=fold.fold_id,
            is_start=fold.is_start,
            is_end=fold.is_end,
            oos_start=fold.oos_start,
            oos_end=fold.oos_end,
            signal_long=signal_long,
            signal_short=signal_short,
            engine_cfg=engine_cfg,
            costs_cfg=costs_cfg,
            risk_cfg=risk_cfg,
            symbol=symbol,
        )
        trades_df = trades_to_dataframe(trades)
        if not trades_df.is_empty():
            oos_only = trades_df.filter(pl.col("segment") == "OOS")
            if not oos_only.is_empty():
                all_oos_trades.append(oos_only)

    oos_combined = pl.concat(all_oos_trades) if all_oos_trades else pl.DataFrame()
    oos_kpis = compute_kpis(oos_combined) if not oos_combined.is_empty() else {}

    # ── Holdout final ────────────────────────────────────────────────────
    # Aplica los parámetros más frecuentemente seleccionados al periodo
    # de holdout (nunca visto durante WFO). Validación final sin sesgo.
    holdout_trades = pl.DataFrame()
    holdout_kpis: dict = {}
    if holdout_months > 0 and best_per_fold:
        modal = _modal_params(best_per_fold)
        if modal:
            h_engine_cfg = EngineConfig(**modal)
            h_trades = run_engine(
                df=df,
                fold_id="HOLDOUT",
                is_start=start,
                is_end=holdout_start,
                oos_start=holdout_start,
                oos_end=end,
                signal_long=signal_long,
                signal_short=signal_short,
                engine_cfg=h_engine_cfg,
                costs_cfg=costs_cfg,
                risk_cfg=risk_cfg,
                symbol=symbol,
            )
            h_df = trades_to_dataframe(h_trades)
            if not h_df.is_empty():
                holdout_trades = h_df.filter(pl.col("segment") == "OOS")
            holdout_kpis = compute_kpis(holdout_trades) if not holdout_trades.is_empty() else {}

    return WFOResult(
        folds=folds,
        best_per_fold=best_per_fold,
        all_combos_per_fold=all_combos_per_fold,
        oos_trades=oos_combined,
        oos_kpis=oos_kpis,
        holdout_trades=holdout_trades,
        holdout_kpis=holdout_kpis,
    )
