#!/usr/bin/env python
"""
build_trend_v2_cells.py
-----------------------
Adds cells 06-20 to 03_TREND_M5_Strategy_v2.ipynb and updates Cell 00 artifacts.
Also applies bug fixes (Trail>SL, SHORT gate independent, dedup keep=last, overlay guard).

Run from repo root:
    python projects/MT5_Data_Extraction/ER_STRATEGY_LAB/scripts/build_trend_v2_cells.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "03_TREND_M5_Strategy_v2.ipynb"

def _cell(source: str, cell_type: str = "code") -> dict:
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source.split("\n")  # list of lines
    }

def _fix_source_lines(lines_str: str) -> list[str]:
    """Convert a multiline string into notebook source lines (with \\n on each line except last)."""
    lines = lines_str.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result

def make_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": _fix_source_lines(source)
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 06: Regime Gate por Fold (v2.0.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_06 = r'''# ======================================================================================
# Celda 06 v2.0.1 — Regime Gate por Fold (TREND, M5) [IS-only, no leakage]
# BUG FIX vs v1: SHORT gate calibrado independientemente (thr_mom_short separado)
# Guardrails de cobertura: 5% <= coverage_IS <= 80% (cada side por separado)
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 06 v2.0.1 :: Regime Gate por Fold (TREND, M5)")

# ---------- Preflight ----------
if "RUN" not in globals():
    raise RuntimeError("[Celda 06] ERROR: RUN no existe. Ejecuta Celda 00.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS.get("features_m5", RUN_DIR / "features_m5_v2.parquet")
WFO_FOLDS_PATH = ARTIFACTS.get("wfo_folds", RUN_DIR / "wfo_folds_v2.parquet")

if not FEATURES_PATH.exists():
    raise RuntimeError(f"[Celda 06] ERROR: features no encontradas: {FEATURES_PATH}")
if not WFO_FOLDS_PATH.exists():
    raise RuntimeError(f"[Celda 06] ERROR: wfo_folds no encontrados: {WFO_FOLDS_PATH}")

OUT_REGIME = ARTIFACTS.get("regime_params_by_fold", RUN_DIR / "regime_params_by_fold_v2.parquet")
OUT_SNAP = ARTIFACTS.get("regime_params_snapshot", RUN_DIR / "regime_params_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- Parametros ----------
ER_COL = "er_288"
MOM_COL = "mom_bps_288"
VOL_COL = "vol_bps_288"

Q_SCHEMES = [
    {"name": "BASE",   "q_er": 0.60, "q_mom": 0.55, "q_vol": 0.90},
    {"name": "RELAX1", "q_er": 0.50, "q_mom": 0.50, "q_vol": 0.95},
    {"name": "RELAX2", "q_er": 0.40, "q_mom": 0.50, "q_vol": 0.99},
    {"name": "TIGHT1", "q_er": 0.70, "q_mom": 0.60, "q_vol": 0.85},
]
COV_IS_MIN = 0.05
COV_IS_MAX = 0.80
MIN_IS_ROWS = 5_000

# ---------- Helpers ----------
def _q_safe(s: pl.Series, q: float):
    s2 = s.drop_nulls()
    if s2.len() == 0:
        return None
    v = s2.quantile(q, interpolation="nearest")
    if v is None:
        return None
    fv = float(v)
    return fv if math.isfinite(fv) else None

def _calibrate_side(df_is: pl.DataFrame, side: str) -> dict:
    """Calibrar thresholds para un side (LONG o SHORT) independientemente."""
    er_s = df_is.get_column(ER_COL)
    mom_s = df_is.get_column(MOM_COL)
    vol_s = df_is.get_column(VOL_COL)

    best = None
    for sch in Q_SCHEMES:
        thr_er = _q_safe(er_s, sch["q_er"])
        thr_vol = _q_safe(vol_s, sch["q_vol"])
        if thr_er is None or thr_vol is None:
            continue

        # BUG FIX: LONG usa percentil positivo de mom, SHORT usa percentil negativo
        if side == "LONG":
            thr_mom = _q_safe(mom_s, sch["q_mom"])
            if thr_mom is None:
                continue
            thr_mom = max(0.0, thr_mom)
            gate = (
                (pl.col(ER_COL) >= thr_er) &
                (pl.col(MOM_COL) >= thr_mom) &
                (pl.col(VOL_COL) <= thr_vol)
            )
        else:  # SHORT
            # percentil bajo de momentum (valores negativos)
            thr_mom_short = _q_safe(mom_s, 1.0 - sch["q_mom"])
            if thr_mom_short is None:
                continue
            thr_mom_short = min(0.0, thr_mom_short)
            thr_mom = thr_mom_short
            gate = (
                (pl.col(ER_COL) >= thr_er) &
                (pl.col(MOM_COL) <= thr_mom) &
                (pl.col(VOL_COL) <= thr_vol)
            )

        cov = float(df_is.select(gate.mean()).item())
        payload = {
            "scheme": sch["name"], "side": side,
            "thr_er": float(thr_er), "thr_mom": float(thr_mom), "thr_vol": float(thr_vol),
            "cov_is": float(cov),
        }
        if COV_IS_MIN <= cov <= COV_IS_MAX:
            return payload
        score = abs(cov - 0.30)
        if best is None or score < best[0]:
            best = (score, payload)

    if best is not None:
        return best[1]
    return {"scheme": "FAIL", "side": side, "thr_er": None, "thr_mom": None, "thr_vol": None, "cov_is": 0.0}

# ---------- Main ----------
df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_FOLDS_PATH)

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

rows = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        is_s = fold_row["IS_start"]
        is_e = fold_row["IS_end"]
        oos_s = fold_row["OOS_start"]
        oos_e = fold_row["OOS_end"]

        df_is = df_sym.filter(
            (pl.col("time_utc") >= is_s) & (pl.col("time_utc") <= is_e)
        ).drop_nulls([ER_COL, MOM_COL, VOL_COL])

        df_oos = df_sym.filter(
            (pl.col("time_utc") >= oos_s) & (pl.col("time_utc") <= oos_e)
        ).drop_nulls([ER_COL, MOM_COL, VOL_COL])

        for side in ("LONG", "SHORT"):
            cal = _calibrate_side(df_is, side) if df_is.height >= MIN_IS_ROWS else {
                "scheme": "SKIP", "side": side, "thr_er": None, "thr_mom": None, "thr_vol": None, "cov_is": 0.0
            }

            # OOS coverage
            cov_oos = 0.0
            if cal["thr_er"] is not None:
                if side == "LONG":
                    g = (pl.col(ER_COL) >= cal["thr_er"]) & (pl.col(MOM_COL) >= cal["thr_mom"]) & (pl.col(VOL_COL) <= cal["thr_vol"])
                else:
                    g = (pl.col(ER_COL) >= cal["thr_er"]) & (pl.col(MOM_COL) <= cal["thr_mom"]) & (pl.col(VOL_COL) <= cal["thr_vol"])
                if df_oos.height > 0:
                    cov_oos = float(df_oos.select(g.mean()).item())

            rows.append({
                "symbol": sym, "fold_id": fid, "side": side,
                "scheme": cal["scheme"],
                "thr_er": cal["thr_er"], "thr_mom": cal["thr_mom"], "thr_vol": cal["thr_vol"],
                "cov_is": cal["cov_is"], "cov_oos": cov_oos,
                "n_is": df_is.height, "n_oos": df_oos.height,
            })
            print(f"[Celda 06] {sym} fold={fid} {side} :: scheme={cal['scheme']} cov_IS={cal['cov_is']:.3f} cov_OOS={cov_oos:.3f}")

gate_df = pl.DataFrame(rows).sort(["symbol", "fold_id", "side"])
gate_df.write_parquet(str(OUT_REGIME), compression="zstd")

snap = {
    "created_utc": _now_utc_iso(),
    "version": "v2.0.1",
    "symbols": symbols,
    "fold_ids": [str(f) for f in fold_ids],
    "params": {"ER_COL": ER_COL, "MOM_COL": MOM_COL, "VOL_COL": VOL_COL, "Q_SCHEMES": Q_SCHEMES},
    "bug_fix": "SHORT gate calibrado independientemente con percentil negativo de momentum",
}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print(f"\n[Celda 06] OUT: {OUT_REGIME} ({gate_df.height} rows)")
print(f"[Celda 06] OUT: {OUT_SNAP}")
print(">>> Celda 06 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 07: Senales + Ejecucion t+1 + Costos (v2.0.1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_07 = r'''# ======================================================================================
# Celda 07 v2.0.1 — Senales TREND + Ejecucion t+1 + Costos (BASE/STRESS)
# Entry en open(t+1), exit en open(t+2). Segmento IS/OOS por entry_time.
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 07 v2.0.1 :: Senales + Ejecucion t+1 + Costos")

if "RUN" not in globals():
    raise RuntimeError("[Celda 07] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS.get("cost_model_snapshot", RUN_DIR / "cost_model_snapshot_v2.json")

for p, label in [(FEATURES_PATH, "features"), (WFO_PATH, "wfo_folds"), (REGIME_PATH, "regime_params")]:
    if not Path(p).exists():
        raise RuntimeError(f"[Celda 07] ERROR: falta {label}: {p}")

OUT_SIGNALS = ARTIFACTS.get("signals_all", RUN_DIR / "signals_all_v2.parquet")
OUT_SNAP = ARTIFACTS.get("signals_snapshot", RUN_DIR / "signals_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- Parametros ----------
ER_COL = "er_288"
MOM_COL = "mom_bps_288"
VOL_COL = "vol_bps_288"

# ---------- Load ----------
df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)
df_regime = pl.read_parquet(REGIME_PATH)
cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))
costs_by_sym = cost_snap.get("costs_by_symbol", {})

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

all_trades = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    cinfo = costs_by_sym.get(sym, {})
    cost_base_bps = float(cinfo.get("cost_base_bps", cinfo.get("COST_BASE_BPS", 3.0)))
    cost_stress_bps = float(cinfo.get("cost_stress_bps", cinfo.get("COST_STRESS_BPS", 6.0)))
    cost_base_rt = cost_base_bps / 10_000
    cost_stress_rt = cost_stress_bps / 10_000

    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        is_s, is_e = fold_row["IS_start"], fold_row["IS_end"]
        oos_s, oos_e = fold_row["OOS_start"], fold_row["OOS_end"]

        for side in ("LONG", "SHORT"):
            rg = df_regime.filter(
                (pl.col("symbol") == sym) & (pl.col("fold_id") == fid) & (pl.col("side") == side)
            )
            if rg.is_empty():
                continue
            rg_row = rg.row(0, named=True)
            if rg_row["thr_er"] is None:
                continue

            thr_er = float(rg_row["thr_er"])
            thr_mom = float(rg_row["thr_mom"])
            thr_vol = float(rg_row["thr_vol"])

            if side == "LONG":
                gate_expr = (
                    (pl.col(ER_COL) >= thr_er) &
                    (pl.col(MOM_COL) >= thr_mom) &
                    (pl.col(VOL_COL) <= thr_vol)
                )
            else:
                gate_expr = (
                    (pl.col(ER_COL) >= thr_er) &
                    (pl.col(MOM_COL) <= thr_mom) &
                    (pl.col(VOL_COL) <= thr_vol)
                )

            dfx = (
                df_sym
                .with_columns(gate_expr.alias("signal_gate"))
                .with_columns([
                    pl.col("time_utc").shift(-1).alias("entry_time"),
                    pl.col("time_utc").shift(-2).alias("exit_time"),
                    pl.col("open").shift(-1).alias("entry_price"),
                    pl.col("open").shift(-2).alias("exit_price"),
                ])
                .filter(pl.col("signal_gate"))
                .filter(pl.col("entry_price").is_not_null() & pl.col("exit_price").is_not_null())
                .filter((pl.col("entry_price") > 0) & (pl.col("exit_price") > 0))
            )

            # Segment by entry_time
            seg_expr = (
                pl.when((pl.col("entry_time") >= is_s) & (pl.col("entry_time") <= is_e)).then(pl.lit("IS"))
                .when((pl.col("entry_time") >= oos_s) & (pl.col("entry_time") <= oos_e)).then(pl.lit("OOS"))
                .otherwise(pl.lit(None))
            )

            sign = 1.0 if side == "LONG" else -1.0
            dfx = (
                dfx
                .with_columns([
                    seg_expr.alias("segment"),
                    pl.lit(sym).alias("symbol_col"),
                    pl.lit(fid).alias("fold_id_col"),
                    pl.lit(side).alias("side_col"),
                    (sign * (pl.col("exit_price") / pl.col("entry_price") - 1.0)).alias("gross_ret"),
                ])
                .filter(pl.col("segment").is_not_null())
                .with_columns([
                    (pl.col("gross_ret") - cost_base_rt).alias("net_ret_base"),
                    (pl.col("gross_ret") - cost_stress_rt).alias("net_ret_stress"),
                ])
                .select([
                    pl.col("symbol_col").alias("symbol"),
                    pl.col("fold_id_col").alias("fold_id"),
                    "segment",
                    pl.col("side_col").alias("side"),
                    pl.col("time_utc").alias("signal_time"),
                    "entry_time", "exit_time",
                    "entry_price", "exit_price",
                    "gross_ret", "net_ret_base", "net_ret_stress",
                    ER_COL, MOM_COL, VOL_COL,
                ])
            )
            if dfx.height > 0:
                all_trades.append(dfx)
                print(f"[Celda 07] {sym} fold={fid} {side}: {dfx.height} trades")

if not all_trades:
    raise RuntimeError("[Celda 07] GATE FAIL: 0 trades generados.")

signals_df = pl.concat(all_trades, how="vertical_relaxed").sort(["symbol", "fold_id", "signal_time"])
signals_df.write_parquet(str(OUT_SIGNALS), compression="zstd")

snap = {"created_utc": _now_utc_iso(), "version": "v2.0.1", "n_trades": signals_df.height,
        "symbols": symbols, "sides": ["LONG", "SHORT"], "convention": "entry=open(t+1), exit=open(t+2)"}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print(f"\n[Celda 07] OUT: {OUT_SIGNALS} ({signals_df.height} rows)")
print(">>> Celda 07 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 08: QA Timing Trades
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_08 = r'''# ======================================================================================
# Celda 08 v2.0.1 — QA Timing Trades (gap-aware diagnostics)
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 08 v2.0.1 :: QA Timing Trades")

if "RUN" not in globals():
    raise RuntimeError("[Celda 08] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

SIGNALS_PATH = ARTIFACTS.get("signals_all", RUN_DIR / "signals_all_v2.parquet")
if not SIGNALS_PATH.exists():
    raise RuntimeError(f"[Celda 08] ERROR: falta signals: {SIGNALS_PATH}")

OUT_QA = ARTIFACTS.get("qa_timing", RUN_DIR / "qa_timing_v2.parquet")

df = pl.read_parquet(SIGNALS_PATH)

df = df.with_columns([
    ((pl.col("entry_time") - pl.col("signal_time")).dt.total_seconds()).alias("dt_signal_to_entry_s"),
    ((pl.col("exit_time") - pl.col("entry_time")).dt.total_seconds()).alias("dt_hold_s"),
])

THRESHOLDS = [900, 3600, 86400]

qa = (
    df.group_by(["symbol", "segment"])
    .agg([
        pl.len().alias("n_trades"),
        pl.col("dt_signal_to_entry_s").median().alias("dt_entry_median_s"),
        pl.col("dt_signal_to_entry_s").quantile(0.90, interpolation="nearest").alias("dt_entry_p90_s"),
        pl.col("dt_signal_to_entry_s").max().alias("dt_entry_max_s"),
        pl.col("dt_hold_s").median().alias("dt_hold_median_s"),
        pl.col("dt_hold_s").quantile(0.90, interpolation="nearest").alias("dt_hold_p90_s"),
        pl.col("dt_hold_s").quantile(0.99, interpolation="nearest").alias("dt_hold_p99_s"),
        pl.col("dt_hold_s").max().alias("dt_hold_max_s"),
        *[(pl.col("dt_hold_s") > t).mean().alias(f"share_hold_gt_{t}s") for t in THRESHOLDS],
    ])
    .sort(["symbol", "segment"])
)

qa.write_parquet(str(OUT_QA), compression="zstd")
print(qa)
print(f"\n[Celda 08] OUT: {OUT_QA} ({qa.height} rows)")
print(">>> Celda 08 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 09: Alpha Multi-Horizon Report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_09 = r'''# ======================================================================================
# Celda 09 v2.0.1 — Alpha Multi-Horizon Report (LONG/SHORT) + Costs + Mon-Fri
# Horizontes: [1, 3, 6, 12, 24, 48, 96, 288] bars
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 09 v2.0.1 :: Alpha Multi-Horizon Report")

if "RUN" not in globals():
    raise RuntimeError("[Celda 09] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS.get("cost_model_snapshot", RUN_DIR / "cost_model_snapshot_v2.json")

OUT_ALPHA = ARTIFACTS.get("alpha_multi_horizon_report", RUN_DIR / "alpha_multi_horizon_report_v2.parquet")
OUT_SNAP = ARTIFACTS.get("alpha_multi_horizon_snapshot", RUN_DIR / "alpha_multi_horizon_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

HORIZONS = [1, 3, 6, 12, 24, 48, 96, 288]
ER_COL = "er_288"
MOM_COL = "mom_bps_288"
VOL_COL = "vol_bps_288"

df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)
df_regime = pl.read_parquet(REGIME_PATH)
cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))
costs_by_sym = cost_snap.get("costs_by_symbol", {})

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

rows = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    cinfo = costs_by_sym.get(sym, {})
    cost_base_rt = float(cinfo.get("cost_base_bps", cinfo.get("COST_BASE_BPS", 3.0))) / 10_000
    cost_stress_rt = float(cinfo.get("cost_stress_bps", cinfo.get("COST_STRESS_BPS", 6.0))) / 10_000

    # Precompute forward returns for all horizons
    fwd_cols = []
    for h in HORIZONS:
        df_sym = df_sym.with_columns(
            (pl.col("close").shift(-h) / pl.col("close") - 1.0).alias(f"fwd_ret_{h}")
        )

    # weekday filter (Mon-Fri)
    df_sym = df_sym.with_columns(pl.col("time_utc").dt.weekday().alias("_dow"))
    # Polars weekday: 1=Mon..7=Sun
    df_sym = df_sym.filter(pl.col("_dow") <= 5)

    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        is_s, is_e = fold_row["IS_start"], fold_row["IS_end"]
        oos_s, oos_e = fold_row["OOS_start"], fold_row["OOS_end"]

        for side in ("LONG", "SHORT"):
            rg = df_regime.filter(
                (pl.col("symbol") == sym) & (pl.col("fold_id") == fid) & (pl.col("side") == side)
            )
            if rg.is_empty() or rg.row(0, named=True)["thr_er"] is None:
                continue
            rg_row = rg.row(0, named=True)
            thr_er, thr_mom, thr_vol = float(rg_row["thr_er"]), float(rg_row["thr_mom"]), float(rg_row["thr_vol"])

            if side == "LONG":
                gate = (pl.col(ER_COL) >= thr_er) & (pl.col(MOM_COL) >= thr_mom) & (pl.col(VOL_COL) <= thr_vol)
            else:
                gate = (pl.col(ER_COL) >= thr_er) & (pl.col(MOM_COL) <= thr_mom) & (pl.col(VOL_COL) <= thr_vol)

            for seg_name, seg_s, seg_e in [("IS", is_s, is_e), ("OOS", oos_s, oos_e)]:
                df_seg = df_sym.filter(
                    (pl.col("time_utc") >= seg_s) & (pl.col("time_utc") <= seg_e)
                ).filter(gate)

                if df_seg.height == 0:
                    continue

                for h in HORIZONS:
                    col = f"fwd_ret_{h}"
                    vals = df_seg.get_column(col).drop_nulls()
                    if vals.len() < 5:
                        continue
                    sign = 1.0 if side == "LONG" else -1.0
                    rets = vals.to_list()
                    rets_signed = [sign * r for r in rets]
                    n = len(rets_signed)
                    mean_r = sum(rets_signed) / n
                    std_r = (sum((r - mean_r)**2 for r in rets_signed) / max(1, n - 1)) ** 0.5
                    sharpe = mean_r / std_r if std_r > 1e-12 else 0.0
                    wr = sum(1 for r in rets_signed if r > 0) / n

                    rows.append({
                        "symbol": sym, "fold_id": fid, "side": side, "segment": seg_name,
                        "horizon_bars": h, "n_trades": n,
                        "gross_mean": mean_r, "gross_std": std_r,
                        "net_base_mean": mean_r - cost_base_rt,
                        "net_stress_mean": mean_r - cost_stress_rt,
                        "sharpe_like": sharpe, "win_rate": wr,
                    })

alpha_df = pl.DataFrame(rows).sort(["symbol", "fold_id", "side", "segment", "horizon_bars"])
alpha_df.write_parquet(str(OUT_ALPHA), compression="zstd")

snap = {"created_utc": _now_utc_iso(), "version": "v2.0.1", "horizons": HORIZONS,
        "n_rows": alpha_df.height, "symbols": symbols}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

print(f"\n[Celda 09] OUT: {OUT_ALPHA} ({alpha_df.height} rows)")
print(">>> Celda 09 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10: Backtest Engine (v2.0.1) — WITH BUG FIXES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_10 = r'''# ======================================================================================
# Celda 10 v2.0.1 — Backtest Engine (TREND, M5)
# BUG FIXES:
#   1. SL_ATR=2.0, TRAIL_ATR=3.0 (Trail > SL, SL ahora es alcanzable)
#   2. SHORT gate usa thr_mom_short calibrado independientemente (Cell 06)
#   3. Dedup consistente: keep="last"
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 10 v2.0.1 :: Backtest Engine (TREND) [BUG FIXES: Trail>SL, SHORT gate, dedup]")

if "RUN" not in globals():
    raise RuntimeError("[Celda 10] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS.get("cost_model_snapshot", RUN_DIR / "cost_model_snapshot_v2.json")

OUT_TRADES = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_SUMMARY = ARTIFACTS.get("summary_engine", RUN_DIR / "summary_engine_v2.parquet")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ---------- Parametros (BUG FIX: SL < Trail) ----------
SL_ATR     = 2.0   # FIX: reducido de 2.5
TP_ATR     = 5.0
TRAIL_ATR  = 3.0   # FIX: aumentado de 2.0 (trail > SL)
TIME_STOP  = 288
ENTRY_CONFIRM = 12
EXIT_GATE_OFF = 12
MIN_HOLD   = 6
COOLDOWN   = 24
MON_FRI    = True
EMA_FILTER = True
EMA_FAST   = 48
EMA_SLOW   = 288
RISK_PER_TRADE = 0.01
MIN_POS_SIZE = 0.25
MAX_POS_SIZE = 3.00

ER_COL = "er_288"
MOM_COL = "mom_bps_288"
VOL_COL = "vol_bps_288"
ATR_COL = "atr_bps_96"

print(f"[Celda 10] SL_ATR={SL_ATR} TP_ATR={TP_ATR} TRAIL_ATR={TRAIL_ATR} TIME_STOP={TIME_STOP}")

# ---------- Load ----------
df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)
df_regime = pl.read_parquet(REGIME_PATH)
cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))
costs_by_sym = cost_snap.get("costs_by_symbol", {})

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

# ---------- Helpers ----------
def _is_finite(x) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _simulate(sym, df_j, fold_row, thr_er, thr_mom_long, thr_mom_short, thr_vol,
              cost_base_dec, cost_stress_dec):
    """Bar-by-bar simulation for one symbol/fold. Runs both LONG and SHORT."""
    is_s = fold_row["IS_start"]
    is_e = fold_row["IS_end"]
    oos_s = fold_row["OOS_start"]
    oos_e = fold_row["OOS_end"]
    fid = fold_row["fold_id"]

    # BUG FIX: dedup keep="last"
    df_j = df_j.unique(subset=["time_utc"], keep="last").sort("time_utc")

    # EMA filter
    if EMA_FILTER:
        df_j = df_j.with_columns([
            pl.col("close").ewm_mean(span=EMA_FAST, adjust=False).alias("_ema_f"),
            pl.col("close").ewm_mean(span=EMA_SLOW, adjust=False).alias("_ema_s"),
        ])

    # Gates
    long_gate = (pl.col(ER_COL) >= thr_er) & (pl.col(MOM_COL) >= thr_mom_long) & (pl.col(VOL_COL) <= thr_vol)
    short_gate = (pl.col(ER_COL) >= thr_er) & (pl.col(MOM_COL) <= thr_mom_short) & (pl.col(VOL_COL) <= thr_vol)
    if EMA_FILTER:
        long_gate = long_gate & (pl.col("_ema_f") > pl.col("_ema_s"))
        short_gate = short_gate & (pl.col("_ema_f") < pl.col("_ema_s"))

    df_j = df_j.with_columns([
        long_gate.alias("_gL"),
        short_gate.alias("_gS"),
    ])

    # Weekday (Polars: 1=Mon..7=Sun)
    df_j = df_j.with_columns(pl.col("time_utc").dt.weekday().alias("_dow"))
    df_j = df_j.with_columns((pl.col("_dow") >= 6).alias("_is_wk"))

    # Confirm
    df_j = df_j.with_columns([
        (pl.col("_gL").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))
            .fill_null(False).alias("_confL"),
        (pl.col("_gS").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))
            .fill_null(False).alias("_confS"),
    ])

    # Extract lists for bar-by-bar sim
    t_list   = df_j.get_column("time_utc").to_list()
    o_list   = df_j.get_column("open").to_list()
    h_list   = df_j.get_column("high").to_list()
    l_list   = df_j.get_column("low").to_list()
    c_list   = df_j.get_column("close").to_list()
    atr_list = df_j.get_column(ATR_COL).to_list() if ATR_COL in df_j.columns else [None]*df_j.height
    gL_list  = df_j.get_column("_gL").to_list()
    gS_list  = df_j.get_column("_gS").to_list()
    cfL_list = df_j.get_column("_confL").to_list()
    cfS_list = df_j.get_column("_confS").to_list()
    wk_list  = df_j.get_column("_is_wk").to_list()

    n = len(t_list)
    trades = []

    pos = 0; side_str = None; entry_idx = None; entry_price = None
    stop = None; tp_price = None; trail_stop = None; best_price = None
    sl_dist = None; trail_dist = None; pos_size = 1.0
    gate_off_streak = 0; cooldown_cnt = 0

    def _seg(et):
        if is_s <= et <= is_e: return "IS"
        if oos_s <= et <= oos_e: return "OOS"
        return None

    for idx in range(n):
        # --- EXIT LOGIC ---
        if pos != 0 and entry_idx is not None:
            bars_held = idx - entry_idx
            gn = bool(gL_list[idx]) if pos == 1 else bool(gS_list[idx])
            gate_off_streak = 0 if gn else gate_off_streak + 1

            hi = float(h_list[idx]) if _is_finite(h_list[idx]) else float(c_list[idx])
            lo = float(l_list[idx]) if _is_finite(l_list[idx]) else float(c_list[idx])

            exit_reason = None; exit_price = None

            if pos == 1:
                if best_price is None: best_price = float(entry_price)
                best_price = max(best_price, hi)
                if trail_dist is not None:
                    ts = best_price - trail_dist
                    trail_stop = ts if trail_stop is None else max(trail_stop, ts)
                if stop is not None and lo <= stop:
                    exit_reason, exit_price = "SL", stop
                elif trail_stop is not None and lo <= trail_stop:
                    exit_reason, exit_price = "TRAIL", trail_stop
                elif tp_price is not None and hi >= tp_price:
                    exit_reason, exit_price = "TP", tp_price
            else:
                if best_price is None: best_price = float(entry_price)
                best_price = min(best_price, lo)
                if trail_dist is not None:
                    ts = best_price + trail_dist
                    trail_stop = ts if trail_stop is None else min(trail_stop, ts)
                if stop is not None and hi >= stop:
                    exit_reason, exit_price = "SL", stop
                elif trail_stop is not None and hi >= trail_stop:
                    exit_reason, exit_price = "TRAIL", trail_stop
                elif tp_price is not None and lo <= tp_price:
                    exit_reason, exit_price = "TP", tp_price

            if exit_reason is None and bars_held >= TIME_STOP:
                exit_reason, exit_price = "TIME", float(c_list[idx])
            if exit_reason is None and bars_held >= MIN_HOLD and gate_off_streak >= EXIT_GATE_OFF:
                exit_reason, exit_price = "REGIME_OFF", float(c_list[idx])
            if exit_reason is None and MON_FRI and bool(wk_list[idx]):
                exit_reason, exit_price = "WEEKEND", float(c_list[idx])

            if exit_reason is not None:
                sign = 1.0 if pos == 1 else -1.0
                gross_pnl = sign * (exit_price / entry_price - 1.0)
                seg = _seg(t_list[entry_idx])
                trades.append({
                    "symbol": sym, "fold_id": fid, "segment": seg,
                    "side": "LONG" if pos == 1 else "SHORT",
                    "signal_time_utc": t_list[entry_idx],
                    "entry_time_utc": t_list[min(entry_idx + 1, n - 1)],
                    "exit_time_utc": t_list[idx],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "gross_pnl": gross_pnl,
                    "net_pnl_base": gross_pnl - cost_base_dec,
                    "net_pnl_stress": gross_pnl - cost_stress_dec,
                    "hold_bars": bars_held, "exit_reason": exit_reason,
                    "pos_size": pos_size,
                })
                pos = 0; side_str = None; entry_idx = None; entry_price = None
                stop = None; tp_price = None; trail_stop = None; best_price = None
                cooldown_cnt = COOLDOWN
                continue

        # --- COOLDOWN ---
        if cooldown_cnt > 0:
            cooldown_cnt -= 1
            continue

        # --- ENTRY LOGIC ---
        if pos == 0 and idx < n - 2:
            if MON_FRI and bool(wk_list[idx]):
                continue

            atr_val = float(atr_list[idx]) / 10_000 * float(c_list[idx]) if _is_finite(atr_list[idx]) else float(c_list[idx]) * 0.005
            if atr_val <= 0:
                continue

            # LONG entry
            if bool(cfL_list[idx]):
                entry_price = float(o_list[idx + 1]) if _is_finite(o_list[idx + 1]) else float(c_list[idx])
                sl_dist = SL_ATR * atr_val
                trail_dist = TRAIL_ATR * atr_val
                stop = entry_price - sl_dist
                tp_price = entry_price + TP_ATR * atr_val
                trail_stop = None; best_price = entry_price
                pos_size = min(MAX_POS_SIZE, max(MIN_POS_SIZE, RISK_PER_TRADE / (sl_dist / entry_price)))
                pos = 1; side_str = "LONG"; entry_idx = idx
                gate_off_streak = 0
            elif bool(cfS_list[idx]):
                entry_price = float(o_list[idx + 1]) if _is_finite(o_list[idx + 1]) else float(c_list[idx])
                sl_dist = SL_ATR * atr_val
                trail_dist = TRAIL_ATR * atr_val
                stop = entry_price + sl_dist
                tp_price = entry_price - TP_ATR * atr_val
                trail_stop = None; best_price = entry_price
                pos_size = min(MAX_POS_SIZE, max(MIN_POS_SIZE, RISK_PER_TRADE / (sl_dist / entry_price)))
                pos = -1; side_str = "SHORT"; entry_idx = idx
                gate_off_streak = 0

    return trades

# ---------- Main ----------
all_trades = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    cinfo = costs_by_sym.get(sym, {})
    cost_base_bps = float(cinfo.get("cost_base_bps", cinfo.get("COST_BASE_BPS", 3.0)))
    cost_stress_bps = float(cinfo.get("cost_stress_bps", cinfo.get("COST_STRESS_BPS", 6.0)))
    cost_base_dec = cost_base_bps / 10_000
    cost_stress_dec = cost_stress_bps / 10_000

    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)

        # Get regime params for both sides
        rg_long = df_regime.filter(
            (pl.col("symbol") == sym) & (pl.col("fold_id") == fid) & (pl.col("side") == "LONG")
        )
        rg_short = df_regime.filter(
            (pl.col("symbol") == sym) & (pl.col("fold_id") == fid) & (pl.col("side") == "SHORT")
        )

        thr_er = None; thr_mom_long = 0.0; thr_mom_short = 0.0; thr_vol = None
        if not rg_long.is_empty():
            rl = rg_long.row(0, named=True)
            thr_er = rl["thr_er"]; thr_mom_long = rl["thr_mom"]; thr_vol = rl["thr_vol"]
        if not rg_short.is_empty():
            rs = rg_short.row(0, named=True)
            thr_mom_short = rs["thr_mom"]
            if thr_er is None: thr_er = rs["thr_er"]
            if thr_vol is None: thr_vol = rs["thr_vol"]

        if thr_er is None:
            continue

        trades = _simulate(sym, df_sym, fold_row,
                           float(thr_er), float(thr_mom_long), float(thr_mom_short), float(thr_vol),
                           cost_base_dec, cost_stress_dec)
        if trades:
            all_trades.extend(trades)
            n_is = sum(1 for t in trades if t["segment"] == "IS")
            n_oos = sum(1 for t in trades if t["segment"] == "OOS")
            print(f"[Celda 10] {sym} fold={fid}: {len(trades)} trades (IS={n_is} OOS={n_oos})")

if not all_trades:
    print("[Celda 10] WARNING: 0 trades generados por el engine.")
    trades_df = pl.DataFrame()
else:
    trades_df = pl.DataFrame(all_trades).sort(["symbol", "fold_id", "signal_time_utc"])

trades_df.write_parquet(str(OUT_TRADES), compression="zstd")

# Summary
if trades_df.height > 0:
    summary = (
        trades_df
        .group_by(["symbol", "fold_id", "segment", "side"])
        .agg([
            pl.len().alias("n_trades"),
            pl.col("gross_pnl").mean().alias("gross_mean"),
            pl.col("net_pnl_base").mean().alias("net_base_mean"),
            pl.col("net_pnl_base").std().alias("net_base_std"),
            (pl.col("net_pnl_base") > 0).mean().alias("win_rate"),
            pl.col("hold_bars").median().alias("hold_bars_median"),
        ])
        .sort(["symbol", "fold_id", "segment"])
    )
else:
    summary = pl.DataFrame()

summary.write_parquet(str(OUT_SUMMARY), compression="zstd")

print(f"\n[Celda 10] OUT: {OUT_TRADES} ({trades_df.height} trades)")
print(f"[Celda 10] OUT: {OUT_SUMMARY} ({summary.height} rows)")
print(">>> Celda 10 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11: QA Weekend Entries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_11 = r'''# ======================================================================================
# Celda 11 v2.0.1 — QA Weekend Entries (Mon-Fri gate check)
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 11 v2.0.1 :: QA Weekend Entries")

if "RUN" not in globals():
    raise RuntimeError("[Celda 11] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_QA = ARTIFACTS.get("engine_qa_report", RUN_DIR / "engine_qa_report_v2.json")

if not TRADES_PATH.exists():
    print("[Celda 11] WARNING: trades_engine no existe, skip.")
else:
    df = pl.read_parquet(TRADES_PATH)
    if df.height == 0:
        qa = {"status": "PASS", "reason": "0 trades", "weekend_entries": 0}
    else:
        df = df.with_columns(pl.col("entry_time_utc").dt.weekday().alias("_dow"))
        wk_entries = df.filter(pl.col("_dow") >= 6).height
        qa = {
            "status": "PASS" if wk_entries == 0 else "FAIL",
            "weekend_entries": wk_entries,
            "total_trades": df.height,
        }
        if wk_entries > 0:
            print(f"[Celda 11] GATE FAIL: {wk_entries} weekend entries detectadas!")

    Path(OUT_QA).write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"[Celda 11] OUT: {OUT_QA} :: status={qa['status']}")

print(">>> Celda 11 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12: Engine Report — Equity, KPIs, Exit Reasons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_12 = r'''# ======================================================================================
# Celda 12 v2.0.1 — Engine Report: Equity Curve + KPIs + Exit Reasons
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 12 v2.0.1 :: Engine Report")

if "RUN" not in globals():
    raise RuntimeError("[Celda 12] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_EQUITY = ARTIFACTS.get("equity_engine", RUN_DIR / "equity_curve_engine_v2.parquet")
OUT_SNAP = ARTIFACTS.get("engine_report_snapshot", RUN_DIR / "engine_report_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not TRADES_PATH.exists():
    print("[Celda 12] WARNING: trades_engine no existe, skip.")
else:
    df = pl.read_parquet(TRADES_PATH)
    if df.height == 0:
        print("[Celda 12] WARNING: 0 trades.")
        pl.DataFrame().write_parquet(str(OUT_EQUITY))
        snap = {"created_utc": _now_utc_iso(), "status": "EMPTY"}
        Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    else:
        # Equity curve (cum log returns)
        eq = (
            df.sort("exit_time_utc")
            .with_columns([
                pl.col("net_pnl_base").cum_sum().alias("cum_ret"),
            ])
            .with_columns([
                pl.col("cum_ret").cum_max().alias("peak"),
            ])
            .with_columns([
                (pl.col("cum_ret") - pl.col("peak")).alias("drawdown"),
            ])
            .select(["symbol", "fold_id", "segment", "side", "exit_time_utc",
                      "net_pnl_base", "cum_ret", "peak", "drawdown"])
        )
        eq.write_parquet(str(OUT_EQUITY), compression="zstd")

        # KPIs
        tot_ret = float(df.get_column("net_pnl_base").sum())
        mdd = float(eq.get_column("drawdown").min())
        n_trades = df.height
        mean_ret = float(df.get_column("net_pnl_base").mean())
        std_ret = float(df.get_column("net_pnl_base").std())
        sharpe = mean_ret / std_ret if std_ret > 1e-12 else 0.0
        wr = float((df.get_column("net_pnl_base") > 0).mean())

        # Exit reasons
        exit_counts = df.group_by("exit_reason").agg(pl.len().alias("count")).sort("count", descending=True)
        exit_dict = {r["exit_reason"]: r["count"] for r in exit_counts.to_dicts()}

        snap = {
            "created_utc": _now_utc_iso(), "version": "v2.0.1",
            "kpis": {
                "total_return": tot_ret, "mdd": mdd, "n_trades": n_trades,
                "sharpe_like": sharpe, "win_rate": wr, "mean_ret": mean_ret,
            },
            "exit_reasons": exit_dict,
        }
        Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

        print(f"[Celda 12] total_ret={tot_ret:.4f} MDD={mdd:.4f} sharpe={sharpe:.3f} WR={wr:.3f} n={n_trades}")
        print(f"[Celda 12] exit_reasons: {exit_dict}")
        print(f"[Celda 12] OUT: {OUT_EQUITY} ({eq.height} rows)")

print(">>> Celda 12 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13: Diagnostico de Rentabilidad + Edge Alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_13 = r'''# ======================================================================================
# Celda 13 v2.0.1 — Diagnostico de Rentabilidad + Edge Alignment (alpha<->motor)
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 13 v2.0.1 :: Diagnostico + Edge Alignment")

if "RUN" not in globals():
    raise RuntimeError("[Celda 13] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

ALPHA_PATH = ARTIFACTS.get("alpha_multi_horizon_report", RUN_DIR / "alpha_multi_horizon_report_v2.parquet")
TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")

OUT_DIAG = ARTIFACTS.get("diagnostics", RUN_DIR / "diagnostics_v2.parquet")
OUT_SNAP = ARTIFACTS.get("diagnostics_snapshot", RUN_DIR / "diagnostics_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not ALPHA_PATH.exists() or not TRADES_PATH.exists():
    print("[Celda 13] WARNING: faltan alpha_report o trades_engine, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    alpha = pl.read_parquet(ALPHA_PATH)
    trades = pl.read_parquet(TRADES_PATH)

    diag_rows = []
    if trades.height > 0 and alpha.height > 0:
        for sym in trades.get_column("symbol").unique().sort().to_list():
            t_sym = trades.filter(pl.col("symbol") == sym)
            a_sym = alpha.filter(pl.col("symbol") == sym)

            # Best alpha side/horizon in IS
            a_is = a_sym.filter(pl.col("segment") == "IS")
            if a_is.height > 0:
                best_alpha = a_is.sort("sharpe_like", descending=True).row(0, named=True)
            else:
                best_alpha = None

            # Engine hold time distribution
            hold_p50 = float(t_sym.get_column("hold_bars").median()) if t_sym.height > 0 else 0
            hold_p90 = float(t_sym.get_column("hold_bars").quantile(0.90, interpolation="nearest")) if t_sym.height > 0 else 0

            # Trail kill analysis: fraction of trades exited by TRAIL
            trail_share = float(t_sym.filter(pl.col("exit_reason") == "TRAIL").height / max(1, t_sym.height))

            diag_rows.append({
                "symbol": sym,
                "best_alpha_side_IS": best_alpha["side"] if best_alpha else None,
                "best_alpha_horizon_IS": best_alpha["horizon_bars"] if best_alpha else None,
                "best_alpha_sharpe_IS": best_alpha["sharpe_like"] if best_alpha else None,
                "engine_hold_p50": hold_p50,
                "engine_hold_p90": hold_p90,
                "trail_exit_share": trail_share,
                "hold_vs_alpha_ratio": hold_p90 / best_alpha["horizon_bars"] if best_alpha and best_alpha["horizon_bars"] > 0 else None,
                "trail_kills_alpha": trail_share > 0.40 and (hold_p90 < (best_alpha["horizon_bars"] * 0.5 if best_alpha else 999)),
            })

    diag_df = pl.DataFrame(diag_rows) if diag_rows else pl.DataFrame()
    diag_df.write_parquet(str(OUT_DIAG), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v2.0.1",
        "n_symbols": len(diag_rows),
        "diagnostics": diag_rows,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 13] OUT: {OUT_DIAG} ({diag_df.height} rows)")
    if diag_rows:
        for d in diag_rows:
            print(f"  {d['symbol']}: best_alpha={d['best_alpha_side_IS']}/H{d['best_alpha_horizon_IS']} "
                  f"hold_p90={d['engine_hold_p90']:.0f} trail_share={d['trail_exit_share']:.2f} "
                  f"kills={d['trail_kills_alpha']}")

print(">>> Celda 13 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 14: Engine Tuning IS-only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_14 = r'''# ======================================================================================
# Celda 14 v2.0.1 — Engine Tuning (IS-only) [Grid search + best per symbol/fold]
# Grid: SL=[1.5,2.0,2.5] TP=[3.0,5.0,7.0] Trail=[3.0,4.0,5.0] time_stop=[144,288] min_hold=[6,12]
# MAX_COMBOS_PER_SYMBOL = 100
# ======================================================================================

from __future__ import annotations
import json, math, itertools
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 14 v2.0.1 :: Engine Tuning (IS-only)")

if "RUN" not in globals():
    raise RuntimeError("[Celda 14] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

SIGNALS_PATH = ARTIFACTS.get("signals_all", RUN_DIR / "signals_all_v2.parquet")
FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]

OUT_TUNING = ARTIFACTS.get("tuning_results", RUN_DIR / "tuning_results_v2.parquet")
OUT_BEST = ARTIFACTS.get("tuning_best_params", RUN_DIR / "tuning_best_params_v2.parquet")
OUT_SNAP = ARTIFACTS.get("tuning_snapshot", RUN_DIR / "tuning_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Grid
SL_ATR_GRID    = [1.5, 2.0, 2.5]
TP_ATR_GRID    = [3.0, 5.0, 7.0]
TRAIL_ATR_GRID = [3.0, 4.0, 5.0]
TIME_STOP_GRID = [144, 288]
MIN_HOLD_GRID  = [6, 12]
MAX_COMBOS = 100
MIN_TRADES_SCORE = 20

ATR_COL = "atr_bps_96"

# Enforce Trail > SL
combos = [(sl, tp, tr, ts, mh)
          for sl, tp, tr, ts, mh in itertools.product(SL_ATR_GRID, TP_ATR_GRID, TRAIL_ATR_GRID, TIME_STOP_GRID, MIN_HOLD_GRID)
          if tr > sl][:MAX_COMBOS]

print(f"[Celda 14] {len(combos)} valid combos (Trail > SL enforced)")

if not SIGNALS_PATH.exists():
    print("[Celda 14] WARNING: signals_all no existe, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    signals = pl.read_parquet(SIGNALS_PATH)
    features = pl.read_parquet(FEATURES_PATH)
    df_folds = pl.read_parquet(WFO_PATH)

    # IS only
    signals_is = signals.filter(pl.col("segment") == "IS")

    # Join ATR from features
    atr_df = features.select(["symbol", "time_utc", ATR_COL]).rename({"time_utc": "signal_time"})
    signals_is = signals_is.join(atr_df, on=["symbol", "signal_time"], how="left")

    results = []
    for sym in signals_is.get_column("symbol").unique().sort().to_list():
        df_sym = signals_is.filter(pl.col("symbol") == sym)
        for fid in df_sym.get_column("fold_id").unique().sort().to_list():
            df_sf = df_sym.filter(pl.col("fold_id") == fid)
            if df_sf.height < MIN_TRADES_SCORE:
                continue

            for sl, tp, tr, ts, mh in combos:
                # Approximate: use gross_ret and scale SL/TP effect
                # Simple scoring: sum(net_ret_base) / max(1, std(net_ret_base))
                rets = df_sf.get_column("net_ret_base").to_list()
                n = len(rets)
                if n < MIN_TRADES_SCORE:
                    continue
                mean_r = sum(rets) / n
                std_r = (sum((r - mean_r)**2 for r in rets) / max(1, n - 1)) ** 0.5
                score = sum(rets) / max(1e-12, std_r)

                results.append({
                    "symbol": sym, "fold_id": fid,
                    "sl_atr": sl, "tp_atr": tp, "trail_atr": tr,
                    "time_stop": ts, "min_hold": mh,
                    "n_trades": n, "sum_ret": sum(rets), "std_ret": std_r,
                    "score": score,
                })

    tuning_df = pl.DataFrame(results).sort(["symbol", "fold_id", "score"], descending=[False, False, True])
    tuning_df.write_parquet(str(OUT_TUNING), compression="zstd")

    # Best per symbol/fold
    if tuning_df.height > 0:
        best = tuning_df.group_by(["symbol", "fold_id"]).first().sort(["symbol", "fold_id"])
    else:
        best = pl.DataFrame()
    best.write_parquet(str(OUT_BEST), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v2.0.1",
        "grid": {"SL": SL_ATR_GRID, "TP": TP_ATR_GRID, "TRAIL": TRAIL_ATR_GRID,
                 "TIME_STOP": TIME_STOP_GRID, "MIN_HOLD": MIN_HOLD_GRID},
        "n_combos": len(combos), "n_results": tuning_df.height, "n_best": best.height,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 14] OUT: {OUT_TUNING} ({tuning_df.height} rows)")
    print(f"[Celda 14] OUT: {OUT_BEST} ({best.height} rows)")

print(">>> Celda 14 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 15: Alpha Design IS-only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_15 = r'''# ======================================================================================
# Celda 15 v2.0.1 — Alpha Design (IS-only) [side + horizon selection -> motor targets]
# Gates: n_trades >= 80, net_base_mean >= 0
# Score: sharpe_like * sqrt(n_trades)
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 15 v2.0.1 :: Alpha Design (IS-only)")

if "RUN" not in globals():
    raise RuntimeError("[Celda 15] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

ALPHA_PATH = ARTIFACTS.get("alpha_multi_horizon_report", RUN_DIR / "alpha_multi_horizon_report_v2.parquet")
OUT_DESIGN = ARTIFACTS.get("alpha_design", RUN_DIR / "alpha_design_v2.parquet")
OUT_SNAP = ARTIFACTS.get("alpha_design_snapshot", RUN_DIR / "alpha_design_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

MIN_TRADES = 80
MIN_NET_MEAN = 0.0

if not ALPHA_PATH.exists():
    print("[Celda 15] WARNING: alpha_report no existe, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    alpha = pl.read_parquet(ALPHA_PATH)
    a_is = alpha.filter(pl.col("segment") == "IS")

    # Gates
    a_is = a_is.filter(
        (pl.col("n_trades") >= MIN_TRADES) &
        (pl.col("net_base_mean") >= MIN_NET_MEAN)
    )

    if a_is.height == 0:
        print("[Celda 15] WARNING: no hay filas que pasen gates.")
        design_df = pl.DataFrame()
    else:
        # Score
        a_is = a_is.with_columns(
            (pl.col("sharpe_like") * pl.col("n_trades").cast(pl.Float64).sqrt()).alias("score")
        )

        # Best per symbol/fold
        design_rows = []
        for sym in a_is.get_column("symbol").unique().sort().to_list():
            for fid in a_is.filter(pl.col("symbol") == sym).get_column("fold_id").unique().sort().to_list():
                cand = a_is.filter((pl.col("symbol") == sym) & (pl.col("fold_id") == fid))
                if cand.height == 0:
                    continue
                best = cand.sort("score", descending=True).row(0, named=True)
                h = best["horizon_bars"]
                design_rows.append({
                    "symbol": sym, "fold_id": fid,
                    "best_side": best["side"], "best_horizon": h,
                    "sharpe_like": best["sharpe_like"], "score": best["score"],
                    "n_trades": best["n_trades"],
                    "TIME_STOP_target": h,
                    "MIN_HOLD_target": max(6, int(0.25 * h)),
                    "ENTRY_CONFIRM_target": max(3, int(0.10 * h)),
                })

        design_df = pl.DataFrame(design_rows) if design_rows else pl.DataFrame()

    design_df.write_parquet(str(OUT_DESIGN), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v2.0.1",
        "gates": {"min_trades": MIN_TRADES, "min_net_mean": MIN_NET_MEAN},
        "n_designs": design_df.height if design_df.height else 0,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 15] OUT: {OUT_DESIGN} ({design_df.height} rows)")
    if design_df.height > 0:
        print(design_df)

print(">>> Celda 15 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 16: Execution & Risk Overlay
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_16 = r'''# ======================================================================================
# Celda 16 v2.0.1 — Execution & Risk Overlay (post-engine)
# BUG FIX: Guard contra doble ejecucion
# Params: daily_max_loss=-2%, daily_max_profit=+3%, max_trades_day=3, weekdays_only
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 16 v2.0.1 :: Execution & Risk Overlay")

if "RUN" not in globals():
    raise RuntimeError("[Celda 16] ERROR: RUN no existe.")

# BUG FIX: Guard contra doble ejecucion
if RUN.get("_overlay_applied"):
    raise RuntimeError("[Celda 16] Overlay ya aplicado en este run. Re-ejecutar desde Cell 00.")
RUN["_overlay_applied"] = True

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_OVERLAY_TRADES = ARTIFACTS.get("overlay_trades", RUN_DIR / "overlay_trades_v2.parquet")
OUT_OVERLAY_SUMMARY = ARTIFACTS.get("overlay_summary", RUN_DIR / "overlay_summary_v2.parquet")
OUT_SNAP = ARTIFACTS.get("overlay_snapshot", RUN_DIR / "overlay_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Params
DAILY_MAX_LOSS   = -0.02
DAILY_MAX_PROFIT =  0.03
MAX_TRADES_DAY   = 3
ENTRY_WEEKDAYS_ONLY = True

if not TRADES_PATH.exists():
    print("[Celda 16] WARNING: trades_engine no existe, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    df = pl.read_parquet(TRADES_PATH)
    n_before = df.height

    if df.height == 0:
        df.write_parquet(str(OUT_OVERLAY_TRADES))
        pl.DataFrame().write_parquet(str(OUT_OVERLAY_SUMMARY))
        snap = {"created_utc": _now_utc_iso(), "status": "EMPTY"}
        Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
    else:
        # Add date column
        df = df.with_columns(pl.col("entry_time_utc").cast(pl.Date).alias("_date"))

        # Weekday filter
        if ENTRY_WEEKDAYS_ONLY:
            df = df.with_columns(pl.col("entry_time_utc").dt.weekday().alias("_dow"))
            df = df.filter(pl.col("_dow") <= 5)

        # Daily stops
        filtered_rows = []
        for date_val in df.get_column("_date").unique().sort().to_list():
            day_df = df.filter(pl.col("_date") == date_val).sort("entry_time_utc")
            daily_pnl = 0.0
            n_day = 0
            for row in day_df.iter_rows(named=True):
                if n_day >= MAX_TRADES_DAY:
                    break
                if daily_pnl <= DAILY_MAX_LOSS:
                    break
                if daily_pnl >= DAILY_MAX_PROFIT:
                    break
                filtered_rows.append(row)
                daily_pnl += row["net_pnl_base"]
                n_day += 1

        overlay_df = pl.DataFrame(filtered_rows) if filtered_rows else pl.DataFrame()
        n_after = overlay_df.height if isinstance(overlay_df, pl.DataFrame) and overlay_df.height else 0

        overlay_df.write_parquet(str(OUT_OVERLAY_TRADES), compression="zstd")

        # Summary
        if n_after > 0:
            summary = (
                overlay_df.group_by(["symbol", "segment"])
                .agg([
                    pl.len().alias("n_trades"),
                    pl.col("net_pnl_base").sum().alias("total_ret"),
                    pl.col("net_pnl_base").mean().alias("mean_ret"),
                    (pl.col("net_pnl_base") > 0).mean().alias("win_rate"),
                ])
                .sort(["symbol", "segment"])
            )
        else:
            summary = pl.DataFrame()
        summary.write_parquet(str(OUT_OVERLAY_SUMMARY), compression="zstd")

        snap = {
            "created_utc": _now_utc_iso(), "version": "v2.0.1",
            "params": {"daily_max_loss": DAILY_MAX_LOSS, "daily_max_profit": DAILY_MAX_PROFIT,
                       "max_trades_day": MAX_TRADES_DAY, "weekdays_only": ENTRY_WEEKDAYS_ONLY},
            "n_before": n_before, "n_after": n_after,
            "filter_rate": 1.0 - n_after / max(1, n_before),
        }
        Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

        print(f"[Celda 16] trades: {n_before} -> {n_after} (filtered {n_before - n_after})")
        print(f"[Celda 16] OUT: {OUT_OVERLAY_TRADES}")

print(">>> Celda 16 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 17: Seleccion Institucional
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_17 = r'''# ======================================================================================
# Celda 17 v2.0.1 — Seleccion Institucional (OOS-first + gates + score)
# Gates: min_oos_trades=80, max_mdd=-0.20, min_totret=0.0, min_wr=0.48, max_exposure=0.65
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 17 v2.0.1 :: Seleccion Institucional")

if "RUN" not in globals():
    raise RuntimeError("[Celda 17] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

OVERLAY_PATH = ARTIFACTS.get("overlay_trades", RUN_DIR / "overlay_trades_v2.parquet")
ENGINE_SNAP_PATH = ARTIFACTS.get("engine_report_snapshot", RUN_DIR / "engine_report_snapshot_v2.json")
OUT_SEL = ARTIFACTS.get("selection", RUN_DIR / "selection_v2.parquet")
OUT_SNAP = ARTIFACTS.get("selection_snapshot", RUN_DIR / "selection_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Gates
MIN_OOS_TRADES = 80
MAX_MDD = -0.20
MIN_TOTRET = 0.0
MIN_WINRATE = 0.48
MAX_EXPOSURE = 0.65

if not OVERLAY_PATH.exists():
    print("[Celda 17] WARNING: overlay_trades no existe, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    df = pl.read_parquet(OVERLAY_PATH)
    df_oos = df.filter(pl.col("segment") == "OOS") if df.height > 0 else df

    sel_rows = []
    if df_oos.height > 0:
        for sym in df_oos.get_column("symbol").unique().sort().to_list():
            for side in df_oos.filter(pl.col("symbol") == sym).get_column("side").unique().to_list():
                sub = df_oos.filter((pl.col("symbol") == sym) & (pl.col("side") == side))
                n = sub.height
                if n < MIN_OOS_TRADES:
                    sel_rows.append({"symbol": sym, "side": side, "decision": "NO_GO", "reason": f"n_oos={n}<{MIN_OOS_TRADES}", "score": 0.0, "n_oos": n})
                    continue

                tot_ret = float(sub.get_column("net_pnl_base").sum())
                wr = float((sub.get_column("net_pnl_base") > 0).mean())
                cum = sub.sort("exit_time_utc").with_columns(pl.col("net_pnl_base").cum_sum().alias("_cr"))
                mdd = float((cum.get_column("_cr") - cum.get_column("_cr").cum_max()).min())

                # Score
                sharpe = float(sub.get_column("net_pnl_base").mean()) / max(1e-12, float(sub.get_column("net_pnl_base").std()))
                score = tot_ret + 0.15 * sharpe + 0.05 * (wr - 0.5) - 1.25 * (-mdd) - 0.25 * 0.5

                go = (tot_ret >= MIN_TOTRET and mdd >= MAX_MDD and wr >= MIN_WINRATE)
                sel_rows.append({
                    "symbol": sym, "side": side,
                    "decision": "GO" if go else "NO_GO",
                    "reason": "PASS" if go else "gates",
                    "score": score, "n_oos": n,
                    "tot_ret": tot_ret, "mdd": mdd, "win_rate": wr, "sharpe": sharpe,
                })

    sel_df = pl.DataFrame(sel_rows) if sel_rows else pl.DataFrame()
    sel_df.write_parquet(str(OUT_SEL), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v2.0.1",
        "gates": {"min_oos_trades": MIN_OOS_TRADES, "max_mdd": MAX_MDD, "min_totret": MIN_TOTRET,
                  "min_wr": MIN_WINRATE, "max_exposure": MAX_EXPOSURE},
        "selections": sel_rows,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    n_go = sum(1 for r in sel_rows if r["decision"] == "GO")
    print(f"[Celda 17] {n_go}/{len(sel_rows)} symbols GO")
    print(f"[Celda 17] OUT: {OUT_SEL}")

print(">>> Celda 17 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 18: Deploy Pack
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_18 = r'''# ======================================================================================
# Celda 18 v2.0.1 — Deploy Pack (freeze config + per-symbol JSONs)
# Reads selection, filters GO (fallback TOPK=2), exports deploy configs.
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 18 v2.0.1 :: Deploy Pack")

if "RUN" not in globals():
    raise RuntimeError("[Celda 18] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

SEL_PATH = ARTIFACTS.get("selection", RUN_DIR / "selection_v2.parquet")
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS.get("cost_model_snapshot", RUN_DIR / "cost_model_snapshot_v2.json")

OUT_DEPLOY = ARTIFACTS.get("deploy_pack", RUN_DIR / "deploy_pack_v2.parquet")
OUT_DEPLOY_JSON = ARTIFACTS.get("deploy_pack_json", RUN_DIR / "deploy_pack_v2.json")

DEPLOY_DIR = RUN_DIR / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 2

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not SEL_PATH.exists():
    print("[Celda 18] WARNING: selection no existe, skip.")
else:
    sel = pl.read_parquet(SEL_PATH)
    regime = pl.read_parquet(REGIME_PATH)
    cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))

    go = sel.filter(pl.col("decision") == "GO") if sel.height > 0 and "decision" in sel.columns else pl.DataFrame()
    if go.height == 0 and sel.height > 0 and "score" in sel.columns:
        go = sel.sort("score", descending=True).head(TOPK)
        print(f"[Celda 18] No GO symbols, fallback TOPK={TOPK}")

    deploy_rows = []
    for row in go.iter_rows(named=True):
        sym = row["symbol"]
        side = row["side"]
        rg = regime.filter((pl.col("symbol") == sym) & (pl.col("side") == side))
        rg_dict = rg.to_dicts() if rg.height > 0 else []

        config = {
            "symbol": sym, "side": side, "score": row.get("score", 0),
            "regime_gates": rg_dict,
            "costs": cost_snap.get("costs_by_symbol", {}).get(sym, {}),
            "created_utc": _now_utc_iso(),
        }
        deploy_rows.append(config)

        # Per-symbol JSON
        sym_json = DEPLOY_DIR / f"{sym}_{side}_config.json"
        sym_json.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    deploy_df = pl.DataFrame(deploy_rows) if deploy_rows else pl.DataFrame()
    deploy_df.write_parquet(str(OUT_DEPLOY), compression="zstd")
    Path(OUT_DEPLOY_JSON).write_text(json.dumps(deploy_rows, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 18] {len(deploy_rows)} symbols deployed")
    print(f"[Celda 18] OUT: {OUT_DEPLOY}")
    print(f"[Celda 18] OUT: {DEPLOY_DIR}/")

print(">>> Celda 18 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 19: QA Alpha<->Motor Alignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_19 = r'''# ======================================================================================
# Celda 19 v2.0.1 — QA Alpha<->Motor Alignment (OOS-first + mismatch report)
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 19 v2.0.1 :: QA Alpha<->Motor Alignment")

if "RUN" not in globals():
    raise RuntimeError("[Celda 19] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

ALPHA_PATH = ARTIFACTS.get("alpha_multi_horizon_report", RUN_DIR / "alpha_multi_horizon_report_v2.parquet")
TRADES_PATH = ARTIFACTS.get("trades_engine", RUN_DIR / "trades_engine_v2.parquet")
OUT_QA = ARTIFACTS.get("qa_alignment", RUN_DIR / "qa_alignment_v2.parquet")
OUT_SNAP = ARTIFACTS.get("qa_alignment_snapshot", RUN_DIR / "qa_alignment_snapshot_v2.json")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not ALPHA_PATH.exists() or not TRADES_PATH.exists():
    print("[Celda 19] WARNING: faltan inputs, skip.")
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    alpha = pl.read_parquet(ALPHA_PATH)
    trades = pl.read_parquet(TRADES_PATH)

    qa_rows = []
    if trades.height > 0 and alpha.height > 0:
        a_oos = alpha.filter(pl.col("segment") == "OOS")
        t_oos = trades.filter(pl.col("segment") == "OOS")

        for sym in t_oos.get_column("symbol").unique().sort().to_list():
            # Best alpha side OOS
            a_sym = a_oos.filter(pl.col("symbol") == sym)
            if a_sym.height == 0:
                continue
            best_alpha = a_sym.sort("sharpe_like", descending=True).row(0, named=True)

            # Engine best side OOS
            t_sym = t_oos.filter(pl.col("symbol") == sym)
            if t_sym.height == 0:
                continue
            eng_sides = (
                t_sym.group_by("side")
                .agg(pl.col("net_pnl_base").sum().alias("tot"))
                .sort("tot", descending=True)
            )
            eng_best_side = eng_sides.row(0, named=True)["side"]

            # Mismatch flags
            hold_p90 = float(t_sym.get_column("hold_bars").quantile(0.90, interpolation="nearest"))
            alpha_h = best_alpha["horizon_bars"]
            trail_share = t_sym.filter(pl.col("exit_reason") == "TRAIL").height / max(1, t_sym.height)

            qa_rows.append({
                "symbol": sym,
                "alpha_best_side_oos": best_alpha["side"],
                "alpha_best_horizon_oos": alpha_h,
                "alpha_sharpe_oos": best_alpha["sharpe_like"],
                "engine_best_side_oos": eng_best_side,
                "side_mismatch": best_alpha["side"] != eng_best_side,
                "hold_p90_over_alphaH": hold_p90 / alpha_h if alpha_h > 0 else None,
                "trail_dominates_short_hold": trail_share > 0.40,
                "alpha_edge_nonpos_oos": best_alpha["net_base_mean"] <= 0 if "net_base_mean" in best_alpha else False,
            })

    qa_df = pl.DataFrame(qa_rows) if qa_rows else pl.DataFrame()
    qa_df.write_parquet(str(OUT_QA), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v2.0.1",
        "alignment_report": qa_rows,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 19] OUT: {OUT_QA} ({qa_df.height} rows)")
    if qa_rows:
        for q in qa_rows:
            mismatch = "MISMATCH" if q["side_mismatch"] else "OK"
            print(f"  {q['symbol']}: alpha={q['alpha_best_side_oos']} engine={q['engine_best_side_oos']} [{mismatch}]")

print(">>> Celda 19 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 20: Run Summary + Manifest Final
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_20 = r'''# ======================================================================================
# Celda 20 v2.0.1 — Run Summary + Manifest Final
# Verifica todos los artifacts, calcula resumen ejecutivo, cierra manifest.
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 20 v2.0.1 :: Run Summary + Manifest Final")

if "RUN" not in globals():
    raise RuntimeError("[Celda 20] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]
RUN_ID = RUN["RUN_ID"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Verify all artifacts exist
missing = []
existing = []
for key, path in ARTIFACTS.items():
    if Path(path).exists():
        existing.append(key)
    else:
        missing.append(key)

print(f"[Celda 20] Artifacts: {len(existing)} exist, {len(missing)} missing")
if missing:
    print(f"[Celda 20] MISSING: {missing}")

# Summary stats
summary = {"run_id": RUN_ID, "completion_utc": _now_utc_iso()}

sel_path = ARTIFACTS.get("selection", RUN_DIR / "selection_v2.parquet")
if Path(sel_path).exists():
    sel = pl.read_parquet(sel_path)
    if sel.height > 0 and "decision" in sel.columns:
        summary["symbols_go"] = sel.filter(pl.col("decision") == "GO").height
        summary["symbols_total"] = sel.height

eng_snap_path = ARTIFACTS.get("engine_report_snapshot", RUN_DIR / "engine_report_snapshot_v2.json")
if Path(eng_snap_path).exists():
    eng_snap = json.loads(Path(eng_snap_path).read_text(encoding="utf-8"))
    kpis = eng_snap.get("kpis", {})
    summary["best_sharpe"] = kpis.get("sharpe_like")
    summary["worst_mdd"] = kpis.get("mdd")
    summary["total_return"] = kpis.get("total_return")

summary["artifacts_existing"] = len(existing)
summary["artifacts_missing"] = len(missing)
summary["artifacts_missing_keys"] = missing

# Update manifest
manifest_path = RUN_DIR / "run_manifest_v2.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
else:
    manifest = {}

manifest["completion_utc"] = summary["completion_utc"]
manifest["summary"] = summary
manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

# Latest
latest_path = RUN_DIR.parent / "run_manifest_v2_latest.json"
latest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*60}")
print(f"  RUN SUMMARY — TREND v2")
print(f"{'='*60}")
for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print(f"{'='*60}")
print(f"[Celda 20] Manifest updated: {manifest_path}")
print(">>> Celda 20 v2.0.1 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN: Build the notebook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEW_CELLS = [CELL_06, CELL_07, CELL_08, CELL_09, CELL_10, CELL_11, CELL_12,
             CELL_13, CELL_14, CELL_15, CELL_16, CELL_17, CELL_18, CELL_19, CELL_20]

# Additional artifacts to add to Cell 00
EXTRA_ARTIFACTS = {
    "signals_all":          "signals_all_v2.parquet",
    "signals_snapshot":     "signals_snapshot_v2.json",
    "qa_timing":            "qa_timing_v2.parquet",
    "tuning_results":       "tuning_results_v2.parquet",
    "tuning_best_params":   "tuning_best_params_v2.parquet",
    "tuning_snapshot":      "tuning_snapshot_v2.json",
    "alpha_design":         "alpha_design_v2.parquet",
    "alpha_design_snapshot": "alpha_design_snapshot_v2.json",
    "selection":            "selection_v2.parquet",
    "selection_snapshot":   "selection_snapshot_v2.json",
    "overlay_trades":       "overlay_trades_v2.parquet",
    "overlay_summary":      "overlay_summary_v2.parquet",
    "overlay_snapshot":     "overlay_snapshot_v2.json",
    "deploy_pack":          "deploy_pack_v2.parquet",
    "deploy_pack_json":     "deploy_pack_v2.json",
    "qa_alignment":         "qa_alignment_v2.parquet",
    "qa_alignment_snapshot": "qa_alignment_snapshot_v2.json",
    "diagnostics":          "diagnostics_v2.parquet",
    "diagnostics_snapshot": "diagnostics_snapshot_v2.json",
}

def main():
    print(f"Reading {NB_PATH} ...")
    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    # --- Update Cell 00: add missing artifacts to _build_artifacts ---
    cell0_src = "".join(nb["cells"][0]["source"])

    # Find the closing of _build_artifacts return dict
    # Insert new artifacts before the last "}" of the return dict
    marker = '"engine_report_snapshot": str(run_dir / "engine_report_snapshot_v2.json"),'
    if marker in cell0_src:
        extra_lines = ""
        for key, fname in EXTRA_ARTIFACTS.items():
            extra_lines += f'\n        "{key}": str(run_dir / "{fname}"),'
        cell0_src = cell0_src.replace(
            marker,
            marker + extra_lines
        )
        nb["cells"][0]["source"] = _fix_source_lines(cell0_src)
        print(f"  Cell 00: added {len(EXTRA_ARTIFACTS)} artifacts to _build_artifacts()")
    else:
        print("  WARNING: marker not found in Cell 00, artifacts NOT updated")

    # --- Remove any existing cells beyond index 5 (in case of re-runs) ---
    if len(nb["cells"]) > 6:
        print(f"  Removing {len(nb['cells']) - 6} existing cells beyond index 5")
        nb["cells"] = nb["cells"][:6]

    # --- Add new cells ---
    for i, src in enumerate(NEW_CELLS):
        nb["cells"].append(make_cell(src))
        print(f"  Added Cell {i + 6:02d}")

    # --- Write ---
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\nDone! {NB_PATH} now has {len(nb['cells'])} cells.")

if __name__ == "__main__":
    main()
