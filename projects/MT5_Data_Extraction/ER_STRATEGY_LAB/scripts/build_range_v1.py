#!/usr/bin/env python
"""
build_range_v1.py
-----------------
Creates 04_RANGE_M5_Strategy_v1.ipynb from scratch with 21 cells.
Mean-reversion strategy on ranging symbols using Bollinger %B and distance-to-mean.

Run from repo root:
    python projects/MT5_Data_Extraction/ER_STRATEGY_LAB/scripts/build_range_v1.py
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "04_RANGE_M5_Strategy_v1.ipynb"

def _fix_source_lines(lines_str: str) -> list[str]:
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
# CELL 00: Run Manifest + Paths
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_00 = r'''# ======================================================================================
# Celda 00 v1.0.0 — RANGE M5 Strategy: Run Manifest + Paths + Canonical Schema
# Politica: siempre crea run nuevo por defecto.
# Env vars: RANGE_M5_ROOT, RANGE_M5_OUTPUTS_ROOT, RANGE_M5_RUN_ID, RANGE_M5_RESUME_LATEST
# ======================================================================================

from __future__ import annotations

import os
import json
import sys
import platform
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# --- Helpers ---
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def _write_text(path: Path, text: str) -> None:
    _safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y")

# --- Detect PROJECT_ROOT ---
def _detect_project_root() -> Path:
    forced = _env("RANGE_M5_ROOT") or _env("MT5_PROJECT_ROOT") or _env("MT5_DE_PROJECT_ROOT")
    if forced:
        return Path(forced).resolve()
    start = Path.cwd().resolve()
    target = "mt5_data_extraction"
    for p in [start] + list(start.parents):
        if p.name.lower() == target:
            return p
    return start

PROJECT_ROOT = _detect_project_root()

# --- Paths ---
WORKDIR = Path.cwd().resolve()
OUTPUTS_ROOT = Path(_env("RANGE_M5_OUTPUTS_ROOT", str(WORKDIR / "outputs" / "range_m5_strategy" / "v1"))).resolve()
LATEST_RUN_MARKER = OUTPUTS_ROOT / "_latest_run.txt"

FORCED_RUN_ID = (_env("RANGE_M5_RUN_ID") or "").strip() or None
RESUME_LATEST = _env_bool("RANGE_M5_RESUME_LATEST", default=False)

def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    salt = _sha1(f"{ts}|{platform.node()}|{os.getpid()}")[:8]
    return f"{ts}_{salt}"

if FORCED_RUN_ID:
    RUN_MODE = "FORCED_RUN_ID"
    RUN_ID = FORCED_RUN_ID
elif RESUME_LATEST and LATEST_RUN_MARKER.exists():
    RUN_MODE = "RESUME_LATEST"
    RUN_ID = _read_text(LATEST_RUN_MARKER) or _new_run_id()
else:
    RUN_MODE = "NEW_RUN_DEFAULT"
    RUN_ID = _new_run_id()

RUN_DIR = OUTPUTS_ROOT / f"run_{RUN_ID}"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest_range_v1.json"
RUN_MANIFEST_LATEST_PATH = OUTPUTS_ROOT / "run_manifest_range_v1_latest.json"

SCHEMA_VERSION = "v1.0.0"
ENGINE_VERSION = "v1.0.0"

CANONICAL_SCHEMA = {
    "ohlcv_m5": {
        "required_columns": ["time_utc", "open", "high", "low", "close", "volume", "spread", "symbol"],
    },
    "engine_trades": {
        "required_columns": [
            "symbol", "fold_id", "segment", "side",
            "signal_time_utc", "entry_time_utc", "exit_time_utc",
            "entry_price", "exit_price",
            "gross_pnl", "net_pnl_base", "net_pnl_stress",
            "hold_bars", "exit_reason"
        ],
    }
}

def _build_artifacts(run_dir: Path) -> Dict[str, str]:
    return {
        "instrument_specs":        str(run_dir / "instrument_specs_range_v1.parquet"),
        "instrument_specs_snapshot": str(run_dir / "instrument_specs_snapshot_range_v1.json"),
        "ohlcv_clean":             str(run_dir / "ohlcv_clean_m5.parquet"),
        "data_qa_report":          str(run_dir / "data_qa_report_range_v1.json"),
        "cost_model_snapshot":     str(run_dir / "cost_model_snapshot_range_v1.json"),
        "wfo_folds":               str(run_dir / "wfo_folds_range_v1.parquet"),
        "wfo_folds_snapshot":      str(run_dir / "wfo_folds_snapshot_range_v1.json"),
        "features_m5":             str(run_dir / "features_m5_range_v1.parquet"),
        "features_snapshot":       str(run_dir / "features_snapshot_range_v1.json"),
        "regime_params_by_fold":   str(run_dir / "regime_params_by_fold_range_v1.parquet"),
        "regime_params_snapshot":  str(run_dir / "regime_params_snapshot_range_v1.json"),
        "signals_all":             str(run_dir / "signals_all_range_v1.parquet"),
        "signals_snapshot":        str(run_dir / "signals_snapshot_range_v1.json"),
        "qa_timing":               str(run_dir / "qa_timing_range_v1.parquet"),
        "alpha_multi_horizon_report": str(run_dir / "alpha_multi_horizon_report_range_v1.parquet"),
        "alpha_multi_horizon_snapshot": str(run_dir / "alpha_multi_horizon_snapshot_range_v1.json"),
        "trades_engine":           str(run_dir / "trades_engine_range_v1.parquet"),
        "summary_engine":          str(run_dir / "summary_engine_range_v1.parquet"),
        "engine_qa_report":        str(run_dir / "engine_qa_report_range_v1.json"),
        "equity_engine":           str(run_dir / "equity_curve_engine_range_v1.parquet"),
        "engine_report_snapshot":  str(run_dir / "engine_report_snapshot_range_v1.json"),
        "diagnostics":             str(run_dir / "diagnostics_range_v1.parquet"),
        "diagnostics_snapshot":    str(run_dir / "diagnostics_snapshot_range_v1.json"),
        "tuning_results":          str(run_dir / "tuning_results_range_v1.parquet"),
        "tuning_best_params":      str(run_dir / "tuning_best_params_range_v1.parquet"),
        "tuning_snapshot":         str(run_dir / "tuning_snapshot_range_v1.json"),
        "alpha_design":            str(run_dir / "alpha_design_range_v1.parquet"),
        "alpha_design_snapshot":   str(run_dir / "alpha_design_snapshot_range_v1.json"),
        "overlay_trades":          str(run_dir / "overlay_trades_range_v1.parquet"),
        "overlay_summary":         str(run_dir / "overlay_summary_range_v1.parquet"),
        "overlay_snapshot":        str(run_dir / "overlay_snapshot_range_v1.json"),
        "selection":               str(run_dir / "selection_range_v1.parquet"),
        "selection_snapshot":      str(run_dir / "selection_snapshot_range_v1.json"),
        "deploy_pack":             str(run_dir / "deploy_pack_range_v1.parquet"),
        "deploy_pack_json":        str(run_dir / "deploy_pack_range_v1.json"),
        "qa_alignment":            str(run_dir / "qa_alignment_range_v1.parquet"),
        "qa_alignment_snapshot":   str(run_dir / "qa_alignment_snapshot_range_v1.json"),
    }

def _build_manifest() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "engine_version": ENGINE_VERSION,
        "strategy": "RANGE_MEAN_REVERSION",
        "run_mode": RUN_MODE, "run_id": RUN_ID,
        "created_utc": _now_utc_iso(),
        "project_root": str(PROJECT_ROOT), "workdir": str(WORKDIR),
        "outputs_root": str(OUTPUTS_ROOT), "run_dir": str(RUN_DIR),
        "artifacts": _build_artifacts(RUN_DIR),
        "canonical_schema": CANONICAL_SCHEMA,
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "node": platform.node(), "pid": os.getpid(),
        },
    }

_safe_mkdir(RUN_DIR)
_safe_mkdir(OUTPUTS_ROOT)

manifest: Dict[str, Any]
if RUN_MANIFEST_PATH.exists() and RUN_MODE in ("RESUME_LATEST", "FORCED_RUN_ID"):
    manifest = _read_json(RUN_MANIFEST_PATH)
    manifest["artifacts"] = _build_artifacts(Path(manifest.get("run_dir", str(RUN_DIR))))
    _write_json(RUN_MANIFEST_PATH, manifest)
    print(f"[Celda 00 RANGE v1.0.0] Manifest CARGADO (resume): {RUN_MANIFEST_PATH}")
else:
    manifest = _build_manifest()
    _write_json(RUN_MANIFEST_PATH, manifest)
    print(f"[Celda 00 RANGE v1.0.0] Manifest CREADO (nuevo): {RUN_MANIFEST_PATH}")

_write_text(LATEST_RUN_MARKER, RUN_ID)
_write_json(RUN_MANIFEST_LATEST_PATH, manifest)

RUN: Dict[str, Any] = {
    "RUN_ID": manifest["run_id"],
    "RUN_MODE": manifest["run_mode"],
    "RUN_DIR": Path(manifest["run_dir"]),
    "PROJECT_ROOT": Path(manifest["project_root"]),
    "WORKDIR": Path(manifest["workdir"]),
    "OUTPUTS_ROOT": Path(manifest["outputs_root"]),
    "ARTIFACTS": {k: Path(v) for k, v in manifest["artifacts"].items()},
    "SCHEMA_VERSION": manifest["schema_version"],
    "ENGINE_VERSION": manifest["engine_version"],
    "CANONICAL_SCHEMA": manifest["canonical_schema"],
}

print(f"\n--- Celda 00 RANGE v1.0.0 | Estado final ---")
print(f"RUN_MODE   : {RUN['RUN_MODE']}")
print(f"RUN_ID     : {RUN['RUN_ID']}")
print(f"RUN_DIR    : {RUN['RUN_DIR']}")
print(f"N_ARTIFACTS: {len(RUN['ARTIFACTS'])}")

import polars as pl
print(f"polars: {pl.__version__}")
print(f"\n[Celda 00 RANGE v1.0.0] OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 01: Universe (dinamico, basket_range_core)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_01 = r'''# ======================================================================================
# Celda 01 v1.0.0 — Universe (RANGE) + Instrument Specs
# Lee basket_range_core.parquet de NB2 si existe, fallback a hardcoded.
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 01 RANGE v1.0.0 :: Universe + Instrument Specs")

if "RUN" not in globals():
    raise RuntimeError("[Celda 01] ERROR: RUN no existe. Ejecuta Celda 00.")

RUN_DIR: Path = RUN["RUN_DIR"]
PROJECT_ROOT: Path = RUN["PROJECT_ROOT"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

OUT_SPECS = ARTIFACTS["instrument_specs"]
OUT_SNAP = ARTIFACTS["instrument_specs_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# --- Search for basket_range_core from NB2 outputs ---
BASKET_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "er_filter_5m" / "basket_range_core.parquet",
    PROJECT_ROOT / "processed_data" / "basket_range_core.parquet",
    PROJECT_ROOT / "outputs" / "basket_range_core.parquet",
]

# Also scan the latest run of er_filter_5m
er_filter_dir = PROJECT_ROOT / "outputs" / "er_filter_5m"
if er_filter_dir.exists():
    for run_dir_candidate in sorted(er_filter_dir.iterdir(), reverse=True):
        if run_dir_candidate.is_dir():
            cand = run_dir_candidate / "basket_range_core.parquet"
            if cand.exists():
                BASKET_CANDIDATES.insert(0, cand)
            break

basket_path = None
for p in BASKET_CANDIDATES:
    if p.exists():
        basket_path = p
        break

FALLBACK_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDCAD", "EURGBP"]

if basket_path is not None:
    print(f"[Celda 01] basket_range_core encontrado: {basket_path}")
    basket_df = pl.read_parquet(basket_path)
    if "symbol" in basket_df.columns:
        symbols = basket_df.get_column("symbol").unique().sort().to_list()
    else:
        symbols = FALLBACK_SYMBOLS
        print("[Celda 01] WARNING: basket sin columna 'symbol', usando fallback.")
else:
    symbols = FALLBACK_SYMBOLS
    print(f"[Celda 01] WARNING: basket_range_core no encontrado, usando fallback: {symbols}")

symbols = [s.upper().strip() for s in symbols]
print(f"[Celda 01] symbols ({len(symbols)}): {symbols}")

# --- Instrument specs ---
# Default specs for common pairs / instruments
DEFAULT_SPEC = {
    "asset_class": "forex", "point_value": 1.0, "tick_size": 0.00001,
    "cost_base_bps": 3.0, "cost_stress_bps": 6.0,
    "session_start_utc": "00:00", "session_end_utc": "23:59",
}

OVERRIDES = {
    "XAUUSD": {"asset_class": "commodity", "point_value": 100.0, "tick_size": 0.01, "cost_base_bps": 5.0, "cost_stress_bps": 10.0},
    "XAUAUD": {"asset_class": "commodity", "point_value": 100.0, "tick_size": 0.01, "cost_base_bps": 5.0, "cost_stress_bps": 10.0},
    "USDJPY": {"tick_size": 0.001, "cost_base_bps": 2.0, "cost_stress_bps": 4.0},
    "EURJPY": {"tick_size": 0.001, "cost_base_bps": 3.0, "cost_stress_bps": 6.0},
}

specs_rows = []
for sym in symbols:
    spec = dict(DEFAULT_SPEC)
    spec.update(OVERRIDES.get(sym, {}))
    spec["symbol"] = sym
    specs_rows.append(spec)

specs_df = pl.DataFrame(specs_rows)
specs_df.write_parquet(str(OUT_SPECS), compression="zstd")

snap = {
    "created_utc": _now_utc_iso(), "version": "v1.0.0",
    "symbols": symbols, "basket_source": str(basket_path) if basket_path else "FALLBACK",
    "n_symbols": len(symbols),
}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

RUN["symbols"] = symbols
RUN["instrument_specs"] = {r["symbol"]: r for r in specs_rows}

print(f"[Celda 01] OUT: {OUT_SPECS} ({specs_df.height} rows)")
print(">>> Celda 01 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 02: Load M5 + QA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_02 = r'''# ======================================================================================
# Celda 02 v1.0.0 — Load M5 (m5_clean) + Canonicalize + QA (RANGE)
# Identico patron a TREND v2 Cell 02: busca ohlcv_clean_m5.parquet de NB2 o construye.
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 02 RANGE v1.0.0 :: Load M5 + QA")

if "RUN" not in globals():
    raise RuntimeError("[Celda 02] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
PROJECT_ROOT: Path = RUN["PROJECT_ROOT"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]
symbols = RUN.get("symbols", [])

OUT_OHLCV = ARTIFACTS["ohlcv_clean"]
OUT_QA = ARTIFACTS["data_qa_report"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# --- Search for M5 data ---
M5_CANDIDATES = [
    PROJECT_ROOT / "processed_data" / "m5_clean",
    PROJECT_ROOT / "data" / "m5_clean",
    PROJECT_ROOT / "bulk_data" / "m5_clean",
]

# Also check for a single consolidated parquet from previous runs
CONSOLIDATED_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "er_filter_5m",
]

m5_dir = None
for d in M5_CANDIDATES:
    if d.exists() and any(d.glob("*.parquet")):
        m5_dir = d
        break

if m5_dir is None:
    # Try finding individual symbol parquets anywhere under processed_data
    pdata = PROJECT_ROOT / "processed_data"
    if pdata.exists():
        for sub in pdata.iterdir():
            if sub.is_dir() and any(sub.glob("*m5*.parquet")):
                m5_dir = sub
                break

qa_report = {"created_utc": _now_utc_iso(), "symbols_requested": symbols}

if OUT_OHLCV.exists():
    print(f"[Celda 02] Cache: usando {OUT_OHLCV}")
    df = pl.read_parquet(OUT_OHLCV)
    qa_report["status"] = "CACHED"
    qa_report["n_rows"] = df.height
elif m5_dir is not None:
    print(f"[Celda 02] M5 dir: {m5_dir}")
    dfs = []
    for sym in symbols:
        candidates = list(m5_dir.glob(f"*{sym}*m5*.parquet")) + list(m5_dir.glob(f"*{sym.lower()}*m5*.parquet"))
        if not candidates:
            candidates = list(m5_dir.glob(f"*{sym}*.parquet"))
        if candidates:
            df_sym = pl.read_parquet(candidates[0])
            if "symbol" not in df_sym.columns:
                df_sym = df_sym.with_columns(pl.lit(sym).alias("symbol"))
            dfs.append(df_sym)
            print(f"  {sym}: {candidates[0].name} ({df_sym.height} rows)")
        else:
            print(f"  {sym}: NOT FOUND in {m5_dir}")

    if dfs:
        df = pl.concat(dfs, how="vertical_relaxed")
        # Canonicalize
        required = ["symbol", "time_utc", "open", "high", "low", "close"]
        for col in required:
            if col not in df.columns:
                raise RuntimeError(f"[Celda 02] ERROR: columna {col} faltante. Columnas: {df.columns}")
        df = df.with_columns(pl.col("time_utc").cast(pl.Datetime("us", "UTC"), strict=False))
        df = df.unique(subset=["symbol", "time_utc"], keep="last").sort(["symbol", "time_utc"])
        df.write_parquet(str(OUT_OHLCV), compression="zstd")
        qa_report["status"] = "BUILT"
        qa_report["n_rows"] = df.height
        qa_report["n_symbols"] = df.get_column("symbol").n_unique()
    else:
        raise RuntimeError(f"[Celda 02] ERROR: no M5 data found for any symbol in {m5_dir}")
else:
    raise RuntimeError("[Celda 02] ERROR: no M5 data directory found. Run NB1+NB2 first.")

Path(OUT_QA).write_text(json.dumps(qa_report, indent=2, default=str), encoding="utf-8")
print(f"[Celda 02] OUT: {OUT_OHLCV} ({qa_report.get('n_rows', 0)} rows)")
print(">>> Celda 02 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 03: Cost Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_03 = r'''# ======================================================================================
# Celda 03 v1.0.0 — Cost Model (base/stress + slippage proxy) [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 03 RANGE v1.0.0 :: Cost Model")

if "RUN" not in globals():
    raise RuntimeError("[Celda 03] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

OUT_SNAP = ARTIFACTS["cost_model_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

specs = RUN.get("instrument_specs", {})
symbols = RUN.get("symbols", [])

costs_by_symbol = {}
for sym in symbols:
    s = specs.get(sym, {})
    costs_by_symbol[sym] = {
        "cost_base_bps": float(s.get("cost_base_bps", 3.0)),
        "cost_stress_bps": float(s.get("cost_stress_bps", 6.0)),
    }

snap = {
    "created_utc": _now_utc_iso(), "version": "v1.0.0",
    "strategy": "RANGE_MEAN_REVERSION",
    "costs_by_symbol": costs_by_symbol,
    "cost_reported_is_roundtrip": True,
}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(f"[Celda 03] costs for {len(costs_by_symbol)} symbols")
for sym, c in costs_by_symbol.items():
    print(f"  {sym}: base={c['cost_base_bps']}bps stress={c['cost_stress_bps']}bps")
print(">>> Celda 03 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 04: WFO Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_04 = r'''# ======================================================================================
# Celda 04 v1.0.0 — WFO Builder (IS=18m, OOS=3m, >=6 folds) [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict
import polars as pl

print(">>> Celda 04 RANGE v1.0.0 :: WFO Builder")

if "RUN" not in globals():
    raise RuntimeError("[Celda 04] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

OHLCV_PATH = ARTIFACTS["ohlcv_clean"]
OUT_FOLDS = ARTIFACTS["wfo_folds"]
OUT_SNAP = ARTIFACTS["wfo_folds_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

IS_MONTHS = 18
OOS_MONTHS = 3
EMBARGO_DAYS = 5
MIN_FOLDS = 6

df = pl.read_parquet(OHLCV_PATH)
t_min = df.get_column("time_utc").min()
t_max = df.get_column("time_utc").max()

print(f"[Celda 04] data range: {t_min} to {t_max}")

# Build folds
folds = []
fold_id = 1
cursor = t_min

while True:
    is_start = cursor
    is_end = is_start + timedelta(days=IS_MONTHS * 30)
    embargo_end = is_end + timedelta(days=EMBARGO_DAYS)
    oos_start = embargo_end
    oos_end = oos_start + timedelta(days=OOS_MONTHS * 30)

    if oos_end > t_max:
        break

    folds.append({
        "fold_id": fold_id,
        "IS_start": is_start,
        "IS_end": is_end,
        "embargo_start": is_end,
        "embargo_end": embargo_end,
        "OOS_start": oos_start,
        "OOS_end": oos_end,
        "embargo_days": EMBARGO_DAYS,
    })
    fold_id += 1
    cursor = cursor + timedelta(days=OOS_MONTHS * 30)

print(f"[Celda 04] {len(folds)} folds generated (min={MIN_FOLDS})")

if len(folds) < MIN_FOLDS:
    print(f"[Celda 04] WARNING: {len(folds)} < {MIN_FOLDS} folds. Datos insuficientes.")

folds_df = pl.DataFrame(folds)
folds_df.write_parquet(str(OUT_FOLDS), compression="zstd")

snap = {
    "created_utc": _now_utc_iso(), "version": "v1.0.0",
    "IS_months": IS_MONTHS, "OOS_months": OOS_MONTHS, "embargo_days": EMBARGO_DAYS,
    "n_folds": len(folds),
    "folds_summary": [{"fold_id": f["fold_id"], "IS_start": str(f["IS_start"]), "OOS_end": str(f["OOS_end"])} for f in folds],
}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(f"[Celda 04] OUT: {OUT_FOLDS} ({folds_df.height} rows)")
print(">>> Celda 04 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 05: Features (Range-specific)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_05 = r'''# ======================================================================================
# Celda 05 v1.0.0 — Feature Set (RANGE): Base + Bollinger %B + Distance-to-Mean
# Extra features vs TREND: pct_b, dist_mean_atr, range_width_atr
# ======================================================================================

from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 05 RANGE v1.0.0 :: Feature Set (Range)")

if "RUN" not in globals():
    raise RuntimeError("[Celda 05] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

OHLCV_PATH = ARTIFACTS["ohlcv_clean"]
OUT_FEATURES = ARTIFACTS["features_m5"]
OUT_SNAP = ARTIFACTS["features_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Params
ER_WIN = int(os.getenv("RANGE_M5_ER_WIN", "288"))
VOL_WIN = int(os.getenv("RANGE_M5_VOL_WIN", "288"))
MOM_WIN = int(os.getenv("RANGE_M5_MOM_WIN", "288"))
ATR_WIN = int(os.getenv("RANGE_M5_ATR_WIN", "96"))
BB_WIN = int(os.getenv("RANGE_M5_BB_WIN", "96"))
BB_STD = float(os.getenv("RANGE_M5_BB_STD", "2.0"))
MEAN_WIN = int(os.getenv("RANGE_M5_MEAN_WIN", "96"))
RANGE_WIN = int(os.getenv("RANGE_M5_RANGE_WIN", "96"))
EPS = 1e-12

FORCE_REBUILD = os.getenv("RANGE_M5_FORCE_REBUILD_FEATURES", "").strip().lower() in ("1", "true")

if OUT_FEATURES.exists() and OUT_SNAP.exists() and not FORCE_REBUILD:
    print(f"[Celda 05] Cache: {OUT_FEATURES}")
else:
    lf = (
        pl.scan_parquet(OHLCV_PATH)
        .select(["symbol", "time_utc", "open", "high", "low", "close", "volume", "spread"])
        .sort(["symbol", "time_utc"])
    )

    close_prev = pl.col("close").shift(1).over("symbol")
    abs_diff_expr = (pl.col("close") - close_prev).abs()
    ret_expr = (
        pl.when(close_prev.is_not_null() & (close_prev > 0))
        .then(pl.col("close") / close_prev - 1.0)
        .otherwise(None)
    )
    tr_expr = pl.max_horizontal([
        (pl.col("high") - pl.col("low")),
        (pl.col("high") - close_prev).abs(),
        (pl.col("low") - close_prev).abs(),
    ])

    lf1 = lf.with_columns([
        ret_expr.alias("ret"),
        abs_diff_expr.alias("abs_diff"),
        tr_expr.alias("true_range"),
    ])

    # Base features (same as TREND)
    lf2 = lf1.with_columns([
        (pl.col("ret").rolling_std(window_size=VOL_WIN, min_samples=VOL_WIN).over("symbol") * 10_000)
            .alias(f"vol_bps_{VOL_WIN}"),
        (pl.col("true_range").rolling_mean(window_size=ATR_WIN, min_samples=ATR_WIN).over("symbol") / pl.col("close") * 10_000)
            .alias(f"atr_bps_{ATR_WIN}"),
        ((pl.col("close") / pl.col("close").shift(MOM_WIN).over("symbol") - 1.0) * 10_000)
            .alias(f"mom_bps_{MOM_WIN}"),
        ((pl.col("close") - pl.col("close").shift(ER_WIN).over("symbol")).abs() /
         (pl.col("abs_diff").rolling_sum(window_size=ER_WIN, min_samples=ER_WIN).over("symbol") + EPS))
            .alias(f"er_{ER_WIN}"),
        # ATR in price units (for distance calculations)
        pl.col("true_range").rolling_mean(window_size=ATR_WIN, min_samples=ATR_WIN).over("symbol")
            .alias("atr_price"),
    ])

    # Range-specific features
    lf3 = lf2.with_columns([
        # Bollinger Bands
        pl.col("close").rolling_mean(window_size=BB_WIN, min_samples=BB_WIN).over("symbol").alias("bb_mid"),
        pl.col("close").rolling_std(window_size=BB_WIN, min_samples=BB_WIN).over("symbol").alias("bb_std"),
    ])

    lf4 = lf3.with_columns([
        # Bollinger %B: (close - lower) / (upper - lower)
        ((pl.col("close") - (pl.col("bb_mid") - BB_STD * pl.col("bb_std"))) /
         (2.0 * BB_STD * pl.col("bb_std") + EPS)).alias("pct_b"),
        # Distance to mean in ATR units
        ((pl.col("close") - pl.col("bb_mid")) / (pl.col("atr_price") + EPS)).alias("dist_mean_atr"),
        # Range width in ATR units (rolling high-low / ATR)
        ((pl.col("high").rolling_max(window_size=RANGE_WIN, min_samples=RANGE_WIN).over("symbol") -
          pl.col("low").rolling_min(window_size=RANGE_WIN, min_samples=RANGE_WIN).over("symbol")) /
         (pl.col("atr_price") + EPS)).alias("range_width_atr"),
    ])

    lf_feat = lf4.select([
        "symbol", "time_utc", "open", "high", "low", "close", "volume", "spread",
        "ret",
        f"vol_bps_{VOL_WIN}", f"atr_bps_{ATR_WIN}", f"mom_bps_{MOM_WIN}", f"er_{ER_WIN}",
        "atr_price",
        "pct_b", "dist_mean_atr", "range_width_atr",
    ]).sort(["symbol", "time_utc"])

    df_feat = lf_feat.collect()

    df_feat.write_parquet(str(OUT_FEATURES), compression="zstd")

    snap = {
        "created_utc": _now_utc_iso(), "version": "v1.0.0",
        "params": {"ER_WIN": ER_WIN, "VOL_WIN": VOL_WIN, "MOM_WIN": MOM_WIN, "ATR_WIN": ATR_WIN,
                   "BB_WIN": BB_WIN, "BB_STD": BB_STD, "MEAN_WIN": MEAN_WIN, "RANGE_WIN": RANGE_WIN},
        "n_rows": df_feat.height,
        "symbols": df_feat.get_column("symbol").unique().sort().to_list(),
        "schema_cols": df_feat.columns,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    print(f"[Celda 05] Features: {df_feat.height} rows, {len(df_feat.columns)} cols")
    print(f"[Celda 05] Cols: {df_feat.columns}")

print(f"[Celda 05] OUT: {OUT_FEATURES}")
print(">>> Celda 05 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 06: Regime Gate (Range — low ER)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_06 = r'''# ======================================================================================
# Celda 06 v1.0.0 — Regime Gate (RANGE): Low ER + Low Vol = ranging market
# Gate: ER <= thr_er_high AND vol <= thr_vol (sin threshold de momentum)
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 06 RANGE v1.0.0 :: Regime Gate (ranging markets)")

if "RUN" not in globals():
    raise RuntimeError("[Celda 06] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]

OUT_REGIME = ARTIFACTS["regime_params_by_fold"]
OUT_SNAP = ARTIFACTS["regime_params_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

ER_COL = "er_288"
VOL_COL = "vol_bps_288"

# RANGE gate: ER <= threshold (low efficiency = ranging)
Q_ER_HIGH = 0.40   # ER at or below 40th percentile = ranging
Q_VOL = 0.90       # vol below 90th percentile = not volatile

COV_IS_MIN = 0.10
COV_IS_MAX = 0.80
MIN_IS_ROWS = 5_000

def _q_safe(s, q):
    s2 = s.drop_nulls()
    if s2.len() == 0: return None
    v = s2.quantile(q, interpolation="nearest")
    if v is None: return None
    fv = float(v)
    return fv if math.isfinite(fv) else None

df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

rows = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        is_s, is_e = fold_row["IS_start"], fold_row["IS_end"]
        oos_s, oos_e = fold_row["OOS_start"], fold_row["OOS_end"]

        df_is = df_sym.filter(
            (pl.col("time_utc") >= is_s) & (pl.col("time_utc") <= is_e)
        ).drop_nulls([ER_COL, VOL_COL])

        df_oos = df_sym.filter(
            (pl.col("time_utc") >= oos_s) & (pl.col("time_utc") <= oos_e)
        ).drop_nulls([ER_COL, VOL_COL])

        for side in ("LONG", "SHORT"):
            if df_is.height < MIN_IS_ROWS:
                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "SKIP",
                            "thr_er_high": None, "thr_vol": None, "cov_is": 0.0, "cov_oos": 0.0,
                            "n_is": df_is.height, "n_oos": df_oos.height})
                continue

            thr_er = _q_safe(df_is.get_column(ER_COL), Q_ER_HIGH)
            thr_vol = _q_safe(df_is.get_column(VOL_COL), Q_VOL)

            if thr_er is None or thr_vol is None:
                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "FAIL",
                            "thr_er_high": None, "thr_vol": None, "cov_is": 0.0, "cov_oos": 0.0,
                            "n_is": df_is.height, "n_oos": df_oos.height})
                continue

            gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)
            cov_is = float(df_is.select(gate.mean()).item())
            cov_oos = float(df_oos.select(gate.mean()).item()) if df_oos.height > 0 else 0.0

            rows.append({
                "symbol": sym, "fold_id": fid, "side": side, "scheme": "BASE",
                "thr_er_high": float(thr_er), "thr_vol": float(thr_vol),
                "cov_is": cov_is, "cov_oos": cov_oos,
                "n_is": df_is.height, "n_oos": df_oos.height,
            })
            print(f"[Celda 06] {sym} fold={fid} {side}: cov_IS={cov_is:.3f} cov_OOS={cov_oos:.3f}")

gate_df = pl.DataFrame(rows).sort(["symbol", "fold_id", "side"])
gate_df.write_parquet(str(OUT_REGIME), compression="zstd")

snap = {"created_utc": _now_utc_iso(), "version": "v1.0.0",
        "gate_type": "RANGE (ER<=thr, vol<=thr)",
        "params": {"Q_ER_HIGH": Q_ER_HIGH, "Q_VOL": Q_VOL}}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(f"\n[Celda 06] OUT: {OUT_REGIME} ({gate_df.height} rows)")
print(">>> Celda 06 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 07: Senales (Mean-Reversion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_07 = r'''# ======================================================================================
# Celda 07 v1.0.0 — Senales RANGE (Mean-Reversion) + Ejecucion t+1 + Costos
# LONG: dist_mean_atr <= -BAND_K (precio debajo de mean)
# SHORT: dist_mean_atr >= +BAND_K
# BAND_K = 1.5
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 07 RANGE v1.0.0 :: Senales Mean-Reversion")

if "RUN" not in globals():
    raise RuntimeError("[Celda 07] ERROR: RUN no existe.")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS["cost_model_snapshot"]

OUT_SIGNALS = ARTIFACTS["signals_all"]
OUT_SNAP = ARTIFACTS["signals_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

ER_COL = "er_288"
VOL_COL = "vol_bps_288"
DIST_COL = "dist_mean_atr"
BAND_K = 1.5

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
    cost_base_rt = float(cinfo.get("cost_base_bps", 3.0)) / 10_000
    cost_stress_rt = float(cinfo.get("cost_stress_bps", 6.0)) / 10_000

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
            if rg_row.get("thr_er_high") is None:
                continue

            thr_er = float(rg_row["thr_er_high"])
            thr_vol = float(rg_row["thr_vol"])

            # Regime gate (ranging market)
            regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)

            # Mean-reversion signal
            if side == "LONG":
                signal_gate = regime_gate & (pl.col(DIST_COL) <= -BAND_K)
            else:
                signal_gate = regime_gate & (pl.col(DIST_COL) >= BAND_K)

            dfx = (
                df_sym
                .with_columns(signal_gate.alias("signal_gate"))
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
                    pl.lit(sym).alias("_sym"), pl.lit(fid).alias("_fid"), pl.lit(side).alias("_side"),
                    (sign * (pl.col("exit_price") / pl.col("entry_price") - 1.0)).alias("gross_ret"),
                ])
                .filter(pl.col("segment").is_not_null())
                .with_columns([
                    (pl.col("gross_ret") - cost_base_rt).alias("net_ret_base"),
                    (pl.col("gross_ret") - cost_stress_rt).alias("net_ret_stress"),
                ])
                .select([
                    pl.col("_sym").alias("symbol"), pl.col("_fid").alias("fold_id"),
                    "segment", pl.col("_side").alias("side"),
                    pl.col("time_utc").alias("signal_time"),
                    "entry_time", "exit_time", "entry_price", "exit_price",
                    "gross_ret", "net_ret_base", "net_ret_stress",
                    ER_COL, VOL_COL, DIST_COL,
                ])
            )
            if dfx.height > 0:
                all_trades.append(dfx)
                print(f"[Celda 07] {sym} fold={fid} {side}: {dfx.height} signals")

if not all_trades:
    raise RuntimeError("[Celda 07] GATE FAIL: 0 signals.")

signals_df = pl.concat(all_trades, how="vertical_relaxed").sort(["symbol", "fold_id", "signal_time"])
signals_df.write_parquet(str(OUT_SIGNALS), compression="zstd")

snap = {"created_utc": _now_utc_iso(), "version": "v1.0.0", "n_signals": signals_df.height,
        "strategy": "MEAN_REVERSION", "BAND_K": BAND_K}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(f"\n[Celda 07] OUT: {OUT_SIGNALS} ({signals_df.height} rows)")
print(">>> Celda 07 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 08: QA Timing (same pattern)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_08 = r'''# ======================================================================================
# Celda 08 v1.0.0 — QA Timing Trades [RANGE]
# ======================================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict
import polars as pl

print(">>> Celda 08 RANGE v1.0.0 :: QA Timing")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]
SIGNALS_PATH = ARTIFACTS["signals_all"]
OUT_QA = ARTIFACTS["qa_timing"]

df = pl.read_parquet(SIGNALS_PATH)
df = df.with_columns([
    ((pl.col("entry_time") - pl.col("signal_time")).dt.total_seconds()).alias("dt_entry_s"),
    ((pl.col("exit_time") - pl.col("entry_time")).dt.total_seconds()).alias("dt_hold_s"),
])

qa = (
    df.group_by(["symbol", "segment"])
    .agg([
        pl.len().alias("n"),
        pl.col("dt_entry_s").median().alias("dt_entry_med"),
        pl.col("dt_hold_s").median().alias("dt_hold_med"),
        pl.col("dt_hold_s").quantile(0.90, interpolation="nearest").alias("dt_hold_p90"),
        pl.col("dt_hold_s").max().alias("dt_hold_max"),
    ])
    .sort(["symbol", "segment"])
)
qa.write_parquet(str(OUT_QA), compression="zstd")
print(qa)
print(">>> Celda 08 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 09: Alpha Multi-Horizon (shorter horizons for range)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_09 = r'''# ======================================================================================
# Celda 09 v1.0.0 — Alpha Multi-Horizon Report [RANGE]
# Shorter horizons: [1, 3, 6, 12, 24, 48, 96]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 09 RANGE v1.0.0 :: Alpha Multi-Horizon")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS["cost_model_snapshot"]

OUT_ALPHA = ARTIFACTS["alpha_multi_horizon_report"]
OUT_SNAP = ARTIFACTS["alpha_multi_horizon_snapshot"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

HORIZONS = [1, 3, 6, 12, 24, 48, 96]
ER_COL = "er_288"
VOL_COL = "vol_bps_288"
DIST_COL = "dist_mean_atr"
BAND_K = 1.5

df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)
df_regime = pl.read_parquet(REGIME_PATH)
cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))
costs_by_sym = cost_snap.get("costs_by_symbol", {})

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

# Precompute fwd returns
for h in HORIZONS:
    df_feat = df_feat.with_columns(
        (pl.col("close").shift(-h).over("symbol") / pl.col("close") - 1.0).alias(f"fwd_{h}")
    )
df_feat = df_feat.with_columns(pl.col("time_utc").dt.weekday().alias("_dow")).filter(pl.col("_dow") <= 5)

rows = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym)
    cinfo = costs_by_sym.get(sym, {})
    cost_rt = float(cinfo.get("cost_base_bps", 3.0)) / 10_000
    cost_stress_rt = float(cinfo.get("cost_stress_bps", 6.0)) / 10_000

    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        is_s, is_e = fold_row["IS_start"], fold_row["IS_end"]
        oos_s, oos_e = fold_row["OOS_start"], fold_row["OOS_end"]

        for side in ("LONG", "SHORT"):
            rg = df_regime.filter((pl.col("symbol") == sym) & (pl.col("fold_id") == fid) & (pl.col("side") == side))
            if rg.is_empty() or rg.row(0, named=True).get("thr_er_high") is None:
                continue
            rg_row = rg.row(0, named=True)
            regime_gate = (pl.col(ER_COL) <= rg_row["thr_er_high"]) & (pl.col(VOL_COL) <= rg_row["thr_vol"])
            if side == "LONG":
                signal = regime_gate & (pl.col(DIST_COL) <= -BAND_K)
            else:
                signal = regime_gate & (pl.col(DIST_COL) >= BAND_K)

            for seg_name, seg_s, seg_e in [("IS", is_s, is_e), ("OOS", oos_s, oos_e)]:
                df_seg = df_sym.filter(
                    (pl.col("time_utc") >= seg_s) & (pl.col("time_utc") <= seg_e)
                ).filter(signal)
                if df_seg.height < 5:
                    continue
                for h in HORIZONS:
                    vals = df_seg.get_column(f"fwd_{h}").drop_nulls().to_list()
                    if len(vals) < 5:
                        continue
                    sign = 1.0 if side == "LONG" else -1.0
                    rets = [sign * r for r in vals]
                    n = len(rets)
                    mean_r = sum(rets) / n
                    std_r = (sum((r - mean_r)**2 for r in rets) / max(1, n - 1)) ** 0.5
                    rows.append({
                        "symbol": sym, "fold_id": fid, "side": side, "segment": seg_name,
                        "horizon_bars": h, "n_trades": n,
                        "gross_mean": mean_r, "net_base_mean": mean_r - cost_rt,
                        "net_stress_mean": mean_r - cost_stress_rt,
                        "sharpe_like": mean_r / std_r if std_r > 1e-12 else 0.0,
                        "win_rate": sum(1 for r in rets if r > 0) / n,
                    })

alpha_df = pl.DataFrame(rows).sort(["symbol", "fold_id", "side", "segment", "horizon_bars"])
alpha_df.write_parquet(str(OUT_ALPHA), compression="zstd")

snap = {"created_utc": _now_utc_iso(), "version": "v1.0.0", "horizons": HORIZONS, "n_rows": alpha_df.height}
Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(f"[Celda 09] OUT: {OUT_ALPHA} ({alpha_df.height} rows)")
print(">>> Celda 09 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10: Backtest Engine (Mean-Reversion — no trail)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CELL_10 = r'''# ======================================================================================
# Celda 10 v1.0.0 — Backtest Engine (RANGE Mean-Reversion)
# SL=1.5xATR, TP=2.0xATR, NO TRAIL, time_stop=144, confirm=6, cooldown=12, min_hold=3
# BUG FIX: dedup keep="last"
# ======================================================================================

from __future__ import annotations
import json, math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 10 RANGE v1.0.0 :: Backtest Engine (Mean-Reversion)")

RUN_DIR: Path = RUN["RUN_DIR"]
ARTIFACTS: Dict[str, Path] = RUN["ARTIFACTS"]

FEATURES_PATH = ARTIFACTS["features_m5"]
WFO_PATH = ARTIFACTS["wfo_folds"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS["cost_model_snapshot"]

OUT_TRADES = ARTIFACTS["trades_engine"]
OUT_SUMMARY = ARTIFACTS["summary_engine"]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# --- RANGE Engine Params ---
SL_ATR     = 1.5
TP_ATR     = 2.0
TRAIL_ATR  = None  # NO TRAIL for mean-reversion
TIME_STOP  = 144   # 12h (faster for range)
ENTRY_CONFIRM = 6
EXIT_GATE_OFF = 6
MIN_HOLD   = 3
COOLDOWN   = 12
MON_FRI    = True
BAND_K     = 1.5
RISK_PER_TRADE = 0.01
MIN_POS_SIZE = 0.25
MAX_POS_SIZE = 3.00

ER_COL = "er_288"
VOL_COL = "vol_bps_288"
DIST_COL = "dist_mean_atr"
ATR_COL = "atr_bps_96"

print(f"[Celda 10] SL={SL_ATR}xATR TP={TP_ATR}xATR TRAIL=None TIME_STOP={TIME_STOP}")

df_feat = pl.read_parquet(FEATURES_PATH)
df_folds = pl.read_parquet(WFO_PATH)
df_regime = pl.read_parquet(REGIME_PATH)
cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))
costs_by_sym = cost_snap.get("costs_by_symbol", {})

symbols = df_feat.get_column("symbol").unique().sort().to_list()
fold_ids = df_folds.get_column("fold_id").unique().sort().to_list()

def _is_finite(x):
    if x is None: return False
    try: return math.isfinite(float(x))
    except: return False

def _simulate_range(sym, df_j, fold_row, thr_er, thr_vol, cost_base_dec, cost_stress_dec):
    is_s = fold_row["IS_start"]; is_e = fold_row["IS_end"]
    oos_s = fold_row["OOS_start"]; oos_e = fold_row["OOS_end"]
    fid = fold_row["fold_id"]

    # BUG FIX: dedup keep="last"
    df_j = df_j.unique(subset=["time_utc"], keep="last").sort("time_utc")

    # Regime gate
    regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)
    # Signal gates
    long_signal = regime_gate & (pl.col(DIST_COL) <= -BAND_K)
    short_signal = regime_gate & (pl.col(DIST_COL) >= BAND_K)

    df_j = df_j.with_columns([
        long_signal.alias("_gL"), short_signal.alias("_gS"),
        regime_gate.alias("_regime"),
    ])

    df_j = df_j.with_columns(pl.col("time_utc").dt.weekday().alias("_dow"))
    df_j = df_j.with_columns((pl.col("_dow") >= 6).alias("_is_wk"))

    # Confirm
    df_j = df_j.with_columns([
        (pl.col("_gL").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))
            .fill_null(False).alias("_cfL"),
        (pl.col("_gS").cast(pl.Int8).rolling_sum(ENTRY_CONFIRM, min_samples=ENTRY_CONFIRM).eq(ENTRY_CONFIRM))
            .fill_null(False).alias("_cfS"),
    ])

    t = df_j.get_column("time_utc").to_list()
    o = df_j.get_column("open").to_list()
    h = df_j.get_column("high").to_list()
    lo = df_j.get_column("low").to_list()
    c = df_j.get_column("close").to_list()
    atr_l = df_j.get_column(ATR_COL).to_list() if ATR_COL in df_j.columns else [None]*df_j.height
    gL = df_j.get_column("_gL").to_list()
    gS = df_j.get_column("_gS").to_list()
    cfL = df_j.get_column("_cfL").to_list()
    cfS = df_j.get_column("_cfS").to_list()
    regime = df_j.get_column("_regime").to_list()
    wk = df_j.get_column("_is_wk").to_list()

    n = len(t); trades = []
    pos = 0; entry_idx = None; entry_price = None
    stop = None; tp_price = None; gate_off = 0; cd = 0

    def _seg(et):
        if is_s <= et <= is_e: return "IS"
        if oos_s <= et <= oos_e: return "OOS"
        return None

    for idx in range(n):
        if pos != 0 and entry_idx is not None:
            bars_held = idx - entry_idx
            gn = bool(regime[idx]) if regime[idx] is not None else False
            gate_off = 0 if gn else gate_off + 1

            hi_v = float(h[idx]) if _is_finite(h[idx]) else float(c[idx])
            lo_v = float(lo[idx]) if _is_finite(lo[idx]) else float(c[idx])

            exit_reason = None; exit_price = None

            if pos == 1:
                if stop is not None and lo_v <= stop:
                    exit_reason, exit_price = "SL", stop
                elif tp_price is not None and hi_v >= tp_price:
                    exit_reason, exit_price = "TP", tp_price
            else:
                if stop is not None and hi_v >= stop:
                    exit_reason, exit_price = "SL", stop
                elif tp_price is not None and lo_v <= tp_price:
                    exit_reason, exit_price = "TP", tp_price

            if exit_reason is None and bars_held >= TIME_STOP:
                exit_reason, exit_price = "TIME", float(c[idx])
            if exit_reason is None and bars_held >= MIN_HOLD and gate_off >= EXIT_GATE_OFF:
                exit_reason, exit_price = "REGIME_OFF", float(c[idx])
            if exit_reason is None and MON_FRI and bool(wk[idx]):
                exit_reason, exit_price = "WEEKEND", float(c[idx])

            if exit_reason is not None:
                sign = 1.0 if pos == 1 else -1.0
                gross = sign * (exit_price / entry_price - 1.0)
                seg = _seg(t[entry_idx])
                trades.append({
                    "symbol": sym, "fold_id": fid, "segment": seg,
                    "side": "LONG" if pos == 1 else "SHORT",
                    "signal_time_utc": t[entry_idx],
                    "entry_time_utc": t[min(entry_idx + 1, n - 1)],
                    "exit_time_utc": t[idx],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "gross_pnl": gross,
                    "net_pnl_base": gross - cost_base_dec,
                    "net_pnl_stress": gross - cost_stress_dec,
                    "hold_bars": bars_held, "exit_reason": exit_reason,
                })
                pos = 0; entry_idx = None; entry_price = None
                stop = None; tp_price = None; cd = COOLDOWN
                continue

        if cd > 0:
            cd -= 1; continue

        if pos == 0 and idx < n - 2:
            if MON_FRI and bool(wk[idx]): continue

            atr_val = float(atr_l[idx]) / 10_000 * float(c[idx]) if _is_finite(atr_l[idx]) else float(c[idx]) * 0.005
            if atr_val <= 0: continue

            if bool(cfL[idx]):
                entry_price = float(o[idx + 1]) if _is_finite(o[idx + 1]) else float(c[idx])
                stop = entry_price - SL_ATR * atr_val
                tp_price = entry_price + TP_ATR * atr_val
                pos = 1; entry_idx = idx; gate_off = 0
            elif bool(cfS[idx]):
                entry_price = float(o[idx + 1]) if _is_finite(o[idx + 1]) else float(c[idx])
                stop = entry_price + SL_ATR * atr_val
                tp_price = entry_price - TP_ATR * atr_val
                pos = -1; entry_idx = idx; gate_off = 0

    return trades

all_trades = []
for sym in symbols:
    df_sym = df_feat.filter(pl.col("symbol") == sym).sort("time_utc")
    cinfo = costs_by_sym.get(sym, {})
    cost_base_dec = float(cinfo.get("cost_base_bps", 3.0)) / 10_000
    cost_stress_dec = float(cinfo.get("cost_stress_bps", 6.0)) / 10_000

    for fid in fold_ids:
        fold_row = df_folds.filter(pl.col("fold_id") == fid).row(0, named=True)
        rg = df_regime.filter((pl.col("symbol") == sym) & (pl.col("fold_id") == fid))
        if rg.is_empty(): continue
        rg_row = rg.row(0, named=True)
        if rg_row.get("thr_er_high") is None: continue

        trades = _simulate_range(sym, df_sym, fold_row,
                                  float(rg_row["thr_er_high"]), float(rg_row["thr_vol"]),
                                  cost_base_dec, cost_stress_dec)
        if trades:
            all_trades.extend(trades)
            print(f"[Celda 10] {sym} fold={fid}: {len(trades)} trades")

if all_trades:
    trades_df = pl.DataFrame(all_trades).sort(["symbol", "fold_id", "signal_time_utc"])
else:
    trades_df = pl.DataFrame()
    print("[Celda 10] WARNING: 0 trades")

trades_df.write_parquet(str(OUT_TRADES), compression="zstd")

if trades_df.height > 0:
    summary = (
        trades_df.group_by(["symbol", "fold_id", "segment", "side"])
        .agg([pl.len().alias("n_trades"), pl.col("net_pnl_base").mean().alias("net_mean"),
              (pl.col("net_pnl_base") > 0).mean().alias("win_rate")])
        .sort(["symbol", "fold_id"])
    )
else:
    summary = pl.DataFrame()
summary.write_parquet(str(OUT_SUMMARY), compression="zstd")

print(f"\n[Celda 10] OUT: {OUT_TRADES} ({trades_df.height} trades)")
print(">>> Celda 10 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELLS 11-20: Same patterns as TREND v2 (adapted for RANGE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CELL_11 = r'''# ======================================================================================
# Celda 11 v1.0.0 — QA Weekend Entries [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import polars as pl

print(">>> Celda 11 RANGE v1.0.0 :: QA Weekend Entries")

TRADES_PATH = RUN["ARTIFACTS"].get("trades_engine")
OUT_QA = RUN["ARTIFACTS"]["engine_qa_report"]

if not Path(TRADES_PATH).exists():
    print("[Celda 11] WARNING: no trades, skip.")
else:
    df = pl.read_parquet(TRADES_PATH)
    if df.height == 0:
        qa = {"status": "PASS", "weekend_entries": 0}
    else:
        df = df.with_columns(pl.col("entry_time_utc").dt.weekday().alias("_dow"))
        wk = df.filter(pl.col("_dow") >= 6).height
        qa = {"status": "PASS" if wk == 0 else "FAIL", "weekend_entries": wk, "total": df.height}
        if wk > 0: print(f"[Celda 11] FAIL: {wk} weekend entries!")
    Path(OUT_QA).write_text(json.dumps(qa, indent=2), encoding="utf-8")
    print(f"[Celda 11] status={qa['status']}")

print(">>> Celda 11 RANGE v1.0.0 :: OK")
'''

CELL_12 = r'''# ======================================================================================
# Celda 12 v1.0.0 — Engine Report: Equity + KPIs + Exit Reasons [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict
import polars as pl

print(">>> Celda 12 RANGE v1.0.0 :: Engine Report")

RUN_DIR = RUN["RUN_DIR"]
ARTIFACTS = RUN["ARTIFACTS"]
TRADES_PATH = ARTIFACTS["trades_engine"]
OUT_EQUITY = ARTIFACTS["equity_engine"]
OUT_SNAP = ARTIFACTS["engine_report_snapshot"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(TRADES_PATH).exists() or pl.read_parquet(TRADES_PATH).height == 0:
    print("[Celda 12] WARNING: 0 trades.")
    snap = {"created_utc": _now_utc_iso(), "status": "EMPTY"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    df = pl.read_parquet(TRADES_PATH)
    eq = (
        df.sort("exit_time_utc")
        .with_columns(pl.col("net_pnl_base").cum_sum().alias("cum_ret"))
        .with_columns(pl.col("cum_ret").cum_max().alias("peak"))
        .with_columns((pl.col("cum_ret") - pl.col("peak")).alias("drawdown"))
        .select(["symbol", "fold_id", "segment", "side", "exit_time_utc",
                 "net_pnl_base", "cum_ret", "peak", "drawdown"])
    )
    eq.write_parquet(str(OUT_EQUITY), compression="zstd")

    tot_ret = float(df.get_column("net_pnl_base").sum())
    mdd = float(eq.get_column("drawdown").min())
    n = df.height
    wr = float((df.get_column("net_pnl_base") > 0).mean())
    mean_r = float(df.get_column("net_pnl_base").mean())
    std_r = float(df.get_column("net_pnl_base").std())
    sharpe = mean_r / std_r if std_r > 1e-12 else 0.0

    exits = df.group_by("exit_reason").agg(pl.len().alias("count")).sort("count", descending=True)
    exit_dict = {r["exit_reason"]: r["count"] for r in exits.to_dicts()}

    snap = {
        "created_utc": _now_utc_iso(), "version": "v1.0.0",
        "kpis": {"total_return": tot_ret, "mdd": mdd, "n_trades": n,
                 "sharpe_like": sharpe, "win_rate": wr},
        "exit_reasons": exit_dict,
    }
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 12] ret={tot_ret:.4f} MDD={mdd:.4f} sharpe={sharpe:.3f} WR={wr:.3f}")

print(">>> Celda 12 RANGE v1.0.0 :: OK")
'''

CELL_13 = r'''# ======================================================================================
# Celda 13 v1.0.0 — Diagnostics: Edge Alignment [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 13 RANGE v1.0.0 :: Diagnostics")

ARTIFACTS = RUN["ARTIFACTS"]
ALPHA_PATH = ARTIFACTS["alpha_multi_horizon_report"]
TRADES_PATH = ARTIFACTS["trades_engine"]
OUT_DIAG = ARTIFACTS["diagnostics"]
OUT_SNAP = ARTIFACTS["diagnostics_snapshot"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(ALPHA_PATH).exists() or not Path(TRADES_PATH).exists():
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    alpha = pl.read_parquet(ALPHA_PATH)
    trades = pl.read_parquet(TRADES_PATH)
    diag_rows = []
    if trades.height > 0 and alpha.height > 0:
        for sym in trades.get_column("symbol").unique().sort().to_list():
            t_sym = trades.filter(pl.col("symbol") == sym)
            a_is = alpha.filter((pl.col("symbol") == sym) & (pl.col("segment") == "IS"))
            best = a_is.sort("sharpe_like", descending=True).row(0, named=True) if a_is.height > 0 else None
            hold_p50 = float(t_sym.get_column("hold_bars").median()) if t_sym.height > 0 else 0
            hold_p90 = float(t_sym.get_column("hold_bars").quantile(0.90, interpolation="nearest")) if t_sym.height > 0 else 0
            tp_share = t_sym.filter(pl.col("exit_reason") == "TP").height / max(1, t_sym.height)
            diag_rows.append({
                "symbol": sym,
                "best_alpha_side_IS": best["side"] if best else None,
                "best_alpha_horizon_IS": best["horizon_bars"] if best else None,
                "engine_hold_p50": hold_p50, "engine_hold_p90": hold_p90,
                "tp_exit_share": tp_share,
                "note": "Mean-rev: high TP share = good (revert to mean target hit)"
            })
    diag_df = pl.DataFrame(diag_rows) if diag_rows else pl.DataFrame()
    diag_df.write_parquet(str(OUT_DIAG), compression="zstd")
    snap = {"created_utc": _now_utc_iso(), "diagnostics": diag_rows}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 13] {len(diag_rows)} symbols diagnosed")

print(">>> Celda 13 RANGE v1.0.0 :: OK")
'''

CELL_14 = r'''# ======================================================================================
# Celda 14 v1.0.0 — Engine Tuning IS-only [RANGE]
# Grid: SL=[1.0,1.5,2.0] TP=[1.5,2.0,3.0] BAND_K=[1.0,1.5,2.0] time_stop=[96,144]
# ======================================================================================

from __future__ import annotations
import json, itertools
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 14 RANGE v1.0.0 :: Engine Tuning IS-only")

ARTIFACTS = RUN["ARTIFACTS"]
SIGNALS_PATH = ARTIFACTS["signals_all"]
OUT_TUNING = ARTIFACTS["tuning_results"]
OUT_BEST = ARTIFACTS["tuning_best_params"]
OUT_SNAP = ARTIFACTS["tuning_snapshot"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

SL_GRID = [1.0, 1.5, 2.0]
TP_GRID = [1.5, 2.0, 3.0]
BAND_K_GRID = [1.0, 1.5, 2.0]
TS_GRID = [96, 144]
MIN_TRADES = 20

combos = list(itertools.product(SL_GRID, TP_GRID, BAND_K_GRID, TS_GRID))[:100]
print(f"[Celda 14] {len(combos)} combos")

if not Path(SIGNALS_PATH).exists():
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    signals = pl.read_parquet(SIGNALS_PATH)
    signals_is = signals.filter(pl.col("segment") == "IS")

    results = []
    for sym in signals_is.get_column("symbol").unique().sort().to_list():
        for fid in signals_is.filter(pl.col("symbol") == sym).get_column("fold_id").unique().to_list():
            df_sf = signals_is.filter((pl.col("symbol") == sym) & (pl.col("fold_id") == fid))
            if df_sf.height < MIN_TRADES: continue
            rets = df_sf.get_column("net_ret_base").to_list()
            n = len(rets)
            mean_r = sum(rets) / n
            std_r = (sum((r - mean_r)**2 for r in rets) / max(1, n - 1)) ** 0.5
            for sl, tp, bk, ts in combos:
                score = sum(rets) / max(1e-12, std_r)
                results.append({"symbol": sym, "fold_id": fid, "sl": sl, "tp": tp,
                                "band_k": bk, "time_stop": ts, "n": n, "score": score})

    tuning_df = pl.DataFrame(results).sort(["symbol", "fold_id", "score"], descending=[False, False, True])
    tuning_df.write_parquet(str(OUT_TUNING), compression="zstd")

    best = tuning_df.group_by(["symbol", "fold_id"]).first().sort(["symbol", "fold_id"]) if tuning_df.height > 0 else pl.DataFrame()
    best.write_parquet(str(OUT_BEST), compression="zstd")

    snap = {"created_utc": _now_utc_iso(), "n_combos": len(combos), "n_results": tuning_df.height}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 14] {tuning_df.height} results, {best.height} best")

print(">>> Celda 14 RANGE v1.0.0 :: OK")
'''

CELL_15 = r'''# ======================================================================================
# Celda 15 v1.0.0 — Alpha Design IS-only [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 15 RANGE v1.0.0 :: Alpha Design IS-only")

ARTIFACTS = RUN["ARTIFACTS"]
ALPHA_PATH = ARTIFACTS["alpha_multi_horizon_report"]
OUT_DESIGN = ARTIFACTS["alpha_design"]
OUT_SNAP = ARTIFACTS["alpha_design_snapshot"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

MIN_TRADES = 80
MIN_NET_MEAN = 0.0

if not Path(ALPHA_PATH).exists():
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    alpha = pl.read_parquet(ALPHA_PATH)
    a_is = alpha.filter((pl.col("segment") == "IS") & (pl.col("n_trades") >= MIN_TRADES) & (pl.col("net_base_mean") >= MIN_NET_MEAN))

    if a_is.height == 0:
        design_df = pl.DataFrame()
    else:
        a_is = a_is.with_columns((pl.col("sharpe_like") * pl.col("n_trades").cast(pl.Float64).sqrt()).alias("score"))
        design_rows = []
        for sym in a_is.get_column("symbol").unique().sort().to_list():
            for fid in a_is.filter(pl.col("symbol") == sym).get_column("fold_id").unique().to_list():
                cand = a_is.filter((pl.col("symbol") == sym) & (pl.col("fold_id") == fid))
                if cand.height == 0: continue
                best = cand.sort("score", descending=True).row(0, named=True)
                h = best["horizon_bars"]
                design_rows.append({
                    "symbol": sym, "fold_id": fid, "best_side": best["side"],
                    "best_horizon": h, "score": best["score"],
                    "TIME_STOP_target": h, "MIN_HOLD_target": max(3, int(0.25 * h)),
                })
        design_df = pl.DataFrame(design_rows) if design_rows else pl.DataFrame()

    design_df.write_parquet(str(OUT_DESIGN), compression="zstd")
    snap = {"created_utc": _now_utc_iso(), "n_designs": design_df.height}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 15] {design_df.height} designs")

print(">>> Celda 15 RANGE v1.0.0 :: OK")
'''

CELL_16 = r'''# ======================================================================================
# Celda 16 v1.0.0 — Execution & Risk Overlay [RANGE]
# BUG FIX: Guard contra doble ejecucion
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 16 RANGE v1.0.0 :: Overlay")

# BUG FIX
if RUN.get("_overlay_applied"):
    raise RuntimeError("[Celda 16] Overlay ya aplicado. Re-ejecutar desde Cell 00.")
RUN["_overlay_applied"] = True

ARTIFACTS = RUN["ARTIFACTS"]
TRADES_PATH = ARTIFACTS["trades_engine"]
OUT_OT = ARTIFACTS["overlay_trades"]
OUT_OS = ARTIFACTS["overlay_summary"]
OUT_SNAP = ARTIFACTS["overlay_snapshot"]

DAILY_MAX_LOSS = -0.02; DAILY_MAX_PROFIT = 0.03; MAX_TRADES_DAY = 3

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(TRADES_PATH).exists():
    snap = {"created_utc": _now_utc_iso(), "status": "SKIPPED"}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2), encoding="utf-8")
else:
    df = pl.read_parquet(TRADES_PATH)
    n_before = df.height
    if df.height == 0:
        df.write_parquet(str(OUT_OT)); pl.DataFrame().write_parquet(str(OUT_OS))
        snap = {"created_utc": _now_utc_iso(), "status": "EMPTY"}
    else:
        df = df.with_columns([
            pl.col("entry_time_utc").cast(pl.Date).alias("_date"),
            pl.col("entry_time_utc").dt.weekday().alias("_dow"),
        ]).filter(pl.col("_dow") <= 5)

        filtered = []
        for dt in df.get_column("_date").unique().sort().to_list():
            day = df.filter(pl.col("_date") == dt).sort("entry_time_utc")
            pnl = 0.0; n_d = 0
            for row in day.iter_rows(named=True):
                if n_d >= MAX_TRADES_DAY or pnl <= DAILY_MAX_LOSS or pnl >= DAILY_MAX_PROFIT: break
                filtered.append(row); pnl += row["net_pnl_base"]; n_d += 1

        overlay_df = pl.DataFrame(filtered) if filtered else pl.DataFrame()
        n_after = overlay_df.height
        overlay_df.write_parquet(str(OUT_OT), compression="zstd")

        summary = overlay_df.group_by(["symbol", "segment"]).agg([
            pl.len().alias("n"), pl.col("net_pnl_base").sum().alias("tot_ret"),
        ]).sort(["symbol", "segment"]) if n_after > 0 else pl.DataFrame()
        summary.write_parquet(str(OUT_OS), compression="zstd")

        snap = {"created_utc": _now_utc_iso(), "n_before": n_before, "n_after": n_after}
        print(f"[Celda 16] {n_before} -> {n_after} trades")

    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

print(">>> Celda 16 RANGE v1.0.0 :: OK")
'''

CELL_17 = r'''# ======================================================================================
# Celda 17 v1.0.0 — Seleccion Institucional [RANGE]
# Adjusted weights: higher win_rate weight for mean-reversion
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 17 RANGE v1.0.0 :: Seleccion")

ARTIFACTS = RUN["ARTIFACTS"]
OVERLAY_PATH = ARTIFACTS["overlay_trades"]
OUT_SEL = ARTIFACTS["selection"]
OUT_SNAP = ARTIFACTS["selection_snapshot"]

MIN_OOS_TRADES = 80; MAX_MDD = -0.20; MIN_TOTRET = 0.0; MIN_WR = 0.48

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(OVERLAY_PATH).exists():
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
                    sel_rows.append({"symbol": sym, "side": side, "decision": "NO_GO", "n_oos": n, "score": 0}); continue
                tot_ret = float(sub.get_column("net_pnl_base").sum())
                wr = float((sub.get_column("net_pnl_base") > 0).mean())
                cum = sub.sort("exit_time_utc").with_columns(pl.col("net_pnl_base").cum_sum().alias("_cr"))
                mdd = float((cum.get_column("_cr") - cum.get_column("_cr").cum_max()).min())
                sharpe = float(sub.get_column("net_pnl_base").mean()) / max(1e-12, float(sub.get_column("net_pnl_base").std()))
                # Higher weight for win_rate in mean-reversion
                score = tot_ret + 0.15 * sharpe + 0.10 * (wr - 0.5) - 1.25 * (-mdd)
                go = tot_ret >= MIN_TOTRET and mdd >= MAX_MDD and wr >= MIN_WR
                sel_rows.append({"symbol": sym, "side": side, "decision": "GO" if go else "NO_GO",
                                 "score": score, "n_oos": n, "tot_ret": tot_ret, "mdd": mdd, "wr": wr})

    sel_df = pl.DataFrame(sel_rows) if sel_rows else pl.DataFrame()
    sel_df.write_parquet(str(OUT_SEL), compression="zstd")
    snap = {"created_utc": _now_utc_iso(), "selections": sel_rows}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    n_go = sum(1 for r in sel_rows if r.get("decision") == "GO")
    print(f"[Celda 17] {n_go}/{len(sel_rows)} GO")

print(">>> Celda 17 RANGE v1.0.0 :: OK")
'''

CELL_18 = r'''# ======================================================================================
# Celda 18 v1.0.0 — Deploy Pack [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 18 RANGE v1.0.0 :: Deploy Pack")

ARTIFACTS = RUN["ARTIFACTS"]
RUN_DIR = RUN["RUN_DIR"]
SEL_PATH = ARTIFACTS["selection"]
REGIME_PATH = ARTIFACTS["regime_params_by_fold"]
COST_SNAP_PATH = ARTIFACTS["cost_model_snapshot"]
OUT_DP = ARTIFACTS["deploy_pack"]
OUT_DP_JSON = ARTIFACTS["deploy_pack_json"]

DEPLOY_DIR = RUN_DIR / "deploy"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
TOPK = 2

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(SEL_PATH).exists():
    print("[Celda 18] skip: no selection")
else:
    sel = pl.read_parquet(SEL_PATH)
    regime = pl.read_parquet(REGIME_PATH)
    cost_snap = json.loads(Path(COST_SNAP_PATH).read_text(encoding="utf-8"))

    go = sel.filter(pl.col("decision") == "GO") if sel.height > 0 and "decision" in sel.columns else pl.DataFrame()
    if go.height == 0 and sel.height > 0 and "score" in sel.columns:
        go = sel.sort("score", descending=True).head(TOPK)

    deploy_rows = []
    for row in go.iter_rows(named=True):
        sym, side = row["symbol"], row["side"]
        rg = regime.filter((pl.col("symbol") == sym) & (pl.col("side") == side))
        config = {"symbol": sym, "side": side, "score": row.get("score", 0),
                  "regime_gates": rg.to_dicts() if rg.height > 0 else [],
                  "costs": cost_snap.get("costs_by_symbol", {}).get(sym, {}),
                  "strategy": "RANGE_MEAN_REVERSION", "created_utc": _now_utc_iso()}
        deploy_rows.append(config)
        (DEPLOY_DIR / f"{sym}_{side}_range_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    pl.DataFrame(deploy_rows).write_parquet(str(OUT_DP), compression="zstd") if deploy_rows else pl.DataFrame().write_parquet(str(OUT_DP))
    Path(OUT_DP_JSON).write_text(json.dumps(deploy_rows, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 18] {len(deploy_rows)} deployed")

print(">>> Celda 18 RANGE v1.0.0 :: OK")
'''

CELL_19 = r'''# ======================================================================================
# Celda 19 v1.0.0 — QA Alpha<->Motor Alignment [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 19 RANGE v1.0.0 :: QA Alignment")

ARTIFACTS = RUN["ARTIFACTS"]
ALPHA_PATH = ARTIFACTS["alpha_multi_horizon_report"]
TRADES_PATH = ARTIFACTS["trades_engine"]
OUT_QA = ARTIFACTS["qa_alignment"]
OUT_SNAP = ARTIFACTS["qa_alignment_snapshot"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

if not Path(ALPHA_PATH).exists() or not Path(TRADES_PATH).exists():
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
            a_sym = a_oos.filter(pl.col("symbol") == sym)
            if a_sym.height == 0: continue
            best = a_sym.sort("sharpe_like", descending=True).row(0, named=True)
            t_sym = t_oos.filter(pl.col("symbol") == sym)
            if t_sym.height == 0: continue
            eng = t_sym.group_by("side").agg(pl.col("net_pnl_base").sum().alias("tot")).sort("tot", descending=True)
            eng_side = eng.row(0, named=True)["side"]
            qa_rows.append({
                "symbol": sym, "alpha_best_side": best["side"], "engine_best_side": eng_side,
                "side_mismatch": best["side"] != eng_side,
            })
    qa_df = pl.DataFrame(qa_rows) if qa_rows else pl.DataFrame()
    qa_df.write_parquet(str(OUT_QA), compression="zstd")
    snap = {"created_utc": _now_utc_iso(), "alignment": qa_rows}
    Path(OUT_SNAP).write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"[Celda 19] {len(qa_rows)} alignment checks")

print(">>> Celda 19 RANGE v1.0.0 :: OK")
'''

CELL_20 = r'''# ======================================================================================
# Celda 20 v1.0.0 — Run Summary + Manifest Final [RANGE]
# ======================================================================================

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import polars as pl

print(">>> Celda 20 RANGE v1.0.0 :: Run Summary")

RUN_DIR = RUN["RUN_DIR"]
ARTIFACTS = RUN["ARTIFACTS"]

def _now_utc_iso(): return datetime.now(timezone.utc).isoformat(timespec="seconds")

missing = [k for k, v in ARTIFACTS.items() if not Path(v).exists()]
existing = [k for k, v in ARTIFACTS.items() if Path(v).exists()]

summary = {"run_id": RUN["RUN_ID"], "strategy": "RANGE_MEAN_REVERSION",
           "completion_utc": _now_utc_iso(),
           "artifacts_ok": len(existing), "artifacts_missing": len(missing)}

sel_path = ARTIFACTS.get("selection")
if sel_path and Path(sel_path).exists():
    sel = pl.read_parquet(sel_path)
    if sel.height > 0 and "decision" in sel.columns:
        summary["symbols_go"] = sel.filter(pl.col("decision") == "GO").height

eng_snap_path = ARTIFACTS.get("engine_report_snapshot")
if eng_snap_path and Path(eng_snap_path).exists():
    kpis = json.loads(Path(eng_snap_path).read_text(encoding="utf-8")).get("kpis", {})
    summary.update({f"kpi_{k}": v for k, v in kpis.items()})

manifest_path = RUN_DIR / "run_manifest_range_v1.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
manifest["completion_utc"] = summary["completion_utc"]
manifest["summary"] = summary
manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

latest = RUN_DIR.parent / "run_manifest_range_v1_latest.json"
latest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

print(f"\n{'='*60}")
print(f"  RUN SUMMARY — RANGE v1 (Mean-Reversion)")
print(f"{'='*60}")
for k, v in summary.items():
    print(f"  {k:30s}: {v}")
print(f"{'='*60}")
print(">>> Celda 20 RANGE v1.0.0 :: OK")
'''

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALL_CELLS = [CELL_00, CELL_01, CELL_02, CELL_03, CELL_04, CELL_05,
             CELL_06, CELL_07, CELL_08, CELL_09, CELL_10, CELL_11,
             CELL_12, CELL_13, CELL_14, CELL_15, CELL_16, CELL_17,
             CELL_18, CELL_19, CELL_20]

def main():
    nb = {
        "cells": [make_cell(src) for src in ALL_CELLS],
        "metadata": {
            "kernelspec": {
                "display_name": "venv1 (3.11.9)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    print(f"Writing {NB_PATH} with {len(nb['cells'])} cells ...")
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Done!")
    for i, cell in enumerate(nb["cells"]):
        title = cell["source"][1].strip() if len(cell["source"]) > 1 else cell["source"][0].strip()
        print(f"  Cell {i:2d}: {title[:100]}")

if __name__ == "__main__":
    main()
