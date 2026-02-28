"""
Patch NB4: Rediseño filtros entrada RANGE
  Filtro A — range_width_atr compacto (Celdas 06 + 10 + 14)
  Filtro B — dist_mean_atr_v3 velocidad de reversión (Celdas 05 + 07 + 10 + 14)

Modifica 5 celdas del NB4 via string replace.

Uso:
    python tools/_patch_nb4_filter.py [--dry-run]
"""
import json
import sys
from pathlib import Path

NB4_PATH = Path(__file__).parent.parent / "03_STRATEGY_LAB" / "notebooks" / "04_RANGE_M5_Strategy_v1.ipynb"


def assert_replace(src: str, old: str, new: str, cell_id: str) -> str:
    if old not in src:
        # Show context around where we expected to find it
        raise ValueError(
            f"[{cell_id}] String NOT FOUND in cell source.\n"
            f"Expected:\n{old!r}\n"
        )
    count = src.count(old)
    if count > 1:
        raise ValueError(
            f"[{cell_id}] String found {count} times (expected 1):\n{old!r}\n"
        )
    return src.replace(old, new, 1)


def get_source(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def set_source(cell, src: str) -> None:
    cell["source"] = src.splitlines(keepends=True)


# =============================================================================
# Celda 05 — v1.0.0 → v1.1.0
# Añade dist_mean_atr_v3 (velocidad a 3 barras) como nueva feature
# =============================================================================
def patch_cell05(src: str) -> str:
    cid = "16060e55"

    # 1. Header version
    src = assert_replace(src,
        "# Celda 05 v1.0.0",
        "# Celda 05 v1.1.0",
        cid)

    # 2. Print statement
    src = assert_replace(src,
        'print(">>> Celda 05 RANGE v1.0.0 :: Feature Set (Range)")',
        'print(">>> Celda 05 RANGE v1.1.0 :: Feature Set (Range)")',
        cid)

    # 3. Añadir bloque lf4.with_columns([dist_mean_atr_v3]) antes de lf_feat = lf4.select
    src = assert_replace(src,
        '    lf_feat = lf4.select([',
        '    lf4 = lf4.with_columns([\n'
        '        # Velocity of dist_mean_atr over 3 bars (positive = turning toward mean for LONG)\n'
        '        ((pl.col("dist_mean_atr") - pl.col("dist_mean_atr").shift(3).over("symbol"))\n'
        '         .alias("dist_mean_atr_v3")),\n'
        '    ])\n'
        '\n'
        '    lf_feat = lf4.select([',
        cid)

    # 4. Añadir "dist_mean_atr_v3" al select
    src = assert_replace(src,
        '        "pct_b", "dist_mean_atr", "range_width_atr",\n'
        '    ]).sort(["symbol", "time_utc"])',
        '        "pct_b", "dist_mean_atr", "range_width_atr", "dist_mean_atr_v3",\n'
        '    ]).sort(["symbol", "time_utc"])',
        cid)

    # 5. Snap version
    src = assert_replace(src,
        '"version": "v1.0.0",\n        "params": {"ER_WIN":',
        '"version": "v1.1.0",\n        "params": {"ER_WIN":',
        cid)

    # 6. End print
    src = assert_replace(src,
        'print(">>> Celda 05 RANGE v1.0.0 :: OK")',
        'print(">>> Celda 05 RANGE v1.1.0 :: OK")',
        cid)

    return src


# =============================================================================
# Celda 06 — v1.0.0 → v1.1.0
# Añade Q_RANGE_WIDTH y thr_range_width al regime gate
# =============================================================================
def patch_cell06(src: str) -> str:
    cid = "defd40fc"

    # 1. Header version
    src = assert_replace(src,
        "# Celda 06 v1.0.0",
        "# Celda 06 v1.1.0",
        cid)

    # 2. Print statement
    src = assert_replace(src,
        'print(">>> Celda 06 RANGE v1.0.0 :: Regime Gate (ranging markets)")',
        'print(">>> Celda 06 RANGE v1.1.0 :: Regime Gate (ranging markets)")',
        cid)

    # 3. Añadir Q_RANGE_WIDTH y RANGE_WIDTH_COL después de Q_VOL
    src = assert_replace(src,
        'Q_VOL = 0.60       # vol below 80th percentile = low volatility\n'
        '\n'
        'COV_IS_MIN',
        'Q_VOL = 0.60       # vol below 80th percentile = low volatility\n'
        'Q_RANGE_WIDTH = 0.80   # range_width_atr at or below 80th pct = non-extreme range\n'
        'RANGE_WIDTH_COL = "range_width_atr"\n'
        '\n'
        'COV_IS_MIN',
        cid)

    # 4. SKIP row: añadir thr_range_width: None
    src = assert_replace(src,
        '                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "SKIP",\n'
        '                            "thr_er_high": None, "thr_vol": None, "cov_is": 0.0, "cov_oos": 0.0,\n'
        '                            "n_is": df_is.height, "n_oos": df_oos.height})',
        '                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "SKIP",\n'
        '                            "thr_er_high": None, "thr_vol": None, "thr_range_width": None,\n'
        '                            "cov_is": 0.0, "cov_oos": 0.0,\n'
        '                            "n_is": df_is.height, "n_oos": df_oos.height})',
        cid)

    # 5. FAIL row: añadir thr_range_width: None
    src = assert_replace(src,
        '                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "FAIL",\n'
        '                            "thr_er_high": None, "thr_vol": None, "cov_is": 0.0, "cov_oos": 0.0,\n'
        '                            "n_is": df_is.height, "n_oos": df_oos.height})',
        '                rows.append({"symbol": sym, "fold_id": fid, "side": side, "scheme": "FAIL",\n'
        '                            "thr_er_high": None, "thr_vol": None, "thr_range_width": None,\n'
        '                            "cov_is": 0.0, "cov_oos": 0.0,\n'
        '                            "n_is": df_is.height, "n_oos": df_oos.height})',
        cid)

    # 6. Añadir cálculo thr_range_width después de thr_vol (antes del None check)
    src = assert_replace(src,
        '            thr_er = _q_safe(df_is.get_column(ER_COL), Q_ER_HIGH)\n'
        '            thr_vol = _q_safe(df_is.get_column(VOL_COL), Q_VOL)\n'
        '\n'
        '            if thr_er is None or thr_vol is None:',
        '            thr_er = _q_safe(df_is.get_column(ER_COL), Q_ER_HIGH)\n'
        '            thr_vol = _q_safe(df_is.get_column(VOL_COL), Q_VOL)\n'
        '            thr_range_width = _q_safe(df_is.get_column(RANGE_WIDTH_COL), Q_RANGE_WIDTH)\n'
        '\n'
        '            if thr_er is None or thr_vol is None:',
        cid)

    # 7. Actualizar gate condition (añadir range_width condicional)
    src = assert_replace(src,
        '            gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '            cov_is',
        '            gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '            if thr_range_width is not None:\n'
        '                gate = gate & (pl.col(RANGE_WIDTH_COL) <= thr_range_width)\n'
        '            cov_is',
        cid)

    # 8. BASE row: añadir thr_range_width
    src = assert_replace(src,
        '                "thr_er_high": float(thr_er), "thr_vol": float(thr_vol),\n'
        '                "cov_is": cov_is,',
        '                "thr_er_high": float(thr_er), "thr_vol": float(thr_vol),\n'
        '                "thr_range_width": float(thr_range_width) if thr_range_width is not None else None,\n'
        '                "cov_is": cov_is,',
        cid)

    # 9. Snap: versión + gate_type + params
    src = assert_replace(src,
        'snap = {"created_utc": _now_utc_iso(), "version": "v1.0.0",\n'
        '        "gate_type": "RANGE (ER<=thr, vol<=thr)",\n'
        '        "params": {"Q_ER_HIGH": Q_ER_HIGH, "Q_VOL": Q_VOL}}',
        'snap = {"created_utc": _now_utc_iso(), "version": "v1.1.0",\n'
        '        "gate_type": "RANGE (ER<=thr, vol<=thr, range_width<=thr)",\n'
        '        "params": {"Q_ER_HIGH": Q_ER_HIGH, "Q_VOL": Q_VOL, "Q_RANGE_WIDTH": Q_RANGE_WIDTH}}',
        cid)

    # 10. End print
    src = assert_replace(src,
        'print(">>> Celda 06 RANGE v1.0.0 :: OK")',
        'print(">>> Celda 06 RANGE v1.1.0 :: OK")',
        cid)

    return src


# =============================================================================
# Celda 07 — v1.0.0 → v1.1.0
# Añade thr_range_width al regime gate + dist_mean_atr_v3 a signal gates
# =============================================================================
def patch_cell07(src: str) -> str:
    cid = "bd71f11d"

    # 1. Header version
    src = assert_replace(src,
        "# Celda 07 v1.0.0",
        "# Celda 07 v1.1.0",
        cid)

    # 2. Print statement
    src = assert_replace(src,
        'print(">>> Celda 07 RANGE v1.0.0 :: Senales Mean-Reversion")',
        'print(">>> Celda 07 RANGE v1.1.0 :: Senales Mean-Reversion")',
        cid)

    # 3. Añadir RANGE_WIDTH_COL y DIST_V3_COL junto a DIST_COL
    src = assert_replace(src,
        'ER_COL = "er_288"\n'
        'VOL_COL = "vol_bps_288"\n'
        'DIST_COL = "dist_mean_atr"\n'
        'BAND_K = 1.5',
        'ER_COL = "er_288"\n'
        'VOL_COL = "vol_bps_288"\n'
        'DIST_COL = "dist_mean_atr"\n'
        'RANGE_WIDTH_COL = "range_width_atr"\n'
        'DIST_V3_COL = "dist_mean_atr_v3"\n'
        'BAND_K = 1.5',
        cid)

    # 4. Actualizar thr_er/thr_vol + regime_gate + signal_gate
    src = assert_replace(src,
        '            thr_er = float(rg_row["thr_er_high"])\n'
        '            thr_vol = float(rg_row["thr_vol"])\n'
        '\n'
        '            # Regime gate (ranging market)\n'
        '            regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '\n'
        '            # Mean-reversion signal\n'
        '            if side == "LONG":\n'
        '                signal_gate = regime_gate & (pl.col(DIST_COL) <= -BAND_K)\n'
        '            else:\n'
        '                signal_gate = regime_gate & (pl.col(DIST_COL) >= BAND_K)\n',
        '            thr_er = float(rg_row["thr_er_high"])\n'
        '            thr_vol = float(rg_row["thr_vol"])\n'
        '            thr_range_width = rg_row.get("thr_range_width")\n'
        '\n'
        '            # Regime gate (ranging market)\n'
        '            regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '            if thr_range_width is not None:\n'
        '                regime_gate = regime_gate & (pl.col(RANGE_WIDTH_COL) <= thr_range_width)\n'
        '\n'
        '            # Mean-reversion signal + velocity confirmation\n'
        '            DIST_V3_THRESHOLD = 0.1   # ATR units\n'
        '            if side == "LONG":\n'
        '                signal_gate = regime_gate & (pl.col(DIST_COL) <= -BAND_K) & (pl.col(DIST_V3_COL) > DIST_V3_THRESHOLD)\n'
        '            else:\n'
        '                signal_gate = regime_gate & (pl.col(DIST_COL) >= BAND_K) & (pl.col(DIST_V3_COL) < -DIST_V3_THRESHOLD)\n',
        cid)

    # 5. Añadir DIST_V3_COL al select final
    src = assert_replace(src,
        '                    ER_COL, VOL_COL, DIST_COL,\n'
        '                ])\n'
        '            )\n',
        '                    ER_COL, VOL_COL, DIST_COL, DIST_V3_COL,\n'
        '                ])\n'
        '            )\n',
        cid)

    # 6. Snap version
    src = assert_replace(src,
        '"version": "v1.0.0", "n_signals": signals_df.height,',
        '"version": "v1.1.0", "n_signals": signals_df.height,',
        cid)

    # 7. End print
    src = assert_replace(src,
        'print(">>> Celda 07 RANGE v1.0.0 :: OK")',
        'print(">>> Celda 07 RANGE v1.1.0 :: OK")',
        cid)

    return src


# =============================================================================
# Celda 10 — v1.2.0 → v1.3.0
# Añade thr_range_width y DIST_V3_COL al engine bar-by-bar
# =============================================================================
def patch_cell10(src: str) -> str:
    cid = "8f3f03ad"

    # 1. Header version
    src = assert_replace(src,
        "# Celda 10 v1.2.0",
        "# Celda 10 v1.3.0",
        cid)

    # 2. Print statement
    src = assert_replace(src,
        'print(">>> Celda 10 RANGE v1.2.0 :: Backtest Engine (Mean-Reversion) [weekend exec-bar fix]")',
        'print(">>> Celda 10 RANGE v1.3.0 :: Backtest Engine (Mean-Reversion) [range_width + v3 filters]")',
        cid)

    # 3. Añadir RANGE_WIDTH_COL y DIST_V3_COL
    src = assert_replace(src,
        'ER_COL = "er_288"\n'
        'VOL_COL = "vol_bps_288"\n'
        'DIST_COL = "dist_mean_atr"\n'
        'ATR_COL = "atr_bps_96"',
        'ER_COL = "er_288"\n'
        'VOL_COL = "vol_bps_288"\n'
        'DIST_COL = "dist_mean_atr"\n'
        'RANGE_WIDTH_COL = "range_width_atr"\n'
        'DIST_V3_COL     = "dist_mean_atr_v3"\n'
        'ATR_COL = "atr_bps_96"',
        cid)

    # 4. Actualizar firma de _simulate_range (añadir thr_range_width=None)
    src = assert_replace(src,
        'def _simulate_range(sym, df_j, fold_row, thr_er, thr_vol, cost_base_dec, cost_stress_dec,\n'
        '                    *, sl_atr=None, tp_atr=None, band_k=None, time_stop=None):',
        'def _simulate_range(sym, df_j, fold_row, thr_er, thr_vol, cost_base_dec, cost_stress_dec,\n'
        '                    *, sl_atr=None, tp_atr=None, band_k=None, time_stop=None, thr_range_width=None):',
        cid)

    # 5. Actualizar regime_gate + signals dentro de _simulate_range
    src = assert_replace(src,
        '    # Regime gate + signal gates (using _BK)\n'
        '    regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '    long_signal = regime_gate & (pl.col(DIST_COL) <= -_BK)\n'
        '    short_signal = regime_gate & (pl.col(DIST_COL) >= _BK)\n',
        '    # Regime gate + signal gates (using _BK)\n'
        '    _DIST_V3_THR = 0.1   # ATR units\n'
        '    regime_gate = (pl.col(ER_COL) <= thr_er) & (pl.col(VOL_COL) <= thr_vol)\n'
        '    if thr_range_width is not None:\n'
        '        regime_gate = regime_gate & (pl.col(RANGE_WIDTH_COL) <= thr_range_width)\n'
        '    long_signal  = regime_gate & (pl.col(DIST_COL) <= -_BK) & (pl.col(DIST_V3_COL) >  _DIST_V3_THR)\n'
        '    short_signal = regime_gate & (pl.col(DIST_COL) >= _BK)  & (pl.col(DIST_V3_COL) < -_DIST_V3_THR)\n',
        cid)

    # 6. Actualizar llamada a _simulate_range en el loop principal
    src = assert_replace(src,
        '        trades = _simulate_range(sym, df_sym, fold_row,\n'
        '                                  float(rg_row["thr_er_high"]), float(rg_row["thr_vol"]),\n'
        '                                  cost_base_dec, cost_stress_dec)\n',
        '        thr_range_width = rg_row.get("thr_range_width")\n'
        '        trades = _simulate_range(sym, df_sym, fold_row,\n'
        '                                  float(rg_row["thr_er_high"]), float(rg_row["thr_vol"]),\n'
        '                                  cost_base_dec, cost_stress_dec,\n'
        '                                  thr_range_width=thr_range_width)\n',
        cid)

    # 7. End print
    src = assert_replace(src,
        'print(">>> Celda 10 RANGE v1.2.0 :: OK")',
        'print(">>> Celda 10 RANGE v1.3.0 :: OK")',
        cid)

    return src


# =============================================================================
# Celda 14 — v1.1.0 → v1.2.0
# Pasa thr_range_width al _simulate_range en el tuning loop
# =============================================================================
def patch_cell14(src: str) -> str:
    cid = "b41156f4"

    # 1. Header version
    src = assert_replace(src,
        "# Celda 14 v1.1.0",
        "# Celda 14 v1.2.0",
        cid)

    # 2. Print statement
    src = assert_replace(src,
        'print(">>> Celda 14 RANGE v1.1.0 :: Engine Tuning REAL (IS-only)")',
        'print(">>> Celda 14 RANGE v1.2.0 :: Engine Tuning REAL (IS-only)")',
        cid)

    # 3. Añadir thr_range_width después de thr_vol en el loop de folds
    src = assert_replace(src,
        '        thr_er = float(rg_row["thr_er_high"])\n'
        '        thr_vol = float(rg_row["thr_vol"])\n'
        '\n'
        '        for sl, tp, bk, ts in combos:',
        '        thr_er = float(rg_row["thr_er_high"])\n'
        '        thr_vol = float(rg_row["thr_vol"])\n'
        '        thr_range_width = rg_row.get("thr_range_width")\n'
        '\n'
        '        for sl, tp, bk, ts in combos:',
        cid)

    # 4. Actualizar llamada a _simulate_range en el tuning loop
    src = assert_replace(src,
        '            trades = _simulate_range(sym, df_sym, fold_row,\n'
        '                                      thr_er, thr_vol,\n'
        '                                      cost_base_dec, cost_stress_dec,\n'
        '                                      sl_atr=sl, tp_atr=tp, band_k=bk, time_stop=ts)\n',
        '            trades = _simulate_range(sym, df_sym, fold_row,\n'
        '                                      thr_er, thr_vol,\n'
        '                                      cost_base_dec, cost_stress_dec,\n'
        '                                      sl_atr=sl, tp_atr=tp, band_k=bk, time_stop=ts,\n'
        '                                      thr_range_width=thr_range_width)\n',
        cid)

    # 5. Snap version
    src = assert_replace(src,
        '"version": "v1.1.0",\n    "grid": {"SL": SL_GRID,',
        '"version": "v1.2.0",\n    "grid": {"SL": SL_GRID,',
        cid)

    # 6. End print
    src = assert_replace(src,
        'print(">>> Celda 14 RANGE v1.1.0 :: OK")',
        'print(">>> Celda 14 RANGE v1.2.0 :: OK")',
        cid)

    return src


# =============================================================================
# Main
# =============================================================================
PATCHES = {
    "16060e55": patch_cell05,
    "defd40fc": patch_cell06,
    "bd71f11d": patch_cell07,
    "8f3f03ad": patch_cell10,
    "b41156f4": patch_cell14,
}


def patch(dry_run: bool = False) -> None:
    nb = json.loads(NB4_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    patched = 0
    for cell in cells:
        cid = cell.get("id")
        if cid not in PATCHES:
            continue
        src_before = get_source(cell)
        src_after = PATCHES[cid](src_before)
        if dry_run:
            # Show diff summary
            lines_before = src_before.count("\n")
            lines_after = src_after.count("\n")
            delta = lines_after - lines_before
            print(f"[dry-run] Celda {cid}: {lines_before} -> {lines_after} lines ({delta:+d})")
        else:
            set_source(cell, src_after)
        patched += 1

    print(f"\n[patch] {patched}/5 celdas parchadas.")

    if dry_run:
        print("[patch] DRY-RUN: no se escribió nada.")
        return

    NB4_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[patch] NB4 guardado: {NB4_PATH}")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    patch(dry_run=dry)
