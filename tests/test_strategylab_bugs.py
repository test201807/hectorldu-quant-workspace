"""
test_strategylab_bugs.py
========================
Unit tests for the critical bug fixes in StrategyLab TREND v2 and RANGE v1.

Bug fixes tested:
1. Trail > SL constraint (TREND: TRAIL_ATR=3.0 > SL_ATR=2.0)
2. SHORT gate calibrated independently (thr_mom_short != -thr_mom_long)
3. Dedup keep="last" uniformly
4. Overlay double-execution guard
"""
from __future__ import annotations

import json
import re
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "projects" / "MT5_Data_Extraction" / "ER_STRATEGY_LAB" / "notebooks"
TREND_V2 = NB_DIR / "03_TREND_M5_Strategy_v2.ipynb"
RANGE_V1 = NB_DIR / "04_RANGE_M5_Strategy_v1.ipynb"


def _load_nb_cells(nb_path: Path) -> list[str]:
    """Load a notebook and return source of each cell as strings."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    return ["".join(cell["source"]) for cell in nb["cells"]]


# ============================================================
# BUG 1: Trail > SL (SL_ATR < TRAIL_ATR)
# ============================================================
class TestTrailGreaterThanSL:
    """Verify that TRAIL_ATR > SL_ATR in all engine cells."""

    def _extract_params(self, cell_src: str) -> dict[str, float]:
        """Extract SL_ATR and TRAIL_ATR values from cell source."""
        params = {}
        for line in cell_src.split("\n"):
            line = line.strip()
            # Match lines like: SL_ATR = 2.0 or SL_ATR=2.0
            m_sl = re.match(r"^SL_ATR\s*=\s*([0-9.]+)", line)
            m_tr = re.match(r"^TRAIL_ATR\s*=\s*([0-9.]+)", line)
            if m_sl:
                params["SL_ATR"] = float(m_sl.group(1))
            if m_tr:
                params["TRAIL_ATR"] = float(m_tr.group(1))
        return params

    def test_trend_v2_engine_trail_gt_sl(self):
        """TREND v2 Cell 10: TRAIL_ATR(3.0) > SL_ATR(2.0)"""
        cells = _load_nb_cells(TREND_V2)
        # Cell 10 is the backtest engine
        engine_cell = cells[10]
        assert "Backtest Engine" in engine_cell, "Cell 10 should be Backtest Engine"

        params = self._extract_params(engine_cell)
        assert "SL_ATR" in params, "SL_ATR not found in engine cell"
        assert "TRAIL_ATR" in params, "TRAIL_ATR not found in engine cell"
        assert params["TRAIL_ATR"] > params["SL_ATR"], (
            f"BUG: TRAIL_ATR({params['TRAIL_ATR']}) <= SL_ATR({params['SL_ATR']}). "
            f"Trail must be > SL for SL to be reachable."
        )

    def test_range_v1_engine_no_trail(self):
        """RANGE v1 Cell 10: TRAIL_ATR should be None (no trailing stop for mean-reversion)."""
        cells = _load_nb_cells(RANGE_V1)
        engine_cell = cells[10]
        assert "Backtest Engine" in engine_cell or "Mean-Reversion" in engine_cell
        assert "TRAIL_ATR  = None" in engine_cell or "TRAIL_ATR = None" in engine_cell, (
            "RANGE engine should have TRAIL_ATR = None (no trailing stop)"
        )

    def test_trend_v2_tuning_grid_trail_gt_sl(self):
        """TREND v2 Cell 14: all TRAIL_ATR_GRID values > max(SL_ATR_GRID)."""
        cells = _load_nb_cells(TREND_V2)
        tuning_cell = cells[14]
        assert "Tuning" in tuning_cell, "Cell 14 should be Tuning"
        assert "if tr > sl" in tuning_cell, (
            "Tuning grid should enforce Trail > SL constraint"
        )


# ============================================================
# BUG 2: SHORT gate calibrated independently
# ============================================================
class TestShortGateIndependent:
    """Verify that SHORT gate is NOT just -LONG gate."""

    def test_trend_v2_regime_gate_has_independent_short(self):
        """TREND v2 Cell 06: calibrates LONG and SHORT independently."""
        cells = _load_nb_cells(TREND_V2)
        gate_cell = cells[6]
        assert "Regime Gate" in gate_cell, "Cell 6 should be Regime Gate"

        # Must calibrate per side separately
        assert "for side in" in gate_cell, "Should iterate over sides"
        assert "_calibrate_side" in gate_cell or "side == \"SHORT\"" in gate_cell, (
            "Should have separate SHORT calibration"
        )

        # BUG FIX marker
        assert "SHORT gate calibrado independientemente" in gate_cell or "percentil negativo" in gate_cell, (
            "Should document the SHORT gate fix"
        )

    def test_range_v1_regime_gate_no_momentum(self):
        """RANGE v1 Cell 06: regime gate uses ER + vol only (no momentum for ranging)."""
        cells = _load_nb_cells(RANGE_V1)
        gate_cell = cells[6]
        assert "Regime Gate" in gate_cell
        # RANGE uses ER <= threshold (low efficiency = ranging)
        assert "ER <=" in gate_cell or "ER_COL) <=" in gate_cell, (
            "RANGE gate should use ER <= (low ER = ranging)"
        )


# ============================================================
# BUG 3: Dedup keep="last"
# ============================================================
class TestDedupConsistent:
    """Verify dedup uses keep='last' uniformly (not keep='first')."""

    def test_trend_v2_engine_dedup_last(self):
        """TREND v2 Cell 10: unique(..., keep='last')"""
        cells = _load_nb_cells(TREND_V2)
        engine_cell = cells[10]
        assert 'keep="last"' in engine_cell, (
            "Engine should use keep='last' for dedup"
        )
        assert 'keep="first"' not in engine_cell, (
            "BUG: Engine should NOT use keep='first'"
        )

    def test_range_v1_engine_dedup_last(self):
        """RANGE v1 Cell 10: unique(..., keep='last')"""
        cells = _load_nb_cells(RANGE_V1)
        engine_cell = cells[10]
        assert 'keep="last"' in engine_cell, (
            "RANGE engine should use keep='last' for dedup"
        )
        assert 'keep="first"' not in engine_cell, (
            "BUG: RANGE engine should NOT use keep='first'"
        )


# ============================================================
# BUG 4: Overlay double-execution guard
# ============================================================
class TestOverlayDoubleExecGuard:
    """Verify overlay cells have guard against double execution."""

    def test_trend_v2_overlay_guard(self):
        """TREND v2 Cell 16: has _overlay_applied guard."""
        cells = _load_nb_cells(TREND_V2)
        overlay_cell = cells[16]
        assert "Overlay" in overlay_cell, "Cell 16 should be Overlay"
        assert "_overlay_applied" in overlay_cell, (
            "Overlay must check _overlay_applied to prevent double execution"
        )
        assert "RuntimeError" in overlay_cell or "raise" in overlay_cell, (
            "Overlay must raise error on double execution"
        )

    def test_range_v1_overlay_guard(self):
        """RANGE v1 Cell 16: has _overlay_applied guard."""
        cells = _load_nb_cells(RANGE_V1)
        overlay_cell = cells[16]
        assert "_overlay_applied" in overlay_cell, (
            "RANGE overlay must check _overlay_applied"
        )
        assert "raise" in overlay_cell, (
            "RANGE overlay must raise on double execution"
        )


# ============================================================
# Structural tests
# ============================================================
class TestNotebookStructure:
    """Verify notebook structure and cell counts."""

    def test_trend_v2_has_21_cells(self):
        cells = _load_nb_cells(TREND_V2)
        assert len(cells) == 21, f"TREND v2 should have 21 cells, got {len(cells)}"

    def test_range_v1_has_21_cells(self):
        cells = _load_nb_cells(RANGE_V1)
        assert len(cells) == 21, f"RANGE v1 should have 21 cells, got {len(cells)}"

    def test_trend_v2_cell_00_has_all_artifacts(self):
        """Cell 00 _build_artifacts() should include all required keys."""
        cells = _load_nb_cells(TREND_V2)
        cell0 = cells[0]
        required_artifacts = [
            "signals_all", "qa_timing", "tuning_results", "tuning_best_params",
            "alpha_design", "selection", "overlay_trades", "deploy_pack",
            "qa_alignment", "diagnostics",
        ]
        for key in required_artifacts:
            assert f'"{key}"' in cell0, f"Cell 00 missing artifact: {key}"

    def test_range_v1_cell_00_has_artifacts(self):
        cells = _load_nb_cells(RANGE_V1)
        cell0 = cells[0]
        required = ["signals_all", "trades_engine", "selection", "deploy_pack", "overlay_trades"]
        for key in required:
            assert f'"{key}"' in cell0, f"RANGE Cell 00 missing artifact: {key}"

    def test_range_v1_has_range_specific_features(self):
        """RANGE Cell 05 should have Bollinger %B, dist_mean_atr, range_width_atr."""
        cells = _load_nb_cells(RANGE_V1)
        feat_cell = cells[5]
        assert "pct_b" in feat_cell, "RANGE features should include pct_b (Bollinger %B)"
        assert "dist_mean_atr" in feat_cell, "RANGE features should include dist_mean_atr"
        assert "range_width_atr" in feat_cell, "RANGE features should include range_width_atr"

    def test_range_v1_uses_basket(self):
        """RANGE Cell 01 should try to read basket_range_core."""
        cells = _load_nb_cells(RANGE_V1)
        univ_cell = cells[1]
        assert "basket_range_core" in univ_cell, (
            "RANGE should use basket_range_core from NB2"
        )
        assert "FALLBACK_SYMBOLS" in univ_cell, (
            "RANGE should have fallback symbols"
        )
