"""
generate_compilation_report.py
==============================
Genera un informe unificado de compilacion de los 4 notebooks del pipeline
MT5 Quant, leyendo todos los artefactos JSON/parquet de los ultimos runs.

Salida:
  reports/COMPILATION_REPORT.md   — informe markdown unico
  reports/compilation_report.zip  — ZIP con el .md

Uso:
  venv1\Scripts\python.exe tools\generate_compilation_report.py
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS      = PROJECT_ROOT / "outputs"
METADATA     = PROJECT_ROOT / "data" / "metadata"
REPORTS_DIR  = PROJECT_ROOT / "reports"

NB_PATHS = {
    "NB1": PROJECT_ROOT / "01_DATA_EXTRACTION" / "notebooks" / "01_MT5_DE_5M_V1.ipynb",
    "NB2": PROJECT_ROOT / "02_ER_FILTER"       / "notebooks" / "02_ER_FILTER_5M_V4.ipynb",
    "NB3": PROJECT_ROOT / "03_STRATEGY_LAB"    / "notebooks" / "03_TREND_M5_Strategy_v2.ipynb",
    "NB4": PROJECT_ROOT / "03_STRATEGY_LAB"    / "notebooks" / "04_RANGE_M5_Strategy_v1.ipynb",
}

NB_TITLES = {
    "NB1": "MT5 Data Extraction",
    "NB2": "ER Filter 5M",
    "NB3": "TREND M5 Strategy v2",
    "NB4": "RANGE M5 Strategy v1",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_json(path: Path) -> dict | list | None:
    """Lee un JSON, retorna None si no existe."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latest_run_dir(strategy_dir: Path) -> Path | None:
    """Retorna el directorio del ultimo run leyendo _latest_run.txt."""
    latest_file = strategy_dir / "_latest_run.txt"
    if latest_file.exists():
        run_id = latest_file.read_text(encoding="utf-8").strip()
        d = strategy_dir / f"run_{run_id}"
        if d.is_dir():
            return d
    # fallback: buscar el directorio run_* mas reciente
    runs = sorted(strategy_dir.glob("run_*"), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


def _latest_er_dir() -> Path | None:
    """Retorna el directorio del ultimo run de ER filter (sin prefijo run_)."""
    er_root = OUTPUTS / "er_filter_5m"
    if not er_root.is_dir():
        return None
    runs = sorted(
        [d for d in er_root.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda p: p.name,
        reverse=True,
    )
    return runs[0] if runs else None


def _extract_nb_cells(nb_path: Path) -> list[dict]:
    """Extrae info de cada celda de un notebook."""
    if not nb_path.exists():
        return []
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        ct = cell.get("cell_type", "unknown")
        src = "".join(cell.get("source", []))
        lines = src.split("\n")
        # titulo: buscar linea con "Celda" o primera linea significativa
        title = ""
        for line in lines[:10]:
            s = line.strip().lstrip("#").strip()
            if s and not s.startswith("===") and not s.startswith("---"):
                title = s[:120]
                break
        if not title:
            for line in lines[:15]:
                if "Celda" in line or "Cell" in line:
                    title = line.strip().lstrip("#").strip()[:120]
                    break
        if not title:
            title = lines[0][:120] if lines else "(vacia)"

        has_error = any(
            o.get("output_type") == "error"
            for o in cell.get("outputs", [])
        )
        has_output = bool(cell.get("outputs"))
        status = "ERROR" if has_error else ("OK" if has_output else "NO_OUTPUT")
        n_lines = len(lines)
        cells.append({
            "idx": i, "type": ct, "title": title,
            "status": status, "n_lines": n_lines,
        })
    return cells


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2f}%"


def _fmt_num(val: float | int | None, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, int) or (isinstance(val, float) and val == int(val)):
        return f"{int(val):,}"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------
def _section_header(report: list[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report.append(f"# Informe de Compilacion — MT5 Quant Pipeline\n")
    report.append(f"**Generado**: {now}\n")
    report.append(f"**Proyecto**: `{PROJECT_ROOT}`\n")
    report.append("")


def _section_executive_summary(report: list[str], data: dict) -> None:
    report.append("## 1. Resumen Ejecutivo\n")

    # Pipeline overview
    report.append("### Pipeline Overview\n")
    report.append("```")
    report.append("NB1 (Data Extraction)  -->  NB2 (ER Filter)  -->  NB3 (TREND Strategy)")
    report.append("                                              -->  NB4 (RANGE Strategy)")
    report.append("```\n")

    # Tabla resumen
    report.append("### Resultados Globales\n")
    report.append("| NB | Nombre | Celdas | OK | FAIL | Symbols | Trades | Sharpe | MDD | Return |")
    report.append("|:---|:-------|-------:|---:|-----:|--------:|-------:|-------:|----:|-------:|")

    for nb_key in ["NB1", "NB2", "NB3", "NB4"]:
        nb = data.get(nb_key, {})
        cells = nb.get("cells", [])
        code_cells = [c for c in cells if c["type"] == "code"]
        n_total = len(code_cells)
        n_ok = sum(1 for c in code_cells if c["status"] == "OK")
        n_fail = n_total - n_ok

        symbols = nb.get("n_symbols", "")
        trades = nb.get("n_trades", "")
        sharpe = nb.get("sharpe", "")
        mdd = nb.get("mdd", "")
        total_ret = nb.get("total_return", "")

        symbols_s = str(symbols) if symbols != "" else "-"
        trades_s = _fmt_num(trades) if trades != "" else "-"
        sharpe_s = _fmt_num(sharpe, 3) if sharpe != "" else "-"
        mdd_s = _fmt_pct(mdd) if mdd != "" else "-"
        ret_s = _fmt_pct(total_ret) if total_ret != "" else "-"

        report.append(
            f"| {nb_key} | {NB_TITLES[nb_key]} | {n_total} | {n_ok} | {n_fail} "
            f"| {symbols_s} | {trades_s} | {sharpe_s} | {mdd_s} | {ret_s} |"
        )
    report.append("")


def _section_architecture(report: list[str], data: dict) -> None:
    report.append("## 2. Arquitectura y Flujo de Datos\n")

    report.append("### 2.1 Pipeline Secuencial\n")
    report.append("1. **NB1 — Data Extraction**: Descarga datos M5 de MetaTrader5, QA, "
                  "genera capa GOLD (`m5_clean/`), universe snapshot, cost filter.\n")
    report.append("2. **NB2 — ER Filter**: Calcula Efficiency Ratio (Kaufman), Price Density, "
                  "detecta regimenes TREND/RANGE/NOISE, scoring, estabilidad, baskets decorrelacionadas.\n")
    report.append("3. **NB3 — TREND Strategy**: Backtesting WFO de estrategia trend-following "
                  "con motor de ejecucion, tuning, overlay, seleccion institucional, deploy pack.\n")
    report.append("4. **NB4 — RANGE Strategy**: Backtesting WFO de estrategia mean-reversion "
                  "con el mismo framework que NB3, adaptado a regimen RANGE.\n")

    report.append("### 2.2 Path Contract\n")
    report.append("- `path_contract.py` = single source of truth para resolucion de rutas")
    report.append("- Layout: Hive partitioning (`symbol=XXX/year=YYYY/month=MM/*.parquet`)")
    report.append("- Datos NUNCA en Git (`.gitignore` bloquea `*.parquet`, `*.csv`, `**/data/`, `**/outputs/`)")
    report.append("")

    # Runtime info
    manifest = data.get("metadata_manifest")
    if manifest:
        report.append("### 2.3 Entorno de Ejecucion\n")
        versions = manifest.get("versions", {})
        report.append(f"- **Python**: {manifest.get('python', '?')}")
        report.append(f"- **Polars**: {versions.get('polars', '?')}")
        report.append(f"- **PyArrow**: {versions.get('pyarrow', '?')}")
        report.append(f"- **MetaTrader5**: {versions.get('MetaTrader5', '?')}")
        report.append(f"- **Plataforma**: {manifest.get('platform', '?')}")
        report.append(f"- **TZ local**: {manifest.get('tz_local', '?')}")
        report.append("")


def _section_nb1(report: list[str], data: dict) -> None:
    report.append("## 3. NB1 — MT5 Data Extraction (23 celdas)\n")

    report.append("### 3.1 Proposito\n")
    report.append("Descarga masiva de datos M5 (4+ anos) desde MetaTrader5, "
                  "con QA integral, filtro de costes 3B, ingesta incremental, "
                  "backups y capa GOLD para consumo downstream.\n")

    # Cell-by-cell
    nb = data.get("NB1", {})
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c["type"] == "code"]
    n_ok = sum(1 for c in code_cells if c["status"] == "OK")
    report.append(f"### 3.2 Compilacion: {n_ok}/{len(code_cells)} celdas OK\n")
    report.append("| # | Status | Descripcion |")
    report.append("|--:|:------:|:------------|")
    for c in code_cells:
        st = c["status"]
        icon = "OK" if st == "OK" else "FAIL" if st == "ERROR" else "-"
        report.append(f"| {c['idx']} | {icon} | {c['title'][:100]} |")
    report.append("")

    # Metricas desde metadata
    manifest = data.get("metadata_manifest")
    qa_ready = data.get("qa_trading_ready")

    if manifest:
        counts = manifest.get("counts", {})
        report.append("### 3.3 Metricas\n")
        report.append(f"- **Symbols en GOLD**: {counts.get('symbols_source', '?')}")
        report.append(f"- **Archivos M5 clean**: {counts.get('files_source', '?'):,}")
        report.append(f"- **Tamano M5 clean**: {counts.get('bytes_source', 0) / 1e9:.2f} GiB")
        report.append("")

    if qa_ready:
        gold = qa_ready.get("gold", {})
        univ = qa_ready.get("universe_summary", {})
        costs = qa_ready.get("costs_summary", {})
        qa_op = qa_ready.get("qa_operativa_summary", {})
        gate = qa_ready.get("gate_result", {})

        report.append("### 3.4 QA Trading Ready\n")
        report.append(f"- **Status general**: `{qa_ready.get('status', '?')}`")
        report.append(f"- **Gold symbols**: {gold.get('symbols', '?')} ({gold.get('bytes_human', '?')})")
        report.append(f"- **Eligible 3B**: {univ.get('eligible_3b_symbols', '?')}")
        report.append(f"- **Interseccion Gold vs 3B**: {univ.get('intersection_base_vs_gold', '?')}")

        cost_dist = costs.get("cost_flag_distribution", {})
        report.append(f"- **Costes**: OK={cost_dist.get('OK', 0)}, "
                      f"CARO={cost_dist.get('CARO', 0)}, "
                      f"PROHIBITIVO={cost_dist.get('PROHIBITIVO', 0)} "
                      f"(ratio OK: {costs.get('cost_flag_ok_ratio_pct', 0):.1f}%)")

        qa_dist = qa_op.get("qa_operativa_flag_distribution", {})
        report.append(f"- **QA operativa**: OK={qa_dist.get('OK', 0)}, "
                      f"WARN={qa_dist.get('WARN', 0)} "
                      f"(ratio OK: {qa_op.get('qa_operativa_flag_ok_ratio_pct', 0):.1f}%)")

        report.append(f"- **Gate computed**: `{gate.get('computed_status', '?')}`")

        issues = qa_ready.get("issues", [])
        if issues:
            report.append("\n**Issues detectados:**")
            for iss in issues:
                report.append(f"- {iss}")
        report.append("")


def _section_nb2(report: list[str], data: dict) -> None:
    report.append("## 4. NB2 — ER Filter 5M (28 celdas)\n")

    report.append("### 4.1 Proposito\n")
    report.append("Detecta regimenes de mercado (TREND/RANGE/NOISE) usando Efficiency Ratio (Kaufman) "
                  "y Price Density. Calcula viabilidad economica, estabilidad estadistica, "
                  "scoring compuesto y genera baskets decorrelacionadas para NB3/NB4.\n")

    # Cell-by-cell
    nb = data.get("NB2", {})
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c["type"] == "code"]
    n_ok = sum(1 for c in code_cells if c["status"] == "OK")
    report.append(f"### 4.2 Compilacion: {n_ok}/{len(code_cells)} celdas OK\n")
    report.append("| # | Status | Descripcion |")
    report.append("|--:|:------:|:------------|")
    for c in code_cells:
        st = c["status"]
        icon = "OK" if st == "OK" else "FAIL" if st == "ERROR" else "-"
        report.append(f"| {c['idx']} | {icon} | {c['title'][:100]} |")
    report.append("")

    # Baskets
    er_dir = data.get("er_dir")
    if er_dir:
        report.append("### 4.3 Baskets Generadas\n")

        trend_syms_file = er_dir / "baskets" / "basket_trend_core_symbols.txt"
        range_syms_file = er_dir / "baskets" / "basket_range_core_symbols.txt"

        if trend_syms_file.exists():
            trend_syms = trend_syms_file.read_text(encoding="utf-8").strip().split("\n")
            report.append(f"- **TREND core**: {', '.join(trend_syms)} ({len(trend_syms)} symbols)")
        if range_syms_file.exists():
            range_syms = range_syms_file.read_text(encoding="utf-8").strip().split("\n")
            report.append(f"- **RANGE core**: {', '.join(range_syms)} ({len(range_syms)} symbols)")
        report.append("")

    # Config ER
    er_config = data.get("er_config")
    if er_config:
        report.append("### 4.4 Configuracion ER\n")
        er_windows = er_config.get("er_pd_windows", {})
        if er_windows:
            report.append(f"- **ER windows**: {er_windows}")

        baskets_cfg = er_config.get("baskets", {})
        if baskets_cfg:
            report.append(f"- **Baskets config**: max_symbols={baskets_cfg.get('max_symbols_per_basket', '?')}, "
                          f"corr_threshold={baskets_cfg.get('corr_threshold', '?')}")

        scoring = er_config.get("scoring_weights", {})
        if scoring:
            report.append(f"- **Scoring weights**: {json.dumps(scoring, indent=None)}")
        report.append("")


def _section_strategy(
    report: list[str],
    data: dict,
    nb_key: str,
    section_num: int,
    strategy_name: str,
    n_cells_expected: int,
) -> None:
    report.append(f"## {section_num}. {nb_key} — {strategy_name} ({n_cells_expected} celdas)\n")

    # --- Manifest ---
    manifest = data.get(f"{nb_key}_manifest")
    nb_data = data.get(nb_key, {})
    cells = nb_data.get("cells", [])
    code_cells = [c for c in cells if c["type"] == "code"]
    n_ok = sum(1 for c in code_cells if c["status"] == "OK")

    if manifest:
        report.append(f"- **Run ID**: `{manifest.get('run_id', '?')}`")
        report.append(f"- **Schema version**: {manifest.get('schema_version', '?')}")
        report.append(f"- **Created**: {manifest.get('created_utc', '?')}")
        report.append(f"- **Completed**: {manifest.get('completion_utc', '?')}")
        summary = manifest.get("summary", {})
        report.append(f"- **Artifacts OK**: {summary.get('artifacts_existing', summary.get('artifacts_ok', '?'))}")
        missing = summary.get("artifacts_missing_keys", [])
        if missing:
            report.append(f"- **Artifacts missing**: {', '.join(missing)}")
        report.append("")

    # --- Compilacion ---
    report.append(f"### {section_num}.1 Compilacion: {n_ok}/{len(code_cells)} celdas OK\n")
    report.append("| # | Status | Descripcion |")
    report.append("|--:|:------:|:------------|")
    for c in code_cells:
        st = c["status"]
        icon = "OK" if st == "OK" else "FAIL" if st == "ERROR" else "-"
        report.append(f"| {c['idx']} | {icon} | {c['title'][:100]} |")
    report.append("")

    # --- Data QA ---
    data_qa = data.get(f"{nb_key}_data_qa")
    if data_qa:
        report.append(f"### {section_num}.2 Data QA\n")
        report.append(f"- **Rows totales**: {_fmt_num(data_qa.get('n_rows_total', data_qa.get('n_rows', 0)))}")
        report.append(f"- **Symbols cargados**: {data_qa.get('n_symbols', '?')}")
        per_sym = data_qa.get("per_symbol", [])
        if per_sym:
            report.append("\n| Symbol | Rows | Start | End | Coverage % |")
            report.append("|:-------|-----:|:------|:----|----------:|")
            for s in per_sym:
                report.append(
                    f"| {s.get('symbol', '?')} | {_fmt_num(s.get('rows', 0))} "
                    f"| {s.get('start_utc', '?')[:10]} | {s.get('end_utc', '?')[:10]} "
                    f"| {_fmt_pct(s.get('coverage_intraday_pct'))} |"
                )
        report.append("")

    # --- WFO Folds ---
    wfo = data.get(f"{nb_key}_wfo")
    if wfo:
        report.append(f"### {section_num}.3 WFO Folds\n")
        report.append(f"- **Modo**: {wfo.get('wfo_mode', '?')}")
        per_sym = wfo.get("per_symbol", [])
        folds_summary = wfo.get("folds_summary", [])

        if per_sym:
            for s in per_sym:
                report.append(
                    f"- **{s['symbol']}** ({s.get('asset_class', '?')}): "
                    f"{s.get('n_folds', '?')} folds, "
                    f"IS={s.get('config', {}).get('is_months', '?')}m / "
                    f"OOS={s.get('config', {}).get('oos_months', '?')}m, "
                    f"embargo={s.get('embargo_days', '?')}d, "
                    f"{_fmt_num(s.get('n_rows', 0))} rows"
                )
        elif folds_summary:
            report.append(f"- **IS months**: {wfo.get('IS_months', '?')}")
            report.append(f"- **OOS months**: {wfo.get('OOS_months', '?')}")
            report.append(f"- **Embargo days**: {wfo.get('embargo_days', '?')}")
            report.append(f"- **N folds**: {wfo.get('n_folds', '?')}")
            if folds_summary:
                report.append(f"- **Fold 1 IS start**: {folds_summary[0].get('IS_start', '?')[:10]}")
                report.append(f"- **Fold {len(folds_summary)} OOS end**: "
                              f"{folds_summary[-1].get('OOS_end', '?')[:10]}")
        report.append("")

    # --- Engine KPIs ---
    engine = data.get(f"{nb_key}_engine")
    if engine:
        kpis = engine.get("kpis", {})
        exits = engine.get("exit_reasons", {})
        report.append(f"### {section_num}.4 Engine KPIs\n")
        report.append("| Metrica | Valor |")
        report.append("|:--------|------:|")
        report.append(f"| Total Return | {_fmt_pct(kpis.get('total_return'))} |")
        report.append(f"| Max Drawdown | {_fmt_pct(kpis.get('mdd'))} |")
        report.append(f"| Sharpe-like | {_fmt_num(kpis.get('sharpe_like'), 4)} |")
        report.append(f"| Win Rate | {_fmt_pct(kpis.get('win_rate', 0) * 100 if kpis.get('win_rate') else None)} |")
        report.append(f"| N Trades | {_fmt_num(kpis.get('n_trades'))} |")
        report.append(f"| Mean Return | {_fmt_num(kpis.get('mean_ret'), 6)} |")
        report.append("")

        if exits:
            report.append("**Exit Reasons:**\n")
            report.append("| Reason | Count | % |")
            report.append("|:-------|------:|--:|")
            total_exits = sum(exits.values())
            for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
                pct = count / total_exits * 100 if total_exits else 0
                report.append(f"| {reason} | {_fmt_num(count)} | {pct:.1f}% |")
            report.append("")

    # --- Selection ---
    selection = data.get(f"{nb_key}_selection")
    if selection:
        sels = selection.get("selections", [])
        gates = selection.get("gates", {})
        report.append(f"### {section_num}.5 Seleccion Institucional\n")
        if gates:
            report.append("**Gates:**\n")
            for k, v in gates.items():
                report.append(f"- `{k}`: {v}")
            report.append("")

        n_go = sum(1 for s in sels if s.get("decision") == "GO")
        n_nogo = sum(1 for s in sels if s.get("decision") == "NO_GO")
        report.append(f"**Resultado**: {n_go} GO, {n_nogo} NO_GO\n")

        if sels:
            report.append("| Symbol | Side | Decision | N OOS | Score | Razon |")
            report.append("|:-------|:-----|:---------|------:|------:|:------|")
            for s in sels:
                report.append(
                    f"| {s.get('symbol', '?')} | {s.get('side', '?')} "
                    f"| {s.get('decision', '?')} | {s.get('n_oos', '?')} "
                    f"| {s.get('score', '?')} | {s.get('reason', '-')} |"
                )
        report.append("")


def _section_comparison(report: list[str], data: dict) -> None:
    report.append("## 7. Comparativa TREND vs RANGE\n")

    t_eng = data.get("NB3_engine", {}).get("kpis", {})
    r_eng = data.get("NB4_engine", {}).get("kpis", {})

    if not t_eng and not r_eng:
        report.append("_No hay datos de engine para comparar._\n")
        return

    report.append("| Metrica | TREND v2 | RANGE v1 |")
    report.append("|:--------|:---------|:---------|")

    metrics = [
        ("Total Return", "total_return", "%", 2),
        ("Max Drawdown", "mdd", "%", 2),
        ("Sharpe-like", "sharpe_like", "", 4),
        ("Win Rate", "win_rate", "x100%", 2),
        ("N Trades", "n_trades", "int", 0),
        ("Mean Return", "mean_ret", "", 6),
    ]

    for label, key, fmt, dec in metrics:
        tv = t_eng.get(key)
        rv = r_eng.get(key)
        if fmt == "%":
            ts = _fmt_pct(tv)
            rs = _fmt_pct(rv)
        elif fmt == "x100%":
            ts = _fmt_pct(tv * 100 if tv else None)
            rs = _fmt_pct(rv * 100 if rv else None)
        elif fmt == "int":
            ts = _fmt_num(tv)
            rs = _fmt_num(rv)
        else:
            ts = _fmt_num(tv, dec)
            rs = _fmt_num(rv, dec)
        report.append(f"| {label} | {ts} | {rs} |")

    # Exit reasons side by side
    t_exits = data.get("NB3_engine", {}).get("exit_reasons", {})
    r_exits = data.get("NB4_engine", {}).get("exit_reasons", {})
    all_reasons = sorted(set(list(t_exits.keys()) + list(r_exits.keys())))
    if all_reasons:
        report.append("")
        report.append("**Exit Reasons Comparison:**\n")
        report.append("| Reason | TREND | RANGE |")
        report.append("|:-------|------:|------:|")
        for r in all_reasons:
            report.append(f"| {r} | {_fmt_num(t_exits.get(r, 0))} | {_fmt_num(r_exits.get(r, 0))} |")
    report.append("")


def _section_issues(report: list[str], data: dict) -> None:
    report.append("## 8. Issues Conocidos y Recomendaciones\n")

    issues = []

    # Check NB1/NB2 compilation
    for nb_key in ["NB1", "NB2", "NB3", "NB4"]:
        cells = data.get(nb_key, {}).get("cells", [])
        code_cells = [c for c in cells if c["type"] == "code"]
        fails = [c for c in code_cells if c["status"] != "OK"]
        if fails:
            for f in fails:
                issues.append(f"**{nb_key} celda {f['idx']}**: status `{f['status']}` — {f['title'][:80]}")

    # Check engine KPIs
    for nb_key, label in [("NB3", "TREND"), ("NB4", "RANGE")]:
        eng = data.get(f"{nb_key}_engine", {}).get("kpis", {})
        if eng:
            ret = eng.get("total_return")
            if ret is not None and ret < 0:
                issues.append(f"**{label}**: Return negativo ({_fmt_pct(ret)}). "
                              "Requiere revision de parametros, features o logica de entrada.")
            sharpe = eng.get("sharpe_like")
            if sharpe is not None and sharpe < 0:
                issues.append(f"**{label}**: Sharpe negativo ({_fmt_num(sharpe, 4)}). "
                              "La estrategia no supera costes en el periodo evaluado.")
            wr = eng.get("win_rate")
            if wr is not None and wr < 0.4:
                issues.append(f"**{label}**: Win rate bajo ({_fmt_pct(wr * 100)}). "
                              "Considerar ajustar SL/TP o filtros de regimen.")

    # Check selections
    for nb_key, label in [("NB3", "TREND"), ("NB4", "RANGE")]:
        sel = data.get(f"{nb_key}_selection", {}).get("selections", [])
        n_go = sum(1 for s in sel if s.get("decision") == "GO")
        if sel and n_go == 0:
            issues.append(f"**{label}**: 0 symbols aprobados (GO). "
                          "Ninguna config pasa los gates institucionales.")

    # QA trading ready
    qa_ready = data.get("qa_trading_ready")
    if qa_ready:
        for iss in qa_ready.get("issues", []):
            issues.append(f"**NB1 QA**: {iss}")

    if issues:
        for iss in issues:
            report.append(f"- {iss}")
    else:
        report.append("No se detectaron issues criticos.")

    report.append("")

    # Recomendaciones
    report.append("### Recomendaciones\n")
    report.append("1. **Ampliar universo**: Incluir mas symbols en baskets TREND/RANGE "
                  "para mayor diversificacion y mas trades OOS.")
    report.append("2. **Revisar features**: Evaluar si los features actuales capturan "
                  "suficiente edge (ER, momentum, volatilidad).")
    report.append("3. **Tuning de parametros**: Explorar grids mas amplios para SL/TP/TRAIL.")
    report.append("4. **Gate de seleccion**: Reducir `min_oos_trades` si el universo "
                  "es pequeno, o aumentar OOS window.")
    report.append("5. **Cost model**: Validar slippage y comisiones con datos reales de ejecucion.")
    report.append("")


def _section_annexes(report: list[str], data: dict) -> None:
    report.append("## 9. Anexos\n")

    report.append("### 9.1 Artefactos por Estrategia\n")

    for nb_key, label in [("NB3", "TREND v2"), ("NB4", "RANGE v1")]:
        manifest = data.get(f"{nb_key}_manifest")
        if manifest:
            artifacts = manifest.get("artifacts", {})
            report.append(f"**{label}** — {len(artifacts)} artefactos:\n")
            report.append("| Clave | Archivo |")
            report.append("|:------|:--------|")
            for k, v in sorted(artifacts.items()):
                fname = Path(v).name if v else "?"
                report.append(f"| {k} | `{fname}` |")
            report.append("")

    # Canonical schema
    manifest = data.get("NB3_manifest")
    if manifest:
        schema = manifest.get("canonical_schema", {})
        if schema:
            report.append("### 9.2 Schema Canonico\n")
            for table_name, info in schema.items():
                cols = info.get("required_columns", [])
                notes = info.get("notes", "")
                report.append(f"**{table_name}**: `{', '.join(cols)}`")
                if notes:
                    report.append(f"  - {notes}")
            report.append("")

    # Runtime
    manifest = data.get("NB3_manifest")
    if manifest:
        rt = manifest.get("runtime", {})
        if rt:
            report.append("### 9.3 Runtime\n")
            report.append(f"- Python: {rt.get('python', '?')}")
            report.append(f"- Platform: {rt.get('platform', '?')}")
            report.append(f"- Node: {rt.get('node', '?')}")
            report.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[INFO] Generando informe de compilacion...")

    data: dict = {}

    # --- 1. Extraer celdas de cada notebook ---
    for nb_key, nb_path in NB_PATHS.items():
        cells = _extract_nb_cells(nb_path)
        data[nb_key] = {"cells": cells}
        n_code = sum(1 for c in cells if c["type"] == "code")
        n_ok = sum(1 for c in cells if c["type"] == "code" and c["status"] == "OK")
        print(f"  {nb_key}: {n_ok}/{n_code} celdas OK")

    # --- 2. Leer metadata NB1 ---
    data["metadata_manifest"] = _safe_json(METADATA / "manifest.json")
    data["qa_trading_ready"] = _safe_json(METADATA / "qa_trading_ready_summary.json")

    # Metricas NB1
    if data["metadata_manifest"]:
        counts = data["metadata_manifest"].get("counts", {})
        data["NB1"]["n_symbols"] = counts.get("symbols_source", 0)

    # --- 3. Leer ER filter (NB2) ---
    er_dir = _latest_er_dir()
    data["er_dir"] = er_dir
    if er_dir:
        data["er_config"] = _safe_json(er_dir / "diagnostics" / "config.json")
        # count basket symbols
        trend_f = er_dir / "baskets" / "basket_trend_core_symbols.txt"
        range_f = er_dir / "baskets" / "basket_range_core_symbols.txt"
        trend_n = len(trend_f.read_text(encoding="utf-8").strip().split("\n")) if trend_f.exists() else 0
        range_n = len(range_f.read_text(encoding="utf-8").strip().split("\n")) if range_f.exists() else 0
        data["NB2"]["n_symbols"] = trend_n + range_n

    # --- 4. Leer TREND v2 (NB3) ---
    trend_dir = _latest_run_dir(OUTPUTS / "trend_v2")
    if trend_dir:
        data["NB3_manifest"]  = _safe_json(trend_dir / "run_manifest_v2.json")
        data["NB3_data_qa"]   = _safe_json(trend_dir / "data_qa_report_v2.json")
        data["NB3_engine"]    = _safe_json(trend_dir / "engine_report_snapshot_v2.json")
        data["NB3_wfo"]       = _safe_json(trend_dir / "wfo_folds_snapshot_v2.json")
        data["NB3_selection"] = _safe_json(trend_dir / "selection_snapshot_v2.json")

        if data["NB3_manifest"]:
            s = data["NB3_manifest"].get("summary", {})
            data["NB3"]["n_symbols"] = data.get("NB3_data_qa", {}).get("n_symbols", s.get("symbols_total", ""))
            data["NB3"]["n_trades"] = s.get("n_trades", data.get("NB3_engine", {}).get("kpis", {}).get("n_trades", ""))
            data["NB3"]["sharpe"] = s.get("best_sharpe", data.get("NB3_engine", {}).get("kpis", {}).get("sharpe_like", ""))
            data["NB3"]["mdd"] = s.get("worst_mdd", data.get("NB3_engine", {}).get("kpis", {}).get("mdd", ""))
            data["NB3"]["total_return"] = s.get("total_return", data.get("NB3_engine", {}).get("kpis", {}).get("total_return", ""))
        elif data.get("NB3_engine"):
            kpis = data["NB3_engine"].get("kpis", {})
            data["NB3"]["n_trades"] = kpis.get("n_trades", "")
            data["NB3"]["sharpe"] = kpis.get("sharpe_like", "")
            data["NB3"]["mdd"] = kpis.get("mdd", "")
            data["NB3"]["total_return"] = kpis.get("total_return", "")

    # --- 5. Leer RANGE v1 (NB4) ---
    range_dir = _latest_run_dir(OUTPUTS / "range_v1")
    if range_dir:
        data["NB4_manifest"]  = _safe_json(range_dir / "run_manifest_range_v1.json")
        data["NB4_data_qa"]   = _safe_json(range_dir / "data_qa_report_range_v1.json")
        data["NB4_engine"]    = _safe_json(range_dir / "engine_report_snapshot_range_v1.json")
        data["NB4_wfo"]       = _safe_json(range_dir / "wfo_folds_snapshot_range_v1.json")
        data["NB4_selection"] = _safe_json(range_dir / "selection_snapshot_range_v1.json")

        if data["NB4_manifest"]:
            s = data["NB4_manifest"].get("summary", {})
            data["NB4"]["n_symbols"] = data.get("NB4_data_qa", {}).get("n_symbols", "")
            data["NB4"]["n_trades"] = s.get("kpi_n_trades", data.get("NB4_engine", {}).get("kpis", {}).get("n_trades", ""))
            data["NB4"]["sharpe"] = s.get("kpi_sharpe_like", data.get("NB4_engine", {}).get("kpis", {}).get("sharpe_like", ""))
            data["NB4"]["mdd"] = s.get("kpi_mdd", data.get("NB4_engine", {}).get("kpis", {}).get("mdd", ""))
            data["NB4"]["total_return"] = s.get("kpi_total_return", data.get("NB4_engine", {}).get("kpis", {}).get("total_return", ""))
        elif data.get("NB4_engine"):
            kpis = data["NB4_engine"].get("kpis", {})
            data["NB4"]["n_trades"] = kpis.get("n_trades", "")
            data["NB4"]["sharpe"] = kpis.get("sharpe_like", "")
            data["NB4"]["mdd"] = kpis.get("mdd", "")
            data["NB4"]["total_return"] = kpis.get("total_return", "")

    # --- 6. Generar markdown ---
    report: list[str] = []

    _section_header(report)
    _section_executive_summary(report, data)
    _section_architecture(report, data)
    _section_nb1(report, data)
    _section_nb2(report, data)
    _section_strategy(report, data, "NB3", 5, "TREND M5 Strategy v2", 21)
    _section_strategy(report, data, "NB4", 6, "RANGE M5 Strategy v1", 21)
    _section_comparison(report, data)
    _section_issues(report, data)
    _section_annexes(report, data)

    # Footer
    report.append("---")
    report.append(f"*Generado automaticamente por `tools/generate_compilation_report.py`*")

    md_content = "\n".join(report)

    # --- 7. Escribir archivos ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS_DIR / "COMPILATION_REPORT.md"
    zip_path = REPORTS_DIR / "compilation_report.zip"

    md_path.write_text(md_content, encoding="utf-8")
    print(f"[OK] {md_path} ({len(md_content):,} chars)")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(md_path, "COMPILATION_REPORT.md")
    print(f"[OK] {zip_path} ({zip_path.stat().st_size:,} bytes)")

    print("\n[DONE] Informe generado exitosamente.")


if __name__ == "__main__":
    main()
