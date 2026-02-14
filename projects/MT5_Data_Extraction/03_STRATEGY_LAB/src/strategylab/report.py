"""Report generation: parquet/csv export, markdown summary, JSON snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .metrics import compute_kpis


def export_trades(trades_df: pl.DataFrame, out_dir: Path, prefix: str = "trades") -> Path:
    """Export trades DataFrame to parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}.parquet"
    trades_df.write_parquet(str(path))
    return path


def export_csv(trades_df: pl.DataFrame, out_dir: Path, prefix: str = "trades") -> Path:
    """Export trades DataFrame to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}.csv"
    trades_df.write_csv(str(path))
    return path


def snapshot_json(data: dict[str, Any], out_dir: Path, name: str = "snapshot") -> Path:
    """Write a JSON snapshot file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    # Convert non-serialisable types
    cleaned = _clean_for_json(data)
    path.write_text(json.dumps(cleaned, indent=2, default=str), encoding="utf-8")
    return path


def _clean_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        return obj
    return obj


def markdown_report(
    kpis: dict[str, Any],
    symbol: str = "ALL",
    strategy: str = "unknown",
    segment: str = "OOS",
) -> str:
    """Generate a markdown summary string."""
    lines = [
        f"# StrategyLab Report — {strategy}",
        f"**Symbol**: {symbol} | **Segment**: {segment}",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## KPIs",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    fmt = {
        "n_trades": "{:.0f}",
        "total_return": "{:.4f}",
        "cagr": "{:.4f}",
        "sharpe": "{:.3f}",
        "sortino": "{:.3f}",
        "calmar": "{:.3f}",
        "mdd": "{:.4f}",
        "hit_rate": "{:.3f}",
        "expectancy": "{:.6f}",
        "profit_factor": "{:.3f}",
        "turnover": "{:.6f}",
        "exposure": "{:.3f}",
    }

    for key, template in fmt.items():
        val = kpis.get(key, 0)
        lines.append(f"| {key} | {template.format(val)} |")

    lines.append("")
    return "\n".join(lines)


def generate_full_report(
    trades_df: pl.DataFrame,
    out_dir: Path,
    symbol: str = "ALL",
    strategy: str = "TREND_v2",
    prefix: str = "report",
) -> dict[str, Path]:
    """Generate all report artifacts: parquet, csv, json, markdown.

    Returns dict of artifact names to file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # Export trades
    paths["trades_parquet"] = export_trades(trades_df, out_dir, f"{prefix}_trades")
    paths["trades_csv"] = export_csv(trades_df, out_dir, f"{prefix}_trades")

    # Compute KPIs
    kpis = compute_kpis(trades_df) if not trades_df.is_empty() else {}

    # JSON snapshot
    snap = {
        "symbol": symbol,
        "strategy": strategy,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kpis": kpis,
        "n_trades": trades_df.height,
    }
    paths["snapshot"] = snapshot_json(snap, out_dir, f"{prefix}_snapshot")

    # Markdown
    md = markdown_report(kpis, symbol=symbol, strategy=strategy)
    md_path = out_dir / f"{prefix}_summary.md"
    md_path.write_text(md, encoding="utf-8")
    paths["markdown"] = md_path

    return paths
