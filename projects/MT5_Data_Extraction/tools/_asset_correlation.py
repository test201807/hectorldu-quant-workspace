"""
Asset Correlation Analyzer — Correlacion de returns diarios entre candidatos.

Responde la pregunta: "¿Cuántos activos se pueden operar simultáneamente
sin exponerse a un drawdown sincronizado que viole el daily cap FTMO?"

Método:
  1. Carga datos M5 de NVDA/META/TSLA/AAPL/AIRF (5 candidatos MARGINAL)
  2. Calcula daily returns (close-to-close, calendario)
  3. Para AIRF SHORT: invierte el signo del return (posición SHORT)
  4. Calcula matriz de correlación pairwise (Pearson)
  5. Calcula max drawdown condicional conjunto (% días donde ≥N activos pierden)
  6. Recomienda max posiciones simultáneas basado en correlación y FTMO risk

Lógica FTMO:
  Si 2 activos altamente correlados (ρ > 0.6) abren simultáneamente y el
  mercado cae, ambos pueden perder el mismo día. Con daily cap de $1,250
  sobre $25k (5%), 2 posiciones perdedoras simultáneas son el doble de
  riesgo que 1. La recomendación considera:
    - Correlación media entre activos a operar
    - Multiplicador de riesgo conjunto

Uso:
  cd C:\\Quant\\projects\\MT5_Data_Extraction
  venv1\\Scripts\\python.exe tools\\_asset_correlation.py
  venv1\\Scripts\\python.exe tools\\_asset_correlation.py --period oos --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "shared" / "contracts"))
from path_contract import m5_clean_dir  # noqa: E402

# Candidatos MARGINAL del selector institucional
CANDIDATES: dict[str, str] = {
    "NVDA": "LONG",
    "META": "LONG",
    "TSLA": "LONG",
    "AAPL": "LONG",
    "AIRF": "SHORT",
}

# Periodos disponibles en datos M5
# El OOS aproximado es los últimos 3 meses del dataset
OOS_MONTHS = 3   # meses de OOS al final


# ── Carga de datos ───────────────────────────────────────────────────

def load_daily_returns(
    symbol: str,
    side: str,
    data_root: Path,
    period: str = "all",
) -> pl.DataFrame:
    """Carga M5, resamplea a diario y calcula returns close-to-close."""
    sym_dir = data_root / f"symbol={symbol}"
    parquets = list(sym_dir.rglob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No hay datos M5 para {symbol} en {sym_dir}")

    df = (
        pl.scan_parquet(parquets)
        .select(["timestamp_utc", "close"])
        .collect()
        .with_columns(
            pl.from_epoch(pl.col("timestamp_utc"), time_unit="ms")
            .alias("time_utc")
        )
        .with_columns(pl.col("time_utc").dt.date().alias("date"))
        .sort("time_utc")
    )

    # Daily close = último precio de cada día
    daily = (
        df.group_by("date")
        .agg(pl.col("close").last().alias("close"))
        .sort("date")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret")
        )
        .drop_nulls("ret")
    )

    # Filtro de periodo
    if period == "oos":
        # Últimos OOS_MONTHS meses
        max_date = daily["date"].max()
        cutoff = max_date - pl.duration(days=OOS_MONTHS * 30)
        daily = daily.filter(pl.col("date") > cutoff)
    elif period == "is":
        max_date = daily["date"].max()
        cutoff = max_date - pl.duration(days=OOS_MONTHS * 30)
        daily = daily.filter(pl.col("date") <= cutoff)

    # Winsoriza outliers extremos (splits, errores de datos, acciones corporativas)
    # Límite conservador: ±40% diario. Movimientos legítimos no superan esto.
    WINSOR_LIMIT = 0.40
    outliers = daily.filter(pl.col("ret").abs() > WINSOR_LIMIT)
    if outliers.height > 0:
        print(f"  WARN {symbol}: {outliers.height} dia(s) con |ret| > {WINSOR_LIMIT:.0%} "
              f"→ winsorizado a ±{WINSOR_LIMIT:.0%}")
        daily = daily.with_columns(
            pl.col("ret").clip(-WINSOR_LIMIT, WINSOR_LIMIT)
        )

    # Para SHORT: invertir el return (ganamos cuando el activo cae)
    if side == "SHORT":
        daily = daily.with_columns((-pl.col("ret")).alias("ret"))

    return daily.select(["date", "ret"]).rename({"ret": symbol})


# ── Correlación ──────────────────────────────────────────────────────

def build_return_matrix(
    data_root: Path,
    period: str = "all",
) -> pl.DataFrame:
    """Construye matriz de returns diarios alineados por fecha."""
    frames = []
    for sym, side in CANDIDATES.items():
        try:
            df = load_daily_returns(sym, side, data_root, period)
            frames.append(df)
        except FileNotFoundError as e:
            print(f"  WARN: {e}")

    if not frames:
        raise RuntimeError("No se pudieron cargar datos para ningún símbolo")

    # Join por fecha (inner join — solo fechas con todos los activos)
    result = frames[0]
    for df in frames[1:]:
        result = result.join(df, on="date", how="inner")

    return result.sort("date")


def pearson_corr_matrix(mat: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Calcula matriz de correlación Pearson."""
    symbols = [c for c in mat.columns if c != "date"]
    arr = mat.select(symbols).to_numpy()
    corr = np.corrcoef(arr.T)
    return corr, symbols


# ── Análisis conjunto ────────────────────────────────────────────────

def joint_loss_analysis(
    mat: pl.DataFrame,
    symbols: list[str],
) -> dict:
    """
    Calcula qué % de días pierde simultáneamente N activos.
    Útil para estimar riesgo de daily cap FTMO con varias posiciones.
    """
    arr = mat.select(symbols).to_numpy()
    n_days = arr.shape[0]

    # % días donde exactamente N activos son negativos
    neg_per_day = (arr < 0).sum(axis=1)
    joint_loss = {}
    for k in range(1, len(symbols) + 1):
        pct = float((neg_per_day >= k).mean())
        joint_loss[f"pct_days_neg_ge{k}"] = round(pct, 4)

    # Peor día: suma de losses simultáneos (como % de equity)
    daily_sum = arr.sum(axis=1)  # suma de returns de todos los activos
    joint_loss["worst_day_sum_ret"] = round(float(daily_sum.min()), 6)
    joint_loss["best_day_sum_ret"]  = round(float(daily_sum.max()), 6)
    joint_loss["n_days"]            = n_days

    return joint_loss


def _subgroup_avg_rho(corr: np.ndarray, symbols: list[str], group: list[str]) -> float:
    """Correlación media dentro de un subgrupo de símbolos."""
    idx = [symbols.index(s) for s in group if s in symbols]
    if len(idx) < 2:
        return 0.0
    vals = [corr[i, j] for i in idx for j in idx if i < j]
    return float(np.mean(vals)) if vals else 0.0


def recommend_max_positions(corr: np.ndarray, symbols: list[str]) -> dict:
    """
    Recomienda max posiciones simultáneas basado en correlación promedio.

    Lógica FTMO ($25k, daily cap $1,250 = 5%):
    - 1 posición: riesgo contenido
    - 2 posiciones con ρ < 0.40: diversificadas → OK
    - 2 posiciones con 0.40 <= ρ < 0.70: moderadas → monitorear
    - 2+ posiciones con ρ >= 0.70: altamente correladas → reducir a 1
    """
    n = len(symbols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "pair": f"{symbols[i]}/{symbols[j]}",
                "rho": round(float(corr[i, j]), 4),
            })

    pairs_sorted = sorted(pairs, key=lambda x: -x["rho"])
    avg_rho = float(np.mean([p["rho"] for p in pairs]))
    max_rho = float(max(p["rho"] for p in pairs)) if pairs else 0.0
    min_rho = float(min(p["rho"] for p in pairs)) if pairs else 0.0

    # Recomendación — umbral 0.60 según spec Tier 1:
    # "Si ρ > 0.6, operar máximo 1-2 simultáneamente"
    if max_rho >= 0.60:
        rec_max  = 1
        rec_text = ("Correlacion maxima >= 0.60 (umbral spec). MAX 1 posicion simultanea. "
                    "Riesgo alto de drawdown conjunto si el mercado cae.")
    elif avg_rho >= 0.40:
        rec_max  = 2
        rec_text = ("Correlacion media >= 0.40. MAX 2 posiciones simultaneas. "
                    "Priorizar el par con menor rho entre si.")
    elif avg_rho >= 0.20:
        rec_max  = 2
        rec_text = ("Correlacion media moderada (0.20-0.40). MAX 2 posiciones, "
                    "aceptable con monitoreo diario.")
    else:
        rec_max  = 3
        rec_text = ("Correlacion baja (<0.20). Hasta 3 posiciones simultaneas "
                    "manteniendo discipline FTMO.")

    # Advertencia especial para el grupo tech US (NVDA/META/TSLA/AAPL)
    tech_group = [s for s in ["NVDA", "META", "TSLA", "AAPL"] if s in symbols]
    tech_avg_rho = _subgroup_avg_rho(corr, symbols, tech_group)
    tech_warning = None
    if len(tech_group) >= 3 and tech_avg_rho >= 0.40:
        tech_warning = (
            f"ATENCION: {'/'.join(tech_group)} tienen ρ_media={tech_avg_rho:.2f} entre si. "
            f"Si las 3-4 posiciones simultaneas son tech US, limitar a MAX 2 "
            f"(riesgo de drawdown conjunto en dias de caida de Nasdaq)."
        )

    return {
        "avg_rho": round(avg_rho, 4),
        "max_rho": round(max_rho, 4),
        "min_rho": round(min_rho, 4),
        "tech_group_avg_rho": round(tech_avg_rho, 4),
        "tech_warning": tech_warning,
        "pairs": pairs_sorted,
        "recommended_max_positions": rec_max,
        "recommendation": rec_text,
    }


# ── Display ──────────────────────────────────────────────────────────

def print_report(
    corr: np.ndarray,
    symbols: list[str],
    rec: dict,
    joint: dict,
    period: str,
) -> None:
    n = len(symbols)
    sides = [CANDIDATES[s] for s in symbols]

    print(f"\n{'='*65}")
    print(f"  CORRELACION DE ACTIVOS — periodo: {period.upper()}")
    print(f"{'='*65}")
    print(f"  Activos: {', '.join(f'{s}({sides[i]})' for i, s in enumerate(symbols))}")
    print(f"  N dias: {joint['n_days']}")
    print()

    # Matriz de correlación
    print(f"  Matriz de correlacion Pearson (returns diarios):")
    print()
    header = f"  {'':>6}" + "".join(f"  {s:>6}" for s in symbols)
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for i, s1 in enumerate(symbols):
        row = f"  {s1:>6}"
        for j in range(n):
            val = corr[i, j]
            row += f"  {val:>+6.3f}"
        print(row)

    print()
    print(f"  Pares ordenados por correlacion:")
    for p in rec["pairs"]:
        bar = "█" * int(abs(p["rho"]) * 20)
        level = ("ALTA" if abs(p["rho"]) >= 0.70
                 else "MEDIA" if abs(p["rho"]) >= 0.40
                 else "BAJA")
        print(f"    {p['pair']:>12}  ρ={p['rho']:>+.3f}  {bar:<20} {level}")

    print()
    print(f"  Estadisticas:")
    print(f"    Correlacion media:  {rec['avg_rho']:>+.3f}")
    print(f"    Correlacion maxima: {rec['max_rho']:>+.3f}")
    print(f"    Correlacion minima: {rec['min_rho']:>+.3f}")

    print(f"\n  Analisis de losses conjuntos:")
    for k in range(1, n + 1):
        key = f"pct_days_neg_ge{k}"
        if key in joint:
            print(f"    >= {k} activos negativos el mismo dia: {joint[key]:.1%}")
    print(f"    Peor dia (suma returns): {joint['worst_day_sum_ret']:.2%}")

    print(f"\n{'─'*65}")
    print(f"  RECOMENDACION FTMO ($25k, daily cap $1,250):")
    print(f"  Max posiciones simultaneas: {rec['recommended_max_positions']}")
    print(f"  {rec['recommendation']}")
    if rec.get("tech_warning"):
        print(f"\n  *** {rec['tech_warning']}")
    print(f"{'='*65}")


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Asset Correlation Analyzer")
    parser.add_argument("--period", choices=["all", "oos", "is"], default="oos",
                        help="Periodo de datos a analizar (default: oos — spec Tier 1)")
    parser.add_argument("--save", action="store_true",
                        help="Guardar resultados en outputs/screening/")
    args = parser.parse_args()

    data_root = m5_clean_dir()
    print(f"Cargando datos M5 desde: {data_root}")
    print(f"Candidatos: {list(CANDIDATES.keys())} | periodo: {args.period}")

    mat = build_return_matrix(data_root, period=args.period)
    symbols = [c for c in mat.columns if c != "date"]

    # Fallback: si hay muy pocas fechas comunes (típico cuando AIRF tiene
    # datos escasos al final del dataset), usar periodo completo.
    MIN_DAYS_REQUIRED = 60
    if mat.height < MIN_DAYS_REQUIRED and args.period != "all":
        print(f"  WARN: solo {mat.height} dias comunes en periodo '{args.period}' "
              f"(minimo {MIN_DAYS_REQUIRED}). Usando 'all' para resultados fiables.")
        mat = build_return_matrix(data_root, period="all")
        symbols = [c for c in mat.columns if c != "date"]

    print(f"Fechas comunes: {mat.height} dias | "
          f"rango: {mat['date'].min()} — {mat['date'].max()}")

    corr, syms = pearson_corr_matrix(mat)
    joint = joint_loss_analysis(mat, syms)
    rec = recommend_max_positions(corr, syms)

    print_report(corr, syms, rec, joint, args.period)

    if args.save:
        out_dir = PROJECT / "outputs" / "screening"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "period": args.period,
            "symbols": syms,
            "sides": {s: CANDIDATES[s] for s in syms},
            "n_days": joint["n_days"],
            "correlation_matrix": {
                syms[i]: {syms[j]: round(float(corr[i, j]), 6)
                           for j in range(len(syms))}
                for i in range(len(syms))
            },
            "pairs": rec["pairs"],
            "summary": {
                "avg_rho": rec["avg_rho"],
                "max_rho": rec["max_rho"],
                "min_rho": rec["min_rho"],
                "recommended_max_positions": rec["recommended_max_positions"],
                "recommendation": rec["recommendation"],
            },
            "joint_loss": joint,
        }
        out_path = out_dir / "asset_correlation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
