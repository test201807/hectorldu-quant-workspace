# Notebook 01_MT5_DE_5M_V1 — Registro Cerrado de Rutas y Artefactos (CLOSED)

Fecha de cierre: 2025-12-23  
Estado: **CERRADO / Single Source of Truth**  
Aplica a: `01_MT5_DE_5M_V1.ipynb`

Este registro consolida y “cierra” **todas las rutas (paths) y artefactos** relevantes del notebook, eliminando ambigüedades por aliases (p. ej. `DATA_ROOT`, `MANIFEST_PATH`). Todas las rutas listadas abajo son **relativas a `DATA_ROOT`**.

---

## 1) Regla única de raíz (`DATA_ROOT`)

1. **Regla base:** el notebook trabaja bajo un único `DATA_ROOT`.
2. **Default:** `DATA_ROOT = <PROJECT_ROOT>/data`
3. **Override soportado (recomendado):**
   - `MT5_DE_DATA_ROOT` (si existe, reemplaza `DATA_ROOT`)
4. **Alias histórico (evitar divergencia):**
   - `M5_DATA_ROOT` (si se usa, mantenerlo igual a `MT5_DE_DATA_ROOT`)

Convención operativa: antes de correr el notebook, definir **solo un** root efectivo y mantenerlo estable entre notebooks encadenados.

---

## 2) Árbol canónico de carpetas bajo `DATA_ROOT`

```text
DATA_ROOT/
  bulk_data/
    m5_raw/
    ticks_recent/           (opcional)
  historical_data/
    m5_clean/
  processed_data/
    m5_windows/
  metadata/
    filters/
    fees/
  logs/
  reports/                  (opcional)
  backups/
```

---

## 3) Tokens/Placeholders usados en rutas

- `<SYM>`: símbolo (ej. `EURUSD`, `XAUUSD`, etc.)
- `<YYYY>`: año (4 dígitos)
- `<MM>`: mes (2 dígitos)
- `<YYYYMMDD>`: día (8 dígitos)
- `<RUN_ID>`: identificador de corrida (definido por el notebook)
- `<TS>`: timestamp compacto (formato exacto depende del notebook)

---

## 4) Lista única final (rutas relativas a `DATA_ROOT`)

> Notación:  
> - `[DIR]` carpeta/namespace  
> - `[OUT]` archivo/carpeta escrita por el notebook  
> - `[IN-OPT]` input opcional (el notebook lo lee si existe; no es generado obligatoriamente)

- [DIR] `bulk_data/`
- [OUT] `bulk_data/m5_raw/`
- [OUT] `bulk_data/m5_raw/symbol=<SYM>/year=<YYYY>/month=<MM>/part=<YYYYMMDD>.parquet`
- [DIR] `bulk_data/ticks_recent/` (opcional / reservado)

- [DIR] `historical_data/`
- [OUT] `historical_data/m5_clean/`
- [OUT] `historical_data/m5_clean/symbol=<SYM>/year=<YYYY>/month=<MM>/part=<YYYYMMDD>.parquet`
- [OUT] `historical_data/m5_clean/symbol=<SYM>/part=<YYYYMMDD>.parquet` (compatibilidad legacy)

- [DIR] `processed_data/`
- [OUT] `processed_data/m5_windows/`
- [OUT] `processed_data/m5_windows/window=<NAME>/symbol=<SYM>/part=<YYYYMMDD>.parquet`

- [DIR] `metadata/`

- [OUT] `metadata/config_snapshot.json`
- [OUT] `metadata/schema_m5.json`
- [OUT] `metadata/symbols_broker.parquet`
- [OUT] `metadata/server_time_info.json`

- [OUT] `metadata/dataset_catalog.parquet`
- [OUT] `metadata/m5_manifest.parquet`
- [OUT] `metadata/manifest.json`

- [OUT] `metadata/costs_summary.parquet`

- [DIR] `metadata/filters/`
- [OUT] `metadata/filters/eligible_symbols_by_cost.parquet`
- [OUT] `metadata/filters/eligible_symbols_by_cost.txt`
- [OUT] `metadata/filters/cost_filter_report.json`
- [IN-OPT] `metadata/filters/cost_filter_config.json`

- [DIR] `metadata/fees/`
- [IN-OPT] `metadata/fees/commissions.json`

- [OUT] `metadata/qa_m5_bulk.parquet`
- [OUT] `metadata/qa_operativa_summary.parquet`

- [OUT] `metadata/data_quality_summary.parquet`
- [OUT] `metadata/universe_snapshot_latest.parquet`
- [OUT] `metadata/universe_snapshot_<RUN_ID>.parquet`

- [OUT] `metadata/day_index_m5.parquet`
- [OUT] `metadata/symbol_index_m5.parquet`
- [OUT] `metadata/window_catalog_m5.parquet`

- [OUT] `metadata/run_log.jsonl`
- [OUT] `metadata/checksums.jsonl`

- [OUT] `metadata/qa_trading_ready_summary.json`
- [OUT] `metadata/pipeline_health_report.json`
- [OUT] `metadata/TRADING_READY.flag`
- [OUT] `metadata/TRADING_NOT_READY.flag`

- [DIR] `logs/`
- [OUT] `logs/mt5_de_5m_<RUN_ID>.log`
- [OUT] `logs/parquet_sample_<TS>.txt`

- [DIR] `reports/` (opcional / reservado)

- [DIR] `backups/`
- [OUT] `backups/backup_m5_<RUN_ID>.zip`

---

## 5) Regla de compatibilidad para notebooks siguientes

Los notebooks posteriores deben:
1) Resolver `DATA_ROOT` **una sola vez** (mismo valor del notebook 01).  
2) Leer `metadata/config_snapshot.json` como fuente canónica de paths.  
3) Tratar `metadata/m5_manifest.parquet` (detalle por archivo) y `metadata/manifest.json` (manifest de corrida) como artefactos distintos.

Fin del registro.
