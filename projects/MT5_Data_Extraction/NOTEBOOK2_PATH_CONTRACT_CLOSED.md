# NOTEBOOK2_PATH_CONTRACT — CLOSED (lista final única)

**Notebook:** `02_ER_FILTER_5M_V4.ipynb`  
**Estado:** **CERRADO** (contrato final para registro)  
**Fecha:** 2025-12-23 (America/Guayaquil)

---

## 1) Convenciones y raíz del proyecto

### 1.1 PROJECT_ROOT (autodetección)
El notebook resuelve `PROJECT_ROOT` ascendiendo desde `Path.cwd()` hasta encontrar **al menos una** de estas carpetas “marcador”:

- `bulk_data/`
- `processed_data/`
- `outputs/`

Si no encuentra ninguna, aborta.

### 1.2 RUN_ID (identificador de corrida)
`RUN_ID` se genera en **UTC** con el formato:

- `YYYYMMDD_HHMMSS`  
  Ejemplo: `20251223_142501`

---

## 2) Rutas base (inputs y config persistente)

### 2.1 Config maestro (persistente)
- `CONFIG_DIR = PROJECT_ROOT/config`
- **Master config** (entrada/salida; se crea si no existe):
  - `CONFIG_DIR/er_filter_5m.json`

> Este archivo es “persistente” (no pertenece a un run). Si no existe, el notebook lo autogenera con valores conservadores.

### 2.2 Snapshot del Data Engine (recomendado)
- `CONFIG_SNAPSHOT_PATH = PROJECT_ROOT/data/metadata/config_snapshot.json`

**Uso:**
- Si existe, el notebook intenta leer `dataset.paths` y usar esos paths como **overrides** (DATA_ROOT, M5_CLEAN_DIR, etc.).
- Si no existe, usa defaults (ver 2.3).

### 2.3 Defaults (si no hay `config_snapshot.json`)
- `DATA_ROOT = PROJECT_ROOT/data`
- `METADATA_DIR = DATA_ROOT/metadata`
- `M5_CLEAN_DIR = DATA_ROOT/bulk_data/m5_clean`
- `M5_RAW_DIR = DATA_ROOT/bulk_data/m5_raw`
- `PROCESSED_DATA_DIR = DATA_ROOT/processed_data`

---

## 3) Outputs del RUN (estructura oficial)

### 3.1 OUT_ROOT (sin prefijo `run_`)
**Base de outputs del run:**

- `OUT_ROOT = PROJECT_ROOT/outputs/er_filter_5m/<RUN_ID>`

Subcarpetas creadas por el notebook:

- `OUT_ROOT/logs/`
- `OUT_ROOT/diagnostics/`
- `OUT_ROOT/metrics/`
- `OUT_ROOT/events/`
- `OUT_ROOT/stability/`
- `OUT_ROOT/scores/`
- `OUT_ROOT/baskets/`
- `OUT_ROOT/exports/`
- `OUT_ROOT/reports/`

### 3.2 logs/
- `OUT_ROOT/logs/run_metadata.json`

### 3.3 diagnostics/
- `OUT_ROOT/diagnostics/config.json`  *(snapshot exacto del config maestro para este run)*
- `OUT_ROOT/diagnostics/handshake_summary.json`
- `OUT_ROOT/diagnostics/artifacts_summary.json`

**Autogenerados (solo si faltan inputs seleccionados):**
- `OUT_ROOT/diagnostics/watchlist_autogen.parquet`
- `OUT_ROOT/diagnostics/params_autogen.parquet`

**Cobertura M5:**
- `OUT_ROOT/diagnostics/coverage_table_5m.parquet`
- `OUT_ROOT/diagnostics/coverage_table_5m.csv`

**Percentiles ER/PD (para umbrales/régimen):**
- `OUT_ROOT/diagnostics/er_pd_percentiles_summary.json`

**Snapshots de estado (debug controlado):**
- `OUT_ROOT/diagnostics/global_state_snapshot_c05.json`
- `OUT_ROOT/diagnostics/global_state_snapshot_c06.json`
- `OUT_ROOT/diagnostics/global_state_snapshot_c07.json`

**Correlación (builder QA):**
- `OUT_ROOT/diagnostics/corr_matrix_builder_mapping.parquet`
- `OUT_ROOT/diagnostics/corr_matrix_builder_report.html`

**Reportes HTML auxiliares:**
- `OUT_ROOT/diagnostics/regimen_selector_report.html`
- `OUT_ROOT/diagnostics/baskets_core_export_report.html`

**Hashes de exports:**
- `OUT_ROOT/diagnostics/exports_hashes.json`

### 3.4 metrics/
- `OUT_ROOT/metrics/er_series.parquet`
- `OUT_ROOT/metrics/pd_series.parquet`
- `OUT_ROOT/metrics/regime_volatility_summary.parquet`
- `OUT_ROOT/metrics/regime_thresholds.parquet`
- `OUT_ROOT/metrics/regime_labels.parquet`
- `OUT_ROOT/metrics/economic_viability.parquet`
- `OUT_ROOT/metrics/economic_viability_meta.json`
- `OUT_ROOT/metrics/structure_summary.parquet`
- `OUT_ROOT/metrics/microstructure_summary.parquet`
- `OUT_ROOT/metrics/frequency_opportunity_table.parquet`

### 3.5 events/
- `OUT_ROOT/events/trend_events.parquet`
- `OUT_ROOT/events/range_events.parquet`

### 3.6 stability/
- `OUT_ROOT/stability/stab_folds.parquet`
- `OUT_ROOT/stability/stability_table.parquet`
- `OUT_ROOT/stability/stability_table_advanced.parquet`

### 3.7 scores/
- `OUT_ROOT/scores/scores_table.parquet`
- `OUT_ROOT/scores/scores_meta.json`
- `OUT_ROOT/scores/candidates_table.parquet`
- `OUT_ROOT/scores/best_per_symbol.parquet`
- `OUT_ROOT/scores/freq_only_watchlist.parquet`
- `OUT_ROOT/scores/asset_strategy_profiles.parquet`
- `OUT_ROOT/scores/asset_strategy_shortlist.parquet`
- `OUT_ROOT/scores/handoff_operational.parquet`
- `OUT_ROOT/scores/selection_table.parquet`
- `OUT_ROOT/scores/selection_table_enriched.parquet`

### 3.8 baskets/
- `OUT_ROOT/baskets/basket_trend_core.parquet`
- `OUT_ROOT/baskets/basket_range_core.parquet`
- `OUT_ROOT/baskets/basket_trend_core_symbols.txt`
- `OUT_ROOT/baskets/basket_range_core_symbols.txt`
- `OUT_ROOT/baskets/selection_symbols.txt`

### 3.9 exports/
Se escriben exports por basket y un export consolidado:

**Por basket (puede haber múltiples):**
- `OUT_ROOT/exports/ea_universe_<family>_<preset>.csv`
- `OUT_ROOT/exports/ea_universe_<family>_<preset>.parquet`

**Consolidado:**
- `OUT_ROOT/exports/ea_universe_all.csv`
- `OUT_ROOT/exports/ea_universe_all.parquet`

### 3.10 reports/
**Reporte final de run (ligero, sin backtesting):**
- `OUT_ROOT/reports/regimen_selector_report.html`

---

## 4) Inputs requeridos (dependencias)

### 4.1 M5 (data)
El notebook selecciona una fuente de M5 en este orden:

1) `M5_CLEAN_DIR` (preferido)  
2) `M5_RAW_DIR` (fallback)  
3) Restore (fallback extremo):
   - `PROJECT_ROOT/data/restore/<restore_id>/historical_data/m5_clean/*.parquet`

> El restore se usa solo si no hay parquets disponibles en clean/raw.

### 4.2 PADs (metadata, normalmente producidos por el Data Engine)
En `METADATA_DIR`:

- `day_index.parquet`
- `symbol_index.parquet`
- `window_catalog.parquet`
- `dataset_catalog.parquet`
- `m5_manifest.parquet`
- `data_quality_summary.json`
- `qa_operativa_summary.json`
- `ticks_recent_qc_summary.json`

### 4.3 Universo y correlación (processed_data)
En `PROCESSED_DATA_DIR`:

- `universe_ranked.parquet` *(requerido)*  
- `corr_matrix_5m.csv` *(opcional; si falta, se construye — ver 5)*

### 4.4 Selección (watchlist/params) — opcional
En `DATA_ROOT`:

- `ea_watchlist.parquet` o `ea_watchlist.csv`
- `ea_params.parquet` o `ea_params.csv`

Si no existen, el notebook autogenera archivos dentro de `OUT_ROOT/diagnostics/`.

---

## 5) Outputs compartidos (fuera de OUT_ROOT)

### 5.1 Correlación global (si se construye)
Si el builder de correlación se ejecuta, se escribe:

- `PROCESSED_DATA_DIR/corr_matrix_5m.csv`

Durante escritura atómica puede aparecer temporalmente:
- `PROCESSED_DATA_DIR/corr_matrix_5m__tmp_<RUN_ID>.csv` *(temporal; luego se renombra a corr_matrix_5m.csv)*

### 5.2 Config maestro (persistente)
- `CONFIG_DIR/er_filter_5m.json` *(se crea si no existe; luego se reutiliza)*

---

## 6) Modo fallback (standalone, no recomendado para registro)

Si el notebook no puede resolver `GLOBAL_STATE["paths"]` (por ejecución parcial/aislada), algunas salidas pueden ir a:

- `ART_DIR = <cfg.ARTIFACTS_DIR> (default: PROJECT_ROOT/artifacts)`

En ese modo, podrían aparecer:
- `ART_DIR/trend_events.parquet`
- `ART_DIR/range_events.parquet`
- `ART_DIR/regime_labels.parquet`
- `ART_DIR/frequency_opportunity_table.parquet`

Para “registro” oficial, el modo esperado es el estándar con `OUT_ROOT`.

---

## 7) Reglas de inmutabilidad del contrato (registro)
1) No renombrar archivos listados en la sección 3.  
2) No mover carpetas internas del run (`logs/diagnostics/metrics/...`).  
3) Mantener el OUT_ROOT como `outputs/er_filter_5m/<RUN_ID>` (sin prefijos adicionales).

---
