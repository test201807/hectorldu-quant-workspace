# 00_AUDIT_REPORT.md — Auditoría Global del Workspace

**Fecha inicial:** 2026-02-07
**Última actualización:** 2026-02-07 (post-migración)
**Alcance:** `C:\Quant\` completo (nivel 1 y 2, excluyendo `.venv`/`venv1`/`btc_env`)

---

## 1) Árbol de Carpetas Actual (Post-Migración)

```
C:\Quant\
│
├── .git/                                        Monorepo Git (inicializado 2026-02-07)
├── .gitignore                                   Global: excluye venvs, data, outputs, .env, binarios
├── .claude/                                     Metadata Claude Code
├── CLAUDE.md                                    Guía para Claude Code
├── 00_AUDIT_REPORT.md                           ← este archivo
│
├── 00_shared/                                   Recursos compartidos
│   ├── docs/                                    Documentos de planificación
│   │   ├── PLAN DE DESARROLLO DE QUANT FINAL.docx
│   │   ├── MOTOR DE ESTRATEGIAS.docx
│   │   ├── errores a aislar y ya no repetir.pdf
│   │   └── errores a aislar y ya no repetir 2.pdf
│   └── tools/
│       └── _audit_tools/                        Scripts de auditoría
│           ├── quant_audit_local.py
│           ├── quant_audit_openai.py
│           └── quant_audit_full.ps1
│
├── 99_archive/                                  Respaldos y versiones históricas
│   ├── audit_out/                               Resultados de auditorías previas (9 archivos)
│   ├── mt5_respaldo/                            Backups MT5 (notebooks, docs, configs anteriores)
│   │   ├── MT5_DE_5M_V1.ipynb
│   │   ├── ER_FILTER_5M_V4.ipynb
│   │   ├── ER_FILTER_5M_V4.docx
│   │   ├── Data Engine MT5 M5.docx / .pdf
│   │   ├── CONTEXTO.txt
│   │   └── backup_20251126_1534/config/
│   ├── twf_respaldo/
│   │   └── A_E_P.ipynb                          Versión anterior del notebook TWF
│   ├── strategy_lab_old_features/               Señales y features obsoletas
│   │   ├── features_base/                       (v1 — reemplazada por features_base_v11)
│   │   ├── signals_base_v1/                     (reemplazada por signals_base_v3)
│   │   └── signals_base_v2/                     (reemplazada por signals_base_v3)
│   └── filtered_symbols.parquet                 Output transitorio archivado de MT5 raíz
│
├── MT5_Data_Extraction/                         ★ PROYECTO PRINCIPAL
│   ├── 01_MT5_DE_5M_V1.ipynb                   (Data Engine)
│   ├── 02_ER_FILTER_5M_V4.ipynb                (ER Filter)
│   ├── AUDITOR_CODE.ipynb                       (auditoría interna)
│   ├── NOTEBOOK1_PATH_CONTRACT.md               (contrato CERRADO)
│   ├── NOTEBOOK2_PATH_CONTRACT_CLOSED.md        (contrato CERRADO)
│   ├── pipeline.log                             (log activo)
│   ├── artifacts/
│   │   └── v2/                                  (runs con UUID: run_20251222_*, run_20251223_*)
│   ├── bulk_data/                               Marcador vacío (necesario para autodetección NB2)
│   │   └── rates_5m/                            (vacío)
│   ├── config/
│   │   └── er_filter_5m.json                    (master config persistente)
│   ├── data/
│   │   ├── backups/                             (3 ZIPs — política: últimos 3)
│   │   │   ├── backup_m5_20251121_161336.zip
│   │   │   ├── backup_m5_20251201_220842.zip
│   │   │   └── backup_m5_20251202_232253.zip
│   │   ├── bulk_data/
│   │   │   └── m5_raw/                          (130 símbolos Hive-partitioned)
│   │   ├── historical_data/                     (legacy — mantener por contrato NB1)
│   │   ├── logs/                                (35 logs de pipeline)
│   │   ├── metadata/                            (config_snapshot, manifests, QA, schemas)
│   │   │   ├── filters/
│   │   │   └── fees/
│   │   ├── processed_data/
│   │   │   ├── m5_windows/
│   │   │   └── corr_matrix_5m.csv               (copiado desde processed_data/ raíz)
│   │   ├── reports/                             (vacío/reservado)
│   │   ├── restore/                             (3 puntos — política: últimos 3)
│   │   │   ├── restore_20251121_161336/
│   │   │   ├── restore_20251201_220842/
│   │   │   └── restore_20251202_232253/
│   │   └── ea_params.parquet
│   ├── diagnostics_global/
│   │   └── inventory_extracts/
│   ├── ER_STRATEGY_LAB/
│   │   ├── README.md                            (documentado — estructura, notebooks, convenciones)
│   │   ├── config/                              (6 YAMLs: project, data_contract, costs_model, wfo, risk, sessions)
│   │   ├── docs/
│   │   │   ├── methodology.md                   (documentado — pipeline, parámetros, versionado)
│   │   │   └── changelog.md                     (documentado — historial de runs y cambios)
│   │   ├── inputs/
│   │   │   ├── diagnostics/
│   │   │   └── shortlist/
│   │   ├── notebooks/
│   │   │   ├── 03_TREND_M5_Strategy_v1.ipynb
│   │   │   ├── 03_TREND_M5_Strategy_v2.ipynb
│   │   │   ├── 04_RANGE_M5_Strategy_v1.ipynb
│   │   │   ├── artifacts/                       (NO MOVER — v2 usa paths relativos desde notebooks/)
│   │   │   └── outputs/                         (NO MOVER — v2 usa WORKDIR / "outputs")
│   │   ├── artifacts/                           (resultados v1: backtests, features, deploy, wfo)
│   │   │   ├── alpha_design/
│   │   │   ├── backtests/
│   │   │   ├── deploy/
│   │   │   ├── features/                        (activos: features_base_v11, m5_clean, m5_ohlcv_clean, regime_gate, signals_base_v3)
│   │   │   ├── selection/
│   │   │   └── wfo/
│   │   └── research_logs/
│   │       └── runs/                            (6 carpetas timestamped)
│   ├── outputs/
│   │   └── er_filter_5m/
│   │       └── 20251218_190810/                 (último run — inmutable por contrato)
│   │           ├── baskets/
│   │           ├── diagnostics/
│   │           ├── events/
│   │           ├── exports/
│   │           ├── logs/
│   │           ├── metrics/
│   │           ├── reports/
│   │           ├── scores/
│   │           └── stability/
│   ├── processed_data/                          (original preservado — corr_matrix_5m.csv)
│   ├── snapshots/
│   └── venv1/                                   (excluido)
│
├── TWF/
│   ├── A_E_P_v1.ipynb                           (notebook principal, 365 KB)
│   ├── README.md
│   ├── requirements.txt                         (13 dependencias)
│   ├── runtime_defaults.json                    (n_jobs: 8)
│   ├── Versión Revisada TWF.docx
│   ├── codigo compilado.txt                     (83 KB)
│   ├── .gitignore
│   ├── .vscode/                                 (settings.json, tasks.json)
│   ├── data/
│   │   ├── BTCUSDT/                             (5m/ y 15m/)
│   │   ├── external/                            (vacío)
│   │   ├── intermediate/                        (vacío)
│   │   └── raw/BTCUSDT/5m/                      (6 JSON klines)
│   ├── logs/
│   │   └── run_log_records.csv                  (626 KB)
│   ├── notebooks/                               (vacío)
│   ├── outputs/BTCUSDT/15m/                     (reports, features, forward, models, wf, etc.)
│   ├── scripts/                                 (vacío — referenciado en README y tasks.json)
│   ├── src/twf/                                 (io/binance.py, utils/config.py, utils/logging.py)
│   └── .venv/                                   (excluido)
│
├── BTC_ANALIST/
│   ├── requirements.txt                         (17 dependencias — poblado desde btc_env)
│   ├── data/
│   │   ├── raw/                                 (btcusd_bitstamp csvs)
│   │   └── processed/                           (btc_4h_full.parquet, on-chain csvs, _onchain_cache/)
│   ├── notebooks/
│   │   ├── BTC_ANALIST_v1.ipynb                 (notebook principal)
│   │   └── data/processed/                      (16 CSVs bitcoinisdata_* — no migrados, ver nota)
│   ├── results/                                 (figures/, logs/, tables/)
│   ├── src/                                     (backtest, config, cycles, data, indicators, optimization, utils)
│   ├── tests/                                   (vacío)
│   ├── workshop_report/figs/
│   └── btc_env/                                 (excluido)
│
├── GESTOR DE IA/
│   ├── .env                                     (API keys — protegido por .gitignore)
│   ├── .gitignore
│   ├── .vscode/settings.json
│   ├── requirements.txt                         (20 dependencias — generado desde .venv)
│   ├── configs/
│   │   ├── runtime.yaml                         (gpt-5.2, rebalance mensual, riesgo)
│   │   └── universe.csv
│   ├── data/
│   │   ├── cache/                               (yahoo_prices/, simfin/, ai_scores/, ai_portfolio/)
│   │   ├── processed/                           (3 parquets)
│   │   └── raw/simfin_universe/                 (6 parquets)
│   ├── logs/
│   ├── notebooks/
│   │   └── GESTOR.ipynb                         (741 KB)
│   ├── reports/                                 (holdings, audits, backtests, strategy comparison)
│   ├── src/                                     (vacío)
│   ├── workshop_report/figs/
│   └── .venv/                                   (excluido)
```

---

## 2) Resumen por Carpeta Principal

| Carpeta | Propósito | Tamaño aprox. | # Archivos (sin venv) |
|---------|-----------|---------------|----------------------|
| **MT5_Data_Extraction** | Pipeline: extracción MT5 → limpieza → ER filter → strategy lab | ~1.4 GB (parquets m5_raw) | ~1,400,000+ |
| **TWF** | Pipeline estadístico de microestructura (BTCUSDT, Binance). NO trading. | ~73 MB | ~148 |
| **BTC_ANALIST** | Análisis de ciclos BTC con indicadores on-chain (MVRV, profit addresses) | ~100 MB | ~48 |
| **GESTOR DE IA** | Gestión de portafolio AI-driven (GPT-5.2 + SimFin + Yahoo Finance) | ~50 MB | ~48 |
| **00_shared** | Documentos de planificación + herramientas de auditoría | ~3.8 MB | 7 |
| **99_archive** | Respaldos, features obsoletas, resultados de auditoría | ~15 MB + old features | ~30+ |

---

## 3) Mapa de Rutas Exactas

### Extracción MT5 (Data Engine)
| Recurso | Ruta |
|---------|------|
| Notebook | `MT5_Data_Extraction\01_MT5_DE_5M_V1.ipynb` |
| Contrato de rutas | `MT5_Data_Extraction\NOTEBOOK1_PATH_CONTRACT.md` |
| Output m5_raw | `MT5_Data_Extraction\data\bulk_data\m5_raw\symbol=<SYM>\year=<YYYY>\month=<MM>\` |
| Output m5_clean | `MT5_Data_Extraction\data\historical_data\m5_clean\symbol=<SYM>\...` |
| Metadata | `MT5_Data_Extraction\data\metadata\` |
| Logs del pipeline | `MT5_Data_Extraction\data\logs\mt5_de_5m_<RUN_ID>.log` |
| Backups (ZIP) | `MT5_Data_Extraction\data\backups\` (3 archivos, últimos 3) |
| Puntos de restauración | `MT5_Data_Extraction\data\restore\` (3 carpetas, últimos 3) |

### Limpieza / Processing
| Recurso | Ruta |
|---------|------|
| m5_clean (candles limpias) | `MT5_Data_Extraction\data\historical_data\m5_clean\` |
| processed_data (ventanas) | `MT5_Data_Extraction\data\processed_data\m5_windows\` |
| Correlación global (canónica) | `MT5_Data_Extraction\data\processed_data\corr_matrix_5m.csv` |
| Correlación global (original) | `MT5_Data_Extraction\processed_data\corr_matrix_5m.csv` |
| universe_ranked | `MT5_Data_Extraction\data\processed_data\universe_ranked.parquet` |

### ER Filter
| Recurso | Ruta |
|---------|------|
| Notebook | `MT5_Data_Extraction\02_ER_FILTER_5M_V4.ipynb` |
| Contrato de rutas | `MT5_Data_Extraction\NOTEBOOK2_PATH_CONTRACT_CLOSED.md` |
| Config maestro | `MT5_Data_Extraction\config\er_filter_5m.json` |
| Último run | `MT5_Data_Extraction\outputs\er_filter_5m\20251218_190810\` |
| — diagnostics/ | 13 archivos: config.json, handshake, coverage, state snapshots, hashes |
| — metrics/ | er_series, pd_series, regime_*, economic_viability, structure, microstructure |
| — events/ | trend_events.parquet, range_events.parquet |
| — stability/ | stab_folds, stability_table, stability_table_advanced |
| — scores/ | scores_table, candidates, best_per_symbol, asset_strategy_*, selection_* |
| — baskets/ | basket_trend_core, basket_range_core, .txt symbol lists |
| — exports/ | ea_universe_TREND_v1, ea_universe_RANGE_v1, ea_universe_all (CSV+parquet) |
| — reports/ | regimen_selector_report.html |

### Strategy Lab / Backtests
| Recurso | Ruta |
|---------|------|
| Notebook TREND v1 | `MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\03_TREND_M5_Strategy_v1.ipynb` |
| Notebook TREND v2 | `MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\03_TREND_M5_Strategy_v2.ipynb` |
| Notebook RANGE v1 | `MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\04_RANGE_M5_Strategy_v1.ipynb` |
| Configs YAML | `ER_STRATEGY_LAB\config\` (project, data_contract, costs_model, wfo, risk, sessions) |
| Inputs/shortlist | `ER_STRATEGY_LAB\inputs\shortlist\` |
| Artifacts v1 | `ER_STRATEGY_LAB\artifacts\` (backtests/, features/, deploy/, wfo/, selection/) |
| Features activas | `ER_STRATEGY_LAB\artifacts\features\` (features_base_v11, signals_base_v3, m5_clean, m5_ohlcv_clean, regime_gate) |
| Outputs v2 | `ER_STRATEGY_LAB\notebooks\outputs\trend_m5_strategy\v2\` (NO MOVER) |
| Research logs | `ER_STRATEGY_LAB\research_logs\runs\` (6 carpetas) |
| Documentación | `ER_STRATEGY_LAB\docs\` (methodology.md, changelog.md) y `README.md` |

### Todos los Notebooks
| # | Notebook | Ruta | Estado |
|---|----------|------|--------|
| 1 | Data Engine MT5 | `MT5_Data_Extraction\01_MT5_DE_5M_V1.ipynb` | Activo |
| 2 | ER Filter | `MT5_Data_Extraction\02_ER_FILTER_5M_V4.ipynb` | Activo |
| 3 | Auditor interno | `MT5_Data_Extraction\AUDITOR_CODE.ipynb` | Activo |
| 4 | TREND Strategy v1 | `ER_STRATEGY_LAB\notebooks\03_TREND_M5_Strategy_v1.ipynb` | Activo |
| 5 | TREND Strategy v2 | `ER_STRATEGY_LAB\notebooks\03_TREND_M5_Strategy_v2.ipynb` | Activo |
| 6 | RANGE Strategy v1 | `ER_STRATEGY_LAB\notebooks\04_RANGE_M5_Strategy_v1.ipynb` | Activo |
| 7 | TWF Pipeline | `TWF\A_E_P_v1.ipynb` | Activo |
| 8 | BTC Analyst | `BTC_ANALIST\notebooks\BTC_ANALIST_v1.ipynb` | Activo |
| 9 | GESTOR AI | `GESTOR DE IA\notebooks\GESTOR.ipynb` | Activo |
| 10 | TWF backup | `99_archive\twf_respaldo\A_E_P.ipynb` | Archivo |
| 11 | MT5 DE backup | `99_archive\mt5_respaldo\MT5_DE_5M_V1.ipynb` | Archivo |
| 12 | ER Filter backup | `99_archive\mt5_respaldo\ER_FILTER_5M_V4.ipynb` | Archivo |
| 13 | ER Filter V1 old | `99_archive\mt5_respaldo\backup_20251126_1534\ER_FILTER_5M_V1.ipynb` | Archivo |

### Contratos de Rutas (Documentos CERRADOS — no modificar)
| Contrato | Ruta | Estado | Fecha |
|----------|------|--------|-------|
| Notebook 1 (Data Engine) | `MT5_Data_Extraction\NOTEBOOK1_PATH_CONTRACT.md` | CERRADO | 2025-12-23 |
| Notebook 2 (ER Filter) | `MT5_Data_Extraction\NOTEBOOK2_PATH_CONTRACT_CLOSED.md` | CERRADO | 2025-12-23 |

---

## 4) Problemas — Estado Post-Migración

### ALTA

| # | Problema | Estado | Detalle |
|---|----------|--------|---------|
| A1 | API keys en texto plano | **ABIERTO** (riesgo aceptado) | `.env` protegido por `.gitignore`. Keys no rotadas — el usuario confirmó que no es necesario mientras el workspace no se comparta. |
| A2 | Sin Git | **RESUELTO** | Monorepo inicializado en `C:\Quant\.git\` con `.gitignore` global. |
| A3 | `bulk_data/` duplicado | **MITIGADO** | `bulk_data/rates_5m/` está vacío — se mantiene como marcador para autodetección de PROJECT_ROOT por NB2. No contiene datos duplicados. |
| A4 | `processed_data/` duplicado | **MITIGADO** | `corr_matrix_5m.csv` copiado a `data/processed_data/`. Original preservado en `processed_data/` raíz por seguridad (NB2 podría leerlo desde ahí). |

### MEDIA

| # | Problema | Estado | Detalle |
|---|----------|--------|---------|
| M1 | Outputs anidados en notebooks/ | **NO RESOLUBLE** | NB v2 usa `WORKDIR / "outputs"` con path relativo desde `notebooks/`. Mover rompería el notebook. Documentado como limitación arquitectónica. |
| M2 | `filtered_symbols.parquet` suelto | **RESUELTO** | Archivado en `99_archive/`. |
| M3 | `historical_data/` legacy | **ABIERTO** | Se mantiene por contrato NB1 (define m5_clean ahí). Requiere auditoría de contenido. |
| M4 | `scripts/` vacío en TWF | **ABIERTO** | Pendiente de verificar si se generan en runtime. |
| M5 | Documentación placeholder | **RESUELTO** | README.md, methodology.md y changelog.md del Strategy Lab ahora tienen contenido real. |
| M6 | 9 restore + 9 backups sin retención | **RESUELTO** | Política aplicada: últimos 3 retenidos, 6 eliminados (~4.8 GB liberados). |
| M7 | Versiones de señales acumuladas | **RESUELTO** | `features_base` (v1), `signals_base_v1`, `signals_base_v2` archivados en `99_archive/strategy_lab_old_features/`. Activos: `features_base_v11`, `signals_base_v3`. |
| M8 | BTC_ANALIST: requirements vacío | **RESUELTO** | Poblado con 17 dependencias desde `btc_env`. |
| M9 | GESTOR DE IA: sin requirements | **RESUELTO** | Creado con 20 dependencias desde `.venv`. |

### BAJA

| # | Problema | Estado | Detalle |
|---|----------|--------|---------|
| B1 | Docs sueltos en raíz | **RESUELTO** | Movidos a `00_shared/docs/`. |
| B2 | Docs sueltos en TWF | **ABIERTO** | `Versión Revisada TWF.docx` y `codigo compilado.txt` siguen en TWF. |
| B3 | Docs en respaldo MT5 | **RESUELTO** | Respaldo completo migrado a `99_archive/mt5_respaldo/`. |
| B4 | `data/` dentro de `notebooks/` BTC | **ABIERTO** | CSVs `bitcoinisdata_*` permanecen en `notebooks/data/processed/`. No se pudo verificar si el notebook los referencia internamente. |
| B5 | Nombres inconsistentes de venvs | **ABIERTO** (pospuesto) | Requiere recrear entornos. Mejor como tarea independiente. |
| B6 | Placeholders _DROP_HERE | **ABIERTO** | Menor — sirven como guía visual para el usuario. |
| B7 | `workshop_report/` duplicado | **ABIERTO** | Menor — residuo de template. |

---

## 5) Estructura Objetivo vs. Estado Actual

| Objetivo | Estado | Nota |
|----------|--------|------|
| `00_shared/docs/` con documentos centralizados | ✅ Completado | 4 documentos migrados |
| `00_shared/tools/` con audit scripts | ✅ Completado | 3 scripts migrados |
| `99_archive/` con respaldos centralizados | ✅ Completado | mt5_respaldo, twf_respaldo, audit_out, old features |
| Git monorepo con .gitignore | ✅ Completado | Protege data, venvs, .env, outputs |
| requirements.txt en cada proyecto | ✅ Completado | TWF (existía), BTC_ANALIST (poblado), GESTOR DE IA (creado) |
| Política de retención backups/restore | ✅ Completado | Últimos 3 retenidos |
| Features obsoletas archivadas | ✅ Completado | v1/v2 → 99_archive/ |
| Documentación Strategy Lab | ✅ Completado | README, methodology, changelog |
| Sacar outputs de notebooks/ Strategy Lab | ❌ No viable | NB v2 depende de paths relativos desde notebooks/ |
| Mover notebooks/data/ BTC_ANALIST | ❌ No viable sin auditoría celda-por-celda | Riesgo de romper references internas |
| Estandarizar nombres de venvs | ⏸ Pospuesto | Requiere recrear entornos |
| Eliminar bulk_data/ duplicado | ❌ No viable | Necesario como marcador para autodetección NB2 |

---

## 6) Registro de Migración Ejecutada

### Bloque A — Preparación ✅
| Paso | Acción | Resultado |
|------|--------|----------|
| A1 | Crear `00_shared/docs/` y `00_shared/tools/` | Hecho |
| A2 | Crear `99_archive/` | Hecho |
| A3 | Rotar API keys | Descartado — no necesario |
| A4 | Generar `GESTOR DE IA\requirements.txt` | Hecho (20 deps) |
| A5 | Poblar `BTC_ANALIST\requirements.txt` | Hecho (17 deps) |
| A6 | Inicializar Git monorepo + `.gitignore` | Hecho |

### Bloque B — Mover documentos y respaldos ✅
Ejecutado externamente (fuera de esta sesión). Verificado intacto.

| Paso | Resultado |
|------|----------|
| B1 | 4 docs raíz → `00_shared/docs/` |
| B2 | `_audit_tools/` → `00_shared/tools/_audit_tools/` |
| B3 | `_audit_out/` → `99_archive/audit_out/` |
| B4 | `MT5_Data_Extraction/respaldo/` → `99_archive/mt5_respaldo/` |
| B5 | `TWF/respaldo/` → `99_archive/twf_respaldo/` |

### Bloque C — Resolver duplicados MT5 ✅
| Paso | Acción | Resultado |
|------|--------|----------|
| C1 | Verificar `bulk_data/` raíz | Vacío (rates_5m/ sin archivos) |
| C2 | Eliminar `bulk_data/` | Descartado — marcador necesario para NB2 |
| C3 | Copiar `corr_matrix_5m.csv` → `data/processed_data/` | Hecho (original preservado) |
| C4 | Archivar `filtered_symbols.parquet` | Hecho → `99_archive/` |
| C5 | Retención restore points: últimos 3 | 6 eliminados |
| C6 | Retención backups ZIP: últimos 3 | 6 eliminados (~4.8 GB liberados) |

### Bloque D — Limpieza Strategy Lab ✅
| Paso | Acción | Resultado |
|------|--------|----------|
| D1 | Migrar `notebooks/outputs/` | Descartado — NB v2 depende de paths relativos |
| D2 | Migrar `notebooks/artifacts/` | Descartado — misma razón |
| D3 | Archivar features/señales obsoletas | Hecho → `99_archive/strategy_lab_old_features/` |
| D4 | Documentar Strategy Lab | Hecho — README.md, methodology.md, changelog.md |
| D5 | Mover `notebooks/data/` BTC_ANALIST | Descartado — riesgo sin auditoría celda-por-celda |
| D6 | Estandarizar venvs | Pospuesto |

---

## 7) Problemas Residuales (pendientes para futuras sesiones)

| # | Problema | Prioridad | Acción sugerida |
|---|----------|-----------|-----------------|
| 1 | `historical_data/` en MT5 — ¿en uso? | Media | Auditar si NB1/NB2 escriben o leen de `m5_clean/` |
| 2 | `scripts/` vacío en TWF | Baja | Verificar si el notebook genera scripts en runtime |
| 3 | `notebooks/data/processed/` en BTC_ANALIST | Baja | Auditar notebook celda por celda para migrar datos |
| 4 | Nombres de venvs inconsistentes | Baja | Recrear como `.venv` en todos los proyectos |
| 5 | `notebooks/outputs/` y `notebooks/artifacts/` en Strategy Lab | Baja | Refactorizar NB v2 para escribir a nivel de ER_STRATEGY_LAB |
| 6 | `bulk_data/` vacío como marcador | Baja | Documentar en el contrato o crear `.marker` file |
| 7 | `processed_data/` duplicado en MT5 raíz | Baja | Eliminar una vez confirmado que NB2 lee de `data/processed_data/` |
| 8 | Primer commit Git pendiente | Media | Ejecutar cuando el usuario lo solicite |

---

*Auditoría completada y migración ejecutada. Todos los cambios documentados.*
