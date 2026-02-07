# Workspace Index — C:\Quant

## Proyectos

| Proyecto | Ruta real | Junction (compatibilidad) | Descripción |
|----------|-----------|---------------------------|-------------|
| MT5_Data_Extraction | `projects\MT5_Data_Extraction\` | `C:\Quant\MT5_Data_Extraction` | Motor de datos M5 + ER Filter + Strategy Lab |
| TWF | `projects\TWF\` | `C:\Quant\TWF` | Pipeline A_E_P (Alphas, Ensembles, Portfolio) |
| BTC_ANALIST | `projects\BTC_ANALIST\` | `C:\Quant\BTC_ANALIST` | Análisis de BTC |
| GESTOR DE IA | `projects\GESTOR DE IA\` | `C:\Quant\GESTOR DE IA` | Gestor de inversión con LLM (GPT-5.2) |

## Entry Points por proyecto

### MT5_Data_Extraction
| Orden | Notebook | Ubicación |
|-------|----------|-----------|
| 1 | `01_MT5_DE_5M_V1.ipynb` | `projects\MT5_Data_Extraction\` |
| 2 | `02_ER_FILTER_5M_V4.ipynb` | `projects\MT5_Data_Extraction\` |
| 3 | `03_TREND_M5_Strategy_v1.ipynb` | `projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\` |
| 3 | `03_TREND_M5_Strategy_v2.ipynb` | `projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\` |
| 4 | `04_RANGE_M5_Strategy_v1.ipynb` | `projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\` |

### TWF
| Orden | Notebook | Ubicación |
|-------|----------|-----------|
| 1 | `A_E_P_v1.ipynb` | `projects\TWF\` |

### BTC_ANALIST
| Orden | Notebook | Ubicación |
|-------|----------|-----------|
| 1 | `BTC_ANALIST_v1.ipynb` | `projects\BTC_ANALIST\notebooks\` |

### GESTOR DE IA
| Orden | Notebook | Ubicación |
|-------|----------|-----------|
| 1 | `GESTOR.ipynb` | `projects\GESTOR DE IA\notebooks\` |

## Orden de ejecución del CORE (MT5 pipeline)

```
1) 01_MT5_DE_5M_V1.ipynb     — Ingesta y limpieza de datos M5 desde MetaTrader 5
2) 02_ER_FILTER_5M_V4.ipynb  — ER Filter: clasifica símbolos TREND/RANGE, genera baskets
3) Strategy Lab notebooks    — Features, señales, WFO, backtest, alpha design, deploy
     03_TREND_M5_Strategy_v1/v2  (TREND)
     04_RANGE_M5_Strategy_v1     (RANGE)
```

## Regla de ejecución

**Correr notebooks desde la raíz del proyecto** (`cd` al root del proyecto antes de ejecutar).

- NB1 y NB2: ejecutar desde `C:\Quant\projects\MT5_Data_Extraction\`
- Strategy Lab: ejecutar desde `C:\Quant\projects\MT5_Data_Extraction\ER_STRATEGY_LAB\notebooks\`
- TWF: ejecutar desde `C:\Quant\projects\TWF\`
- BTC_ANALIST: ejecutar desde `C:\Quant\projects\BTC_ANALIST\`
- GESTOR DE IA: ejecutar desde `C:\Quant\projects\GESTOR DE IA\`

## Estructura del workspace

```
C:\Quant\
├── .gitignore                 # Git ignore para monorepo
├── WORKSPACE_INDEX.md         # Este archivo
├── projects/                  # Raíz real de todos los proyectos
│   ├── MT5_Data_Extraction/   # 21 GB
│   ├── TWF/                   # 819 MB
│   ├── BTC_ANALIST/           # 1 GB
│   └── GESTOR DE IA/          # 1.1 GB
├── shared/
│   ├── audit/                 # Audit report + herramientas
│   └── workspace_docs/        # CLAUDE.md, PDFs, DOCXs
├── _archive/
│   ├── backup_20260207_1435/  # Snapshot pre-migración
│   ├── legacy/                # Contenido de 99_archive anterior
│   └── MIGRATION_LOG_*.md     # Logs de migración
├── _inbox/                    # Archivos no clasificados
├── MT5_Data_Extraction → projects\MT5_Data_Extraction  (junction)
├── TWF → projects\TWF  (junction)
├── BTC_ANALIST → projects\BTC_ANALIST  (junction)
└── GESTOR DE IA → projects\GESTOR DE IA  (junction)
```

## Notas importantes

- Las **junctions** garantizan que rutas antiguas (`C:\Quant\MT5_Data_Extraction\...`) sigan funcionando.
- **NO eliminar las junctions** sin antes verificar que ningún script/notebook referencia la ruta antigua.
- Los **path contracts** de NB1 y NB2 están CERRADOS y no deben modificarse.
