# Path Contract — MT5_Data_Extraction

## Estructura del proyecto

```
MT5_Data_Extraction/
  01_DATA_EXTRACTION/
    notebooks/01_MT5_DE_5M_V1.ipynb        NB1: descarga y limpieza M5
  02_ER_FILTER/
    notebooks/02_ER_FILTER_5M_V4.ipynb      NB2: filtro ER/PD + baskets
  03_STRATEGY_LAB/
    notebooks/
      03_TREND_M5_Strategy_v2.ipynb         NB3: backtest TREND
      04_RANGE_M5_Strategy_v1.ipynb         NB4: backtest RANGE
    src/strategylab/                         Modulo Python (17 archivos)
    tests/                                   Tests (61 tests)
    configs/strategylab.yaml                 Config CLI
  shared/
    contracts/path_contract.py               UNICO contrato de rutas
  data/                                      Datos compartidos (NO en git)
    bulk_data/m5_raw/                        Hive raw (NB1 output)
    historical_data/m5_clean/                Hive gold (NB1 output)
    metadata/                                Metadata y manifests
    processed_data/                          Datos procesados
    backups/                                 Respaldos ZIP
    logs/                                    Logs de ejecucion
  outputs/                                   Resultados de runs (NO en git)
    er_filter_5m/<RUN_ID>/                   NB2 runs (baskets/, diagnostics/, etc.)
    trend_v2/<RUN_ID>/                       NB3 runs (37 artefactos por run)
    range_v1/<RUN_ID>/                       NB4 runs (37 artefactos por run)
  config/
    er_filter_5m.json                        Config persistente del filtro
  tools/
    validate_paths_static.py                 Validador estatico de rutas
  reports/
    paths_audit_report.md                    Reporte del validador
  _trash_review/                             Archivos obsoletos (revision pendiente)
  venv1/                                     Entorno virtual (Python 3.11.9)
  requirements.txt
```

## Contrato de rutas (path_contract.py)

**Ubicacion**: `shared/contracts/path_contract.py`

**Regla fundamental**: Ningun notebook debe hardcodear `C:\Quant\...` ni construir
rutas absolutas manualmente. Todas las rutas se derivan de `PROJECT_ROOT` via
las funciones del contrato.

### Deteccion de PROJECT_ROOT

Prioridad (deterministica):
1. Env var: `MT5_PROJECT_ROOT` o `MT5_DE_PROJECT_ROOT`
2. `__file__` hint: `shared/contracts/` -> parent.parent
3. Walk up desde CWD buscando directorio `mt5_data_extraction`
4. Walk up desde CWD buscando markers: `data/` + `outputs/`
5. Fallback: CWD

### Funciones disponibles

Todas aceptan `project_root: Path | None = None` opcional.

| Funcion | Ruta canonica |
|---------|---------------|
| `detect_project_root()` | PROJECT_ROOT |
| `data_root()` | PROJECT_ROOT/data |
| `m5_clean_dir()` | data/historical_data/m5_clean |
| `m5_raw_dir()` | data/bulk_data/m5_raw |
| `metadata_dir()` | data/metadata |
| `processed_data_dir()` | data/processed_data |
| `config_dir()` | PROJECT_ROOT/config |
| `outputs_root()` | PROJECT_ROOT/outputs |
| `nb2_outputs_dir()` | outputs/er_filter_5m |
| `trend_outputs_dir()` | outputs/trend_v2 |
| `range_outputs_dir()` | outputs/range_v1 |
| `strategy_lab_root()` | PROJECT_ROOT/03_STRATEGY_LAB |
| `nb2_latest_run_dir()` | Ultimo run de NB2 |
| `nb2_basket(strategy)` | Basket parquet del ultimo run NB2 |
| `m5_data_dir()` | Primer directorio M5 con datos (clean > raw) |
| `nb1_notebook_dir()` | 01_DATA_EXTRACTION/notebooks |
| `nb2_notebook_dir()` | 02_ER_FILTER/notebooks |
| `nb3_notebook_dir()` | 03_STRATEGY_LAB/notebooks |

### Como importar desde cualquier notebook

```python
import sys
from pathlib import Path

for _p in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
    _contract = _p / "shared" / "contracts" / "path_contract.py"
    if _contract.exists():
        if str(_contract.parent) not in sys.path:
            sys.path.insert(0, str(_contract.parent))
        break

import path_contract

PROJECT_ROOT = path_contract.detect_project_root()
DATA_ROOT = path_contract.data_root(PROJECT_ROOT)
```

## Pipeline de datos

```
NB1 (Extraction)
  Escribe: data/bulk_data/m5_raw/, data/historical_data/m5_clean/, data/metadata/
     |
     v
NB2 (ER Filter)
  Lee: data/ (M5, metadata)
  Escribe: outputs/er_filter_5m/<RUN_ID>/ (baskets, diagnostics, metrics, etc.)
     |
     v
NB3 (TREND)                    NB4 (RANGE)
  Lee: data/m5_clean            Lee: data/m5_clean
       outputs/er_filter_5m          outputs/er_filter_5m
         (basket_trend_core)           (basket_range_core)
  Escribe: outputs/trend_v2/   Escribe: outputs/range_v1/
```

## Variables de entorno soportadas

| Variable | Notebook | Funcion |
|----------|----------|---------|
| `MT5_PROJECT_ROOT` | Todos | Override PROJECT_ROOT |
| `MT5_DE_PROJECT_ROOT` | Todos | Override PROJECT_ROOT (alias) |
| `MT5_DE_DATA_ROOT` | NB1 | Override DATA_ROOT |
| `TREND_M5_ROOT` | NB3 | Override PROJECT_ROOT |
| `TREND_M5_OUTPUTS_ROOT` | NB3 | Override directorio de outputs |
| `TREND_M5_RUN_ID` | NB3 | Forzar RUN_ID |
| `TREND_M5_M5_CLEAN_DIR` | NB3 | Forzar directorio M5 |
| `RANGE_M5_ROOT` | NB4 | Override PROJECT_ROOT |
| `RANGE_M5_OUTPUTS_ROOT` | NB4 | Override directorio de outputs |
| `RANGE_M5_RUN_ID` | NB4 | Forzar RUN_ID |

## Validacion

Ejecutar el validador estatico:
```bash
python tools/validate_paths_static.py
```

Genera `reports/paths_audit_report.md` con estado de cada notebook y directorio.
