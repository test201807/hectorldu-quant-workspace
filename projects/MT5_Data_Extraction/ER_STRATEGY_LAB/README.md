# ER_STRATEGY_LAB

Laboratorio de diseño de estrategias basado en los resultados del ER Filter (`02_ER_FILTER_5M_V4`).

## Notebooks

| Notebook | Estrategia | Descripción |
|----------|-----------|-------------|
| `03_TREND_M5_Strategy_v1` | TREND M5 | Pipeline completo: features → WFO → señales → backtests → alpha design → deploy config. Escribe a `artifacts/`. |
| `03_TREND_M5_Strategy_v2` | TREND M5 v2 | Pipeline refactorizado con run manifests v2. Escribe a `notebooks/outputs/trend_m5_strategy/v2/`. |
| `04_RANGE_M5_Strategy_v1` | RANGE M5 | Estrategia para regímenes laterales. |

## Estructura

```
ER_STRATEGY_LAB/
├── config/          6 YAMLs: project, data_contract, costs_model, wfo, risk, sessions
├── inputs/          Shortlists y diagnósticos del ER Filter (NB2)
├── notebooks/       Notebooks de estrategia + outputs de v2
├── artifacts/       Resultados centralizados de v1 (backtests, features, deploy, wfo)
├── research_logs/   Logs de runs con snapshots de config
└── docs/            Metodología y changelog
```

## Inputs requeridos

- Baskets y exports del ER Filter: `outputs/er_filter_5m/<RUN_ID>/baskets/` y `exports/`
- Shortlist: `inputs/shortlist/selected_symbols_TREND.json`
- Datos M5 limpios vía `data_contract.yaml`

## Convenciones de paths

- **v1**: `PATH_ARTIFACTS = ER_STRATEGY_LAB_ROOT / "artifacts"` (resuelto con `paths["artifacts"]`)
- **v2**: `OUTPUTS_ROOT = WORKDIR / "outputs" / "trend_m5_strategy" / "v2"` (relativo a `notebooks/`)
