# Metodología — ER Strategy Lab

## Pipeline general

1. **Selección de universo**: El ER Filter (NB2) clasifica símbolos en TREND/RANGE y produce baskets con scores de estabilidad y viabilidad económica.
2. **Ingestión de datos**: M5 OHLCV limpio, filtrado por `data_contract.yaml` (cobertura mínima, spread máximo).
3. **Feature engineering**: Features base (v11) + regime gates por fold WFO.
4. **Generación de señales**: Señales por símbolo y fold (`signals_base_v3/`).
5. **Walk-Forward Optimization (WFO)**: Folds definidos en `config/wfo.yaml`. Cada fold tiene train/test con embargo.
6. **Backtesting**: Motor v10 con policy sweep, quality sweep, calibración (v17) y overlay (v16).
7. **Alpha design**: Evaluación multi-horizonte, selección de candidatos.
8. **Deploy**: Configuraciones por símbolo para EA en `artifacts/deploy/per_symbol_configs_v10/`.

## Parámetros clave (configs/)

- `costs_model.yaml`: Modelo de costos de transacción (spread + comisiones).
- `risk.yaml`: Gestión de riesgo (sizing, drawdown limits).
- `sessions.yaml`: Ventanas horarias de trading por mercado.
- `wfo.yaml`: Estructura de folds walk-forward.

## Versionado de artefactos

- Features: `features_base_v11` (actual), señales: `signals_base_v3` (actual).
- Versiones anteriores archivadas en `99_archive/strategy_lab_old_features/`.
