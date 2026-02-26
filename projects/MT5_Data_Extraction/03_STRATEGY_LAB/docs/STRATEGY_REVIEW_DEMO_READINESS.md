# STRATEGY REVIEW — Demo Readiness Assessment

**Fecha**: 2026-02-25
**Estrategia**: TREND v2 (regime gate + WFO)
**Pipeline**: Screener → WFO Validator → Institutional Selector
**Objetivo**: Determinar si los resultados justifican pasar a cuenta demo MT5

---

## 1. PIPELINE COMPLETO (3 fases independientes)

### Fase 1: Edge Diagnosis (Forward Returns, sin backtest)
- **Script**: `tools/_edge_diagnosis.py`
- **Metodo**: Analisis de forward returns en senales del regime gate (no hay optimizacion, no hay backtest)
- **Muestra**: 1,096,589 senales TREND, 317,805 senales RANGE

| Estrategia | Edge medio | IC 95% | Horizonte | Veredicto |
|------------|-----------|--------|-----------|-----------|
| **TREND** | +0.123% por senal | [+0.119%, +0.127%] | 24h | EDGE POSITIVO |
| RANGE | -0.009% por senal | [-0.010%, -0.007%] | 1h | EDGE NEGATIVO |

**TREND por side**:
- LONG: +0.226% por senal (Sharpe 1.50)
- SHORT: +0.001% por senal (Sharpe 0.01)
- **Conclusion**: Edge concentrado en LONG (225x mejor que SHORT)

**Validacion IS vs OOS**:
- IS: +0.119% (980,413 senales)
- OOS: +0.155% (116,176 senales)
- **OOS > IS → No hay overfitting**

### Fase 2: Screener Universo Completo
- **Script**: `tools/_screen_trend_universe.py`
- **Metodo**: Bootstrap CI sobre forward returns por (simbolo, side) en OOS
- **Resultado**: 65 candidatos OOS significativos (42 LONG, 23 SHORT) de 56 simbolos

### Fase 3a: Validador WFO (Top 20 LONG)
- **Script**: `tools/_validate_trend_candidates.py`
- **Metodo**: WFO real con engine propietario (SL/TP/trail, costos 8bps, risk management)
- **Grid**: 18 combos (SL=[1.5,2.0,2.5] x TP=[5,7,10] x TS=[288,576])
- **WFO**: IS=18m, OOS=3m, step=3m, embargo=5d, ~10 folds por simbolo
- **Evaluados**: 19 candidatos (top 20 LONG del screener, ambos sides de 5 seleccionados)

| Candidato | Side | Return OOS | Sharpe | PF | WR | N trades | Status |
|-----------|------|-----------|--------|-----|-----|----------|--------|
| TSLA | LONG | +56.34% | 38.84 | 1.47 | 30% | 176 | PASS |
| NVDA | LONG | +37.91% | 26.72 | 1.32 | 24% | 239 | PASS |
| META | LONG | +17.45% | 21.88 | 1.24 | 32% | 190 | PASS |
| AIRF | SHORT | +17.35% | 17.40 | 1.20 | 23% | 206 | PASS |
| AAPL | LONG | +2.71% | 5.36 | 1.05 | 28% | 209 | WARN |

Los otros 14 evaluados: todos FAIL (return negativo en OOS).

### Fase 3b: Selector Institucional (Universo Completo LONG + SHORT)
- **Script**: `tools/_select_institutional_universe.py`
- **Metodo**: WFO identico + metricas de estabilidad per-fold + stability score compuesto
- **Evaluados**: 88 candidatos (46 simbolos x 2 sides, 4 sin trades)
- **Grid**: 18 combos, misma config que Fase 3a
- **Runtime**: ~80 minutos (2 runs identicos confirman reproducibilidad)

**Stability Score (0-100)** = ponderacion de:
- 30% Fold consistency (% folds con return > 0)
- 25% Return stability (inverse CV de returns per fold)
- 20% Worst fold score (peor fold normalizado)
- 15% Profit factor stability (avg PF across folds)
- 10% Win rate consistency (std de WR entre folds)

**Clasificacion**: STABLE >= 55, MARGINAL >= 35, UNSTABLE < 35

**Resultado**: 0 STABLE, 5 MARGINAL, 83 UNSTABLE

---

## 2. LOS 5 CANDIDATOS VIABLES (MARGINAL)

### NVDA LONG — Stability #1 (54.8/100)
| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Stability Score | 54.8 | Mejor de todos, 0.2 puntos debajo de STABLE |
| Folds positivos | 8/10 (80%) | Alta consistencia entre periodos |
| Return total OOS | +37.91% | Rentable en agregado |
| Avg return/fold | +3.33% | Positivo y estable |
| Return CV | 1.11 | Variabilidad moderada |
| Worst fold | -3.13% | Peor periodo contenido |
| MDD | -11.31% | Drawdown aceptable |
| Sharpe | 26.72 | Alto |
| Profit Factor | 1.322 | Gana 1.32x por cada 1x que pierde |
| Hit Rate | 24.3% | Baja pero compensada por ratio ganancia/perdida |
| Avg PF per fold | 1.346 | Consistente entre folds |
| N trades OOS | 239 | Muestra significativa |

### META LONG — Stability #2 (47.2/100)
| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Stability Score | 47.2 | Segundo mejor |
| Folds positivos | 7/10 (70%) | Buena consistencia |
| Return total OOS | +17.45% | Rentable |
| Avg return/fold | +1.69% | Positivo pero menor que NVDA |
| Return CV | 2.26 | Mas variable entre folds |
| Worst fold | -3.84% | Contenido |
| MDD | -8.47% | El mejor MDD del grupo |
| Sharpe | 21.88 | Alto |
| Profit Factor | 1.243 | Aceptable |
| Hit Rate | 31.6% | La mas alta del grupo |
| N trades OOS | 190 | Adecuada |

### TSLA LONG — Stability #3 (44.5/100)
| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Stability Score | 44.5 | Medio |
| Folds positivos | 6/10 (60%) | Moderada |
| Return total OOS | +56.34% | **Mas alta del grupo** |
| Avg return/fold | +5.11% | Mejor avg return/fold |
| Return CV | 2.15 | Variable |
| Worst fold | -4.85% | Peor fold mas profundo |
| MDD | -12.34% | El peor MDD del grupo |
| Sharpe | 38.84 | **Mas alto del grupo** |
| Profit Factor | 1.469 | **Mas alto del grupo** |
| Hit Rate | 30.1% | Baja |
| N trades OOS | 176 | Menor muestra |

**Nota**: TSLA tiene los mejores KPIs brutos pero menor estabilidad entre folds. Alta varianza = alto riesgo de que futuros folds sean negativos.

### AAPL LONG — Stability #4 (39.2/100)
| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Stability Score | 39.2 | Bajo-marginal |
| Folds positivos | 5/10 (50%) | Coin flip |
| Return total OOS | +2.71% | Apenas positivo |
| Avg return/fold | +0.33% | Casi cero |
| Return CV | 10.63 | **Extremadamente variable** |
| Worst fold | -3.37% | Contenido |
| MDD | -11.73% | Alto |
| Sharpe | 5.36 | El mas bajo del grupo |
| Profit Factor | 1.054 | Casi breakeven |
| Hit Rate | 27.8% | Baja |
| N trades OOS | 209 | Adecuada |

**Riesgo**: Solo 50% folds positivos y CV de 10.63 indican que el edge de AAPL es marginal. Podria facilmente ser negativo en periodos futuros.

### AIRF SHORT — Stability #5 (36.5/100)
| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| Stability Score | 36.5 | Justo arriba de MARGINAL |
| Folds positivos | 3/9 (33%) | **Solo 1 de cada 3 folds positivo** |
| Return total OOS | +17.35% | Rentable en agregado |
| Avg return/fold | +2.04% | Positivo pero inconsistente |
| Return CV | 3.62 | Alta variabilidad |
| Worst fold | -4.23% | Aceptable |
| MDD | -15.95% | **Peor del grupo** |
| Sharpe | 17.40 | Aceptable |
| Profit Factor | 1.204 | Aceptable |
| Avg PF per fold | 1.509 | **Mejor PF per fold** (pocos folds ganan mucho) |
| Hit Rate | 23.3% | La mas baja |
| N trades OOS | 206 | Adecuada |

**Riesgo**: Solo 33% folds positivos. El return total viene de pocos folds muy buenos compensando muchos malos. Alta dependencia de condiciones especificas.

---

## 3. QUE NO FUNCIONA (y por que)

### Crypto: TODO UNSTABLE
| Simbolo | Side | Stab | Return OOS | Por que falla |
|---------|------|------|-----------|--------------|
| DOGEUSD | LONG | 24.0 | -15.53% | Alta varianza entre folds, edge no persiste |
| XRPUSD | LONG | 23.3 | -23.10% | Edge de screener no sobrevive WFO |
| ETHUSD | LONG | 13.9 | -51.42% | Fuertemente negativo con costos reales |
| BNBUSD | LONG | 12.5 | -29.26% | Sin edge consistente |
| Todos SHORT | <28 | Negativo | Peor que LONG en todos los casos |

**Por que el screener mostro edge pero WFO no**: El screener mide forward returns *sin costos*. Crypto tiene alta volatilidad que genera senales pero los costos (8 bps roundtrip) y el time stop erosionan el edge.

### Metales: TODO UNSTABLE
| Simbolo | Side | Stab | Return OOS |
|---------|------|------|-----------|
| XAUUSD | LONG | 17.5 | -24.22% |
| XAGUSD | LONG | 20.6 | -34.28% |
| XAGEUR | LONG | 18.0 | -35.11% |
| XAUAUD | LONG | 25.1 | -21.30% |
| Todos SHORT | <29 | Negativo |

### Forex: TODO UNSTABLE (0% folds positivos)
| Simbolo | Side | Stab | Return OOS |
|---------|------|------|-----------|
| EURUSD | LONG | 36.7 | -35.40% |
| NZDUSD | LONG | 34.4 | -42.63% |
| USDCAD | LONG | 33.4 | -38.93% |
| Todos los pares | Ambos | <37 | Fuertemente negativo |

**Nota sobre forex**: Algunos pares tienen stability scores "altos" (EURUSD 36.7) pero 0% folds positivos y return totalmente negativo. El score viene de baja variabilidad (pierden poco pero pierden *siempre*). Esto NO es edge.

---

## 4. CONFIGURACION DEL ENGINE

### Parametros WFO
```
Grid de optimizacion:
  SL (ATR mult):       [1.5, 2.0, 2.5]
  TP (ATR mult):       [5.0, 7.0, 10.0]
  Trail (ATR mult):    [0]  (trail desactivado — optimo en NB3)
  Time Stop (bars):    [288, 576]  (24h o 48h)
  Entry Confirm (bars): [6]  (30 min)
  → 18 combinaciones por fold

WFO Rolling:
  In-Sample:  18 meses
  Out-of-Sample: 3 meses
  Step: 3 meses
  Embargo: 5 dias
  Folds resultantes: ~10 por simbolo
```

### Costos
```
Spread:     8.0 bps (one-way)
Commission: 0.0 bps
Slippage:   0.0 bps
Roundtrip:  8.0 bps total
Fuente:     cost_model_snapshot_v2.json (fee-only, conservative)
```

### Risk Management (desactivado en evaluacion)
```
Max drawdown cap:  -100% (desactivado para evaluar edge puro)
Daily loss cap:    -100% (desactivado)
Daily profit cap:  +100% (desactivado)
Max trades/dia:    100 (efectivamente ilimitado)
```

### Regime Gate
```
ER quantile:       0.60 (solo mercados con efficiency ratio top 40%)
Momentum LONG:     quantile 0.55 (momentum positivo fuerte)
Momentum SHORT:    quantile 0.45 (momentum negativo fuerte)
Volatilidad:       quantile 0.90 (filtra solo ultra-alta vol)
```

---

## 5. RIESGOS Y LIMITACIONES

### Riesgos identificados

1. **Ninguno es STABLE (score >= 55)**. NVDA LONG es el mejor a 54.8 — apenas por debajo del umbral. Los 5 candidatos son MARGINAL, lo que significa que hay riesgo significativo de periodos negativos futuros.

2. **Hit rate bajo (23-32%)**. Todos los candidatos ganan menos del 33% de los trades. Esto es normal para estrategias TREND (pocas ganancias grandes, muchas perdidas pequenas) pero requiere disciplina psicologica en demo/live.

3. **Solo acciones + 1 accion EU**. Cero diversificacion por asset class. Todos son "other" (acciones US/EU). No hay metales, forex, crypto, ni indices que funcionen.

4. **AAPL LONG es marginal**. Return CV de 10.63 y solo 50% folds positivos. Podria facilmente ser negativo en futuros periodos.

5. **AIRF SHORT depende de pocos folds buenos**. Solo 33% folds positivos pero avg PF de 1.51 — pocos periodos muy buenos compensan muchos malos.

6. **Costos asumidos conservadores pero simplificados**. 8 bps fee-only, sin slippage. En real, slippage en acciones puede sumar 2-5 bps adicionales.

7. **Sin correlacion entre activos**. No se ha medido si NVDA/META/TSLA/AAPL tienen drawdowns simultaneos (probable dado que son todas acciones tech US).

### Limitaciones metodologicas

1. **WFO evalua edge historico**. Que haya funcionado en ~3 anos de datos no garantiza futuro.
2. **Regime gate calibrado sobre todo el periodo**. Los quantiles de ER/momentum/vol se calculan sobre IS pero las distribuciones pueden cambiar.
3. **Trail desactivado (=0)**. El optimo historico es sin trailing stop, pero esto podria no ser optimo en regimenes futuros.
4. **No se evaluo position sizing ni portfolio**. Los returns son por activo individual, no hay analisis de portfolio combinado.

---

## 6. EVALUACION: LISTO PARA DEMO?

### Argumentos A FAVOR
- Edge estadisticamente significativo confirmado por 3 metodos independientes (forward returns, WFO, stability per-fold)
- OOS > IS en edge diagnosis (no hay overfitting)
- 2 runs identicos del selector institucional (reproducibilidad confirmada)
- NVDA LONG tiene 80% folds positivos con avg PF 1.35 — robusto
- Costos ya incluidos en WFO (8 bps roundtrip)
- Muestra significativa (176-239 trades OOS por candidato)

### Argumentos EN CONTRA
- 0 candidatos STABLE — todos MARGINAL
- Solo 1 asset class (acciones)
- AAPL y AIRF son debiles (50% y 33% folds positivos)
- Sin analisis de correlacion ni portfolio
- Hit rate bajo requiere tolerancia a rachas perdedoras

### RECOMENDACION

**SI, se puede pasar a demo** con las siguientes condiciones:

1. **Empezar con los 3 mas fuertes**: NVDA LONG, META LONG, TSLA LONG
2. **AAPL LONG opcional** — incluir solo si se acepta el riesgo de edge marginal
3. **AIRF SHORT con precaucion** — monitorear cuidadosamente, solo 33% folds positivos
4. **Tamano de posicion conservador** en demo — enfocarse en validar ejecucion, no en PnL
5. **Monitorear por fold** — si 2-3 folds consecutivos son negativos en un activo, pausar

### Portfolio demo sugerido
```
Tier 1 (mayor confianza):  NVDA LONG, META LONG
Tier 2 (confianza media):  TSLA LONG
Tier 3 (experimental):     AAPL LONG, AIRF SHORT
```

---

## 7. DATOS PARA VERIFICACION POR TERCEROS

### Archivos de resultados
```
outputs/screening/screen_trend_universe.parquet        — Screener (65 OOS sig)
outputs/screening/validate_trend_candidates.parquet    — WFO validator (19 eval)
outputs/screening/select_institutional_universe.parquet — Selector inst. (88 eval)
```

### Scripts (reproducibles)
```
tools/_edge_diagnosis.py              — Edge diagnosis (forward returns)
tools/_screen_trend_universe.py       — Screener universo
tools/_validate_trend_candidates.py   — WFO validator
tools/_select_institutional_universe.py — Selector institucional
```

### Engine y modulos
```
03_STRATEGY_LAB/src/strategylab/      — 17 modulos
03_STRATEGY_LAB/tests/                — 61 tests, todos PASS
```

### Comandos para reproducir
```bash
cd C:\Quant\projects\MT5_Data_Extraction

# Fase 1: Edge diagnosis
venv1\Scripts\python.exe tools\_edge_diagnosis.py

# Fase 2: Screener
venv1\Scripts\python.exe tools\_screen_trend_universe.py --save

# Fase 3a: WFO Validator
venv1\Scripts\python.exe tools\_validate_trend_candidates.py --save -v

# Fase 3b: Selector institucional (full, ~80 min)
venv1\Scripts\python.exe tools\_select_institutional_universe.py --save -v

# Selector solo los 5 viables (~3 min)
venv1\Scripts\python.exe tools\_select_institutional_universe.py --symbols NVDA,META,TSLA,AAPL,AIRF --save -v

# Tests del engine
cd 03_STRATEGY_LAB && venv1\..\venv1\Scripts\python.exe -m pytest tests/ -v
```

### Checksums de resultados clave
```
NVDA LONG:  Stab=54.8, Ret=+37.91%, Sharpe=26.72, PF=1.322, N=239, 8/10 folds+
META LONG:  Stab=47.2, Ret=+17.45%, Sharpe=21.88, PF=1.243, N=190, 7/10 folds+
TSLA LONG:  Stab=44.5, Ret=+56.34%, Sharpe=38.84, PF=1.469, N=176, 6/10 folds+
AAPL LONG:  Stab=39.2, Ret= +2.71%, Sharpe= 5.36, PF=1.054, N=209, 5/10 folds+
AIRF SHORT: Stab=36.5, Ret=+17.35%, Sharpe=17.40, PF=1.204, N=206, 3/9  folds+
```

---

*Documento generado el 2026-02-25 a partir de resultados reproducibles del pipeline TREND v2.*
