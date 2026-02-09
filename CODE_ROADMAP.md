# Code Roadmap — Next Sprint

## Current state of code

| Project | Notebooks | Python modules | Status |
|---------|-----------|----------------|--------|
| MT5_Data_Extraction | 3 (.ipynb, working) | 0 .py | All logic in notebooks |
| ER_STRATEGY_LAB | 3 (.ipynb, working) | 0 .py | All logic in notebooks |
| TWF | 1 (.ipynb, working) | 86 lines across 3 modules | `io/binance.py`, `utils/config.py`, `utils/logging.py` |
| BTC_ANALIST | 1 (.ipynb, working) | 0 (12 empty stubs) | `src/` is scaffolding only |
| GESTOR DE IA | 1 (.ipynb, working) | 0 .py | All logic in notebook |

## Priority 1 — Extract reusable code from MT5 notebooks

**Why first**: MT5_Data_Extraction is the foundation of the entire pipeline. Its notebooks contain the most complex logic (data ingestion, ER computation, regime classification) that is currently monolithic and untestable.

**Tasks**:
1. Extract data-loading helpers from NB1 into `projects/MT5_Data_Extraction/src/mt5de/io/mt5_loader.py`
2. Extract parquet I/O utilities into `src/mt5de/io/parquet_utils.py`
3. Extract ER computation logic from NB2 into `src/mt5de/er/filter.py`
4. Extract regime classification into `src/mt5de/er/regimes.py`
5. Create `src/mt5de/utils/paths.py` — centralize path resolution (honor path contracts)
6. Add unit tests for pure functions (ER computation, regime thresholds)

**Risk**: Path contracts are CLOSED. The extraction must not change any path behavior. Notebooks call extracted functions but retain control of path setup.

## Priority 2 — Implement BTC_ANALIST stubs

**Why second**: The scaffolding exists (`src/` with 12 modules), the notebook works, but the stubs are empty. This is the easiest win to get testable code.

**Tasks**:
1. `src/indicators/mvrv.py` — MVRV Z-Score computation
2. `src/indicators/profit_addresses.py` — Profit address ratio
3. `src/cycles/btc_cycles.py` — Halving cycle detection
4. `src/backtest/btc_backtest.py` — Simple signal backtest
5. `src/utils/dates.py` — Date alignment helpers
6. `src/utils/plotting.py` — Reusable chart templates
7. `src/config/settings.py` — Centralized config
8. Refactor notebook to import from `src/` instead of inline code

## Priority 3 — TWF module completeness

**Why third**: TWF already has the best module structure. Extend it.

**Tasks**:
1. Extract pipeline cells from `A_E_P_v1.ipynb` into `src/twf/pipeline/` modules
2. Add `src/twf/stats/` implementations (currently just `placeholders.py`)
3. Add tests for `io/binance.py` (mock httpx calls)

## Priority 4 — GESTOR DE IA modularization

**Why last**: Single notebook, most logic depends on OpenAI API calls (hard to test without mocking, costs money).

**Tasks**:
1. Extract SIMFIN data fetching into `src/gestor/data/simfin_loader.py`
2. Extract portfolio optimization into `src/gestor/portfolio/optimizer.py`
3. Extract LLM interaction into `src/gestor/llm/client.py`
4. Add `.env.example` with placeholder keys

## Cross-cutting tasks

| Task | Effort | Impact |
|------|--------|--------|
| Add `pyproject.toml` per project (replace raw requirements.txt) | Low | Enables `pip install -e .` for src/ imports |
| Add `tests/` directories with pytest structure | Low | Foundation for CI |
| Expand CI to run pytest (after tests exist) | Low | Catch regressions |
| Add notebook output stripping (nbstripout) to pre-commit | Low | Prevent output diffs |
| Document data schemas (parquet column contracts) | Medium | Prevent silent breakage |

## Definition of "sprint done"

- [ ] At least one module extracted from MT5 notebooks with tests
- [ ] At least 3 BTC_ANALIST stubs implemented
- [ ] `pytest` runs in CI with >0 tests passing
- [ ] No new data files accidentally committed
