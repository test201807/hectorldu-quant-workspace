"""Single source of truth — FTMO challenge ruleset.

Usage:
    from challenge_config import CAPITAL, DAILY_MAX, MAX_LOSS, TARGET, RISK_USD
    from challenge_config import RULESET  # full dict

Ruleset file: 03_STRATEGY_LAB/configs/challenge_ruleset.json
"""
import json
from pathlib import Path

_RULESET_PATH = Path(__file__).parent.parent / "03_STRATEGY_LAB" / "configs" / "challenge_ruleset.json"
RULESET = json.loads(_RULESET_PATH.read_text(encoding="utf-8"))

CAPITAL      = RULESET["starting_balance"]       # 10_000
DAILY_MAX    = RULESET["daily_max_loss"]          #    500  (5%)
MAX_LOSS     = RULESET["max_loss"]                #  1_000  (10%)
TARGET       = RULESET["profit_target"]           #    500  (5%)
MIN_DAYS     = RULESET["min_trading_days"]        #      2
RISK_PCT     = RULESET["risk_per_trade_pct"]      # 0.0025  (0.25%)
RISK_USD     = RULESET["risk_per_trade_usd"]      #     25
ACCOUNT_TYPE = RULESET["account_type"]            # "standard"
CB_PCT       = RULESET["circuit_breaker_pct"]     #    1.0
