"""Patch NB3 Cell 16 (ab67bb0e): replace hardcoded challenge params with ruleset JSON read.

Changes:
- 6 hardcoded constants → read from challenge_ruleset.json
- Add ACCOUNT_TYPE, _CB_THRESHOLD variables
- Add consistency assertions
- Update circuit breaker to use _CB_THRESHOLD
- Add account_type + circuit_breaker_pct to challenge_result["challenge"] dict
- Update header comment with new values
"""
import json
from pathlib import Path

NB3 = Path(r"C:\Quant\projects\MT5_Data_Extraction\03_STRATEGY_LAB\notebooks\03_TREND_M5_Strategy_v2.ipynb")


def ar(src: str, old: str, new: str, label: str) -> str:
    assert old in src, f"ASSERT FAIL [{label}]:\n  {old!r}"
    return src.replace(old, new)


nb = json.load(open(NB3, encoding="utf-8"))

checks = 0
for cell in nb["cells"]:
    if cell.get("id") != "ab67bb0e":
        continue

    src = "".join(cell.get("source", []))

    # 1. Update header comment
    src = ar(src,
        "# Challenge rules: daily -$1,250 / total -$2,500 / target +$1,250 / min 2 days",
        "# Challenge rules: daily -$500 / total -$1,000 / target +$500 / min 2 days (from ruleset)",
        "header-comment")
    checks += 1

    # 2. Replace hardcoded constants block with ruleset read + assertions
    OLD_BLOCK = (
        "# ── Challenge params (prop-firm exam) ──\n"
        "CHALLENGE_CAPITAL        = 25_000\n"
        "CHALLENGE_DAILY_MAX_LOSS = 1_250   # USD\n"
        "CHALLENGE_TOTAL_MAX_LOSS = 2_500   # USD\n"
        "CHALLENGE_PROFIT_TARGET  = 1_250   # USD\n"
        "CHALLENGE_MIN_DAYS       = 2\n"
        "RISK_PER_TRADE_USD       = 75     # optimal from sweep (worst fold DD = -$1,955, no violations)"
    )
    NEW_BLOCK = (
        "# ── Challenge params — single source of truth (ruleset JSON) ──\n"
        "# CWD durante ejecución via nbconvert = 03_STRATEGY_LAB/notebooks/\n"
        "_ruleset_path = Path.cwd().parent / \"configs\" / \"challenge_ruleset.json\"\n"
        "_ruleset = json.loads(_ruleset_path.read_text(encoding=\"utf-8\"))\n"
        "CHALLENGE_CAPITAL        = _ruleset[\"starting_balance\"]\n"
        "CHALLENGE_DAILY_MAX_LOSS = _ruleset[\"daily_max_loss\"]\n"
        "CHALLENGE_TOTAL_MAX_LOSS = _ruleset[\"max_loss\"]\n"
        "CHALLENGE_PROFIT_TARGET  = _ruleset[\"profit_target\"]\n"
        "CHALLENGE_MIN_DAYS       = _ruleset[\"min_trading_days\"]\n"
        "RISK_PER_TRADE_USD       = _ruleset[\"risk_per_trade_usd\"]\n"
        "ACCOUNT_TYPE             = _ruleset[\"account_type\"]\n"
        "_CB_THRESHOLD            = CHALLENGE_DAILY_MAX_LOSS * _ruleset.get(\"circuit_breaker_pct\", 1.0)\n"
        "# ── Consistency assertions ──\n"
        "assert CHALLENGE_CAPITAL > 0, \"capital must be positive\"\n"
        "assert 0 < CHALLENGE_DAILY_MAX_LOSS <= CHALLENGE_TOTAL_MAX_LOSS < CHALLENGE_CAPITAL\n"
        "assert 0 < CHALLENGE_PROFIT_TARGET <= CHALLENGE_CAPITAL, \"target out of range\"\n"
        "assert RISK_PER_TRADE_USD < CHALLENGE_DAILY_MAX_LOSS, \"risk/trade >= daily limit (suicida)\"\n"
        "assert ACCOUNT_TYPE in (\"standard\", \"swing\"), f\"unknown account_type: {ACCOUNT_TYPE!r}\"\n"
        "print(f\"[Celda 16] Ruleset: ${CHALLENGE_CAPITAL:,} capital | daily-${CHALLENGE_DAILY_MAX_LOSS} \"\n"
        "      f\"| total-${CHALLENGE_TOTAL_MAX_LOSS} | target+${CHALLENGE_PROFIT_TARGET} \"\n"
        "      f\"| risk=${RISK_PER_TRADE_USD}/trade | type={ACCOUNT_TYPE}\")"
    )
    src = ar(src, OLD_BLOCK, NEW_BLOCK, "constants-block")
    checks += 1

    # 3. Update circuit breaker threshold
    src = ar(src,
        "if day[\"pnl_usd\"] <= -CHALLENGE_DAILY_MAX_LOSS:",
        "if day[\"pnl_usd\"] <= -_CB_THRESHOLD:",
        "circuit-breaker")
    checks += 1

    # 4. Add account_type + circuit_breaker_pct to challenge_result["challenge"] dict
    src = ar(src,
        '"challenge": {\n'
        '                "initial_capital": CHALLENGE_CAPITAL,\n'
        '                "daily_max_loss_usd": CHALLENGE_DAILY_MAX_LOSS,\n'
        '                "total_max_loss_usd": CHALLENGE_TOTAL_MAX_LOSS,\n'
        '                "profit_target_usd": CHALLENGE_PROFIT_TARGET,\n'
        '                "min_trading_days": CHALLENGE_MIN_DAYS,\n'
        '            },',
        '"challenge": {\n'
        '                "account_type": ACCOUNT_TYPE,\n'
        '                "initial_capital": CHALLENGE_CAPITAL,\n'
        '                "daily_max_loss_usd": CHALLENGE_DAILY_MAX_LOSS,\n'
        '                "total_max_loss_usd": CHALLENGE_TOTAL_MAX_LOSS,\n'
        '                "profit_target_usd": CHALLENGE_PROFIT_TARGET,\n'
        '                "min_trading_days": CHALLENGE_MIN_DAYS,\n'
        '                "circuit_breaker_pct": _ruleset.get("circuit_breaker_pct", 1.0),\n'
        '            },',
        "challenge-result-dict")
    checks += 1

    # 5. Update overlay_snapshot "challenge" section
    src = ar(src,
        '"challenge": {\n'
        '            "capital": CHALLENGE_CAPITAL,\n'
        '            "daily_max_loss": CHALLENGE_DAILY_MAX_LOSS,\n'
        '            "total_max_loss": CHALLENGE_TOTAL_MAX_LOSS,\n'
        '            "profit_target": CHALLENGE_PROFIT_TARGET,\n'
        '            "risk_per_trade": RISK_PER_TRADE_USD,\n'
        '        },',
        '"challenge": {\n'
        '            "account_type": ACCOUNT_TYPE,\n'
        '            "capital": CHALLENGE_CAPITAL,\n'
        '            "daily_max_loss": CHALLENGE_DAILY_MAX_LOSS,\n'
        '            "total_max_loss": CHALLENGE_TOTAL_MAX_LOSS,\n'
        '            "profit_target": CHALLENGE_PROFIT_TARGET,\n'
        '            "risk_per_trade": RISK_PER_TRADE_USD,\n'
        '        },',
        "snap-challenge-dict")
    checks += 1

    cell["source"] = [src]
    print(f"Cell 16 (ab67bb0e): {checks} patches OK")
    break

assert checks == 5, f"Expected 5 checks, got {checks}"

json.dump(nb, open(NB3, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"NB3 saved. {checks}/5 checks OK")
