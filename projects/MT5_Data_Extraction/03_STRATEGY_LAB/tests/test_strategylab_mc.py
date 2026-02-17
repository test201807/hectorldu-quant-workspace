"""Tests for Monte Carlo simulation — synthetic data only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from strategylab.monte_carlo import (
    MCResult,
    block_bootstrap,
    iid_bootstrap,
    run_all_mc,
    stress_test,
)


def _sample_returns(n: int = 200, seed: int = 42) -> list[float]:
    """Generate deterministic sample trade returns."""
    import random
    rng = random.Random(seed)
    return [rng.gauss(0.0003, 0.005) for _ in range(n)]


class TestIIDBootstrap:

    def test_returns_mc_result(self):
        rets = _sample_returns()
        result = iid_bootstrap(rets, n_sims=50, seed=42)
        assert isinstance(result, MCResult)
        assert result.method == "iid_bootstrap"
        assert result.n_sims == 50

    def test_deterministic(self):
        rets = _sample_returns()
        r1 = iid_bootstrap(rets, n_sims=50, seed=42)
        r2 = iid_bootstrap(rets, n_sims=50, seed=42)
        assert r1.final_equities == r2.final_equities

    def test_percentiles_ordered(self):
        rets = _sample_returns()
        result = iid_bootstrap(rets, n_sims=100, seed=42)
        p = result.percentiles
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]

    def test_empty_returns(self):
        result = iid_bootstrap([], n_sims=10)
        assert result.n_sims == 0
        assert result.final_equities == []

    def test_n_sims_matches(self):
        rets = _sample_returns()
        result = iid_bootstrap(rets, n_sims=77, seed=1)
        assert len(result.final_equities) == 77
        assert len(result.max_drawdowns) == 77


class TestBlockBootstrap:

    def test_returns_mc_result(self):
        rets = _sample_returns()
        result = block_bootstrap(rets, block_size=10, n_sims=50, seed=42)
        assert isinstance(result, MCResult)
        assert result.method == "block_bootstrap"

    def test_block_size_affects_results(self):
        rets = _sample_returns()
        r1 = block_bootstrap(rets, block_size=5, n_sims=50, seed=42)
        r2 = block_bootstrap(rets, block_size=50, n_sims=50, seed=42)
        # Different block sizes should give different results
        assert r1.final_equities != r2.final_equities

    def test_percentiles_ordered(self):
        rets = _sample_returns()
        result = block_bootstrap(rets, n_sims=100, seed=42)
        p = result.percentiles
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]


class TestStressTest:

    def test_returns_mc_result(self):
        rets = _sample_returns()
        result = stress_test(rets, n_sims=50, seed=42)
        assert isinstance(result, MCResult)
        assert result.method == "stress"

    def test_stress_worse_than_iid(self):
        """Stressed equity should be worse (lower median) than IID."""
        rets = _sample_returns(n=200, seed=42)
        iid = iid_bootstrap(rets, n_sims=200, seed=42)
        stressed = stress_test(rets, cost_multiplier=3.0, n_sims=200, seed=42)
        assert stressed.percentiles["p50"] <= iid.percentiles["p50"]


class TestRunAllMC:

    def test_returns_three_results(self):
        rets = _sample_returns()
        results = run_all_mc(rets, n_sims=50, seed=42)
        assert len(results) == 3
        methods = {r.method for r in results}
        assert methods == {"iid_bootstrap", "block_bootstrap", "stress"}

    def test_all_have_data(self):
        rets = _sample_returns()
        results = run_all_mc(rets, n_sims=30, seed=42)
        for r in results:
            assert len(r.final_equities) == 30
            assert len(r.max_drawdowns) == 30
