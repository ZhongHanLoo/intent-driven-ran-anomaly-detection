"""Tests for src.calibrate — written FIRST (TDD). Operating points for the
Phase-2 threshold knob must come from BENIGN VALIDATION scores (deployable),
never from test scores."""

import numpy as np
import pytest

from src.calibrate import benign_quantile_table


def test_benign_quantile_table_uses_benign_scores_only():
    # benign scores ramp 0..1; attack scores sit far above at 5.0 — if attacks
    # leaked into the quantile computation, thresholds would inflate visibly.
    scores = np.r_[np.linspace(0.0, 1.0, 101), np.full(50, 5.0)]
    y = np.r_[np.zeros(101, int), np.ones(50, int)]
    t = benign_quantile_table(scores, y, qs=(0.90, 0.99))
    assert list(t) == [0.90, 0.99]  # insertion order preserved
    assert t[0.90] == pytest.approx(0.90, abs=0.02)
    assert t[0.99] == pytest.approx(0.99, abs=0.02)


def test_benign_quantile_table_default_ladder_is_monotonic():
    rng = np.random.default_rng(7)
    scores = rng.random(10_000)
    y = np.zeros(10_000, int)
    t = benign_quantile_table(scores, y)
    qs = list(t)
    assert qs == sorted(qs) and 0.90 in qs and 0.995 in qs
    vals = list(t.values())
    assert all(a <= b for a, b in zip(vals, vals[1:]))  # thresholds rise with q
