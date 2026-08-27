"""Tests for src.intent.registry — (model, window) -> artifacts, thresholds."""

import json
from pathlib import Path

import pytest

from src.intent.registry import Registry

ART = Path(__file__).resolve().parents[1] / "artifacts"


@pytest.fixture(scope="module")
def reg():
    return Registry(ART)


def test_all_twelve_temporal_pairs_present(reg):
    assert reg.pairs() == {(m, w) for m in ("lstm", "tcn", "transformer", "ae") for w in (3, 5, 7)}


def test_resolve_quantile_matches_operating_points_csv(reg):
    import pandas as pd
    op = pd.read_csv(ART / "operating_points.csv")
    row = op[(op.model == "transformer") & (op.window == 3) & (op.q == 0.995)].iloc[0]
    assert reg.resolve_threshold("transformer", 3, 0.995) == pytest.approx(row.threshold, rel=1e-9)


def test_resolve_default_uses_metrics_json(reg):
    stored = json.load(open(ART / "ae_w3_temporal_metrics.json"))["threshold"]
    assert reg.resolve_threshold("ae", 3, "default") == pytest.approx(stored, rel=1e-9)
    assert reg.resolve_threshold("lstm", 3, "default") == 0.5


def test_unknown_pair_or_q_raises(reg):
    with pytest.raises(KeyError):
        reg.score_archive("lstm", 9)
    with pytest.raises(KeyError):
        reg.resolve_threshold("lstm", 3, 0.42)


def test_score_archive_paths_exist(reg):
    for m, w in reg.pairs():
        assert reg.score_archive(m, w).exists()


def test_full_ladder_completeness(reg):
    import pandas as pd
    op = pd.read_csv(ART / "operating_points.csv")
    for r in op.itertuples():
        got = reg.resolve_threshold(r.model, r.window, r.q)
        assert got == pytest.approx(r.threshold, rel=1e-9), \
            f"mismatch at ({r.model}, w{r.window}, q={r.q})"


def test_metrics_carry_threshold_for_all_pairs(reg):
    for m, w in reg.pairs():
        assert "threshold" in reg.metrics(m, w)


def test_pairs_returns_a_copy(reg):
    s = reg.pairs()
    s.add(("fake", 0))
    assert ("fake", 0) not in reg.pairs()
