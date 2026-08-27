"""Tests for src.intent.capability_card — auto-generated evidence text."""

from pathlib import Path

from src.intent.capability_card import build_card

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_card_contains_measured_anchors():
    card = build_card(ART)
    for anchor in ["transformer", "ae", "GTP-U", "0.995", "persistence",
                   "false alarms", "novel", "window"]:
        assert anchor.lower() in card.lower()


def test_card_is_deterministic_and_bounded():
    a, b = build_card(ART), build_card(ART)
    assert a == b
    assert a.count("\n") <= 60          # stays compact for the prompt
    assert len(a) < 6000                # ~1.5k tokens ceiling


def test_card_quotes_real_operating_points():
    import pandas as pd
    op = pd.read_csv(ART / "operating_points.csv")
    thr = op[(op.model == "ae") & (op.window == 3) & (op.q == 0.995)].threshold.iloc[0]
    assert f"{thr:.4f}"[:5] in build_card(ART)  # the AE w3 q0.995 value appears


def test_card_has_default_threshold_reference_rows():
    card = build_card(ART)
    # the temporal-replay reference block lists all four models at their defaults
    block = card[card.find("TEMPORAL REPLAY"):]
    for m in ("lstm", "tcn", "transformer", "ae"):
        assert f"- {m}:" in block
