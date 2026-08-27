"""Tests for src.intent.knobs — the single source of legal knob values."""

from src.intent.knobs import DEFAULT_THRESHOLD, MODELS, PERSISTENCES, THRESHOLD_QS, WINDOWS, knob_menu_text


def test_knob_constants_match_spec():
    assert MODELS == ("lstm", "tcn", "transformer", "ae")
    assert WINDOWS == (3, 5, 7)
    assert THRESHOLD_QS == (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)
    assert PERSISTENCES == (1, 2, 3, 5)


def test_schema_ladder_stays_in_sync():
    from src.intent import schema
    assert schema.THRESHOLD_QS == THRESHOLD_QS


def test_knob_menu_text_lists_every_value():
    menu = knob_menu_text()
    for m in MODELS:
        assert m in menu
    for w in WINDOWS:
        assert str(w) in menu
    for q in THRESHOLD_QS:
        assert str(q) in menu  # str(0.9) == "0.9" matches the f-string's rendering
    for p in PERSISTENCES:
        assert str(p) in menu
    assert f'"{DEFAULT_THRESHOLD}"' in menu
    assert menu.count("\n") <= 8  # tight compactness guard (currently 3 newlines)
