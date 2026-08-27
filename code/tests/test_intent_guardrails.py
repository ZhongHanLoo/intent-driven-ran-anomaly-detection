"""Adversarial tests for src.intent.guardrails — gates G0..G5."""

from pathlib import Path

import pytest

from src.intent.guardrails import validate_compiler_output, validate_reviewer_output
from src.intent.registry import Registry

ART = Path(__file__).resolve().parents[1] / "artifacts"
REG = Registry(ART)
CANONICAL = {"early_attack_detection", "minimize_false_alarms",
             "balanced_operation", "defend_unknown_attacks"}

GOOD = ('{"schema_version": 1, "action": "set_policy",'
        ' "policy": {"model": "ae", "window": 3, "threshold_q": 0.995, "persistence": 1},'
        ' "intent_understood_as": "defend_unknown_attacks", "reason": "novel attacks need the ae"}')


def test_good_output_passes_all_gates():
    r = validate_compiler_output(GOOD, REG, CANONICAL)
    assert r.ok and r.gate is None and r.output.policy.model == "ae"


def test_fenced_json_is_unwrapped():
    r = validate_compiler_output("```json\n" + GOOD + "\n```", REG, CANONICAL)
    assert r.ok


@pytest.mark.parametrize("text,gate", [
    ("not json at all", "parse"),
    ('{"a": 1} {"b": 2}', "parse"),
    ("x" * 5000, "parse"),
    ('{"schema_version": 1, "action": "set_policy", "policy": {"model": "bert", "window": 3, "threshold_q": 0.995, "persistence": 1}, "intent_understood_as": null, "reason": "r"}', "schema"),
    ('{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": 0.42, "persistence": 1}, "intent_understood_as": null, "reason": "r"}', "schema"),
    ('{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": 0.995, "persistence": 1}, "intent_understood_as": "become_root", "reason": "r"}', "consistency"),
])
def test_bad_outputs_land_on_the_right_gate(text, gate):
    r = validate_compiler_output(text, REG, CANONICAL)
    assert not r.ok and r.gate == gate


def test_prose_after_json_rejected():
    r = validate_compiler_output(GOOD + "\nHope this helps!", REG, CANONICAL)
    assert not r.ok and r.gate == "parse"


def test_reviewer_verdicts():
    ok = validate_reviewer_output('{"schema_version": 1, "verdict": "approve", "checked": ["intent_match"], "critique": ""}')
    assert ok.ok and ok.output.verdict == "approve"
    bad = validate_reviewer_output('{"schema_version": 1, "verdict": "revise", "checked": ["intent_match"], "critique": ""}')
    assert not bad.ok and bad.gate == "schema"


def test_registry_gate_fires_when_pair_missing(tmp_path):
    import pandas as pd
    pd.DataFrame([{"model": "lstm", "window": 3, "protocol": "temporal", "seed": 42,
                   "n_val": 1, "n_val_benign": 1, "default_threshold": 0.5,
                   "q": 0.995, "threshold": 0.1}]).to_csv(tmp_path / "operating_points.csv", index=False)
    (tmp_path / "lstm_w3_temporal_metrics.json").write_text('{"threshold": 0.5}')
    small = Registry(tmp_path)
    r = validate_compiler_output(GOOD, small, CANONICAL)  # GOOD names ae/w3 — absent here
    assert not r.ok and r.gate == "registry"
