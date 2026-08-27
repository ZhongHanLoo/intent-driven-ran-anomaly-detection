"""Tests for src.intent.schema — written FIRST (TDD). Strict shapes for the
Compiler output (set_policy | refuse) and the Reviewer verdict."""

import pytest
from pydantic import ValidationError

from src.intent.schema import ReviewerVerdict, parse_compiler_output, parse_reviewer_verdict


def test_set_policy_parses_and_types():
    out = parse_compiler_output(
        '{"schema_version": 1, "action": "set_policy",'
        ' "policy": {"model": "transformer", "window": 3, "threshold_q": 0.995, "persistence": 1},'
        ' "intent_understood_as": "early_attack_detection", "reason": "r"}')
    assert out.action == "set_policy"
    assert out.policy.model == "transformer" and out.policy.threshold_q == 0.995


def test_threshold_q_accepts_default_literal():
    out = parse_compiler_output(
        '{"schema_version": 1, "action": "set_policy",'
        ' "policy": {"model": "ae", "window": 5, "threshold_q": "default", "persistence": 3},'
        ' "intent_understood_as": null, "reason": "r"}')
    assert out.policy.threshold_q == "default"


@pytest.mark.parametrize("bad", [
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "gpt", "window": 3, "threshold_q": 0.995, "persistence": 1}, "intent_understood_as": null, "reason": "r"}',
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 4, "threshold_q": 0.995, "persistence": 1}, "intent_understood_as": null, "reason": "r"}',
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": 0.42, "persistence": 1}, "intent_understood_as": null, "reason": "r"}',
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": 0.995, "persistence": 4}, "intent_understood_as": null, "reason": "r"}',
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": 0.995, "persistence": 1}, "intent_understood_as": null, "reason": "r", "extra": 1}',
    '{"schema_version": 1, "action": "refuse", "intent_understood_as": null, "reason": ""}',
    '{"schema_version": 2, "action": "refuse", "intent_understood_as": null, "reason": "r"}',
    '{"schema_version": 1, "action": "set_policy", "policy": {"model": "lstm", "window": 3, "threshold_q": "0.995", "persistence": 1}, "intent_understood_as": null, "reason": "r"}',
])
def test_bad_compiler_outputs_rejected(bad):
    with pytest.raises(ValidationError):
        parse_compiler_output(bad)


def test_refusal_parses():
    out = parse_compiler_output(
        '{"schema_version": 1, "action": "refuse", "intent_understood_as": null, "reason": "out of scope"}')
    assert out.action == "refuse"


def test_reviewer_verdict_revise_requires_critique():
    ReviewerVerdict.model_validate_json(
        '{"schema_version": 1, "verdict": "approve", "checked": ["intent_match"], "critique": ""}')
    with pytest.raises(ValidationError):
        ReviewerVerdict.model_validate_json(
            '{"schema_version": 1, "verdict": "revise", "checked": ["intent_match"], "critique": ""}')


def test_reviewer_verdict_empty_checked_rejected():
    with pytest.raises(ValidationError):
        ReviewerVerdict.model_validate_json(
            '{"schema_version": 1, "verdict": "approve", "checked": [], "critique": ""}')


def test_set_policy_requires_intent_understood_as():
    with pytest.raises(ValidationError):
        parse_compiler_output(
            '{"schema_version": 1, "action": "set_policy",'
            ' "policy": {"model": "lstm", "window": 3, "threshold_q": 0.995, "persistence": 1},'
            ' "reason": "r"}')
