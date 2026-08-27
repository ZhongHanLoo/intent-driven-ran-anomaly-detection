"""Tests for src.intent.reviewer — verdict call + the two-agent loop
(approve / revise-then-approve / never-approve -> no_convergence)."""

from pathlib import Path

from src.intent.llm_client import FakeLLM
from src.intent.registry import Registry
from src.intent.reviewer import run_two_agent

ART = Path(__file__).resolve().parents[1] / "artifacts"
REG = Registry(ART)

POLICY_AE = ('{"schema_version": 1, "action": "set_policy",'
             ' "policy": {"model": "ae", "window": 3, "threshold_q": 0.995, "persistence": 1},'
             ' "intent_understood_as": "defend_unknown_attacks", "reason": "novel needs ae"}')
POLICY_TF = ('{"schema_version": 1, "action": "set_policy",'
             ' "policy": {"model": "transformer", "window": 3, "threshold_q": 0.99, "persistence": 1},'
             ' "intent_understood_as": "early_attack_detection", "reason": "fast + robust"}')
APPROVE = ('{"schema_version": 1, "verdict": "approve",'
           ' "checked": ["intent_match", "tradeoff_direction", "knob_legality"], "critique": ""}')
REVISE = ('{"schema_version": 1, "verdict": "revise",'
          ' "checked": ["intent_match"], "critique": "use the ae for novel attacks"}')


def test_approve_first_round():
    # one client serves both agents: compiler answer, then reviewer verdict
    r = run_two_agent("defend against unknown attacks", FakeLLM([POLICY_AE, APPROVE]), REG, ART)
    assert r.status == "policy" and r.rounds == 1 and r.output.policy.model == "ae"


def test_revise_then_approve():
    fake = FakeLLM([POLICY_TF, REVISE, POLICY_AE, APPROVE])
    r = run_two_agent("defend against unknown attacks", fake, REG, ART)
    assert r.status == "policy" and r.rounds == 2 and r.output.policy.model == "ae"


def test_never_approve_is_no_convergence():
    fake = FakeLLM([POLICY_TF, REVISE, POLICY_TF, REVISE, POLICY_TF, REVISE])
    r = run_two_agent("defend against unknown attacks", fake, REG, ART, max_rounds=3)
    assert r.status == "no_convergence" and r.output is None and r.rounds == 3


def test_compiler_refusal_skips_review():
    refuse = '{"schema_version": 1, "action": "refuse", "intent_understood_as": null, "reason": "out of scope"}'
    r = run_two_agent("block all traffic", FakeLLM([refuse]), REG, ART)
    assert r.status == "refusal" and r.rounds == 1


def test_reviewer_disabled_passthrough():
    r = run_two_agent("defend against unknown attacks", FakeLLM([POLICY_AE]), REG, ART, use_reviewer=False)
    assert r.status == "policy" and r.rounds == 1


def test_invalid_reviewer_output_gets_repair_then_generic_critique():
    # reviewer emits junk twice -> generic critique -> compiler retries -> approve
    fake = FakeLLM([POLICY_TF, "junk", "junk", POLICY_AE, APPROVE])
    r = run_two_agent("defend against unknown attacks", fake, REG, ART)
    assert r.status == "policy" and r.rounds == 2 and r.output.policy.model == "ae"
