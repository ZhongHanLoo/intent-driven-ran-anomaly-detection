"""Tests for src.intent.loop — the staged Loop-B metric-feedback driver
(bounded Plan-Act-Observe-Reflect behind a flag). FakeLLM
scripts the agents. The executor is real where cheap (integration) and
stubbed where only the loop logic is under test."""

import json
from pathlib import Path

import src.intent.loop as loop
from src.intent.llm_client import FakeLLM
from src.intent.loop import run_loop
from src.intent.registry import Registry

ART = Path(__file__).resolve().parents[1] / "artifacts"
REG = Registry(ART)


def _policy(persistence=1):
    return ('{"schema_version": 1, "action": "set_policy",'
            f' "policy": {{"model": "ae", "window": 3, "threshold_q": "default", "persistence": {persistence}}},'
            ' "intent_understood_as": "defend_unknown_attacks", "reason": "r"}')


POL_P1, POL_P3, POL_P5 = _policy(1), _policy(3), _policy(5)
APPROVE = ('{"schema_version": 1, "verdict": "approve",'
           ' "checked": ["intent_match", "tradeoff_direction", "knob_legality"], "critique": ""}')
REFUSE = '{"schema_version": 1, "action": "refuse", "intent_understood_as": null, "reason": "no"}'


def _stub_apply(monkeypatch):
    monkeypatch.setattr(loop, "apply_policy", lambda p, reg: {
        "fa_per_hour": 24.3 if p.persistence == 1 else 7.7, "dns_delay_s": 221.0,
        "gtpu_delay_s": 57.0, "dns_detected": True, "gtpu_detected": True,
        "mcc": 0.13, "fpr": 0.18, "fnr": 0.58, "macro_f1": 0.51})


def test_converges_on_fixed_point_and_feeds_observation(tmp_path):
    # iteration 2 re-emits the same policy -> converged, no re-execution
    fake = FakeLLM([POL_P1, APPROVE, POL_P1, APPROVE])
    log = tmp_path / "loop.jsonl"
    r = run_loop("defend against unknown attacks", fake, REG, ART,
                 max_iterations=3, runlog=log)
    assert r.status == "converged" and r.iterations == 2
    assert r.final_policy["persistence"] == 1
    assert r.final_metrics["fa_per_hour"] > 0          # real executed metrics
    assert r.trajectory[1]["converged"] is True
    it2_user = fake.calls[2]["messages"][0]["content"]  # iteration-2 compiler prompt
    assert "[OBSERVED METRICS]" in it2_user and '"fa_per_hour"' in it2_user
    assert '"persistence": 1' in it2_user               # current policy visible
    assert len(log.read_text().splitlines()) == 3       # 2 iterations + summary


def test_refines_then_converges(monkeypatch):
    _stub_apply(monkeypatch)
    fake = FakeLLM([POL_P1, APPROVE, POL_P3, APPROVE, POL_P3, APPROVE])
    r = run_loop("defend against unknown attacks", fake, REG, ART, max_iterations=3)
    assert r.status == "converged"
    assert [t["policy"]["persistence"] for t in r.trajectory] == [1, 3, 3]
    assert r.final_policy["persistence"] == 3
    assert r.final_metrics["fa_per_hour"] == 7.7        # the refined policy's metrics


def test_budget_exhausted_when_policies_keep_changing(monkeypatch):
    _stub_apply(monkeypatch)
    fake = FakeLLM([POL_P1, APPROVE, POL_P3, APPROVE, POL_P5, APPROVE])
    r = run_loop("defend against unknown attacks", fake, REG, ART, max_iterations=3)
    assert r.status == "max_iterations" and r.iterations == 3
    assert r.final_policy["persistence"] == 5


def test_halt_mid_loop_keeps_last_good_policy():
    fake = FakeLLM([POL_P1, APPROVE, REFUSE])
    r = run_loop("defend against unknown attacks", fake, REG, ART, max_iterations=3)
    assert r.status == "halted_refusal"
    assert r.final_policy["persistence"] == 1           # rollback semantics: keep it
    assert r.iterations == 1


def test_halt_on_first_iteration_returns_no_policy():
    r = run_loop("block all traffic", FakeLLM([REFUSE]), REG, ART, max_iterations=3)
    assert r.status == "halted_refusal"
    assert r.final_policy is None and r.trajectory == []


def test_cli_loop_flag(tmp_path):
    from src.intent.cli import run_cli
    fake = FakeLLM([POL_P1, APPROVE, POL_P1, APPROVE])
    rec = run_cli(intent="defend against unknown attacks", client=fake, art_dir=ART,
                  runlog=tmp_path / "l.jsonl", execute=True, use_reviewer=True,
                  loop_iterations=3)
    assert rec["status"] == "policy"
    assert rec["loop"]["status"] == "converged" and rec["loop"]["iterations"] == 2
    assert rec["policy"]["persistence"] == 1
    lines = [json.loads(l) for l in (tmp_path / "l.jsonl").read_text().splitlines()]
    assert lines[-1]["loop"]["status"] == "converged"   # final CLI record wraps the loop
