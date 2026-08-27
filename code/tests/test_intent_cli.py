"""End-to-end CLI test through FakeLLM (no network)."""

import json
from pathlib import Path

from src.intent.cli import run_cli
from src.intent.llm_client import FakeLLM

ART = Path(__file__).resolve().parents[1] / "artifacts"

GOOD = ('{"schema_version": 1, "action": "set_policy",'
        ' "policy": {"model": "ae", "window": 3, "threshold_q": 0.995, "persistence": 1},'
        ' "intent_understood_as": "defend_unknown_attacks", "reason": "novel needs ae"}')
APPROVE = ('{"schema_version": 1, "verdict": "approve",'
           ' "checked": ["intent_match", "tradeoff_direction", "knob_legality"], "critique": ""}')


def test_cli_end_to_end_with_fake_llm(tmp_path):
    log = tmp_path / "runs.jsonl"
    record = run_cli(intent="defend against unknown attacks",
                     client=FakeLLM([GOOD, APPROVE]),
                     art_dir=ART, runlog=log, execute=True, use_reviewer=True)
    assert record["status"] == "policy"
    assert record["policy"]["model"] == "ae"
    assert record["metrics"]["fa_per_hour"] > 0
    logged = [json.loads(l) for l in log.read_text().splitlines()]
    from src.intent.prompting import PROMPT_VERSION
    assert logged and logged[-1]["status"] == "policy"
    assert logged[-1]["prompt_version"] == PROMPT_VERSION  # record carries the live version


def test_cli_records_per_run_token_delta(tmp_path):
    # topic-6 fix: one shared client across runs must not inflate per-run token
    # figures (the 07-04 smoke narrative summed cumulative counters -> ~64k
    # instead of the true ~24k). Records carry delta + running total.
    fake = FakeLLM([GOOD, APPROVE, GOOD, APPROVE])
    r1 = run_cli(intent="defend against unknown attacks", client=fake, art_dir=ART,
                 runlog=tmp_path / "l.jsonl", execute=False, use_reviewer=True)
    r2 = run_cli(intent="defend against unknown attacks", client=fake, art_dir=ART,
                 runlog=tmp_path / "l.jsonl", execute=False, use_reviewer=True)
    assert r1["tokens_delta"] > 0 and r2["tokens_delta"] > 0
    assert r2["tokens_total"] == r1["tokens_delta"] + r2["tokens_delta"]


def test_cli_refusal_logs_without_metrics(tmp_path):
    refuse = '{"schema_version": 1, "action": "refuse", "intent_understood_as": null, "reason": "out of scope"}'
    log = tmp_path / "runs.jsonl"
    record = run_cli(intent="rm -rf the detector", client=FakeLLM([refuse]),
                     art_dir=ART, runlog=log, execute=True, use_reviewer=True)
    assert record["status"] == "refusal" and "metrics" not in record
