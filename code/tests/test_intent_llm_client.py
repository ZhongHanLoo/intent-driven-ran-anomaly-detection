"""Tests for src.intent.llm_client — FakeLLM, cache, token ceiling, JSONL log.
No network anywhere: the real client is constructed but never called."""

import json

import pytest

from src.intent.llm_client import (
    FakeLLM, LLMClient, TokenCeilingExceeded, append_jsonl, parse_retry_delay,
)


class _Clock:
    """Deterministic fake clock: sleeping advances 'now' by the slept amount."""

    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def test_parse_retry_delay_reads_googles_hint():
    msg = ("Error code: 429 - quota exceeded ... 'retryDelay': '47s' ... "
           "Please retry in 47.109167917s.")
    d = parse_retry_delay(msg)
    assert d is not None and 40 < d < 50


def test_parse_retry_delay_none_when_absent():
    assert parse_retry_delay("some unrelated error") is None


def test_throttle_spaces_network_calls():
    clock = _Clock()
    fake = FakeLLM(["a", "b", "c"], min_interval_s=13.0, _clock=clock.now, _sleep=clock.sleep)
    for _ in range(3):
        fake.complete("s", [{"role": "user", "content": "u"}])
    # first call is immediate; the next two are each spaced by the interval
    assert clock.sleeps == [13.0, 13.0]


def test_no_throttle_by_default():
    clock = _Clock()
    fake = FakeLLM(["a", "b"], _clock=clock.now, _sleep=clock.sleep)  # min_interval_s defaults to 0
    fake.complete("s", [{"role": "user", "content": "u"}])
    fake.complete("s", [{"role": "user", "content": "u"}])
    assert clock.sleeps == []


def test_fake_llm_replays_script_in_order():
    fake = FakeLLM(["one", "two"])
    assert fake.complete("sys", [{"role": "user", "content": "u"}]).text == "one"
    assert fake.complete("sys", [{"role": "user", "content": "u"}]).text == "two"
    with pytest.raises(IndexError):
        fake.complete("sys", [{"role": "user", "content": "u"}])


def test_fake_llm_counts_tokens_and_ceiling_aborts():
    fake = FakeLLM(["x" * 400] * 50, max_total_tokens=120)
    fake.complete("s", [{"role": "user", "content": "u"}])
    with pytest.raises(TokenCeilingExceeded):
        for _ in range(50):
            fake.complete("s", [{"role": "user", "content": "u"}])


def test_cache_round_trip(tmp_path):
    fake = FakeLLM(["only-answer"], cache_dir=tmp_path, prompt_version="v1")
    r1 = fake.complete("sys", [{"role": "user", "content": "u"}], use_cache=True)
    r2 = fake.complete("sys", [{"role": "user", "content": "u"}], use_cache=True)  # script exhausted -> must hit cache
    assert r1.text == r2.text == "only-answer" and r2.cached


def test_cache_hit_spends_zero_tokens(tmp_path):
    # B4 / audit C7: cache hits return before _spend, so replayed cells bill
    # nothing — the delta-based per-cell accounting depends on this
    fake = FakeLLM(["answer"], cache_dir=tmp_path)
    r1 = fake.complete("s", [{"role": "user", "content": "u"}])
    spent = fake.tokens_used
    assert not r1.cached and spent > 0
    r2 = fake.complete("s", [{"role": "user", "content": "u"}])
    assert r2.cached and fake.tokens_used == spent


def test_cache_bypass_goes_back_to_the_model(tmp_path):
    fake = FakeLLM(["first", "second"], cache_dir=tmp_path)
    assert fake.complete("s", [{"role": "user", "content": "u"}], use_cache=True).text == "first"
    assert fake.complete("s", [{"role": "user", "content": "u"}], use_cache=False).text == "second"


def test_real_client_constructs_without_key():
    c = LLMClient(model="gemini-2.5-flash", base_url="https://example.invalid/v1",
                  api_key_env="MISSING_KEY_ENV_VAR")
    assert c.model == "gemini-2.5-flash"


def test_append_jsonl(tmp_path):
    p = tmp_path / "runs.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"b": 2})
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert rows == [{"a": 1}, {"b": 2}]
