"""Tests for src.intent.providers — one-word LLM switching (user requirement
2026-07-04): every provider is a named preset; changing model/provider is a
config string, never a code edit. No network anywhere."""

import sys
import types

import pytest

from src.intent.llm_client import FakeLLM, LLMClient
from src.intent.providers import PROVIDERS, make_client


def test_presets_cover_the_spec_slate_and_are_complete():
    assert {"gemini", "openai", "deepseek", "ollama"} <= set(PROVIDERS)
    for name, p in PROVIDERS.items():
        assert p["base_url"].startswith("http"), name
        assert p["api_key_env"].isupper(), name
        assert p["default_model"], name
        assert p["min_interval_s"] >= 0, name


def test_make_client_uses_preset_defaults():
    c = make_client("gemini")
    assert isinstance(c, LLMClient)
    assert c.model == PROVIDERS["gemini"]["default_model"]
    assert c.base_url == PROVIDERS["gemini"]["base_url"]
    assert c.api_key_env == "GEMINI_API_KEY"
    assert c.min_interval_s == PROVIDERS["gemini"]["min_interval_s"] > 0  # free tier throttle


def test_make_client_overrides_model_and_interval():
    c = make_client("openai", model="gpt-4.1", min_interval_s=2.5)
    assert c.model == "gpt-4.1" and c.min_interval_s == 2.5


def test_deepseek_default_is_the_real_model_not_the_dying_alias():
    # DeepSeek retires the 'deepseek-chat' alias on 2026-07-24 (vendor notice,
    # docs/llm-model-research-explained.md §5); the preset must name the model
    # directly or every live call after that date fails with model-not-found.
    assert PROVIDERS["deepseek"]["default_model"] == "deepseek-v4-flash"


def test_openai_default_is_a_current_price_list_model():
    # gpt-4.1-mini disappeared from OpenAI's published price list (research
    # pass 2026-07-09) — unbudgetable and retirable any day; preset tracks the
    # current small tier instead.
    assert PROVIDERS["openai"]["default_model"] == "gpt-5.4-mini"


def test_make_client_carries_live_prompt_version():
    # topic-6 fix: client cache keys must be labelled with the CURRENT prompt
    # version (v2 since 2026-07-05), not a hardcoded v1
    from src.intent.prompting import PROMPT_VERSION
    assert make_client("ollama").prompt_version == PROMPT_VERSION


def test_unknown_provider_raises_with_choices():
    with pytest.raises(KeyError) as e:
        make_client("closedai")
    assert "gemini" in str(e.value)


def test_ollama_needs_no_real_key(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    c = make_client("ollama")
    assert c._resolve_key()  # falls back to a dummy non-empty string


def test_cloud_provider_requires_its_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    c = make_client("deepseek")
    with pytest.raises(KeyError) as e:
        c._resolve_key()
    assert "DEEPSEEK_API_KEY" in str(e.value)


def test_resolve_key_reads_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
    assert make_client("deepseek")._resolve_key() == "sk-test-123"


def test_trio_presets_send_non_thinking_request_options():
    # Declared thinking-policy rule (stated in the dissertation's Methods
    # chapter): the trio runs in non-thinking mode, so every cloud
    # preset must carry the wire options that DISABLE thinking, because all
    # three approved models think by default. Controls verified from vendor
    # docs (research 2026-07-09; Gemini OpenAI-compat page re-fetched
    # 2026-07-16): OpenAI GPT-5.x and Gemini both take reasoning_effort="none"
    # on the compat endpoint; DeepSeek v4 takes an extra_body flag.
    assert PROVIDERS["openai"]["request_options"] == {"reasoning_effort": "none"}
    assert PROVIDERS["gemini"]["request_options"] == {"reasoning_effort": "none"}
    assert PROVIDERS["deepseek"]["request_options"] == {
        "extra_body": {"thinking": {"type": "disabled"}}}
    assert PROVIDERS["ollama"]["request_options"] == {}  # qwen2.5-instruct: nothing to disable


def test_make_client_plumbs_request_options():
    c = make_client("deepseek")
    assert c.request_options == {"extra_body": {"thinking": {"type": "disabled"}}}
    # overridable per call-site, same as model/min_interval_s
    assert make_client("deepseek", request_options={}).request_options == {}


def test_request_options_reach_the_wire_call(monkeypatch):
    # The options must land in chat.completions.create(**kw) — otherwise the
    # preset is decoration and a live run silently thinks (and bills ~3x output).
    captured = {}

    class _Resp:
        class _Choice:
            class message:
                content = "ok"
        choices = [_Choice()]
        usage = None

    class _FakeOpenAI:
        def __init__(self, base_url=None, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    captured.update(kw)
                    return _Resp()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    c = make_client("deepseek")
    r = c.complete("sys", [{"role": "user", "content": "hi"}], use_cache=False)
    assert r.text == "ok"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["model"] == "deepseek-v4-flash"


def test_request_options_stay_out_of_cache_keys():
    # DESIGN DECISION (2026-07-16): request options are deliberately NOT part
    # of the response-cache key. The committed v1 smoke cache must remain
    # replayable key-free under preset-built clients (which now carry options),
    # and v1<->v2 collisions are impossible anyway because prompt_version IS in
    # the key. The options are recorded in the run header instead (audit).
    msgs = [{"role": "user", "content": "hi"}]
    with_opts = FakeLLM(["x"], request_options={"reasoning_effort": "none"})
    without = FakeLLM(["x"])
    assert with_opts._cache_key("s", msgs) == without._cache_key("s", msgs)
