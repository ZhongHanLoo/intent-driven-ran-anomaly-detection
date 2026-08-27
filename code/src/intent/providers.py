"""Named LLM provider presets: switching model/provider is a one-word
config change. All presets speak
the OpenAI-compatible protocol, so one client class serves every row. Keys
live in environment variables (see code/.secrets.env.example) — never here.

Model names and free-tier terms drift; verify at signup. min_interval_s is a
proactive throttle for per-minute quotas (0 = no throttle)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.intent.llm_client import LLMClient

PROVIDERS: dict[str, dict] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "min_interval_s": 2.0,  # paid tier (user-enabled 2026-07-17): was 13.0 for
        # the free tier's 5 req/min; lowered 2026-08-07 (user decision) for the
        # extension leg — pacing only, no measured value depends on it
        # 2.5 Flash THINKS BY DEFAULT; the compat endpoint disables it via
        # reasoning_effort="none" (Gemini OpenAI-compat docs, verified 2026-07-16).
        "request_options": {"reasoning_effort": "none"},
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        # gpt-4.1-mini left OpenAI's price list (research 2026-07-09); current small tier.
        # GPT-5.x are reasoning models — the declared non-thinking rule (report
        # ch4 §4.3) is enforced by request_options below.
        "default_model": "gpt-5.4-mini",
        "min_interval_s": 0.0,
        "request_options": {"reasoning_effort": "none"},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        # 'deepseek-chat' alias retires 2026-07-24 (vendor notice). Unlike the old
        # alias, v4 models THINK BY DEFAULT — disabled per request via the
        # request_options below (docs/llm-model-research-explained.md §2).
        "default_model": "deepseek-v4-flash",
        "min_interval_s": 0.0,
        "request_options": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
    "ollama": {  # local, no account, no key (SDK still wants a non-empty string)
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "default_model": "qwen2.5:14b-instruct",
        "min_interval_s": 0.0,
        "api_key_fallback": "ollama",
        "request_options": {},  # qwen2.5-instruct is non-thinking; nothing to disable
    },
}


def make_client(provider: str, model: Optional[str] = None,
                min_interval_s: Optional[float] = None,
                cache_dir: Optional[Path] = None, **kw) -> LLMClient:
    """Preset -> ready client. `model`/`min_interval_s` override the preset.
    The client is labelled with the LIVE prompt version (cache-key metadata)."""
    from src.intent.prompting import PROMPT_VERSION  # no cycle: prompting never imports providers
    if provider not in PROVIDERS:
        raise KeyError(f"unknown provider '{provider}'; choose from {sorted(PROVIDERS)}")
    kw.setdefault("prompt_version", PROMPT_VERSION)
    p = PROVIDERS[provider]
    kw.setdefault("request_options", p.get("request_options"))
    return LLMClient(model=model or p["default_model"],
                     base_url=p["base_url"],
                     api_key_env=p["api_key_env"],
                     api_key_fallback=p.get("api_key_fallback"),
                     min_interval_s=p["min_interval_s"] if min_interval_s is None else min_interval_s,
                     cache_dir=cache_dir, **kw)
