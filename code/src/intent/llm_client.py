"""Provider-agnostic LLM access. One OpenAI-compatible adapter
covers Gemini/OpenAI/DeepSeek/Ollama (base_url + key). Token ceiling guards
spend; responses cache to disk keyed on (model, prompt_version, messages,
temperature) so every downstream number is re-derivable without API access.
FakeLLM implements the same interface for the no-network test suite."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class TokenCeilingExceeded(RuntimeError):
    pass


class TransportFailure(RuntimeError):
    """Raised when a network call still fails after retries/backoff. The
    experiment harness records it as a reliability datum and continues;
    interactive callers (CLI/smoke) surface it directly."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _rough_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # ~4 chars/token heuristic for ceilings only


_RETRY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s"),  # Google structured hint
    re.compile(r"retry in (\d+(?:\.\d+)?)\s*s"),                    # prose form
)


def parse_retry_delay(err) -> Optional[float]:
    """Extract the server-suggested cooldown (seconds) from a rate-limit error,
    or None if the error carries no hint. Lets the client honour a 429's
    'retry in Ns' instead of blind backoff — important on free-tier quotas."""
    s = str(err)
    for pat in _RETRY_PATTERNS:
        m = pat.search(s)
        if m:
            return float(m.group(1))
    return None


class _Base:
    def __init__(self, model: str, temperature: float = 0.0,
                 max_total_tokens: int = 2_000_000,
                 cache_dir: Optional[Path] = None, prompt_version: str = "v1",
                 min_interval_s: float = 0.0, request_options: Optional[dict] = None,
                 _clock=time.monotonic, _sleep=time.sleep):
        self.model, self.temperature = model, temperature
        # extra wire options sent on every live call (e.g. reasoning_effort=
        # "none", extra_body={"thinking": ...}) — how the declared non-thinking
        # rule (report ch4 §4.3) is enforced in code. Deliberately NOT part of
        # the cache key (the committed v1 smoke cache must replay key-free, and
        # prompt_version in the key already separates v1/v2); recorded in every
        # run header instead so each matrix is attributable to one thinking policy.
        self.request_options = dict(request_options or {})
        self.max_total_tokens, self.tokens_used = max_total_tokens, 0
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.prompt_version = prompt_version
        # proactive throttle: keep >= min_interval_s between real network calls
        # (free-tier quotas are per-minute; 0 disables it, so tests/FakeLLM are unaffected)
        self.min_interval_s = min_interval_s
        self._clock, self._sleep, self._last_call = _clock, _sleep, None
        # experiment repeats set a salt so each repeat gets its own cache slot:
        # repeats measure provider nondeterminism live, yet replay offline later
        self.cache_salt = ""

    def _cache_key(self, system: str, messages: list[dict]) -> str:
        blob = json.dumps({"m": self.model, "v": self.prompt_version, "t": self.temperature,
                           "s": system, "msgs": messages, "salt": self.cache_salt},
                          sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[LLMResponse]:
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.json"
        if p.exists():
            d = json.loads(p.read_text())
            return LLMResponse(d["text"], d["input_tokens"], d["output_tokens"], d["model"], cached=True)
        return None

    def _cache_put(self, key: str, r: LLMResponse) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(json.dumps(
                {"text": r.text, "input_tokens": r.input_tokens,
                 "output_tokens": r.output_tokens, "model": r.model}))

    def _spend(self, in_tok: int, out_tok: int) -> None:
        self.tokens_used += in_tok + out_tok
        if self.tokens_used > self.max_total_tokens:
            raise TokenCeilingExceeded(f"{self.tokens_used} > ceiling {self.max_total_tokens}")

    def _throttle(self) -> None:
        if self.min_interval_s > 0 and self._last_call is not None:
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)

    def complete(self, system: str, messages: list[dict], use_cache: bool = True) -> LLMResponse:
        key = self._cache_key(system, messages)
        if use_cache:
            hit = self._cache_get(key)
            if hit:  # cache hits are free and unthrottled
                return hit
        self._throttle()
        r = self._complete(system, messages)
        if self.min_interval_s > 0:
            self._last_call = self._clock()
        self._spend(r.input_tokens, r.output_tokens)
        self._cache_put(key, r)
        return r

    def _complete(self, system: str, messages: list[dict]) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class FakeLLM(_Base):
    """Scripted stand-in for tests: returns canned texts in order."""

    def __init__(self, scripted: list[str], **kw):
        super().__init__(model="fake", **kw)
        self._script = list(scripted)
        self.calls: list[dict] = []  # every real (non-cached) call, for test assertions

    def _complete(self, system: str, messages: list[dict]) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages})
        text = self._script.pop(0)  # IndexError when exhausted (intentional)
        return LLMResponse(text,
                           _rough_tokens(system) + sum(_rough_tokens(m["content"]) for m in messages),
                           _rough_tokens(text), self.model)


class LLMClient(_Base):
    """Real client. Lazy-imports the openai SDK; never touched by the suite."""

    def __init__(self, model: str, base_url: str, api_key_env: str,
                 max_retries: int = 3, max_backoff_s: float = 90.0,
                 api_key_fallback: Optional[str] = None, **kw):
        super().__init__(model=model, **kw)
        self.base_url, self.api_key_env = base_url, api_key_env
        self.max_retries, self.max_backoff_s = max_retries, max_backoff_s
        self.api_key_fallback = api_key_fallback  # for keyless local providers (Ollama)

    def _resolve_key(self) -> str:
        key = os.environ.get(self.api_key_env) or self.api_key_fallback
        if not key:
            raise KeyError(f"set the {self.api_key_env} environment variable "
                           f"(see code/.secrets.env.example)")
        return key

    def _complete(self, system: str, messages: list[dict]) -> LLMResponse:  # pragma: no cover (live only)
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key=self._resolve_key())
        last = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system}, *messages],
                    **self.request_options)
                u = resp.usage
                return LLMResponse(resp.choices[0].message.content or "",
                                   u.prompt_tokens if u else 0,
                                   u.completion_tokens if u else 0, self.model)
            except Exception as e:  # transport/rate-limit: honour server hint, else backoff
                last = e
                hint = parse_retry_delay(e)
                self._sleep(min(hint + 1.0, self.max_backoff_s) if hint is not None else 2 ** attempt)
        raise TransportFailure(f"after {self.max_retries} retries: {last}")
