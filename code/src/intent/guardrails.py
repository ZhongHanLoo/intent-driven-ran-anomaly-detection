"""Deterministic guardrails. Gates: G0 size/fence -> G1 parse ->
G2 schema -> (G3 range folds into schema validators) -> G4 registry ->
G5 consistency. A GateResult with ok=False carries the failing gate tag —
the Phase-3 rejection taxonomy. No LLM anywhere in this module."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from src.intent.schema import Refuse, ReviewerVerdict, SetPolicy, parse_compiler_output, parse_reviewer_verdict

MAX_BYTES = 4096
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class GateResult:
    ok: bool
    output: Optional[object] = None   # SetPolicy | Refuse | ReviewerVerdict
    gate: Optional[str] = None        # parse | schema | registry | consistency
    error: Optional[str] = None


def _extract_json(text: str) -> str:
    """G0: size cap, single fence strip, must be exactly one JSON object."""
    if len(text.encode()) > MAX_BYTES:
        raise ValueError(f"response exceeds {MAX_BYTES} bytes")
    t = text.strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    obj, end = json.JSONDecoder().raw_decode(t)
    if t[end:].strip():
        raise ValueError("trailing content after the JSON object")
    if not isinstance(obj, dict):
        raise ValueError("top level is not a JSON object")
    return t[:end]


def validate_compiler_output(text: str, registry, canonical_ids: set[str]) -> GateResult:
    try:
        payload = _extract_json(text)                       # G0 + G1
    except (ValueError, json.JSONDecodeError) as e:
        return GateResult(False, gate="parse", error=str(e))
    try:
        out = parse_compiler_output(payload)                # G2 (+G3 via validators)
    except ValidationError as e:
        return GateResult(False, gate="schema", error=str(e)[:1000])
    if isinstance(out, SetPolicy):
        p = out.policy
        try:                                                # G4
            registry.resolve_threshold(p.model, p.window, p.threshold_q)
        except KeyError as e:
            return GateResult(False, gate="registry", error=str(e))
    if out.intent_understood_as is not None and out.intent_understood_as not in canonical_ids:
        return GateResult(False, gate="consistency",        # G5
                          error=f"intent_understood_as '{out.intent_understood_as}' is not canonical")
    return GateResult(True, output=out)


def validate_reviewer_output(text: str) -> GateResult:
    try:
        payload = _extract_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        return GateResult(False, gate="parse", error=str(e))
    try:
        v = parse_reviewer_verdict(payload)
    except ValidationError as e:
        return GateResult(False, gate="schema", error=str(e)[:1000])
    return GateResult(True, output=v)
