"""Compiler agent: prompt -> LLM -> guardrails, one repair round
per output. Returns a CompileResult; never raises on model misbehaviour —
misbehaviour is data (the Phase-3 rejection taxonomy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.intent.guardrails import validate_compiler_output
from src.intent.prompting import CANONICAL_IDS, build_compiler_system, build_user_message
from src.intent.schema import SetPolicy

REPAIR_INSTRUCTION = ("Your output failed validation: {error}\n"
                      "Emit exactly one corrected JSON object and nothing else.")


@dataclass
class CompileResult:
    status: str                      # "policy" | "refusal" | "rejected"
    output: Optional[object] = None  # SetPolicy | Refuse
    gate: Optional[str] = None       # failing gate when rejected
    repair_used: bool = False
    transcript: list = field(default_factory=list)  # [{"role","content"}...] for the audit log


def compile_intent(intent_text: str, client, registry, art_dir: Path,
                   current_policy: dict | None = None,
                   observed: dict | None = None,
                   critique: str | None = None,
                   include_card: bool = True,
                   generic_revise: bool = False) -> CompileResult:
    system = build_compiler_system(art_dir, include_card=include_card)
    user = build_user_message(intent_text, current_policy, observed)
    if critique:  # reviewer round: critique appended as a follow-up instruction
        user += f"\n\n[REVIEWER CRITIQUE OF YOUR PREVIOUS DRAFT]\n{critique}\nEmit a corrected JSON object."
    elif generic_revise:  # A3 arm: a retry must differ from round 1 (cache) but carry no critique content
        user += ("\n\n[REVISION REQUESTED]\nYour previous draft was not approved. "
                 "Re-derive the policy from the intent above and emit a corrected JSON object.")
    messages = [{"role": "user", "content": user}]
    transcript = list(messages)
    resp = client.complete(system, messages)
    transcript.append({"role": "assistant", "content": resp.text})
    result = validate_compiler_output(resp.text, registry, set(CANONICAL_IDS))
    repair_used = False
    if not result.ok:
        repair_used = True
        messages = messages + [{"role": "assistant", "content": resp.text},
                               {"role": "user", "content": REPAIR_INSTRUCTION.format(error=result.error)}]
        resp = client.complete(system, messages)
        transcript += [messages[-1], {"role": "assistant", "content": resp.text}]
        result = validate_compiler_output(resp.text, registry, set(CANONICAL_IDS))
    if not result.ok:
        return CompileResult("rejected", gate=result.gate, repair_used=repair_used, transcript=transcript)
    status = "policy" if isinstance(result.output, SetPolicy) else "refusal"
    return CompileResult(status, output=result.output, repair_used=repair_used, transcript=transcript)
