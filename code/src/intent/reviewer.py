"""Reviewer agent + two-agent loop control. Compiler drafts ->
Reviewer verdicts; 'revise' returns the critique to the Compiler; at most
max_rounds rounds; non-convergence applies NOTHING (current policy stands).
A reviewer output that fails its own guardrails gets the same one-repair
treatment; a second failure counts as 'revise' with a generic critique —
the loop remains bounded either way."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.intent.guardrails import validate_reviewer_output
from src.intent.prompting import build_reviewer_system
from src.intent.translator import REPAIR_INSTRUCTION, CompileResult, compile_intent


@dataclass
class PipelineResult:
    status: str                      # "policy" | "refusal" | "rejected" | "no_convergence"
    output: Optional[object] = None  # SetPolicy when status == "policy"
    gate: Optional[str] = None
    rounds: int = 0
    transcript: list = field(default_factory=list)


def _review(policy_out, intent_text: str, client, art_dir: Path, include_card: bool = True):
    system = build_reviewer_system(art_dir, include_card=include_card)
    user = (f"[INTENT]\n{intent_text}\n[/INTENT]\n\n"
            f"[PROPOSED POLICY]\n{policy_out.model_dump_json()}")
    messages = [{"role": "user", "content": user}]
    resp = client.complete(system, messages)
    result = validate_reviewer_output(resp.text)
    if not result.ok:
        messages = messages + [{"role": "assistant", "content": resp.text},
                               {"role": "user", "content": REPAIR_INSTRUCTION.format(error=result.error)}]
        resp = client.complete(system, messages)
        result = validate_reviewer_output(resp.text)
    return result, resp.text


def run_two_agent(intent_text: str, client, registry, art_dir: Path,
                  max_rounds: int = 3, use_reviewer: bool = True,
                  current_policy: dict | None = None,
                  observed: dict | None = None,
                  include_card: bool = True,
                  carry_critique: bool = True) -> PipelineResult:
    critique = None
    generic_revise = False
    transcript = []
    for rnd in range(1, max_rounds + 1):
        c: CompileResult = compile_intent(intent_text, client, registry, art_dir,
                                          current_policy=current_policy,
                                          observed=observed, critique=critique,
                                          include_card=include_card,
                                          generic_revise=generic_revise)
        transcript.append({"round": rnd, "compiler": c.transcript, "status": c.status, "gate": c.gate})
        if c.status in ("rejected", "refusal"):
            return PipelineResult(c.status, output=c.output, gate=c.gate, rounds=rnd, transcript=transcript)
        if not use_reviewer:
            return PipelineResult("policy", output=c.output, rounds=rnd, transcript=transcript)
        verdict_result, verdict_text = _review(c.output, intent_text, client, art_dir,
                                               include_card=include_card)
        transcript[-1]["reviewer"] = verdict_text
        if verdict_result.ok and verdict_result.output.verdict == "approve":
            return PipelineResult("policy", output=c.output, rounds=rnd, transcript=transcript)
        if carry_critique:
            critique = (verdict_result.output.critique if verdict_result.ok
                        else "Your previous draft could not be reviewed; simplify and re-emit.")
        else:  # A3 regeneration arm: the verdict still gates, the content does not travel
            critique, generic_revise = None, True
    return PipelineResult("no_convergence", rounds=max_rounds, transcript=transcript)
