"""Loop-B: the staged metric-feedback driver — Paper 2's
Plan-Act-Observe-Reflect loop, replay-grounded and BOUNDED. Off by default
everywhere; enable via run_loop() or the CLI --loop flag.

Honest scope framing (for the report): on a replay, a policy's measured
metrics are static, so the loop tests whether observation-guided refinement
finds better policies in the knob space — not adaptation to a drifting
environment (that needs the live O-RAN loop; Future Work). The observation
feeds the COMPILER only; the Reviewer keeps judging against the capability
card (v1 semantics).

Stopping rules (all bounded):
- fixed point: the compiler re-emits the current policy -> converged
  (no re-execution needed — same policy, same replay metrics);
- non-policy outcome mid-loop (refusal/rejected/no_convergence) -> halt and
  KEEP the last good policy (Paper 2's 'wait/rollback' commit gate);
- max_iterations reached -> stop with the latest policy."""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.intent.apply import apply_policy
from src.intent.llm_client import append_jsonl
from src.intent.prompting import PROMPT_VERSION
from src.intent.reviewer import run_two_agent

# the compact operator-facing observation the compiler sees each iteration
OBSERVED_KEYS = ("fa_per_hour", "dns_delay_s", "gtpu_delay_s", "dns_detected",
                 "gtpu_detected", "mcc", "fpr", "fnr", "macro_f1")


def _observation(metrics: dict) -> dict:
    out = {}
    for k in OBSERVED_KEYS:
        v = metrics.get(k)
        if isinstance(v, float):
            v = None if not math.isfinite(v) else round(v, 4)
        out[k] = v
    return out


@dataclass
class LoopResult:
    status: str                          # converged | max_iterations | halted_<pipeline status>
    final_policy: Optional[dict] = None
    final_metrics: Optional[dict] = None
    iterations: int = 0
    trajectory: list = field(default_factory=list)


def run_loop(intent_text: str, client, registry, art_dir: Path,
             max_iterations: int = 3, use_reviewer: bool = True,
             include_card: bool = True, runlog: Optional[Path] = None) -> LoopResult:
    current_policy: Optional[dict] = None
    current_metrics: Optional[dict] = None
    trajectory: list = []
    status = "max_iterations"
    for it in range(1, max_iterations + 1):
        res = run_two_agent(intent_text, client, registry, art_dir,
                            use_reviewer=use_reviewer, include_card=include_card,
                            current_policy=current_policy,
                            observed=_observation(current_metrics) if current_metrics else None)
        if res.status != "policy":
            status = f"halted_{res.status}"
            break
        new_policy = res.output.policy.model_dump()
        converged = current_policy is not None and new_policy == current_policy
        metrics = current_metrics if converged else apply_policy(res.output.policy, registry)
        entry = {"iteration": it, "pipeline_status": res.status, "rounds": res.rounds,
                 "policy": new_policy, "metrics": _observation(metrics),
                 "converged": converged}
        trajectory.append(entry)
        if runlog:
            append_jsonl(runlog, {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  "kind": "loop_iteration", "intent": intent_text,
                                  "llm_model": client.model, "prompt_version": PROMPT_VERSION,
                                  **entry})
        current_policy, current_metrics = new_policy, metrics
        if converged:
            status = "converged"
            break
    result = LoopResult(status=status, final_policy=current_policy,
                        final_metrics=current_metrics, iterations=len(trajectory),
                        trajectory=trajectory)
    if runlog:
        append_jsonl(runlog, {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              "kind": "loop_summary", "intent": intent_text,
                              "llm_model": client.model, "status": result.status,
                              "iterations": result.iterations,
                              "final_policy": result.final_policy})
    return result
