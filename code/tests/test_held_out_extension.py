"""Pin-tests for the 2026-08-06 held-out phrasing extension (freeze 32817a9).

Everything offline. These are pins of already-built behaviour (the --held-out
path + FIX-6) against the newly frozen held_out_phrasings.json — expected to
pass immediately, guarding shape, tagging, order and population separation
before any live extension cell is fired."""

import json
from pathlib import Path

from src.intent.analysis import arm_mcnemar, paired_cells
from src.intent.experiment import Arm, ExperimentConfig, run_matrix
from src.intent.llm_client import FakeLLM
from src.intent.prompting import load_held_out_phrasings, load_intents
from src.intent.registry import Registry

ART = Path(__file__).resolve().parents[1] / "artifacts"
REG = Registry(ART)

POLICY_TF = ('{"schema_version": 1, "action": "set_policy",'
             ' "policy": {"model": "transformer", "window": 3, "threshold_q": 0.995, "persistence": 1},'
             ' "intent_understood_as": "early_attack_detection", "reason": "fast robust"}')
APPROVE = ('{"schema_version": 1, "verdict": "approve",'
           ' "checked": ["intent_match", "tradeoff_direction", "knob_legality"], "critique": ""}')


class RoleAwareFakeLLM(FakeLLM):
    """Answers by role, so any cell order works: reviewer systems get APPROVE,
    compiler systems get a valid policy. No script to exhaust."""

    def __init__(self, **kw):
        super().__init__([], **kw)

    def _complete(self, system, messages):
        self.calls.append({"system": system, "messages": messages})
        text = APPROVE if "policy reviewer" in system else POLICY_TF
        from src.intent.llm_client import LLMResponse
        return LLMResponse(text, 10, 5, self.model)


def _cfg(**kw):
    base = dict(phrasings_per_intent=3, repeats=2,
                arms=(Arm(reviewer=True, card=True), Arm(reviewer=False, card=True)),
                adversarial=False, held_out=True, execute=False)
    base.update(kw)
    return ExperimentConfig(**base)


def test_frozen_file_shape_is_15_per_intent_at_5_5_5():
    """Data pin on the frozen file itself: 12 extension sentences per intent;
    with the 3 fielded canonical ones every intent is 15 at 5 short/5 medium/5 long."""
    ho = load_held_out_phrasings()
    intents = load_intents()

    def cls(t):
        n = len(t.split())
        return "short" if n <= 4 else ("medium" if n <= 7 else "long")

    assert sum(len(v) for v in ho["phrasings"].values()) == 48
    for iid, spec in intents.items():
        ext = ho["phrasings"][iid]
        assert len(ext) == 12
        pool = [cls(t) for t in spec["phrasings"][:3]] + [cls(t) for t in ext]
        assert len(pool) == 15
        assert {c: pool.count(c) for c in ("short", "medium", "long")} == \
               {"short": 5, "medium": 5, "long": 5}, iid
        # reserves-first ordering: the first entries are intents.json's own tail
        reserves = spec["phrasings"][3:]
        assert ext[:len(reserves)] == reserves


def test_held_out_cells_shape_tags_and_order():
    """--held-out appends exactly 48 sentences x repeats x arms, every record
    tagged held_out=True with phrasing_index = position in the frozen file."""
    recs = run_matrix(RoleAwareFakeLLM(), _cfg(), REG, ART)
    ho_file = load_held_out_phrasings()
    ho = [r for r in recs if r.get("held_out")]
    canonical = [r for r in recs if not r.get("held_out")]

    assert len(canonical) == 4 * 3 * 2 * 2            # intents x phrasings x repeats x arms
    assert len(ho) == 48 * 2 * 2                      # sentences x repeats x arms
    assert all(r["held_out"] is True for r in ho)
    assert all(r.get("adversarial_kind") is None for r in ho)
    assert all("held_out" not in r for r in canonical)
    for iid, texts in ho_file["phrasings"].items():
        for pi, text in enumerate(texts):
            sub = [r for r in ho if r["intent_id"] == iid and r["phrasing_index"] == pi]
            assert len(sub) == 2 * 2 and all(r["text"] == text for r in sub)


def test_analysis_population_separation_with_held_out_present():
    """The FIX-1 firewall extends to the extension: default (canonical) pairing
    never sees held-out units; population='held_out' sees exactly the 48."""
    recs = run_matrix(RoleAwareFakeLLM(), _cfg(), REG, ART)
    a, b = "reviewer|card", "compiler-only|card"
    assert arm_mcnemar(recs, a, b)["n_pairs"] == 12                       # 4 intents x 3 phrasings
    assert arm_mcnemar(recs, a, b, population="held_out")["n_pairs"] == 48
    assert len(paired_cells(recs, a, b, population="held_out")) == 48 * 2  # repeat-level pairs
