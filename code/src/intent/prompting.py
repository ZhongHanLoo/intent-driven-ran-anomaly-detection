"""Prompt assembly. System prompts are versioned templates under
prompts/<version>/; any wording change bumps the version directory. The
intent text is the ONLY untrusted input and is delimited in the user message."""

from __future__ import annotations

import json
from pathlib import Path

from src.intent.capability_card import build_card
from src.intent.knobs import knob_menu_text

_HERE = Path(__file__).resolve().parent
PROMPT_VERSION = "v2"  # v2 (2026-07-05): oracle budgets entered
# the intent definitions + a hard-constraint line in both templates. v1 remains
# on disk; the response cache keys on this version, so v1 runs stay replayable.


def load_intents() -> dict:
    return json.loads((_HERE / "intents.json").read_text())


def load_held_out_phrasings(path: Path | None = None) -> dict | None:
    """Plan T3: the frozen held-out phrasing set (authored blind to the compiler
    prompt; provenance + freeze date recorded in the file). None until the file
    exists — the matrix simply skips held-out cells then."""
    p = path if path is not None else _HERE / "held_out_phrasings.json"
    if not p.exists():
        return None
    ho = json.loads(p.read_text())
    assert {"provenance", "frozen", "phrasings"} <= set(ho), "held-out file missing provenance fields"
    return ho


def load_adversarial_intents() -> list[dict]:
    """The Phase-3 injection/robustness suite: out-of-scope, contradictory and
    instruction-override intents. Every one must end as a refusal (the LLM
    declines) or a gate rejection (guardrails catch it) — a returned policy
    counts as a security failure in the evaluation."""
    return json.loads((_HERE / "adversarial_intents.json").read_text())


CANONICAL_IDS = frozenset(load_intents())


def _intents_block() -> str:
    return "\n".join(f"- {iid}: {spec['definition']} Success: {spec['success']}"
                     for iid, spec in load_intents().items())


def _template(name: str) -> str:
    return (_HERE / "prompts" / PROMPT_VERSION / f"{name}.txt").read_text()


CARD_WITHHELD = "(withheld for this ablation arm)"


def build_compiler_system(art_dir: Path, include_card: bool = True) -> str:
    card = build_card(art_dir) if include_card else CARD_WITHHELD
    return _template("compiler").format(knob_menu=knob_menu_text(),
                                        capability_card=card,
                                        intents_block=_intents_block())


def build_reviewer_system(art_dir: Path, include_card: bool = True) -> str:
    card = build_card(art_dir) if include_card else CARD_WITHHELD
    return _template("reviewer").format(knob_menu=knob_menu_text(),
                                        capability_card=card,
                                        intents_block=_intents_block())


def build_user_message(intent_text: str, current_policy: dict | None,
                       observed: dict | None) -> str:
    return (f"[INTENT]\n{intent_text}\n[/INTENT]\n\n"
            f"[CURRENT POLICY]\n{json.dumps(current_policy) if current_policy else 'none'}\n\n"
            f"[OBSERVED METRICS]\n{json.dumps(observed) if observed else 'none'}")
