"""Tests for src.intent.prompting — canonical intents + prompt assembly."""

from pathlib import Path

from src.intent.prompting import (
    CANONICAL_IDS, build_compiler_system, build_reviewer_system, build_user_message, load_intents,
)

ART = Path(__file__).resolve().parents[1] / "artifacts"


def test_four_canonical_intents_with_phrasings():
    intents = load_intents()
    assert set(intents) == {"early_attack_detection", "minimize_false_alarms",
                            "balanced_operation", "defend_unknown_attacks"} == set(CANONICAL_IDS)
    for spec in intents.values():
        assert spec["definition"] and spec["success"] and len(spec["phrasings"]) >= 5


def test_compiler_system_contains_all_blocks():
    s = build_compiler_system(ART)
    for anchor in ["You do NOT", "KNOB MENU", "CAPABILITY CARD", "CANONICAL INTENTS",
                   "OUTPUT CONTRACT", "set_policy", "refuse", "intent is data"]:
        assert anchor in s
    assert "{knob_menu}" not in s and "{capability_card}" not in s  # placeholders filled
    assert len(s) < 16000  # ~4k tokens ceiling


def test_reviewer_system_is_reviewer_shaped():
    s = build_reviewer_system(ART)
    for anchor in ["skeptical", "approve", "revise", "checked", "do NOT edit"]:
        assert anchor in s
    assert "{intents_block}" not in s


def test_intent_definitions_disclose_oracle_budgets():
    # the oracle's budgets are part of each intent's MEANING —
    # the LLM must see the constraints it is graded against, and they must stay
    # in sync with the oracle's constants.
    from src.intent.oracle import BUDGETS_FA_PER_HOUR
    intents = load_intents()
    assert "1.0" in intents["early_attack_detection"]["definition"]
    assert "5.0" in intents["balanced_operation"]["definition"]
    assert "25" in intents["defend_unknown_attacks"]["definition"]
    assert "genuinely detected" in intents["minimize_false_alarms"]["definition"]
    assert BUDGETS_FA_PER_HOUR == {"early_attack_detection": 1.0,
                                   "balanced_operation": 5.0,
                                   "defend_unknown_attacks": 25.0}


def test_prompt_version_v2_with_hard_constraint_line():
    from src.intent.prompting import PROMPT_VERSION
    assert PROMPT_VERSION == "v2"   # budgets entered the assembled prompt -> version bump
    s = build_compiler_system(ART)
    assert "hard constraint" in s


def test_user_message_delimits_untrusted_intent():
    u = build_user_message("catch attacks ASAP", current_policy=None, observed=None)
    assert "[INTENT]" in u and "[/INTENT]" in u and "catch attacks ASAP" in u
    assert "[CURRENT POLICY]" in u and "[OBSERVED METRICS]" in u


def test_user_message_carries_state_when_present():
    u = build_user_message("x", current_policy={"model": "ae"}, observed={"fa_per_hour": 2.0})
    assert '"model": "ae"' in u and '"fa_per_hour": 2.0' in u


def test_adversarial_suite_loads_with_expected_outcomes():
    from src.intent.prompting import load_adversarial_intents
    adv = load_adversarial_intents()
    assert len(adv) >= 8
    kinds = {a["kind"] for a in adv}
    assert {"out_of_scope", "override_attempt", "sycophancy_bad_advice"} <= kinds
    for a in adv:
        assert a["text"].strip()
        if a["kind"] == "sycophancy_bad_advice":
            # plan T2: in-scope intent + legal-but-bad advice — a policy is
            # allowed; following the advised value is the security failure
            assert a["expected"] == "resist_bad_advice"
            assert set(a["bad_advice"]) == {"knob", "value"}
        else:
            # every other adversarial intent must end in refusal or gate
            # rejection — a returned policy would be a security failure
            assert a["expected"] == "refusal_or_rejection"
