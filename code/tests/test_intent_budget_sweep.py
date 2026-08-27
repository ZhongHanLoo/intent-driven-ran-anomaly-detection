"""Tests for src.intent.budget_sweep — plan T4: the offline budget-sensitivity
sweep (R2-04 B-C). Oracle-only; zero LLM cost; runs on the committed grid."""

from pathlib import Path

import pandas as pd
import pytest

from src.intent.oracle import BUDGETS_FA_PER_HOUR, feasible, oracle_policies

ART = Path(__file__).resolve().parents[1] / "artifacts"
GRID = pd.read_csv(ART / "policy_grid.csv")


def test_feasible_accepts_budget_override():
    row = {"fa_per_hour": 10.0, "dns_detected": True, "dns_sustained": 1.0,
           "gtpu_detected": False, "gtpu_sustained": 0.0,
           "dns_delay_s": 5.0, "gtpu_delay_s": float("nan"), "mcc": 0.5}
    assert feasible("defend_unknown_attacks", row) is True          # default budget 25.0
    assert feasible("defend_unknown_attacks", row,
                    budgets={"defend_unknown_attacks": 5.0}) is False


def test_oracle_policies_accepts_budget_override():
    # a tighter budget can only shrink the feasible set; the winner under the
    # default budget must itself be feasible under the default budget
    top_default = oracle_policies("early_attack_detection", GRID, top=1).iloc[0]
    assert float(top_default["fa_per_hour"]) <= BUDGETS_FA_PER_HOUR["early_attack_detection"]
    loose = oracle_policies("early_attack_detection", GRID, top=1,
                            budgets={**BUDGETS_FA_PER_HOUR, "early_attack_detection": 100.0})
    assert not loose.empty


def test_budget_sweep_covers_budgeted_intents_and_flags_winner_moves():
    from src.intent.budget_sweep import budget_sweep
    df = budget_sweep(GRID, factors=(0.5, 1.0, 2.0))
    budgeted = sorted(BUDGETS_FA_PER_HOUR)
    assert sorted(df["intent"].unique()) == budgeted
    assert len(df) == len(budgeted) * 3
    for intent in budgeted:
        sub = df[df["intent"] == intent].sort_values("factor")
        # n_feasible is nondecreasing in the budget factor
        assert sub["n_feasible"].is_monotonic_increasing or sub["n_feasible"].nunique() == 1
        # the reference row (factor 1.0) never counts as a move
        assert bool(sub[sub["factor"] == 1.0]["winner_moved"].iloc[0]) is False
    # every winner respects its swept budget
    ok = df.dropna(subset=["winner_fa_per_hour"])
    assert (ok["winner_fa_per_hour"] <= ok["budget_fa_per_hour"] + 1e-9).all()
