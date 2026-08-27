"""Tests for src.intent.oracle — the deterministic evidence-optimal planner.
Exact winners on a tiny synthetic grid + property tests on the
real committed policy_grid.csv (no magic numbers: the oracle must be optimal
BY CONSTRUCTION, so we assert undominatedness, not memorized winners)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intent.oracle import (
    BUDGETS_FA_PER_HOUR, GENUINE_SUSTAINED, annotate, objective_key,
    oracle_gap, oracle_policies,
)

ART = Path(__file__).resolve().parents[1] / "artifacts"
GRID = pd.read_csv(ART / "policy_grid.csv")
CANONICAL = ["early_attack_detection", "minimize_false_alarms",
             "balanced_operation", "defend_unknown_attacks"]


def _synth_grid():
    """Four hand-crafted policies with obvious winners per intent."""
    base = dict(model="lstm", window=3, threshold_q="default", persistence=1,
                mcc=0.5, dns_detected=True, dns_delay_s=100.0, dns_sustained=0.5,
                gtpu_detected=False, gtpu_delay_s=np.nan, gtpu_sustained=0.0)
    rows = [
        # A: fast + noisy (best for early detection within budget)
        {**base, "model": "transformer", "fa_per_hour": 0.9, "dns_delay_s": 5.0},
        # B: quietest genuine-DNS detector (best for minimize_false_alarms)
        {**base, "model": "tcn", "fa_per_hour": 0.1, "dns_delay_s": 300.0, "mcc": 0.4},
        # C: highest MCC (best for balanced_operation)
        {**base, "fa_per_hour": 2.0, "mcc": 0.9},
        # D: detects BOTH incidents (best for defend_unknown_attacks)
        {**base, "model": "ae", "fa_per_hour": 20.0, "gtpu_detected": True,
         "gtpu_delay_s": 60.0, "gtpu_sustained": 0.4, "mcc": 0.3},
    ]
    return pd.DataFrame(rows)


def test_annotate_genuine_rule():
    g = annotate(_synth_grid())
    assert list(g.n_genuine) == [1, 1, 1, 2]
    assert g.mean_genuine_delay_s.iloc[3] == pytest.approx((100.0 + 60.0) / 2)


def test_synthetic_winners_per_intent():
    g = _synth_grid()
    assert oracle_policies("early_attack_detection", g).iloc[0].model == "transformer"
    assert oracle_policies("minimize_false_alarms", g).iloc[0].model == "tcn"
    assert oracle_policies("balanced_operation", g).iloc[0].mcc == 0.9
    assert oracle_policies("defend_unknown_attacks", g).iloc[0].model == "ae"


def test_budgets_are_the_disclosed_constants():
    assert BUDGETS_FA_PER_HOUR == {"early_attack_detection": 1.0,
                                   "balanced_operation": 5.0,
                                   "defend_unknown_attacks": 25.0}
    assert GENUINE_SUSTAINED == 0.3


@pytest.mark.parametrize("intent", CANONICAL)
def test_real_grid_every_intent_has_feasible_policies(intent):
    top = oracle_policies(intent, GRID)
    assert 1 <= len(top) <= 3


@pytest.mark.parametrize("intent", CANONICAL)
def test_real_grid_winner_is_undominated_and_top3_sorted(intent):
    top = oracle_policies(intent, GRID)
    ann = annotate(GRID)
    keys = [objective_key(intent, r) for _, r in top.iterrows()]
    assert keys == sorted(keys)  # best-first
    best = keys[0]
    # no feasible row anywhere in the grid strictly beats the winner
    for _, row in ann.iterrows():
        if _feasible(intent, row):
            assert objective_key(intent, row) >= best


def _feasible(intent, row):
    from src.intent.oracle import feasible
    return feasible(intent, row)


def test_real_grid_budget_respected():
    for intent, budget in BUDGETS_FA_PER_HOUR.items():
        top = oracle_policies(intent, GRID)
        assert (top.fa_per_hour <= budget).all()


def test_real_grid_min_fa_winner_detects_dns_genuinely():
    top = oracle_policies("minimize_false_alarms", GRID)
    r = top.iloc[0]
    assert bool(r.dns_detected) and r.dns_sustained >= GENUINE_SUSTAINED


def test_oracle_gap_of_winner_is_rank_one():
    top = oracle_policies("balanced_operation", GRID)
    gap = oracle_gap("balanced_operation", top.iloc[0].to_dict(), GRID)
    assert gap["rank_of_chosen"] == 1 and gap["chosen_feasible"] is True
    assert gap["knob_match"]["model"] is True


def test_oracle_gap_of_bad_feasible_policy_ranks_worse():
    ann = annotate(GRID)
    feas = ann[[_feasible("balanced_operation", r) for _, r in ann.iterrows()]]
    worst = feas.sort_values("mcc").iloc[0]  # lowest-MCC feasible row
    gap = oracle_gap("balanced_operation", worst.to_dict(), GRID)
    assert gap["chosen_feasible"] is True and gap["rank_of_chosen"] > 1


def test_knob_match_accepts_any_tied_optimum():
    # co-optimal rows tie on the objective key; a chosen knob
    # value matching ANY tied optimum must count as agreement, not sort-order luck
    base = dict(model="tcn", window=3, persistence=1, mcc=0.5,
                dns_detected=True, dns_delay_s=10.0, dns_sustained=0.5,
                gtpu_detected=False, gtpu_delay_s=np.nan, gtpu_sustained=0.0,
                fa_per_hour=0.0)
    g = pd.DataFrame([
        {**base, "threshold_q": "0.999", "threshold_resolved": 0.7},
        {**base, "threshold_q": "0.9999", "threshold_resolved": 0.9},   # ties row 1 on (fa, dns_delay)
        {**base, "threshold_q": "0.99", "threshold_resolved": 0.5, "fa_per_hour": 3.0},
    ])
    chosen = g.iloc[1].to_dict()
    gap = oracle_gap("minimize_false_alarms", chosen, g)
    assert gap["rank_of_chosen"] == 1
    assert gap["knob_match"]["threshold_q"] is True


def test_threshold_match_resolves_default_equivalence():
    # ae "default" IS its q0.995 threshold: string inequality must not deny a match
    base = dict(model="ae", window=3, persistence=1, mcc=0.4,
                dns_detected=True, dns_delay_s=20.0, dns_sustained=0.6,
                gtpu_detected=False, gtpu_delay_s=np.nan, gtpu_sustained=0.0,
                fa_per_hour=0.5)
    g = pd.DataFrame([{**base, "threshold_q": "0.995", "threshold_resolved": 0.109505}])
    chosen = {**base, "threshold_q": "default", "threshold_resolved": 0.109505}
    gap = oracle_gap("minimize_false_alarms", chosen, g)
    assert gap["knob_match"]["threshold_q"] is True  # resolved thresholds identical


def test_real_grid_has_324_distinct_policies():
    # ae "default" duplicates its q0.995 row for every (window, persistence):
    # 336 grid cells = 324 physically distinct policies (disclosed, topic-5 F2)
    dups = GRID.duplicated(subset=["model", "window", "threshold_resolved", "persistence"]).sum()
    assert len(GRID) == 336 and dups == 12


def test_oracle_gap_flags_infeasible_choice():
    ann = annotate(GRID)
    over = ann[ann.fa_per_hour > BUDGETS_FA_PER_HOUR["early_attack_detection"]].iloc[0]
    gap = oracle_gap("early_attack_detection", over.to_dict(), GRID)
    assert gap["chosen_feasible"] is False
