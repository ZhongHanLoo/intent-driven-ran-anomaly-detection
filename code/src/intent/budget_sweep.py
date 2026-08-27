"""Budget-sensitivity sweep (plan T4; R2-04 B-C) — offline, oracle-only, zero
LLM cost. For each budgeted intent, the FA/h budget is scaled by a factor grid
and the oracle winner recomputed: the report states whether the headline
policy choices are budget-robust or artifacts of the chosen constants
(NP-ROC-style range reporting, `tong2018np`). Winner and feasibility only —
event counts behind realized FA/h belong to the live-matrix analysis, where
per-policy replay metrics exist.

  PYTHONPATH=code .venv/bin/python -m src.intent.budget_sweep
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from src.intent.oracle import BUDGETS_FA_PER_HOUR, KNOBS, annotate, feasible, oracle_policies


def _winner(intent_id: str, grid: pd.DataFrame, budgets: dict):
    try:
        return oracle_policies(intent_id, grid, top=1, budgets=budgets).iloc[0]
    except ValueError:  # empty feasible set under a tightened budget
        return None


def budget_sweep(grid: pd.DataFrame, factors=(0.5, 1.0, 2.0)) -> pd.DataFrame:
    """One row per budgeted intent × factor: swept budget, feasible-set size,
    winning policy knobs, the winner's FA/h, and whether the winner moved
    relative to the disclosed (factor 1.0) budget."""
    ann = annotate(grid)
    rows = []
    for intent, base_b in sorted(BUDGETS_FA_PER_HOUR.items()):
        ref = _winner(intent, grid, {**BUDGETS_FA_PER_HOUR, intent: base_b})
        ref_key = tuple(str(ref[k]) for k in KNOBS) if ref is not None else None
        for f in factors:
            budgets = {**BUDGETS_FA_PER_HOUR, intent: base_b * f}
            w = _winner(intent, grid, budgets)
            w_key = tuple(str(w[k]) for k in KNOBS) if w is not None else None
            rows.append({
                "intent": intent, "factor": f, "budget_fa_per_hour": base_b * f,
                "n_feasible": int(sum(feasible(intent, r, budgets=budgets)
                                      for _, r in ann.iterrows())),
                **{f"winner_{k}": (w[k] if w is not None else None) for k in KNOBS},
                "winner_fa_per_hour": (float(w["fa_per_hour"]) if w is not None else math.nan),
                "winner_moved": (w_key != ref_key),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover — offline sweep entry point
    art = Path(__file__).resolve().parents[3] / "code" / "artifacts"
    grid = pd.read_csv(art / "policy_grid.csv")
    df = budget_sweep(grid)
    out = art / "budget_sweep.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")
