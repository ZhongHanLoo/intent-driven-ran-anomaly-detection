"""Deterministic oracle planner — EVALUATION ONLY, never consulted
at runtime (fairness firewall). For each canonical intent it finds the
evidence-optimal policies by exhaustive lookup over policy_grid.csv, and
scores how far an LLM-chosen policy falls from optimal.

Disclosed constants (fixed 2026-07-04):
- FA/h budgets: early 1.0 (strict operations tolerance), balanced 5.0 (loose),
  defend-unknown 25.0 (deliberately generous — this intent buys sensitivity;
  admits the AE's default operating point at 24.3 FA/h).
- genuine detection = detected AND sustained fraction >= 0.3 (the Phase-1
  honesty rule, unchanged).

Objective keys are tuples where LOWER IS BETTER (lexicographic):
- early_attack_detection:  (-n_genuine, mean_genuine_delay_s)      s.t. fa <= 1.0
- minimize_false_alarms:   (fa_per_hour, dns_delay_s)              s.t. genuine DNS
- balanced_operation:      (-mcc,)                                 s.t. fa <= 5.0
- defend_unknown_attacks:  (-n_genuine, fa_per_hour, mean_delay)   s.t. fa <= 25.0
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BUDGETS_FA_PER_HOUR = {"early_attack_detection": 1.0,
                       "balanced_operation": 5.0,
                       "defend_unknown_attacks": 25.0}
GENUINE_SUSTAINED = 0.3
KNOBS = ("model", "window", "threshold_q", "persistence")


def annotate(grid: pd.DataFrame) -> pd.DataFrame:
    """Add genuine-detection columns (idempotent; returns a copy)."""
    g = grid.copy()
    g["genuine_dns"] = g.dns_detected.astype(bool) & (g.dns_sustained >= GENUINE_SUSTAINED)
    g["genuine_gtpu"] = g.gtpu_detected.astype(bool) & (g.gtpu_sustained >= GENUINE_SUSTAINED)
    g["n_genuine"] = g.genuine_dns.astype(int) + g.genuine_gtpu.astype(int)
    stack = np.column_stack([np.where(g.genuine_dns, g.dns_delay_s, np.nan),
                             np.where(g.genuine_gtpu, g.gtpu_delay_s, np.nan)])
    cnt = (~np.isnan(stack)).sum(axis=1)
    g["mean_genuine_delay_s"] = np.where(cnt > 0, np.nansum(stack, axis=1) / np.maximum(cnt, 1), np.nan)
    return g


def _row(r) -> dict:
    return r if isinstance(r, dict) else r.to_dict()


def _annotate_one(row: dict) -> dict:
    """Annotate a single metrics dict (e.g. straight from apply_policy)."""
    if "n_genuine" in row:
        return row
    one = annotate(pd.DataFrame([row]))
    return one.iloc[0].to_dict()


def feasible(intent_id: str, row, budgets: dict | None = None) -> bool:
    """budgets overrides BUDGETS_FA_PER_HOUR — evaluation-side sensitivity
    sweeps only (plan T4); the disclosed defaults stay the runtime truth."""
    r = _annotate_one(_row(row))
    if intent_id == "minimize_false_alarms":
        return bool(r["genuine_dns"])
    return float(r["fa_per_hour"]) <= (budgets or BUDGETS_FA_PER_HOUR)[intent_id]


def objective_key(intent_id: str, row) -> tuple:
    """Comparable tuple, lower = better. NaN delays sort last (treated as inf)."""
    r = _annotate_one(_row(row))
    delay = r.get("mean_genuine_delay_s")
    delay = math.inf if delay is None or (isinstance(delay, float) and math.isnan(delay)) else float(delay)
    if intent_id == "early_attack_detection":
        return (-int(r["n_genuine"]), delay)
    if intent_id == "minimize_false_alarms":
        dns = float(r["dns_delay_s"]) if not pd.isna(r["dns_delay_s"]) else math.inf
        return (float(r["fa_per_hour"]), dns)
    if intent_id == "balanced_operation":
        return (-float(r["mcc"]),)
    if intent_id == "defend_unknown_attacks":
        return (-int(r["n_genuine"]), float(r["fa_per_hour"]), delay)
    raise KeyError(f"unknown intent '{intent_id}'")


def oracle_policies(intent_id: str, grid: pd.DataFrame, top: int = 3,
                    budgets: dict | None = None) -> pd.DataFrame:
    """Feasible policies sorted best-first; the head is the oracle's choice."""
    ann = annotate(grid)
    mask = [feasible(intent_id, r, budgets=budgets) for _, r in ann.iterrows()]
    feas = ann[mask]
    if feas.empty:
        raise ValueError(f"no feasible policy in the grid for '{intent_id}'")
    keys = feas.apply(lambda r: objective_key(intent_id, r), axis=1)
    return feas.loc[keys.sort_values(kind="stable").index].head(top)


def oracle_gap(intent_id: str, chosen, grid: pd.DataFrame) -> dict:
    """How far is a chosen policy (metrics dict/row) from the oracle's best?
    rank_of_chosen: 1 = optimal (count of strictly better feasible keys + 1).

    Knob agreement is judged against the SET of key-tied
    optimal rows — co-optimal policies are common (e.g. balanced_operation's
    MCC is persistence-invariant, a built-in 4-way tie) — and threshold_q also
    matches on the RESOLVED threshold, so ae's 'default' (physically its
    q0.995) is never penalized by string inequality. Disclosure: the 336-cell
    grid holds 324 distinct policies (ae default ≡ q0.995, 3 windows × 4
    persistences)."""
    chosen = _annotate_one(_row(chosen))
    ann = annotate(grid)
    feas_rows = [r for _, r in ann.iterrows() if feasible(intent_id, r)]
    feas_keys = [objective_key(intent_id, r) for r in feas_rows]
    if not feas_keys:
        raise ValueError(f"no feasible policy in the grid for '{intent_id}'")
    best_key = min(feas_keys)
    best_rows = [r for r, k in zip(feas_rows, feas_keys) if k == best_key]

    def _knob_ok(k: str) -> bool:
        if str(chosen.get(k)) in {str(br[k]) for br in best_rows}:
            return True
        if k == "threshold_q":  # 'default' aliases a quantile: compare resolved values
            cr = chosen.get("threshold_resolved")
            if cr is not None and not pd.isna(cr):
                return any(abs(float(br["threshold_resolved"]) - float(cr)) < 1e-12
                           for br in best_rows
                           if "threshold_resolved" in br and not pd.isna(br["threshold_resolved"]))
        return False

    out = {"chosen_feasible": feasible(intent_id, chosen),
           "n_feasible": len(feas_keys),
           "n_tied_optima": len(best_rows),
           "best_key": best_key,
           "chosen_key": objective_key(intent_id, chosen),
           "knob_match": {k: _knob_ok(k) for k in KNOBS}}
    out["rank_of_chosen"] = (1 + sum(1 for k in feas_keys if k < out["chosen_key"])
                             if out["chosen_feasible"] else None)
    return out
