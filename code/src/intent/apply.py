"""Executor: a policy selects an archived score file; threshold +
persistence produce alarms via src.replay (latch=True for FA counting,
latch=False + fresh_only for delay — fixed semantics); window
metrics via src.evaluate. Never trains or loads a network. Grid mode sweeps
the full legal knob space -> policy_grid.csv (the Phase-3 oracle's table)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import incidents_dict
from src.evaluate import compute_metrics
from src.intent.knobs import MODELS, PERSISTENCES, THRESHOLD_QS, WINDOWS
from src.intent.schema import PolicyKnobs
from src.replay import alarm_events, false_alarms_per_hour, incident_delays

MEDIAN_DT_S = 5.14  # same constant as eda/delay_analysis.py


def apply_policy(policy: PolicyKnobs, registry) -> dict:
    thr = registry.resolve_threshold(policy.model, policy.window, policy.threshold_q)
    z = np.load(registry.score_archive(policy.model, policy.window), allow_pickle=True)
    scores, y, ts, ue = z["scores"], z["y"], z["ts"], z["ue"]
    inc = incidents_dict([4, 5])
    benign_hours = float((y == 0).sum()) * MEDIAN_DT_S / 3600
    fa_ev = alarm_events(scores, ts, ue, thr, persistence=policy.persistence, latch=True)
    dl_ev = alarm_events(scores, ts, ue, thr, persistence=policy.persistence,
                         latch=False, with_start=True)
    d = incident_delays(dl_ev, inc, fresh_only=True)
    tsx = pd.DatetimeIndex(ts)

    def sustained(k):
        s0, e0 = (t.tz_localize(None) for t in inc[k][:2])
        mask = (tsx >= s0) & (tsx < e0) & np.isin(ue, list(inc[k][2]))
        return float((scores[mask] >= thr).mean()) if mask.sum() else 0.0

    def delay_f(v):
        # replay reports a missed incident as delay_s=None; grading needs float
        # NaN — a None makes oracle annotate()'s single-row frame object-dtype
        # and np.isnan raises (fire-night erratum 2026-07-16, leg-1 crash)
        return float("nan") if v is None else float(v)

    m = compute_metrics(y, scores, threshold=thr)
    m.update({"model": policy.model, "window": policy.window,
              "threshold_q": policy.threshold_q, "persistence": policy.persistence,
              "threshold_resolved": float(thr),
              "fa_per_hour": false_alarms_per_hour(fa_ev, inc, benign_hours),
              "dns_delay_s": delay_f(d[4]["delay_s"]), "dns_detected": d[4]["detected"],
              "gtpu_delay_s": delay_f(d[5]["delay_s"]), "gtpu_detected": d[5]["detected"],
              "dns_sustained": sustained(4), "gtpu_sustained": sustained(5)})
    return m


def grid(registry, models=MODELS, windows=WINDOWS,
         qs=(*THRESHOLD_QS, "default"), persists=PERSISTENCES,
         out_csv: Path | None = None) -> pd.DataFrame:
    rows = []
    for mo in models:
        for w in windows:
            if (mo, w) not in registry.pairs():
                continue
            for q in qs:
                for p in persists:
                    rows.append(apply_policy(
                        PolicyKnobs(model=mo, window=w, threshold_q=q, persistence=p), registry))
                    print(f"grid: {mo} w{w} q{q} p{p} done", flush=True)
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df
