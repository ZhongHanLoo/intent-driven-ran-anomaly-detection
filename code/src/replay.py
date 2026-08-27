"""Replay evaluation: turn per-window scores into alarm EVENTS, then into the
operator-facing metrics — per-incident detection delay and false alarms per hour.

Definitions (documented for the report):
- Alarm event: the moment a UE's count of consecutive above-threshold windows
  reaches `persistence`. The counter resets on a below-threshold window or a
  time gap > gap_reset_s (streaks must be contiguous observations). After an
  event the alarm is "latched" until a reset — a continuing attack re-fires
  only after the state clears (operators act on events, not on every window).
- Incident delay: earliest alarm event of any PARTICIPATING UE inside the
  attack window, minus the window start. No such event = missed incident.
- False alarm: any event that is not (participating UE ∧ inside its window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def alarm_events(scores, ts, ue, threshold: float, persistence: int = 1,
                 gap_reset_s: float = 30.0, latch: bool = True,
                 with_start: bool = False):
    """Per-UE chronological scan -> list of alarm events.

    latch=True (default): after firing, do not re-fire until a reset (below-
    threshold window or a gap > gap_reset_s). Use for counting operator-facing
    alarm events -> false alarms per hour.
    latch=False: emit an event for EVERY window whose trailing `persistence`
    run is satisfied. Use for detection delay, so a prior latched alarm cannot
    suppress a later genuine in-window detection.
    with_start=False: events are (ue, event_ts) 2-tuples.
    with_start=True:  events are (ue, event_ts, streak_start_ts) 3-tuples —
    streak_start_ts is when the current above-threshold run began. Used to
    reject "detections" that are really standing pre-onset false alarms
    (incident_delays fresh_only).
    """
    scores = np.asarray(scores)
    ts = np.asarray(ts, dtype="datetime64[ns]")
    ue = np.asarray(ue)
    events = []
    for u in pd.unique(ue):
        m = ue == u
        order = np.argsort(ts[m], kind="stable")
        t_u, s_u = ts[m][order], scores[m][order]
        streak, armed, last_t, start_t = 0, True, None, None
        for t, s in zip(t_u, s_u):
            if last_t is not None and (t - last_t) / np.timedelta64(1, "s") > gap_reset_s:
                streak, armed = 0, True
            if s >= threshold:
                if streak == 0:
                    start_t = t
                streak += 1
                if streak >= persistence and armed:
                    events.append((u, t, start_t) if with_start else (u, t))
                    if latch:
                        armed = False
            else:
                streak, armed = 0, True
            last_t = t
    events.sort(key=lambda e: e[1])
    return events


def _win64(t) -> np.datetime64:
    return t.to_datetime64() if hasattr(t, "to_datetime64") else np.datetime64(t)


def incident_delays(events, incidents: dict, fresh_only: bool = False) -> dict:
    """incidents: {k: (start, end, participants:set)} -> {k: {detected, delay_s}}.

    fresh_only=True (requires with_start events): an in-window alarm counts as a
    detection only if its streak STARTED at/after onset — so a model already
    false-alarming on that UE before the attack is not credited a spurious
    near-zero delay (the A1 fix). Events may be 2- or 3-tuples.

    Conservative by design: a pre-onset standing streak that never resets
    in-window scores as a MISS even if every in-window window is above
    threshold (without a dip, "still false-alarming" and "now detecting" are
    indistinguishable); the first streak starting in-window after a dip is
    credited, so reported delays are honest upper bounds at operating points
    with standing false alarms. Report alongside the sustained fraction.
    """
    out = {}
    for k, (start, end, parts) in incidents.items():
        s64, e64 = _win64(start), _win64(end)
        hits = []
        for ev in events:
            u, t = ev[0], ev[1]
            if u not in parts or not (s64 <= t < e64):
                continue
            if fresh_only:
                if len(ev) < 3:
                    raise ValueError("fresh_only requires alarm_events(..., with_start=True)")
                if ev[2] < s64:  # streak began before onset -> standing alarm, not a detection
                    continue
            hits.append(t)
        if hits:
            out[k] = {"detected": True, "delay_s": float((min(hits) - s64) / np.timedelta64(1, "s"))}
        else:
            out[k] = {"detected": False, "delay_s": None}
    return out


def false_alarms_per_hour(events, incidents: dict, benign_ue_hours: float) -> float:
    """Events not attributable to a participating attacker inside its window,
    normalized by observed benign UE-hours."""
    def is_true_alarm(u, t):
        return any(u in parts and _win64(s) <= t < _win64(e)
                   for s, e, parts in incidents.values())
    fa = sum(1 for ev in events if not is_true_alarm(ev[0], ev[1]))  # 2- or 3-tuple events
    return fa / benign_ue_hours
