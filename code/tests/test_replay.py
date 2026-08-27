"""Tests for src.replay — written FIRST (TDD). Detection-delay semantics:
alarm EVENT = the moment a UE's consecutive above-threshold window count
reaches `persistence` (counter resets on a below-threshold window or a time
gap > gap_reset_s). Incident delay = earliest participant alarm within the
attack window, minus window start."""

import numpy as np
import pandas as pd
import pytest

from src.replay import alarm_events, false_alarms_per_hour, incident_delays


def _ts(*seconds):
    base = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    return np.array([(base + pd.Timedelta(seconds=s)).to_datetime64() for s in seconds])


def test_alarm_events_persistence_and_reset():
    ts = _ts(0, 5, 10, 15, 20)
    ue = np.array(["A"] * 5)
    scores = np.array([0.1, 0.9, 0.9, 0.1, 0.9])
    e1 = alarm_events(scores, ts, ue, threshold=0.5, persistence=1)
    assert [t for _, t in e1] == list(_ts(5, 20))  # re-arms only after a reset
    e2 = alarm_events(scores, ts, ue, threshold=0.5, persistence=2)
    assert [t for _, t in e2] == list(_ts(10))  # needs two consecutive hits


def test_alarm_events_latch_false_refires_every_qualifying_window():
    # latch=False: every window meeting the persistence run is an event (used for
    # delay, where a prior latched alarm must not suppress in-window detection).
    ts = _ts(0, 5, 10, 15)
    ue = np.array(["A"] * 4)
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    e = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False)
    assert [t for _, t in e] == list(_ts(0, 5, 10, 15))  # fires every window


def test_delay_not_suppressed_by_prewindow_alarm():
    # A latched pre-window alarm must NOT hide a sustained in-window attack.
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {4: (start, start + pd.Timedelta(hours=1), {"A"})}
    ts = _ts(-10, -5, 5, 10)  # two before window, two inside
    ue = np.array(["A"] * 4)
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    ev_latched = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=True)
    ev_delay = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False)
    # latched stream fires only once (pre-window) -> would wrongly miss the incident
    assert incident_delays(ev_latched, incidents)[4]["detected"] is False
    # non-latched stream detects at the first in-window window (+5 s)
    assert incident_delays(ev_delay, incidents)[4]["delay_s"] == 5.0


def test_alarm_events_gap_resets_counter():
    ts = _ts(0, 100)  # 100 s gap > gap_reset_s
    ue = np.array(["A", "A"])
    scores = np.array([0.9, 0.9])
    assert alarm_events(scores, ts, ue, 0.5, persistence=2, gap_reset_s=30) == []


def test_alarm_events_do_not_mix_ues():
    ts = _ts(0, 5)
    ue = np.array(["A", "B"])
    scores = np.array([0.9, 0.9])
    e = alarm_events(scores, ts, ue, 0.5, persistence=2)
    assert e == []  # two different UEs never form one streak


def test_incident_delays_first_participant_alarm():
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {4: (start, start + pd.Timedelta(hours=1), {"A", "B"})}
    events = [("C", _ts(3)[0]), ("A", _ts(12)[0]), ("A", _ts(30)[0]), ("B", _ts(7)[0])]
    d = incident_delays(events, incidents)
    assert d[4]["detected"] and d[4]["delay_s"] == 7.0  # B first; C not a participant


def test_alarm_events_with_start_returns_streak_origin():
    # with_start=True returns (ue, event_ts, streak_start_ts). A streak that
    # begins before the window keeps its pre-window origin.
    ts = _ts(-10, -5, 5)
    ue = np.array(["A"] * 3)
    scores = np.array([0.9, 0.9, 0.9])
    e = alarm_events(scores, ts, ue, 0.5, persistence=3, with_start=True)
    assert len(e[0]) == 3 and e[0][2] == _ts(-10)[0]  # streak origin is the first hit


def test_incident_delays_fresh_only_rejects_standing_alarm():
    # A model already alarming before onset must NOT be credited a ~0 s detection.
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {5: (start, start + pd.Timedelta(hours=1), {"A"})}
    ts = _ts(-10, -5, 5, 10)  # contiguous 5 s cadence, no gap: streak carries into window
    ue = np.array(["A"] * 4)
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    ev = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False, with_start=True)
    # standing alarm (streak origin -10 s, before onset) -> not a fresh detection
    assert incident_delays(ev, incidents, fresh_only=True)[5]["detected"] is False
    # without the guard, the old behaviour would (wrongly) report a tiny delay
    assert incident_delays(ev, incidents, fresh_only=False)[5]["detected"] is True


def test_incident_delays_fresh_only_accepts_in_window_onset():
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {5: (start, start + pd.Timedelta(hours=1), {"A"})}
    ts = _ts(-10, -5, 5, 10)  # quiet before (contiguous), streak STARTS in-window
    ue = np.array(["A"] * 4)
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    ev = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False, with_start=True)
    d = incident_delays(ev, incidents, fresh_only=True)[5]
    assert d["detected"] and d["delay_s"] == 5.0


def test_incident_delays_missed():
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {5: (start, start + pd.Timedelta(hours=1), {"A"})}
    d = incident_delays([("A", (start - pd.Timedelta(seconds=10)).to_datetime64())], incidents)
    assert not d[5]["detected"] and d[5]["delay_s"] is None  # alarm before window


def test_false_alarms_per_hour():
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {4: (start, start + pd.Timedelta(hours=1), {"A"})}
    events = [("A", _ts(10)[0]),          # participant in window: not an FA
              ("B", _ts(10)[0]),          # innocent UE during window: FA
              ("A", (start + pd.Timedelta(hours=2)).to_datetime64())]  # outside: FA
    fa = false_alarms_per_hour(events, incidents, benign_ue_hours=4.0)
    assert fa == pytest.approx(2 / 4.0)


def test_false_alarms_per_hour_accepts_with_start_events():
    # The same event stream must serve both metrics: FA counting may not crash
    # on the 3-tuple (ue, event_ts, streak_start_ts) events that with_start=True
    # produces for delay measurement.
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {4: (start, start + pd.Timedelta(hours=1), {"A"})}
    late = (start + pd.Timedelta(hours=2)).to_datetime64()
    events3 = [("A", _ts(10)[0], _ts(10)[0]),   # participant in window: not an FA
               ("B", _ts(10)[0], _ts(5)[0]),    # innocent UE during window: FA
               ("A", late, late)]               # outside any window: FA
    fa = false_alarms_per_hour(events3, incidents, benign_ue_hours=4.0)
    assert fa == pytest.approx(2 / 4.0)


def test_fresh_only_gap_straddling_onset_counts_as_fresh():
    # Episode semantics, pinned: a pre-onset alarm
    # separated from its in-window continuation by a silence > gap_reset_s is a
    # NEW streak — its origin lies in-window, so fresh_only credits it.
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {5: (start, start + pd.Timedelta(hours=1), {"A"})}
    ts = _ts(-40, -35, 20)  # 55 s of silence straddles the onset (> 30 s reset)
    ue = np.array(["A"] * 3)
    scores = np.array([0.9, 0.9, 0.9])
    ev = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False, with_start=True)
    d = incident_delays(ev, incidents, fresh_only=True)[5]
    assert d["detected"] and d["delay_s"] == 20.0


def test_fresh_only_credits_first_fresh_streak_after_midwindow_dip():
    # The AE-DNS mechanism: a standing pre-onset
    # streak is never credited even while the attack rages; the first streak
    # that STARTS in-window (after a dip) is — delay lands at the dip-recovery
    # time, honestly late. Pins the conservative side of the fresh_only guard.
    start = pd.Timestamp("2024-08-21 12:00:00", tz="UTC")
    incidents = {4: (start, start + pd.Timedelta(hours=1), {"A"})}
    ts = _ts(-10, -5, 5, 10, 15, 20)
    ue = np.array(["A"] * 6)
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.9])  # dip at +15, fresh rise at +20
    ev = alarm_events(scores, ts, ue, 0.5, persistence=1, latch=False, with_start=True)
    d = incident_delays(ev, incidents, fresh_only=True)[4]
    assert d["detected"] and d["delay_s"] == 20.0
