"""Tests for src.intent.apply — policy execution off archived scores.
Regression pin: results must match the committed delay_temporal.csv and
metrics JSONs exactly (same replay functions, same archives)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.intent.apply import apply_policy, grid
from src.intent.registry import Registry
from src.intent.schema import PolicyKnobs

ART = Path(__file__).resolve().parents[1] / "artifacts"
REG = Registry(ART)


def test_ae_default_matches_delay_temporal_row():
    m = apply_policy(PolicyKnobs(model="ae", window=3, threshold_q="default", persistence=1), REG)
    td = pd.read_csv(ART / "delay_temporal.csv")
    stored_thr = json.load(open(ART / "ae_w3_temporal_metrics.json"))["threshold"]
    row = td[(td.model == "ae") & (td.window == 3) & (td.persistence == 1)
             & (abs(td.threshold - stored_thr) < 1e-9)].iloc[0]
    assert m["fa_per_hour"] == pytest.approx(row.fa_per_hour, rel=1e-6)
    assert m["dns_delay_s"] == pytest.approx(row.dns_delay_s, rel=1e-6)
    assert m["gtpu_delay_s"] == pytest.approx(row.gtpu_delay_s, rel=1e-6)
    assert m["dns_sustained"] == pytest.approx(row.dns_sustained, rel=1e-6)


def test_window_metrics_match_metrics_json_at_default():
    m = apply_policy(PolicyKnobs(model="ae", window=3, threshold_q="default", persistence=1), REG)
    ref = json.load(open(ART / "ae_w3_temporal_metrics.json"))
    assert m["fpr"] == pytest.approx(ref["fpr"], rel=1e-6)
    assert m["fnr"] == pytest.approx(ref["fnr"], rel=1e-6)
    assert m["macro_f1"] == pytest.approx(ref["macro_f1"], rel=1e-6)


def test_quantile_policy_runs_and_reports_all_keys():
    m = apply_policy(PolicyKnobs(model="transformer", window=5, threshold_q=0.999, persistence=3), REG)
    for k in ["threshold_resolved", "fa_per_hour", "dns_delay_s", "gtpu_delay_s",
              "dns_sustained", "gtpu_sustained", "dns_detected", "gtpu_detected",
              "fpr", "fnr", "macro_f1", "mcc"]:
        assert k in m
    assert m["threshold_resolved"] == pytest.approx(REG.resolve_threshold("transformer", 5, 0.999))


def test_mini_grid_shape(tmp_path):
    df = grid(REG, models=["ae"], windows=[3], qs=[0.995, "default"], persists=[1, 3],
              out_csv=tmp_path / "mini.csv")
    assert len(df) == 4 and (tmp_path / "mini.csv").exists()
    assert {"model", "window", "threshold_q", "persistence", "fa_per_hour"} <= set(df.columns)


def test_missed_incident_delays_are_float_nan_and_oracle_gradeable():
    # FIRE-NIGHT ERRATUM (2026-07-16, leg-1 crash at experiment.py:156):
    # replay's incident_delays reports a missed incident as delay_s=None;
    # apply_policy must normalize that to float NaN. A None reaching
    # oracle_gap's single-row annotate() makes an object-dtype column and
    # np.isnan raises TypeError — killing the matrix at the first quiet
    # policy that genuinely misses an incident (expected for the
    # minimize_false_alarms intent). Grid rows are immune (the CSV
    # round-trip stores NaN); only the live dict path carried None.
    import math

    from src.intent.oracle import oracle_gap
    m = apply_policy(PolicyKnobs(model="lstm", window=7, threshold_q=0.9, persistence=1), REG)
    assert m["dns_detected"] is False          # this policy genuinely misses DNS
    assert isinstance(m["dns_delay_s"], float) and math.isnan(m["dns_delay_s"])
    assert isinstance(m["gtpu_delay_s"], float)
    g = oracle_gap("minimize_false_alarms", m, pd.read_csv(ART / "policy_grid.csv"))
    assert g["chosen_feasible"] is False       # DNS missed -> infeasible here
    assert g["rank_of_chosen"] is None         # infeasible policies get no rank
