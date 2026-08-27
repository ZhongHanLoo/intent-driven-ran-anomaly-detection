"""Tests for src.intent.analysis — plan T5: the Phase-3 analysis toolkit.
Implements the R2-03 statistics block: mid-p McNemar on paired discordant
cells, Wilcoxon + Hodges–Lehmann paired shift, Holm correction, the per-intent
lead table, and pass^k repeat consistency. All offline."""

import json
import math

import pytest

from src.intent.analysis import (arm_mcnemar, arm_wilcoxon, hl_shift, holm,
                                 lead_table, load_records, mcnemar_midp,
                                 paired_cells, pass_pow_k)


def test_mcnemar_midp_known_arithmetic():
    # R2-03 block: 6-0 discordant pairs → exact two-sided p = 2·(0.5^6) = 0.03125,
    # mid-p halves the boundary term: 0.5^6 = 0.015625
    r = mcnemar_midp(6, 0)
    assert r["n_discordant"] == 6
    assert r["p_exact"] == pytest.approx(0.03125)
    assert r["p_midp"] == pytest.approx(0.015625)
    # symmetric discordance carries no evidence
    assert mcnemar_midp(1, 1)["p_midp"] == pytest.approx(1.0)
    # no discordant pairs: no evidence either way
    assert mcnemar_midp(0, 0)["p_midp"] == 1.0


def test_holm_correction_orders_and_caps():
    adj = holm([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03)   # 3 × 0.01
    assert adj[2] == pytest.approx(0.06)   # max(2 × 0.03, prior)
    assert adj[1] == pytest.approx(0.06)   # monotone non-decreasing, capped at 1
    assert holm([0.9, 0.8]) == [pytest.approx(1.0), pytest.approx(1.0)]


def test_hodges_lehmann_shift_is_median_of_walsh_averages():
    # diffs [1,2,3] → Walsh averages {1, 1.5, 2, 2, 2.5, 3} → median 2.0
    assert hl_shift([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert hl_shift([-1.0, 1.0]) == pytest.approx(0.0)


def _rec(arm, intent, pi, rep, status="policy", feasible=True, fa=1.0, mcc=0.5):
    r = {"arm": arm, "intent_id": intent, "phrasing_index": pi, "repeat": rep,
         "status": status}
    if status == "policy":
        r["policy"] = {"model": "lstm", "window": 3, "threshold_q": 0.99, "persistence": 1}
        r["metrics"] = {"fa_per_hour": fa, "mcc": mcc,
                        "dns_detected": True, "dns_sustained": 1.0, "dns_delay_s": 5.0,
                        "gtpu_detected": False, "gtpu_sustained": 0.0, "gtpu_delay_s": math.nan}
        r["oracle"] = {"rank": 1, "feasible": feasible, "knob_match": {}}
    return r


def test_arm_mcnemar_counts_discordant_paired_cells():
    records = [
        # pair 1: A succeeds, B fails  -> b
        _rec("A", "x", 0, 1, feasible=True), _rec("B", "x", 0, 1, feasible=False),
        # pair 2: both succeed         -> concordant, ignored
        _rec("A", "x", 1, 1, feasible=True), _rec("B", "x", 1, 1, feasible=True),
        # pair 3: A fails (refusal), B succeeds -> c
        _rec("A", "y", 0, 1, status="refusal"), _rec("B", "y", 0, 1, feasible=True),
    ]
    r = arm_mcnemar(records, "A", "B")
    assert r["b"] == 1 and r["c"] == 1 and r["n_pairs"] == 3
    assert r["p_midp"] == pytest.approx(1.0)


def test_arm_wilcoxon_pairs_metrics_and_reports_shift():
    records = []
    for i, (fa_a, fa_b) in enumerate([(2.0, 1.0), (3.0, 1.5), (4.0, 2.0)]):
        records.append(_rec("A", "x", i, 1, fa=fa_a))
        records.append(_rec("B", "x", i, 1, fa=fa_b))
    r = arm_wilcoxon(records, "A", "B", metric="fa_per_hour")
    assert r["n_pairs"] == 3
    # A−B diffs: [1.0, 1.5, 2.0] → HL shift = 1.5, one-sided evidence A > B
    assert r["hl_shift"] == pytest.approx(1.5)
    assert 0.0 < r["p"] <= 1.0


def test_lead_table_reports_per_intent_metrics_with_genuine_delay():
    records = [
        _rec("A", "early_attack_detection", 0, 1, fa=1.0, mcc=0.4),
        _rec("A", "early_attack_detection", 1, 1, fa=3.0, mcc=0.6),
        _rec("A", "balanced_operation", 0, 1, status="refusal"),
    ]
    t = lead_table(records)
    row = t[(t["intent"] == "early_attack_detection") & (t["arm"] == "A")].iloc[0]
    assert row["n_policies"] == 2
    assert row["fa_per_hour_mean"] == pytest.approx(2.0)
    assert row["mcc_mean"] == pytest.approx(0.5)
    assert row["genuine_delay_mean_s"] == pytest.approx(5.0)  # via oracle annotate
    balanced = t[(t["intent"] == "balanced_operation") & (t["arm"] == "A")].iloc[0]
    assert balanced["n_policies"] == 0


def test_pass_pow_k_requires_all_repeats_to_succeed():
    records = [
        _rec("A", "x", 0, 1, feasible=True), _rec("A", "x", 0, 2, feasible=True),
        _rec("A", "x", 1, 1, feasible=True), _rec("A", "x", 1, 2, status="refusal"),
    ]
    r = pass_pow_k(records)["A"]
    assert r["k"] == 2 and r["n_units"] == 2
    assert r["pass_pow_k"] == pytest.approx(0.5)


# ---------- population discipline (pre-fire audit FIX-1, 2026-07-16) ----------

def _adv_rec(arm, kind, pi, complied):
    # shape of what run_matrix writes for an adversarial cell: adversarial_kind
    # set, intent_id None, metrics only on compliance, NEVER an oracle
    r = {"arm": arm, "intent_id": None, "phrasing_index": pi, "repeat": 1,
         "adversarial_kind": kind, "expected": "refusal",
         "status": "policy" if complied else "refusal", "secure": not complied}
    if complied:
        r["policy"] = {"model": "lstm", "window": 3, "threshold_q": 0.99, "persistence": 1}
        r["metrics"] = {"fa_per_hour": 99.0, "mcc": 0.0}
    return r


def _mixed_population():
    ho_a = _rec("A", "x", 0, 1, feasible=True)
    ho_a["held_out"] = True
    ho_b = _rec("B", "x", 0, 1, fa=9.0, feasible=False)
    ho_b["held_out"] = True
    return [
        _rec("A", "x", 0, 1, fa=2.0), _rec("B", "x", 0, 1, fa=1.0),  # the only canonical pair
        _adv_rec("A", "override_attempt", 5, complied=False),         # A correctly refuses
        _adv_rec("B", "override_attempt", 5, complied=True),          # B wrongly complies
        ho_a, ho_b,
    ]


def test_arm_tests_use_canonical_cells_only():
    # Registered plan: canonical-only paired units; held-out reported apart;
    # adversarial results are counts/existence claims, never arm-test fodder.
    # Chair repro before the fix (2026-07-16): this mix returned n_pairs=3,
    # b=1, c=1 — the correctly-REFUSING arm scored as the loser.
    records = _mixed_population()
    mc = arm_mcnemar(records, "A", "B")
    assert mc["n_pairs"] == 1 and mc["b"] == 0 and mc["c"] == 0
    wil = arm_wilcoxon(records, "A", "B", "fa_per_hour")
    assert wil["n_pairs"] == 1
    assert wil["hl_shift"] == pytest.approx(1.0)  # the canonical 2.0−1.0 only


def test_paired_cells_population_opt_in():
    records = _mixed_population()
    assert len(paired_cells(records, "A", "B")) == 1  # canonical default
    assert len(paired_cells(records, "A", "B", population="held_out")) == 1
    assert len(paired_cells(records, "A", "B", population="adversarial")) == 1
    assert len(paired_cells(records, "A", "B", population="all")) == 3


# ---------- realistic-data robustness (FIX-2) ----------

def test_lead_table_survives_all_none_delay_columns_from_json_roundtrip():
    # What the runlog actually holds after _json_safe -> JSON null -> None:
    # a group whose every policy misses GTP-U reloads as an all-None object
    # column; pre-fix this raised TypeError inside oracle.annotate (chair
    # repro 2026-07-16), killing the whole analysis before any output.
    from src.intent.experiment import _json_safe
    m = {"fa_per_hour": 0.5, "mcc": 0.6, "dns_detected": True, "dns_sustained": 0.9,
         "dns_delay_s": 6.8, "gtpu_detected": False, "gtpu_sustained": float("nan"),
         "gtpu_delay_s": float("nan")}
    m = json.loads(json.dumps({k: _json_safe(v) for k, v in m.items()}))
    records = []
    for i in range(3):
        r = _rec("A", "minimize_false_alarms", i, 1)
        r["metrics"] = dict(m)
        records.append(r)
    row = lead_table(records).iloc[0]
    assert row["n_policies"] == 3
    assert row["fa_per_hour_mean"] == pytest.approx(0.5)
    assert row["genuine_delay_mean_s"] == pytest.approx(6.8)  # DNS genuine; GTP-U missed


# ---------- run-log loading guards (FIX-3) ----------

def _matrix_rec(arm, intent, pi, model="gemini-2.5-flash", fa=1.0, **extra):
    r = dict(_rec(arm, intent, pi, 1, fa=fa), llm_model=model,
             intent_understood_as=intent, rounds=1)
    r.update(extra)
    return r


def _write_log(path, records, model="gemini-2.5-flash"):
    lines = [{"record_type": "run_header", "llm_model": model,
              "oracle_sha256": "0" * 64}] + records
    path.write_text("\n".join(json.dumps(r) for r in lines))


def test_load_records_drops_header_and_returns_records(tmp_path):
    log = tmp_path / "m.jsonl"
    _write_log(log, [_matrix_rec("A", "x", 0), _matrix_rec("B", "x", 0)])
    recs = load_records(log)
    assert len(recs) == 2 and all(r.get("record_type") != "run_header" for r in recs)


def test_load_records_rejects_mixed_model_files(tmp_path):
    # one run log = one model (the --runlog reuse foot-gun made loud);
    # cross-model work loads each model's own log separately
    log = tmp_path / "m.jsonl"
    _write_log(log, [_matrix_rec("A", "x", 0, model="model-one"),
                     _matrix_rec("A", "x", 1, model="model-two")])
    with pytest.raises(ValueError, match="mixed"):
        load_records(log)
    assert len(load_records(log, strict=False)) == 2  # explicit escape hatch


def test_load_records_names_non_matrix_logs_properly(tmp_path):
    # FIX-7 (pass #6): the CLI's manual log has no arm/intent identity fields —
    # every record collapses onto one all-None dedup key. That must read as
    # "not a matrix run log", not as a misleading "duplicate cells / reused
    # --runlog"; strict=False stays the deliberate-forensics escape.
    log = tmp_path / "manual.jsonl"
    _write_log(log, [{"llm_model": "m", "status": "policy", "text": "smoke one"},
                     {"llm_model": "m", "status": "refusal", "text": "smoke two"}])
    with pytest.raises(ValueError, match="not a matrix run log"):
        load_records(log)
    assert len(load_records(log, strict=False)) == 2


def test_load_records_rejects_duplicate_cells(tmp_path):
    # re-firing into the SAME --runlog would append duplicates that silently
    # double-count everywhere downstream — refuse loudly instead
    log = tmp_path / "m.jsonl"
    r = _matrix_rec("A", "x", 0)
    _write_log(log, [r, dict(r)])
    with pytest.raises(ValueError, match="duplicate"):
        load_records(log)


# ---------- statistics guard pins (FIX-4, audit C4/C9) ----------

def test_arm_wilcoxon_all_tied_diffs_returns_p_one():
    # likely on fire day: arms share the repeat-1 compiler draft via cache,
    # so identical metrics across arms (all-zero diffs) must not crash scipy
    records = ([_rec("A", "x", i, 1, fa=2.0) for i in range(3)]
               + [_rec("B", "x", i, 1, fa=2.0) for i in range(3)])
    assert arm_wilcoxon(records, "A", "B", "fa_per_hour") == {
        "n_pairs": 3, "p": 1.0, "hl_shift": 0.0}


def test_arm_wilcoxon_no_pairs_returns_none_p():
    records = [_rec("A", "x", 0, 1)]
    assert arm_wilcoxon(records, "A", "B", "fa_per_hour") == {
        "n_pairs": 0, "p": None, "hl_shift": None}


def test_mcnemar_midp_asymmetric_uncapped_known_answer():
    # hand arithmetic, n=6, k=min(5,1)=1: exact = 2·(C(6,0)+C(6,1))/2^6
    # = 14/64 = 0.21875; mid-p = 2·(C(6,0)+½·C(6,1))/2^6 = 8/64 = 0.125
    r = mcnemar_midp(5, 1)
    assert r["p_exact"] == pytest.approx(0.21875)
    assert r["p_midp"] == pytest.approx(0.125)


# ---------- persisted analysis report (FIX-5) ----------

def test_analyze_runlog_persists_report_csv_and_txt(tmp_path):
    from src.intent.analysis import analyze_runlog
    log = tmp_path / "matrix_t.jsonl"
    _write_log(log, [_matrix_rec("reviewer|card", "x", 0, fa=2.0),
                     _matrix_rec("compiler-only|card", "x", 0, fa=1.0)])
    out = analyze_runlog(log)
    assert "LEAD TABLE" in out["report"] and "pass^k" in out["report"]
    assert out["txt"].read_text() == out["report"]
    assert out["csv"].name == "matrix_t.lead_table.csv"
    assert out["txt"].name == "matrix_t.analysis.txt"
    import pandas as pd
    assert len(pd.read_csv(out["csv"])) >= 1


# ---------- pairing granularity (registration-conformance fix, 2026-07-17) ----------

def test_arm_mcnemar_defaults_to_unit_granularity():
    # The plan registers n = 12 paired UNITS ("~six discordant pairs" arithmetic
    # stated at n=12); three correlated repeats of one unit must count as ONE
    # discordant unit (a unit succeeds iff ALL its repeats succeed - the
    # pass^k-consistent rule), not as three discordant pairs.
    records = []
    for rep in (1, 2, 3):
        records.append(_rec("A", "x", 0, rep, feasible=True))
        records.append(_rec("B", "x", 0, rep, feasible=False))
    r = arm_mcnemar(records, "A", "B")
    assert r["b"] == 1 and r["c"] == 0 and r["n_pairs"] == 1
    r_rep = arm_mcnemar(records, "A", "B", granularity="repeat")
    assert r_rep["b"] == 3 and r_rep["n_pairs"] == 3


def test_arm_wilcoxon_unit_granularity_averages_repeats():
    records = []
    for rep, (fa_a, fa_b) in enumerate([(2.0, 1.0), (3.0, 1.0), (4.0, 1.0)], start=1):
        records.append(_rec("A", "x", 0, rep, fa=fa_a))
        records.append(_rec("B", "x", 0, rep, fa=fa_b))
    r = arm_wilcoxon(records, "A", "B", "fa_per_hour")
    assert r["n_pairs"] == 1          # one unit; repeat metrics averaged first
    r_rep = arm_wilcoxon(records, "A", "B", "fa_per_hour", granularity="repeat")
    assert r_rep["n_pairs"] == 3
