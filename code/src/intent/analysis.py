"""Phase-3 analysis toolkit (plan T5) — implements the R2-03 statistics block.

Paired arm comparisons on the shared-cache design: mid-p McNemar on discordant
binary outcomes (`mcnemar1947`; `dietterich1998`; mid-p per `fagerland2013midp`),
Wilcoxon signed-rank with the Hodges–Lehmann paired shift for continuous
metrics (`wilcoxon1945`; `demsar2006`), Holm correction across the handful of
arm contrasts, the per-intent LEAD table (intent→metric effects
first, oracle-gap second), and pass^k repeat consistency (`yao2025taubench`).
No CLT error bars anywhere at this n (`bowyer2025clt`).

  PYTHONPATH=code .venv/bin/python -m src.intent.analysis <runlog.jsonl>
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon as _scipy_wilcoxon

from src.intent.oracle import annotate


def load_records(path, strict: bool = True) -> list[dict]:
    """Read a matrix run log, dropping the freeze/run header line.

    strict (default) enforces the one-run-per-file design (pre-fire audit
    FIX-3, 2026-07-16): a single llm_model and no duplicated cells. Both
    guards exist to make the --runlog-reuse foot-gun loud — every downstream
    aggregate would otherwise silently pool models or double-count cells."""
    lines = Path(path).read_text().splitlines()
    recs = [json.loads(l) for l in lines if l.strip()]
    recs = [r for r in recs if r.get("record_type") != "run_header"]
    if strict and recs:
        models = sorted({r["llm_model"] for r in recs if "llm_model" in r})
        if len(models) > 1:
            raise ValueError(f"mixed-model run log {path}: {models} — one run "
                             f"per file by design; analyze each model's own log "
                             f"(or pass strict=False)")
        # matrix records always carry cell identity; a log where NO record has
        # any (e.g. the CLI's manual smoke log) is a different artifact, not a
        # duplicated matrix log — name it properly instead of crying duplicates
        id_fields = ("arm", "intent_id", "phrasing_index", "repeat")
        if not any(r.get(k) is not None for r in recs for k in id_fields):
            raise ValueError(f"{path} is not a matrix run log (no record carries "
                             f"arm/intent cell identity — e.g. a CLI manual log); "
                             f"pass strict=False to load it anyway")
        seen = set()
        for r in recs:
            key = (r.get("llm_model"), r.get("arm"), r.get("intent_id"),
                   r.get("phrasing_index"), r.get("repeat"),
                   bool(r.get("held_out")), r.get("adversarial_kind"))
            if key in seen:
                raise ValueError(f"duplicate cell records in {path} (same "
                                 f"--runlog reused across runs?): {key}")
            seen.add(key)
    return recs


# ---------- statistics primitives ----------

def mcnemar_midp(b: int, c: int) -> dict:
    """Two-sided McNemar on discordant pair counts b (A-only success) and
    c (B-only success): exact conditional p and the recommended mid-p variant."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_exact": 1.0, "p_midp": 1.0}
    k = min(b, c)
    cdf_below = sum(math.comb(n, i) for i in range(k)) / 2 ** n
    pmf_k = math.comb(n, k) / 2 ** n
    return {"b": b, "c": c, "n_discordant": n,
            "p_exact": min(1.0, 2 * (cdf_below + pmf_k)),
            "p_midp": min(1.0, 2 * (cdf_below + 0.5 * pmf_k))}


def holm(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjustment; returns adjusted p in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def hl_shift(diffs) -> float:
    """Hodges–Lehmann one-sample estimator: median of Walsh averages."""
    ds = sorted(float(d) for d in diffs)
    walsh = sorted((a + b) / 2 for i, a in enumerate(ds) for b in ds[i:])
    n = len(walsh)
    mid = n // 2
    return walsh[mid] if n % 2 else (walsh[mid - 1] + walsh[mid]) / 2


# ---------- pairing over the shared-cache design ----------

def _cell_key(r: dict) -> tuple:
    return (r.get("intent_id"), r.get("phrasing_index"), r.get("repeat"),
            bool(r.get("held_out")), r.get("adversarial_kind"))


def paired_cells(records: list[dict], arm_a: str, arm_b: str,
                 *, population: str = "canonical") -> list[tuple]:
    """Match cells across two arms by (intent, phrasing, repeat, held-out,
    adversarial-kind) — the repeat-1 cache sharing makes these true pairs.

    population selects WHICH cells may form pairs (pre-fire audit FIX-1,
    2026-07-16 — the registered plan's rules): 'canonical' (default — the
    registered paired units), 'held_out' (the generalization probes, reported
    apart), 'adversarial' (counts/existence claims only; note _success has no
    security meaning there), or 'all' (no filter; caller takes responsibility)."""
    def _in(r: dict) -> bool:
        adv = r.get("adversarial_kind") is not None
        ho = bool(r.get("held_out"))
        return {"canonical": not adv and not ho,
                "held_out": ho and not adv,
                "adversarial": adv,
                "all": True}[population]
    by: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        if r["arm"] in (arm_a, arm_b) and _in(r):
            by[_cell_key(r)][r["arm"]] = r
    return [(v[arm_a], v[arm_b]) for v in by.values() if arm_a in v and arm_b in v]


def _success(r: dict) -> bool:
    """Default binary outcome: a policy that the oracle grades feasible
    (falls back to 'emitted a policy' when the oracle was not run)."""
    if r["status"] != "policy":
        return False
    o = r.get("oracle")
    return bool(o["feasible"]) if o else True


def _unit_key(r: dict) -> tuple:
    # one row per intent×phrasing unit — the plan's registered n (12/arm)
    return (r.get("intent_id"), r.get("phrasing_index"),
            bool(r.get("held_out")), r.get("adversarial_kind"))


def arm_mcnemar(records, arm_a: str, arm_b: str, success=_success,
                population: str = "canonical", granularity: str = "unit") -> dict:
    """Mid-p McNemar between two arms on paired known-outcome cells.

    granularity='unit' (default — the registration-conformance fix, 2026-07-17):
    one paired row per intent×phrasing unit, a unit succeeding iff ALL its
    repeats succeed (the pass^k-consistent rule); this matches the plan's
    registered n = 12 units/arm and its "~six discordant pairs" arithmetic.
    granularity='repeat' pairs each repeat separately (up to 36 correlated
    pairs) — supplementary only: correlated repeats inflate discordant counts."""
    pairs = [(a, b) for a, b in paired_cells(records, arm_a, arm_b, population=population)
             if a["status"] != "transport_failure" and b["status"] != "transport_failure"]
    if granularity == "unit":
        units: dict[tuple, tuple] = defaultdict(lambda: ([], []))
        for a, b in pairs:
            u = units[_unit_key(a)]
            u[0].append(success(a))
            u[1].append(success(b))
        b_ = sum(1 for sa, sb in units.values() if all(sa) and not all(sb))
        c_ = sum(1 for sa, sb in units.values() if all(sb) and not all(sa))
        n = len(units)
    else:
        b_ = sum(1 for a, b in pairs if success(a) and not success(b))
        c_ = sum(1 for a, b in pairs if success(b) and not success(a))
        n = len(pairs)
    out = mcnemar_midp(b_, c_)
    out["n_pairs"] = n
    return out


def arm_wilcoxon(records, arm_a: str, arm_b: str, metric: str,
                 population: str = "canonical", granularity: str = "unit") -> dict:
    """Wilcoxon signed-rank on paired metric differences (A − B), with the
    Hodges–Lehmann shift as the effect estimate. granularity='unit' (default)
    averages each unit's valid repeat metrics before pairing (n ≤ 12/arm);
    'repeat' pairs each repeat separately (supplementary)."""
    valid = []
    for a, b in paired_cells(records, arm_a, arm_b, population=population):
        ma, mb = a.get("metrics"), b.get("metrics")
        if not (ma and mb) or metric not in ma or metric not in mb:
            continue
        va, vb = ma[metric], mb[metric]
        if va is None or vb is None:
            continue
        va, vb = float(va), float(vb)
        if math.isnan(va) or math.isnan(vb):
            continue
        valid.append((a, va, vb))
    if granularity == "unit":
        units: dict[tuple, tuple] = defaultdict(lambda: ([], []))
        for a, va, vb in valid:
            u = units[_unit_key(a)]
            u[0].append(va)
            u[1].append(vb)
        diffs = [sum(x) / len(x) - sum(y) / len(y) for x, y in units.values() if x and y]
    else:
        diffs = [va - vb for _, va, vb in valid]
    if not diffs:
        return {"n_pairs": 0, "p": None, "hl_shift": None}
    if all(d == 0 for d in diffs):
        return {"n_pairs": len(diffs), "p": 1.0, "hl_shift": 0.0}
    return {"n_pairs": len(diffs),
            "p": float(_scipy_wilcoxon(diffs).pvalue),
            "hl_shift": hl_shift(diffs)}


# ---------- the report's lead table & consistency ----------

def lead_table(records) -> pd.DataFrame:
    """Per-intent × arm (× held-out) executed-metric means — the table that
    LEADS the analysis. Delay is the genuine-detection mean via
    the oracle's annotate(), i.e. the same rule the grid uses."""
    canon = [r for r in records if "adversarial_kind" not in r
             and r["status"] != "transport_failure"]
    rows = []
    for intent, arm, ho in sorted({(r["intent_id"], r["arm"], bool(r.get("held_out")))
                                   for r in canon}):
        cell = [r for r in canon if r["intent_id"] == intent and r["arm"] == arm
                and bool(r.get("held_out")) == ho]
        pol = [r for r in cell if r["status"] == "policy" and r.get("metrics")]
        row = {"intent": intent, "arm": arm, "held_out": ho,
               "n_cells": len(cell), "n_policies": len(pol)}
        if pol:
            mdf = pd.DataFrame([r["metrics"] for r in pol])
            # JSON round-trip turns NaN into null/None; an all-None column
            # reloads as object dtype, which annotate's isnan cannot take
            # (pre-fire audit FIX-2 — an all-missed-GTP-U group is the
            # EXPECTED shape for the minimize/early intents, not an edge case)
            for c in ("fa_per_hour", "mcc", "dns_sustained", "gtpu_sustained",
                      "dns_delay_s", "gtpu_delay_s"):
                if c in mdf:
                    mdf[c] = pd.to_numeric(mdf[c])
            mdf = annotate(mdf)
            row["fa_per_hour_mean"] = float(mdf["fa_per_hour"].mean())
            row["mcc_mean"] = float(mdf["mcc"].mean())
            delays = mdf["mean_genuine_delay_s"].dropna()
            row["genuine_delay_mean_s"] = float(delays.mean()) if not delays.empty else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pass_pow_k(records) -> dict:
    """τ-bench-style pass^k per arm: the fraction of (intent, phrasing) units
    whose EVERY repeat succeeded (canonical, known-outcome cells only)."""
    out = {}
    canon = [r for r in records if "adversarial_kind" not in r and not r.get("held_out")
             and r["status"] != "transport_failure"]
    for arm in sorted({r["arm"] for r in canon}):
        groups: dict[tuple, list] = defaultdict(list)
        for r in (r for r in canon if r["arm"] == arm):
            groups[(r["intent_id"], r["phrasing_index"])].append(r)
        multi = {k: v for k, v in groups.items() if len(v) > 1}
        if not multi:
            out[arm] = {"k": None, "n_units": 0, "pass_pow_k": None}
            continue
        out[arm] = {"k": max(len(v) for v in multi.values()),
                    "n_units": len(multi),
                    "pass_pow_k": sum(1 for v in multi.values()
                                      if all(_success(r) for r in v)) / len(multi)}
    return out


def render_report(records, arm_a: str = "reviewer|card",
                  arm_b: str = "compiler-only|card") -> str:
    """The full post-run analysis as one text report (pre-fire audit FIX-5):
    lead table, the three registered paired tests + Holm, pass^k, summarize().
    Pure — no file I/O — so it is testable and reusable (e.g. per-model
    sections of a combined report)."""
    from src.intent.experiment import summarize  # lazy: one-way dep, kept local
    mc = arm_mcnemar(records, arm_a, arm_b)
    wil_fa = arm_wilcoxon(records, arm_a, arm_b, "fa_per_hour")
    wil_mcc = arm_wilcoxon(records, arm_a, arm_b, "mcc")
    ps = [p for p in (mc["p_midp"], wil_fa["p"], wil_mcc["p"]) if p is not None]
    mc_r = arm_mcnemar(records, arm_a, arm_b, granularity="repeat")
    wil_fa_r = arm_wilcoxon(records, arm_a, arm_b, "fa_per_hour", granularity="repeat")
    wil_mcc_r = arm_wilcoxon(records, arm_a, arm_b, "mcc", granularity="repeat")
    ps_r = [p for p in (mc_r["p_midp"], wil_fa_r["p"], wil_mcc_r["p"]) if p is not None]
    return "\n".join([
        "== LEAD TABLE (per-intent metric effects) ==",
        lead_table(records).to_string(index=False),
        "",
        "== PAIRED ARM TESTS (canonical cells; UNIT granularity — the registered n) ==",
        f"mcnemar(midp) {arm_a} vs {arm_b}: {mc}",
        f"wilcoxon fa_per_hour: {wil_fa}",
        f"wilcoxon mcc: {wil_mcc}",
        f"holm-adjusted: {holm(ps)}",
        "-- supplementary: REPEAT granularity (correlated repeats paired separately; inflates counts) --",
        f"mcnemar(midp) {arm_a} vs {arm_b}: {mc_r}",
        f"wilcoxon fa_per_hour: {wil_fa_r}",
        f"wilcoxon mcc: {wil_mcc_r}",
        f"holm-adjusted: {holm(ps_r)}",
        "",
        "== pass^k ==",
        json.dumps(pass_pow_k(records), indent=2),
        "",
        "== summarize() ==",
        json.dumps(summarize(records), indent=2, default=str),
    ])


def analyze_runlog(runlog, arm_a: str = "reviewer|card",
                   arm_b: str = "compiler-only|card") -> dict:
    """Load a matrix run log, render the report, and persist BOTH artifacts
    beside the log — `<stem>.lead_table.csv` and `<stem>.analysis.txt` — so
    every number the analysis prints exists as a committed file, not terminal
    scrollback (pre-fire audit FIX-5). Returns {'report', 'csv', 'txt'}."""
    runlog = Path(runlog)
    recs = load_records(runlog)
    report = render_report(recs, arm_a, arm_b)
    csv_path = runlog.with_suffix(".lead_table.csv")
    lead_table(recs).to_csv(csv_path, index=False)
    txt_path = runlog.with_suffix(".analysis.txt")
    txt_path.write_text(report)
    return {"report": report, "csv": csv_path, "txt": txt_path}


if __name__ == "__main__":  # pragma: no cover — post-run analysis entry point
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("runlog")
    ap.add_argument("--arm-a", default="reviewer|card")
    ap.add_argument("--arm-b", default="compiler-only|card")
    a = ap.parse_args()
    out = analyze_runlog(a.runlog, a.arm_a, a.arm_b)
    print(out["report"])
    print(f"-> {out['csv']}")
    print(f"-> {out['txt']}")
