"""Held-out phrasing extension analysis.

Reads the three extension run logs + held_out_phrasings.json metadata and emits
code/artifacts/intent_runs/extension_analysis.md.

The registered scope: n = 15 sentences/intent with exact Clopper-Pearson,
arm effects over the 48 extension units, the four length questions at 5/5/5
with the terse-vs-descriptive confound, no pooling into canonical statistics,
authorship-tier disclosure attached to robustness claims. The sentence-level
success rules were fixed at ANALYSIS TIME, before computation, mirroring the
registered pass^k semantics: on the primary arm A0, a sentence is
acceptance-robust iff all 3 repeats emitted a policy, and compliance-robust
(an analysis-time quantity) iff all 3 emitted budget-feasible policies. The
unit-collapse rule of the arm tests IS registered (analysis.py granularity
default, 2026-07-17)."""
import json, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scipy.stats import beta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.intent.analysis import arm_mcnemar, arm_wilcoxon, holm, load_records

ART = Path(__file__).resolve().parents[1] / "artifacts"
RUNS = ART / "intent_runs"
LEGS = {"deepseek-v4-pro": "matrix_20260807T042723Z",
        "gpt-5.4-nano": "matrix_20260807T053146Z",
        "gemini-2.5-flash": "matrix_20260807T061129Z"}
ORDER = list(LEGS)
INTENTS = ["early_attack_detection", "minimize_false_alarms",
           "balanced_operation", "defend_unknown_attacks"]
A0, A1, A2, A3 = "reviewer|card", "compiler-only|card", "reviewer|no-card", "reviewer|card|no-critique"

def cls(t):
    n = len(t.split())
    return "short" if n <= 4 else ("medium" if n <= 7 else "long")

def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi

assert abs(cp(15, 15)[0] - 0.7820) < 0.001

ho_file = json.loads((Path(__file__).resolve().parents[1] / "src/intent/held_out_phrasings.json").read_text())
intents_spec = json.loads((Path(__file__).resolve().parents[1] / "src/intent/intents.json").read_text())
PROV = {m["text"]: ("new" if m["provenance"].startswith("new") else "reserve")
        for metas in ho_file["metadata"].values() for m in metas}

R = {m: load_records(RUNS / f"{s}.jsonl") for m, s in LEGS.items()}
CANON_ACC = {"deepseek-v4-pro": 141, "gpt-5.4-nano": 127, "gemini-2.5-flash": 144}
for m, recs in R.items():
    can = [r for r in recs if not r.get("held_out") and not r.get("adversarial_kind")]
    assert sum(1 for r in can if r["status"] == "policy") == CANON_ACC[m], m
    # ranked == feasible coincidence check (caption claim)
    for r in recs:
        o = r.get("oracle")
        if o is not None:
            assert (o.get("rank") is not None) == bool(o.get("feasible")), (m, "rank/feasible mismatch")

def sentences(recs, model):
    out = {}
    for iid in INTENTS:
        for tier, texts in (("canonical", intents_spec[iid]["phrasings"][:3]),
                            ("extension", ho_file["phrasings"][iid])):
            for t in texts:
                cells = [r for r in recs if r.get("intent_id") == iid and r.get("text") == t
                         and r.get("arm") == A0 and not r.get("adversarial_kind")]
                assert len(cells) == 3, (model, iid, t, len(cells))
                out[(iid, t)] = dict(cells=cells, cls=cls(t),
                                     prov=("canonical" if tier == "canonical" else PROV[t]))
    return out

S = {m: sentences(recs, m) for m, recs in R.items()}
for m in ORDER:  # 5/5/5 per intent over the 15
    for iid in INTENTS:
        cc = [v["cls"] for (i, _), v in S[m].items() if i == iid]
        assert {c: cc.count(c) for c in ("short", "medium", "long")} == {"short": 5, "medium": 5, "long": 5}

def acc_ok(c):  return c["status"] == "policy"
def feas_ok(c): return c["status"] == "policy" and bool((c.get("oracle") or {}).get("feasible"))

NAMES = {"early_attack_detection": "early", "minimize_false_alarms": "minimize",
         "balanced_operation": "balanced", "defend_unknown_attacks": "defend"}

L = []
L.append("# Held-out phrasing extension — analysis of the pre-registered scope\n")
L.append(f"*Generated {pd.Timestamp.utcnow().strftime('%Y-%m-%d')} from the three extension run logs "
         f"({', '.join(LEGS.values())}). **Scope** per the plan addendum of 2026-08-06 (sealed with freeze "
         "32817a9; grading state byte-identical to 64eead0): n = 15 sentences per intent (3 canonical + 12 "
         "extension, all from these logs), exact Clopper-Pearson 95% intervals, arm effects over the 48 "
         "extension units, the four length questions, no pooling into canonical statistics. **Sentence-level "
         "success rules were fixed at analysis time (2026-08-07), before computation, mirroring the "
         "registered pass^k semantics** — they are not themselves in the addendum: on the primary arm A0, "
         "acceptance-robust = all 3 repeats emitted a policy; compliance-robust (an analysis-time quantity) "
         "= all 3 repeats emitted a budget-feasible policy. The arm tests' unit-collapse rule IS registered "
         "(analysis.py default since 2026-07-17). **Conventions:** A0 = reviewer|card (full pipeline), "
         "A1 = compiler-only|card, A2 = reviewer|no-card, A3 = reviewer|card|no-critique; in arm tables, "
         "b = units where the first-named arm succeeded and the second failed, c = the reverse; "
         "Hodges-Lehmann (HL) shifts are first-named arm minus second. Extension results are reported apart "
         "from the registered canonical core.*\n")

L.append("\n## 1. Wording robustness per intent (the formerly refused claim, now licensed)\n")
L.append("*Authorship disclosure, attached per the addendum: 18 of the 48 extension sentences are pre-freeze, "
         "pre-results reserves; 30 were authored post-results under the T3 fallback with selection mitigation "
         "— see section 4; the tier comparison shows no consistent advantage, with the largest gap running against the contamination direction. "
         "Intervals to 3 dp so 14/15's upper bound (0.998) is distinct from a perfect record's 1.000.*\n")
L.append("| Model | Intent | Acceptance-robust (of 15) | 95% CI | Compliance-robust (of 15) | 95% CI |")
L.append("|---|---|---|---|---|---|")
for m in ORDER:
    for iid in INTENTS:
        ss = [v for (i, _), v in S[m].items() if i == iid]
        ka = sum(1 for v in ss if all(acc_ok(c) for c in v["cells"]))
        kf = sum(1 for v in ss if all(feas_ok(c) for c in v["cells"]))
        (al, ah), (fl, fh) = cp(ka, 15), cp(kf, 15)
        L.append(f"| {m} | {NAMES[iid]} | {ka}/15 | [{al:.3f}, {ah:.3f}] | {kf}/15 | [{fl:.3f}, {fh:.3f}] |")

L.append("\n## 2. The four pre-registered length questions\n")
L.append("*Length classes short/medium/long = 2-4 / 5-7 / 8+ words; 5/5/5 per intent, 20 sentences per class "
         "per model pooled. **Disclosed confound (per the addendum):** longer phrasings naturally carry more "
         "context and urgency, so these results read as terse-vs-descriptive requests, not word-count "
         "causality. Cell-level rates over A0 cells; rank over ranked cells (= budget-feasible cells; the "
         "coincidence is asserted in code); consistency = sentences (>=2 emitted policies) whose policies are "
         "all identical — a **survivorship denominator**: sentences with <=1 policy are excluded, which on "
         "gpt-5.4-nano removes its worst sentences (denominators 16/17/16 vs 20/20/20 elsewhere), biasing its "
         "consistency upward in cross-model reads.*\n")
L.append("| Model | Class | (a) acceptance (cells) | (b) budget-compliance (of emitted) | (c) mean oracle rank | (d) repeat-consistency |")
L.append("|---|---|---|---|---|---|")
for m in ORDER:
    for c in ("short", "medium", "long"):
        vs = [v for v in S[m].values() if v["cls"] == c]
        assert len(vs) == 20
        cells = [x for v in vs for x in v["cells"]]
        pol = [x for x in cells if acc_ok(x)]
        feas = [x for x in pol if feas_ok(x)]
        ranked = [x["oracle"]["rank"] for x in pol if (x.get("oracle") or {}).get("rank") is not None]
        multi = [[json.dumps(p["policy"], sort_keys=True) for p in v["cells"] if acc_ok(p)] for v in vs]
        multi = [ps for ps in multi if len(ps) >= 2]
        cons = sum(1 for ps in multi if all(p == ps[0] for p in ps))
        L.append(f"| {m} | {c} | {len(pol)/len(cells):.2f} ({len(pol)}/{len(cells)}) | "
                 f"{len(feas)/len(pol):.2f} ({len(feas)}/{len(pol)}) | "
                 f"{(sum(ranked)/len(ranked)) if ranked else float('nan'):.1f} (n={len(ranked)}) | "
                 f"{cons/len(multi):.2f} ({cons}/{len(multi)}) |")

L.append("\n**Per-intent breakdowns (the addendum's per-intent-first rule), by class:**\n")
def per_intent_table(title, cellfn):
    L.append(f"\n*{title}*\n")
    L.append("| Model | Intent | short | medium | long |")
    L.append("|---|---|---|---|---|")
    for m in ORDER:
        for iid in INTENTS:
            row = []
            for c in ("short", "medium", "long"):
                vs = [v for (i, _), v in S[m].items() if i == iid and v["cls"] == c]
                row.append(cellfn(vs))
            L.append(f"| {m} | {NAMES[iid]} | {row[0]} | {row[1]} | {row[2]} |")

def f_acc(vs):
    k = sum(1 for v in vs if all(acc_ok(x) for x in v["cells"]))
    return f"{k}/5"
def f_comp(vs):
    pol = [x for v in vs for x in v["cells"] if acc_ok(x)]
    feas = [x for x in pol if feas_ok(x)]
    return f"{len(feas)}/{len(pol)}" if pol else "0/0"
def f_rank(vs):
    ranked = [x["oracle"]["rank"] for v in vs for x in v["cells"]
              if acc_ok(x) and (x.get("oracle") or {}).get("rank") is not None]
    return f"{sum(ranked)/len(ranked):.0f} (n={len(ranked)})" if ranked else "— (n=0)"
def f_cons(vs):
    multi = [[json.dumps(p["policy"], sort_keys=True) for p in v["cells"] if acc_ok(p)] for v in vs]
    multi = [ps for ps in multi if len(ps) >= 2]
    if not multi: return "— (0)"
    cons = sum(1 for ps in multi if all(p == ps[0] for p in ps))
    return f"{cons}/{len(multi)}"

per_intent_table("(a) acceptance-robust sentences (of 5 per class)", f_acc)
per_intent_table("(b) budget-compliant cells / emitted cells", f_comp)
per_intent_table("(c) mean oracle rank over ranked cells", f_rank)
per_intent_table("(d) consistent sentences / sentences with >=2 policies", f_cons)

L.append("\n## 3. Arm effects on the extension population\n")
L.append("*The registered comparisons on population='held_out' at the registered unit granularity "
         "(unit = one extension sentence; success = emitted AND budget-feasible in all repeats). "
         "**The 48-unit population applies to the b/c (McNemar) column.** The HL columns are computed over "
         "the co-surviving units only — pairs where both arms produced the metric — with that n shown per "
         "cell; the dropped units are informatively censored (a unit vanishes exactly when one arm emitted "
         "no metric-bearing policy), so a near-zero HL with a large b/c split means the effect lives in the "
         "missingness, not the metric. Holm over each comparison's three tests; the first column is the "
         "**Holm-adjusted** mid-p.*\n")
L.append("| Model | Comparison | Discordant b/c | Holm-adj. mid-p | FA/h HL (Holm p; n) | MCC HL (Holm p; n) |")
L.append("|---|---|---|---|---|---|")
for m in ORDER:
    recs = R[m]
    for cname, (a, b) in {"±Reviewer (A0 vs A1)": (A0, A1), "±Card (A0 vs A2)": (A0, A2),
                          "retry value (A1 vs A3)": (A1, A3)}.items():
        mu = arm_mcnemar(recs, a, b, population="held_out")
        wf = arm_wilcoxon(recs, a, b, "fa_per_hour", population="held_out")
        wm = arm_wilcoxon(recs, a, b, "mcc", population="held_out")
        assert mu["n_pairs"] == 48
        h = holm([mu["p_midp"], wf["p"], wm["p"]])
        L.append(f"| {m} | {cname} | {mu['b']}/{mu['c']} | {h[0]:.3g} | "
                 f"{wf['hl_shift']:+.2f} ({h[1]:.3g}; n={wf['n_pairs']}) | "
                 f"{wm['hl_shift']:+.3f} ({h[2]:.3g}; n={wm['n_pairs']}) |")

L.append("\n## 4. Authorship-tier honesty check (reserves vs newly authored)\n")
L.append("*Do the 30 post-results-authored sentences behave differently from the 18 pre-results reserves? "
         "(Cell-level A0 rates; a gap favouring the new sentences would flag authorship contamination.)*\n")
L.append("| Model | Tier | acceptance (cells) | compliance (of emitted) |")
L.append("|---|---|---|---|")
for m in ORDER:
    for tier in ("reserve", "new"):
        vs = [v for v in S[m].values() if v["prov"] == tier]
        cells = [x for v in vs for x in v["cells"]]
        pol = [x for x in cells if acc_ok(x)]
        feas = [x for x in pol if feas_ok(x)]
        L.append(f"| {m} | {tier} ({len(vs)} sentences) | {len(pol)/len(cells):.2f} ({len(pol)}/{len(cells)}) | "
                 f"{len(feas)/len(pol):.2f} ({len(feas)}/{len(pol)}) |")

L.append("\n## 5. Notable sentences (descriptive, analysis-time; tier-tagged)\n")
L.append("*Sentences not acceptance-robust on A0. Tags: [canonical] = one of the registered 3 per intent "
         "(its behaviour is part of the canonical core, replayed here from cache); [reserve]/[new] = "
         "extension tiers. Extension-only counts stated per model.*\n")
for m in ORDER:
    bad = [(i, t, v["prov"], pd.Series([c['status'] for c in v['cells']]).value_counts().to_dict())
           for (i, t), v in S[m].items() if not all(acc_ok(c) for c in v["cells"])]
    n_ext = sum(1 for *_, p, _s in [(a, b, c, d) for a, b, c, d in bad] if p != "canonical")
    L.append(f"\n**{m} — {len(bad)} sentences not acceptance-robust ({n_ext} extension, {len(bad)-n_ext} canonical):**\n")
    for i, t, p, st in sorted(bad):
        L.append(f"- [{p}] {NAMES[i]}: \"{t}\" — {st}")

out = RUNS / "extension_analysis.md"
out.write_text("\n".join(L) + "\n")
print("written", out, len(L), "lines")
