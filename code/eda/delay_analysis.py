"""First detection-delay results on NCSRD-DS-5GDDoS (never reported before).

Part 1 — TEMPORAL stream (deployment-realistic): for each model's saved test
scores (contiguous final ~21 h containing the DNS and GTP-U incidents), sweep
threshold x persistence -> false alarms/hour vs per-incident detection delay.
Thresholds: model's default operating point + benign-score quantiles (curve
characterization, ROC-style — computed from test scores, stated openly).

Part 2 — LOAO folds: delay for each NEVER-SEEN attack at the default threshold
and at a lowered threshold 0.1 (persistence 1) — previews the intent knob's
effect on novel attacks. (FA/hour is not computed here: LOAO benign windows
are a scattered sample, not a stream; FPR lives in the run JSONs.)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import ATTACK_NAMES, ATTACK_WINDOWS, PARTICIPANTS, incidents_dict
from src.replay import alarm_events, false_alarms_per_hour, incident_delays

ART = Path(__file__).resolve().parents[1] / "artifacts"
FIGS = Path(__file__).resolve().parent / "figures"
MODELS = ["lstm", "tcn", "transformer", "ae"]
MEDIAN_DT_S = 5.14

print("=" * 100)
print("PART 1 — temporal stream (incidents in test period: DNS, GTP-U)")
print("=" * 100)
rows = []
for m in MODELS:
    for w in (3, 5, 7):  # window knob armed 2026-07-03
        fp = ART / f"{m}_w{w}_temporal_scores.npz"
        if not fp.exists():
            continue
        z = np.load(fp, allow_pickle=True)
        scores, y, ts, ue = z["scores"], z["y"], z["ts"], z["ue"]
        thr0 = json.load(open(ART / f"{m}_w{w}_temporal_metrics.json"))["threshold"]
        ben = scores[y == 0]
        benign_hours = (y == 0).sum() * MEDIAN_DT_S / 3600
        ladder = sorted({thr0, *np.quantile(ben, [0.995, 0.999, 0.9999]).tolist()})
        inc = incidents_dict([4, 5])
        tsx = pd.DatetimeIndex(ts)
        def sustained(k, thr):  # fraction of in-window participant windows above threshold
            s0, e0 = (t.tz_localize(None) for t in inc[k][:2])
            mask = (tsx >= s0) & (tsx < e0) & np.isin(ue, list(inc[k][2]))
            return float((scores[mask] >= thr).mean()) if mask.sum() else 0.0
        for thr in ladder:
            for persist in [1, 3]:
                fa_ev = alarm_events(scores, ts, ue, thr, persistence=persist, latch=True)   # count events
                dl_ev = alarm_events(scores, ts, ue, thr, persistence=persist, latch=False,
                                     with_start=True)                                         # measure delay
                fa = false_alarms_per_hour(fa_ev, inc, benign_hours)
                d = incident_delays(dl_ev, inc, fresh_only=True)  # A1 fix: reject standing pre-onset alarms
                rows.append({"model": m, "window": w, "threshold": thr, "persistence": persist,
                             "fa_per_hour": fa,
                             "dns_delay_s": d[4]["delay_s"], "dns_sustained": sustained(4, thr),
                             "gtpu_delay_s": d[5]["delay_s"], "gtpu_sustained": sustained(5, thr)})
tdf = pd.DataFrame(rows)
tdf.to_csv(ART / "delay_temporal.csv", index=False)
with pd.option_context("display.width", 160, "display.max_rows", 100):
    print(tdf.round(3).to_string(index=False))

print()
print("=" * 100)
print("PART 2 — LOAO folds: delay to detect the NEVER-SEEN attack (persistence 1)")
print("=" * 100)
rows2 = []
SEEDS = [42, 43, 44]  # A5 fix: LOAO delay across all 3 seeds (score archives exist)
for m in MODELS:
    for k in range(1, 6):
        for seed in SEEDS:
            suffix = "" if seed == 42 else f"_s{seed}"
            fp = ART / f"{m}_w3_loao{k}{suffix}_scores.npz"
            if not fp.exists():
                continue
            z = np.load(fp, allow_pickle=True)
            scores, ts, ue = z["scores"], z["ts"], z["ue"]
            thr0 = json.load(open(ART / f"{m}_w3_loao{k}{suffix}_metrics.json"))["threshold"]
            start, end = ATTACK_WINDOWS[k]
            sn, en = start.tz_localize(None), end.tz_localize(None)
            tsx = pd.DatetimeIndex(ts)
            inwin = (tsx >= sn) & (tsx < en) & np.isin(ue, list(PARTICIPANTS[k]))
            for thr in ([thr0] if m == "ae" else [thr0, 0.1]):
                ev = alarm_events(scores, ts, ue, thr, persistence=1, latch=False, with_start=True)
                d = incident_delays(ev, incidents_dict([k]), fresh_only=True)[k]  # A1 fix
                sus = float((scores[inwin] >= thr).mean()) if inwin.sum() else 0.0
                rows2.append({"model": m, "attack": ATTACK_NAMES[k], "seed": seed,
                              "threshold_kind": "default" if thr == thr0 else "0.1",
                              "detected": d["detected"], "delay_s": d["delay_s"],
                              "sustained_frac": round(sus, 3)})
ldf = pd.DataFrame(rows2)
ldf.to_csv(ART / "delay_loao.csv", index=False)
# genuine detections only: sustained >= 0.3 AND detected (A1/A3 honesty guard)
ldf["genuine"] = ldf.detected & (ldf.sustained_frac >= 0.3)
d0 = ldf[ldf.threshold_kind == "default"]
summ = d0.groupby(["model", "attack"]).agg(
    genuine_seeds=("genuine", "sum"), n_seeds=("seed", "nunique"),
    sust_mean=("sustained_frac", "mean"))
# delay statistics over GENUINE seeds only (a mean
# mixing non-genuine rows is meaningless); min/max expose bimodal cases like
# guard-censored AE-DNS (4.8 s / 426.1 s) that a lone mean would hide.
gen = d0[d0.genuine].groupby(["model", "attack"]).delay_s.agg(
    delay_mean_gen="mean", delay_min_gen="min", delay_max_gen="max")
summ = summ.join(gen).round(2)
print("Per (model,attack) at DEFAULT threshold, across seeds — genuine = detected & sustained>=0.3;")
print("delay columns over GENUINE seeds only:")
print(summ.to_string())

tdf = pd.DataFrame(rows)  # (defined above; kept for the figure)
# figure: temporal trade-off (persistence 1), one panel per incident.
# y capped at 60 s (all genuine detections fall below); misses drawn as x at
# the "MISS" band above the cap so they don't distort the readable range.
CAP, MISS_Y = 60, 66
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, col, title in [(axes[0], "dns_delay_s", "DNS incident"), (axes[1], "gtpu_delay_s", "GTP-U incident")]:
    for m in MODELS:
        g = tdf[(tdf.model == m) & (tdf.persistence == 1) & (tdf.window == 3)].sort_values("fa_per_hour")
        det = g[g[col].notna() & (g[col] <= CAP)]
        ax.plot(det.fa_per_hour, det[col], "o-", label=m.upper())
        bad = g[g[col].isna() | (g[col] > CAP)]  # missed or absurdly late (>cap) = effectively missed
        if len(bad):
            ax.scatter(bad.fa_per_hour, [MISS_Y] * len(bad), marker="x", color="gray")
    ax.axhline(CAP, ls=":", c="gray", lw=0.7)
    ax.text(ax.get_xlim()[0], MISS_Y, " MISS/>60s", va="center", fontsize=7, color="gray")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("false alarms per benign UE-hour")
    ax.set_ylabel("detection delay (s)")
    ax.set_ylim(-3, 72)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="center right")
fig.suptitle("Temporal replay: false-alarm vs detection-delay trade-off (unseen attacks, persistence 1, window 3)")
fig.savefig(FIGS / "delay_tradeoff.png", dpi=140, bbox_inches="tight")
print(f"\nsaved {ART / 'delay_temporal.csv'}, {ART / 'delay_loao.csv'}, {FIGS / 'delay_tradeoff.png'}")
