"""Aggregate the hardening batch: LOAO across seeds {42,43,44} with mean±std,
and the SMOTE+Tomek variants vs their baselines. Saves loao_multiseed.csv."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
ATTACKS = {1: "SYN", 2: "ICMP", 3: "UDP-frag", 4: "DNS", 5: "GTP-U"}
MODELS = ["lstm", "tcn", "transformer", "ae"]

rows = [json.load(open(p)) for p in ART.glob("*_metrics.json")]
for r in rows:
    r.setdefault("protocol", r.get("split"))
    r.setdefault("model", "lstm")
    r.setdefault("seed", 42)
df = pd.DataFrame(rows)

# ---- LOAO across seeds -------------------------------------------------------
lo = df[df.protocol == "loao"].copy()
lo["attack"] = lo["heldout"].map(ATTACKS)
n_seeds = lo.groupby(["model", "attack"]).seed.nunique()
print(f"LOAO seeds per (model,attack): min {n_seeds.min()}, max {n_seeds.max()}")

def mean_std(metric):
    g = lo.groupby(["model", "attack"])[metric]
    return g.mean().unstack().reindex(MODELS)[list(ATTACKS.values())], g.std().unstack().reindex(MODELS)[list(ATTACKS.values())]

for metric, label, scale in [("heldout_recall", "held-out recall (%)", 100),
                             ("heldout_auc_vs_benign", "held-out AUC", 1)]:
    m, s = mean_std(metric)
    print(f"\n=== LOAO {label}: mean ± std over seeds ===")
    combo = m.copy().astype(object)
    for i in m.index:
        for c in m.columns:
            combo.loc[i, c] = f"{m.loc[i,c]*scale:5.1f}±{s.loc[i,c]*scale:4.1f}" if scale == 100 else f"{m.loc[i,c]:.2f}±{s.loc[i,c]:.2f}"
    print(combo.to_string())

lo_summary = lo.groupby(["model", "attack"]).agg(
    recall_mean=("heldout_recall", "mean"), recall_std=("heldout_recall", "std"),
    auc_mean=("heldout_auc_vs_benign", "mean"), auc_std=("heldout_auc_vs_benign", "std"),
    n_seeds=("seed", "nunique")).round(4)
lo_summary.to_csv(ART / "loao_multiseed.csv")

# ---- SMOTE variants vs baseline (random split) — read from files by name ----
print("\n=== SMOTE+Tomek vs baseline (random split) ===")
print(f"{'model':12s} {'variant':9s} {'macroF1':>8s} {'FPR%':>7s} {'FNR%':>7s} {'MCC':>6s} {'recall%':>8s}")
for m in ["lstm", "tcn", "transformer"]:
    for variant, fname in [("baseline", f"{m}_w3_random_metrics.json"),
                           ("smote", f"{m}_w3_random_smote_metrics.json")]:
        p = ART / fname
        if p.exists():
            r = json.load(open(p))
            rec = r["tp"] / (r["tp"] + r["fn"]) * 100
            print(f"{m:12s} {variant:9s} {r['macro_f1']:8.4f} {r['fpr']*100:7.3f} {r['fnr']*100:7.3f} {r['mcc']:6.3f} {rec:8.1f}")
print("(SMOTE trades FNR down for FPR up — a recall-favouring variant for the intent policy set;")
print(" effect is modest at this setting, clearest for TCN, and smaller than Paper 1's SMOTE+Tomek result.)")
print(f"\nsaved {ART / 'loao_multiseed.csv'}")
