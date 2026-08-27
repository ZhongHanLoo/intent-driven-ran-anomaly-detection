"""Aggregate Option C results (code/artifacts/*_metrics.json) into the study's
key tables and headline figure. Saves: option_c_summary.csv + loao_heatmap.png."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
FIGS = Path(__file__).resolve().parent / "figures"
ATTACKS = {1: "SYN", 2: "ICMP", 3: "UDP-frag", 4: "DNS", 5: "GTP-U"}
MODELS = ["lstm", "tcn", "transformer", "ae"]

rows = [json.load(open(p)) for p in sorted(ART.glob("*_metrics.json"))]
for r in rows:  # legacy keys from the pre-unification LSTM run
    r.setdefault("protocol", r.get("split"))
    r.setdefault("model", "lstm")
df = pd.DataFrame(rows)
df.to_csv(ART / "option_c_summary.csv", index=False)

pd.set_option("display.width", 200)
fmt = lambda d, cols: d[cols].round(4).to_string(index=False)

print("=== RANDOM SPLIT (Paper 1-comparable) ===")
r = df[df.protocol == "random"].set_index("model").reindex(MODELS).reset_index()
print(fmt(r, ["model", "accuracy", "macro_f1", "fpr", "fnr", "mcc", "auc", "params", "epochs_ran"]))
print("Paper 1 Table VII:   lstm .9985/.9775/FPR .0004/FNR .0614 | tcn .9985/.9767 | transformer .9984/.9754 (win 7)")

print("\n=== TEMPORAL SPLIT (fixed validation) ===")
t = df[df.protocol == "temporal"].set_index("model").reindex(MODELS).reset_index()
print(fmt(t, ["model", "accuracy", "macro_f1", "fpr", "fnr", "mcc", "auc", "epochs_ran"]))

print("\n=== LOAO: recall on the HELD-OUT (never-seen) attack type ===")
lo = df[df.protocol == "loao"].copy()
lo["attack"] = lo["heldout"].map(ATTACKS)
piv_rec = lo.pivot_table(index="model", columns="attack", values="heldout_recall").reindex(MODELS)[list(ATTACKS.values())]
print((piv_rec * 100).round(1).to_string())
print("\n=== LOAO: ranking quality vs benign (AUC, held-out type) ===")
piv_auc = lo.pivot_table(index="model", columns="attack", values="heldout_auc_vs_benign").reindex(MODELS)[list(ATTACKS.values())]
print(piv_auc.round(3).to_string())
print("\n=== LOAO: control — recall on SEEN attack types / FPR ===")
piv_seen = lo.pivot_table(index="model", columns="attack", values="seen_recall").reindex(MODELS)[list(ATTACKS.values())]
print((piv_seen * 100).round(1).to_string())
piv_fpr = lo.pivot_table(index="model", columns="attack", values="fpr").reindex(MODELS)[list(ATTACKS.values())]
print("\nFPR (%):")
print((piv_fpr * 100).round(3).to_string())

# headline figure: two heatmaps (held-out recall, held-out AUC)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, piv, title, vmax in [
    (axes[0], piv_rec * 100, "Recall on NEVER-SEEN attack type (%)", 100),
    (axes[1], piv_auc, "Ranking quality vs benign (AUC)", 1.0),
]:
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_yticks(range(len(piv.index)), [m.upper() for m in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.0f}" if vmax == 100 else f"{v:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Leave-one-attack-out: supervised models vs benign-only autoencoder", y=1.04)
fig.savefig(FIGS / "loao_heatmap.png", dpi=140, bbox_inches="tight")
print(f"\nsaved {FIGS / 'loao_heatmap.png'} and {ART / 'option_c_summary.csv'}")
