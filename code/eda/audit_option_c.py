"""Audit of the Option C batch (run matrix, reported-number
spot-checks, and verification of the AE-drift explanation)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# A. run matrix completeness
print("=== A. run matrix ===")
runs = {}
for p in ART.glob("*_metrics.json"):
    r = json.load(open(p))
    r.setdefault("protocol", r.get("split"))
    r.setdefault("model", "lstm")
    key = (r["model"], r["protocol"], r.get("heldout", 0))
    runs[key] = r
expected = (
    [(m, "random", 0) for m in ["lstm", "tcn", "transformer", "ae"]]
    + [(m, "temporal", 0) for m in ["lstm", "tcn", "transformer", "ae"]]
    + [(m, "loao", k) for m in ["lstm", "tcn", "transformer", "ae"] for k in range(1, 6)]
)
missing = [e for e in expected if e not in runs]
check("all 28 expected runs present", not missing, f"missing: {missing or 'none'}")

# B. spot-check the reported numbers
print("\n=== B. reported-number spot checks ===")
S = [
    (("tcn", "random", 0), "macro_f1", 0.9763, 0.0002),
    (("transformer", "random", 0), "macro_f1", 0.9746, 0.0002),
    (("ae", "random", 0), "fpr", 0.0054, 0.0002),
    (("ae", "temporal", 0), "fpr", 0.1775, 0.0005),
    (("lstm", "temporal", 0), "fnr", 0.9297, 0.0005),
    (("transformer", "loao", 4), "heldout_recall", 0.996, 0.001),
    (("tcn", "loao", 4), "heldout_recall", 0.167, 0.001),
    (("lstm", "loao", 5), "heldout_auc_vs_benign", 0.458, 0.02),
    (("ae", "loao", 5), "heldout_auc_vs_benign", 0.789, 0.02),
    (("lstm", "loao", 1), "heldout_n", 1402, 0),
]
for key, field, want, tol in S:
    got = runs[key].get(field)
    check(f"{key} {field}", got is not None and abs(got - want) <= tol, f"got {got} want ~{want}")

# C. heldout isolation invariant across all LOAO runs: heldout_n must equal
# the known per-attack window counts (1402/3756/1402/1399/3497)
print("\n=== C. LOAO heldout sizes ===")
counts = {1: 1402, 2: 3756, 3: 1402, 4: 1399, 5: 3497}
ok = all(runs[(m, "loao", k)]["heldout_n"] == counts[k]
         for m in ["lstm", "tcn", "transformer", "ae"] for k in range(1, 6))
check("heldout_n exact for all 20 LOAO runs", ok, str(counts))

# D. AE temporal drift claim: where do its false positives come from?
print("\n=== D. AE temporal false-positive anatomy ===")
z = np.load(ART / "ae_w3_temporal_scores.npz", allow_pickle=True)
thr = runs[("ae", "temporal", 0)]["threshold"]
scores, y, ts, ue = z["scores"], z["y"], pd.DatetimeIndex(z["ts"]), z["ue"]
fp = (y == 0) & (scores >= thr)
print(f"threshold {thr:.4f} | benign windows {int((y==0).sum()):,} | FPs {int(fp.sum()):,} ({fp.sum()/(y==0).sum()*100:.1f}%)")
per_ue = pd.Series(fp).groupby(pd.Series(ue)).mean().sort_values(ascending=False)
print("FP rate per UE (%):")
print((per_ue * 100).round(1).to_string())
per_hour = pd.Series(fp.astype(float)).groupby(ts.floor("h")).mean()
print(f"hours with FP-rate > 30%: {(per_hour > 0.3).sum()} of {len(per_hour)}")
top3 = float(per_ue.head(3).mean() * 100)
bot6 = float(per_ue.tail(6).mean() * 100)
check("FPs concentrated (drift-like), not uniform", top3 > 3 * max(bot6, 0.1),
      f"top-3 UEs mean {top3:.1f}% vs remaining mean {bot6:.1f}%")

print(f"\n{sum(results)}/{len(results)} checks passed" + ("" if all(results) else "  <-- FAILURES ABOVE"))
