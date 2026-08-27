"""Archive validation scores + benign-quantile operating points. For every
temporal-protocol model artifact: rebuild its exact
training-time validation set (shared carve functions — see
data_pipeline.carve_validation{,_benign}), score it with the saved scaler +
weights, archive `{base}_valscores.npz`, and emit operating_points.csv — the
validation-derived threshold ladder the Phase-2 intent layer will quote.

Built-in end-to-end proof: the AE's stored default threshold IS its benign-val
p99.5, so our reconstruction must reproduce it exactly; any mismatch aborts.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.calibrate import DEFAULT_QS, benign_quantile_table
from src.data_pipeline import (
    build_windows, carve_validation, carve_validation_benign, derive_labels, load_raw,
)
from src.models import make_model
from src.train import score_all

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "code" / "artifacts"
PAT = re.compile(r"^(lstm|tcn|transformer|ae)_w(\d+)_temporal_metrics\.json$")

targets = sorted((PAT.match(p.name).groups(), p) for p in ART.glob("*_temporal_metrics.json")
                 if PAT.match(p.name))
if not targets:
    sys.exit("no temporal metrics artifacts found")
windows = sorted({int(w) for (_, w), _ in targets})
print(f"targets: {[(m, w) for (m, w), _ in targets]}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
df = derive_labels(load_raw(ART.parents[1] / "data" / "amari_ue_data_merged_with_attack_number.csv"))

rows = []
for w in windows:
    X, y, ts, ue = build_windows(df, window=w)
    from src.data_pipeline import split_temporal
    tr, te = split_temporal(ts)
    ytr, Xtr_raw, tstr, uetr = y[tr], X[tr], ts[tr], ue[tr]
    for (name, w_s), mp in targets:
        if int(w_s) != w:
            continue
        meta = json.load(open(mp))
        seed, base = meta.get("seed", 42), mp.name.replace("_metrics.json", "")
        va_i = (carve_validation_benign(ytr, seed) if name == "ae"
                else carve_validation(ytr, seed))[1]
        scaler = joblib.load(ART / f"{base}_scaler.joblib")
        f = Xtr_raw.shape[-1]
        Xva = scaler.transform(Xtr_raw[va_i].reshape(-1, f)).reshape(
            Xtr_raw[va_i].shape).astype(np.float32)
        model = make_model(name, window=w).to(device)
        model.load_state_dict(torch.load(ART / f"{base}.pt", map_location=device))
        model.eval()
        scores = score_all(model, Xva, device, 64, name == "ae")
        yva = ytr[va_i]
        np.savez_compressed(ART / f"{base}_valscores.npz", scores=scores, y=yva,
                            ts=tstr[va_i].astype("datetime64[ns]"), ue=uetr[va_i].astype(str))
        table = benign_quantile_table(scores, yva)
        if name == "ae":  # end-to-end reconstruction proof against the stored threshold
            rec = float(np.quantile(scores, 0.995))
            ok = np.isclose(rec, meta["threshold"], rtol=1e-5)
            print(f"AE w{w} threshold reproduction: stored={meta['threshold']:.6f} "
                  f"recomputed={rec:.6f} -> {'PASS' if ok else 'FAIL'}")
            if not ok:
                sys.exit(f"ABORT: validation reconstruction diverged for {base}")
        for q, thr in table.items():
            rows.append({"model": name, "window": w, "protocol": "temporal",
                         "seed": seed, "n_val": len(yva),
                         "n_val_benign": int((yva == 0).sum()),
                         "default_threshold": meta["threshold"], "q": q, "threshold": thr})
        print(f"{base}: archived {len(yva):,} val scores "
              f"({int((yva == 0).sum()):,} benign), ladder {list(table.values())}")

op = pd.DataFrame(rows)
op.to_csv(ART / "operating_points.csv", index=False)
print(f"\nsaved {ART / 'operating_points.csv'} ({len(op)} rows)")
print("\nthreshold ladder (rows=model_w, cols=q):")
piv = op.assign(mw=op.model + "_w" + op.window.astype(str)).pivot(
    index="mw", columns="q", values="threshold")
print(piv.round(6).to_string())
