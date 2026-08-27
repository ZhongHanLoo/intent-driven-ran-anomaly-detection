"""Diagnose the temporal-split FNR=1.0 result: is the model blind to the unseen
attack types, or merely scoring them below the fixed 0.5 threshold?

Prints: split composition (which attacks fall in train/val/test), score
distributions per class and per attack type, ROC-AUC, and detections/FPR at a
ladder of thresholds.
"""

from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_pipeline import build_windows, derive_labels, load_raw, scale_windows, split_temporal
from src.models import LstmDetector

ROOT = Path(__file__).resolve().parents[2]

df = derive_labels(load_raw(ROOT / "data" / "amari_ue_data_merged_with_attack_number.csv"))
X, y, ts, ue = build_windows(df, window=3)
yk = np.zeros(len(y), dtype=int)  # per-attack-type label of window (last row), for breakdown
# recompute per-type label of last rows (build_windows binarizes; redo cheaply)
d2 = df.dropna(subset=[c for c in df.columns if c in set(__import__('src.data_pipeline', fromlist=['KPM_FEATURES']).KPM_FEATURES)])
tr, te = split_temporal(ts)

print("=== split composition ===")
print(f"train: {ts[tr].min()} -> {ts[tr].max()}  ({len(tr):,} windows, {y[tr].sum():,} positives)")
print(f"test : {ts[te].min()} -> {ts[te].max()}  ({len(te):,} windows, {y[te].sum():,} positives)")
order = np.argsort(ts[tr], kind="stable")
va = order[int(round(len(order) * 0.9)):]
print(f"val (last 10% of train): {ts[tr][va].min()} -> {ts[tr][va].max()}  positives: {y[tr][va].sum():,}")

Xtr, Xte = scale_windows(X[tr], X[te])
model = LstmDetector()
model.load_state_dict(torch.load(ROOT / "code" / "artifacts" / "lstm_w3_temporal.pt", weights_only=True))
model.eval()
with torch.no_grad():
    scores = torch.sigmoid(model(torch.from_numpy(Xte))).numpy()

yt = y[te]
ben, att = scores[yt == 0], scores[yt == 1]
print("\n=== test score distributions ===")
print(f"benign : p50 {np.percentile(ben,50):.4g}  p99 {np.percentile(ben,99):.4g}  p99.9 {np.percentile(ben,99.9):.4g}  max {ben.max():.4g}")
print(f"attack : p10 {np.percentile(att,10):.4g}  p50 {np.percentile(att,50):.4g}  p90 {np.percentile(att,90):.4g}  max {att.max():.4g}")

# per attack type in test: match window ts against known windows
import pandas as pd
W = {4: ("DNS ", pd.Timestamp("2024-08-21 12:00", tz="UTC"), pd.Timestamp("2024-08-21 13:00", tz="UTC")),
     5: ("GTPU", pd.Timestamp("2024-08-21 17:00", tz="UTC"), pd.Timestamp("2024-08-21 18:00", tz="UTC"))}
tse = pd.DatetimeIndex(ts[te])
for k, (name, s, e) in W.items():
    m = (yt == 1) & (tse >= s) & (tse < e)
    if m.sum():
        a = scores[m.to_numpy() if hasattr(m, "to_numpy") else m]
        print(f"attack {k} {name}: n={int(np.sum(m)):5d}  p50 {np.percentile(a,50):.4g}  p90 {np.percentile(a,90):.4g}  max {a.max():.4g}")

from sklearn.metrics import roc_auc_score
print(f"\nROC-AUC on temporal test: {roc_auc_score(yt, scores):.4f}")

# CRITICAL CONTROL: does this same model detect the attacks it TRAINED on?
# Score the train-period windows (SYN/ICMP/UDP-frag era).
with torch.no_grad():
    str_ = torch.sigmoid(model(torch.from_numpy(Xtr))).numpy()
ytr_ = y[tr]
btr, atr = str_[ytr_ == 0], str_[ytr_ == 1]
print("\n=== control: scores on TRAIN-period windows (seen attack types) ===")
print(f"benign : p50 {np.percentile(btr,50):.4g}  p99 {np.percentile(btr,99):.4g}  max {btr.max():.4g}")
print(f"attack : p10 {np.percentile(atr,10):.4g}  p50 {np.percentile(atr,50):.4g}  p90 {np.percentile(atr,90):.4g}  max {atr.max():.4g}")
print(f"train-period ROC-AUC: {roc_auc_score(ytr_, str_):.4f}")
print(f"train-period recall @0.5: {(atr >= 0.5).mean():.3f}")

print("\n=== threshold ladder (temporal test) ===")
print(f"{'thr':>6s} {'TP':>6s} {'FN':>6s} {'recall':>7s} {'FP':>7s} {'FPR%':>8s}")
for thr in [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
    pred = scores >= thr
    tp = int((pred & (yt == 1)).sum()); fn = int(((~pred) & (yt == 1)).sum())
    fp = int((pred & (yt == 0)).sum())
    print(f"{thr:6.3f} {tp:6d} {fn:6d} {tp/(tp+fn):7.3f} {fp:7d} {fp/max(1,(yt==0).sum())*100:8.4f}")
