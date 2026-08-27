"""Unified detector training: 4 models x 3 protocols.

Models:    lstm | tcn | transformer (supervised, BCE)   ae (unsupervised, benign-only MSE)
Protocols: random   — stratified 80/20 (Paper 1-comparable)
           temporal — train strictly before test
           loao     — leave-one-attack-out (--heldout 1..5), "Design 2"

Validation (all protocols, fixed 2026-07-02 after the blind-early-stopping bug):
stratified 10% sample of TRAINING windows (for temporal: sampled within the
training period — time-purity vs test preserved; used only for early stopping).
Carve functions live in data_pipeline (shared with the operating-point
archiver). AE threshold = 99.5th percentile of benign-validation
reconstruction errors.

Usage examples:
  train.py --model tcn --protocol random
  train.py --model ae  --protocol loao --heldout 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_pipeline import (
    build_windows, carve_validation, carve_validation_benign, derive_labels,
    load_raw, scale_windows, split_loao, split_random, split_temporal,
)
from src.evaluate import compute_metrics
from src.models import make_model

ROOT = Path(__file__).resolve().parents[2]


def loader(X, y, batch, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float()),
                      batch_size=batch, shuffle=shuffle)


def train_supervised(model, Xfit, yfit, Xva, yva, device, args):
    dl_tr, dl_va = loader(Xfit, yfit, args.batch, True), loader(Xva, yva, args.batch, False)
    opt, lossf = torch.optim.Adam(model.parameters()), nn.BCEWithLogitsLoss()
    best, best_state, stall = float("inf"), None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            lossf(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                va += lossf(model(xb.to(device)), yb.to(device)).item() * len(yb)
        va /= len(yva)
        print(f"epoch {epoch:2d} | val {va:.6f}", flush=True)
        if va < best - 1e-6:
            best, best_state, stall = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stall += 1
            if stall >= args.patience:
                break
    model.load_state_dict(best_state)
    return model, epoch


def train_ae(model, Xtr, ytr, device, args, seed):
    tr_i, va_i = carve_validation_benign(ytr, seed)
    dl_tr, dl_va = loader(Xtr[tr_i], ytr[tr_i], args.batch, True), loader(Xtr[va_i], ytr[va_i], args.batch, False)
    opt = torch.optim.Adam(model.parameters())
    best, best_state, stall = float("inf"), None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        for xb, _ in dl_tr:
            xb = xb.to(device)
            opt.zero_grad()
            ((model(xb) - xb) ** 2).mean().backward()
            opt.step()
        model.eval()
        va = 0.0
        with torch.no_grad():
            for xb, _ in dl_va:
                va += model.score(xb.to(device)).sum().item()
        va /= len(va_i)
        print(f"epoch {epoch:2d} | val_mse {va:.6f}", flush=True)
        if va < best - 1e-7:
            best, best_state, stall = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stall += 1
            if stall >= args.patience:
                break
    model.load_state_dict(best_state)
    # threshold: 99.5th percentile of benign-validation reconstruction error
    with torch.no_grad():
        errs = np.concatenate([model.score(xb.to(device)).cpu().numpy() for xb, _ in dl_va])
    return model, epoch, float(np.quantile(errs, 0.995))


def score_all(model, X, device, batch, is_ae):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch * 16):
            xb = torch.from_numpy(X[i: i + batch * 16]).to(device)
            out.append((model.score(xb) if is_ae else torch.sigmoid(model(xb))).cpu().numpy())
    return np.concatenate(out)


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm", "tcn", "transformer", "ae"], required=True)
    ap.add_argument("--protocol", choices=["random", "temporal", "loao"], default="random")
    ap.add_argument("--heldout", type=int, default=0, help="attack type 1-5 for loao")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--resample", choices=["none", "smote-tomek"], default="none")
    ap.add_argument("--csv", default=str(ROOT / "data" / "amari_ue_data_merged_with_attack_number.csv"))
    return ap


def run(args):
    assert args.protocol != "loao" or 1 <= args.heldout <= 5
    assert not (args.resample != "none" and args.model == "ae"), "resampling is for supervised models"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()
    tag = f"loao{args.heldout}" if args.protocol == "loao" else args.protocol
    if args.resample != "none":
        tag += "_smote"

    df = load_raw(args.csv)
    if args.limit_rows:
        df = df.groupby("imeisv", sort=False).head(args.limit_rows // 9)
    df = derive_labels(df)
    X, y, ts, ue, y_type, touch = build_windows(df, window=args.window, details=True)
    held_mask = None
    if args.protocol == "random":
        tr, te = split_random(y, seed=args.seed)
    elif args.protocol == "temporal":
        tr, te = split_temporal(ts)
    else:
        tr, te, held_mask = split_loao(y_type, touch, args.heldout, seed=args.seed)
    print(f"{args.model} | {tag} | train {len(tr):,} ({y[tr].sum():,}+) | test {len(te):,} ({y[te].sum():,}+)", flush=True)

    Xtr, Xte = scale_windows(X[tr], X[te])
    ytr, yte = y[tr], y[te]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = make_model(args.model, window=args.window).to(device)

    if args.model == "ae":
        model, epochs, thr = train_ae(model, Xtr, ytr, device, args, args.seed)
    else:
        tr_i, va_i = carve_validation(ytr, args.seed)
        Xfit, yfit = Xtr[tr_i], ytr[tr_i]
        if args.resample == "smote-tomek":
            from src.data_pipeline import resample_smote_tomek
            Xfit, yfit = resample_smote_tomek(Xfit, yfit, seed=args.seed)
            print(f"resampled train: {len(yfit):,} windows, {yfit.mean()*100:.1f}% positive", flush=True)
        model, epochs = train_supervised(model, Xfit, yfit, Xtr[va_i], ytr[va_i], device, args)
        thr = 0.5

    model.eval()
    scores = score_all(model, Xte, device, args.batch, args.model == "ae")
    m = compute_metrics(yte, scores, threshold=thr)
    if len(np.unique(yte)) == 2:
        m["auc"] = float(roc_auc_score(yte, scores))
    m.update({"model": args.model, "protocol": args.protocol, "heldout": args.heldout,
              "resample": args.resample, "window": args.window, "seed": args.seed, "epochs_ran": epochs,
              "n_train": int(len(tr)), "n_test": int(len(te)),
              "params": int(sum(p.numel() for p in model.parameters())),
              "runtime_s": round(time.time() - t0, 1)})
    if held_mask is not None and held_mask.any():
        pred = scores >= thr
        seen = (yte == 1) & ~held_mask
        m["heldout_n"] = int(held_mask.sum())
        m["heldout_recall"] = float(pred[held_mask].mean())
        ben = yte == 0
        m["heldout_auc_vs_benign"] = float(roc_auc_score(
            np.r_[np.zeros(ben.sum()), np.ones(held_mask.sum())],
            np.r_[scores[ben], scores[held_mask]]))
        m["seen_recall"] = float(pred[seen].mean()) if seen.any() else None
    print("TEST METRICS:", json.dumps(m, indent=2), flush=True)

    if not args.limit_rows:
        outdir = ROOT / "code" / "artifacts"
        outdir.mkdir(exist_ok=True)
        base = f"{args.model}_w{args.window}_{tag}" + (f"_s{args.seed}" if args.seed != 42 else "")
        torch.save(model.state_dict(), outdir / f"{base}.pt")
        json.dump(m, open(outdir / f"{base}_metrics.json", "w"), indent=2)
        np.savez_compressed(outdir / f"{base}_scores.npz", scores=scores, y=yte,
                            held=held_mask if held_mask is not None else np.zeros(0, bool),
                            ts=ts[te].astype("datetime64[ns]"), ue=ue[te].astype(str))
        from sklearn.preprocessing import RobustScaler
        # refit on the same float32 X[tr] is bit-identical to the scaler used in
        # scale_windows (RobustScaler is deterministic; same-dtype input matters)
        joblib.dump(RobustScaler().fit(X[tr].reshape(-1, X.shape[-1])), outdir / f"{base}_scaler.joblib")
        print(f"artifacts saved: code/artifacts/{base}.*", flush=True)
    return m


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
