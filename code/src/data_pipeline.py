"""Data pipeline for NCSRD-DS-5GDDoS (Zenodo record 13900057).

Encodes the Phase-0 audit findings as code:
- true labels are DERIVED: a row is malicious iff its UE participated in that
  specific attack AND the row lies in the attack window (the file's own
  `attack_number` column flags whole windows for all UEs — innocents included);
- identity columns (imeisv, IPs, ...) never enter the feature matrix;
- windows never cross UE boundaries; the window label is the LAST row's label
  (the sequence classifier judges the most recent instant given short history);
- rows with NaN features are dropped before windowing (Paper 1 does the same);
- scaling is fit on training data only (leak-free);
- splits: 'random' (stratified, Paper 1-comparable) and 'temporal'
  (train strictly before test — the methodologically defensible protocol).

Windows MAY span idle gaps in a UE's record — an INTERPRETATION, not a
paper-matching fact: Paper 1 (p. 3311) says "data from periods of continuous
transmission were selected" without specifics; our derived-label counts match
theirs row-for-row, so any such selection had ~zero effect on labelled rows.
Recorded as a reproduction deviation; see the dissertation's Section 5.1.1.
Intra-attack gaps are ≤~10 s per the Phase 0 audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Paper 1's 14 KPM features (its Fig. 2 order); all exist verbatim in the merged CSV.
KPM_FEATURES = [
    "epre", "pusch_snr", "p_ue", "ul_mcs", "cqi", "ul_bitrate", "dl_mcs",
    "dl_retx", "ul_tx", "dl_tx", "ul_retx", "dl_bitrate", "dl_err", "ul_err",
]

# From summary_report.xlsx (audited 2026-07-02): who actually ran each attack.
IP_TO_IMEISV = {
    "10.20.10.2": "8642840401612300", "10.20.10.4": "8642840401624200",
    "10.20.10.6": "8642840401594200", "10.20.10.8": "8677660403123800",
    "10.20.10.10": "3557821101183501",
}
_PIS = {IP_TO_IMEISV["10.20.10.2"], IP_TO_IMEISV["10.20.10.4"]}
PARTICIPANTS = {1: _PIS, 2: _PIS, 3: _PIS, 4: _PIS, 5: set(IP_TO_IMEISV.values())}

# Audited attack windows (Phase 0 audit: attack_number spans match these to <5 s).
ATTACK_WINDOWS = {
    1: (pd.Timestamp("2024-08-18 07:00", tz="UTC"), pd.Timestamp("2024-08-18 08:00", tz="UTC")),
    2: (pd.Timestamp("2024-08-19 07:00", tz="UTC"), pd.Timestamp("2024-08-19 09:41", tz="UTC")),
    3: (pd.Timestamp("2024-08-19 17:00", tz="UTC"), pd.Timestamp("2024-08-19 18:00", tz="UTC")),
    4: (pd.Timestamp("2024-08-21 12:00", tz="UTC"), pd.Timestamp("2024-08-21 13:00", tz="UTC")),
    5: (pd.Timestamp("2024-08-21 17:00", tz="UTC"), pd.Timestamp("2024-08-21 18:00", tz="UTC")),
}
ATTACK_NAMES = {1: "SYN", 2: "ICMP", 3: "UDP-frag", 4: "DNS", 5: "GTP-U"}


def incidents_dict(keys=None):
    """{k: (start, end, participants)} for replay evaluation."""
    keys = keys or list(ATTACK_WINDOWS)
    return {k: (*ATTACK_WINDOWS[k], PARTICIPANTS[k]) for k in keys}


def load_raw(csv_path) -> pd.DataFrame:
    """Load the merged CSV, parse timestamps, drop empty-timestamp rows (audited:
    740 rows, all benign-window, unbiased), sort by (imeisv, time)."""
    df = pd.read_csv(csv_path, low_memory=False)
    df["imeisv"] = df["imeisv"].astype(str)
    df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["_time"])
    return df.sort_values(["imeisv", "_time"], kind="stable").reset_index(drop=True)


def derive_labels(df: pd.DataFrame, participants: dict | None = None) -> pd.DataFrame:
    """Add `y_attack` (0 = benign, k = attacker-active in attack k)."""
    participants = PARTICIPANTS if participants is None else participants
    df = df.copy()
    y = np.zeros(len(df), dtype=np.int64)
    for k, imeis in participants.items():
        y[(df["attack_number"].to_numpy() == k) & df["imeisv"].isin(imeis).to_numpy()] = k
    df["y_attack"] = y
    return df


def build_windows(df: pd.DataFrame, window: int = 3, features: list[str] = KPM_FEATURES,
                  details: bool = False):
    """Per-UE sliding windows. Returns (X, y, ts, ue) — with details=True also
    (y_type, touch):
    X: float32 (n, window, len(features)); y: int (n,) binary label of LAST row;
    ts: datetime64 (n,) of last row; ue: object (n,) UE id;
    y_type: int (n,) attack type (0-5) of the LAST row;
    touch: int (n,) bitmask — bit k set iff ANY row of the window has y_attack==k
    (catches trailing boundary windows whose last row is benign)."""
    df = df.dropna(subset=features)
    df = df.sort_values(["imeisv", "_time"], kind="stable")
    xs, ys, tss, ues, types, touches = [], [], [], [], [], []
    for ue_id, g in df.groupby("imeisv", sort=False):
        if len(g) < window:
            continue
        arr = g[features].to_numpy(dtype=np.float32)
        sw = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # (n-w+1, f, w)
        xs.append(np.ascontiguousarray(sw.transpose(0, 2, 1)))
        ya = g["y_attack"].to_numpy()
        ys.append((ya[window - 1:] > 0).astype(np.int64))
        tss.append(g["_time"].to_numpy()[window - 1:])
        ues.append(np.full(len(g) - window + 1, ue_id, dtype=object))
        if details:
            types.append(ya[window - 1:].astype(np.int64))
            bits = np.left_shift(1, ya).astype(np.int64) * (ya > 0)  # 0 for benign rows
            swb = np.lib.stride_tricks.sliding_window_view(bits, window)
            touches.append(np.bitwise_or.reduce(swb, axis=1))
    out = (np.concatenate(xs), np.concatenate(ys), np.concatenate(tss), np.concatenate(ues))
    if details:
        return out + (np.concatenate(types), np.concatenate(touches))
    return out


def split_temporal(ts: np.ndarray, test_frac: float = 0.2):
    """Chronological split: earliest (1-test_frac) of windows -> train, rest -> test."""
    order = np.argsort(ts, kind="stable")
    cut = int(round(len(order) * (1.0 - test_frac)))
    return order[:cut], order[cut:]


def split_random(y: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Stratified random split over window indices (Paper 1-comparable protocol)."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    return tr, te


def split_loao(y_type: np.ndarray, touch: np.ndarray, heldout: int,
               test_frac: float = 0.2, seed: int = 42):
    """Leave-one-attack-out fold ("Design 2"). Training pool: stratified
    (1-test_frac) of windows that do NOT touch the held-out attack. Test: the
    remaining eligible windows ∪ all windows LABELLED with the held-out attack.
    Touching-but-not-labelled boundary windows are dropped entirely.
    Returns (train_idx, test_idx, heldout_mask_within_test)."""
    touching = ((touch >> heldout) & 1).astype(bool)
    labelled = y_type == heldout
    eligible = np.where(~touching)[0]
    tr, te_e = train_test_split(eligible, test_size=test_frac,
                                stratify=(y_type[eligible] > 0), random_state=seed)
    te = np.concatenate([te_e, np.where(labelled)[0]])
    held_mask = np.zeros(len(te), dtype=bool)
    held_mask[len(te_e):] = True
    return tr, te, held_mask


def carve_validation(y: np.ndarray, seed: int = 42, frac: float = 0.1):
    """Supervised validation carve (fixed 2026-07-02 after the blind-early-
    stopping bug): stratified `frac` sample of training-window indices ->
    (fit_idx, val_idx). SINGLE SOURCE OF TRUTH — train.py and the validation
    operating-point archiver must both call this so archived thresholds match
    training exactly."""
    idx = np.arange(len(y))
    strat = y if y.sum() >= 2 else None
    return train_test_split(idx, test_size=frac, stratify=strat, random_state=seed)


def carve_validation_benign(y: np.ndarray, seed: int = 42, frac: float = 0.1):
    """AE validation carve: `frac` of BENIGN training windows -> (fit_idx,
    val_idx). The AE trains, early-stops AND derives its threshold quantile
    from this benign slice only."""
    ben = np.where(y == 0)[0]
    return train_test_split(ben, test_size=frac, random_state=seed)


def resample_smote_tomek(X: np.ndarray, y: np.ndarray, target: float = 1 / 9, seed: int = 42):
    """SMOTE + Tomek links on flattened windows (Paper 1's imbalance handling:
    oversample minority to ~1:9, then remove borderline pairs). Train data only."""
    from imblearn.combine import SMOTETomek
    n, w, f = X.shape
    Xf, yr = SMOTETomek(sampling_strategy=target, random_state=seed).fit_resample(
        X.reshape(n, -1), y)
    return Xf.reshape(-1, w, f).astype(np.float32), np.asarray(yr)


def scale_windows(X_train: np.ndarray, X_test: np.ndarray):
    """RobustScaler fit on TRAIN rows only; applied to both. Leak-free."""
    f = X_train.shape[-1]
    scaler = RobustScaler().fit(X_train.reshape(-1, f))
    tr = scaler.transform(X_train.reshape(-1, f)).reshape(X_train.shape).astype(np.float32)
    te = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape).astype(np.float32)
    return tr, te
