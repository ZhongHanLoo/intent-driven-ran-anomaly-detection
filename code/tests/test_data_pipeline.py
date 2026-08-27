"""Tests for src.data_pipeline — written FIRST (TDD).

Fixture mimics the real file's semantics discovered in Phase 0:
- `attack_number` flags the whole time window for ALL UEs (innocents included);
- true labels must come from participants-of-that-attack ∧ window;
- identity columns exist and must never reach the feature matrix.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    KPM_FEATURES,
    build_windows,
    carve_validation,
    carve_validation_benign,
    derive_labels,
    resample_smote_tomek,
    scale_windows,
    split_loao,
    split_random,
    split_temporal,
)


def make_synth_df():
    """Two UEs x 10 rows @5s. Rows 4-7 sit inside attack window #1 (flagged for
    BOTH UEs, as in the real file). Only UE 'A' actually participates."""
    frames = []
    for ue, base in [("A", 0.0), ("B", 100.0)]:
        ts = pd.date_range("2024-08-18 06:59:40", periods=10, freq="5s", tz="UTC")
        d = {"_time": ts, "imeisv": ue, "bearer_0_ip": f"10.0.0.{base:.0f}"}
        for j, f in enumerate(KPM_FEATURES):
            d[f] = [base + j + i for i in range(10)]
        f = pd.DataFrame(d)
        f["attack_number"] = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


PARTICIPANTS = {1: {"A"}}


def test_kpm_features_are_papers_14():
    assert len(KPM_FEATURES) == 14
    assert set(KPM_FEATURES) == {
        "epre", "pusch_snr", "p_ue", "ul_mcs", "cqi", "ul_bitrate", "dl_mcs",
        "dl_retx", "ul_tx", "dl_tx", "ul_retx", "dl_bitrate", "dl_err", "ul_err",
    }


def test_derive_labels_marks_only_participants_inside_window():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    a = df[df["imeisv"] == "A"]["y_attack"].tolist()
    b = df[df["imeisv"] == "B"]["y_attack"].tolist()
    assert a == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  # attacker: labelled inside window
    assert b == [0] * 10  # innocent UE inside same window: never labelled


def test_build_windows_shapes_last_row_label_and_no_ue_crossing():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    X, y, ts, ue = build_windows(df, window=3)
    # 10 rows per UE -> 8 windows per UE; never mixes UEs
    assert X.shape == (16, 3, 14)
    assert y.shape == (16,) and set(np.unique(ue)) == {"A", "B"}
    # windows are consecutive rows of one UE: feature 0 increases by exactly 1 per step
    assert np.allclose(X[:, 1, 0] - X[:, 0, 0], 1) and np.allclose(X[:, 2, 0] - X[:, 1, 0], 1)
    # label = last row's label: for UE A, windows ending at rows 4..7 are 1
    ya = y[ue == "A"]
    assert ya.tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    assert y[ue == "B"].sum() == 0


def test_build_windows_drops_rows_with_nan_features():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    df.loc[(df["imeisv"] == "A") & (df[KPM_FEATURES[0]] == 5), KPM_FEATURES[1]] = np.nan
    X, y, ts, ue = build_windows(df, window=3)
    # one A-row dropped -> A has 9 rows -> 7 windows; B unchanged with 8
    assert (ue == "A").sum() == 7 and (ue == "B").sum() == 8


def test_temporal_split_has_no_time_overlap():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    X, y, ts, ue = build_windows(df, window=3)
    tr, te = split_temporal(ts, test_frac=0.25)
    assert len(te) + len(tr) == len(ts)
    assert ts[tr].max() <= ts[te].min()  # strictly ordered: train fully before test


def test_random_split_is_stratified_and_sized():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    tr, te = split_random(y, test_frac=0.2, seed=42)
    assert len(te) == 200 and len(tr) == 800
    assert abs(y[te].mean() - y.mean()) < 0.02  # stratification keeps prevalence


def test_build_windows_details_expose_type_and_touch():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    X, y, ts, ue, y_type, touch = build_windows(df, window=3, details=True)
    a = ue == "A"
    # last-row attack type: windows ending at rows 4..7 carry type 1
    assert y_type[a].tolist() == [0, 0, 1, 1, 1, 1, 0, 0]
    # touch bit 1 set for any window CONTAINING an attack row: rows 4..7 appear
    # in windows ending at rows 4..9 (trailing boundary windows included)
    assert [(t >> 1) & 1 for t in touch[a]] == [0, 0, 1, 1, 1, 1, 1, 1]
    assert touch[ue == "B"].sum() == 0  # innocent UE never touches


def test_split_loao_excludes_heldout_from_train_entirely():
    df = derive_labels(make_synth_df(), PARTICIPANTS)
    X, y, ts, ue, y_type, touch = build_windows(df, window=3, details=True)
    tr, te, held = split_loao(y_type, touch, heldout=1, test_frac=0.25, seed=0)
    # no training window may even TOUCH attack 1
    assert all((touch[i] >> 1) & 1 == 0 for i in tr)
    # every window LABELLED attack 1 is in test, flagged as held-out
    labelled = set(np.where(y_type == 1)[0])
    assert labelled <= set(te[held])
    # train contains no positives of the held-out type
    assert (y_type[tr] == 1).sum() == 0


def test_resample_smote_tomek_reaches_target_prevalence():
    rng = np.random.default_rng(3)
    Xn = rng.normal(0, 1, size=(600, 3, 14)).astype(np.float32)
    Xp = rng.normal(4, 1, size=(12, 3, 14)).astype(np.float32)
    X = np.concatenate([Xn, Xp])
    y = np.r_[np.zeros(600, int), np.ones(12, int)]
    Xr, yr = resample_smote_tomek(X, y, target=1 / 9, seed=0)
    assert Xr.shape[1:] == (3, 14) and len(Xr) == len(yr)
    assert 0.07 <= yr.mean() <= 0.13  # ~10% positives after SMOTE + Tomek
    assert len(Xr) > len(X)  # synthetic positives were added


def test_carve_validation_matches_trainer_convention():
    # Single source of truth for the validation carve:
    # the operating-point archiver must reproduce the trainer's carve EXACTLY,
    # so both call this function. Pins equivalence with the historical inline
    # formula, and pins the 2026-07-02 blind-validation bugfix (val must
    # contain positives when the training pool does).
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(5)
    y = (rng.random(500) < 0.05).astype(int)
    fit_i, va_i = carve_validation(y, seed=42)
    strat = y if y.sum() >= 2 else None
    exp_fit, exp_va = train_test_split(np.arange(len(y)), test_size=0.1,
                                       stratify=strat, random_state=42)
    assert np.array_equal(fit_i, exp_fit) and np.array_equal(va_i, exp_va)
    assert set(fit_i).isdisjoint(va_i) and len(fit_i) + len(va_i) == len(y)
    assert y[va_i].sum() >= 1  # stratified carve keeps positives in validation


def test_carve_validation_benign_for_ae():
    # AE trains/early-stops/thresholds on benign windows only.
    from sklearn.model_selection import train_test_split

    y = np.r_[np.zeros(90, int), np.ones(10, int)]
    tr_i, va_i = carve_validation_benign(y, seed=42)
    assert (y[tr_i] == 0).all() and (y[va_i] == 0).all()
    assert len(tr_i) + len(va_i) == 90
    exp_tr, exp_va = train_test_split(np.where(y == 0)[0], test_size=0.1, random_state=42)
    assert np.array_equal(tr_i, exp_tr) and np.array_equal(va_i, exp_va)


def test_scaler_is_fit_on_train_only():
    rng = np.random.default_rng(1)
    Xtr = rng.normal(0, 1, size=(50, 3, 14)).astype(np.float32)
    Xte = rng.normal(10, 1, size=(20, 3, 14)).astype(np.float32)  # shifted distribution
    Xtr_s, Xte_s = scale_windows(Xtr, Xte)
    # train medians ~0 (RobustScaler centers on TRAIN median)...
    assert np.abs(np.median(Xtr_s.reshape(-1, 14), axis=0)).max() < 0.2
    # ...while shifted test stays far from 0 => test stats were NOT used for fitting
    assert np.median(Xte_s.reshape(-1, 14), axis=0).min() > 5
