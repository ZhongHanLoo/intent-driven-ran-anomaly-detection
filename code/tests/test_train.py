"""Smoke tests for src.train: the trainer's
integration path — synthetic CSV -> windows -> split -> validation carve ->
1-epoch train -> metrics dict — must run end-to-end for each protocol family
and return a self-describing metrics dict. Artifact saving is disabled via
--limit-rows (the trainer skips saving when it is set; the value is large
enough that every synthetic row is kept)."""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import IP_TO_IMEISV, KPM_FEATURES
from src.train import build_parser, run

PI1 = IP_TO_IMEISV["10.20.10.2"]  # participates in every attack
PI2 = IP_TO_IMEISV["10.20.10.4"]
BENIGN_UES = ["1000000000000001", "1000000000000002"]


@pytest.fixture(scope="module")
def synth_csv(tmp_path_factory):
    """4 UEs x 240 rows @5 s. attack_number flags rows [60,100)=1 and
    [140,160)=2 for ALL UEs (the real file's over-labelling semantics); only
    the two Pi IMEISVs are participants, so derived labels mark them alone.
    Attacker features shift +8 inside windows so a 1-epoch run has signal.
    Both attacks sit in the earliest 80%% so the temporal train split holds
    positives (the 2026-07-02 validation-carve bug scenario)."""
    rng = np.random.default_rng(0)
    n = 240
    attack = np.zeros(n, dtype=int)
    attack[60:100] = 1
    attack[140:160] = 2
    frames = []
    for ue in [PI1, PI2] + BENIGN_UES:
        ts = pd.date_range("2024-08-18 00:00", periods=n, freq="5s", tz="UTC")
        d = {"_time": ts.astype(str), "imeisv": ue}
        for feat in KPM_FEATURES:
            x = rng.normal(0.0, 1.0, n)
            if ue in (PI1, PI2):
                x = x + (attack > 0) * 8.0
            d[feat] = x
        df = pd.DataFrame(d)
        df["attack_number"] = attack
        frames.append(df)
    p = tmp_path_factory.mktemp("data") / "synth.csv"
    pd.concat(frames, ignore_index=True).to_csv(p, index=False)
    return p


def _run(synth_csv, *cli):
    args = build_parser().parse_args(
        ["--csv", str(synth_csv), "--max-epochs", "1", "--limit-rows", "9999999", *cli])
    return run(args)


EXPECTED_KEYS = {"accuracy", "macro_f1", "fpr", "fnr", "mcc", "threshold",
                 "model", "protocol", "resample", "window", "seed",
                 "epochs_ran", "n_train", "n_test", "params"}


def test_train_smoke_temporal_lstm(synth_csv):
    m = _run(synth_csv, "--model", "lstm", "--protocol", "temporal")
    assert EXPECTED_KEYS <= set(m)  # self-describing provenance (incl. resample)
    assert m["model"] == "lstm" and m["protocol"] == "temporal" and m["resample"] == "none"
    assert m["epochs_ran"] >= 1 and m["n_train"] > 0 and m["n_test"] > 0
    assert 0.0 <= m["fpr"] <= 1.0 and 0.0 <= m["fnr"] <= 1.0


def test_train_smoke_loao_heldout_metrics(synth_csv):
    m = _run(synth_csv, "--model", "lstm", "--protocol", "loao", "--heldout", "1")
    assert m["heldout_n"] > 0
    assert 0.0 <= m["heldout_recall"] <= 1.0
    assert 0.0 <= m["heldout_auc_vs_benign"] <= 1.0
    assert "auc" in m  # this fold's test set holds both classes
    assert m["seen_recall"] is not None  # attack-2 windows reach the test split


def test_train_smoke_ae_benign_threshold(synth_csv):
    m = _run(synth_csv, "--model", "ae", "--protocol", "temporal")
    assert m["model"] == "ae" and m["epochs_ran"] >= 1
    assert m["threshold"] > 0.0  # benign-validation p99.5 reconstruction error
