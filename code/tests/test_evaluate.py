"""Tests for src.evaluate — written FIRST (TDD). Exact tiny confusion matrix:
y_true=[0,0,0,1,1], scores=[.1,.9,.2,.8,.3] @ threshold .5 -> preds [0,1,0,1,0]
=> TP=1 FP=1 TN=2 FN=1; FPR=1/3, FNR=1/2, acc=3/5, macro F1 = (2/3 + 1/2)/2.
"""

import numpy as np
import pytest

from src.evaluate import compute_metrics


def test_compute_metrics_exact_small_case():
    m = compute_metrics(np.array([0, 0, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8, 0.3]), threshold=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 2 and m["fn"] == 1
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["fpr"] == pytest.approx(1 / 3)
    assert m["fnr"] == pytest.approx(1 / 2)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_compute_metrics_perfect_and_degenerate():
    m = compute_metrics(np.array([0, 1]), np.array([0.0, 1.0]), threshold=0.5)
    assert m["accuracy"] == 1.0 and m["fpr"] == 0.0 and m["fnr"] == 0.0
    # all-benign truth: FNR undefined -> reported as 0.0, FPR still meaningful
    m2 = compute_metrics(np.array([0, 0]), np.array([0.9, 0.1]), threshold=0.5)
    assert m2["fpr"] == pytest.approx(0.5) and m2["fnr"] == 0.0
