"""Evaluation metrics. Definitions:
FPR (false-positive rate / false-alarm rate) = FP / (FP + TN)   [benign wrongly alarmed]
FNR (false-negative rate / missed-attack rate) = FN / (FN + TP) [attacks missed]
MCC computed from the confusion matrix directly (0 when undefined).
Zero-division cases report 0.0 so degenerate slices don't crash sweeps.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "threshold": float(threshold),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="weighted", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "mcc": float((tp * tn - fp * fn) / denom) if denom else 0.0,
    }
