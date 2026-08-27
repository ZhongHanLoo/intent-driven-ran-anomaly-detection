"""Threshold calibration for the Phase-2 intent layer.

Operating points (candidate detection thresholds) are quantiles of BENIGN
VALIDATION scores — computable at deployment time without test data, unlike
the ROC-style test-quantile ladders used for curve characterization in the
Phase-1 delay analysis (disclosed there as such). A quantile q maps to an
expected benign flag rate of ~(1-q): e.g. q=0.995 targets ~0.5% of benign
windows above threshold. The AE already ships its q=0.995 as its default
threshold; this module gives every model the full ladder.
"""

from __future__ import annotations

import numpy as np

DEFAULT_QS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)


def benign_quantile_table(scores: np.ndarray, y: np.ndarray, qs=DEFAULT_QS) -> dict:
    """{q: threshold} from the benign (y == 0) subset of validation scores."""
    ben = np.asarray(scores)[np.asarray(y) == 0]
    return {float(q): float(np.quantile(ben, q)) for q in qs}
