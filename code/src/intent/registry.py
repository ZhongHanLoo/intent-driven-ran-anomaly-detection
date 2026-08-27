"""Artifact registry: maps (model, window) to the
on-disk artifacts a policy needs. Temporal protocol only in v1."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_MODELS = {"lstm", "tcn", "transformer", "ae"}
_PAT = re.compile(r"^([a-z]+)_w(\d+)_temporal_metrics\.json$")


class Registry:
    def __init__(self, art_dir: Path):
        self.art = Path(art_dir)
        self._pairs = {}
        for p in self.art.glob("*_temporal_metrics.json"):
            m = _PAT.match(p.name)
            if m and m.group(1) in _MODELS:
                self._pairs[(m.group(1), int(m.group(2)))] = p
        csv_path = self.art / "operating_points.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"operating_points.csv not found in {self.art}; "
                "run code/eda/archive_val_operating_points.py first")
        op = pd.read_csv(csv_path)
        # {(model, window, q): threshold}
        self._ladder = {(r.model, int(r.window), float(r.q)): float(r.threshold)
                        for r in op.itertuples()}

    def pairs(self) -> set[tuple[str, int]]:
        return set(self._pairs)

    def _check(self, model: str, window: int):
        if (model, int(window)) not in self._pairs:
            raise KeyError(f"no temporal artifacts for ({model}, w{window})")

    def metrics(self, model: str, window: int) -> dict:
        window = int(window)
        self._check(model, window)
        with open(self._pairs[(model, window)]) as fh:
            return json.load(fh)

    def score_archive(self, model: str, window: int) -> Path:
        window = int(window)
        self._check(model, window)
        return self.art / f"{model}_w{window}_temporal_scores.npz"

    def resolve_threshold(self, model: str, window: int, threshold_q) -> float:
        """Quantile choice -> absolute threshold. 'default' = trained operating
        point (0.5 supervised; the AE's stored benign-p99.5)."""
        window = int(window)
        self._check(model, window)
        if threshold_q == "default":
            return float(self.metrics(model, window)["threshold"])
        key = (model, window, float(threshold_q))
        if key not in self._ladder:
            raise KeyError(f"no operating point for {key}")
        return self._ladder[key]
