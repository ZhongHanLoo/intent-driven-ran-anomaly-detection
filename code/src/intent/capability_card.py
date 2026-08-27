"""The capability card: the measured-evidence text both agents see.
Dynamic numbers come from the committed artifact CSVs; the finding lines are
versioned constants sourced from the project's measured Phase-1 results.
Never hand-edit generated output — edit this module and regenerate."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CARD_VERSION = 1

_FINDINGS = """\
REGIMES for NEVER-SEEN attack types (multi-seed LOAO evidence):
- threshold-fixable: ICMP-like (ranking fine, fixed threshold wrong)
- model-choice-sensitive: DNS-like (TCN AUC 0.79 vs transformer 1.00)
- model-class-limited: GTP-U-like — NO supervised policy detects it; only ae ranks it usefully (AUC 0.78)
Other measured facts:
- transformer avoids catastrophic per-attack failures; lstm/tcn each fail badly on one attack type
- no window (3/5/7) rescues unseen-attack generalization; window effects are modest, w3 is a sane default
- ae detects novel attack types but costs ~10x benign false alarms (~0.5% vs ~0.05%), worse under drift
- persistence 1->3 alone cut ae false alarms 24.3 -> 7.7 per hour in the temporal replay
- trade-off directions: lower threshold_q => earlier detection + more false alarms;
  higher persistence => fewer false alarms + slower confirmation"""

_MODEL_ORDER = ("lstm", "tcn", "transformer", "ae")


def build_card(art_dir: Path) -> str:
    art = Path(art_dir)
    op = pd.read_csv(art / "operating_points.csv")
    lines = [f"CAPABILITY CARD v{CARD_VERSION} (auto-generated from measured artifacts)", "",
             _FINDINGS, "",
             "OPERATING POINTS (benign-validation quantile -> absolute threshold; 'default' = trained point):"]
    for (m, w), g in op.groupby(["model", "window"]):
        default = g.default_threshold.iloc[0]
        ladder = ", ".join(f"q{row.q:g}={row.threshold:.4g}" for row in g.sort_values("q").itertuples())
        lines.append(f"- {m} w{int(w)}: default={default:.4g}; {ladder}")
    dt = art / "delay_temporal.csv"
    if dt.exists():
        td = pd.read_csv(dt)
        ref = td[(td.persistence == 1) & (td.window == 3)]
        defaults = {m: float(op[(op.model == m) & (op.window == 3)].default_threshold.iloc[0])
                    for m in _MODEL_ORDER}
        lines.append("")
        lines.append("TEMPORAL REPLAY at default threshold, persistence 1, window 3 "
                     "(false alarms/hour | DNS delay s | GTP-U delay s; NaN = missed):")
        for m in _MODEL_ORDER:
            g = ref[(ref.model == m) & ((ref.threshold - defaults[m]).abs() < 1e-9)]
            if len(g):
                r = g.iloc[0]
                lines.append(f"- {m}: {r.fa_per_hour:.2f}/h | {r.dns_delay_s} | {r.gtpu_delay_s}")
    return "\n".join(lines)
