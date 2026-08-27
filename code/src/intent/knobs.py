"""The knob space: every value a policy may legally set.
Latch and gap-reset stay FIXED constants (not knobs in v1, disclosed):
latch=True for FA counting, latch=False + fresh_only for delay, gap 30 s."""

MODELS = ("lstm", "tcn", "transformer", "ae")
WINDOWS = (3, 5, 7)
THRESHOLD_QS = (0.90, 0.95, 0.99, 0.995, 0.999, 0.9999)
PERSISTENCES = (1, 2, 3, 5)
DEFAULT_THRESHOLD = "default"


def knob_menu_text() -> str:
    """Plain-English knob menu as shown to both agents (kept compact).
    Every value list is interpolated from the constants above so the menu
    can never drift from the legal knob space."""
    supervised = ", ".join(m for m in MODELS if m != "ae")
    return (
        f"- model: one of {supervised} (supervised classifiers) or ae "
        "(benign-only autoencoder: detects novel attack types at ~10x benign false alarms"
        # measured result: AE FPR ~0.5% vs supervised ~0.05%, random split
        ")\n"
        f"- window: one of {', '.join(str(w) for w in WINDOWS)} — consecutive 5-second reports the detector sees at once\n"
        f"- threshold_q: one of {', '.join(str(q) for q in THRESHOLD_QS)} or \"{DEFAULT_THRESHOLD}\" — "
        "operating point as a benign-validation quantile; lower q = earlier detection, more false alarms; "
        f"\"{DEFAULT_THRESHOLD}\" = the model's trained operating point\n"
        f"- persistence: one of {', '.join(str(p) for p in PERSISTENCES)} — consecutive anomalous windows required before an alarm; "
        "higher = fewer false alarms, slower confirmation"
    )
