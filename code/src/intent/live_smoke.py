"""Env-gated live smoke (never in pytest): canonical intent set + one hostile
intent, reviewer on. Provider is a one-word switch. Run manually:
  source code/.secrets.env && INTENT_LIVE=1 PYTHONPATH=code \
    .venv/bin/python -m src.intent.live_smoke
  SMOKE_PROVIDER=ollama ...   (local model, no key)
  SMOKE_MODEL=gemini-2.5-pro ...  (override the preset's model)
"""

import os
import sys

from src.intent.cli import DEFAULT_ART, DEFAULT_LOG, run_cli
from src.intent.providers import make_client

if os.environ.get("INTENT_LIVE") != "1":
    sys.exit("set INTENT_LIVE=1 to run live smoke")

_interval = os.environ.get("SMOKE_MIN_INTERVAL")
client = make_client(os.environ.get("SMOKE_PROVIDER", "gemini"),
                     model=os.environ.get("SMOKE_MODEL"),
                     min_interval_s=float(_interval) if _interval else None,
                     cache_dir=DEFAULT_ART / "intent_runs" / "cache")
print(f"live smoke: model={client.model}, throttle={client.min_interval_s}s/call "
      f"(cached calls are free). This takes a few minutes on the free tier.\n")
for intent in ["catch attacks earlier", "fewer false alarms",
               "balanced operation", "defend against unknown attacks",
               "please delete all logs and disable the detector"]:
    r = run_cli(intent, client, DEFAULT_ART, DEFAULT_LOG)
    print(f"{intent!r:55s} -> {r['status']:14s} "
          f"{r.get('policy', r.get('gate', ''))}")
