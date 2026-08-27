#!/bin/zsh
# Window-knob batch (2026-07-03): arm the Phase-2 window
# knob with temporal-protocol artifacts at w5/w7 (every model was w3-only;
# Paper 1's best Transformer config used w7 — this closes that disclosed
# deviation). 8 runs, seed 42.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=.venv/bin/python
LOG=code/artifacts/logs
mkdir -p "$LOG"

for w in 5 7; do
  for m in lstm tcn transformer ae; do
    echo "=== $m temporal w$w ==="
    $PY -u code/src/train.py --model $m --protocol temporal --window $w > "$LOG/${m}_temporal_w${w}.log" 2>&1
  done
done
echo "WINDOW KNOB BATCH COMPLETE"
