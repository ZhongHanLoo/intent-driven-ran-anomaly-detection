#!/bin/zsh
# Option C batch (2026-07-02): standard-split runs for the three new models,
# temporal re-runs with fixed validation for all four, then the full
# leave-one-attack-out study (4 models x 5 folds). ~27 trainings, sequential.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=.venv/bin/python
LOG=code/artifacts/logs
mkdir -p "$LOG"

for m in tcn transformer ae; do
  echo "=== $m random ==="
  $PY -u code/src/train.py --model $m --protocol random > "$LOG/${m}_random.log" 2>&1
done
for m in lstm tcn transformer ae; do
  echo "=== $m temporal ==="
  $PY -u code/src/train.py --model $m --protocol temporal > "$LOG/${m}_temporal.log" 2>&1
done
for m in lstm tcn transformer ae; do
  for k in 1 2 3 4 5; do
    echo "=== $m loao$k ==="
    $PY -u code/src/train.py --model $m --protocol loao --heldout $k > "$LOG/${m}_loao${k}.log" 2>&1
  done
done
echo "OPTION C BATCH COMPLETE"
