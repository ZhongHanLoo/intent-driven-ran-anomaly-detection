#!/bin/zsh
# Hardening batch (2026-07-02): LOAO seed replication (43, 44) to put error
# bars on the single-seed claims, SMOTE+Tomek variants (Paper 1's resampled
# models), and a unified-trainer LSTM random rerun (adds AUC + score archive).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=code
PY=.venv/bin/python
LOG=code/artifacts/logs
mkdir -p "$LOG"

for seed in 43 44; do
  for m in lstm tcn transformer ae; do
    for k in 1 2 3 4 5; do
      echo "=== $m loao$k seed$seed ==="
      $PY -u code/src/train.py --model $m --protocol loao --heldout $k --seed $seed > "$LOG/${m}_loao${k}_s${seed}.log" 2>&1
    done
  done
done
for m in lstm tcn transformer; do
  echo "=== $m random smote ==="
  $PY -u code/src/train.py --model $m --protocol random --resample smote-tomek > "$LOG/${m}_random_smote.log" 2>&1
done
echo "=== lstm random rerun (unified) ==="
$PY -u code/src/train.py --model lstm --protocol random > "$LOG/lstm_random_rerun.log" 2>&1
echo "HARDENING BATCH COMPLETE"
