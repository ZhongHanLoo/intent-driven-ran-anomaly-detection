# Intent-Driven Policy Control for RAN Anomaly Detection

Code and evaluation artifacts for the MSc dissertation *Intent-Driven Policy
Control for RAN Anomaly Detection* (MSc Artificial Intelligence, University of
Surrey, 2026) by Zhong Han Loo, supervised by Dr Mahdi Boloursaz Mashhadi
(5G/6GIC, Institute for Communication Systems, University of Surrey).

The system translates high-level security intents (for example "early attack
detection" or "minimise false alarms") into concrete detector policies for a
RAN anomaly-detection fleet. A two-agent LLM layer (a Compiler and a Reviewer,
bounded by deterministic guardrails) selects one of 336 pre-evaluated
policies. A deterministic oracle grades every choice against replayed detector
scores from the public NCSRD-DS-5GDDoS dataset, measuring detection accuracy,
false alarms per hour and detection delay.

## Repository map

| Path | Contents |
|---|---|
| `code/src/` | data pipeline, detector models (LSTM, TCN, Transformer, autoencoder), training, replay and delay metrics |
| `code/src/intent/` | the intent layer: schemas, guardrails, Compiler and Reviewer, provider presets, executor, oracle, evaluation harness |
| `code/tests/` | offline test suite — runs with no dataset and no API keys |
| `code/eda/` | analysis scripts that regenerate the derived result tables |
| `code/artifacts/` | trained models, scalers, metrics, temporal score archives, raw experiment logs, LLM response cache |

## Quick start (no dataset, no API keys)

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=code .venv/bin/python -m pytest code/tests -q   # expect: 208 passed
```

## Replaying the recorded experiments

Every live LLM response is committed under
`code/artifacts/intent_runs/cache/`, so the recorded experiment matrices
replay deterministically without any API key. The raw run logs
(`matrix_*.jsonl`) and the script-generated result tables (`*.analysis.txt`,
`*.cmp*.txt`, `*.lead_table.csv`) are committed beside them, and the scripts
under `code/eda/` regenerate the derived tables from the logs.

## Retraining from the dataset

```bash
bash code/download_data.sh   # ~242 MB from Zenodo, checksum-verified
```

This fetches the NCSRD-DS-5GDDoS dataset (v3) from Zenodo with checksum
verification. The dataset is not redistributed in this repository. It is
published at Zenodo record 13900057 (concept DOI 10.5281/zenodo.10671493)
under the Creative Commons Attribution 4.0 International (CC BY 4.0) license;
please observe its terms.

## Live LLM runs (optional)

New live runs need a provider API key: copy `code/.secrets.env.example` to
`code/.secrets.env` and fill in the key for the provider you select. Recorded
results never require this.

## License

MIT — see `LICENSE`. The dataset has its own license and terms (above).
