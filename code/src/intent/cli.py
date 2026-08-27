"""CLI: translate one intent, optionally execute it, log everything.
Usage (live) — provider is a one-word switch, presets in providers.py:
  source code/.secrets.env && PYTHONPATH=code .venv/bin/python -m src.intent.cli \
    --intent "fewer false alarms" --provider gemini
  ... --provider ollama                      (local, no key)
  ... --provider openai --llm-model gpt-4.1  (preset + model override)
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from src.intent.apply import apply_policy
from src.intent.llm_client import LLMClient, append_jsonl
from src.intent.prompting import PROMPT_VERSION
from src.intent.providers import PROVIDERS, make_client
from src.intent.registry import Registry
from src.intent.reviewer import run_two_agent

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ART = ROOT / "code" / "artifacts"
DEFAULT_LOG = DEFAULT_ART / "intent_runs" / "manual.jsonl"


def run_cli(intent: str, client, art_dir: Path, runlog: Path,
            execute: bool = True, use_reviewer: bool = True,
            loop_iterations: int = 0) -> dict:
    registry = Registry(art_dir)
    tok0 = client.tokens_used  # per-run delta: a shared client's counter is cumulative
    if loop_iterations > 0:  # Loop-B (staged): metric-feedback refinement
        from src.intent.loop import run_loop
        lr = run_loop(intent, client, registry, art_dir,
                      max_iterations=loop_iterations, use_reviewer=use_reviewer,
                      runlog=runlog)
        record = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "prompt_version": PROMPT_VERSION, "llm_model": client.model,
                  "intent": intent,
                  "status": "policy" if lr.final_policy else lr.status,
                  "loop": {"status": lr.status, "iterations": lr.iterations,
                           "trajectory": lr.trajectory},
                  "tokens_delta": client.tokens_used - tok0,
                  "tokens_total": client.tokens_used}
        if lr.final_policy:
            record["policy"] = lr.final_policy
            record["metrics"] = lr.final_metrics
        append_jsonl(runlog, record)
        return record
    result = run_two_agent(intent, client, registry, art_dir, use_reviewer=use_reviewer)
    record = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "prompt_version": PROMPT_VERSION, "llm_model": client.model,
              "intent": intent, "status": result.status, "gate": result.gate,
              "rounds": result.rounds, "transcript": result.transcript,
              "tokens_delta": client.tokens_used - tok0,
              "tokens_total": client.tokens_used}
    if result.status == "policy":
        record["policy"] = result.output.policy.model_dump()
        if execute:
            record["metrics"] = apply_policy(result.output.policy, registry)
    append_jsonl(runlog, record)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True)
    ap.add_argument("--provider", default="gemini", choices=sorted(PROVIDERS),
                    help="named preset (base_url/key/model/throttle); see providers.py")
    ap.add_argument("--llm-model", default=None, help="override the preset's model")
    ap.add_argument("--base-url", default=None, help="override the preset's endpoint")
    ap.add_argument("--api-key-env", default=None, help="override the preset's key variable")
    ap.add_argument("--no-reviewer", action="store_true")
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--loop", type=int, default=0, metavar="N",
                    help="Loop-B (staged): up to N metric-feedback iterations (0 = one-shot)")
    ap.add_argument("--min-interval", type=float, default=None,
                    help="seconds between network calls (default: preset value)")
    ap.add_argument("--runlog", default=str(DEFAULT_LOG))
    args = ap.parse_args(argv)
    if args.loop and args.no_execute:
        ap.error("--loop needs execution (it feeds measured metrics back); drop --no-execute")
    client = make_client(args.provider, model=args.llm_model,
                         min_interval_s=args.min_interval,
                         cache_dir=DEFAULT_ART / "intent_runs" / "cache")
    if args.base_url:
        client.base_url = args.base_url
    if args.api_key_env:
        client.api_key_env = args.api_key_env
    record = run_cli(args.intent, client, DEFAULT_ART, Path(args.runlog),
                     execute=not args.no_execute, use_reviewer=not args.no_reviewer,
                     loop_iterations=args.loop)
    print(json.dumps({k: v for k, v in record.items() if k != "transcript"}, indent=2, default=str))


if __name__ == "__main__":
    main()
