"""Small public CLI. Listing, validation and planning never contact a provider."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cps_reason_bench.config import ConfigError
from cps_reason_bench.graph_v2 import GRAPH_V2_FAMILIES, GRAPH_V2_LEVELS
from cps_reason_bench.public_config import BACKENDS, BUILTIN_SUITES, RunConfig, list_conditions, load_config
from cps_reason_bench.public_run import run_experiment, validate_config
from cps_reason_bench.versions import PACKAGE_VERSION


def _plan(config: RunConfig) -> dict:
    selected = list_conditions(config)
    return {
        "name": config.name, "package_version": PACKAGE_VERSION,
        "suite": config.suite.identity, "models": config.to_dict()["models"],
        "conditions": selected, "condition_count": len(selected),
        "repetitions": config.repetitions, "planned_episodes": config.episode_count,
        "simulator_seed_override": list(config.simulator_seeds),
        "output_dir": str(config.output_dir), "protocol": config.suite.protocol.to_dict(),
        "runtime": config.to_dict()["runtime"],
        "missing_credential_variables": sorted({m.api_key_env for m in config.models if m.api_key_env and not os.environ.get(m.api_key_env, "").strip()}),
        "notes": ["No provider requests or output writes in dry-run.",
            "Filtered selections may omit one member of a relational pair.",
            "No monetary estimate or monetary budget enforcement; max_episodes is a scale cap."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cps-reason", description="Graph-based CPS reasoning experiments")
    parser.add_argument("--version", action="version", version=PACKAGE_VERSION)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "dry-run", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("config", type=Path)
        if command == "validate":
            sub.add_argument("--offline", action="store_true", help="Do not require credentials (validation is always network-free)")
        if command == "run":
            sub.add_argument("--yes", action="store_true", help="Authorize the displayed cloud matrix without prompting")
    for command in ("list-providers", "list-benchmarks", "list-families", "list-graphs"):
        commands.add_parser(command)
    for command in ("list-models", "list-conditions"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True, help="List configured entries, not live provider availability")
    args = parser.parse_args(argv)
    try:
        if args.command == "list-providers":
            print("\n".join(BACKENDS))
        elif args.command == "list-benchmarks":
            print("\n".join(BUILTIN_SUITES))
        elif args.command == "list-families":
            print("\n".join(GRAPH_V2_FAMILIES))
        elif args.command == "list-graphs":
            print("Graph roles: i01 (local anchor), i02/i03 (relational pair), i04 (short nonlocal), i05 (multipath)")
            print("Authored profiles: " + ", ".join(GRAPH_V2_LEVELS))
            print("Graphs are generated within complete conditions; use list-conditions --config FILE for IDs.")
        else:
            config = load_config(args.config)
            if args.command == "list-models":
                print(json.dumps(config.to_dict()["models"], indent=2))
            elif args.command == "list-conditions":
                print(json.dumps(list_conditions(config), indent=2))
            elif args.command == "dry-run":
                print(json.dumps(_plan(config), indent=2))
            elif args.command == "validate":
                validate_config(config, check_credentials=not args.offline)
                print(f"Valid: {len(config.conditions)} conditions, {len(config.models)} models, {config.episode_count} episodes. No provider contacted.")
            else:
                print(f"Run: {config.name}\nSuite: {config.suite.name}@{config.suite.version}\nModels: {len(config.models)}\nConditions: {len(config.conditions)}\nRepetitions: {config.repetitions}\nPlanned episodes: {config.episode_count}\nOutput: {config.output_dir}", flush=True)
                for model in config.models:
                    print(f"  {model.name}: {model.backend}/{model.model}, normalization={model.response_normalization['mode']}", flush=True)
                if config.requires_confirmation and not args.yes:
                    print("Cloud calls may incur charges. No monetary budget is enforced.", flush=True)
                    try:
                        answer = input("Type RUN to authorize this matrix: ")
                    except EOFError:
                        answer = ""
                    if answer != "RUN":
                        print("Cancelled; no provider calls made.")
                        return 2
                result = run_experiment(config, authorize_cloud=args.yes or config.requires_confirmation,
                    progress=lambda message: print(message, flush=True))
                if result.failed_runs:
                    print("Stopped on infrastructure failure. Inspect failures.json and the episode summary; a request may have been billed.")
                    return 1
                print(f"Finished: {len(result.run_results)} episodes. Results: {result.output_dir}")
    except ConfigError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        print("Interrupted. Existing evidence is retained; in-flight requests may have been billed.")
        return 130
    return 0
