# Architecture and Public API

## Execution

```text
release YAML -> RunConfig -> suite materialization -> condition selection
                                      |
                    existing experiment/seed expansion
                                      |
               public observation + allowed context
                                      |
                  policy / versioned prompt / backend
                                      |
             response envelope normalization (if selected)
                                      |
                strict decision/schema validation
                                      |
           command feasibility / resources / acceptance
                                      |
         callbacks + subscriptions + accepted plant actions
                                      |
          fixed-step world advance -> next observation
                                      |
               evaluator -> JSONL / summaries / exports
```

`public_run` delegates episodes to `experiments.run_experiment_episode`, which
constructs the same `BenchmarkRunner`, policy and simulator. No alternative
simulator, action parser or scorer was introduced. The only retained runtime API
refactor accepts pre-materialized suites/models in experiment expansion, and
optional pre-collected provenance in workspace preparation. Old file-based callers
retain their existing defaults; the new wrapper shares their identity/seed loop.

## Module Map

| Module(s), under `src/cps_reason_bench/` | Responsibility |
| --- | --- |
| `public_config.py` | Strict release schema, model references, offline materialization/filtering, safe resolved config. |
| `public_cli.py`, `__main__.py` | Listing, validation, dry-run, confirmation and execution. No historical campaign commands. |
| `public_run.py`, `experiments.py` | Matrix expansion, seeds/identities, workspace, per-episode execution, stop/partial export. |
| `config.py`, `suites.py`, `generators.py`, `graph_v2.py`, `design.py` | Existing scenario/suite schemas and generation. Frozen conditions ship under package `data/`. |
| `simulator/graph.py` | World state, public observation, feasibility, actions, resources, coupling, faults, delays, safety/goals. |
| `protocol.py`, `response_normalization.py`, `versions.py` | Decision schema/parser, one-pass envelope modes, exact prompt/version identities. |
| `models/base.py`, `models/config.py`, `models/registry.py` | Backend/policy contracts and factories. |
| `models/ollama.py`, `models/openai_compatible.py`, `models/security.py` | Existing transport, usage/error classification and redaction. |
| `models/catalog.py`, `models/providers.py` | Retained advanced registry/presets; not the new public YAML schema or live discovery. |
| `policies/` | Scripted/random/LLM policies and public-information control implementation/adapter. |
| `context.py`, `scheduling.py`, `subscriptions.py`, `public_events.py` | Context disclosure, callbacks, subscriptions and sanitized events. |
| `metrics.py`, `validation.py`, `invariants.py`, `references.py`, `graph_v2_validation.py` | Scoring, reference replay, feasibility/family and graph coverage checks. |
| `run_logging.py`, `provenance.py`, `aggregation.py`, `family_reports.py`, `cloud_reporting.py` | Evidence, provenance and native exports. |
| `plotting.py`, `pilot_analysis.py`, `catalog.py` | Retained optional generic analysis, plots and advanced catalog functions. Not the full paper-analysis pipeline. |

## Time and State Boundaries

Simulated time is `step * time_step`, rounded by the existing engine. A model's
wall-clock inference latency does not advance the plant. At a physical advance,
the engine increments step/time, activates due faults, applies due action effects,
applies drift and coupling, clamps variables, records sensor history, and checks
safety/goals. Action batches have their authored sequential-per-command or atomic
semantics. These are discrete evaluation checkpoints, not continuous safety proofs.

Logs can contain private world state for auditing, but the policy receives only
the permitted observation/context. Inspection, delayed sensing, rule exposure and
public event filtering retain their existing boundaries. The reference controller
can encode privileged solution knowledge; it is a feasibility check, not an LLM
competitor under equal information.

Accepted command IDs form the scored action history. Rejected commands do not
enter that accepted sequence, but can still affect diagnostic/family checks.
Named no-op actions do count as accepted action IDs. Callbacks/subscriptions are
separate from plant-action history. Full-task scoring for Graph-v2 can require
exact accepted sequence agreement; a different safe plan need not pass this
authored criterion.

## Public Python API

```python
from cps_reason_bench import (
    load_config, instantiate_benchmark, list_conditions,
    plan_experiment, validate_config, run_experiment, load_results,
)

config = load_config("configs/examples/offline_smoke.yaml")
conditions = instantiate_benchmark(config)   # existing SuiteScenarioInstance tuple
print(list_conditions(config))              # JSON-friendly IDs and hashes
specs = plan_experiment(config)             # existing ExperimentRunSpec tuple
validate_config(config)                     # offline reference/semantic checks
result = run_experiment(config)             # no cloud in this example
rows = load_results(result.output_dir)      # native completed episode records
```

`load_config(path, check_credentials=False)` returns `RunConfig` with independent
`to_dict()`, `episode_count`, and selected models/suite/conditions. It is a frozen
dataclass containing mutable nested mappings: treat them as read-only. Edit YAML
and reload instead of mutating a loaded experiment.

`validate_config(config, check_credentials=True)` is always network-free.
`run_experiment(config, authorize_cloud=False, progress=None)` requires explicit
cloud authorization even in Python, returns the existing `ExperimentResult`, and
raises on unexpected failures. `progress` receives text messages. Infrastructure
failures return `failed_runs` and partial exports rather than a task-success score.

Existing top-level `ScenarioConfig`, `load_scenario`, `BenchmarkRunner`, `RunResult`
remain available for lower-level use. In particular, directly using a runner is an
advanced API: it does not add the release CLI's cloud confirmation or scale cap.
Internal modules are retained for compatibility, not all promised as a stable
high-level API. The release interface itself is version 1, separate from package
version 0.5.0.

## Test Structure

Normal tests block real socket connections. Legacy providers use injected mocks;
the scripted/random simulator checks are offline. There is no paid integration
test enabled by default, and no weights are downloaded.

| Tests | Main checks |
| --- | --- |
| `test_public_release.py` | Strict YAML, malformed options, env names/no key leakage, all examples, filters/counts, no-network dry-run, CLI help, cloud gate, partial writes, legacy expansion equivalence, all-240 original condition hashes. |
| `test_graph_v2_suite.py`, `test_graph_v2_engine.py` | Matrix/coverage, stable generation across hash seeds, all 240 feasible scripted references, 48 pair manipulations and graph-engine behavior. |
| `test_config_and_simulator.py`, `test_time_semantics.py` | Schema, feasibility, world evolution, fixed time, wall-clock stalls, faults, atomicity and interruption/error boundaries. |
| `test_protocol_and_interactions.py`, `test_reference_feedback_and_retries.py`, `test_response_normalization.py` | Decision validity, callbacks, public context, strict versus fence-normalized interfaces, correction rules and evidence hashes. |
| `test_backends_and_versions.py`, `test_openai_cloud_backend.py` | Mocked payloads, provider errors, route checks, seeds, bounded retries and versions. |
| Remaining retained modules | Runner traces, adaptation/temporal tasks, suite generation, foundation regression, aggregation/provenance, optional plots and end-to-end public/private boundary. |

Examples:

```sh
python -m pytest tests/test_public_release.py -q
python -m pytest tests/test_graph_v2_suite.py -q
python -m pytest tests/test_response_normalization.py -q
```

`scripts/check_release.py` checks hashes, links, importable modules, source paths,
unexpected files and security patterns. A golden fixture records definitions from
the unmodified development generator. Tests do not claim to reproduce model
outputs. Historical launcher/recovery/add-on tests are intentionally excluded with
their omitted features; see the machine-readable release manifest and report.
