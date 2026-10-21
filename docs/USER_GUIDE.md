# User Guide

## Installation and First Run

Use Python 3.10+ and a fresh virtual environment. Required dependency: PyYAML.
Optional `dev` adds pytest/build; `plots` adds matplotlib and its dependencies.
Provider adapters require no SDK. Run from the standalone repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots]'
cps-reason --help
cps-reason validate configs/examples/offline_smoke.yaml
cps-reason dry-run configs/examples/offline_smoke.yaml
cps-reason run configs/examples/offline_smoke.yaml
```

The first run uses scripted/random controls, not an LLM. For a local model, follow
[Ollama setup](PROVIDERS.md), then validate/dry-run/run
`configs/examples/local_quickstart.yaml`. It explicitly selects `qwen3.5:0.8b`
and one operational smoke condition. Review RAM/disk requirements before downloading
weights. No software test automatically downloads a model.

## Configuration Anatomy

`config_version` identifies the new release schema. `name`/`description` label the
run. `models` contains inline configurations or references. `benchmark` chooses a
frozen suite or complete custom scenario files and optional filters. `repetitions`
and optional `simulator_seeds` define repeated episodes. `protocol` explicitly
selects existing prompt/schema/correction behavior; it normally stays inherited.
`output_dir` and `runtime` control files, scale and call pacing.

All relative paths resolve against the YAML file, including output paths. The
shipped examples are two directories below the repository root, so use
`../../runs/...`. For a run file at the repository root use `runs/...`. Move or
rename an experiment by editing that file, not Python. See the authoritative
[field reference](CONFIGURATION.md) for types/defaults and incompatible settings.

## Selecting Models

For Ollama, set `backend: ollama`, the installed model ID, local base URL and
decoding parameters. Add a second model mapping with a distinct configuration
name to compare two local models. Runs are sequential across the generated
condition/model matrix, with one active model call at a time. Different models can
require different memory, output formats and thinking controls.

For cloud, use the compatible chat adapter and an environment variable name:

```sh
cps-reason validate configs/examples/cloud_quickstart.yaml --offline
cps-reason dry-run configs/examples/cloud_quickstart.yaml
# Set OPENROUTER_API_KEY privately as described in PROVIDERS.md.
cps-reason validate configs/examples/cloud_quickstart.yaml
cps-reason run configs/examples/cloud_quickstart.yaml
```

`list-models --config FILE` lists configured models only; it never queries an
account or claims they are currently available. `mixed_models.yaml` combines one
Ollama and one cloud configuration. The entire matrix requires cloud authorization
before either model begins. `--yes` is an explicit noninteractive authorization.

## Selecting Conditions and Graphs

```sh
cps-reason list-benchmarks
cps-reason list-families
cps-reason list-graphs
cps-reason list-conditions --config configs/examples/selected_conditions.yaml
```

Use `suite: cps_reason_graph_v2` with no filters for all 240 conditions. Filters
intersect, so choose a valid family, profile and role combination. An unknown
identifier or empty intersection fails early. Examples of useful changes:

| Workflow | Configuration change |
| --- | --- |
| One local model, first five conditions | Use the local model and full suite with `benchmark.limit: 5`, repetitions 1. This is ordered, not random sampling. |
| Compare two local models | Two inline/model-file entries with distinct names and installed IDs; no cloud keys needed. |
| One operational family | `families: [planning]`: 40 conditions in the full suite before other filters. |
| One graph role | `roles: [i05]`: 48 conditions before other filters. |
| One authored profile | `profiles: [graph_v2_bridge]`: 30 conditions before other filters. |
| One relational pair | `families: [planning]`, `profiles: [graph_v2_entry_1]`, `roles: [i02, i03]`. |
| Exact authored graph/task | Select one or more actual `condition_ids` from the listing. |
| Complete local benchmark | `local_full_benchmark.yaml`: 240 x 3 = 720 episodes. Inspect dry-run first; potentially long. |
| New graph | `custom_graph.yaml`: complete two-node scenario file. Add further scenario paths for several custom graphs. |

Do not confuse roles with profiles. Do not swap a topology into a frozen condition
while retaining its identity. See [graphs/scenarios](GRAPHS_AND_SCENARIOS.md).

## Repetitions and Running Safely

Each repetition resets the same condition. Simulator seed overrides change the
experiment. The original expansion algorithm derives baseline seeds and, when
requested, model seeds; the suite can fix a model seed across repetitions. A seed
does not force deterministic hosted generation. Treat repeated episodes as
clustered within condition when doing inference, not independent graph samples.

`validate` performs offline feasibility/reference checks plus environment-variable
presence checks. `--offline` skips only credential presence. `dry-run` prints
configuration/model/condition identities, count, protocol and limits without
network or files. `run` validates again, prints progress, and writes native results.
Cloud execution needs confirmation; validation is not a live qualification pass.

The only scale guard is `runtime.max_episodes`; it does not limit money or total
inference calls. Set provider/account spending limits separately. The CLI does not
estimate a cost without trustworthy retained pricing. Each episode may request
multiple decisions and, if configured, output corrections or transport retries.

Ctrl-C interrupts execution and exports completed records plus an `interrupted`
metadata status. An in-flight HTTP call might still run or be billed. A process
kill/power loss can prevent final exports, though already written traces/summaries
remain. This release deliberately has **no resume/retry/overwrite command**: retain
the directory, inspect summaries and provider billing, and create an explicitly
selected new experiment in a new output directory. Do not blindly rerun a paid
matrix. Historical campaign recovery scripts are not part of this portable CLI.

## Output Layout

```text
runs/<name>/
  config.resolved.yaml       # all release settings, credential names only
  metadata.json              # versions, start/end, status, counts, identities
  manifest.json              # existing experiment plan, per-episode seeds/provenance
  condition_snapshots/       # selected complete scenarios keyed by content hash
  generated_scenarios/       # native runner's generated scenario inputs
  episodes/                 # native per-episode directories, JSONL traces/JSON summaries
  episodes.jsonl             # completed records for programmatic analysis
  episodes.csv               # flattened completed records
  aggregates.json/csv        # existing grouped summary statistics
  failures.json              # infrastructure failure records, not task failures
  ...                        # native family/cloud reports
```

Each episode records protocol/prompt/model/interface identity, seeds, timestamps,
decisions, action outcomes, world transitions and metrics. Raw responses and private
audit state may be sensitive even if credentials are redacted. Generated outputs
are ignored by git; review before public release. Infrastructure-failed episodes
are excluded from completed aggregate records and retained separately. A completed
episode can have task failure; `completed` does not mean successful reasoning.

The generic grouping includes model, family, condition and seed; do not average
unmatched partial datasets as if they covered the same matrix. `metadata.json`
and `failures.json` establish whether the requested matrix finished.

## Interpreting Results

- **Interface exposure:** at least one executable decision reached the action
  interface; distinct from merely valid JSON and from physical control. Inspect
  `protocol_compliance` counts/rates.
- **Accepted control:** a feasible plant-control command was accepted. The generic
  `accepted_actions` counter also includes accepted no-ops/inspection and other
  authored action types; it is not automatically the paper's filtered plant-control
  diagnostic. Use action IDs and outcomes when constructing that diagnostic.
- **Safe-goal success:** goal attainment with the configured safety requirement.
  For frozen Graph-v2 the allowed safety violations are zero. Evaluate the actual
  goal/safety fields together, not `task_success` alone.
- **Full-task success:** safe-goal attainment AND active authored behavioral gates.
  Graph-v2 exact accepted sequence agreement and family-specific requirements can
  reject a physically adequate alternative. This is an operational benchmark score,
  not a proof of or exhaustive test for reasoning.
- **Trajectory diagnostics:** reveal state/action timing and graph dependence but
  do not by themselves establish the cause of a decision or reasoning mechanism.
- **Recurrence:** repeatability within the same condition, such as success in 0/3,
  1/3, 2/3 or 3/3 runs. Reconstruct from completed records with explicit denominators;
  generic plots do not reproduce the paper's dedicated recurrence analysis.

Optional plots, without new inference:

```python
from cps_reason_bench.plotting import write_result_plots
write_result_plots(["runs/offline_smoke/episodes.jsonl"], "runs/offline_smoke/plots")
```

These are generic summaries, not the paper-specific paired/clustered analysis.

## Extension Points

New model at an existing backend: add YAML, validate offline, then separately
authorize a small qualification run. New graph/task: complete scenario YAML with
feasible reference and consistent goals/actions. New suite/family/backend: advanced
implementation work, not configuration alone. The backend registry supports Python
factories; a new family needs evaluator/invariant/reference tests and a versioned
definition. Preserve public/private information boundaries. See
[Architecture](ARCHITECTURE.md) for the public API and tests and
[Troubleshooting](TROUBLESHOOTING.md) for failures.
