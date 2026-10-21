# CPS Reason Bench

CPS Reason Bench is a Python research environment for testing whether agents can
make valid, safe, task-completing decisions in small graph-based cyber-physical
systems. An agent observes public state, submits a structured decision, and sees
the consequences as the simulator advances in fixed steps. Logs distinguish
interface errors, rejected actions, infrastructure failures, safety outcomes, and
task success.

The software is a research prototype, **not a production controller**, a validated
physical plant model, or proof of general reasoning ability. Full-task success
includes authored behavioral gates as well as physical goals and safety.

## What Is Included

- Frozen Graph-v2 generators and 240 conditions: six families, eight authored
  profiles, five graph roles, including 48 matched relational pairs.
- Scripted and seeded-random controls; Ollama and OpenAI-compatible chat adapters.
- Strict decision validation and an explicitly selectable single-fence response
  normalization mode. Normalization is not semantic repair.
- Sequential experiment execution, per-episode traces, native summaries and
  aggregate exports. Optional plotting uses the existing reporting API.
- A small YAML interface, network-free validation/dry-run, and offline tests.

## Install

Python 3.10 or newer is required. From this repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cps-reason --help
```

Use `python -m pip install .` for a non-editable install, or
`python -m pip install -e '.[dev,plots]'` for tests and optional plots. Provider
adapters use the standard library; no provider SDK is needed.

## Five-Minute Local Setup

These first commands make **no network requests**, require no API key, and do not
need Ollama to be running:

```sh
cps-reason validate configs/examples/local_quickstart.yaml
cps-reason dry-run configs/examples/local_quickstart.yaml
cps-reason list-conditions --config configs/examples/local_quickstart.yaml
```

For an entirely offline simulator run, with scripted and random policies:

```sh
cps-reason run configs/examples/offline_smoke.yaml
```

For the LLM example, install Ollama separately, start its service, and explicitly
download the configured model if needed. Downloads consume disk and memory; this
package never downloads or starts models automatically.

```sh
ollama serve                 # only if your Ollama service is not already running
ollama pull qwen3.5:0.8b     # in a second terminal; review the download first
cps-reason run configs/examples/local_quickstart.yaml
```

Model names are configurable examples, not guarantees of current availability or
of fitting your hardware. Start with one episode and one worker (the only release
execution mode). See [provider setup](docs/PROVIDERS.md).

## Cloud Setup

The example uses an OpenRouter-compatible chat endpoint. Review the model, route,
privacy settings, and provider account limits before execution.

```sh
cps-reason validate configs/examples/cloud_quickstart.yaml --offline
cps-reason dry-run configs/examples/cloud_quickstart.yaml
read -rs OPENROUTER_API_KEY; export OPENROUTER_API_KEY; printf '\n'
cps-reason validate configs/examples/cloud_quickstart.yaml
cps-reason run configs/examples/cloud_quickstart.yaml
```

Enter the key at the `read` command, not in YAML. Real cloud execution displays the
episode count and requires `RUN`; `--yes` explicitly authorizes automation.
**There is no monetary budget enforcement or cost estimate in this release CLI.**
The scale cap is `runtime.max_episodes`. A timeout may still have been billed.

## Configuration and Selection

One file defines models, benchmark selection, repetitions, interface settings,
runtime limits, and output. References to model/scenario files are optional. All
relative paths, including output paths, resolve against the containing run YAML.

```yaml
config_version: 1
name: planning_pair
models:
  - name: scripted_reference
    backend: scripted
benchmark:
  suite: cps_reason_graph_v2
  families: [planning]
  roles: [i02, i03]
  profiles: [graph_v2_entry_1]
repetitions: 1
output_dir: runs/planning_pair
```

```sh
cps-reason list-benchmarks
cps-reason list-families
cps-reason list-graphs
cps-reason run configs/examples/selected_conditions.yaml
```

Roles (`i01`-`i05`) and profiles (`graph_v2_entry_1`-`graph_v2_core_4`) are distinct.
Filters intersect; `condition_ids` selects exact frozen IDs. Removing all filters
selects the full suite. The full local example plans 720 episodes and can take a
long time; it is not part of the quick start.

## Outputs and Reproducibility

Examples write to `runs/<example>/`. Each run retains resolved YAML, metadata,
condition snapshots, a manifest, per-episode JSONL traces and JSON summaries,
`episodes.csv`, `episodes.jsonl`, grouped summaries, and failure records. Existing
output directories are refused, never overwritten. There is no campaign resume
or automatic episode retry command in this release. See the
[user guide](docs/USER_GUIDE.md) before restarting an interrupted experiment.

The shipped suite and generator reproduce benchmark conditions. They do **not**
guarantee exact historical LLM outputs. The paper used a larger, separately
qualified model-interface panel and retained provenance with limitations. The
[paper-condition preset](configs/benchmark/paper_graph_v2.yaml) runs the scripted
reference over those conditions; it does not claim to replay that model campaign.

## Documentation

| Document | Purpose |
| --- | --- |
| [User guide](docs/USER_GUIDE.md) | Workflows, outputs, interpretation, extension points |
| [Configuration](docs/CONFIGURATION.md) | All release YAML fields and defaults |
| [Architecture](docs/ARCHITECTURE.md) | Execution, modules, public API, testing |
| [Providers](docs/PROVIDERS.md) | Local/cloud setup and interface limitations |
| [Graphs and scenarios](docs/GRAPHS_AND_SCENARIOS.md) | Frozen selection and complete custom scenarios |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Identities, seeds, paper protocol and limits |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Configuration, installation and runtime failures |
| [Release manifest](RELEASE_MANIFEST.md) | File integrity and provenance |
| [Release report](RELEASE_PREPARATION_REPORT.md) | Verification results and release boundaries |

## Tests, License, Citation

```sh
python -m pip install -e '.[dev,plots]'
python -m pytest
```

Normal tests use offline controls or mocked providers, never paid APIs. The
existing [MIT license](LICENSE) is preserved. Version `0.5.0` is retained from the
implementation; this candidate is not an announced publication or tag. Approved
author/citation, repository URL, and archival metadata are pending author choice
in [RELEASE_TODO.md](RELEASE_TODO.md); no DOI or public location is asserted.
