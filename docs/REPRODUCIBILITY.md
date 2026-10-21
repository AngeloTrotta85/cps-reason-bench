# Reproducibility

## What Is Frozen and Shipped

The core physical/scoring modules and Graph-v2 generator are retained unchanged.
The complete suite remains `cps_reason_graph_v2@1.0.0`, package `0.5.0`, generator
`cps_reason_graph_v2_generator@1.0.0`. Its canonical JSON content hash is
`ac1139228d2543ef6b8c58263c63df47158fc7a90a39cba6ce1d62c660f0a89d`.

`artifacts/graph_v2_conditions.json` records all 240 condition IDs, scenario hashes,
generation and simulator seeds from the unmodified development generator. This
was materialization only, not a new LLM campaign. Tests check release definitions
against it and exercise all 240 scripted references and 48 relational pairs.
`benchmark/graph_v2_coverage_report.json` and the construct-validity report are
compact pre-existing offline definition evidence, not raw model generations.

The new schema is a release wrapper. It was **not** the command interface that
produced the historical paper campaign. Its public release defaults (including
cloud one-attempt transport and no discovery) are not retroactive campaign claims.

## Paper Preset

```sh
cps-reason validate configs/benchmark/paper_graph_v2.yaml --offline
cps-reason dry-run configs/benchmark/paper_graph_v2.yaml
# Explicitly run only when desired:
cps-reason run configs/benchmark/paper_graph_v2.yaml
```

This reproduces the paper's **condition definitions, prompt 2.0.0, decision schema
1.0.0 and strict zero-correction configuration**, using one scripted-reference
execution per condition. It does not reconstruct a whole historical cloud panel.
The full local example illustrates three repetitions, but does not claim identical
server state, model binary or historical outputs.

The paper panel had 12 model-interface configurations x 240 x 3 = 8,640 LLM
episodes; scripted reference 240 and random control 720 made 9,600 in that main
matrix. PICC was a separate 240-condition control. Strict and normalized Claude
conditions were different interfaces, not interchangeable records. The suite's
default metadata recommends five repetitions; the campaign explicitly selected
three. The release preserves suite bytes and makes repetitions explicit.

## What Cannot Be Promised

Exact generations depend on model weights/digest, quantization, serving version,
hardware, route, provider implementation, decoding and sometimes nondeterminism
even with fixed seeds. The manuscript's retained-provenance limitations remain.
Do not invent absent details, infer an identical backend from a model name, or
describe a newly available hosted alias as the historical deployment.

The release does not include raw campaign outputs, archival adoption history,
account discovery, historical launcher budgets, Astra-specific add-on machinery,
or the dedicated paper analysis/report pipeline. Reproducing paper numerical
results requires an independently reviewed artifact release with those needed
records. Reproducing the benchmark protocol is a different claim from reproducing
exact historical outputs or re-running the analysis.

## Identity and Seeds

Episode identity uses experiment name, suite identity, entry ID, full scenario
content hash, model configuration identity, simulator seed and repetition index.
The same legacy algorithm derives baseline and optional requested model seeds.
Graph-v2's `fixed_per_model_condition` rule retains requested model seed across
repetitions; baseline seeds still follow their existing rule. Renaming an
experiment/model or changing protocol affects identity and may change derived seeds.
`simulator_seeds: []` retains the authored seed. Overrides are explicit variants.

Filters select existing conditions without modifying the source suite hash.
Actual selections and complete condition snapshots are recorded separately.
Protocol overrides produce a different content hash; custom scenario collections
use a separate content identity. Repetitions and operational limits are retained
in resolved run metadata. `release_manifest.json` is a **file SHA-256** manifest;
these byte hashes are distinct from canonical JSON semantic content hashes.

## Retaining a New Experiment

Preserve resolved YAML, native manifest, condition snapshots, episode summaries
and traces, completed/failure counts, package/wheel hash, dependency versions and
reviewed hardware/provider metadata. The release wrapper records runtime metadata
and existing backend response metadata but deliberately does not query installed
model inventories or account endpoints. `ollama.collection_status: not_probed`
is explicit: collect model binary/server evidence separately if required.

Never publish keys or unrestricted provider/account logs. Generated outputs can
contain local paths and private simulator state; review/redact before archival.
The release file manifest is independent of the historical paper artifact manifest,
which has not been regenerated or replaced.
