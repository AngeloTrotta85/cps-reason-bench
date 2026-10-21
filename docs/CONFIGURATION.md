# Configuration Reference

This documents the release run schema, `config_version: 1`, implemented in
`public_config.py`. YAML is loaded safely; duplicate/unknown release keys, wrong
scalar types, empty selections and incompatible combinations are errors. JSON is
also accepted as YAML-compatible input. No interpolation, `.env` autoloading,
Python imports, or command execution is performed by this schema.

In the tables, **required** means no default. Paths are absolute or relative to
the file containing them: run-relative output/scenarios/model references,
model-file-relative model content. Defaults are inserted in resolved output.

## Run Fields

| Key | Type; requirement/default | Values/example and effect |
| --- | --- | --- |
| `config_version` | integer; optional `1` | Only `1`; release wrapper schema, not scenario/suite version. |
| `name` | string; required | `planning_pair`; starts alphanumeric, then alphanumeric, `_`, `-`, `.`. Part of episode/derived seed identity. |
| `description` | string; optional empty | Human-readable metadata, not an agent instruction. Do not put secrets here. |
| `models` | nonempty list of mappings; required | Inline models or `{path: ../models/qwen3_5_0_8b.yaml}`; unique configuration names. |
| `benchmark` | mapping; required | Exactly one source, `suite` or `scenarios`, plus optional filters below. |
| `repetitions` | positive integer; optional `1` | Fresh episodes per condition/model/seed, not new independent conditions. Explicitly overrides suite recommendation metadata. |
| `simulator_seeds` | list of distinct nonnegative integers; optional `[]` | Empty preserves each condition's frozen seeds. `[11, 12]` replaces them, creating a seed variant. |
| `protocol` | mapping; optional suite protocol | Explicit prompt/schema/validation/retry overrides. Any actual change changes the suite content identity. |
| `output_dir` | nonempty path string; optional `runs/<name>` | Relative to run YAML. Must not already exist when running. |
| `runtime` | mapping; optional `{}` | Operational limits, never physical horizon/scoring overrides. |
| `runtime.max_episodes` | positive integer; optional `1000` | Rejects a matrix above this scale cap. Not a monetary or per-episode call cap. |
| `runtime.model_call_delay_seconds` | finite nonnegative number; optional `0` | Wall-clock delay before model calls. Does not advance simulated time. |

Episode count is `models * repetitions * sum(selected simulator-seed counts)`.
No concurrency, resume, overwrite, budget currency or automatic episode-retry
field is accepted. Fresh output protects historical evidence.

## Benchmark Selection

| Key | Type; requirement/default | Values/example and effect |
| --- | --- | --- |
| `suite` | string; required unless `scenarios` | `cps_reason_graph_v2` (240), `cps_reason_graph_v2_pilot` (9), `cps_reason_graph_v2_local_smoke` (1), `cps_reason_graph_v2_cloud_qualification` (3). Version 1.0.0; files unchanged. |
| `scenarios` | nonempty list of path strings; alternative to `suite` | `[../graphs/two_node_transfer.yaml]`; complete scenario YAML files, distinct `name` values become condition IDs. Several paths select several authored graphs/tasks. |
| `families` | list of distinct strings; optional `[]` | All if empty. `prospective_reasoning`, `planning`, `interference`, `conditional_safety`, `adaptation`, `temporal_tracking`. Must occur in selected source. |
| `roles` | list of distinct strings; optional `[]` | `i01`, `i02`, `i03`, `i04`, `i05`; not profile labels. Custom scenarios without Graph-v2 role metadata cannot use these filters. |
| `profiles` | list of distinct strings; optional `[]` | `graph_v2_entry_1`, `graph_v2_entry_2`, `graph_v2_entry_3`, `graph_v2_bridge`, `graph_v2_core_1`, `graph_v2_core_2`, `graph_v2_core_3`, `graph_v2_core_4`. Custom source can use its authored level, e.g. `custom`. |
| `condition_ids` | list of distinct strings; optional `[]` | Exact IDs from `list-conditions`, e.g. `graph_v2_planning_graph_v2_entry_1_i02`. |
| `limit` | positive integer; optional no limit | First N remaining conditions in suite order, after intersecting every filter. Not random sampling. |

Filters never rewrite a condition. Selecting a role/limit can break relational
pairs; do not report paired comparisons unless both members are present.

## Model Fields

An element of `models` can instead contain **only** `path` (required string for
that reference), pointing to a single model mapping with the fields below.

| Key | Type; requirement/default | Values/example and effect |
| --- | --- | --- |
| `model_config_schema_version` | integer; optional `1` | Existing model schema; only `1`. |
| `name` | nonempty string; required | `qwen_local`; configuration identity, distinct from service model ID. |
| `backend` | string; required | `scripted`, `random`, `ollama`, `openai-compatible` (alias `openai_compatible`). |
| `model` | nonempty string; optional backend name | Use the actual installed/service ID, e.g. `qwen3.5:0.8b`. Never checks availability during validation. |
| `base_url` | URL string; required for compatible API, Ollama default `http://localhost:11434` | HTTP(S), hostname required; no embedded credentials, query, fragment or whitespace. Compatible API requires HTTPS except loopback. Unsupported for baselines. |
| `api_key_env` | environment variable name; required for compatible API | E.g. `OPENROUTER_API_KEY`. Values never enter config. Unsupported for Ollama/baselines. For unauthenticated local service use the native Ollama adapter or advanced backend API. |
| `timeout_seconds` | positive finite number; optional `120` | Per-request wall-clock timeout, not simulated time. |
| `decoding` | mapping; optional `{}` | Fields below; actual provider support can differ. |
| `backend_parameters` | mapping; optional `{}` | Backend-specific transport/interface controls below. Empty for baselines. |
| `response_normalization` | mapping; optional `{mode: none}` | Only child `mode`, string `none` or `markdown_single_fence`. One strict whole-response fence removal, no semantic repair; changes configuration identity. |

Legacy model-file `backend_import`, `capability_overrides`, model-specific retry
and arbitrary headers are deliberately not exposed in public YAML. Advanced
`ModelConfig`/backend registry APIs remain available. This prevents a YAML file
from importing arbitrary Python or concealing credentials in headers. Set output
corrections only through the explicit run `protocol`.

### Decoding

| Key | Type; requirement/default | Effect/example |
| --- | --- | --- |
| `temperature` | finite number >=0; optional `0` | Sampling temperature, provider-specific accepted range. |
| `top_p` | finite number >0 and <=1; optional `1` | Nucleus-sampling parameter. |
| `max_tokens` | positive integer; optional `768` | Output cap; Ollama sends `num_predict`; compatible API uses configured token field. |
| `seed` | nonnegative integer, `null`, or `derived`; optional `null` | No requested model seed by default. `derived` uses the existing experiment identity rules; provider honoring is not guaranteed. |
| `extra` | mapping; optional `{}` | Ollama only: optional positive integers `num_ctx` (context budget) and `num_thread` (CPU threads). No other keys accepted; cannot shadow sampling/seed controls. |

Baselines accept decoding metadata for compatibility; they do not perform LLM
sampling. The random policy uses the separately derived baseline seed. Editing
metadata that affects model identity can also affect derived baseline seeds.

### Ollama Parameters

| Key | Type; requirement/default | Effect/example |
| --- | --- | --- |
| `think` | boolean; optional `false` | Existing adapter's thinking toggle; supported models only. |
| `structured_output_mode` | string; optional `json_schema` | `json_schema`, `json_object`, `prompt_only`; controls native format request. Paper-like examples explicitly use `prompt_only`, not this default. |

### OpenAI-Compatible Parameters

| Key | Type; requirement/default | Effect/example |
| --- | --- | --- |
| `provider` | string; optional `custom` | `custom`, `openai`, `openrouter`; selects adapter transport behavior, not a new SDK. |
| `structured_output_mode` | string; optional `prompt_only` | `prompt_only`, `json_object`, `json_schema`; provider must support requested mode. |
| `routing` | mapping; optional `{}` | Direct for custom/openai; dynamic for OpenRouter unless explicitly benchmark mode. Detailed fields below. |
| `request_parameters` | mapping; optional `{}` | Include/omit specific outbound fields; detailed fields below. |
| `transport_retry` | mapping; optional one attempt | Bounded retry of classified transport failures only, distinct from output correction. |

`routing` fields are optional:

| Key | Type; default | Effect/example |
| --- | --- | --- |
| `mode` | string; `direct` except OpenRouter `dynamic` | `direct`, `dynamic`, `benchmark`; non-OpenRouter supports only direct. |
| `provider_order`, `only` | lists of strings; `[]` | OpenRouter preferences/allowlist, e.g. provider names verified for your account. Do not guess current route availability. |
| `allow_fallbacks` | boolean; unspecified, benchmark `false` | Benchmark mode forbids true. |
| `require_parameters` | boolean; unspecified, benchmark `true` | Benchmark mode forbids false. |
| `quantizations` | list of strings; `[]` | OpenRouter route filter, server-supported values. |
| `data_collection` | string; unspecified | `allow` or `deny`; requested routing policy, not a release privacy certification. |
| `zdr` | boolean; unspecified | Requested zero-data-retention routing, subject to availability. |
| `expected_provider`, `expected_route`, `expected_canonical_model` | nonempty strings; unspecified | Explicit returned-identity expectations, e.g. an approved route ID; mismatch is infrastructure failure. |

All routing fields other than `mode: direct` require `provider: openrouter`.

`request_parameters` fields are optional:

| Key | Type; default | Effect/example |
| --- | --- | --- |
| `temperature`, `top_p`, `seed` | booleans; `true` | Include supported parameters; false omits rather than renames them. |
| `max_tokens_field` | string; `max_tokens` | `max_tokens` or `max_completion_tokens`. |
| `reasoning_effort` | string or null; `null` | `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; adapter-supported values, not a guarantee for every model. |
| `reasoning_exclude` | boolean; `true` | OpenRouter reasoning exclusion control when reasoning is requested. |

`transport_retry` fields are optional:

| Key | Type; default | Effect/example |
| --- | --- | --- |
| `max_attempts` | integer 1-10; `1` | Includes initial attempt. Release default differs from the advanced adapter's historical default of 3. |
| `base_backoff_seconds` | finite number >=0; `1` | Backoff base. |
| `max_backoff_seconds` | finite number >=0; `30` | Backoff cap. |
| `jitter_seconds` | finite number >=0; `0.25` | Added transport retry jitter. |

Timeout/ambiguous failures are not automatically retried. These settings do not
authorize retrying whole episodes or guarantee a total wall-clock deadline.

## Protocol

All fields optional; omitted nested fields retain the suite protocol. Graph-v2
defaults below. Changes are experimental variants and have a changed content hash.

| Key | Type; default | Allowed values/effect |
| --- | --- | --- |
| `prompt.id` | string; `cps-reason-json-control` | Existing prompt ID only. |
| `prompt.version` | string; `2.0.0` | `1.0.0` or `2.0.0`; quoted strings, exact registered template. |
| `schema.id` | string; `cps-reason-decision` | Existing decision schema only. |
| `schema.version` | string; `1.0.0` | Only `1.0.0`. |
| `output_validation_policy` | string; `strict` | Only `strict`; no tolerant parsing option. |
| `retry_policy.name` | string; `strict_zero_correction` | Descriptive label, e.g. `correction_assisted`; retain a truthful label. |
| `retry_policy.max_corrections` | nonnegative integer; `0` | Extra output-correction requests; incompatible with non-none response normalization. Zero for paper protocol. |

No public setting changes a frozen condition's physical horizon, context,
observability, actions, graph or scoring. For a new condition, author a complete
scenario as described in [Graphs and scenarios](GRAPHS_AND_SCENARIOS.md).

## Validation and Credentials

`load_config`/`dry-run` check syntax, field values, selections and backend
construction with no network. `validate` additionally runs offline semantic and
scripted-reference checks and checks credential presence. `validate --offline`
omits only the credential requirement. Real execution repeats validation before
any request. Presence is not proof that a key/model/route works. Unknown programming
errors propagate; they are not converted to successful results.
