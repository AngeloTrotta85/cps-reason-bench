# Release Audit

Date: 2026-10-21. Prepared before creating the separate `git/` release tree.
This is a source audit, not an experiment or a publication of campaign records.
The existing development tree remains authoritative; no simulator, scoring,
paper, result or historical artifact manifest is to be edited there.

## Current Architecture

The package is `cps_reason_bench`, version 0.5.0, Python >=3.10, with PyYAML
as its runtime dependency. Plotting additionally uses matplotlib. Both Ollama
and OpenAI-compatible Chat Completions use the Python standard-library HTTP
client; neither requires a provider SDK. The existing LICENSE and packaging
explicitly declare MIT. No new author identity or public release tag is inferred.

Actual execution path:

1. `config.py`, `suites.py`, `models/config.py` and `experiments.py` load YAML/JSON
   scenario, suite, model and experiment documents. `models/catalog.py` adds a
   research-campaign registry with routing, discovery and pricing metadata.
2. `suites.materialize_suite` invokes a declared generator or loads complete
   scenario files. Graph-v2 uses `graph_v2.generate_graph_v2_scenario` at generator
   1.0.0. The complete suite is `cps_reason_graph_v2@1.0.0` with 240 conditions:
   six families, eight authored profiles and roles i01--i05. Profiles are not
   graph roles or a pure node-count difficulty scale.
3. `experiments.expand_experiment` combines conditions, configured models,
   simulator seeds and repetitions. Episode identities and derived baseline/
   model seeds depend on the existing content-hash rules. Repetitions reset a
   condition rather than creating new conditions.
4. `models.config.build_policy` constructs scripted/random controls or an
   `LLMPolicy` through `models.registry`. Native Ollama and OpenAI-compatible
   adapters implement `ModelBackend`. Provider metadata is not model reasoning.
5. `runner.BenchmarkRunner` constructs the graph simulator, context manager,
   callback scheduler and subscription manager. Public observations and allowed
   history enter a fresh versioned prompt, not arbitrary private simulator state.
6. `policies/llm.py` builds messages and requests inference. Optional
   `markdown_single_fence` normalization changes only the envelope; the strict
   parser, schema and decision checks remain authoritative. Correction retries
   and transport retries are distinct mechanisms.
7. `protocol.py` validates the decision. The simulator checks prerequisites,
   resources and action feasibility before accepting commands. Rejected commands
   are distinct from accepted commands and from later state changes.
8. `simulator/graph.py` executes accepted effects and advances discrete simulated
   time: scheduled effects, drift, coupling, faults and sensor/public projections.
   Callback/subscription/context operations are not accepted plant-action IDs.
9. `metrics.py` computes outcome layers and family gates. For Graph-v2, full-task
   success requires safe-goal attainment, exact accepted-action sequence agreement
   and active family requirements. This is deliberately narrower than every
   useful controller. Safety is checked at discrete checkpoints.
10. `run_logging.py` writes JSONL events and JSON summaries; `aggregation.py`,
    `family_reports.py`, `cloud_reporting.py` export CSV/JSON summaries.
    `plotting.py` provides optional generic plots. Later graph campaign/report
    modules combine frozen/adopted runs and paper-specific analyses.

The current CLI imports many campaign-specific helpers. Research launchers encode
fixed model panels, qualification, budget freezes, adoption from old results,
recovery, parallel execution and add-ons. They are not the necessary core for a
new researcher's configurable experiment. The release should expose a small new
CLI while retaining the underlying loader/runner/provider APIs.

The complete suite's metadata recommends five repetitions, but the paper campaign
explicitly used three model/random repetitions and one scripted-reference
execution. The wrapper must preserve the suite bytes and make its explicit run
repetition count authoritative, not rewrite this historical distinction.

## Code Categories and Release Decisions

Inventory inspected all major directories, source-module imports, core execution
and scoring paths, provider constructors, suite data, tests, launchers and paper
methods/provenance. Approximate sizes are observations at audit time, not release
limits. No directory is copied wholesale merely because it exists.

| Major location | Category | Decision | Reason |
|---|---|---|---|
| `src/cps_reason_bench/` core loader, runner, simulator, metrics, protocol, context, scheduling, subscriptions, references, types, invariants | CORE_RUNTIME | INCLUDE | Existing scientific runtime; preserve code and test it, not rewrite it. |
| `src/cps_reason_bench/graph_v2.py`, `graph_v2_validation.py`, generators/design/suites | BENCHMARK_DEFINITION | INCLUDE | Frozen generation semantics and identifiers; older generators remain required by suite APIs/tests. |
| `src/cps_reason_bench/models/` adapters, config, registry, security | PROVIDER_INTEGRATION | INCLUDE_AFTER_CLEANUP | Native and compatible HTTP adapters; expose actual fields, prohibit credentials in config. No account discovery needed for dry-run. |
| `src/cps_reason_bench/cli.py`, model/campaign commands | CLI, HISTORICAL | EXCLUDE | Current CLI is coupled to campaign inventories and archival paths. Replace only the release-facing entry point with a thin wrapper, not the runtime. |
| `src/cps_reason_bench/*campaign*`, freeze/reuse/qualification/add-on/discovery modules | HISTORICAL, PRIVATE/LOCAL | ARCHIVE_NOT_PUBLIC | Keep in development; fixed historical campaign/recovery machinery is not a portable public workflow. |
| Generic experiment, aggregation, family/cloud reporting and plotting modules | CORE_RUNTIME | INCLUDE | Reuse native episode/export formats and optional plots. Avoid duplicate result formats. |
| PICC policy and public-input adapter | CORE_RUNTIME | INCLUDE | Useful public-information control implementation; preserve without claiming a new historical execution. |
| PICC frozen experiment/analysis launcher | REPRODUCIBILITY_ARTIFACT, PAPER_ONLY | ARCHIVE_NOT_PUBLIC | Depends on frozen inventory and paper audit files; not a generic standalone runner. |
| `benchmark/suites/` | BENCHMARK_DEFINITION | INCLUDE | Complete, pilot, local smoke and qualification suites, unchanged. |
| `benchmark/` other JSON/CSV/model freezes (~15 MiB) | REPRODUCIBILITY_ARTIFACT, GENERATED_RESULT, PRIVATE/LOCAL | INCLUDE_AFTER_CLEANUP | Include only compact public-safe condition/hash evidence actually needed; do not copy discovery/account/inventory/budget or failed-run records. |
| `examples/` (~0.5 MiB) | BENCHMARK_DEFINITION, HISTORICAL | INCLUDE_AFTER_CLEANUP | Retain scenario templates and legacy suite/model fixtures used by runtime/tests; author separate beginner examples. Exclude old campaign experiment matrices. |
| `tests/` | TEST | INCLUDE_AFTER_CLEANUP | Retain offline runtime/provider/generation/regression tests whose dependencies ship; add release config/CLI/isolation tests. Exclude campaign-only tests with omitted features, with explicit inventory. |
| Root shell launchers, `local_experiments.yaml`, `models.yaml`, comprehensive design | CLI, HISTORICAL, PRIVATE/LOCAL | ARCHIVE_NOT_PUBLIC | Encode research campaign choices and paths; not the beginner public interface. |
| `scripts/` | PAPER_ONLY, HISTORICAL | EXCLUDE | Analysis/report building, frozen-campaign auditing and machine-specific Ollama tuning. Add small release verification scripts instead. |
| `extensions/graph_v2_astra/` | PROVIDER_INTEGRATION, HISTORICAL | ARCHIVE_NOT_PUBLIC | Historical add-on with experiment-specific backend/analysis; do not imply generic adapter can recreate this route exactly. |
| `experiments/` | HISTORICAL, REPRODUCIBILITY_ARTIFACT | ARCHIVE_NOT_PUBLIC | Pilot configurations and separate PICC execution record. Preserve originals in development. |
| `runs/` (~4.1 GiB, >41,000 files) | GENERATED_RESULT, REPRODUCIBILITY_ARTIFACT, PRIVATE/LOCAL | ARCHIVE_NOT_PUBLIC | Raw prompts/responses, failures, provider records, adopted runs and histories require a separate reviewed artifact release. No raw campaign records copied. |
| `analysis/` (~1 GiB), `figures/`, `tables/` (~144 MiB) | GENERATED_RESULT, PAPER_ONLY | ARCHIVE_NOT_PUBLIC | Paper analysis/results are not required to install or execute the benchmark. No recomputation or mutation. |
| `paper/` (~912 MiB) | PAPER_ONLY, REPRODUCIBILITY_ARTIFACT, PRIVATE/LOCAL | ARCHIVE_NOT_PUBLIC | Current/obsolete drafts, reviews, literature PDFs and supplements are not software package contents. Current paper is read-only consistency evidence. |
| `docs/` (~69 MiB) | PAPER_ONLY, HISTORICAL, UNKNOWN | INCLUDE_AFTER_CLEANUP | Write focused public manuals from inspected code; do not export internal reviews, checkpoints or unapproved citation metadata. No unexplained UNKNOWN content enters release. |
| `ARTIFACTS.md`, `artifact_manifest.json` | REPRODUCIBILITY_ARTIFACT | ARCHIVE_NOT_PUBLIC | Paper manifest refresh remains deferred. Release gets a separate manifest; original stays untouched. |
| `pyproject.toml`, `LICENSE`, README, changelog | CORE_RUNTIME, CLI | INCLUDE_AFTER_CLEANUP | Keep package version/license; replace historical packaging and very large research README with standalone install/data metadata and public guide. |
| `build/`, egg-info, bytecode, test caches | CACHE | EXCLUDE | Regenerable machine outputs. |
| `.git/`, `.codex/`, `.agents/`, `.aws/`, environment/credential files | PRIVATE/LOCAL | EXCLUDE | Never copy or initialize git metadata, user tooling or secrets. |
| `tmpp8gu_r9i/` | CACHE, GENERATED_RESULT | EXCLUDE | Temporary interrupted/debug execution. |
| `git/` | Release destination | EXCLUDE_FROM_SOURCE_INVENTORY_AND_COPY | Explicit recursion guard. It becomes a standalone root, not a nested copy of itself. |

## Release Design and Boundaries

Use existing PyYAML and argparse. New public configuration maps to existing
ModelConfig, BenchmarkSuiteConfig, SuiteScenarioInstance and ExperimentRunSpec;
the public runner delegates actual episodes to the existing runner. A known
built-in suite is materialized before filtering by actual family, role, profile
or condition ID. A custom graph requires a complete scenario, not a topology
substitution that silently invalidates its authored task/reference.

Retain fixed-step time, private/public separation, edge direction/enabled/sign/
strength/delay, resources/faults, command histories, response modes and exact
scoring. Do not offer frozen-suite horizon/scoring overrides as convenience
settings. Runtime timeout, token budget and repetitions are explicit policy/run
configuration, not hidden benchmark changes.

Validation and dry-run must perform no inference or network request. Cloud run
requires credentials and explicit confirmation. Runtime outputs are untracked,
with secret-free resolved config and existing episode summaries/JSONL. A reused
output directory must not silently overwrite an experiment. Sequential execution
is the default; no automatic parallelism or model downloads.

Package canonical data with the package rather than depend on sibling development
paths. Keep historical 0.5.0 and suite identities; document the YAML wrapper as
release-only, not as the historical campaign's execution interface. Reconstruct
condition/prompt identity and preserve a compact original-generator hash fixture;
do not promise identical hosted/local generations or fabricate missing provenance.

## Verification Plan

- Snapshot the development files independently, excluding the release destination
  and caches, and verify no pre-existing file changed at the end.
- Preserve/hash scientific modules and all 240 materialized condition definitions.
- Run retained offline tests and new strict-config, selection, credential safety,
  dry-run, CLI, native-runner equivalence and offline smoke tests.
- Test editable installation in a clean venv, then a wheel from a relocated copy
  with the development source absent from sys.path.
- Scan every intended release file for secrets, personal paths, cache/build
  debris, oversized files and broken internal documentation links.
- Generate a release file/hash manifest and a clear author-decision checklist.
  Do not initialize a repository, publish a package, create a remote or push.
