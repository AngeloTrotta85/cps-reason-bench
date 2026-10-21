# Release Preparation Report

Date: 2026-10-21. Candidate root: this directory. Existing package version 0.5.0.
No repository has been initialized, no package published and no model campaign
rerun. This report separates software verification from live provider qualification.

Post-preparation correction: `scripts/check_release.py` now loads this checkout's
`src/` explicitly instead of assuming an editable package installation. It retains
source-isolation verification and gives a concise installation hint for missing
PyYAML. Four subprocess regressions cover an uninstalled benchmark, a competing
package on PYTHONPATH, missing PyYAML and an incomplete checkout. No benchmark or
provider code changed. The release file manifest was refreshed for this correction.

## A. Existing Repository Audit

The pre-copy [audit](RELEASE_AUDIT.md) covers package/module imports, loaders,
simulator, Graph-v2 generation, prompts/protocol, policies/adapters, validators,
metrics, logging, generic reporting/plots, tests, campaign machinery, frozen data,
paper/support files, caches and private/local assumptions. It was created before
the release tree. The runtime path is configuration -> condition/model -> public
observation -> prompt/decision -> validation -> accepted commands -> fixed-step
world update -> evaluator -> trace/summary/export.

Included: core runtime/import closure, native/compatible adapters, frozen suites,
needed legacy regression fixtures, offline tests, compact definition evidence and
new public documentation. Excluded: raw model outputs, research launchers, model
discovery/account caches, reuse/add-on histories, paper/reviews/literature and build
or private files. The per-file Python/test and copy-origin decisions are in
`artifacts/source_inventory.json`.

The independent baseline covered 47,395 development files. Final comparison found
no removed files and no development code/frozen-data changes. Six concurrent
paper changes were observed: the introduction source plus main PDF/log/build
sidecars. This task did not write them; they were left untouched. No historical
paper artifact manifest was refreshed. The only intentional development-root
addition was the requested `RELEASE_AUDIT.md`; all release work is separate.

## B. Release Tree

```text
.
|-- README.md, LICENSE, pyproject.toml, MANIFEST.in
|-- .gitignore, .env.example
|-- configs/
|   |-- examples/       # seven complete public run configs
|   |-- models/         # reusable local and compatible-cloud models
|   |-- graphs/         # complete two-node custom scenario
|   `-- benchmark/      # paper-condition scripted preset
|-- src/cps_reason_bench/
|   |-- public_config.py, public_run.py, public_cli.py, __main__.py
|   |-- simulator/, policies/, models/
|   |-- graph_v2.py, runner.py, protocol.py, metrics.py, ...
|   `-- data/           # installed suite/template resources
|-- tests/              # 17 retained modules + public API/checker tests/fixtures
|-- examples/           # advanced legacy regression fixtures, not campaign history
|-- benchmark/          # unchanged Graph-v2 suites and compact definition evidence
|-- artifacts/          # source inventory and original condition-hash fixture
|-- scripts/check_release.py
|-- docs/               # seven focused manuals
|-- RELEASE_AUDIT.md, RELEASE_MANIFEST.md, release_manifest.json
|-- PAPER_SOFTWARE_CONSISTENCY.md, SECURITY_AUDIT.md
`-- GIT_READY_CHECKLIST.md, RELEASE_TODO.md, RELEASE_PREPARATION_REPORT.md
```

The package contains 52 Python modules: 48 selectively retained plus the new
entry point and three public wrapper modules. Packaged data is self-contained.
Source-level fixture copies support retained path-based tests; no symlink points
back to the development tree.

## C. Core-Code Changes

| Change | Category | Scientific effect |
| --- | --- | --- |
| `public_config.py` | RELEASE WRAPPER / NEW FEATURE | Strict YAML, real identifiers, deterministic filtering, reference resolution, backend option checks. No condition rewriting. Explicit protocol overrides create variants with different hashes. |
| `public_run.py` | RELEASE WRAPPER | Same experiment expansion/episode runner/scorer; fresh-output protection, scale cap, cloud authorization, stop on infrastructure failure, snapshots and partial exports. Operational scheduling differs from historical launchers, not physical/scoring rules. |
| `public_cli.py`, `__main__.py`, top-level exports | RELEASE WRAPPER / NEW FEATURE | Public CLI/API; historical CLI not carried over. Alias `cps-reason-bench` now invokes the release CLI, not old campaign subcommands. |
| `experiments.expand_experiment` optional `suite_instances`/`model_configs` | REFACTOR | Accepts pre-materialized inputs while sharing the original identity/seed loop. Existing file-based default preserved; regression equivalence checked. |
| `experiments.prepare_experiment` optional `provenance` | REFACTOR | Avoids unsolicited localhost discovery in wrapper; existing caller default preserved. No simulation effect. |
| `paths.examples_root` packaged-data priority | BUG FIX | Installed-wheel resources no longer depend on a sibling checkout. Resource bytes unchanged. |
| Packaging/CLI entry points/package data | CLEANUP | Explicit PyYAML runtime, dev and plot extras; setuptools >=77, existing MIT SPDX expression, Python >=3.10, existing 0.5.0. No new dependency on provider SDKs. |
| Test fixture adaptation | CLEANUP | Graph documentation paths use public equivalents; normalization registry test uses a compact synthetic registry instead of campaign inventory; time tests omit one historical-launcher classifier assertion while retaining runtime timing/error assertions. |
| New public tests/network guard/check script | NEW FEATURE | Covers wrapper correctness/security/isolation without changing benchmark behavior. |
| Source-checker bootstrap and dependency error | BUG FIX | Checks this checkout without a package installation; preserves import-isolation checks, disables checker-generated bytecode and reports missing PyYAML. No benchmark effect. |

Simulator, graph generator, protocol, prompts, response normalization, metrics,
family gates, scheduling/context/subscriptions and provider transport modules are
byte-identical to their retained development counterparts. The release does not
make numerical/scientific edits to these modules. Source hashes make this auditable.

## D. Configuration System

YAML uses the existing required PyYAML dependency. One file contains
`config_version`, `name`, optional `description`, `models`, `benchmark`,
`repetitions`, optional `simulator_seeds`/`protocol`, `output_dir`, and `runtime`.
Models can be inline or referenced; graphs are either complete frozen conditions
or complete scenario YAML, never standalone topology substituted into a frozen task.

Filters expose actual family/role/profile/condition IDs and optional ordered limit.
Model options use existing `decoding`, `backend_parameters`, `timeout_seconds`,
`api_key_env` and `response_normalization` concepts. Unknown fields and incompatible
combinations fail early. Dry-run is network-free and credential-independent.
Validation checks credentials unless `--offline`; actual availability remains a
separate authorized operational check.

The runtime is sequential, capped by `max_episodes`, with optional wall-clock call
pacing. Cloud retries default to one transport attempt. There is no monetary cap,
concurrency or public campaign resume/retry command. Existing output is refused.
These boundaries are documented, not hidden behind compatibility claims.

## E. CLI

Commands: `validate`, `dry-run`, `run`, `list-providers`, `list-benchmarks`,
`list-families`, `list-graphs`, `list-models --config`, `list-conditions --config`,
plus `--help` and `--version`. `python -m cps_reason_bench` uses the same CLI.

```sh
cps-reason validate configs/examples/local_quickstart.yaml
cps-reason dry-run configs/examples/local_quickstart.yaml
cps-reason list-conditions --config configs/examples/selected_conditions.yaml
cps-reason run configs/examples/offline_smoke.yaml
```

Cloud execution prints model/condition/repetition/episode counts and requires
`RUN`, or explicitly supplied `--yes`. Python callers must pass
`authorize_cloud=True`. Errors are not silently reclassified as task results.

## F. Documentation

`README.md` provides purpose, nonclaims and quick starts. `docs/USER_GUIDE.md`
covers practical workflows, outputs, interpretation and extension.
`CONFIGURATION.md` is the release field/default reference. `ARCHITECTURE.md`
explains modules, execution, APIs and tests. `PROVIDERS.md` explains existing
adapters, setup and limits, with official Ollama documentation links.
`GRAPHS_AND_SCENARIOS.md` explains axes and complete custom scenarios.
`REPRODUCIBILITY.md` distinguishes frozen definitions from exact generations.
`TROUBLESHOOTING.md` covers install/config/runtime errors. Root audit/manifest,
security, consistency, checklist and author-decision files complete the release.

## G. Examples

| Run configuration | Purpose / planned episodes |
| --- | --- |
| `local_quickstart.yaml` | Small Ollama operational check / 1. |
| `local_full_benchmark.yaml` | Complete local Graph-v2, three repetitions / 720; not executed. |
| `cloud_quickstart.yaml` | One OpenRouter model with env credential / 1; no real call made. |
| `mixed_models.yaml` | One local and one cloud model sequentially / 2. |
| `selected_conditions.yaml` | Family/profile/role/exact-ID intersection retaining a relational pair, scripted / 2. |
| `custom_graph.yaml` | Complete authored two-node graph/task with feasible scripted policy / 1. |
| `offline_smoke.yaml` | Scripted and random smoke controls, no service needed / 2. |
| `configs/benchmark/paper_graph_v2.yaml` | Separate paper-condition preset, one scripted-reference pass / 240. |

All seven run examples and the paper preset validated offline. Custom, offline
smoke and isolated-wheel offline execution were run. No complete local LLM or
cloud benchmark was executed; no weights downloaded. Referenced provider/model
availability is not claimed to have been live-qualified.

## H. Reproducibility

All 240 definitions/seeds and original generator hashes are retained, as are
prompt/schema identities and response-mode semantics. The condition preset uses
strict prompt 2.0.0/schema 1.0.0 with zero corrections. It reproduces a condition
protocol, not the historical 12-interface LLM panel or exact model generations.

Paper repetitions, matched-pair boundaries, separate PICC control and provenance
limits are documented. Raw model evidence and historical route/add-on machinery
were not guessed or repackaged. The new file manifest does not touch the paper
artifact manifest.

## I. Tests

- Editable environment after checker correction: **225 passed, 0 failed, 0 skipped; 120 subtests passed**.
- Original independent wheel environment with relocated source/tests: **221 passed,
  0 failed, 0 skipped; 120 subtests passed**.
- Of the current 225 tests, 56 cover the public release layer, four cover the
  standalone source checker, and 165 are retained runtime, generation,
  normalization, provider, logging/reporting/plot tests. The checker-only fix does
  not change wheel runtime contents.
- Original-generator equality covers 240 conditions; existing references and
  counterfactual checks cover all 240 conditions / 48 pairs.
- Isolated `python -I` verified site-packages import, packaged suite data and a
  completed offline episode with native result loading, from outside the checkout.
- Syntax/imports, CLI commands, examples, local docs links, secret/path scan and
  manifest integrity checked. Actual plotting tests ran with matplotlib installed.

Tests block real sockets or use explicit mock backends. No real cloud/Ollama
generation was used. Python 3.13 was tested on this machine; the existing >=3.10
support declaration is retained, not a claim that every supported interpreter/OS
was tested here. Optional dependency ranges are not a historical dependency lock.

## J. Fresh Installation

Commands tested from the development root for the candidate editable install:

```sh
python3 -m venv /tmp/cps-release-editable-venv
/tmp/cps-release-editable-venv/bin/python -m pip install -e './git[dev,plots]'
cd git
/tmp/cps-release-editable-venv/bin/cps-reason --help
/tmp/cps-release-editable-venv/bin/cps-reason validate configs/examples/local_quickstart.yaml
/tmp/cps-release-editable-venv/bin/cps-reason dry-run configs/examples/local_quickstart.yaml
/tmp/cps-release-editable-venv/bin/python -m pytest -q --tb=short
```

Package-index downloads initially failed in the restricted network sandbox, then
succeeded with approved network access in temporary environments. No development
venv was modified. Build and isolated-wheel checks:

```sh
/tmp/cps-release-editable-venv/bin/python -m pip install 'setuptools>=77' wheel
/tmp/cps-release-editable-venv/bin/python -m build --no-isolation --outdir /tmp/cps-release-dist
python3 -m venv /tmp/cps-release-wheel-venv
/tmp/cps-release-wheel-venv/bin/python -m pip install '/tmp/cps-release-dist/cps_reason_bench-0.5.0-py3-none-any.whl[dev,plots]'
cd /tmp/cps-release-standalone
env -u PYTHONPATH /tmp/cps-release-wheel-venv/bin/python -m pytest -q --tb=short
```

The standalone directory was an independent file copy, not a symlink; no sibling
development paths were needed. A separate `python -I` execution in `/tmp` loaded
only the wheel package and ran a scripted smoke episode. The source distribution
and wheel built successfully. Build/license/package-data warnings encountered
during preparation were corrected; no missing packaged resource remains.

## K. Security / Privacy

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md). Intended files contain no detected real
key/private-key material, personal home path, credential-bearing runtime URL,
unapproved email address, raw model output or private manuscript material. Only
`.gitignore` and placeholder `.env.example` are intended hidden files. Synthetic
test credentials are explicitly test-only. Resolved/runtime-output credential
leakage is checked with mocked cloud execution.

No scanner proves arbitrary future free text is secret-free. Generated logs may
include local paths, prompts, responses and private simulator state; a separately
reviewed publication process is required. Endpoint privacy remains a user/provider
responsibility. No money/continuous-control safety guarantee is provided.

## L. Paper Consistency

[PAPER_SOFTWARE_CONSISTENCY.md](PAPER_SOFTWARE_CONSISTENCY.md) maps retained
identities, time/dynamics, public observations, accepted histories/gates, outcome
layers, pairs, repetitions, interfaces and provenance limitations to the paper.
**No frozen scientific semantics changed.** Wrapper operational defaults are
documented separately. Explicit protocol/custom scenario/seed/model changes are
new experimental configurations, not silent substitutes for historical evidence.

## M. Excluded Material

Historical budget/qualification/recovery/adoption/parallel/add-on launchers and
their tests; private machine/account settings; raw runs/failures/provider caches;
paper analysis/tables/figures; manuscripts, reviews, checkpoints and local literature;
model binaries; build/editor caches. Unused comprehensive/factorial/robustness suite
fixtures were removed from the candidate, while necessary older-suite regression
fixtures remain. All originals remain in the development tree.

## N. Git Readiness

The standalone source tree is ready for author review and initialization. File
hashes, links, syntax/import isolation and source scan are reproducible with
`python scripts/check_release.py`. Generated/cache files are ignored and removed
from the delivered candidate. `git init`, `git add`, commit, remote and push have
**not** been executed. Inspect staged content before publishing; no software check
replaces that author review.

## O. Remaining Author Decisions

Public host/URL/visibility; first public tag/version policy; approved authors and
citation/archival metadata; whether and under which reviewed terms to publish a
separate results bundle. Existing MIT licensing is preserved, not unresolved.
See [RELEASE_TODO.md](RELEASE_TODO.md). No implementation work is deferred there.

## P. Recommended First Public Release Workflow

From this candidate root, review README, consistency/security reports and author
decisions; then:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,plots]'
python -m pytest
python scripts/check_release.py
cps-reason dry-run configs/examples/local_quickstart.yaml
# Only after your review, run these yourself:
git init
git add .
git status --short
git diff --cached --stat
git diff --cached
```

If reviewed source changes are needed, regenerate the release manifest and rerun
checks before staging. Choose real repository/citation metadata before publishing;
configure a remote or tag only after author approval. No remote URL or DOI is
invented here. Run live local/cloud smoke tests separately under your own explicit
authorization and account/hardware constraints.
