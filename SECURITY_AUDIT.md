# Security and Privacy Audit

Scope: intended standalone source release, not the development account, private
credential stores, historical campaign outputs or future user-generated results.

## Inclusion Review

Only audited runtime modules, offline tests/fixtures, public configs/docs and
compact definition evidence are included. No environment file, API credential,
SSH/cloud account material, raw model generation, manuscript, literature PDF,
internal review, model weights, provider cache or machine-tuning file was copied.
`.env.example` contains empty placeholders only; it is not loaded automatically.

## Checks

`scripts/check_release.py` scans every intended file for private-key blocks,
probable provider/AWS key patterns, credential-bearing URLs, personal home paths,
unexpected binary/cache types, oversized files and broken local links. Additional
manual review covers credential assignments, bearer/header handling, email-like
strings, hidden files and endpoint declarations. Matching values are never printed.

The tests intentionally contain synthetic keys and one credential-bearing loopback
URL rejected by validation. That exact fixture file has a narrow URL-pattern
exception in the scanner; no runtime configuration has such an exception. Provider
endpoint constants are public APIs or documented loopback/test endpoints. Environment
variable *names* and code constructing an Authorization header are not credentials.

## Runtime Handling

- Keys are required by environment-variable name; resolved configs never expand
  environment values. Inline key/header/import fields are not accepted in release
  YAML. URL userinfo/query/fragment are rejected.
- Compatible remote endpoints require HTTPS (loopback HTTP is allowed). The native
  Ollama adapter assumes a trusted local deployment; it is not a secure public proxy.
- Public validation, listing and dry-run do not contact providers. Normal tests block
  real socket connections. Model factories validate options without generation.
- Mock cloud execution tests check that the environment key never appears in any
  generated file. Existing provider error/metadata redaction tests are retained.
- Cloud calls require explicit authorization and stop on infrastructure failure;
  no automatic whole-episode retry or model download is introduced.

## Limits

Pattern scans cannot prove absence of every possible secret. Arbitrary free-text
descriptions or custom scenarios should never contain credentials. Users control
where a configured service receives public task content; inspect endpoints and
provider privacy terms. Logs contain prompts, responses, timestamps, local paths
and private simulator audit state and need a separate publication review. Ignored
files are not a substitute for access controls. This is research software, not a
production safety or security certification.
