# Git-Ready Checklist

This is a standalone source candidate, not an initialized or published repository.

- [x] Core runtime retained; physical/scoring modules and frozen Graph-v2 definitions unchanged.
- [x] Original-generator hashes verify all 240 condition definitions.
- [x] Existing MIT license retained; no invented authors, URL, DOI or new release tag.
- [x] No real credentials, personal paths, raw model outputs, private manuscript material or unintended binary artifacts in intended source files.
- [x] `.env.example` contains empty placeholders only; runtime uses environment-variable names.
- [x] `.gitignore` covers secrets, caches, environments, outputs, logs, model weights and editor/build debris.
- [x] Editable installation and independent wheel installation verified in clean temporary environments.
- [x] Offline test suite passes: 225 tests and 120 subtests, no failures/skips with plot dependencies installed.
- [x] Source checker runs without installing the benchmark package; missing PyYAML is reported clearly.
- [x] README help/validation/dry-run and offline simulator commands verified.
- [x] All seven example run configs and paper-condition preset validate offline.
- [x] Real cloud execution requires explicit confirmation; no paid/live model calls made during preparation.
- [x] Standalone packaged data/imports work without a development-source dependency.
- [x] Documentation links, Python syntax/imports and release file hashes checked.
- [x] No source file exceeds the release's 2 MB audit threshold.
- [x] Release manifest and per-file source inclusion/exclusion inventory generated.
- [x] No git initialization, staging, commit, remote configuration or push performed.

After review, the intended `git init` / `git add .` workflow should stage only
release source files, not ignored runtime/build/cache contents. Re-run
`python scripts/check_release.py` before staging after any edits. This checklist
does not certify future additions; inspect `git diff --cached` yourself.

Only publication metadata/artifact decisions remain in [RELEASE_TODO.md](RELEASE_TODO.md).
