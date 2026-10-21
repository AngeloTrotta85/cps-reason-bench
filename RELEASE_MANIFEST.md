# Release Manifest

`release_manifest.json` is the standalone software release's own manifest. It is
not the paper artifact manifest and does not replace or refresh it. It records
package/config/generator/suite identities and SHA-256/size for every intended source
file, using paths relative to this repository root.

The manifest excludes itself to avoid a self-hash cycle. It also excludes ignored
build/cache/virtualenv/git directories and generated runs. No runtime result or
private account log is intentionally included. `artifacts/source_inventory.json`
records copy origins (relative development paths and original hashes), included
and excluded modules/tests, and whether copied bytes were changed in the release.
`artifacts/graph_v2_conditions.json` records semantic condition hashes independently
of file hashes. Tests verify all 240 against the release generator.

Verify a source checkout:

```sh
python scripts/check_release.py
```

The checker loads this checkout's `src/` directly; installing `cps_reason_bench`
or setting `PYTHONPATH` is not required. The selected interpreter still needs
PyYAML. If it is missing, activate your release virtual environment or run
`python -m pip install 'PyYAML>=6.0'` with that interpreter. The checker never
installs dependencies itself and retains the source-import isolation checks.

After an intentional reviewed edit, regenerate only this manifest:

```sh
python scripts/check_release.py --write-manifest
python scripts/check_release.py
```

The check scans intended files, local documentation links, probable secret/path
patterns, size/type constraints, syntax/imports and byte hashes. It is not a proof
that arbitrary future input is secret-free. Inspect generated result bundles
separately before publishing them. Never regenerate the manifest merely to silence
an unexplained mismatch.

The source distribution includes public docs/configs/tests/fixtures; the wheel
includes runtime modules and packaged definitions/templates. A wheel installation
does not install the repository's manuals into the current directory. Keep the
source checkout for examples or author a run YAML using built-in suite IDs.
