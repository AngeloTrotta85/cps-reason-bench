# Troubleshooting

| Symptom | Action |
| --- | --- |
| `cps-reason` not found | Activate the venv, install with that interpreter, check `python -m cps_reason_bench --help`. |
| Install cannot reach package index | Allow package downloads or install approved wheels from a local wheelhouse. Runtime requires PyYAML; testing/plots have separate extras. |
| Wrong package imported | Run `python -c 'import cps_reason_bench; print(cps_reason_bench.__file__)'`. Remove an old `PYTHONPATH`; the installed release must not import a sibling checkout. |
| Config/scenario/model file missing | Paths are relative to the YAML containing the reference, not your shell directory. |
| Unknown role/profile | Roles are `i01`-`i05`; profiles are `graph_v2_entry_1` through `graph_v2_core_4`, including `graph_v2_bridge`. Use listing commands. |
| Empty condition selection | Filters intersect. Inspect IDs in the selected source; use full suite when asking for conditions absent from a smoke suite. |
| Episode cap exceeded | Inspect dry-run intent, then explicitly increase `runtime.max_episodes`. This is not a spending budget. |
| Missing credential variable | Export the exact `api_key_env` name privately. `.env` is not automatically loaded. Use `validate --offline` only to skip presence checks, not to run without credentials. |
| Ollama connection/model failure | Check service and installed model ID yourself using `ollama ls`. Validation intentionally does not probe it. Review timeout and hardware; avoid oversized context/token settings. |
| OpenRouter model/route unavailable | Verify current model and route/account permissions. Requested privacy/parameter constraints can make a route unavailable. Do not silently weaken privacy or substitute a model under an old identity. |
| Rate limit | Provider capacity/account request or token throughput limit; inspect provider evidence. Reduce pacing/scale or retry later with deliberate authorization. |
| Timeout/empty/provider envelope failure | Infrastructure failure, possibly billed. Inspect `failures.json` and episode summary before another attempt. No automatic whole-episode retry. |
| Invalid decision JSON/fence/prose | Protocol failure, not necessarily infrastructure. Choose an explicitly separate normalized interface only if scientifically intended; it does not repair JSON or reasoning. |
| Safe state but task failure | Check accepted action sequence and active inspection/timing/credit gates, not only final state. |
| Custom graph reference fails | Inspect action prerequisites, energy, coupling (`coupling`, not `weight`), observability, horizon and family invariants; validate the full authored scenario. |
| Existing output directory | Preserve it. Choose a different `output_dir`; no automatic overwrite or campaign resume is provided. |
| Ctrl-C or power loss | Keep existing evidence. Normal Ctrl-C exports completed records; abrupt power loss may not. Review in-flight billing before constructing a new selected run. |
| No plots | Install `.[plots]` and use the documented `write_result_plots` API. Tests for actual rendering skip without matplotlib. |

Expected configuration errors are concise CLI messages. Unexpected programming
exceptions keep their traceback and partial metadata with status `error`; report
them with a minimal credential-free config and software versions. Never paste a
key, authorization header or unreviewed account dump into an issue.
