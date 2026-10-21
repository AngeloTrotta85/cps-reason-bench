# Paper / Software Consistency

The current human-edited paper's design, methods and limitations were read as
read-only evidence. No manuscript, frozen results or historical artifact manifest
was changed by this release task. Paper text and internal reviews are not shipped.

| Scientific object | Release treatment |
| --- | --- |
| Package / generator / suite | Existing 0.5.0 / Graph-v2 generator 1.0.0 / `cps_reason_graph_v2@1.0.0`; preserved, not a new release tag. |
| Conditions | Same six families, eight profiles, roles i01-i05 and 240 authored conditions. Golden per-condition hashes computed from the unmodified development generator; release comparisons cover all 240. |
| Public vs private information | Simulator, observation, context and policy boundaries unchanged. Audit logs are not agent observations. |
| Time and graph dynamics | Fixed steps, fault/effect/drift/coupling ordering, delayed effects/sensing, resources and action semantics unchanged. |
| Accepted histories and gates | Accepted action sequence, named no-op membership, inspection/timing/credit and family-specific requirements unchanged. |
| Outcomes | Safe-goal attainment and full-task success remain distinct; exact authored sequence agreement is not a universal optimal-control metric. |
| Response modes | Strict decision schema unchanged. Single-fence normalization remains an explicitly different model-interface condition with zero corrections. |
| Relational pairs | All 48 i02/i03 pairs retained. Filtering does not confer paired validity when a member is omitted. |
| Repetitions / recurrence | Paper LLM panel: 12 model-interface configurations x 240 x 3 = 8,640 episodes. Scripted: 240 once; random: 720. Repetitions are clustered within a condition, not 720 independent graphs. PICC is a separate 240-condition control, not a thirteenth LLM. |
| Reference information | Scripted reference validates feasibility; it is not information-equivalent to a model. PICC policy/adapter retained, historical PICC campaign/analyzer excluded. |
| Provenance | Exact historical generations, provider serving state and unretained metadata cannot be reconstructed by a fresh YAML invocation. No missing values invented. |

## Release-Only Changes

The YAML schema and CLI are a new **wrapper**, not the historical campaign
interface. They retain the existing physical/scoring code and exact frozen suite
files. Experiment expansion gained optional pre-materialized inputs; its shared
identity/seed algorithm remains unchanged. Workspace preparation gained optional
provenance injection to avoid an unsolicited local service probe. Resource lookup
now finds packaged data in installed wheels.

Operational defaults are explicitly documented: sequential execution, an episode
cap, fresh output directories, cloud confirmation, no automatic episode retries,
one cloud transport attempt by default, and no model discovery. These differ from
some historical launchers' scheduling/recovery behavior and must not be presented
as exact campaign replay. Provider sampling/output/format overrides can change
agent performance; the wrapper records them rather than calling them scientifically
equivalent. Explicit protocol changes alter the suite content hash.

**No frozen benchmark scientific semantics were changed.** A custom scenario or
changed protocol/seed/model configuration is a new experiment, not the frozen
paper condition. Generic aggregate/plot exports are not replacements for the
paper's dedicated denominator, recurrence, pairing and provenance analysis.
