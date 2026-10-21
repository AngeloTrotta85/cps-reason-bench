# Graphs and Scenarios

## Frozen Graph-v2

`cps_reason_graph_v2@1.0.0` is materialized by
`cps_reason_graph_v2_generator@1.0.0` from an unchanged suite. It contains 240
conditions, not 240 unrelated physical graphs: six families x eight authored
profiles x five semantic graph roles. The generator couples topology to the task,
observation, action catalog, reference and evaluator. Do not replace just the
graph inside an existing condition and call the result frozen Graph-v2.

| Axis | Actual identifiers and meaning |
| --- | --- |
| Family | `prospective_reasoning` (future intentions), `planning` (prerequisite sequences), `interference` (competing/misleading evidence), `conditional_safety` (inspection and conditional rules), `adaptation` (fault-sensitive changes), `temporal_tracking` (delayed-event credit/attribution). These operationalizations are not diagnoses of human faculties. |
| Profile | `graph_v2_entry_1`, `graph_v2_entry_2`, `graph_v2_entry_3`, `graph_v2_bridge`, `graph_v2_core_1`, `graph_v2_core_2`, `graph_v2_core_3`, `graph_v2_core_4`. Authored demand profiles, not calibrated universal difficulty scores. |
| Role | `i01`: local anchor; `i02`/`i03`: matched relation-sensitive pair; `i04`: short nonlocal chain/branch; `i05`: multipath role. Node count is not identical to profile. |

The 48 i02/i03 pairs manipulate direction, coupling sign, enabled state or coupling
strength. A pair shares relevant non-relational attributes but requires different
correct behavior. Retaining only one member does not support paired inference.
An altered trajectory by itself does not show that a model reasoned about a graph.

```yaml
benchmark:
  suite: cps_reason_graph_v2
  families: [adaptation]
  profiles: [graph_v2_bridge, graph_v2_core_1]
  roles: [i02, i03]
```

This selects four conditions. Use `condition_ids` for a particular generated graph
and task. Use multiple IDs for several named conditions, retaining their existing
meaning. `list-graphs` explains axes; `list-conditions --config FILE` returns exact
IDs and hashes. Offline generation has no provider dependency.

The pilot has nine conditions and is not a representative reduced statistical
sample. Local smoke (one) and cloud qualification (three) are operational checks,
not substitutes for the scientific suite.

## Complete Custom Graphs

The public layer also accepts the existing complete scenario YAML schema. See
[`two_node_transfer.yaml`](../configs/graphs/two_node_transfer.yaml), a two-node
directed-coupling example with sensors, finite energy, prerequisites, safety,
goal, horizon and feasible scripted reference. Its condition is explicitly named
`custom`, not a new Graph-v2 role or official benchmark version.

```yaml
benchmark:
  scenarios:
    - ../graphs/two_node_transfer.yaml
    - ../graphs/my_second_scenario.yaml
```

The second file above is a user-authored file, not a shipped example. Scenario
names must differ. For one graph, use a one-element list. To create a variation,
duplicate the complete example, change its `name`, author consistent actions,
sensors/goals/reference, and run `validate` before calling a model. Each scenario
is hashed and snapshotted. Custom selections have a separate `custom_scenarios`
suite identity based on their full content. They must not be pooled with frozen
conditions without explicitly labeling the changed design.

### Scenario Contract

This is the existing advanced scenario schema, not a second graph engine. Its
loader retains authored mappings for domain-specific fields. The public wrapper
does not coerce a custom file into a frozen template. Refer to the executable
examples in `examples/scenarios/` and `config.ScenarioConfig` for advanced cases.

| Section | Contents and behavior |
| --- | --- |
| Identity | `schema_version: 1`, required `name`, `task_family`; description, difficulty and `difficulty_level`, authored/computed `difficulty_profile`, `generation` metadata. |
| Time | `seed`, positive `max_steps`, positive `time_step`; a fixed simulator grid, not wall-clock time. |
| `task` | Public instructions/objectives/rules and optional `terminate_on_goals`. |
| `graph.nodes` | Named nodes with named variables. Each has an `initial` value; numeric variables may have drift and hard bounds. Labels/observable attributes remain as authored. |
| `graph.edges` | Unique `id`, existing `source`/`target`, optional shared `variable`, numeric `coupling`, `directed`, `enabled` and supported public/hidden metadata. The coupling key is not `weight`. |
| `graph.resources` | Named initial resources, minimum bounds, public visibility. Action costs consume them. A resource `max` is declarative in this engine, not an enforced storage limit. |
| `sensors` | Unique ID and target, observability, noise/delay and runtime fault effects. Hidden state does not become public simply because it is logged. |
| `actions` | Unique ID and operation: `noop`, `adjust`, `set`, `toggle_edge`, `inspect`, `repair`, `communicate`; operation-specific target/value, prerequisites, costs and optional delayed effects. |
| `faults` | Scheduled sensor bias/stuck/unavailable/delay, disabled/scaled/remapped actuator, disabled edge or drift change; target/type-specific data. |
| `constraints` / `goals` | Bounds/conditional safety rules and goal conditions (`condition`, `all`/`any` forms where supported). Validates targets and feasible reference. |
| `observability` | Public-state projection and optional full state; changing it changes the information available to the agent. |
| `interaction` | Periodic/scheduled/event or mixed modes, interval, initial invocation, callback limits, allowed event types, batch size and sequential/atomic semantics. |
| `context` | Current/previous/history/summary context, history/request limits, action outcomes and public events. |
| `evaluation` | Required/forbidden actions, inspections, expected sequence, safety policy, family-specific behavioral gates and reference quantities. |
| `baseline` / `reference` | Step-indexed scripted decisions/default and declared evidence/optimality. A feasible reference is not automatically an optimal or public-information controller. |

The legacy schema is intentionally not broadly rewritten here; not every nested
mapping has strict unknown-key checking. Semantic validation and reference replay
are important when authoring a graph. Do not infer physical correctness from YAML
parsing alone. See `tests/test_graph_v2_engine.py` and
`tests/test_config_and_simulator.py` for executable contracts.

## Dynamics and Delays

Each active coupling computes `(source - target) * coupling` from a shared
pre-coupling snapshot and accumulates deltas. A directed edge updates the target;
an undirected edge also applies the opposite delta to the source. Sign, strength
and enabled state therefore matter. These are abstract discrete dynamics, not a
general-purpose physical solver. The release does not introduce a per-edge
propagation-delay field; delayed action effects and delayed sensor observations
retain their existing distinct mechanisms.

## Extending the Design

Adding models at existing backends or complete scenario YAML is a configuration
operation. Adding a new frozen suite/generator version or a new operational
family requires implementation, invariant/reference tests, documented scoring,
new identity/version decisions, and separate scientific validation. Do not add a
new family name to YAML and assume the evaluator implements its construct.
