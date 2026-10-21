"""Public release contracts: no sockets, exact selections, and existing seed rules."""

from __future__ import annotations

import io
import json
import socket
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cps_reason_bench import (
    instantiate_benchmark, list_conditions, load_config, load_results,
    plan_experiment, run_experiment, validate_config,
)
from cps_reason_bench.config import ConfigError, dump_scenario
from cps_reason_bench.models.base import ProviderError, ModelResponse
from cps_reason_bench.experiments import expand_experiment, load_experiment
from cps_reason_bench.public_cli import main
from cps_reason_bench.public_config import BUILTIN_SUITES, DATA
from cps_reason_bench.suites import load_suite, materialize_suite
from cps_reason_bench.versions import content_hash


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Public release test attempted a real network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def base():
    return {"config_version": 1, "name": "unit_test", "models": [{"name": "scripted", "backend": "scripted"}],
        "benchmark": {"suite": "cps_reason_graph_v2_local_smoke"}, "output_dir": "out"}


def write_config(tmp_path, data):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["local_quickstart", "local_full_benchmark", "cloud_quickstart", "mixed_models", "selected_conditions", "custom_graph", "offline_smoke"])
def test_examples_validate_without_provider(name):
    config = load_config(ROOT / f"configs/examples/{name}.yaml")
    validate_config(config, check_credentials=False)
    assert config.episode_count > 0


@pytest.mark.parametrize("field,value", [("repetitions", 0), ("repetitions", True), ("repetitions", "3"), ("models", []),
    ("simulator_seeds", [1, 1]), ("simulator_seeds", [True]), ("runtime", {"max_episodes": 0}),
    ("runtime", {"max_episodes": 1, "unknown": 1}), ("runtime", {"model_call_delay_seconds": float("nan")}),
    ("config_version", 2), ("description", []), ("unknown", 1)])
def test_invalid_top_fields(tmp_path, base, field, value):
    base[field] = value
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, base))


@pytest.mark.parametrize("field,value", [("suite", "unknown"), ("families", ["unknown"]), ("roles", ["Core1"]),
    ("profiles", ["i01"]), ("condition_ids", ["missing"]), ("roles", "i01"), ("limit", 0), ("graph", "fake")])
def test_invalid_selections(tmp_path, base, field, value):
    base["benchmark"][field] = value
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, base))


@pytest.mark.parametrize("model", [
    {"backend": "fake"}, {"path": "missing-model.yaml"}, {"backend": "ollama", "base_url": "localhost:11434"},
    {"backend": "ollama", "base_url": "http://user:pass@localhost:11434"},
    {"backend": "ollama", "backend_parameters": {"think": "false"}},
    {"backend": "ollama", "backend_parameters": {"unknown": 1}},
    {"backend": "ollama", "decoding": {"extra": {"temperature": 1}}},
    {"backend": "ollama", "decoding": {"seed": []}},
    {"backend": "ollama", "decoding": {"max_tokens": False}},
    {"backend": "ollama", "decoding": {"top_p": 2}},
    {"backend": "ollama", "response_normalization": {"mode": "repair"}},
    {"backend": "ollama", "response_normalization": {"mode": []}},
    {"backend": "ollama", "api_key": "never-allowed"},
    {"backend": "openai-compatible", "base_url": "http://example.com/v1", "api_key_env": "TEST_KEY"},
])
def test_invalid_models(tmp_path, base, model):
    base["models"] = [{"name": "invalid", **model}] if "path" not in model else [model]
    with pytest.raises(ConfigError):
        load_config(write_config(tmp_path, base))


def cloud_model():
    return {"name": "mock_cloud", "backend": "openai-compatible", "model": "test/model",
        "base_url": "https://example.invalid/v1", "api_key_env": "CPS_RELEASE_TEST_KEY",
        "backend_parameters": {"provider": "custom", "transport_retry": {"max_attempts": 1}}}


def test_credentials_are_checked_but_not_resolved(tmp_path, base, monkeypatch):
    base["models"] = [cloud_model()]
    path = write_config(tmp_path, base)
    monkeypatch.delenv("CPS_RELEASE_TEST_KEY", raising=False)
    config = load_config(path)
    with pytest.raises(ConfigError, match="CPS_RELEASE_TEST_KEY"):
        validate_config(config)
    assert main(["dry-run", str(path)]) == 0
    monkeypatch.setenv("CPS_RELEASE_TEST_KEY", "synthetic-private-value-for-test")
    config = load_config(path, check_credentials=True)
    assert "synthetic-private-value-for-test" not in json.dumps(config.to_dict())
    assert "CPS_RELEASE_TEST_KEY" in json.dumps(config.to_dict())
    with pytest.raises(ConfigError, match="authorize_cloud"):
        run_experiment(config)
    assert not config.output_dir.exists()


def test_cloud_failure_stops_before_second_episode(tmp_path, base, monkeypatch):
    base["models"] = [cloud_model()]
    base["repetitions"] = 2
    monkeypatch.setenv("CPS_RELEASE_TEST_KEY", "synthetic-value")
    config = load_config(write_config(tmp_path, base))
    failure = ProviderError("offline timeout", category="timeout", retryable=False, billing_ambiguity=True)
    with patch("cps_reason_bench.models.openai_compatible.OpenAICompatibleBackend.generate", side_effect=failure) as generate:
        result = run_experiment(config, authorize_cloud=True)
    assert generate.call_count == 1
    assert len(result.failed_runs) == 1
    assert len(result.run_results) == 0
    assert load_results(config.output_dir) == []
    assert result.failed_runs[0]["error_classification"]["billing_ambiguity"] is True


def test_mock_cloud_output_and_metadata_do_not_leak_key(tmp_path, base, monkeypatch):
    base["models"] = [cloud_model()]
    monkeypatch.setenv("CPS_RELEASE_TEST_KEY", "synthetic-private-runtime-value")
    config = load_config(write_config(tmp_path, base))
    response = ModelResponse('{"actions": [], "callback": null, "subscriptions": [], "context_requests": []}')
    with patch("cps_reason_bench.models.openai_compatible.OpenAICompatibleBackend.generate", return_value=response):
        result = run_experiment(config, authorize_cloud=True)
    assert len(result.run_results) == 1
    for path in config.output_dir.rglob("*"):
        if path.is_file():
            assert "synthetic-private-runtime-value" not in path.read_text()


def test_no_network_or_writes_in_dry_run(tmp_path, base):
    path = write_config(tmp_path, base)
    before = sorted(tmp_path.rglob("*"))
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        assert main(["dry-run", str(path)]) == 0
    output = json.loads(stdout.getvalue())
    assert output["planned_episodes"] == 1
    assert sorted(tmp_path.rglob("*")) == before


def test_selection_and_seed_counts(tmp_path, base):
    base["benchmark"] = {"suite": "cps_reason_graph_v2", "families": ["planning"], "roles": ["i02", "i03"], "profiles": ["graph_v2_core_1"]}
    base["models"].append({"name": "random", "backend": "random"})
    base["repetitions"] = 3
    base["simulator_seeds"] = [9, 10]
    config = load_config(write_config(tmp_path, base))
    assert len(instantiate_benchmark(config)) == 2
    assert {row["role"] for row in list_conditions(config)} == {"i02", "i03"}
    assert len(plan_experiment(config)) == config.episode_count == 24
    base["benchmark"]["condition_ids"] = [list_conditions(config)[0]["id"]]
    assert load_config(write_config(tmp_path, base)).episode_count == 12
    base["benchmark"]["roles"] = ["i01"]
    with pytest.raises(ConfigError, match="empty"):
        load_config(write_config(tmp_path, base))


def test_resolved_roundtrip_and_defaults(tmp_path, base):
    config = load_config(write_config(tmp_path, base))
    assert config.output_dir == tmp_path / "out"
    again = load_config(write_config(tmp_path, config.to_dict()))
    assert config.to_dict() == again.to_dict()
    assert [s.episode_id for s in plan_experiment(config)] == [s.episode_id for s in plan_experiment(again)]


def test_frozen_conditions_and_packaged_resources_are_identical(tmp_path, base):
    for name, filename in BUILTIN_SUITES.items():
        assert (DATA / "benchmark/suites" / filename).read_bytes() == (ROOT / "benchmark/suites" / filename).read_bytes()
        base["benchmark"] = {"suite": name}
        config = load_config(write_config(tmp_path, base))
        original = materialize_suite(load_suite(ROOT / "benchmark/suites" / filename))
        assert [content_hash(i.scenario.to_mapping()) for i in config.conditions] == [content_hash(i.scenario.to_mapping()) for i in original]
    golden = json.loads((ROOT / "artifacts/graph_v2_conditions.json").read_text())
    base["benchmark"] = {"suite": "cps_reason_graph_v2"}
    assert list_conditions(load_config(write_config(tmp_path, base))) == golden["conditions"]


def test_expansion_matches_existing_file_based_algorithm(tmp_path, base):
    base["models"] = [{"name": "random", "backend": "random", "decoding": {"seed": "derived"}}]
    base["repetitions"] = 3
    config = load_config(write_config(tmp_path, base))
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump(config.models[0].safe_dict()))
    legacy = {"name": config.name, "suites": [str(ROOT / "benchmark/suites/graph_v2_local_smoke.yaml")],
        "models": [str(model)], "repetitions": 3, "output_dir": str(tmp_path / "out")}
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy))
    old_specs = expand_experiment(load_experiment(path))
    new_specs = plan_experiment(config)
    assert [(s.episode_id, s.random_baseline_seed, s.requested_model_seed) for s in old_specs] == [
        (s.episode_id, s.random_baseline_seed, s.requested_model_seed) for s in new_specs]


def test_protocol_override_is_explicit_and_normalization_cannot_correct(tmp_path, base):
    original = load_config(write_config(tmp_path, base))
    base["protocol"] = {"prompt": {"version": "1.0.0"}}
    variant = load_config(write_config(tmp_path, base))
    assert variant.suite.identity != original.suite.identity
    assert variant.conditions == original.conditions
    base["models"][0]["response_normalization"] = {"mode": "markdown_single_fence"}
    base["protocol"]["retry_policy"] = {"name": "assisted", "max_corrections": 1}
    with pytest.raises(ConfigError, match="zero output corrections"):
        load_config(write_config(tmp_path, base))


def test_offline_execution_and_existing_output_guard(tmp_path, base):
    config = load_config(write_config(tmp_path, base))
    result = run_experiment(config)
    assert len(result.run_results) == 1
    assert result.run_results[0].completion_status == "completed"
    assert len(load_results(config.output_dir)) == 1
    assert (config.output_dir / "config.resolved.yaml").is_file()
    assert json.loads((config.output_dir / "metadata.json").read_text())["status"] == "completed"
    with pytest.raises(ConfigError, match="already exists"):
        run_experiment(config)


def test_interruption_keeps_partial_outputs(tmp_path, base):
    config = load_config(write_config(tmp_path, base))
    with patch("cps_reason_bench.public_run.run_experiment_episode", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_experiment(config)
    assert json.loads((config.output_dir / "metadata.json").read_text())["status"] == "interrupted"
    assert load_results(config.output_dir) == []


def test_unexpected_errors_are_not_swallowed(tmp_path, base):
    config = load_config(write_config(tmp_path, base))
    with patch("cps_reason_bench.public_run.run_experiment_episode", side_effect=RuntimeError("programming error")):
        with pytest.raises(RuntimeError, match="programming error"):
            run_experiment(config)
    assert json.loads((config.output_dir / "metadata.json").read_text())["status"] == "error"


def test_two_custom_graphs(tmp_path, base):
    original = load_config(ROOT / "configs/examples/custom_graph.yaml").conditions[0].scenario
    dump_scenario(original, tmp_path / "one.yaml")
    changed = original.to_mapping()
    changed["name"] = "other_graph"
    (tmp_path / "two.yaml").write_text(yaml.safe_dump(changed))
    base["benchmark"] = {"scenarios": ["one.yaml", "two.yaml"]}
    config = load_config(write_config(tmp_path, base))
    validate_config(config)
    assert config.episode_count == 2


def test_cli_help_and_list_commands():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    for command in ("list-providers", "list-benchmarks", "list-families", "list-graphs"):
        assert main([command]) == 0


def test_duplicate_yaml_keys_and_scale_cap(tmp_path, base):
    path = tmp_path / "bad.yaml"
    path.write_text("name: first\nname: second\n")
    with pytest.raises(ConfigError, match="Duplicate"):
        load_config(path)
    base["repetitions"] = 1001
    with pytest.raises(ConfigError, match="exceeds"):
        load_config(write_config(tmp_path, base))
