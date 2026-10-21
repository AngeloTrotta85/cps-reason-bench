from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cps_reason_bench.config import ConfigError, load_scenario
from cps_reason_bench.models.config import ModelConfig, build_policy
from cps_reason_bench.models.registry import (
    BackendBuildConfig,
    create_backend,
    register_backend,
)
from cps_reason_bench.runner import BenchmarkRunner
from cps_reason_bench.versions import (
    PROMPT_TEMPLATE_CONTENT_HASH,
    PROTOCOL_SCHEMA_VERSION,
)
from cps_reason_bench.policies.llm import _build_messages
from cps_reason_bench.protocol import DecisionConstraints
from cps_reason_bench.policies import PolicyRequest
from tests.custom_backend import CustomBackend


ROOT = Path(__file__).resolve().parents[1]


class BackendAndVersionTests(unittest.TestCase):
    def test_backend_registration_and_custom_import_loading(self) -> None:
        register_backend(
            "unit-custom",
            lambda config: CustomBackend(config.model),
            replace=True,
        )
        registered = create_backend(
            "unit-custom", BackendBuildConfig(model="registered-model")
        )
        imported = create_backend(
            "external-name",
            BackendBuildConfig(model="imported-model"),
            "tests.custom_backend:create_backend",
        )
        self.assertEqual("registered-model", registered.model)
        self.assertEqual("imported-model", imported.model)
        self.assertFalse(imported.capabilities().model_seed)

    def test_model_configuration_rejects_secret_values(self) -> None:
        with self.assertRaisesRegex(ConfigError, "forbidden"):
            ModelConfig.from_mapping(
                {
                    "name": "unsafe",
                    "backend": "ollama",
                    "model": "model",
                    "api_key": "must-not-be-stored",
                }
            )

    def test_prompt_protocol_and_schema_hashes_are_logged(self) -> None:
        model = ModelConfig.from_mapping(
            {
                "name": "custom-import",
                "backend": {
                    "name": "custom-test",
                    "import_path": "tests.custom_backend:create_backend",
                },
                "model": "fixed-json",
                "retry_policy": {"name": "strict_zero_correction", "max_corrections": 0},
            }
        )
        policy = build_policy(model, random_seed=1, requested_model_seed=None)
        scenario = load_scenario(ROOT / "examples" / "scenarios" / "prospective_basic.yaml")
        with tempfile.TemporaryDirectory() as directory:
            result = BenchmarkRunner(scenario, policy, directory).run()
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            interactions = [
                json.loads(line)
                for line in result.events_path.read_text(encoding="utf-8").splitlines()
                if '"record_type": "model_interaction"' in line
            ]
        versions = summary["benchmark_versions"]
        self.assertEqual(PROTOCOL_SCHEMA_VERSION, versions["protocol_schema_version"])
        self.assertEqual(PROMPT_TEMPLATE_CONTENT_HASH, versions["prompt_template_hash"])
        self.assertTrue(interactions[0]["rendered_prompt_hash"])
        self.assertTrue(interactions[0]["response_schema_hash"])
        self.assertEqual(1, summary["scenario_configuration"]["schema_version"])

    def test_prompt_v2_adds_protocol_detail_without_changing_v1_shape(self) -> None:
        scenario = load_scenario(
            ROOT / "examples" / "scenarios" / "planning_basic.yaml"
        )
        constraints = DecisionConstraints(
            allowed_action_ids=("restore_power",),
            action_parameter_schemas={"restore_power": {}},
            max_actions=1,
            current_step=0,
            current_time=0.0,
            max_steps=6,
            time_step=1.0,
            allows_callbacks=False,
            min_callback_steps=1,
            max_callback_steps=1,
            allows_subscriptions=False,
            allowed_event_types=(),
            active_subscription_ids=(),
            allowed_context_requests=("action_history",),
            reasoning_enabled=False,
        )
        request = PolicyRequest(scenario, {"observation": {}}, constraints, 0)
        v1 = json.loads(_build_messages(request, {})[1]["content"])
        v2 = json.loads(
            _build_messages(
                request,
                {},
                prompt_template_version="2.0.0",
            )[1]["content"]
        )
        self.assertNotIn(
            "action_parameter_schemas", v1["interaction_protocol"]
        )
        self.assertEqual(
            {"restore_power": {}},
            v2["interaction_protocol"]["action_parameter_schemas"],
        )


if __name__ == "__main__":
    unittest.main()
