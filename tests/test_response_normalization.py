from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cps_reason_bench.config import ConfigError, load_scenario
from cps_reason_bench.models.base import BackendError, ModelBackend, ModelResponse
from cps_reason_bench.models.catalog import load_model_catalog
from cps_reason_bench.models.config import ModelConfig
from cps_reason_bench.policies.base import PolicyRequest
from cps_reason_bench.policies.llm import LLMPolicy
from cps_reason_bench.protocol import DecisionConstraints, parse_decision
from cps_reason_bench.response_normalization import (
    STRICT_FENCE, DIAGNOSTIC_EXTRACTION, normalize_response, response_hash,
)
from cps_reason_bench.runner import BenchmarkRunner


ROOT = Path(__file__).resolve().parents[1]
PLAIN = '{"actions":[{"action_id":"cool_chamber"}],"callback":null,"subscriptions":[],"context_requests":[]}'


class FixedBackend(ModelBackend):
    provider = "offline-test"
    model = "fixed-response"

    def __init__(self, text=PLAIN):
        self.text = text
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return ModelResponse(self.text, {"prompt_tokens": 10, "completion_tokens": 10}, .01)


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.scenario = load_scenario(ROOT / "examples/scenarios/protocol_sanity/direct_action.yaml")
        self.constraints = DecisionConstraints(
            ("cool_chamber", "wait"), {"cool_chamber": {}, "wait": {}}, 1,
            0, 0, 1, 1, False, 1, 1, False, (), (), (), False,
        )

    def test_plain_json_and_whitespace_are_byte_identical(self):
        raw = " \r\n" + PLAIN + " \n"
        result = normalize_response(raw, STRICT_FENCE)
        self.assertEqual(raw, result.normalized_response)
        self.assertFalse(result.applied)

    def test_json_and_generic_fences_are_removed(self):
        for label in ("json", ""):
            with self.subTest(label=label):
                result = normalize_response(f"```{label}\n{PLAIN}\n```", STRICT_FENCE)
                self.assertEqual(PLAIN, result.normalized_response)
                self.assertTrue(result.applied)

    def test_surrounding_whitespace_and_crlf(self):
        raw = " \t\r\n  ```json\r\n \t" + PLAIN + "\r\n  ``` \r\n\t"
        result = normalize_response(raw, STRICT_FENCE)
        self.assertEqual(PLAIN, result.normalized_response)
        self.assertTrue(result.success)

    def test_interior_representation_is_preserved(self):
        payload = '{\r\n "value": -1.20e+03, "text": "\\u00e9", "values": [2, 1]\r\n}'
        result = normalize_response("```json\n " + payload + " \n```", STRICT_FENCE)
        self.assertEqual(payload.encode(), result.normalized_response.encode())

    def test_malformed_json_remains_invalid(self):
        result = normalize_response('```json\n{"actions":[],}\n```', STRICT_FENCE)
        parsed = parse_decision(result.normalized_response, self.constraints)
        self.assertTrue(result.applied)
        self.assertFalse(parsed.json_valid)

    def test_schema_error_remains_invalid(self):
        result = normalize_response('```json\n{}\n```', STRICT_FENCE)
        parsed = parse_decision(result.normalized_response, self.constraints)
        self.assertTrue(parsed.json_valid)
        self.assertFalse(parsed.schema_valid)

    def test_unknown_action_remains_invalid(self):
        raw = PLAIN.replace("cool_chamber", "invented_action")
        result = normalize_response("```json\n" + raw + "\n```", STRICT_FENCE)
        parsed = parse_decision(result.normalized_response, self.constraints)
        self.assertFalse(parsed.valid)
        self.assertIn("invented_action", " ".join(parsed.errors))

    def test_multiple_nested_and_unbalanced_fences_are_rejected(self):
        for raw in (
            f"```json\n{PLAIN}\n```\n```json\n{PLAIN}\n```",
            f"```json\n```json\n{PLAIN}\n```\n```",
            f"```json\n{PLAIN}",
        ):
            with self.subTest(raw=raw):
                result = normalize_response(raw, STRICT_FENCE)
                self.assertFalse(result.success)
                self.assertEqual(raw, result.normalized_response)

    def test_unsupported_fence_languages_and_lengths(self):
        for opening, closing in (("```python", "```"), ("````json", "````"), ("```JSON", "```")):
            with self.subTest(opening=opening):
                result = normalize_response(f"{opening}\n{PLAIN}\n{closing}", STRICT_FENCE)
                self.assertFalse(result.applied)

    def test_policy_a_preserves_outside_prose_and_policy_b_extracts(self):
        for raw in (f"```json\n{PLAIN}\n```\nExplanation.", f"Explanation.\n```json\n{PLAIN}\n```"):
            with self.subTest(raw=raw):
                a = normalize_response(raw, STRICT_FENCE)
                b = normalize_response(raw, DIAGNOSTIC_EXTRACTION)
                self.assertEqual("outside_prose", a.reason)
                self.assertEqual(raw, a.normalized_response)
                self.assertEqual(PLAIN, b.normalized_response)
                self.assertTrue(b.applied)

    def test_policy_b_does_not_repair_invalid_json_or_choose_multiple_blocks(self):
        for raw in ('```json\n{"a":}\n```\nProse', f"```json\n{PLAIN}\n```\n```\n{{}}\n```"):
            self.assertFalse(normalize_response(raw, DIAGNOSTIC_EXTRACTION).applied)

    def test_policy_b_not_selectable_for_live_models(self):
        with self.assertRaises(ConfigError):
            ModelConfig.from_mapping({"name": "test", "backend": "ollama", "response_normalization": {"mode": DIAGNOSTIC_EXTRACTION}})

    def test_strict_default_and_explicit_none_keep_model_identity(self):
        data = {"name": "test", "backend": "ollama", "model": "test"}
        strict = ModelConfig.from_mapping(data)
        explicit = ModelConfig.from_mapping({**data, "response_normalization": {"mode": "none"}})
        self.assertEqual(strict.identity, explicit.identity)
        self.assertNotIn("response_normalization", strict.safe_dict())

    def test_normalized_model_has_distinct_identity(self):
        strict = ModelConfig.from_mapping({"name": "test", "backend": "ollama"})
        normalized = ModelConfig.from_mapping({**strict.safe_dict(), "response_normalization": {"mode": STRICT_FENCE}})
        self.assertNotEqual(strict.identity, normalized.identity)

    def test_normalization_cannot_enable_output_corrections(self):
        with self.assertRaises(ConfigError):
            LLMPolicy(FixedBackend(), max_output_retries=1, response_normalization={"mode": STRICT_FENCE})

    def test_one_model_call_one_normalization_and_unchanged_prompt(self):
        backend = FixedBackend("```json\n" + PLAIN + "\n```\nOutside prose")
        request = PolicyRequest(self.scenario, {}, self.constraints, 0)
        strict = LLMPolicy(backend).decide(request)
        with patch("cps_reason_bench.policies.llm.normalize_response", wraps=normalize_response) as normalizer:
            result = LLMPolicy(backend, response_normalization={"mode": STRICT_FENCE}).decide(request)
        normalizer.assert_called_once()
        self.assertEqual(2, backend.calls)
        self.assertEqual(strict.prompt, result.prompt)
        self.assertEqual(strict.response_schema_hash, result.response_schema_hash)
        self.assertFalse(result.valid)
        self.assertEqual(1, len(result.attempts))

    def test_raw_and_normalized_hashes_and_executed_action_logged(self):
        raw = "```json\n" + PLAIN + "\n```"
        policy = LLMPolicy(FixedBackend(raw), response_normalization={"mode": STRICT_FENCE})
        with tempfile.TemporaryDirectory() as directory:
            run = BenchmarkRunner(self.scenario, policy, directory).run()
            events = [json.loads(line) for line in run.events_path.read_text().splitlines()]
            event = next(item for item in events if item["record_type"] == "model_interaction")
            summary = json.loads(run.summary_path.read_text())
        self.assertEqual(raw, event["raw_response"])
        self.assertEqual(PLAIN, event["normalized_response"])
        self.assertEqual(response_hash(raw), event["raw_response_hash"])
        self.assertEqual(response_hash(PLAIN), event["normalized_response_hash"])
        self.assertEqual(event["normalized_response_hash"], event["output_attempts"][0]["normalized_response_hash"])
        self.assertEqual(1, run.metrics["general"]["accepted_actions"])
        self.assertTrue(run.metrics["general"]["task_success"])
        self.assertEqual("markdown_fence_normalized", summary["run_metadata"]["interface_condition"])

    def test_strict_fenced_output_still_never_reaches_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            run = BenchmarkRunner(self.scenario, LLMPolicy(FixedBackend("```json\n" + PLAIN + "\n```")), directory).run()
        self.assertEqual(0, run.metrics["general"]["accepted_actions"])
        self.assertEqual(0, run.metrics["protocol_compliance"]["first_attempt_decision_valid_rate"])

    def test_action_preconditions_are_still_checked_by_simulator(self):
        raw = self.scenario.to_mapping()
        raw.pop("difficulty_profile", None)
        raw["actions"][0]["preconditions"] = [{"target": "chamber.temperature", "operator": "le", "value": 1}]
        scenario = type(self.scenario).from_mapping(raw)
        with tempfile.TemporaryDirectory() as directory:
            run = BenchmarkRunner(scenario, LLMPolicy(FixedBackend("```json\n" + PLAIN + "\n```"), response_normalization={"mode": STRICT_FENCE}), directory).run()
        self.assertEqual(0, run.metrics["general"]["accepted_actions"])
        self.assertGreater(run.metrics["general"]["invalid_actions"], 0)

    def test_registry_roundtrip_preserves_normalization_and_legacy_registry(self):
        strict = load_model_catalog(ROOT / "tests/fixtures/model_registry.yaml")
        before = strict.safe_dict()
        data = {"model_registry_schema_version": 1, "models": {
            "test": {"backend": "ollama", "model": "test", "response_normalization": {"mode": STRICT_FENCE}}
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.json"
            path.write_text(json.dumps(data))
            catalog = load_model_catalog(path)
            path.write_text(json.dumps(catalog.safe_dict()))
            restored = load_model_catalog(path)
        self.assertEqual(catalog.safe_dict(), restored.safe_dict())
        self.assertEqual({"mode": STRICT_FENCE}, restored.entries["test"].config.response_normalization)
        self.assertEqual(before, load_model_catalog(ROOT / "tests/fixtures/model_registry.yaml").safe_dict())

    def test_backend_failure_records_unavailable_normalized_response(self):
        backend = FixedBackend()
        with patch.object(backend, "generate", side_effect=BackendError("offline failure")):
            result = LLMPolicy(backend, response_normalization={"mode": STRICT_FENCE}).decide(PolicyRequest(self.scenario, {}, self.constraints, 0))
        self.assertFalse(result.valid)
        self.assertEqual("no_response_backend_error", result.response_processing["normalization_reason"])
        self.assertIsNone(result.response_processing["normalized_response"])
        self.assertEqual("1.0.0", result.response_processing["normalization_version"])
        self.assertEqual(64, len(result.response_processing["normalization_rule_hash"]))


if __name__ == "__main__":
    unittest.main()
