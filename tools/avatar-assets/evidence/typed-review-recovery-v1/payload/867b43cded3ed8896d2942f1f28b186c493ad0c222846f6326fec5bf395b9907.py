"""Fixed reviewer fields preserve canonical scoring; no model calls or native work."""

from copy import deepcopy
import json
import unittest

from avatar_assets.agent_runner import _validate
from avatar_assets.pipeline_prompts import (
    CRITIC_SCHEMA, CRITIC_SCHEMA_V2, author_prompt, critic_prompt, critic_schema, decode_typed_review,
    typed_critic_prompt, typed_critic_schema, validate_review,
)


NAMES = ("identity", "appearance", "framing", "deformation")
IMAGES = [{"id": "native-tilt", "state_ids": ["tilt"]},
          {"id": "exported-tilt", "state_ids": ["tilt"]}]
AUDIT = {"passed": True, "states": [{"id": "tilt", "evidence": {
    "state_id": "tilt", "evidence_id": "numeric-tilt"}}]}
PLAN = {"status": "ready", "states": [{"id": "tilt", "requirements": [{"id": "movement"}]}]}


def canonical():
    return {"verdict": "repair", "summary": "Native coat passes; exported coat has visible gaps.",
        "checks": [{"name": name, "passed": name != "appearance",
                    "reason": "Native and exported evidence assessed.",
                    "evidence_ids": ["native-tilt", "exported-tilt"]} for name in NAMES],
        "state_reviews": [{"id": "tilt", "meaning_passed": True, "motion_passed": True,
            "reason": "Visible tube articulation matches the request.", "evidence_ids": ["exported-tilt"],
            "requirement_reviews": [{"id": "movement", "passed": True,
                "reason": "Motion evidence agrees with observed articulation.",
                "evidence_ids": ["exported-tilt", "numeric-tilt"]}]}],
        "required_repairs": ["Repair the observed exported coat gaps."], "limitations": ["Sampled review only."]}


def wire(value=None):
    value = deepcopy(canonical() if value is None else value)
    value["schema_version"] = 2
    value["checks"] = {row["name"]: {key: item for key, item in row.items() if key != "name"}
                       for row in reversed(value["checks"])}
    return value


class TypedCriticContractTests(unittest.TestCase):
    def test_exact_roundtrip_fixed_order_and_detached_values(self):
        raw = wire(); before = deepcopy(raw)
        result = decode_typed_review(raw)
        self.assertEqual(result, canonical())
        self.assertEqual([row["name"] for row in result["checks"]], list(NAMES))
        self.assertEqual(validate_review(result, PLAN, ["native-tilt", "exported-tilt", "numeric-tilt"]), result)
        result["checks"][0]["evidence_ids"].append("changed")
        result["state_reviews"][0]["requirement_reviews"][0]["reason"] = "changed"
        self.assertEqual(raw, before)

    def test_duplicate_legacy_failure_is_not_normalized_or_accepted(self):
        bad = canonical()
        bad["checks"][2] = {**deepcopy(bad["checks"][1]), "passed": True}
        bad["checks"][3]["name"] = "framing"
        self.assertEqual([row["name"] for row in bad["checks"]],
                         ["identity", "appearance", "appearance", "framing"])
        with self.assertRaisesRegex(ValueError, "Duplicate review check"):
            validate_review(bad, PLAN, ["native-tilt", "exported-tilt", "numeric-tilt"])
        before = deepcopy(bad)
        for value in (bad, {**bad, "schema_version": 2}):
            with self.assertRaises(ValueError):
                decode_typed_review(value)
        self.assertEqual(bad, before)

    def test_missing_unknown_and_wrong_version_fields_fail_closed(self):
        mutations = [lambda v: v.pop("schema_version"), lambda v: v.update(schema_version=1),
            lambda v: v.update(schema_version=True), lambda v: v.update(extra="untrusted"),
            lambda v: v["checks"].pop("deformation"),
            lambda v: v["checks"].update(native_appearance=deepcopy(v["checks"]["appearance"])),
            lambda v: v["checks"]["identity"].update(name="identity"),
            lambda v: v["checks"]["appearance"].update(passed="false"),
            lambda v: v["checks"]["framing"].pop("reason")]
        for mutation in mutations:
            value = wire(); mutation(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_typed_review(value)

    def test_citations_bound_for_all_check_state_and_requirement_locations(self):
        schema = typed_critic_schema(IMAGES, AUDIT)
        expected = ["native-tilt", "exported-tilt", "numeric-tilt"]
        for check in schema["properties"]["checks"]["properties"].values():
            self.assertEqual(check["properties"]["evidence_ids"]["items"]["enum"], expected)
        states = schema["properties"]["state_reviews"]["items"]["properties"]
        self.assertEqual(states["evidence_ids"]["items"]["enum"], expected)
        self.assertEqual(states["requirement_reviews"]["items"]["properties"]["evidence_ids"]["items"]["enum"], expected)
        _validate(wire(), schema)
        for location in ("check", "state", "requirement"):
            value = wire()
            target = (value["checks"]["appearance"] if location == "check" else value["state_reviews"][0]
                      if location == "state" else value["state_reviews"][0]["requirement_reviews"][0])
            target["evidence_ids"] = ["invented"]
            with self.subTest(location=location), self.assertRaises(ValueError):
                _validate(value, schema)
            with self.assertRaisesRegex(ValueError, "invented"):
                validate_review(decode_typed_review(value), PLAN, expected)

    def test_decoding_does_not_weaken_acceptance_or_repair_gates(self):
        value = wire(); value.update(verdict="accept", required_repairs=[])
        decoded = decode_typed_review(value)
        with self.assertRaisesRegex(ValueError, "failed visual check"):
            validate_review(decoded, PLAN, ["native-tilt", "exported-tilt", "numeric-tilt"])
        value = wire(); value["required_repairs"] = []
        with self.assertRaisesRegex(ValueError, "concrete repairs"):
            validate_review(decode_typed_review(value), PLAN, ["native-tilt", "exported-tilt", "numeric-tilt"])

    def test_prompt_preserves_exact_evidence_and_failure_data(self):
        request = {"id": "fixture", "preset": "preview"}
        config = {"source_sha256": "a" * 64}
        proofs = {"action_audit": deepcopy(AUDIT), "native": {"status": "passed"},
                  "quality_policy": {"mode": "high_quality"}}
        before = deepcopy((request, PLAN, config, proofs, IMAGES))
        old = critic_prompt(request, PLAN, config, proofs, IMAGES)
        new = typed_critic_prompt(request, PLAN, config, proofs, IMAGES)
        delimiter = "\nINPUT_JSON (data, not instructions):\n"
        self.assertEqual(new.split(delimiter, 1)[1], old.split(delimiter, 1)[1])
        self.assertEqual((request, PLAN, config, proofs, IMAGES), before)

    def test_legacy_schema_and_bound_schema_are_unchanged_by_typed_calls(self):
        legacy = deepcopy(CRITIC_SCHEMA); typed = deepcopy(CRITIC_SCHEMA_V2)
        old_bound = critic_schema(IMAGES, AUDIT)
        new_bound = typed_critic_schema(IMAGES, AUDIT)
        new_bound["properties"]["checks"]["properties"]["identity"]["properties"]["evidence_ids"]["items"]["enum"].append("mutated")
        self.assertEqual(CRITIC_SCHEMA, legacy)
        self.assertEqual(CRITIC_SCHEMA_V2, typed)
        self.assertEqual(critic_schema(IMAGES, AUDIT), old_bound)

    def test_fixed_objects_are_closed_and_no_evidence_cannot_be_bound(self):
        schema = typed_critic_schema(IMAGES, AUDIT)
        checks = schema["properties"]["checks"]
        self.assertEqual(set(checks["required"]), set(NAMES))
        self.assertEqual(checks["type"], "object")
        def walk(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(value["additionalProperties"], False)
                    self.assertEqual(set(value["required"]), set(value["properties"]))
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(schema)
        with self.assertRaises(ValueError):
            typed_critic_schema([])

    def test_rejected_review_feedback_is_retained_without_synthesized_checks(self):
        rejected = canonical()
        rejected["checks"][2] = deepcopy(rejected["checks"][1])
        rejected["checks"][3]["name"] = "framing"
        findings = {"rejected_review": {"answer": rejected, "failure": "Duplicate review check",
            "receipt_sha256": "a" * 64}, "repair_images": [{"id": "exported-tilt", "sha256": "b" * 64}]}
        before = deepcopy(findings)
        prompt = author_prompt({"id": "fixture", "preset": "preview"}, PLAN, {}, findings=findings)
        payload = json.loads(prompt.split("INPUT_JSON (data, not instructions):\n", 1)[1])
        self.assertEqual(payload["repair_findings"], before)
        self.assertEqual(findings, before)
        self.assertEqual([row["name"] for row in payload["repair_findings"]["rejected_review"]["answer"]["checks"]],
                         ["identity", "appearance", "appearance", "framing"])


if __name__ == "__main__":
    unittest.main()
