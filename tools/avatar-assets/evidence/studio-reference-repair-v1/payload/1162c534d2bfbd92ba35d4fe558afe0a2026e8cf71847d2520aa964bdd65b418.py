"""Bounded prompt views preserve source evidence; no provider or native execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from avatar_assets.pipeline_prompts import _repair_prompt_view, author_prompt, _payload
from avatar_assets.agent_runner import MAX_PROMPT_BYTES


class RepairPromptViewTests(unittest.TestCase):
    def test_native_counts_and_current_defects_are_unchanged(self):
        findings = {"validation_failure": "Undercoat selected face count is outside its declared bounds",
            "measurements": {"native_failure": {"report": {"message": "Undercoat selection failed",
                "measurements": {"selected_faces": 35758, "min_faces": 18000, "max_faces": 32000, "total_faces": 36124}},
                "receipt": {"path": "native-failure.json", "sha256": "a" * 64}},
                "author_geometry": {"failures": ["left arm has too few vertices"]}}}
        original = deepcopy(findings)
        self.assertEqual(_repair_prompt_view(findings), findings)
        self.assertEqual(findings, original)

    def test_historical_rejections_keep_errors_and_hashes_not_duplicate_answers(self):
        answer = {"status": "ready", "design": {"test": [1, 2, 3]}}
        findings = {"later_rejected_authors": [{"role": "author-2", "answer": answer,
            "validation_failure": "Target outside root hierarchy", "answer_sha256": "b" * 64,
            "receipt_sha256": "c" * 64}]}
        original = deepcopy(findings)
        result = _repair_prompt_view(findings, continuation=True)["later_rejected_authors"][0]
        self.assertNotIn("answer", result)
        self.assertEqual(result["validation_failure"], "Target outside root hierarchy")
        self.assertEqual(result["answer_sha256"], "b" * 64)
        raw = json.dumps(answer, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
        self.assertEqual(result["omitted_answer"]["canonical_json_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(result["omitted_answer"]["reference"], "continuation/context.json#/findings/later_rejected_authors/0/answer")
        self.assertEqual(findings, original)

    def test_only_superseded_audit_details_are_omitted(self):
        state = {"id": "dance", "evidence": {"roles": {"body": {"max_travel_m": .08}}},
            "requirement_results": {"passed": False, "requirements": [{"id": "old_stillness", "status": "failed"}]},
            "role_details": {"body": {"outliers": [1, 2, 3]}}}
        audit = {"states": [state], "passed": False, "model_sha256": "d" * 64}
        error = "Requested action capability failed: " + json.dumps(audit["states"])
        findings = {"validation_failure": error, "measurements": {"action_audit": audit}}
        active = _repair_prompt_view(findings)
        self.assertEqual(active["measurements"]["action_audit"], audit)
        self.assertEqual(active["validation_failure_reference"]["sha256_utf8"], hashlib.sha256(error.encode()).hexdigest())
        historical = _repair_prompt_view({**findings, "semantic_replan": {"reason": "Unrequested stillness"}})
        summary = historical["measurements"]["action_audit"]
        self.assertEqual(summary["states"][0]["evidence"], state["evidence"])
        self.assertEqual(summary["states"][0]["requirement_results"], state["requirement_results"])
        self.assertNotIn("role_details", summary["states"][0])
        self.assertIn("not active requirements", summary["historical_scope"])
        self.assertEqual(_repair_prompt_view({"validation_failure": "Unrecognized error"}), {"validation_failure": "Unrecognized error"})

    def test_actual_b082_author_prompt_fits_without_mutating_saved_context(self):
        root = Path(__file__).resolve().parents[1] / "outputs/studio-b082dc54dbb54f9d9030f75b"
        if not root.exists():
            self.skipTest("Retained b082 local evidence is unavailable")
        files = [root / name for name in ("request.json", "plan.json", "inspection/manifest.json", "continuation/context.json")]
        before = [path.read_bytes() for path in files]
        request, plan, inspected, context = map(json.loads, before)
        findings = {**context["findings"], "semantic_replan": {
            "validation_failure": context["replan"]["validation_failure"],
            "previous_plan_sha256": "a" * 64, "current_plan_sha256": "b" * 64,
            "prior_measurement_scope": "Preserved failures against the superseded invalid plan; not current acceptance. "
                                       "Current-plan independent motion audits must run again."}}
        prompt = author_prompt(request, plan, {"preflight": inspected["preflight"], "native": inspected["native"]},
                               context["previous_answer"], findings)
        self.assertLess(len(prompt.encode()) + 1024, MAX_PROMPT_BYTES)
        payload = json.loads(prompt.split("INPUT_JSON (data, not instructions):\n", 1)[1])
        self.assertEqual(payload["previous_design"], context["previous_answer"])
        self.assertEqual([path.read_bytes() for path in files], before)
        self.assertEqual(prompt, author_prompt(request, plan, {"preflight": inspected["preflight"], "native": inspected["native"]},
                                              context["previous_answer"], findings))

    def test_compact_payload_preserves_every_value(self):
        value = {"text": "Fur 🐼\nKeep spaces inside text", "numbers": [0, .125, -3],
                 "nested": {"optional": None, "flags": [True, False], "rows": [{"x": i} for i in range(50)]}}
        payload = _payload(value).split("INPUT_JSON (data, not instructions):\n", 1)[1]
        self.assertEqual(json.loads(payload), value)
        self.assertLess(len(payload.encode()), len(json.dumps(value, indent=2, ensure_ascii=False).encode()))

    def test_actual_visual_continuation_fits_runner_not_only_payload_limit(self):
        root = Path(__file__).resolve().parents[1] / "outputs/studio-d8a680ce6580429f9fdd016f"
        if not root.exists():
            self.skipTest("Retained d8 evidence is unavailable")
        files = [root / name for name in ("request.json", "plan.json", "inspection/manifest.json", "continuation/context.json")]
        before = [path.read_bytes() for path in files]
        request, plan, inspected, context = map(json.loads, before)
        prompt = author_prompt(request, plan, {"preflight": inspected["preflight"], "native": inspected["native"]},
                               context["previous_answer"], context["findings"])
        self.assertLess(len(prompt.encode()) + 1024, MAX_PROMPT_BYTES)
        payload = json.loads(prompt.split("INPUT_JSON (data, not instructions):\n", 1)[1])
        self.assertEqual(payload["previous_design"], context["previous_answer"])
        self.assertEqual(payload["repair_findings"], _repair_prompt_view(context["findings"], continuation=True))
        self.assertEqual([path.read_bytes() for path in files], before)


if __name__ == "__main__":
    unittest.main()
