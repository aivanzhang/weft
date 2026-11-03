import test from "node:test";
import assert from "node:assert/strict";
import { aiFailureDisplay, canStart } from "../studio.js";

test("unknown charge is explicit without offering an automatic retry", () => {
  const job = { status: "needs_inspection", ai_failure: { code: "ai_charge_unknown", verified: true,
    automatic_retry: false, message: "AI request stopped after 240 seconds. Charge unknown; no automatic retry." } };
  assert.equal(aiFailureDisplay(job).chargeUnknown, true);
  assert.match(aiFailureDisplay(job).summary, /charge remains unknown/);
  assert.equal(canStart(job), false);
});

test("settled refusal and invalid response never show unknown-charge text", () => {
  for (const code of ["ai_declined", "ai_response_unusable", "ai_plan_incomplete"]) {
    const result = aiFailureDisplay({ status: "unsupported", ai_failure: { code, verified: true, automatic_retry: false, message: "Saved work preserved." } });
    assert.equal(result.chargeUnknown, false); assert.doesNotMatch(result.summary, /unknown/);
  }
});

test("old jobs, active jobs and unverified claims preserve the fallback display", () => {
  assert.equal(aiFailureDisplay({ status: "failed" }), null);
  assert.equal(aiFailureDisplay({ status: "running", ai_failure: { code: "ai_charge_unknown", verified: true, automatic_retry: false } }), null);
  assert.equal(aiFailureDisplay({ status: "failed", ai_failure: { code: "ai_charge_unknown", verified: false, automatic_retry: false } }), null);
  const result = aiFailureDisplay({ status: "failed", ai_failure: { code: "ai_status_unverified", verified: false, automatic_retry: false, message: "Inspect preserved records." } });
  assert.match(result.summary, /records need inspection/);
});
