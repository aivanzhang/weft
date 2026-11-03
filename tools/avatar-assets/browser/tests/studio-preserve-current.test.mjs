import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { canPreserveCurrent, retryRequestBody } from "../studio.js";
import { retryOptions } from "./studio-retry-options.mjs";

const eligible = { status: "unsupported", capabilities: { retry: true, retry_preserve_current: true } };

test("keeping the current look requires an explicit supported owner choice", () => {
  assert.equal(canPreserveCurrent(eligible), true);
  assert.deepEqual(retryRequestBody(eligible), {});
  assert.deepEqual(retryRequestBody(eligible, true), { appearance_policy: "preserve_current" });
  assert.deepEqual(eligible.capabilities, { retry: true, retry_preserve_current: true });
  assert.throws(() => retryRequestBody(eligible, "preserve_current"), /Invalid retry mode/);
});

test("missing, advisory-false and nonterminal jobs cannot opt in", () => {
  for (const job of [null, {}, { status: "unsupported", capabilities: { retry: true } },
    { status: "unsupported", capabilities: { retry_preserve_current: "true" } },
    ...["complete", "running", "needs_inspection", "created"].map((status) => ({ ...eligible, status }))]) {
    assert.equal(canPreserveCurrent(job), false);
    assert.throws(() => retryRequestBody(job, true), /not available/);
  }
  assert.throws(() => retryRequestBody({ capabilities: { retry: false } }), /not available/);
});

test("ordinary retry and the explicit harness mode have disjoint exact payloads", () => {
  const args = ["outputs/analytic-fixture", "a".repeat(24)];
  assert.deepEqual(retryOptions(args), { folder: args[0], parent: args[1], preserveCurrent: false, button: "#retry-job", body: {} });
  assert.deepEqual(retryOptions([...args, "--preserve-current"]), { folder: args[0], parent: args[1], preserveCurrent: true,
    button: "#preserve-current-job", body: { appearance_policy: "preserve_current" } });
  for (const bad of [[], [args[0], "../parent"], [...args, "--skip-review"], [...args, "--preserve-current", "--budget=99"]]) {
    assert.throws(() => retryOptions(bad));
  }
});

test("new action starts hidden and explains preserved appearance without claiming acceptance", async () => {
  const html = await readFile(new URL("../studio.html", import.meta.url), "utf8");
  assert.match(html, /id="preserve-current-job"[^>]+aria-describedby="preserve-current-note"[^>]+hidden>Keep look; finish motion<\/button>/);
  const note = html.match(/id="preserve-current-note"[^>]*>([^<]+)<\/p>/)?.[1];
  assert.match(note, /fur and materials/); assert.match(note, /Rig and motion checks still apply/);
  assert.doesNotMatch(note, /human.approved|quality passed|automatically starts/i);
});
