import assert from "node:assert/strict";
import test from "node:test";
import { observePreparation } from "./studio-prepare-observation.mjs";

test("old awaiting DOM cannot finish observation before the accepted POST and new worker", async () => {
  const oldDOM = { status: "awaiting_authorization" };
  let finishPost, reads = 0, clicks = 0, saved;
  const accepted = new Promise((resolve) => { finishPost = resolve; });
  const statuses = ["launching", "running", "planned"];
  const observation = observePreparation({
    receivePost: () => accepted,
    click: async () => { clicks++; assert.equal(oldDOM.status, "awaiting_authorization"); },
    readJob: async () => { reads++; return { status: statuses.shift() }; },
    savePost: async (value) => { saved = value; }, pause: async () => {},
  });
  await Promise.resolve(); await Promise.resolve();
  assert.equal(reads, 0);
  finishPost({ ok: true, status: 202, body: { status: "queued" } });
  assert.equal((await observation).status, "planned");
  assert.equal(reads, 3); assert.equal(clicks, 1); assert.equal(saved.status, 202);
});

test("a new post-submit authorization refusal is preserved, not retried", async () => {
  let clicks = 0;
  const result = await observePreparation({
    receivePost: async () => ({ ok: true, status: 202, body: { status: "queued" } }),
    click: async () => { clicks++; }, readJob: async () => ({ status: "awaiting_authorization", error: "Exact grant missing" }),
    savePost: async () => {}, pause: async () => {},
  });
  assert.equal(result.error, "Exact grant missing"); assert.equal(clicks, 1);
});

test("failed mutation and bounded timeout never start a second preparation", async () => {
  let reads = 0, clicks = 0, clock = 0;
  const args = { receivePost: async () => ({ ok: false, status: 409 }), click: async () => { clicks++; },
    readJob: async () => { reads++; return { status: "running" }; }, savePost: async () => {},
    pause: async () => { clock += 250; }, now: () => clock, maxMs: 500 };
  await assert.rejects(observePreparation(args), /Prepare request failed/);
  assert.equal(reads, 0); assert.equal(clicks, 1);
  await assert.rejects(observePreparation({ ...args, receivePost: async () => ({ ok: true, status: 202 }) }), /preserve the detached worker/);
  assert.equal(reads, 2); assert.equal(clicks, 2);
});
