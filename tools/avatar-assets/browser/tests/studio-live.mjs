// Real local Studio/job/asset verification. Only --start may submit one pinned Start.
import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const usage = "node browser/tests/studio-live.mjs <new-output-folder> <job-id> [--start --plan-id HASH] [--max-seconds 900]";
const args = process.argv.slice(2);
assert(args.length >= 2, usage);
const [outputArg, jobID] = args;
assert(/^[a-f0-9]{24}$/.test(jobID), "Require a canonical 24-character Studio job ID");
let startRequested = false, maxSeconds = 900, expectedPlanID = null;
const flags = new Set();
for (let index = 2; index < args.length; index++) {
  const flag = args[index]; assert(!flags.has(flag), `Duplicate option: ${flag}`); flags.add(flag);
  if (flag === "--start") startRequested = true;
  else if (flag === "--plan-id") expectedPlanID = args[++index];
  else if (flag === "--max-seconds") maxSeconds = Number(args[++index]);
  else assert.fail(usage);
}
assert(Number.isInteger(maxSeconds) && maxSeconds >= 30 && maxSeconds <= 1800, "Lifetime must be 30–1800 seconds");
assert(!expectedPlanID || /^[a-f0-9]{64}$/.test(expectedPlanID), "Invalid frozen plan ID");
assert(!startRequested || expectedPlanID, "--start requires the explicitly reviewed --plan-id");
const out = path.resolve(root, outputArg), outputs = await realpath(path.join(root, "outputs"));
assert.equal(await realpath(path.dirname(out)), outputs, "Use a new direct child of outputs/");
assert.notEqual(out, outputs);
const python = path.join(root, ".venv/bin/python"), helper = path.join(root, "scripts/verify_studio_live.py");
const executable = process.env.AVATAR_BROWSER_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
await Promise.all([access(python, constants.X_OK), access(executable, constants.X_OK), access(helper)]);
await mkdir(out, { mode: 0o700 });

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const execFileAsync = promisify(execFile);
const environment = Object.fromEntries(["HOME", "USER", "LOGNAME", "TMPDIR", "PATH", "LANG"]
  .filter((name) => process.env[name]).map((name) => [name, process.env[name]]));
Object.assign(environment, { PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" });
const localTime = () => new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", year: "numeric",
  month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  timeZoneName: "short" }).format(new Date());
const limits = { wall_seconds: maxSeconds, operation_ms: 15000, sampled_browser_rss_bytes: 3 * 1024 ** 3,
  output_bytes: 64 * 1024 ** 2, server_log_bytes: 1024 ** 2, json_bytes: 1024 ** 2,
  requests: maxSeconds + 256, snapshots: 128, events: 4096, event_bytes: 1024 ** 2 };
const started = performance.now();
const report = { schema_version: 1, status: "running", provenance: "actual_pipeline", recorded_local: localTime(),
  timezone: "America/Los_Angeles", job_id: jobID, explicit_start_requested: startRequested, limits,
  scope: "Actual local HTML/HTTP/canonical job manager, UI progress and real delivered GLB/video/static assets. Explicit reused geometry, not fresh text-to-mesh, visual quality, physical action, seamless-loop or sustained-FPS acceptance.",
  sources: [], snapshots: [], checks: [], requests: [], errors: [], external_requests: [], request_failures: [],
  mutations: [], assets: {}, artifacts: [], cleanup: {},
  resources: { sampled_peak_rss_bytes: 0, samples: 0, sampling_interval_ms: 2000,
    scope: "Tester Node + direct HTTP server + owned Chrome subtree; detached generation workers are EXCLUDED and keep their own bounds. RSS is sampled and shared pages may be double-counted." } };
let outputBytes = 0, eventBytes = 0, eventCount = 0, requestCount = 0;
let server, browserLaunch, browserServer, browser, context, page, origin, before, latest, latestSignature;
let serverLog = "", serverLogBytes = 0, deadline, monitorTimer, monitorTask, cancelled, cleanupStarted = false;
let responseQueue = Promise.resolve(), firstStartAt = null, allowStart = false, startCount = 0;
let backendInFlight = 0, maxBackendInFlight = 0;
const contexts = new Set(), sourceFiles = new Map();
const backendRequests = new Set(), pollTimes = [];
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let rejectCancellation;
const cancellation = new Promise((_, reject) => { rejectCancellation = reject; });
function cancel(reason) {
  if (cancelled || cleanupStarted) return;
  cancelled = new Error(reason); rejectCancellation(cancelled);
}
function running() { if (cancelled) throw cancelled; }
function record(list, value) {
  const size = Buffer.byteLength(JSON.stringify(value));
  if (++eventCount > limits.events || (eventBytes += size) > limits.event_bytes) return cancel("Observation bound exceeded");
  list.push(value);
}
async function bounded(work, milliseconds, label) {
  let timer;
  try { return await Promise.race([work, new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out`)), milliseconds);
  })]); } finally { clearTimeout(timer); }
}
async function save(name, value) {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  assert(outputBytes + bytes.length <= limits.output_bytes - 2 * 1024 ** 2, "Evidence output bound exceeded");
  await writeFile(path.join(out, name), bytes, { flag: "wx", mode: 0o600 }); outputBytes += bytes.length;
  report.artifacts.push({ path: name, bytes: bytes.length, sha256: digest(bytes) });
}
const saveJSON = (name, value) => save(name, JSON.stringify(value, null, 2) + "\n");
async function boundedJSON(filename) {
  const info = await lstat(filename);
  assert(info.isFile() && !info.isSymbolicLink() && info.size <= limits.json_bytes, "Invalid bounded JSON input");
  return JSON.parse(await readFile(filename, "utf8"));
}
async function ownership() {
  const { stdout } = await execFileAsync(python, [helper, "--job", jobID],
    { cwd: root, env: environment, timeout: 5000, maxBuffer: limits.json_bytes });
  return JSON.parse(stdout);
}
async function screenshot(name) {
  const bytes = await page.screenshot({ fullPage: true, timeout: 10000 });
  assert(bytes.length <= 8 * 1024 ** 2, "Screenshot exceeds 8 MiB");
  await save(name, bytes);
}
function appendLog(chunk) {
  const bytes = Buffer.from(chunk), remaining = Math.max(0, limits.server_log_bytes - serverLogBytes);
  serverLog += bytes.subarray(0, remaining).toString("utf8"); serverLogBytes += bytes.length;
  if (serverLogBytes > limits.server_log_bytes) cancel("HTTP server log bound exceeded");
}
async function startServer() {
  server = spawn(python, ["-u", helper, "--job", jobID, "--serve", "--max-seconds", String(maxSeconds + 15)],
    { cwd: root, env: environment, stdio: ["ignore", "pipe", "pipe"] });
  report.server_pid = server.pid;
  server.stderr.on("data", appendLog);
  return bounded(new Promise((resolve, reject) => {
    let pending = "", ready = false;
    server.stdout.on("data", (chunk) => {
      appendLog(chunk); if (ready) return;
      pending += chunk.toString("utf8");
      if (pending.length > 4096) return reject(new Error("Invalid server startup output"));
      const end = pending.indexOf("\n"); if (end < 0) return;
      try {
        const value = JSON.parse(pending.slice(0, end));
        assert(Number.isInteger(value.port) && value.port > 0 && value.port <= 65535);
        assert.equal(value.job_id, jobID); assert.equal(value.canonical_state_dir, before.state_dir);
        report.server = value; ready = true; resolve(`http://127.0.0.1:${value.port}`);
      } catch (error) { reject(error); }
    });
    server.once("error", reject);
    server.once("exit", (code, signal) => {
      const reason = `Owned Studio server exited (${code ?? signal})`;
      if (!ready) reject(new Error(reason)); else if (!cleanupStarted) cancel(reason);
    });
  }), 15000, "Studio startup");
}
async function sampleMemory() {
  const { stdout } = await execFileAsync("/bin/ps", ["-axo", "pid=,ppid=,rss="], { timeout: 2000, maxBuffer: 512 * 1024 });
  const children = new Map(), rss = new Map();
  for (const line of stdout.trim().split("\n")) {
    const [pid, parent, kib] = line.trim().split(/\s+/).map(Number);
    if (![pid, parent, kib].every(Number.isFinite)) continue;
    rss.set(pid, kib * 1024); if (!children.has(parent)) children.set(parent, []); children.get(parent).push(pid);
  }
  const measured = new Set([process.pid, server?.pid].filter(Boolean));
  const queue = browserServer ? [browserServer.process().pid] : [];
  for (const pid of queue) { if (measured.has(pid)) continue; measured.add(pid); queue.push(...(children.get(pid) || [])); }
  const total = [...measured].reduce((sum, pid) => sum + (rss.get(pid) || 0), 0);
  report.resources.samples++; report.resources.sampled_peak_rss_bytes = Math.max(report.resources.sampled_peak_rss_bytes, total);
  if (total > limits.sampled_browser_rss_bytes) cancel("Browser/server sampled RSS exceeded 3 GiB; worker will not be stopped");
}
async function observeResponse(response) {
  if (cleanupStarted || !origin) return;
  const url = new URL(response.url()), target = `/api/studio/jobs/${jobID}`;
  if (url.origin !== origin || ![target, `${target}/start`].includes(url.pathname)) return;
  const expectedStatus = response.request().method() === "POST" && url.pathname === `${target}/start` ? 202 : 200;
  assert.equal(response.status(), expectedStatus, `Actual job request failed: ${response.status()}`);
  const bytes = await response.body(); assert(bytes.length <= limits.json_bytes, "Job response exceeds 1 MiB");
  const value = JSON.parse(bytes);
  assert.equal(value.id, jobID); assert.equal(value.provenance, "actual_pipeline");
  for (const key of ["prompt", "style", "custom_actions", "preset"]) assert.deepEqual(value.request?.[key], before.request[key], `HTTP changed ${key}`);
  assert.equal(value.request.reuse_model_selected, true, "Geometry reuse must be disclosed");
  assert.equal(value.plan_id, before.plan_id, "Frozen plan changed");
  latest = value;
  const signature = JSON.stringify({ status: value.status, stages: value.stages, reviews: value.reviews,
    previews: value.previews, result: value.result, error: value.error, plan_id: value.plan_id });
  if (signature !== latestSignature) {
    assert(report.snapshots.length < limits.snapshots, "Snapshot bound exceeded"); latestSignature = signature;
    const name = `job-${String(report.snapshots.length).padStart(3, "0")}.json`;
    await saveJSON(name, value);
    report.snapshots.push({ path: name, status: value.status, observed_seconds: (performance.now() - started) / 1000,
      elapsed_seconds_measured: value.elapsed_seconds, stages: value.stages, reviews: value.reviews });
  }
}
async function newContext(reduced = false) {
  const active = await browser.newContext({ viewport: { width: 1440, height: 1050 }, deviceScaleFactor: 1,
    reducedMotion: reduced ? "reduce" : "no-preference", serviceWorkers: "block", acceptDownloads: false });
  contexts.add(active); active.setDefaultTimeout(limits.operation_ms);
  await active.route("**/*", async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (++requestCount > limits.requests) { cancel("Request bound exceeded"); return route.abort("blockedbyclient"); }
    if (url.origin !== origin) {
      if (["data:", "blob:"].includes(url.protocol)) return route.continue();
      record(report.external_requests, url.href.slice(0, 2000)); cancel("Unexpected external browser request"); return route.abort("blockedbyclient");
    }
    record(report.requests, { method: request.method(), path: url.pathname, at_seconds: (performance.now() - started) / 1000 });
    if (!["GET", "HEAD"].includes(request.method())) {
      const valid = startRequested && allowStart && request.method() === "POST" && url.pathname === `/api/studio/jobs/${jobID}/start` && startCount === 0;
      if (!valid) { cancel("Blocked unexpected or duplicate mutation"); return route.abort("blockedbyclient"); }
      const body = request.postDataJSON();
      if (Object.keys(body).length !== 1 || body.plan_id !== before.plan_id || !request.headers()["x-avatar-studio-token"]) {
        cancel("Start was not bound to the reviewed plan/session"); return route.abort("blockedbyclient");
      }
      startCount++; firstStartAt = performance.now(); allowStart = false;
      record(report.mutations, { method: "POST", path: url.pathname, plan_id: body.plan_id, csrf_present: true });
    }
    return route.continue();
  });
  await active.routeWebSocket("**/*", (socket) => {
    record(report.external_requests, socket.url().slice(0, 2000)); socket.close({ code: 1008, reason: "Local HTTP-only verification" }); cancel("Unexpected websocket");
  });
  active.on("page", (tab) => {
    tab.on("pageerror", (error) => { record(report.errors, { kind: "pageerror", message: error.message.slice(0, 4000) }); cancel("Page error"); });
    tab.on("console", (message) => { if (message.type() === "error") record(report.errors, { kind: "console", message: message.text().slice(0, 4000) }); });
    tab.on("crash", () => cancel("Owned page crashed"));
    tab.on("download", (download) => { download.cancel().catch(() => {}); cancel("Unexpected full browser download"); });
    tab.on("requestfailed", (request) => record(report.request_failures,
      { path: new URL(request.url()).pathname, reason: request.failure()?.errorText }));
  });
  return active;
}
async function selectJob(tab) {
  await tab.goto(origin, { waitUntil: "domcontentloaded", timeout: 15000 });
  await tab.locator(`[data-job="${jobID}"]`).click();
  await tab.locator("#job-detail").waitFor({ state: "visible" });
}
async function verifyUI(snapshot) {
  await page.waitForFunction((expected) => {
    const stages = [...document.querySelectorAll("#timeline > li")];
    return document.getElementById("job-status").dataset.status === expected.status
      && stages.length === expected.stages.length && stages.every((item, i) => item.dataset.status === expected.stages[i].status
        && item.querySelector("strong").textContent === expected.stages[i].label)
      && JSON.stringify([...document.querySelectorAll("#reviews .review p")].map((item) => item.textContent))
        === JSON.stringify(expected.reviews.map((item) => item.summary));
  }, { status: snapshot.status, stages: snapshot.stages, reviews: snapshot.reviews });
  assert.equal(await page.locator("#request-prompt").textContent(), before.request.prompt);
  assert.deepEqual(await page.locator("#requested-actions li").allTextContents(), before.request.custom_actions);
  const facts = await page.locator("#request-facts").evaluate((element) => Object.fromEntries(
    [...element.children].map((group) => [group.querySelector("dt").textContent, group.querySelector("dd").textContent])));
  assert.equal(facts["Look and materials"] || "", before.request.style); assert.equal(facts.Geometry, "Existing generated model");
  assert.equal(await page.locator("#fixture-notice").isVisible(), false);
  const timeline = await page.locator("#timeline > li").evaluateAll((items) => items.map((item) => ({
    status: item.dataset.status, label: item.querySelector("strong").textContent,
    meters: [...item.querySelectorAll("progress")].map((meter) => ({ value: meter.value, max: meter.max })) })));
  assert.equal(timeline.length, snapshot.stages.length);
  for (const [index, stage] of snapshot.stages.entries()) {
    assert.equal(timeline[index].status, stage.status); assert.equal(timeline[index].label, stage.label);
    const progress = stage.progress, measured = progress && Number.isFinite(progress.current) && progress.current >= 0
      && Number.isFinite(progress.total) && progress.total > 0 && progress.current <= progress.total;
    assert.deepEqual(timeline[index].meters, measured ? [{ value: progress.current, max: progress.total }] : []);
  }
  assert.equal(await page.locator("progress").count(), timeline.reduce((sum, item) => sum + item.meters.length, 0), "Unbound global percent meter");
  const reviews = await page.locator("#reviews .review p").allTextContents();
  assert.deepEqual(reviews, snapshot.reviews.map((review) => review.summary));
  if (snapshot.error?.message) assert.equal(await page.locator("#job-problem").textContent(), snapshot.error.message);
  const seconds = Math.floor(snapshot.elapsed_seconds);
  assert(Number.isFinite(seconds) && seconds >= 0, "Backend supplied no measured elapsed time");
  const elapsed = seconds < 60 ? `${seconds}s` : seconds < 3600 ? `${Math.floor(seconds / 60)}m ${seconds % 60}s`
    : `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
  assert.equal(await page.locator("#job-elapsed").textContent(), `${elapsed} elapsed`);
  return { status: snapshot.status, timeline_entries: timeline.length, reviews: reviews.length,
    elapsed_label: await page.locator("#job-elapsed").textContent() };
}
async function waitSnapshot(predicate) {
  while (!latest || !predicate(latest)) { running(); await pause(200); }
  await responseQueue; running(); return latest;
}
function opaque(value) {
  assert.equal(typeof value, "string");
  assert(new RegExp(`^/api/studio/jobs/${jobID}/files/[a-f0-9]{32}$`).test(value), "Asset URL is not an opaque same-job URL");
  return value;
}
async function verifyAssets(snapshot) {
  const result = snapshot.result; assert(result?.model_url && result.portrait_url, "Complete job lacks model/portrait");
  const states = (await boundedJSON(path.join(root, before.output, "plan.json"))).states.map((state) => state.id);
  assert(states.length >= 1 && states.length <= 8 && new Set(states).size === states.length);
  assert.deepEqual(result.videos.map((video) => video.id).sort(), [...states].sort(), "Rendered videos differ from the actual plan");
  opaque(result.model_url); opaque(result.portrait_url);
  await page.locator('[data-view="portrait"]').click();
  await page.waitForFunction((source) => { const image = document.getElementById("result-portrait");
    return !image.hidden && image.getAttribute("src") === source && image.complete && image.naturalWidth > 0; }, result.portrait_url);
  report.assets.portrait = await page.locator("#result-portrait").evaluate((image) => ({ width: image.naturalWidth, height: image.naturalHeight }));
  await screenshot("finished-portrait.png");
  await page.locator('[data-view="model"]').click();
  await page.waitForFunction(() => document.getElementById("result-model").loaded && !document.getElementById("model-play").disabled, null, { timeout: 30000 });
  const clips = await page.locator("#result-model").evaluate((model) => model.availableAnimations);
  assert.deepEqual([...clips].sort(), [...states].sort(), "Actual GLB animations differ from the plan");
  report.assets.animations = [];
  for (const state of states) {
    await page.locator("#clip").selectOption(state);
    await page.waitForFunction((name) => document.getElementById("result-model").animationName === name && !document.getElementById("model-play").disabled, state);
    const duration = await page.locator("#result-model").evaluate((model) => model.duration);
    assert(Number.isFinite(duration) && duration >= .25 && duration <= 60);
    assert(await page.locator("#result-model").evaluate((model) => model.paused));
    await page.locator("#model-play").click();
    await page.waitForFunction(() => document.getElementById("result-model").currentTime > .08 && !document.getElementById("result-model").paused);
    const advanced = await page.locator("#result-model").evaluate((model) => model.currentTime);
    await page.locator("#model-play").click();
    const paused = await page.locator("#result-model").evaluate((model) => model.currentTime);
    await pause(150); assert(Math.abs(await page.locator("#result-model").evaluate((model) => model.currentTime) - paused) < .025);
    await page.locator("#result-model").evaluate((model) => { model.currentTime = model.duration / 2; });
    const seek = await page.locator("#result-model").evaluate((model) => model.currentTime);
    assert(Math.abs(seek - duration / 2) < .05);
    report.assets.animations.push({ id: state, duration_seconds: duration, played_to_seconds: advanced, seek_seconds: seek,
      play_pause: "Actual UI buttons", seek: "Programmatic model-viewer control; no UI seek control is claimed" });
  }
  await screenshot("finished-model.png");
  report.assets.videos = [];
  for (const video of result.videos) {
    opaque(video.url); opaque(video.poster_url);
    await page.locator('[data-view="video"]').click(); await page.locator("#clip").selectOption(video.id);
    await page.waitForFunction((source) => { const v = document.getElementById("result-video");
      return v.getAttribute("src") === source && v.readyState >= 2 && v.videoWidth > 0; }, video.url);
    const info = await page.locator("#result-video").evaluate((v) => ({ width: v.videoWidth, height: v.videoHeight, duration: v.duration, controls: v.controls, paused: v.paused, loop: v.loop }));
    assert(info.controls && info.paused && !info.loop && info.duration >= .25 && info.duration <= 60);
    await page.locator("#result-video").evaluate((v) => v.play());
    await page.waitForFunction(() => document.getElementById("result-video").currentTime > .08);
    await page.locator("#result-video").evaluate((v) => { v.pause(); v.currentTime = v.duration / 2; });
    await page.waitForFunction(() => { const v = document.getElementById("result-video"); return v.paused && Math.abs(v.currentTime - v.duration / 2) < .05; });
    const fallback = await page.evaluate(async (source) => {
      const image = new Image(); image.src = source; await image.decode(); return { width: image.naturalWidth, height: image.naturalHeight };
    }, video.poster_url);
    assert(fallback.width > 0 && fallback.height > 0);
    report.assets.videos.push({ id: video.id, ...info, static_fallback: fallback, playback: "Actual HTML video API", seamless_loop_claimed: video.loop_verified === true });
  }
  await screenshot("finished-video.png");
  const links = await page.locator("#downloads a").evaluateAll((items) => items.map((a) => ({ label: a.textContent, url: a.getAttribute("href"), filename: a.download })));
  assert(links.some((link) => link.url === result.model_url)); assert(links.some((link) => link.url === result.portrait_url));
  assert(result.downloads.some((item) => item.filename?.endsWith(".blend")), "Editable source download is missing");
  report.assets.downloads = [];
  for (const link of links) {
    opaque(link.url);
    const proof = await page.evaluate(async (url) => {
      const head = await fetch(url, { method: "HEAD", redirect: "error", signal: AbortSignal.timeout(10000) });
      const part = await fetch(url, { headers: { Range: "bytes=0-3" }, redirect: "error", signal: AbortSignal.timeout(10000) });
      if (part.status !== 206 || part.headers.get("content-length") !== "4") { await part.body?.cancel(); throw new Error("Download range did not remain bounded to four bytes"); }
      const reader = part.body.getReader(), bytes = [];
      while (true) { const next = await reader.read(); if (next.done) break;
        if (bytes.length + next.value.length > 4) { await reader.cancel(); throw new Error("Download range byte bound exceeded"); } bytes.push(...next.value); }
      return { head_status: head.status, content_length: Number(head.headers.get("content-length")),
        content_type: head.headers.get("content-type"), content_disposition: head.headers.get("content-disposition"),
        range_status: part.status, content_range: part.headers.get("content-range"), first_four_hex: bytes.map((b) => b.toString(16).padStart(2, "0")).join(""), range_bytes: bytes.length };
    }, link.url);
    assert.equal(proof.head_status, 200); assert.equal(proof.range_status, 206); assert.equal(proof.range_bytes, 4);
    assert(proof.content_length > 0 && proof.content_length <= 512 * 1024 ** 2);
    report.assets.downloads.push({ ...link, ...proof, scope: "Metadata and first four bytes, not a copied full download" });
  }
  report.checks.push("Actual planned GLB clips play/pause/seek, rendered videos decode/play/seek, static images decode, editable/model/media download endpoints stream byte ranges");
  await page.locator('[data-view="portrait"]').click();
  const reduced = await newContext(true), reducedPage = await reduced.newPage();
  const fetchedMotion = [];
  reducedPage.on("request", (request) => { if ([result.model_url, ...result.videos.map((video) => video.url)].includes(new URL(request.url()).pathname)) fetchedMotion.push(request.url()); });
  await selectJob(reducedPage);
  await reducedPage.waitForFunction(() => { const image = document.getElementById("result-portrait"); return !image.hidden && image.complete && image.naturalWidth > 0; });
  assert.equal(await reducedPage.locator("#result-model").getAttribute("src"), null);
  assert.equal(await reducedPage.locator("#result-video").getAttribute("src"), null); assert.deepEqual(fetchedMotion, []);
  await save("reduced-motion.png", await reducedPage.screenshot({ fullPage: true, timeout: 10000 }));
  await reduced.close(); contexts.delete(reduced);
  report.checks.push("Real reduced-motion page selects the static portrait without fetching GLB/video");
  const delivery = path.join(root, before.output, "delivery.json");
  assert.equal((await boundedJSON(delivery)).status, "complete");
  report.delivery = { path: path.relative(root, delivery), sha256: digest(await readFile(delivery)) };
}
async function run() {
  const sourceNames = ["browser/tests/studio-live.mjs", "scripts/verify_studio_live.py", "browser/studio.html", "browser/studio.js", "browser/studio.css",
    "avatar_assets/studio.py", "avatar_assets/studio_jobs.py", "avatar_assets/studio_worker.py", "avatar_assets/pipeline.py",
    "avatar_assets/pipeline_prompts.py", "avatar_assets/agent_runner.py", "avatar_assets/action_intent.py", "avatar_assets/action_audit.py",
    "avatar_assets/quality_policy.py", "avatar_assets/pipeline_evidence.py"];
  for (const name of sourceNames) { const bytes = await readFile(path.join(root, name)); sourceFiles.set(name, digest(bytes)); report.sources.push({ path: name, sha256: digest(bytes) }); }
  await save("source-test.mjs", await readFile(fileURLToPath(import.meta.url)));
  await save("source-helper.py", await readFile(helper));
  before = await ownership(); report.initial_status = before.status; report.frozen_plan_id = before.plan_id;
  if (expectedPlanID) assert.equal(before.plan_id, expectedPlanID, "Canonical plan differs from the explicitly reviewed plan");
  await saveJSON("ownership-before.json", before);
  if (startRequested) assert.equal(before.status, "planned", "--start requires the already planned job; no retries or restart of running/terminal jobs");
  origin = await startServer(); running();
  browserLaunch = chromium.launchServer({ executablePath: executable, headless: true, host: "127.0.0.1", port: 0, timeout: 20000,
    handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false, env: environment,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "--disable-background-networking", "--disable-component-update", "--disable-sync", "--js-flags=--max-old-space-size=768"] })
    .then(async (handle) => { browserServer = handle; if (cleanupStarted) { await handle.kill(); throw new Error("Late browser launch"); } return handle; });
  browserServer = await browserLaunch; report.browser_pid = browserServer.process().pid;
  browser = await chromium.connect(browserServer.wsEndpoint(), { timeout: 10000 });
  report.browser = { version: browser.version(), executable, software_webgl: true };
  context = await newContext(); page = await context.newPage();
  page.on("response", (response) => { responseQueue = responseQueue.then(() => observeResponse(response)).catch((error) => cancel(error.message)); });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.origin === origin && /^\/api\/studio\/(session|jobs(?:\/[a-f0-9]{24})?)$/.test(url.pathname) && request.method() === "GET") {
      backendRequests.add(request); backendInFlight++; maxBackendInFlight = Math.max(maxBackendInFlight, backendInFlight);
      pollTimes.push(performance.now());
    }
  });
  const done = (request) => { if (backendRequests.delete(request)) backendInFlight--; };
  page.on("requestfinished", done); page.on("requestfailed", done);
  await selectJob(page); await waitSnapshot(() => true);
  report.initial_ui = await verifyUI(latest);
  assert.equal(await page.locator("#estimate-total").textContent(), "0 credits");
  assert.equal(latest.plan.estimate.credits, 0); assert.equal(latest.plan.estimate.amount, null);
  await screenshot("initial-job.png");
  report.checks.push("Real HTTP/UI preserve the saved exact prompt/style/actions and disclose source reuse, zero new Meshy credits and unknown account dollar cost");
  if (startRequested) {
    assert.equal(latest.status, "planned"); assert.equal(latest.plan.authorization.can_start, true);
    assert(await page.locator("#start-job").isDisabled(), "Start must require explicit estimate acknowledgement");
    await page.locator("#estimate-ack").check(); allowStart = true; await page.locator("#start-job").click();
    await waitSnapshot((job) => !["planned", "awaiting_authorization"].includes(job.status));
    assert.equal(startCount, 1); report.checks.push("One explicit UI Start sent the pinned plan_id with a session token; no plan/create/cancel/retry mutation");
  } else if (["created", "planned", "awaiting_authorization"].includes(latest.status)) {
    report.status = "observed_only"; report.result = "Saved plan observed without starting generation; no completed-output acceptance"; return;
  }
  const terminal = new Set(["complete", "unsupported", "failed", "needs_inspection", "cancelled"]);
  let lastUI = "", progressShots = 0;
  while (true) {
    running(); await responseQueue;
    const snapshot = latest, signature = latestSignature;
    if (signature !== lastUI) {
      report.last_ui = await verifyUI(snapshot); lastUI = signature;
      if (progressShots < 8) await screenshot(`progress-${String(progressShots++).padStart(2, "0")}.png`);
    }
    if (terminal.has(snapshot.status)) break;
    await pause(500);
  }
  report.terminal_status = latest.status; await saveJSON("terminal-job.json", latest);
  if (latest.status !== "complete") { report.status = latest.status === "unsupported" ? "unsupported" : "failed";
    report.result = "Actual worker stopped without a completed delivery; preserved attempts are not a pass"; return; }
  assert(latest.reviews.some((review) => review.status === "accept"), "Completed output has no real accepted reviewer receipt");
  await verifyAssets(latest);
  report.polling = { source: "Existing page polling only; verifier made no status polling requests", maximum_simultaneous_backend_gets: maxBackendInFlight };
  assert(maxBackendInFlight <= 1, "Overlapping UI backend polling");
  if (firstStartAt && performance.now() - firstStartAt > 20000) {
    const end = performance.now(), polls = pollTimes.filter((time) => time > firstStartAt + 10000);
    report.polling.steady_gets_per_second = polls.length / ((end - firstStartAt - 10000) / 1000);
    assert(report.polling.steady_gets_per_second <= 1.05, "Steady backend polling exceeded one request per second");
  }
  assert.equal(startCount, startRequested ? 1 : 0); assert.deepEqual(report.errors, []); assert.deepEqual(report.external_requests, []);
  report.status = "passed";
}
async function waitExit(child, timeout = 3000) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return true;
  try { await bounded(new Promise((resolve) => child.once("exit", resolve)), timeout, "Owned process exit"); } catch {}
  return child.exitCode !== null || child.signalCode !== null;
}
async function cleanup() {
  cleanupStarted = true; clearTimeout(deadline); clearInterval(monitorTimer);
  if (monitorTask) await bounded(monitorTask, 2500, "Monitor cleanup").catch(() => {});
  if (browserLaunch && !browserServer) await bounded(browserLaunch, 21000, "Browser launch cleanup").catch(() => {});
  await bounded(Promise.allSettled([...contexts].map((item) => item.close())), 3000, "Page cleanup").catch(() => {});
  await bounded(responseQueue, 3000, "Response observation cleanup").catch(() => {});
  if (browserServer) await bounded(browserServer.close(), 5000, "Browser shutdown")
    .catch(() => bounded(browserServer.kill(), 3000, "Owned browser kill").catch(() => {}));
  report.cleanup.browser_exit_verified = await waitExit(browserServer?.process());
  // Signal the direct HTTP process only: its detached pipeline must retain ownership.
  if (server && !await waitExit(server, 10)) { server.kill("SIGTERM"); if (!await waitExit(server)) server.kill("SIGKILL"); }
  report.cleanup.server_exit_verified = await waitExit(server);
  report.cleanup.worker_cancel_requested = false; report.cleanup.worker_signalled = false;
  try {
    const after = await ownership(); await saveJSON("ownership-after.json", after);
    report.worker_handoff = { status: after.status, output: after.output, owner: after.owner,
      next_action: after.owner.identity_verified_live ? "Worker remains owned and live. Observe its canonical job; do not start another copy or reset its lease."
        : "Inspect the canonical status and preserved receipts before any next action; this verifier issued no worker cancellation." };
  } catch (error) { report.worker_handoff = { status: "unavailable", reason: error.message, next_action: "Inspect canonical ownership; never infer the worker stopped" }; }
  if (!report.cleanup.browser_exit_verified || !report.cleanup.server_exit_verified) { report.status = "failed"; report.cleanup.manual_stop_required = true; }
}
const interrupt = () => cancel("Verifier interrupted; worker will not be cancelled");
process.once("SIGINT", interrupt); process.once("SIGTERM", interrupt);
deadline = setTimeout(() => cancel(`Verifier ${maxSeconds}-second deadline; worker will not be cancelled`), maxSeconds * 1000);
monitorTimer = setInterval(() => {
  if (!monitorTask && !cleanupStarted) monitorTask = sampleMemory().catch((error) => cancel(`RSS monitor failed: ${error.message}`)).finally(() => { monitorTask = null; });
}, 2000);
try { await Promise.race([run(), cancellation]); }
catch (error) { report.status = cancelled?.message.includes("deadline") ? "timed_out" : "failed"; report.error = String(error.stack || error).slice(0, 12000);
  if (page) await bounded(screenshot("failure.png"), 5000, "Failure screenshot").catch(() => {}); }
finally {
  await cleanup(); process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", interrupt);
  for (const [name, hash] of sourceFiles) if (digest(await readFile(path.join(root, name))) !== hash) {
    report.status = "failed"; report.error = "Verifier or live backend source changed during the run; no frozen-source acceptance";
  }
  if (report.errors.length || report.external_requests.length) { report.status = "failed"; report.error ||= "Browser errors were recorded"; }
  report.start_requests = startCount; report.request_count = requestCount;
  report.observed_rejected_reviews = [...new Map(report.snapshots.flatMap((item) => item.reviews || [])
    .filter((item) => ["repair", "rejected", "unsupported"].includes(item.status)).map((item) => [item.attempt, item])).values()];
  report.elapsed_seconds = (performance.now() - started) / 1000; report.finished_local = localTime();
  await save("server.log", serverLog);
  const bytes = Buffer.from(JSON.stringify(report, null, 2) + "\n"); assert(bytes.length <= 2 * 1024 ** 2 && outputBytes + bytes.length <= limits.output_bytes);
  await writeFile(path.join(out, "studio-live-test.json"), bytes, { flag: "wx", mode: 0o600 });
  console.log(JSON.stringify({ status: report.status, output: out, job_id: jobID, start_requests: startCount,
    elapsed_seconds: report.elapsed_seconds, cleanup: report.cleanup, worker_handoff: report.worker_handoff }));
  process.exitCode = ["passed", "observed_only"].includes(report.status) ? 0 : 1;
}
