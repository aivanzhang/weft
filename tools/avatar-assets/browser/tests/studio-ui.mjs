// Analytic API and renderer fixtures verify UI behavior, not generation or real GLB playback.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
assert.equal(process.argv.length, 3, "Usage: node browser/tests/studio-ui.mjs <new-output-folder>");
const out = path.resolve(root, process.argv[2]);
const outputs = await realpath(path.join(root, "outputs"));
assert.equal(await realpath(path.dirname(out)), outputs, "Use a new direct child of avatar-assets/outputs");
assert.notEqual(out, outputs);
await mkdir(out, { mode: 0o700 });
const sourceFiles = ["studio.html", "studio.css", "studio.js"];
const sources = new Map(await Promise.all(sourceFiles.map(async (name) => [name, await readFile(path.join(root, "browser", name))])));
const digest = (data) => createHash("sha256").update(data).digest("hex");
const origin = "http://127.0.0.1:43991";
const token = "analytic-studio-fixture-token";
function pngChunk(type, bytes) {
  const content = Buffer.concat([Buffer.from(type), bytes]); let crc = 0xffffffff;
  for (const byte of content) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0); }
  const length = Buffer.alloc(4), checksum = Buffer.alloc(4); length.writeUInt32BE(bytes.length); checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
  return Buffer.concat([length, content, checksum]);
}
const pngHeader = Buffer.alloc(13); pngHeader.writeUInt32BE(1, 0); pngHeader.writeUInt32BE(1, 4); pngHeader[8] = 8; pngHeader[9] = 6;
const png = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), pngChunk("IHDR", pngHeader),
  pngChunk("IDAT", deflateSync(Buffer.from([0, 220, 225, 222, 255]))), pngChunk("IEND", Buffer.alloc(0))]);
const report = { schema_version: 1, provenance: "analytic_fixture", status: "running",
  scope: "Mock API and mock model-viewer UI test. No generation, real GLB/video validation, provider call, or live-server proof.",
  limits: { wall_seconds: 120, requests: 200, output_bytes: 12 * 1024 * 1024, operation_ms: 10000 },
  sources: Object.fromEntries([...sources].map(([name, data]) => [name, digest(data)])), checks: [], errors: [], external_requests: [], artifacts: [], cleanup: {} };
let browserServer, browser, context, deadline, outputBytes = 0, requestCount = 0;
let createAttempts = 0, createFailure = true, uploads = 0, plans = 0, starts = 0, cancels = 0, gets = 0, launchingReads = 0;
const createIDs = [], jobs = new Map(), idempotency = new Map();
const jobURL = (id) => `/api/studio/jobs/${id}`;
const check = (name) => report.checks.push(name);
async function save(name, bytes) {
  assert(outputBytes + bytes.length <= report.limits.output_bytes, "Output bound exceeded");
  await writeFile(path.join(out, name), bytes, { flag: "wx", mode: 0o600 }); outputBytes += bytes.length;
  report.artifacts.push({ path: name, bytes: bytes.length, sha256: digest(bytes) });
}
const fixtureModule = `
class FixtureModel extends HTMLElement {
  static observedAttributes = ['src'];
  constructor() { super(); this.availableAnimations=['idle','dance']; this.paused=true; this.duration=4; this.currentTime=0; this.updateComplete=Promise.resolve(); }
  set src(value) { this.setAttribute('src', value); }
  get src() { return this.getAttribute('src'); }
  attributeChangedCallback(name, previous, value) { if(value) queueMicrotask(()=>this.dispatchEvent(new Event(value.includes('bad.glb')?'error':'load'))); }
  play() { this.paused=false; }
  pause() { this.paused=true; }
}
customElements.define('model-viewer', FixtureModel);
`;

try {
  browserServer = await chromium.launchServer({ executablePath: process.env.AVATAR_BROWSER_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true, timeout: 20000,
    args: ["--disable-background-networking", "--disable-component-update", "--disable-default-apps", "--no-first-run"] });
  deadline = setTimeout(() => { report.errors.push("90-second wall deadline exceeded"); browserServer.process().kill("SIGTERM"); }, 90000);
  browser = await chromium.connect(browserServer.wsEndpoint(), { timeout: 10000 });
  context = await browser.newContext({ viewport: { width: 1440, height: 1050 }, reducedMotion: "no-preference", serviceWorkers: "block" });
  context.setDefaultTimeout(10000);
  const page = await context.newPage();
  page.on("pageerror", (error) => { if (report.errors.length < 30) report.errors.push(error.message); });
  await context.route("**/*", async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (++requestCount > report.limits.requests) { report.errors.push("Request bound exceeded"); return route.abort(); }
    if (url.origin !== origin) { report.external_requests.push(url.origin); return route.abort(); }
    const pathname = url.pathname;
    const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (pathname === "/" || pathname === "/studio.html") return route.fulfill({ contentType: "text/html", body: sources.get("studio.html") });
    if (sources.has(pathname.slice(1))) return route.fulfill({ contentType: pathname.endsWith(".js") ? "text/javascript" : "text/css", body: sources.get(pathname.slice(1)) });
    if (pathname === "/favicon.svg") return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 30"><circle cx="15" cy="15" r="12" fill="#a6cbd6"/></svg>' });
    if (pathname === "/vendor/model-viewer.min.js") return route.fulfill({ contentType: "text/javascript", body: fixtureModule });
    if (pathname.startsWith("/api/studio/assets/") && pathname.endsWith(".png")) return route.fulfill({ contentType: "image/png", body: png });
    if (pathname.startsWith("/api/studio/assets/")) return route.fulfill({ status: 404, body: "Intentional fixture media failure" });
    if (request.method() === "POST") assert.equal(request.headers()["x-avatar-studio-token"], token, "Every mutation needs the local token");
    if (pathname === "/api/studio/session") return json({ csrf_token: token, presets: [{ id: "preview", label: "Preview", description: "Fixture sizes, not a generated output." }],
      limits: { prompt_chars: 4000, style_chars: 800, actions: 8, action_chars: 600, reference_bytes: 8 * 1024 * 1024 }, capabilities: { reference_upload: true },
      planning_cost_note: "Planning uses a bounded AI allowance. Account-specific dollar cost is unknown. No Meshy call is made while planning." });
    if (pathname === "/api/studio/references") { uploads++; assert.match(request.headers()["content-type"], /^multipart\/form-data; boundary=/); return json({ id: "reference-fixture", name: "reference.png", preview_url: "/api/studio/assets/reference.png" }); }
    if (pathname === "/api/studio/jobs" && request.method() === "GET") {
      gets++;
      // A stale list must not suppress polling of the selected launching job.
      return json({ jobs: [...jobs.values()].reverse().map((job) => job.status === "launching" ? { ...job, status: "planned" } : job) });
    }
    if (pathname === "/api/studio/jobs" && request.method() === "POST") {
      createAttempts++; const id = request.headers()["x-avatar-request-id"]; assert.match(id, /^[a-f0-9-]{36}$/); createIDs.push(id);
      if (!idempotency.has(id)) {
        const body = request.postDataJSON(); assert.equal(body.reference_id, "reference-fixture");
        const job = { id: `fixture-${jobs.size + 1}`, title: body.prompt, status: "created", request: { ...body, reference_selected: true },
          elapsed_seconds: 0, stages: [], plan: null, reviews: [], result: null, error: null, provenance: "analytic_fixture" };
        jobs.set(job.id, job); idempotency.set(id, job.id);
      }
      if (createFailure) { createFailure = false; return route.abort("failed"); }
      return json(jobs.get(idempotency.get(id)));
    }
    const match = /^\/api\/studio\/jobs\/([^/]+)(?:\/(plan|start|cancel))?$/.exec(pathname);
    if (match) {
      const job = jobs.get(match[1]); if (!job) return json({ error: { message: "Unknown fixture" } }, 404);
      if (request.method() === "GET") {
        gets++;
        if (job.status === "launching" && ++launchingReads >= 2) job.status = "running";
        return json(job);
      }
      if (match[2] === "plan") {
        plans++; Object.assign(job, { status: "awaiting_authorization", plan_id: "fixed-plan-1", plan: { summary: "Two requested actions, one character and saved editable sources.",
          estimate: { credits: 35, amount: null, currency: "USD", note: "No automatic paid retry.", breakdown: [{ label: "Geometry", credits: 25 }, { label: "Texture", credits: 10 }] },
          authorization: { can_start: false, reason: "Only 8 credits remain in the existing allowance. Starting requires a separate authorized budget decision." }, warnings: [] } });
      } else if (match[2] === "start") {
        starts++; assert.deepEqual(request.postDataJSON(), { plan_id: "fixed-plan-2" });
        Object.assign(job, { status: "launching", elapsed_seconds: 17, stages: [{ id: "author", label: "Author movement", status: "running", detail: "Working from your requested actions." }],
          reviews: [{ attempt: 1, status: "rejected", summary: "The first candidate did not visibly perform the second action. A bounded local repair is in progress." }] });
      } else if (match[2] === "cancel") { cancels++; job.status = "cancel_requested"; }
      return json(job);
    }
    return json({ error: { message: "Unknown fixture route" } }, 404);
  });
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => !document.getElementById("save-request").disabled);
  await page.locator("#prompt").fill('<img src=x onerror="window.injected=true"> Copper gardener');
  await page.locator("#actions").fill(Array(9).fill("Turn around").join("\n"));
  await page.locator("#save-request").click(); assert.equal(createAttempts, 0);
  assert.equal(await page.locator("#actions-error").isVisible(), true); check("More than eight actions rejected before a request");
  await page.locator("#actions").fill("Look around curiously\nLift a watering can and water a plant");
  await page.locator("#reference").setInputFiles({ name: "reference.png", mimeType: "image/png", buffer: png });
  await page.locator("#save-request").click(); await page.locator("#message").waitFor({ state: "visible" });
  assert.equal(createAttempts, 1); assert.equal(jobs.size, 1);
  await page.waitForFunction(() => !document.getElementById("save-request").disabled);
  await page.locator("#save-request").click(); await page.locator("#job-detail").waitFor({ state: "visible" });
  assert.deepEqual(createIDs, [createIDs[0], createIDs[0]]); assert.equal(uploads, 1); assert.equal(jobs.size, 1);
  assert.equal(await page.evaluate(() => window.injected), undefined); assert.equal(await page.locator("#job-title img").count(), 0);
  check("Ambiguous create response retained its idempotency ID; upload was not repeated; titles use text, not HTML");
  assert.equal(plans, 0); assert.equal(starts, 0);
  await page.locator("#prepare-plan").click(); await page.waitForFunction(() => document.getElementById("estimate-total").textContent === "35 credits");
  assert.equal(await page.locator("#start-job").isDisabled(), true); assert.equal(await page.locator("#estimate-ack").isDisabled(), true);
  await page.evaluate(() => document.getElementById("start-job").dispatchEvent(new MouseEvent("click")));
  assert.equal(starts, 0); assert.match(await page.locator("#authorization").textContent(), /8 credits/);
  await save("studio-cost-gate.png", await page.screenshot({ fullPage: true }));
  check("Concrete cost and insufficient allowance block start; UI does not grant spending approval");
  const job = jobs.get("fixture-1"); job.status = "planned"; job.plan_id = "fixed-plan-2"; job.plan.estimate.credits = 0;
  job.plan.authorization = { can_start: true, reason: "Fixture-only existing authorization with reused geometry." };
  await page.waitForFunction(() => !document.getElementById("refresh").disabled); await page.locator("#refresh").click();
  await page.waitForFunction(() => !document.getElementById("estimate-ack").disabled);
  assert.equal(await page.locator("#start-job").isDisabled(), true);
  await page.locator("#estimate-ack").check(); await page.locator("#start-job").click();
  await page.waitForFunction(() => document.getElementById("job-status").textContent === "Starting");
  assert.match(await page.locator("#progress-summary").textContent(), /Starting the local worker/);
  assert.equal(await page.locator("#cancel-job").isVisible(), true);
  await page.waitForFunction(() => document.getElementById("job-status").textContent === "Creating");
  assert.equal(launchingReads, 2); check("Launching continues polling with a stale planned list and reaches the actual running response");
  assert.equal(starts, 1); assert.equal(await page.locator("progress").count(), 0); assert.match(await page.locator("#reviews").textContent(), /bounded local repair/);
  check("Explicit start echoes the frozen plan; review failure is visible; no invented overall percentage");
  await page.evaluate(() => { Object.defineProperty(document, "hidden", { configurable: true, value: true }); document.dispatchEvent(new Event("visibilitychange")); });
  const hiddenGets = gets; await new Promise((resolve) => setTimeout(resolve, 2800)); assert.equal(gets, hiddenGets);
  await page.evaluate(() => { delete document.hidden; document.dispatchEvent(new Event("visibilitychange")); });
  await page.waitForFunction(() => !document.getElementById("cancel-job").disabled); await page.locator("#cancel-job").click();
  await page.waitForFunction(() => document.getElementById("job-status").textContent === "Stopping"); assert.equal(cancels, 1);
  check("Hidden-page polling pauses; stop remains a request rather than instant cancellation");
  job.status = "cancelled";
  const finished = { ...job, id: "fixture-finished", title: "Finished analytic preview", status: "complete", result: { portrait_url: "/api/studio/assets/portrait.png", model_url: "/api/studio/assets/model.glb",
    videos: [{ id: "idle", label: "Look around", url: "/api/studio/assets/invalid.mp4", loop_verified: false }],
    downloads: [{ label: "Editable source", url: "/api/studio/assets/source.blend", filename: "source.blend" }, { label: "Untrusted external URL", url: "https://external.invalid/secret" }],
    limitations: ["Analytic interface fixture, not an actual generated avatar."] } };
  jobs.set(finished.id, finished);
  await page.waitForFunction(() => !document.getElementById("refresh").disabled); await page.locator("#refresh").click();
  await page.locator(`[data-job="${finished.id}"]`).click(); await page.locator("#result-portrait").waitFor({ state: "visible" });
  assert.equal(await page.getByText("Untrusted external URL", { exact: false }).count(), 0);
  assert.equal(await page.locator("#fixture-notice").isVisible(), true);
  await save("studio-ready-fixture.png", await page.screenshot({ fullPage: true }));
  await page.locator('[data-view="model"]').click(); await page.waitForFunction(() => !document.getElementById("model-play").disabled);
  await page.locator("#model-play").click(); await page.waitForFunction(() => document.getElementById("model-play").textContent === "Pause");
  await page.locator("#model-play").click(); assert.equal(await page.locator("#model-play").textContent(), "Play once");
  await page.emulateMedia({ reducedMotion: "reduce" }); await page.locator("#result-portrait").waitFor({ state: "visible" });
  await page.locator('[data-view="video"]').click(); await page.waitForFunction(() => document.getElementById("preview-error").textContent.includes("Video preview could not load"));
  assert.equal(await page.locator("#result-portrait").isVisible(), true);
  check("Fixture model controls, reduced-motion portrait, invalid-video static fallback, and safe downloads work");
  finished.result.model_url = "/api/studio/assets/bad.glb";
  await page.waitForFunction(() => !document.getElementById("refresh").disabled); await page.locator("#refresh").click();
  await page.waitForFunction(() => !document.getElementById("refresh").disabled);
  await page.locator('[data-view="model"]').click(); await page.waitForFunction(() => document.getElementById("preview-error").textContent.includes("3D preview could not load"));
  assert.equal(await page.locator("#result-portrait").isVisible(), true); check("Fixture model failure falls back to portrait");
  await page.locator("#new-request").click(); await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await save("studio-mobile-fixture.png", await page.screenshot({ fullPage: true })); check("390px layout has no horizontal document overflow");
  assert.equal(report.errors.length, 0); assert.equal(report.external_requests.length, 0);
  report.requests = requestCount; report.api = { create_attempts: createAttempts, unique_jobs_created: idempotency.size, uploads, plans, starts, cancels, launching_reads: launchingReads };
  report.status = "passed";
} catch (error) {
  report.status = "failed"; report.errors.push(String(error.stack || error)); process.exitCode = 1;
} finally {
  clearTimeout(deadline);
  const boundedClose = (operation) => Promise.race([operation.catch(() => {}), new Promise((resolve) => setTimeout(resolve, 3000))]);
  if (context) await boundedClose(context.close());
  if (browser) await boundedClose(browser.close());
  if (browserServer) {
    await boundedClose(browserServer.close());
    const child = browserServer.process();
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM");
    report.cleanup.owned_browser_exited = child.exitCode !== null || child.signalCode !== null;
  }
  report.cleanup.server_started = false;
  await writeFile(path.join(out, "studio-ui-test.json"), JSON.stringify(report, null, 2) + "\n", { flag: "wx", mode: 0o600 });
  console.log(JSON.stringify({ status: report.status, report: path.relative(root, path.join(out, "studio-ui-test.json")), checks: report.checks.length, fixture_only: true }));
}
