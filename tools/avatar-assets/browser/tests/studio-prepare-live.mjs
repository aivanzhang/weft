// Prepare one already-saved request through the real website; never click Start.
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile, realpath } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium } from "playwright-core";
import { observePreparation } from "./studio-prepare-observation.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
const [folder, parent] = process.argv.slice(2);
assert(folder && /^[a-f0-9]{24}$/.test(parent), "Require new output folder and saved job ID");
const out = path.resolve(root, folder);
assert.equal(await realpath(path.dirname(out)), await realpath(path.join(root, "outputs")));
await mkdir(out, { mode: 0o700 });
const environment = Object.fromEntries(["HOME", "USER", "LOGNAME", "TMPDIR", "PATH", "LANG"]
  .filter((name) => process.env[name]).map((name) => [name, process.env[name]]));
Object.assign(environment, { PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" });
const report = { schema_version: 1, classification: "actual_website_saved_request_preparation",
  parent_id: parent, mutations: [], errors: [], generation_started: false, cleanup: {} };
const started = performance.now();
let server, browser, deadline, origin, child = parent, logBytes = 0, requestCount = 0;
const save = async (name, data) => writeFile(path.join(out, name), JSON.stringify(data, null, 2) + "\n", { flag: "wx", mode: 0o600 });
const exited = (process) => process.exitCode !== null || process.signalCode !== null;
const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
try {
  for (const name of ["browser/tests/studio-prepare-live.mjs", "browser/tests/studio-prepare-observation.mjs", "browser/studio.js", "avatar_assets/studio_jobs.py", "avatar_assets/studio_worker.py", "avatar_assets/standing_authorization.py", "avatar_assets/pipeline_continuation.py"]) {
    const bytes = await readFile(path.join(root, name));
    (report.sources ??= []).push({ path: name, sha256: createHash("sha256").update(bytes).digest("hex") });
  }
  server = spawn(path.join(root, ".venv/bin/python"), ["-B", "scripts/verify_studio_live.py", "--job", parent, "--allow-unplanned", "--serve", "--max-seconds", "675"],
    { cwd: root, env: environment, stdio: ["ignore", "pipe", "pipe"] });
  report.server_pid = server.pid;
  origin = await new Promise((resolve, reject) => {
    let pending = "";
    const timer = setTimeout(() => reject(new Error("Server startup timed out")), 15000);
    server.stdout.on("data", (chunk) => {
      pending += chunk.toString();
      if (pending.length > 4096) return reject(new Error("Server startup output exceeded bound"));
      if (!pending.includes("\n")) return;
      clearTimeout(timer);
      try { const value = JSON.parse(pending.split("\n")[0]); assert.equal(value.job_id, parent);
        assert(Number.isInteger(value.port) && value.port > 0 && value.port <= 65535); resolve(`http://127.0.0.1:${value.port}`);
      } catch (error) { reject(error); }
    });
    server.once("error", reject);
    server.once("exit", () => { clearTimeout(timer); reject(new Error("Server exited before readiness")); });
    server.stderr.on("data", (chunk) => { logBytes += chunk.length; if (logBytes > 1024 ** 2) server.kill("SIGTERM"); });
  });
  browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true,
    timeout: 20000, env: environment, args: ["--disable-background-networking", "--disable-sync", "--no-first-run"] });
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  await context.route("**/*", async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (++requestCount > 1500 || url.origin !== origin) return route.abort();
    if (request.method() !== "GET") {
      const allowed = url.pathname === `/api/studio/jobs/${child}/plan`;
      assert(allowed && request.method() === "POST", "Unexpected mutation");
      assert.deepEqual(request.postDataJSON(), {});
      assert(request.headers()["x-avatar-studio-token"]);
      report.mutations.push({ method: "POST", path: url.pathname });
      assert(report.mutations.length <= 1, "Duplicate mutation");
    }
    return route.continue();
  });
  const page = await context.newPage(); page.setDefaultTimeout(15000);
  deadline = setTimeout(() => browser.close(), 660000);
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  await page.locator(`[data-job="${parent}"]`).click();
  report.job_id = child;
  const prepared = await observePreparation({
    receivePost: async () => {
      const response = await page.waitForResponse((value) => value.url() === `${origin}/api/studio/jobs/${child}/plan`
        && value.request().method() === "POST", { timeout: 15000 });
      return { ok: response.ok(), status: response.status(), body: await response.json() };
    },
    click: () => page.locator("#prepare-plan").click(),
    readJob: async () => {
      const response = await context.request.get(`${origin}/api/studio/jobs/${child}`);
      assert(response.ok(), "Canonical preparation status request failed");
      const value = await response.json(); return value.job ?? value;
    },
    savePost: (response) => save("prepare-response.json", response), pause, maxMs: 630000,
  });
  await save("prepared.json", prepared);
  report.plan_id = prepared.plan_id ?? prepared.plan?.id;
  assert(["planned", "awaiting_authorization"].includes(prepared.status) && /^[a-f0-9]{64}$/.test(report.plan_id || ""),
    prepared.error ?? "Saved job did not produce a pinned plan");
  report.awaiting_generation_authorization = prepared.status === "awaiting_authorization";
  await page.screenshot({ path: path.join(out, "prepared.png") });
  assert.equal(report.mutations.length, 1); report.status = "passed";
} catch (error) { report.status = "failed"; report.errors.push(error.message); }
finally {
  clearTimeout(deadline);
  if (browser) await browser.close();
  report.cleanup.browser_closed = !browser || !browser.isConnected();
  if (server && !exited(server)) server.kill("SIGTERM");
  for (let i = 0; server && !exited(server) && i < 50; i++) await pause(100);
  report.cleanup.server_exit_verified = !server || exited(server);
  report.elapsed_seconds = (performance.now() - started) / 1000;
  report.request_count = requestCount;
  await save("result.json", report);
  console.log(JSON.stringify({ status: report.status, child_id: child, plan_id: report.plan_id, errors: report.errors, cleanup: report.cleanup }));
  process.exitCode = report.status === "passed" && report.cleanup.server_exit_verified ? 0 : 1;
}
