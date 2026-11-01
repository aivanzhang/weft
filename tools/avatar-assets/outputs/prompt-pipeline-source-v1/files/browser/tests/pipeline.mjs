import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { access, lstat, mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { promisify } from "node:util";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const limits = { wall_ms: 900_000, sampled_rss_bytes: 3 * 1024 ** 3, output_bytes: 64 * 1024 ** 2,
  log_bytes: 1024 ** 2, events: 2000, event_bytes: 512 * 1024, assets: 24, states_per_asset: 8, operation_ms: 15_000 };
assert.equal(process.argv.length, 4, "Usage: node browser/tests/pipeline.mjs <new-output-folder> <catalog.json>");
const out = path.resolve(root, process.argv[2]);
const catalogPath = path.resolve(root, process.argv[3]);
const beneath = (parent, candidate) => {
  const relative = path.relative(parent, candidate);
  return relative !== "" && !relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative);
};
assert(beneath(path.join(root, "outputs"), out), "Output must be a new folder under avatar-assets/outputs");
assert(beneath(root, catalogPath), "Catalog must be inside avatar-assets");
const outputRoot = await realpath(path.join(root, "outputs"));
const outputParent = await realpath(path.dirname(out));
assert(beneath(root, outputRoot) && (outputParent === outputRoot || beneath(outputRoot, outputParent)),
  "Output parent escapes avatar-assets/outputs");
assert(beneath(root, await realpath(catalogPath)), "Catalog escapes avatar-assets");
const catalogStat = await lstat(catalogPath);
assert(catalogStat.isFile() && !catalogStat.isSymbolicLink() && catalogStat.size <= 512 * 1024,
  "Catalog must be a regular JSON file no larger than 512 KiB");
const rawCatalog = await readFile(catalogPath);
const sourceCatalog = JSON.parse(rawCatalog);
assert.equal(sourceCatalog.schema_version, 1);
assert(Array.isArray(sourceCatalog.assets) && sourceCatalog.assets.length >= 1 && sourceCatalog.assets.length <= limits.assets);
const python = path.join(root, ".venv/bin/python");
const executable = process.env.AVATAR_BROWSER_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
await Promise.all([access(python, constants.X_OK), access(executable, constants.X_OK)]);
await mkdir(out, { mode: 0o700 });

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const started = performance.now();
const localTime = () => new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", year: "numeric",
  month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  timeZoneName: "short" }).format(new Date());
const report = { schema_version: 1, status: "running", recorded_local: localTime(), timezone: "America/Los_Angeles",
  scope: "Real browser GLB loading, each declared skeletal clip and supplied video, static/reduced-motion/failure fallback. Not a visual-quality, seamless-loop, physics or sustained-FPS certification.",
  catalog: { path: path.relative(root, catalogPath), sha256: digest(rawCatalog) }, limits, assets: [], checks: [],
  errors: [], expected_failure_observations: [], external_requests: [], request_failures: [],
  resources: { sampled_peak_rss_bytes: 0, sampling_interval_ms: 2000,
    scope: "RSS summed for this test and its descendants; shared pages may be counted twice. Sampled enforcement, not a hard OS allocation cap." },
  artifacts: [], cleanup: {} };
let outputBytes = 0;
async function save(name, bytes) {
  const data = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  assert(outputBytes + data.length <= limits.output_bytes, "Browser test output bound exceeded");
  await writeFile(path.join(out, name), data, { flag: "wx", mode: 0o600 });
  outputBytes += data.length;
  report.artifacts.push({ path: name, bytes: data.length, sha256: digest(data) });
}
await save("source-catalog.json", rawCatalog);
await save("source-test.mjs", await readFile(fileURLToPath(import.meta.url)));

let server, browserLaunch, browserServer, browser, page, origin;
let serverLog = "", serverLogBytes = 0, cancelled = false, cancellation;
let eventCount = 0, eventBytes = 0;
let deadline, monitorTimer, monitorTask;
const contexts = new Set();
const execFileAsync = promisify(execFile);
let rejectCancellation;
const cancelledPromise = new Promise((_, reject) => { rejectCancellation = reject; });
// Resource failure rejects the active work before cleanup; no foreign process is signalled.
function cancel(reason) {
  if (cancelled) return;
  cancelled = true;
  cancellation = new Error(reason);
  rejectCancellation(cancellation);
}
function checkRunning() {
  if (cancelled) throw cancellation;
}
function record(list, item) {
  const bytes = Buffer.byteLength(JSON.stringify(item));
  if (eventCount >= limits.events || eventBytes + bytes > limits.event_bytes)
    return cancel("Browser observations exceeded their count/byte bound");
  eventCount++; eventBytes += bytes;
  list.push(item);
}
function appendLog(data) {
  const buffer = Buffer.from(data);
  const remaining = Math.max(0, limits.log_bytes - serverLogBytes);
  serverLog += buffer.subarray(0, remaining).toString("utf8");
  serverLogBytes += buffer.length;
  if (serverLogBytes > limits.log_bytes) cancel("Gallery server log exceeded 1 MiB");
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function bounded(promise, milliseconds, label) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out`)), milliseconds);
    })]);
  } finally { clearTimeout(timer); }
}
async function sampleMemory() {
  const { stdout } = await execFileAsync("/bin/ps", ["-axo", "pid=,ppid=,rss="], { timeout: 2000, maxBuffer: 512 * 1024 });
  const children = new Map(), rss = new Map();
  for (const line of stdout.trim().split("\n")) {
    const [pid, parent, kib] = line.trim().split(/\s+/).map(Number);
    if (![pid, parent, kib].every(Number.isFinite)) continue;
    rss.set(pid, kib * 1024);
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(pid);
  }
  const owned = [process.pid], seen = new Set();
  let total = 0;
  for (let i = 0; i < owned.length; i++) {
    const pid = owned[i];
    if (seen.has(pid)) continue;
    seen.add(pid); total += rss.get(pid) || 0;
    owned.push(...(children.get(pid) || []));
  }
  report.resources.sampled_peak_rss_bytes = Math.max(report.resources.sampled_peak_rss_bytes, total);
  if (total > limits.sampled_rss_bytes) cancel("Owned browser/server process tree exceeded sampled 3 GiB RSS");
}
function startMonitor() {
  monitorTimer = setInterval(() => {
    if (monitorTask || cancelled) return;
    monitorTask = sampleMemory().catch((error) => cancel(`Memory monitor failed: ${error.message}`))
      .finally(() => { monitorTask = null; });
  }, 2000);
}
const onInterrupt = () => cancel("Browser test interrupted by SIGINT");
const onTerminate = () => cancel("Browser test interrupted by SIGTERM");

async function startServer() {
  const code = "import sys; from avatar_assets.gallery import create_server; s=create_server(port=0,catalog_path=sys.argv[1]); print(s.server_address[1],flush=True); s.serve_forever()";
  server = spawn(python, ["-u", "-c", code, path.relative(root, catalogPath)], {
    cwd: root, detached: true, stdio: ["ignore", "pipe", "pipe"],
    env: { PATH: "/usr/bin:/bin:/usr/sbin:/sbin", PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" },
  });
  report.server_pid = server.pid;
  server.stderr.on("data", appendLog);
  return bounded(new Promise((resolve, reject) => {
    let pending = "", resolved = false;
    server.stdout.on("data", (data) => {
      appendLog(data);
      pending += data.toString("utf8");
      if (pending.length > 4096) return reject(new Error("Invalid bounded server startup response"));
      const newline = pending.indexOf("\n");
      if (resolved || newline < 0) return;
      const line = pending.slice(0, newline).trim();
      if (!/^\d{1,5}$/.test(line)) return reject(new Error("Invalid gallery startup port"));
      const port = Number(line);
      if (!(port > 0 && port <= 65535)) return reject(new Error("Invalid gallery port"));
      resolved = true; pending = ""; resolve(`http://127.0.0.1:${port}`);
    });
    server.once("error", reject);
    server.once("exit", (code, signal) => {
      const message = `Owned gallery server exited (${code ?? signal})`;
      if (!resolved) reject(new Error(message));
      else if (!report.cleanup.started) cancel(message);
    });
  }), 15_000, "Gallery startup");
}

async function newContext({ reduced = false, failureProbe = false } = {}) {
  checkRunning();
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1,
    reducedMotion: reduced ? "reduce" : "no-preference", serviceWorkers: "block", acceptDownloads: false });
  contexts.add(context);
  context.setDefaultTimeout(limits.operation_ms);
  context.setDefaultNavigationTimeout(30_000);
  await context.route("**/*", (route) => {
    const url = new URL(route.request().url());
    if (url.origin === origin || url.protocol === "data:") return route.continue();
    record(report.external_requests, url.href.slice(0, 2048));
    return route.abort("blockedbyclient");
  });
  await context.routeWebSocket("**/*", (socket) => {
    record(report.external_requests, socket.url().slice(0, 2048));
    socket.close({ code: 1008, reason: "Network sockets are not part of the local gallery test" });
  });
  context.on("page", (tab) => {
    tab.on("pageerror", (error) => record(report.errors, { kind: "pageerror", message: error.message.slice(0, 4096) }));
    tab.on("console", (message) => {
      if (message.type() === "error") record(failureProbe ? report.expected_failure_observations : report.errors,
        { kind: "console", message: message.text().slice(0, 4096) });
    });
    tab.on("requestfailed", (request) => record(report.request_failures,
      { url: request.url().slice(0, 2048), error: request.failure()?.errorText, injected: failureProbe }));
    tab.on("crash", () => cancel("Owned browser page crashed"));
    tab.on("download", (download) => { download.cancel().catch(() => {}); cancel("Unexpected browser download"); });
  });
  return context;
}
const select = (tab, name, value) => tab.locator(`[data-${name}=${JSON.stringify(value)}]`).click();
async function portrait(tab, asset) {
  await select(tab, "mode", "portrait");
  await tab.waitForFunction((expected) => {
    const image = document.querySelector("#portrait");
    return !image.hidden && image.getAttribute("src") === expected && image.complete && image.naturalWidth > 0;
  }, asset.portrait || asset.poster);
  assert(await tab.locator("#play").isDisabled());
  assert(await tab.locator("#error").isHidden());
  assert(await tab.locator("#video").evaluate((video) => video.paused));
  return tab.locator("#portrait").evaluate((image) => ({ width: image.naturalWidth, height: image.naturalHeight }));
}
async function exerciseClip(tab, selector) {
  checkRunning();
  const player = tab.locator(selector);
  const duration = await player.evaluate((element) => element.duration);
  assert(Number.isFinite(duration) && duration >= .25 && duration <= 60, "Invalid bounded clip duration");
  assert(await player.evaluate((element) => element.paused), "Playback must begin paused");
  await tab.locator("#repeat").uncheck();
  await tab.locator("#speed").selectOption("1");
  await tab.locator("#seek").fill("0");
  await tab.locator("#play").click();
  await tab.waitForFunction(({ selector, threshold }) => {
    const element = document.querySelector(selector);
    return !element.paused && element.currentTime > threshold;
  }, { selector, threshold: Math.min(.08, duration / 8) });
  const advanced = await player.evaluate((element) => element.currentTime);
  await tab.locator("#play").click();
  assert(await player.evaluate((element) => element.paused), "Pause control did not pause actual playback");
  const paused = await player.evaluate((element) => element.currentTime);
  await tab.waitForTimeout(120);
  assert(Math.abs(await player.evaluate((element) => element.currentTime) - paused) < .025, "Paused pose drifted");
  const target = Math.round(duration * 50) / 100;
  await tab.locator("#seek").fill(String(target));
  await tab.waitForFunction(({ selector, target }) => Math.abs(document.querySelector(selector).currentTime - target) < .06,
    { selector, target });
  return { duration_seconds: duration, playback_advanced_to: advanced, paused_at: paused, seek_seconds: target };
}
async function screenshot(tab, name) {
  checkRunning();
  const bytes = await tab.screenshot({ fullPage: false, timeout: 10_000 });
  assert(bytes.length <= 8 * 1024 ** 2, "Screenshot exceeded 8 MiB");
  assert(outputBytes + bytes.length <= limits.output_bytes - 3 * 1024 ** 2, "Screenshot budget reserves final logs/report");
  await save(name, bytes);
}

const manifestInputs = [];
async function bindManifest(relative) {
  assert.equal(typeof relative, "string", "Missing catalog pack path");
  const filename = path.resolve(root, relative, "manifest.json");
  assert(beneath(root, filename) && beneath(root, await realpath(filename)), "Manifest escapes avatar-assets");
  const info = await lstat(filename);
  assert(info.isFile() && !info.isSymbolicLink() && info.size <= 4 * 1024 ** 2, "Invalid bounded pack manifest");
  const item = { path: path.relative(root, filename), sha256: digest(await readFile(filename)) };
  manifestInputs.push(item);
  return item;
}

async function runChecks() {
  origin = await startServer();
  checkRunning();
  browserLaunch = chromium.launchServer({ executablePath: executable, headless: true, host: "127.0.0.1", port: 0,
    timeout: 20_000, handleSIGINT: false, handleSIGTERM: false, handleSIGHUP: false,
    env: { PATH: "/usr/bin:/bin:/usr/sbin:/sbin", ...(process.env.HOME ? { HOME: process.env.HOME } : {}),
      ...(process.env.TMPDIR ? { TMPDIR: process.env.TMPDIR } : {}) },
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "--disable-background-networking",
      "--disable-component-update", "--disable-sync", "--js-flags=--max-old-space-size=768"] }).then(async (handle) => {
    browserServer = handle;
    if (report.cleanup.started) { await handle.kill(); throw new Error("Browser launch completed after cancellation"); }
    return handle;
  });
  browserServer = await browserLaunch;
  report.browser_pid = browserServer.process().pid;
  assert.equal(new URL(browserServer.wsEndpoint()).hostname, "127.0.0.1");
  checkRunning();
  browser = await chromium.connect(browserServer.wsEndpoint(), { timeout: 10_000 });
  report.browser = { version: browser.version(), executable, headless: true, software_webgl: true };
  const context = await newContext();
  const response = await context.request.get(`${origin}/api/catalog`, { timeout: 10_000, maxRedirects: 0 });
  assert.equal(response.status(), 200);
  const body = await response.body();
  assert(body.length <= 512 * 1024, "Expanded catalog exceeds 512 KiB");
  const catalog = JSON.parse(body);
  report.catalog.expanded_sha256 = digest(body);
  assert.equal(digest(await readFile(catalogPath)), report.catalog.sha256, "Catalog changed during server startup");
  assert.deepEqual(catalog.assets.map((asset) => asset.id), sourceCatalog.assets.map((asset) => asset.id));
  for (const asset of catalog.assets) {
    assert(asset.model && asset.states.length >= 1 && asset.states.length <= limits.states_per_asset,
      `${asset.id}: expected a rigged model with 1–8 states, not an unrigged preview or 2D reference`);
    assert(asset.facts.bones > 0 && asset.facts.animations === asset.states.length, "Catalog lacks real rig/animation facts");
    assert(asset.states.every((state) => state.video && state.poster), "Complete pipeline states require video and static fallback");
    assert(new URL(asset.model, origin).origin === origin);
    assert.equal(new Set(asset.states.map((state) => state.id)).size, asset.states.length);
  }
  page = await context.newPage();
  const initial = performance.now();
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  await page.waitForFunction((count) => document.querySelectorAll(".asset-card").length === count, catalog.assets.length);
  report.initial_library_ms = Math.round(performance.now() - initial);
  assert.equal(await page.locator("model-viewer").count(), 1);
  let videoCount = 0;
  for (const [index, asset] of catalog.assets.entries()) {
    checkRunning();
    const assetStarted = performance.now();
    const observation = { id: asset.id, model: asset.model, animations: [], videos: [], missing_videos: [] };
    report.assets.push(observation);
    observation.model_manifest = await bindManifest(sourceCatalog.assets[index].pack);
    observation.media_manifest = await bindManifest(sourceCatalog.assets[index].media_pack);
    await select(page, "asset", asset.id);
    await select(page, "mode", "model");
    await page.waitForFunction((source) => {
      const model = document.querySelector("model-viewer");
      return model.getAttribute("src") === source && model.loaded && !document.querySelector("#play").disabled;
    }, asset.model, { timeout: 30_000 });
    observation.load_ms = Math.round(performance.now() - assetStarted);
    assert(await page.locator("#error").isHidden());
    const animations = await page.locator("model-viewer").evaluate((model) => model.availableAnimations);
    assert.deepEqual([...animations].sort(), asset.states.map((state) => state.id).sort(), "GLB clips must match declared states");
    for (const state of asset.states) {
      await select(page, "state", state.id);
      await page.waitForFunction((id) => document.querySelector("model-viewer").animationName === id
        && !document.querySelector("#play").disabled, state.id);
      observation.animations.push({ id: state.id, ...await exerciseClip(page, "model-viewer") });
    }
    await screenshot(page, `asset-${String(index + 1).padStart(2, "0")}-model.png`);
    for (const state of asset.states) {
      if (!state.video) { observation.missing_videos.push(state.id); continue; }
      await select(page, "mode", "video");
      await select(page, "state", state.id);
      await page.waitForFunction((source) => {
        const video = document.querySelector("#video");
        return video.getAttribute("src") === source && video.readyState >= 2 && video.videoWidth > 0
          && !document.querySelector("#play").disabled;
      }, state.video);
      const dimensions = await page.locator("#video").evaluate((video) => ({ width: video.videoWidth, height: video.videoHeight }));
      observation.videos.push({ id: state.id, ...dimensions, ...await exerciseClip(page, "#video") });
      videoCount++;
    }
    observation.portrait = await portrait(page, asset);
    observation.static_fallbacks = await page.evaluate(async (states) => {
      const results = [];
      for (const state of states) {
        const image = new Image();
        let timer;
        try {
          await Promise.race([new Promise((resolve, reject) => {
            image.onload = resolve; image.onerror = () => reject(new Error("Static fallback image failed")); image.src = state.poster;
          }), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("Static image deadline")), 5000); })]);
          if (!(image.naturalWidth > 0 && image.naturalWidth * image.naturalHeight <= 16_777_216)) throw new Error("Invalid static image dimensions");
          results.push({ id: state.id, source: state.poster, width: image.naturalWidth, height: image.naturalHeight });
        } finally { clearTimeout(timer); }
      }
      return results;
    }, asset.states);
    observation.elapsed_ms = Math.round(performance.now() - assetStarted);
  }
  assert(videoCount > 0, "A complete pipeline catalog needs rendered video evidence");
  report.checks.push("Every declared GLB clip loads, plays, pauses and seeks; every supplied video decodes and plays; portraits and per-state static images decode");

  const reduced = await newContext({ reduced: true });
  const reducedPage = await reduced.newPage();
  const reducedMotionRequests = [];
  reducedPage.on("request", (request) => {
    if (/\/asset\/[^/]+\/(model|video-[^/]+)$/.test(new URL(request.url()).pathname)) reducedMotionRequests.push(request.url());
  });
  await reducedPage.goto(origin, { waitUntil: "domcontentloaded" });
  await reducedPage.waitForFunction((count) => document.querySelectorAll(".asset-card").length === count, catalog.assets.length);
  for (const asset of catalog.assets) {
    await select(reducedPage, "asset", asset.id);
    assert(await reducedPage.locator("#portrait").isVisible(), "Reduced motion must choose portrait without a mode click");
    await reducedPage.waitForFunction((source) => document.querySelector("#portrait").getAttribute("src") === source
      && document.querySelector("#portrait").complete && document.querySelector("#portrait").naturalWidth > 0, asset.portrait || asset.poster);
    assert(await reducedPage.locator("#play").isDisabled());
    assert.equal(await reducedPage.locator("model-viewer").getAttribute("src"), null);
    assert.equal(await reducedPage.locator("#video").getAttribute("src"), null);
  }
  assert.deepEqual(reducedMotionRequests, []);
  await screenshot(reducedPage, "reduced-motion.png");
  await reduced.close(); contexts.delete(reduced);
  report.checks.push("Reduced motion selects static portraits for every asset without any GLB/video fetch or autoplay");

  const failure = await newContext({ failureProbe: true });
  const failurePage = await failure.newPage();
  const first = catalog.assets[0];
  await failurePage.route(`${origin}${first.model}`, (route) => route.abort("failed"));
  await failurePage.goto(`${origin}/#${encodeURIComponent(first.id)}`, { waitUntil: "domcontentloaded" });
  await failurePage.locator("#error").waitFor({ state: "visible" });
  assert.match(await failurePage.locator("#error").textContent(), /3D model could not load/);
  assert(await failurePage.locator("#play").isDisabled());
  await portrait(failurePage, first);
  await screenshot(failurePage, "model-failure-static-fallback.png");
  const failedVideo = first.states[0];
  await failurePage.route(`${origin}${failedVideo.video}`, (route) => route.abort("failed"));
  await select(failurePage, "state", failedVideo.id);
  await select(failurePage, "mode", "video");
  await failurePage.locator("#error").waitFor({ state: "visible" });
  assert.match(await failurePage.locator("#error").textContent(), /video could not load/);
  assert(await failurePage.locator("#play").isDisabled());
  await portrait(failurePage, first);
  await failure.close(); contexts.delete(failure);
  report.checks.push("Intentionally failed GLB/video requests disable playback and leave the real static portrait usable");

  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), "Mobile layout overflows horizontally");
  await screenshot(page, "mobile-portrait.png");
  report.checks.push("390px portrait layout has no horizontal overflow");
  assert.deepEqual(report.external_requests, [], "External requests were blocked but must not be attempted");
  assert.deepEqual(report.errors, [], "Unexpected browser console/page errors");
  assert.equal(digest(await readFile(catalogPath)), report.catalog.sha256, "Catalog changed during verification");
  for (const input of manifestInputs) assert.equal(digest(await readFile(path.join(root, input.path))), input.sha256,
    "Pack manifest changed during verification");
  checkRunning();
  report.status = "passed";
}

async function waitExited(child, milliseconds) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return true;
  try {
    await bounded(new Promise((resolve) => child.once("exit", resolve)), milliseconds, "Owned process exit");
    return true;
  } catch { return child.exitCode !== null || child.signalCode !== null; }
}
async function cleanup() {
  report.cleanup.started = true;
  clearTimeout(deadline); clearInterval(monitorTimer);
  if (monitorTask) await bounded(monitorTask, 2500, "Memory monitor cleanup").catch(() => {});
  if (browserLaunch && !browserServer) await bounded(browserLaunch, 21_000, "Pending browser launch cleanup").catch(() => {});
  await bounded(Promise.allSettled([...contexts].map((context) => context.close())), 3000, "Context cleanup").catch(() => {});
  if (browserServer) {
    await bounded(browserServer.close(), 5000, "Browser shutdown").catch(async () => {
      await bounded(browserServer.kill(), 3000, "Owned browser kill").catch(() => {});
    });
    report.cleanup.browser_exit_verified = await waitExited(browserServer.process(), 3000);
  } else report.cleanup.browser_exit_verified = true;
  if (server?.pid && server.exitCode === null && server.signalCode === null) {
    try { process.kill(-server.pid, "SIGTERM"); } catch (error) { if (error.code !== "ESRCH") record(report.errors, { kind: "cleanup", message: error.message }); }
    if (!await waitExited(server, 3000)) {
      try { process.kill(-server.pid, "SIGKILL"); } catch (error) { if (error.code !== "ESRCH") record(report.errors, { kind: "cleanup", message: error.message }); }
    }
  }
  report.cleanup.server_exit_verified = await waitExited(server, 3000);
  process.removeListener("SIGINT", onInterrupt); process.removeListener("SIGTERM", onTerminate);
  if (!report.cleanup.browser_exit_verified || !report.cleanup.server_exit_verified) {
    report.status = "failed"; report.cleanup.manual_stop_required = true;
  }
}

process.once("SIGINT", onInterrupt); process.once("SIGTERM", onTerminate);
deadline = setTimeout(() => cancel("Browser test exceeded its 900-second wall deadline"), limits.wall_ms);
startMonitor();
try {
  await Promise.race([runChecks(), cancelledPromise]);
} catch (error) {
  report.status = "failed";
  report.error = String(error.stack || error).slice(0, 16_384);
  if (page && !cancelled) await screenshot(page, "failure.png").catch(() => {});
} finally {
  await cleanup();
  if (cancelled || report.errors.length) {
    report.status = "failed";
    report.error ||= cancellation?.message || "Unexpected browser/cleanup errors were recorded";
  }
  report.elapsed_ms = Math.round(performance.now() - started);
  report.finished_local = localTime();
  await save("server.log", serverLog);
  report.output_bytes_before_report = outputBytes;
  const reportBytes = Buffer.from(`${JSON.stringify(report, null, 2)}\n`);
  assert(reportBytes.length <= 2 * 1024 ** 2 && outputBytes + reportBytes.length <= limits.output_bytes,
    "Final browser report exceeded its output bound");
  await writeFile(path.join(out, "browser-test.json"), reportBytes, { flag: "wx", mode: 0o600 });
  console.log(JSON.stringify({ status: report.status, output: out, assets: report.assets.length, elapsed_ms: report.elapsed_ms,
    error: report.error, cleanup: report.cleanup }));
  process.exitCode = report.status === "passed" ? 0 : 1;
}
