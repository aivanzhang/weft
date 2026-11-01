import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const out = path.resolve(root, process.argv[2] || "outputs/gallery-browser-v1");
const catalogPath = process.argv[3] || "";
await mkdir(out, { recursive: false });
const report = { status: "running", tests: [], loads_ms: [], browser: null, external_requests: [] };
let browser;
let page;
const errors = [];
const server = spawn(path.join(root, ".venv/bin/python"), ["-u", "-c", "import sys; from avatar_assets.gallery import create_server; s=create_server(port=0,catalog_path=sys.argv[1] or None); print(s.server_address[1],flush=True); s.serve_forever()", catalogPath], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
let serverLog = "";
server.stderr.on("data", (data) => { serverLog += data; });
const deadline = setTimeout(() => { server.kill("SIGTERM"); process.exitCode = 1; }, 120000);
const lines = createInterface({ input: server.stdout });
try {
  const port = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Server did not start within ten seconds")), 10000);
    lines.on("line", (line) => {
      if (/^\d+$/.test(line)) { clearTimeout(timer); resolve(Number(line)); }
      else serverLog += `${line}\n`;
    });
    server.on("exit", (code) => { clearTimeout(timer); reject(new Error(`Server exited ${code}: ${serverLog}`)); });
  });
  const origin = `http://127.0.0.1:${port}`;
  browser = await chromium.launch({ executablePath: process.env.AVATAR_BROWSER_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run"] });
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 1440, height: 1040 }, deviceScaleFactor: 1 });
  await context.route("**/*", (route) => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    report.external_requests.push(route.request().url());
    return route.abort();
  });
  page = await context.newPage();
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  const started = performance.now();
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => document.querySelector("model-viewer")?.loaded && !document.querySelector("#play").disabled);
  report.initial_ready_ms = Math.round(performance.now() - started);
  const catalog = await (await page.request.get(`${origin}/api/catalog`)).json();
  assert.ok(catalog.assets.length >= 2 && catalog.assets.length <= 24);
  assert.equal(await page.locator("model-viewer").count(), 1);
  report.tests.push(`${catalog.assets.length} assets, one lazy 3D viewer, local-only loading`);
  await page.screenshot({ path: path.join(out, "desktop.png"), fullPage: true });

  await page.locator("#play").click();
  await page.waitForFunction(() => document.querySelector("model-viewer").currentTime > .2);
  await page.locator("#play").click();
  const paused = await page.locator("model-viewer").evaluate((el) => el.currentTime);
  await page.waitForTimeout(180);
  assert.ok(Math.abs((await page.locator("model-viewer").evaluate((el) => el.currentTime)) - paused) < .02);
  report.tests.push("play advances actual model time; pause holds the pose");

  for (const speed of ["0.5", "2"]) {
    await page.locator("#speed").selectOption(speed);
    await page.locator("#seek").fill("2");
    await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").currentTime - 2) < .05);
  }
  await page.locator("#seek").fill("4");
  await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").currentTime - 4) < .05);
  await page.locator("#seek").fill("2");
  await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").currentTime - 2) < .05);
  await page.locator("#speed").selectOption("1");
  report.tests.push("seeking is absolute at 0.5x/2x and works at/after a finished endpoint");

  for (const asset of catalog.assets.filter((entry) => entry.model)) {
    const start = performance.now();
    await page.locator(`[data-asset="${asset.id}"]`).click();
    await page.locator('[data-mode="model"]').click();
    await page.waitForFunction(() => document.querySelector("model-viewer").loaded && !document.querySelector("#play").disabled);
    report.loads_ms.push({ asset: asset.id, ms: Math.round(performance.now() - start) });
    await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").getCameraOrbit().theta - 25 * Math.PI / 180) < .1);
    const animations = await page.locator("model-viewer").evaluate((el) => el.availableAnimations);
    assert.equal(animations.length, 4);
    for (const animation of animations) {
      await page.locator(`[data-state="${animation}"]`).click();
      await page.waitForFunction((name) => document.querySelector("model-viewer").animationName === name && !document.querySelector("#play").disabled, animation);
      await page.locator("#seek").fill("2");
      await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").currentTime - 2) < .05);
    }
    await page.locator('[data-state="milestone_level_up"]').click();
    await page.locator("#seek").fill("2");
    await page.waitForTimeout(180);
    await page.screenshot({ path: path.join(out, `${asset.id}-celebrate.png`), fullPage: true });
    await page.locator('[data-view="front"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").getCameraOrbit().theta) < .1);
    await page.locator('[data-view="side"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").getCameraOrbit().theta - Math.PI / 2) < .1);
    report.tests.push(`${asset.id}: four skeletal clips, seeking and camera rotation`);
  }

  const mostDetailed = catalog.assets.filter((asset) => asset.model).sort((a, b) => b.facts.vertices-a.facts.vertices)[0];
  await page.locator(`[data-asset="${mostDetailed.id}"]`).click();
  await page.locator('[data-mode="model"]').click();
  await page.waitForFunction(() => document.querySelector("model-viewer").loaded && !document.querySelector("#play").disabled);
  await page.locator('[data-state="milestone_level_up"]').click();
  await page.locator("#seek").fill("0");
  const activeCdp = await context.newCDPSession(page);
  await activeCdp.send("Performance.enable");
  const activeMetrics = async () => Object.fromEntries((await activeCdp.send("Performance.getMetrics")).metrics.map((metric) => [metric.name, metric.value]));
  const activeStart = await activeMetrics();
  await page.locator("#play").click();
  const responsiveness = await page.evaluate(() => new Promise((resolve) => {
    const start = performance.now();
    const gaps = [];
    let previous = start;
    function sample(now) {
      gaps.push(now-previous); previous = now;
      if (now-start < 2000) requestAnimationFrame(sample);
      else resolve({ elapsed_ms: now-start, callbacks: gaps.length, max_gap_ms: Math.max(...gaps) });
    }
    requestAnimationFrame(sample);
  }));
  await page.locator("#play").click();
  const activeEnd = await activeMetrics();
  const advanced = await page.locator("model-viewer").evaluate((el) => el.currentTime);
  assert.ok(advanced > 1, "Detailed skeletal animation must advance");
  report.detailed_playback_observation = { asset: mostDetailed.id, vertices: mostDetailed.facts.vertices, ...responsiveness,
    clip_time_seconds: advanced, renderer_task_seconds: activeEnd.TaskDuration-activeStart.TaskDuration,
    js_heap_bytes: activeEnd.JSHeapUsedSize, scope: "Two-second browser responsiveness sample with software WebGL; RAF callbacks are not measured rendered frames or a sustained performance guarantee." };
  await activeCdp.detach();
  report.tests.push("most detailed model remains controllable during actual skeletal playback");

  const robot = catalog.assets.find((asset) => asset.model && asset.states.some((clip) => clip.video));
  await page.locator(`[data-asset="${robot.id}"]`).click();
  await page.locator('[data-mode="video"]').click();
  await page.waitForFunction(() => !document.querySelector("#play").disabled);
  await page.locator("#seek").fill("0.4");
  assert.ok(Math.abs(await page.locator("#video").evaluate((el) => el.currentTime) - .4) < .05);
  await page.locator("#repeat").check();
  assert.equal(await page.locator("#video").evaluate((el) => el.loop), true);
  await page.locator("#play").click();
  await page.waitForFunction(() => document.querySelector("#video").currentTime > .5);
  await page.locator('[data-mode="portrait"]').click();
  assert.equal(await page.locator("#video").evaluate((el) => el.paused), true);
  await page.waitForFunction(() => document.querySelector("#portrait").complete && document.querySelector("#portrait").naturalWidth > 0);
  assert.ok((await page.locator("#portrait").getAttribute("src")).endsWith(robot.portrait || robot.poster));
  report.tests.push("video seeking/repeat/play; portrait mode stops playback");

  await page.locator('[data-mode="video"]').click();
  await page.waitForFunction(() => !document.querySelector("#play").disabled);
  await page.evaluate(() => {
    const video = document.querySelector("#video");
    window.restoreVideoPlay = () => { delete video.play; };
    video.play = () => new Promise((_, reject) => { window.rejectPendingPlay = reject; });
  });
  await page.locator("#play").click();
  await page.locator('[data-mode="portrait"]').click();
  await page.evaluate(() => { window.rejectPendingPlay(new Error("Interrupted old selection")); window.restoreVideoPlay(); });
  await page.waitForTimeout(30);
  assert.equal(await page.locator("#error").isHidden(), true);
  report.tests.push("injected delayed play rejection cannot show an error on a newer view");

  const reference = catalog.assets.find((asset) => !asset.model);
  await page.locator(`[data-asset="${reference.id}"]`).click();
  assert.equal(await page.locator('[data-mode="model"]').isDisabled(), true);
  assert.match(await page.locator("#asset-kind").textContent(), /2D ONLY/);
  report.tests.push("reference pack is labeled 2D, never presented as geometry");

  await page.locator("#search").fill("no such character");
  assert.equal(await page.locator("#empty-search").isVisible(), true);
  await page.locator("#search").fill("");
  await page.locator(`[data-asset="${robot.id}"]`).click();
  await page.locator('[data-mode="model"]').click();
  await page.waitForFunction(() => document.querySelector("model-viewer").loaded && !document.querySelector("#play").disabled);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  await page.screenshot({ path: path.join(out, "mobile.png"), fullPage: true });
  report.tests.push("search and 390px layout without horizontal document overflow");

  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  const metrics = async () => Object.fromEntries((await cdp.send("Performance.getMetrics")).metrics.map((m) => [m.name, m.value]));
  await page.waitForTimeout(500);
  const idleStart = await metrics();
  await page.waitForTimeout(1000);
  const idleEnd = await metrics();
  report.idle_observation = { seconds: 1, renderer_task_seconds: idleEnd.TaskDuration - idleStart.TaskDuration, js_heap_bytes: idleEnd.JSHeapUsedSize, models_in_dom: await page.locator("model-viewer").count() };

  const motionContext = await browser.newContext({ reducedMotion: "reduce" });
  const motionPage = await motionContext.newPage();
  const reducedModelRequests = [];
  motionPage.on("request", (request) => { if (/\/asset\/[^/]+\/model$/.test(request.url())) reducedModelRequests.push(request.url()); });
  await motionPage.goto(origin);
  await motionPage.waitForFunction(count => document.querySelectorAll(".asset-card").length === count, catalog.assets.length);
  assert.equal(await motionPage.locator("#portrait").isVisible(), true);
  assert.equal(await motionPage.locator("#play").isDisabled(), true);
  assert.equal(await motionPage.locator("model-viewer").getAttribute("src"), null);
  assert.deepEqual(reducedModelRequests, []);
  report.tests.push("reduced motion defaults to static portrait, no GLB fetch or autoplay");
  await motionContext.close();
  const failurePage = await context.newPage();
  await failurePage.route("**/asset/*/model", (route) => route.abort());
  await failurePage.goto(origin);
  await failurePage.locator("#error").waitFor({ state: "visible" });
  await failurePage.locator('[data-mode="portrait"]').click();
  assert.equal(await failurePage.locator("#portrait").isVisible(), true);
  assert.equal(await failurePage.locator("#error").isHidden(), true);
  report.tests.push("failed 3D load leaves the static portrait usable");
  await failurePage.close();
  const slowPage = await context.newPage();
  await slowPage.addInitScript(() => {
    const schedule = window.setTimeout;
    window.setTimeout = (fn, delay, ...args) => schedule(fn, delay === 20000 ? 20 : delay, ...args);
  });
  await slowPage.route("**/asset/*/video-*", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 250));
    await route.continue();
  });
  await slowPage.goto(origin);
  await slowPage.locator('[data-mode="video"]').waitFor();
  await slowPage.waitForFunction(count => document.querySelectorAll(".asset-card").length === count, catalog.assets.length);
  await slowPage.locator('[data-mode="video"]').click();
  await slowPage.locator("#error").filter({ hasText: "longer than expected" }).waitFor({ state: "visible" });
  await slowPage.waitForFunction(() => !document.querySelector("#play").disabled);
  assert.equal(await slowPage.locator("#error").isHidden(), true);
  report.tests.push("accelerated timeout with delayed real video clears its error on successful load");
  await slowPage.close();
  assert.deepEqual(report.external_requests, []);
  assert.deepEqual(errors, []);
  report.status = "passed";
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  report.status = "failed";
  report.error = error.stack;
  report.console_errors = errors;
  if (page) {
    await page.screenshot({ path: path.join(out, "failure.png"), fullPage: true }).catch(() => {});
    report.visible_error = await page.locator("#error").textContent().catch(() => null);
  }
  console.error(error);
  process.exitCode = 1;
} finally {
  clearTimeout(deadline);
  await browser?.close();
  server.kill("SIGTERM");
  lines.close();
  await writeFile(path.join(out, "browser-test.json"), `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(path.join(out, "server.log"), serverLog);
}
