import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir, writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { chromium } from "playwright-core";

const root = fileURLToPath(new URL("../../", import.meta.url));
const out = path.resolve(root, process.argv[2]);
const catalogPath = process.argv[3] || "browser/gallery-fur-comparison-v1.json";
await mkdir(out, { recursive: false });
const report = { scope: "Actual browser models; sampled visual comparison, not an image-similarity or FPS score", loads: [], external: [], errors: [] };
const server = spawn(path.join(root, ".venv/bin/python"), ["-u", "-c", "import sys; from avatar_assets.gallery import create_server; s=create_server(port=0,catalog_path=sys.argv[1]); print(s.server_address[1],flush=True); s.serve_forever()", catalogPath], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
let log = "", browser;
server.stderr.on("data", (value) => { log += value; });
const lines = createInterface({ input: server.stdout });
const guard = setTimeout(() => { server.kill("SIGTERM"); browser?.close(); }, 110000);
try {
  const port = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Gallery startup timeout")), 10000);
    lines.on("line", (line) => { if (/^\d+$/.test(line)) { clearTimeout(timer); resolve(Number(line)); } });
    server.on("exit", () => { clearTimeout(timer); reject(new Error(log)); });
  });
  const origin = `http://127.0.0.1:${port}`;
  browser = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true, args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run"] });
  report.browser = browser.version();
  const context = await browser.newContext({ viewport: { width: 1440, height: 1040 }, deviceScaleFactor: 1 });
  await context.route("**/*", route => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    report.external.push(route.request().url());
    return route.abort();
  });
  const page = await context.newPage();
  page.on("pageerror", error => report.errors.push(error.message));
  await page.goto(origin);
  const catalog = await (await page.request.get(`${origin}/api/catalog`)).json();
  for (const asset of catalog.assets.filter(asset => asset.model)) {
    const start = performance.now();
    await page.locator(`[data-asset="${asset.id}"]`).click();
    await page.waitForFunction(animated => document.querySelector("model-viewer").loaded &&
      (!animated || !document.querySelector("#play").disabled), asset.states.length > 0);
    if (asset.states.length) {
      await page.locator(`[data-state="${asset.states[0].id}"]`).click();
      await page.locator("#seek").fill("0");
    } else {
      assert(await page.locator("#play").isDisabled());
      assert(await page.locator("#error").isHidden());
    }
    const viewer = page.locator("model-viewer");
    await viewer.evaluate(el => { el.cameraOrbit = "0deg 80deg 105%"; el.jumpCameraToGoal(); });
    await page.waitForTimeout(350);
    await viewer.screenshot({ path: path.join(out, `${asset.id}-normal.png`) });
    await viewer.evaluate(el => { el.cameraOrbit = "0deg 80deg 65%"; el.jumpCameraToGoal(); });
    await page.waitForTimeout(350);
    await viewer.screenshot({ path: path.join(out, `${asset.id}-close.png`) });
    const clips = await viewer.evaluate(el => el.availableAnimations);
    assert.equal(clips.length, asset.states.length);
    for (const clip of clips) {
      await page.locator(`[data-state="${clip}"]`).click();
      await page.locator("#seek").fill("2");
      await page.waitForFunction(() => Math.abs(document.querySelector("model-viewer").currentTime - 2) < .05);
    }
    report.loads.push({ id: asset.id, model: asset.model, clips, elapsed_ms: Math.round(performance.now()-start) });
  }
  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.external, []);
  report.status = "passed";
} catch (error) {
  report.status = "failed"; report.failure = String(error); process.exitCode = 1;
} finally {
  clearTimeout(guard);
  await browser?.close();
  lines.close(); server.kill("SIGTERM");
  await writeFile(path.join(out, "browser-test.json"), JSON.stringify(report, null, 2)+"\n");
  await writeFile(path.join(out, "server.log"), log);
  console.log(JSON.stringify(report));
}
