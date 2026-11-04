import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { once } from "node:events";
import { readFile } from "node:fs/promises";
import { landingPort, landingServer } from "./dev.mjs";
import { publicAssets } from "./public-assets.mjs";

let server;
let base;
before(async () => {
  server = landingServer();
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  base = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  if (!server?.listening) return;
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
});

test("landing loads from its own root and all public assets are available", async () => {
  const response = await fetch(base);
  assert.equal(new URL(response.url).pathname, "/website/index.html");
  assert.match(await response.text(), /Weft/);
  for (const asset of publicAssets) {
    const result = await fetch(`${base}/${asset}`, { method: "HEAD" });
    assert.equal(result.status, 200, asset);
    assert.equal(result.headers.get("cache-control"), "no-store");
    assert.ok(Number(result.headers.get("content-length")) > 0, asset);
    assert.equal(await result.text(), "");
  }
});

test("private state, source code and traversal are not served", async () => {
  for (const path of ["/.env", "/.git/config", "/.local/portal", "/src/weft/portal.py", "/sandbox/run.mjs", "/website/README.md", "/website/%2e%2e/.env", "/%2e%2e%2f.env"]) {
    assert.equal((await fetch(`${base}${path}`)).status, 404, path);
  }
});

test("local landing never starts or proxies uploads to the sandbox", async () => {
  const config = await (await fetch(`${base}/api/session`)).json();
  assert.equal(config.enabled, false);
  assert.match(config.message, /npm run sandbox/);
  assert.equal((await fetch(`${base}/api/jobs`, { method: "POST", body: "test" })).status, 405);
  assert.equal((await fetch(`${base}/api/projects`)).status, 404);
});

test("demo routes send the browser to the local sandbox, without proxying private APIs", async () => {
  for (const path of ["/demo", "/demo/", "/demo/build", "/demo/runs", "/emo"]) {
    const response = await fetch(base + path, { redirect: "manual" });
    assert.equal(response.status, 302);
    const destination = ["/demo/", "/emo"].includes(path) ? "/demo" : path;
    assert.equal(response.headers.get("location"), `http://127.0.0.1:8765${destination}`);
  }
  assert.throws(() => landingServer({ sandboxPort: 0 }));
});

test("video seeking returns the exact requested bytes, including suffix ranges", async () => {
  const bytes = await readFile(new URL("../data/rolling-ball.mp4", import.meta.url));
  for (const [range, start, end] of [["bytes=10-29", 10, 29], ["bytes=-16", bytes.length - 16, bytes.length - 1]]) {
    const result = await fetch(`${base}/data/rolling-ball.mp4`, { headers: { Range: range } });
    assert.equal(result.status, 206);
    assert.equal(result.headers.get("content-range"), `bytes ${start}-${end}/${bytes.length}`);
    assert.deepEqual(Buffer.from(await result.arrayBuffer()), bytes.subarray(start, end + 1));
  }
  for (const range of ["bytes=999999999-", "bytes=20-10", "bytes=-0", "bytes=0-1,3-4"]) {
    assert.equal((await fetch(`${base}/data/rolling-ball.mp4`, { headers: { Range: range } })).status, 416);
  }
});

test("invalid paths and command options fail without exposing files or changing ports", async () => {
  assert.equal((await fetch(`${base}/%ZZ`)).status, 400);
  assert.equal(landingPort([]), 3000);
  assert.equal(landingPort(["--port", "3001"]), 3001);
  for (const args of [["--port", "0"], ["--port", "65536"], ["--host", "0.0.0.0"], ["--port"]]) {
    assert.throws(() => landingPort(args));
  }
});
