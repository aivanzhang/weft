import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { once } from "node:events";
import { createServer, request as httpRequest } from "node:http";
import { readFile } from "node:fs/promises";
import { landingPort, landingServer } from "./dev.mjs";
import { publicAssets } from "./public-assets.mjs";

let server, sandbox;
let base;
const forwarded = [];
before(async () => {
  sandbox = createServer(async (request, response) => {
    let body = "";
    for await (const chunk of request) body += chunk;
    forwarded.push({ url: request.url, method: request.method, headers: request.headers, body });
    if (request.url.startsWith("/api/") && !request.headers.authorization) {
      response.writeHead(401); response.end("Unauthorized"); return;
    }
    response.setHeader("Content-Type", "application/json");
    response.setHeader("X-Frame-Options", "SAMEORIGIN");
    response.end(JSON.stringify(forwarded.at(-1)));
  });
  sandbox.listen(0, "127.0.0.1");
  await once(sandbox, "listening");
  server = landingServer({ sandboxPort: sandbox.address().port });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  base = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  if (!server?.listening) return;
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
  sandbox.closeAllConnections();
  await new Promise((resolve) => sandbox.close(resolve));
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
  for (const route of ["/__weft_reload", "/__weft_reload.js"]) {
    assert.equal((await fetch(base + route)).status, 404, "Hot refresh is opt-in at the server boundary");
  }
  for (const path of ["/.env", "/.git/config", "/.local/portal", "/src/weft/portal.py", "/sandbox/run.mjs", "/website/README.md", "/website/%2e%2e/.env", "/%2e%2e%2f.env"]) {
    assert.equal((await fetch(`${base}${path}`)).status, 404, path);
  }
});

test("references routes serve the bibliography with working local assets and navigation", async () => {
  for (const route of ["/references", "/references/", "/website/references.html"]) {
    const response = await fetch(base + route);
    assert.equal(response.status, 200, route);
    assert.match(response.headers.get("content-type"), /text\/html/);
    const html = await response.text();
    assert.match(html, /<title>References — Weft<\/title>/);
    for (const [, href] of html.matchAll(/href="([^"]+)"/g)) {
      const target = new URL(href, base + route);
      if (target.origin !== base) continue;
      if (target.hash && target.pathname === route) {
        assert.ok(html.includes(`id="${target.hash.slice(1)}"`), href);
      } else {
        assert.equal((await fetch(target, { method: "HEAD" })).status, 200, href);
      }
    }
    assert.equal((await fetch(base + route, { method: "POST" })).status, 405);
  }
  assert.equal((await fetch(base + "/references/private")).status, 404);
});

test("local landing keeps uploads disabled and the sandbox enforces API authentication", async () => {
  const config = await (await fetch(`${base}/api/session`)).json();
  assert.equal(config.enabled, false);
  assert.match(config.message, /npm run sandbox/);
  assert.equal((await fetch(`${base}/api/jobs`, { method: "POST", body: "test" })).status, 401);
  assert.equal((await fetch(`${base}/api/projects`)).status, 401);
});

test("demo pages, assets and labs stay on the landing origin and retain query strings", async () => {
  for (const path of ["/demo?access_code=example", "/demo/build", "/demo/runs", "/demo/studio/curated-example", "/demo/studio/run-deadbeef/", "/demo.css", "/demo_focus.js", `/labs/${"a".repeat(32)}/lab.js`]) {
    const response = await fetch(base + path, { redirect: "manual" });
    assert.equal(response.status, 200, path);
    assert.equal(response.headers.get("location"), null);
    assert.equal(response.headers.get("x-frame-options"), "SAMEORIGIN");
    assert.equal((await response.json()).url, path);
  }
  for (const path of ["/demo/", "/emo"]) {
    const response = await fetch(base + path + "?access_code=example", { redirect: "manual" });
    assert.equal(response.headers.get("location"), "/demo?access_code=example");
  }
  assert.throws(() => landingServer({ sandboxPort: 0 }));
});

test("proxy streams authenticated posts with origin translation and no forwarded-header trust", async () => {
  const response = await fetch(base + "/api/demo/start?clip=example", {
    method: "POST", headers: { Authorization: "Bearer test-token", Origin: base,
      "Content-Type": "application/json", "X-Forwarded-Host": "attacker.example" }, body: '{"sample":true}',
  });
  assert.equal(response.status, 200);
  const sent = await response.json();
  assert.equal(sent.method, "POST");
  assert.equal(sent.body, '{"sample":true}');
  assert.equal(sent.headers.authorization, "Bearer test-token");
  assert.equal(sent.headers.host, `127.0.0.1:${sandbox.address().port}`);
  assert.equal(sent.headers.origin, `http://${sent.headers.host}`);
  assert.equal(sent.headers["x-forwarded-host"], undefined);
});

test("host spoofing and cross-origin requests never reach the sandbox", async () => {
  const count = forwarded.length;
  for (const headers of [{ Host: "attacker.example" }, { Origin: "https://attacker.example" },
    { Origin: "null" }, { "Sec-Fetch-Site": "cross-site" }]) {
    const status = await new Promise((resolve, reject) => {
      const request = httpRequest(base + "/api/demo/unlock", {
        method: "POST", headers: { ...headers, "Content-Length": "2" },
      }, (response) => { response.resume(); response.on("end", () => resolve(response.statusCode)); });
      request.on("error", reject);
      request.end("{}");
    });
    assert.equal(status, 403, JSON.stringify(headers));
  }
  assert.equal(forwarded.length, count);
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
