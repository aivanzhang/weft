import { request as httpRequest } from "node:http";

const assets = new Set([
  "/demo", "/demo/build", "/demo/runs", "/demo.js", "/demo.css",
  "/demo_build.js", "/demo_focus.js", "/demo-font.woff2", "/demo-wordmark.png",
  "/portal.js", "/portal.css", "/lab.js", "/lab.css", "/brand.css", "/weft-logo.png",
  "/avatar_retarget.js", "/avatar_worker.js", "/avatar_viewer.js", "/lab_appearance.js",
]);
const hopHeaders = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailer", "transfer-encoding", "upgrade"];

export function sandboxRoute(path) {
  return assets.has(path) || /^\/demo\/studio\/[A-Za-z0-9_-]{1,64}\/?$/.test(path) || path.startsWith("/api/") || /^\/labs\/[a-f0-9]{32}\//.test(path);
}

function endToEnd(headers) {
  const clean = { ...headers };
  for (const name of [...hopHeaders, ...(headers.connection || "").toLowerCase().split(",").map((s) => s.trim())]) {
    delete clean[name];
  }
  for (const name of Object.keys(clean)) {
    if (name === "forwarded" || name.startsWith("x-forwarded-")) delete clean[name];
  }
  return clean;
}

export function proxySandbox(request, response, sandboxPort, { transformHTML } = {}) {
  const fail = (code, message) => {
    if (response.destroyed) return;
    if (response.headersSent) return response.destroy();
    response.writeHead(code, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    response.end(JSON.stringify({ error: message }));
  };
  const host = request.headers.host;
  const port = request.socket.localPort;
  // Validate the public origin before translating it to the loopback-only backend.
  if (!["127.0.0.1", "localhost"].some((name) => host === `${name}:${port}`)
      || (request.headers.origin && request.headers.origin !== `http://${host}`)
      || request.headers["sec-fetch-site"] === "cross-site") {
    return fail(403, "Use this local site's own origin.");
  }
  if (!["GET", "HEAD", "POST"].includes(request.method)) return fail(405, "Method not allowed");
  if (request.headers["transfer-encoding"]) return fail(400, "Use a fixed-length upload");
  if (Number(request.headers["content-length"] || 0) > 100_000_000) return fail(413, "Choose a video up to 100 MB");
  if (sandboxPort === port) return fail(502, "The sandbox must use a separate internal port.");
  const upstreamHost = `127.0.0.1:${sandboxPort}`;
  const headers = endToEnd(request.headers);
  headers.host = upstreamHost;
  if (request.headers.origin) headers.origin = `http://${upstreamHost}`;
  const upstream = httpRequest({ hostname: "127.0.0.1", port: sandboxPort,
    method: request.method, path: request.url, headers, agent: false });
  upstream.setTimeout(180_000, () => upstream.destroy(new Error("Sandbox request timed out")));
  upstream.on("error", () => fail(502, "Sandbox unavailable. Start it with npm run demo or npm run sandbox."));
  upstream.on("response", async (incoming) => {
    const outgoing = endToEnd(incoming.headers);
    outgoing["cache-control"] = "no-store";
    if (transformHTML && request.method === "GET" && incoming.statusCode === 200
        && /^text\/html\b/.test(outgoing["content-type"] || "") && !outgoing["content-encoding"] && !request.headers.range) {
      try {
        const chunks = []; let size = 0;
        for await (const chunk of incoming) {
          size += chunk.length;
          if (size > 2_000_000) throw new Error("HTML exceeds the local refresh limit");
          chunks.push(chunk);
        }
        const html = transformHTML(Buffer.concat(chunks).toString("utf8"));
        outgoing["content-length"] = Buffer.byteLength(html);
        delete outgoing["accept-ranges"];
        response.writeHead(200, outgoing);
        response.end(html);
      } catch { fail(502, "Could not load the local page."); }
      return;
    }
    response.writeHead(incoming.statusCode, outgoing);
    incoming.on("error", () => response.destroy());
    incoming.pipe(response);
    response.on("close", () => incoming.destroy());
  });
  request.on("aborted", () => upstream.destroy());
  request.on("error", () => upstream.destroy());
  response.on("close", () => upstream.destroy());
  request.pipe(upstream);
}
