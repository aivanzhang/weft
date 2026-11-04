import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { pipeline } from "node:stream/promises";
import { publicAssets } from "./public-assets.mjs";
import { proxySandbox, sandboxRoute } from "./sandbox-proxy.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const allowed = new Set(publicAssets.map((name) => `/${name}`));
const types = {
  ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8", ".json": "application/json",
  ".png": "image/png", ".jpg": "image/jpeg", ".mp4": "video/mp4",
  ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8",
};

export function landingServer({ sandboxPort = 8765 } = {}) {
  if (!Number.isInteger(sandboxPort) || sandboxPort < 1 || sandboxPort > 65535) {
    throw new Error("Sandbox port must be between 1 and 65535");
  }
  return createServer(async (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    response.setHeader("X-Content-Type-Options", "nosniff");
    const reply = (code, text, headers = {}) => {
      response.writeHead(code, { "Content-Type": "text/plain; charset=utf-8", ...headers });
      response.end(request.method === "HEAD" ? undefined : text);
    };
    try {
      const parsed = new URL(request.url, "http://localhost");
      const path = decodeURIComponent(parsed.pathname);
      if (["/demo/", "/emo"].includes(path)) {
        return reply(302, "", { Location: `/demo${parsed.search}` });
      }
      if (path !== "/api/session" && sandboxRoute(path)) {
        return proxySandbox(request, response, sandboxPort);
      }
      if (!["GET", "HEAD"].includes(request.method)) {
        return reply(405, "The landing page does not accept uploads.", { Allow: "GET, HEAD" });
      }
      if (path === "/") return reply(302, "", { Location: "/website/index.html" });
      if (path === "/api/session") {
        return reply(200, JSON.stringify({
          enabled: false,
          message: "Open /demo to upload videos and use the sandbox. Start the backend with npm run sandbox if needed.",
        }), { "Content-Type": "application/json" });
      }
      if (!allowed.has(path)) return reply(404, "Not found");
      const file = resolve(root, `.${path}`);
      // An allowlisted path must not become a symlink to private files.
      if (await realpath(file) !== file) return reply(404, "Not found");
      const info = await stat(file);
      let start = 0;
      let end = info.size - 1;
      let status = 200;
      const headers = { "Content-Type": types[extname(file)], "Accept-Ranges": "bytes" };
      if (request.headers.range) {
        const match = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range);
        if (!match || (!match[1] && !match[2])) {
          return reply(416, "Invalid range", { "Content-Range": `bytes */${info.size}` });
        }
        start = match[1] ? Number(match[1]) : Math.max(0, info.size - Number(match[2]));
        end = match[1] && match[2] ? Math.min(Number(match[2]), end) : end;
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= info.size) {
          return reply(416, "Invalid range", { "Content-Range": `bytes */${info.size}` });
        }
        status = 206;
        headers["Content-Range"] = `bytes ${start}-${end}/${info.size}`;
      }
      headers["Content-Length"] = end - start + 1;
      response.writeHead(status, headers);
      if (request.method === "HEAD") return response.end();
      await pipeline(createReadStream(file, { start, end }), response);
    } catch (error) {
      if (response.headersSent) response.destroy();
      else reply(error instanceof URIError ? 400 : 404, "Not found");
    }
  });
}

export function landingPort(args) {
  if (!args.length) return 3000;
  if (args.length !== 2 || args[0] !== "--port" || !/^\d+$/.test(args[1])) {
    throw new Error("Usage: npm run dev -- [--port 3000]");
  }
  const port = Number(args[1]);
  if (port < 1 || port > 65535) throw new Error("Port must be between 1 and 65535");
  return port;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const port = landingPort(process.argv.slice(2));
    const server = landingServer({ sandboxPort: Number(process.env.WEFT_SANDBOX_PORT || 8765) });
    server.on("error", (error) => {
      console.error(error.code === "EADDRINUSE"
        ? `Port ${port} is in use. Choose another with npm run dev -- --port 3001.`
        : error.message);
      process.exitCode = 1;
    });
    server.listen(port, "127.0.0.1", () => {
      console.log(`Weft landing page: http://127.0.0.1:${port}/`);
      console.log("Refresh after edits. Sandbox/API workers are not started. Run npm run sandbox separately.");
    });
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
