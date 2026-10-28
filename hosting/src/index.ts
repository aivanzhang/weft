import { Container, getContainer } from "@cloudflare/containers";
import { DurableObject } from "cloudflare:workers";

const DAY = 86_400_000;
const MAX_UPLOAD = 10 * 1024 * 1024;
const MAX_RESULT = 32 * 1024 * 1024;
const uuid = /^[a-f0-9-]{36}$/;
type Job = {
  id: string;
  owner: string;
  ip: string;
  key: string;
  created: number;
  status: string;
  result: string | null;
  message: string | null;
};
class Rejected extends Error {
  constructor(
    message: string,
    public status = 400,
  ) {
    super(message);
  }
}
const json = (body: unknown, status = 200) =>
  Response.json(body, {
    status,
    headers: { "Cache-Control": "private, no-store" },
  });
const hash = async (text: string) =>
  Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
    ),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
const random = () =>
  Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");

export class GenerationContainer extends Container<Env> {
  defaultPort = 8080;
  sleepAfter = "5m";
  enableInternet = false;
}

// One admission ledger coordinates this bounded, single-runner service, not static traffic.
export class SubmissionQueue extends DurableObject<Env> {
  private processing = false;
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.storage.sql.exec(
      `CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, owner TEXT NOT NULL, ip TEXT NOT NULL, key TEXT NOT NULL, created INTEGER NOT NULL, status TEXT NOT NULL, result TEXT, message TEXT, UNIQUE(owner,key)); CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created); CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner,created);`,
    );
  }
  private rows(query: string, ...args: (string | number)[]) {
    return this.ctx.storage.sql.exec<Job>(query, ...args).toArray();
  }
  async latest(owner: string) {
    return (
      this.rows(
        "SELECT * FROM jobs WHERE owner=? AND created>? AND status IN ('uploading','queued','running','complete') ORDER BY created DESC LIMIT 1",
        owner,
        Date.now() - DAY,
      )[0] ?? null
    );
  }
  async getJob(id: string, owner: string) {
    return (
      this.rows("SELECT * FROM jobs WHERE id=? AND owner=?", id, owner)[0] ??
      null
    );
  }
  async reserve(owner: string, ip: string, key: string) {
    const old = this.rows(
      "SELECT * FROM jobs WHERE owner=? AND key=?",
      owner,
      key,
    )[0];
    if (old) return { job: old, fresh: false };
    const active = this.rows(
      "SELECT * FROM jobs WHERE status IN ('uploading','queued','running')",
    );
    const owned = active.find((job) => job.owner === owner);
    if (owned) return { job: owned, fresh: false };
    const daily = this.rows(
      "SELECT * FROM jobs WHERE created>?",
      Date.now() - DAY,
    );
    if (
      active.length >= 3 ||
      daily.length >= 20 ||
      daily.filter((job) => job.owner === owner || job.ip === ip).length >= 3
    )
      return null;
    const job: Job = {
      id: crypto.randomUUID(),
      owner,
      ip,
      key,
      created: Date.now(),
      status: "uploading",
      result: null,
      message: null,
    };
    this.ctx.storage.sql.exec(
      "INSERT INTO jobs VALUES (?,?,?,?,?,?,NULL,NULL)",
      job.id,
      owner,
      ip,
      key,
      job.created,
      job.status,
    );
    await this.schedule(Date.now() + 60_000);
    return { job, fresh: true };
  }
  async uploaded(id: string, owner: string) {
    this.ctx.storage.sql.exec(
      "UPDATE jobs SET status='queued' WHERE id=? AND owner=? AND status='uploading'",
      id,
      owner,
    );
    await this.schedule(Date.now() + 100);
  }
  async failed(id: string, owner: string) {
    this.ctx.storage.sql.exec(
      "UPDATE jobs SET status='failed', message='Upload did not finish. Try again.' WHERE id=? AND owner=? AND status='uploading'",
      id,
      owner,
    );
  }
  private async schedule(at: number) {
    const current = await this.ctx.storage.getAlarm();
    if (current === null || current > at) await this.ctx.storage.setAlarm(at);
  }
  private finish(
    job: Job,
    status: string,
    result: string | null,
    message: string | null,
  ) {
    this.ctx.storage.sql.exec(
      "UPDATE jobs SET status=?,result=?,message=? WHERE id=?",
      status,
      result,
      message,
      job.id,
    );
  }
  async alarm() {
    if (this.processing) {
      await this.schedule(Date.now() + 60_000);
      return;
    }
    this.processing = true;
    try {
      await this.processNext();
    } finally {
      this.processing = false;
    }
  }
  private async processNext() {
    const now = Date.now();
    this.ctx.storage.sql.exec(
      "UPDATE jobs SET status='failed',message='Upload was interrupted. Try again.' WHERE status='uploading' AND created<?",
      now - 60_000,
    );
    // Recovery never reruns a possibly completed trainer after a crash.
    for (const job of this.rows("SELECT * FROM jobs WHERE status='running'")) {
      const saved = await this.env.UPLOADS.head(`${job.id}/environment.zip`);
      await getContainer(this.env.GENERATOR, "weft-video-generation")
        .stop()
        .catch(() => {});
      this.finish(
        job,
        saved?.customMetadata?.status ?? "failed",
        saved?.customMetadata?.result ?? null,
        saved ? null : "Generation was interrupted. Please try again.",
      );
    }
    for (const job of this.rows(
      "SELECT * FROM jobs WHERE created<? AND status NOT IN ('expired','running','uploading')",
      now - DAY,
    )) {
      await this.env.UPLOADS.delete([
        `${job.id}/source.mp4`,
        `${job.id}/environment.zip`,
      ]);
      this.finish(job, "expired", null, null);
    }
    this.ctx.storage.sql.exec(
      "DELETE FROM jobs WHERE status='expired' AND created<?",
      now - 7 * DAY,
    );
    const job = this.rows(
      "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1",
    )[0];
    if (job) {
      this.ctx.storage.sql.exec(
        "UPDATE jobs SET status='running' WHERE id=?",
        job.id,
      );
      const container = getContainer(
        this.env.GENERATOR,
        "weft-video-generation",
      );
      try {
        const source = await this.env.UPLOADS.get(`${job.id}/source.mp4`);
        if (!source) throw new Error("input_missing");
        const response = await container.fetch(
          new Request("http://container/generate", {
            method: "POST",
            body: source.body,
            headers: {
              "Content-Type": "video/mp4",
              "Content-Length": String(source.size),
            },
            signal: AbortSignal.timeout(210_000),
          }),
        );
        const size = Number(response.headers.get("Content-Length"));
        if (!response.ok || !response.body || !(size > 0 && size <= MAX_RESULT))
          throw new Error(`generation_response_${response.status}_bytes_${size}`);
        const status =
          response.headers.get("X-Weft-Status") === "complete"
            ? "complete"
            : "failed";
        const result =
          status === "complete"
            ? atob(response.headers.get("X-Weft-Result") ?? "")
            : "";
        if (result.length > 8192) throw new Error("result_too_large");
        if (status === "complete") JSON.parse(result);
        // The Container SDK wraps bodies in an unknown-length stream; R2 requires an exact length.
        const transfer = new FixedLengthStream(size);
        const abort = new AbortController();
        const pumping = response.body.pipeTo(transfer.writable, { signal: abort.signal });
        try {
          await Promise.all([
            pumping,
            this.env.UPLOADS.put(`${job.id}/environment.zip`, transfer.readable, {
              httpMetadata: { contentType: "application/zip" },
              customMetadata: { status, result },
            }),
          ]);
        } catch (error) {
          abort.abort();
          await pumping.catch(() => {});
          throw error;
        }
        this.finish(
          job,
          status,
          result || null,
          status === "failed"
            ? "This clip could not produce a verified environment. Try one yellow ball rolling horizontally with a fixed camera."
            : null,
        );
      } catch (error) {
        console.error("weft_generation_failed", error instanceof Error ? error.message.slice(0,240) : "unknown");
        this.finish(
          job,
          "failed",
          null,
          "Generation could not finish. Try a shorter clip with one clearly visible yellow ball.",
        );
      } finally {
        // No idle paid container between visitor jobs.
        await container.stop().catch(() => {});
      }
    }
    const next = this.rows(
      "SELECT * FROM jobs WHERE status IN ('uploading','queued') ORDER BY created LIMIT 1",
    )[0];
    const expiry = this.ctx.storage.sql
      .exec<{ at: number | null }>(
        "SELECT MIN(created)+? as at FROM jobs WHERE status!='expired'",
        DAY,
      )
      .one().at;
    if (next || expiry)
      await this.ctx.storage.setAlarm(
        next ? Date.now() + 1000 : Math.max(Date.now() + 1000, expiry!),
      );
  }
}

async function limitedBody(request: Request, limit: number) {
  const length = Number(request.headers.get("Content-Length"));
  if (!(length > 0 && length <= limit) || !request.body)
    throw new Rejected("Choose an MP4 smaller than 10 MB.", 413);
  const reader = request.body.getReader();
  const bytes = new Uint8Array(length);
  let size = 0;
  const timer = setTimeout(() => {
    void reader.cancel();
  }, 45_000);
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (size + value.length > length)
        throw new Rejected("Upload exceeds the size limit.", 413);
      bytes.set(value, size);
      size += value.length;
    }
    if (
      size !== length ||
      new TextDecoder().decode(bytes.subarray(4, 8)) !== "ftyp"
    )
      throw new Rejected("This file is not a supported MP4.");
    return bytes;
  } finally {
    clearTimeout(timer);
    await reader.cancel().catch(() => {});
  }
}

async function api(request: Request, env: Env) {
  const url = new URL(request.url);
  const queue = env.SUBMISSIONS.getByName("weft-admission-v1");
  const token = request.headers
    .get("Cookie")
    ?.match(/(?:^|;\s*)weft_session=([a-f0-9]{64})(?:;|$)/)?.[1];
  const enabled = env.SUBMISSIONS_ENABLED === "true";
  if (url.pathname === "/api/session" && request.method === "GET") {
    if (!enabled) return json({ enabled: false });
    const value = token ?? random();
    const owner = await hash(value);
    const job = await queue.latest(owner);
    const response = json({
      enabled: true,
      csrf: await hash(value + ":csrf"),
      job: job?.id ?? null,
      active: job
        ? ["uploading", "queued", "running"].includes(job.status)
        : false,
    });
    response.headers.set(
      "Set-Cookie",
      `weft_session=${value}; Path=/; Max-Age=86400; Secure; HttpOnly; SameSite=Strict`,
    );
    return response;
  }
  if (!enabled) throw new Rejected("Submissions are not open yet.", 503);
  if (!token) throw new Rejected("Reload this page before submitting.", 403);
  const owner = await hash(token);
  if (url.pathname === "/api/jobs" && request.method === "POST") {
    if (
      request.headers.get("Origin") !== env.PUBLIC_ORIGIN ||
      request.headers.get("X-Weft-CSRF") !== (await hash(token + ":csrf"))
    )
      throw new Rejected("Reload this page before submitting.", 403);
    if (request.headers.get("X-Weft-Consent") !== "yes")
      throw new Rejected("Confirm you have permission to process this video.");
    if (request.headers.get("Content-Type") !== "video/mp4")
      throw new Rejected("Choose an MP4 video.", 415);
    const key = request.headers.get("Idempotency-Key") ?? "";
    if (!uuid.test(key)) throw new Rejected("Missing upload identifier.");
    const length = Number(request.headers.get("Content-Length"));
    if (!(length > 0 && length <= MAX_UPLOAD))
      throw new Rejected("Choose an MP4 smaller than 10 MB.", 413);
    const ip = await hash(
      (request.headers.get("CF-Connecting-IP") ?? "unknown") +
        ":" +
        Math.floor(Date.now() / DAY),
    );
    const reserved = await queue.reserve(owner, ip, key);
    if (!reserved)
      throw new Rejected(
        "The generation queue is full. Please try again later.",
        429,
      );
    if (reserved.fresh) {
      try {
        const body = await limitedBody(request, MAX_UPLOAD);
        await env.UPLOADS.put(`${reserved.job.id}/source.mp4`, body, {
          httpMetadata: { contentType: "video/mp4" },
        });
        await queue.uploaded(reserved.job.id, owner);
      } catch (error) {
        await queue.failed(reserved.job.id, owner);
        throw error;
      }
    }
    return json({ id: reserved.job.id }, 202);
  }
  const match = url.pathname.match(
    /^\/api\/jobs\/([a-f0-9-]{36})(?:\/(source|download))?$/,
  );
  if (!match || !["GET", "HEAD"].includes(request.method))
    throw new Rejected("Not found.", 404);
  const job = await queue.getJob(match[1], owner);
  if (!job) throw new Rejected("Result not found.", 404);
  if (Date.now() - job.created >= DAY)
    return match[2]
      ? json({ error: "Result expired." }, 410)
      : json({
          status: "expired",
          message: "This result has expired. Submit a new clip to start again.",
        });
  if (!match[2]) {
    const result = job.result ? JSON.parse(job.result) : null;
    if (result)
      Object.assign(result, {
        video: `/api/jobs/${job.id}/source`,
        download: `/api/jobs/${job.id}/download`,
        title: "Your reconstructed world",
      });
    return json({
      id: job.id,
      status: job.status,
      message: job.message,
      result,
    });
  }
  if (job.status !== "complete")
    throw new Rejected("Result is not ready.", 404);
  const download = match[2] === "download";
  const object = await env.UPLOADS.get(
    `${job.id}/${download ? "environment.zip" : "source.mp4"}`,
    { range: request.headers },
  );
  if (!object) throw new Rejected("Result not found.", 404);
  const headers = new Headers({
    "Cache-Control": "private, no-store",
    "Content-Type": download ? "application/zip" : "video/mp4",
    "Accept-Ranges": "bytes",
  });
  if (download)
    headers.set(
      "Content-Disposition",
      'attachment; filename="weft-environment.zip"',
    );
  if (object.range && "offset" in object.range) {
    const offset = object.range.offset ?? 0,
      length = object.range.length ?? object.size;
    headers.set(
      "Content-Range",
      `bytes ${offset}-${offset + length - 1}/${object.size}`,
    );
    headers.set("Content-Length", String(length));
  } else headers.set("Content-Length", String(object.size));
  return new Response(request.method === "HEAD" ? null : object.body, {
    status: object.range ? 206 : 200,
    headers,
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    let response: Response;
    try {
      if (url.pathname.startsWith("/api/")) response = await api(request, env);
      else if (url.pathname === "/health")
        response = json({
          status: "ok",
          version: env.RELEASE,
          submissions: env.SUBMISSIONS_ENABLED === "true",
        });
      else if (url.pathname === "/")
        response = Response.redirect(url.origin + "/website/index.html", 302);
      else response = await env.ASSETS.fetch(request);
    } catch (error) {
      if (!(error instanceof Rejected)) console.error("weft_request_failed", error instanceof Error ? error.message.slice(0, 240) : "unknown");
      response = json(
        {
          error:
            error instanceof Rejected
              ? error.message
              : "The service is temporarily unavailable.",
        },
        error instanceof Rejected ? error.status : 503,
      );
    }
    const secured = new Response(response.body, response);
    secured.headers.set("X-Content-Type-Options", "nosniff");
    secured.headers.set("Referrer-Policy", "no-referrer");
    secured.headers.set("X-Frame-Options", "DENY");
    secured.headers.set(
      "Content-Security-Policy",
      "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
    );
    return secured;
  },
} satisfies ExportedHandler<Env>;
