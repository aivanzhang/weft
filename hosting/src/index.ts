import { Container, getContainer } from "@cloudflare/containers";
import { DurableObject } from "cloudflare:workers";

const DAY = 86_400_000;
const MAX_RESULT = 32 * 1024 * 1024;
const MAX_SUMMARY = 16 * 1024;
const uuid = /^[a-f0-9-]{36}$/;
// One container image runs both fixed pipelines. The exercise lane decodes and poses a human
// clip, so it needs a larger upload and a longer deadline; the ball lane keeps its old,
// tighter limits so existing traffic is unaffected.
const PIPELINES = {
  ball: {
    upload: 10 * 1024 * 1024,
    read: 45_000,
    deadline: 210_000,
    title: "Your reconstructed world",
    constraints: {
      subject: "One isolated yellow ball rolling horizontally.",
      scene: "A fixed camera. This is not arbitrary-video or multi-object reconstruction.",
      motion: "The ball must move across the frame.",
      file: "MP4 up to 10 MB.",
      expect: "Under a minute.",
    },
    refusal:
      "This clip could not produce a verified environment. Try one yellow ball rolling horizontally with a fixed camera.",
    error:
      "Generation could not finish. Try a shorter clip with one clearly visible yellow ball.",
  },
  exercise: {
    upload: 40 * 1024 * 1024,
    read: 120_000,
    deadline: 420_000,
    title: "Your reconstructed exercise environment",
    // Published before upload, not only after a refusal: the envelope is narrow and a
    // visitor who knows it picks a clip that works instead of discovering why one did not.
    // Every line here is a constraint the pipeline actually enforces.
    constraints: {
      subject:
        "One person, upright, whole body from head to feet in frame throughout. Hanging and lying-down exercises are refused: the pose estimator cannot see them confidently, and a more accurate model measured worse.",
      scene:
        "A plain background and a fixed camera, in one continuous shot. Bystanders, wall mirrors and cuts each read as a second person.",
      motion:
        "A moving exercise. A still or near-still clip has no measurable motion to learn from.",
      file:
        "MP4 up to 40 MB. Longer clips are thinned across their whole length, and if only part of a clip is usable that part is found and used automatically.",
      expect:
        "About 1 to 3 minutes: measured 66 s for a clip that admits first try, 185 s for one that needs a narrower window found automatically. A clip outside the envelope comes back with the reason in under a minute: across 19 uncurated inputs the slowest refusal was 42 s, and a wrong file type is rejected in under 3 s.",
    },
    refusal:
      "This clip could not produce a verified environment. Use one person, head to feet in frame, a plain background and a fixed camera in a single continuous shot.",
    error:
      "Generation could not finish. Try a shorter single-person clip with a fixed camera and no cuts.",
  },
} as const;
type Pipeline = keyof typeof PIPELINES;
// What a visitor should change, per refusal the exercise pipeline classifies. The pipeline's
// own wording carries the detail; this only adds the advice, and an unknown code adds none.
const ADVICE: Record<string, string> = {
  multiple_people:
    "More than one person was detected. Use a single continuous shot of one person, with no bystanders, mirrors or camera cuts.",
  no_person_visible:
    "No one was visible for long enough. Keep one person in frame from head to feet throughout.",
  no_motion:
    "No exercise motion was measured. Upload a moving exercise rather than a still or near-still clip.",
  root_reference_gap:
    "The body was lost for too long mid-clip. Avoid occlusions and keep the whole body in frame.",
  body_anchor:
    "The opening of the clip could not be anchored. Start with the person standing still, fully in frame, before the exercise begins.",
  undecodable:
    "This file could not be decoded as a video of at least 12 frames. Re-export it as an MP4.",
  too_large: "This clip is too long or too large. Upload a shorter clip.",
  timeout: "This clip took too long to process. Upload a shorter clip.",
  // Host misconfiguration, not the visitor's clip. Never tell them to change the video.
  pose_cpu:
    "Pose estimation is unavailable on this server. Nothing is wrong with your clip; please try again later.",
};
const pipelineOf = (kind: string | undefined): Pipeline =>
  kind === "exercise" ? "exercise" : "ball";
type Job = {
  id: string;
  owner: string;
  ip: string;
  key: string;
  created: number;
  status: string;
  result: string | null;
  message: string | null;
  lane?: string;
  kind?: string;
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
  // 30 s, not the 5 m default. Container provisioning is about 22 s of a submission -- pose
  // and the ML imports are 0.52 s, so the cost is starting the sandbox, not loading it -- and
  // a judge submitting several clips in a row paid it every time. Thirty seconds carries a
  // warm instance across consecutive submissions without holding idle capacity: 8 GiB for
  // 30 s is roughly $0.0006 per job, about $0.05 a day against the 80-job daily caps, where
  // the 5 m default would have been ten times that for no extra benefit.
  sleepAfter = "30s";
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
    for (const column of [
      "lane TEXT NOT NULL DEFAULT 'public'",
      // Rows predating the exercise lane were all ball submissions.
      "kind TEXT NOT NULL DEFAULT 'ball'",
    ])
      try {
        ctx.storage.sql.exec(`ALTER TABLE jobs ADD COLUMN ${column}`);
      } catch {
        // Column already present.
      }
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
  async reserve(
    owner: string,
    ip: string,
    key: string,
    judge = false,
    kind: Pipeline = "ball",
  ): Promise<{ job: Job; fresh: boolean } | { refused: string }> {
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
    // The judge lane shares the single generation slot but keeps its own bounded
    // daily budget, so public traffic can never exhaust judging capacity.
    const publicDaily = daily.filter((job) => job.lane !== "judge");
    // A momentarily busy queue and an exhausted daily budget both used to refuse with "try
    // again later", which is misleading for the budget: it does not clear for hours.
    if (active.length >= 3)
      return { refused: "Three environments are already generating. Try again in about a minute." };
    if (judge) {
      if (daily.filter((job) => job.lane === "judge").length >= 60)
        return { refused: "This access code has used its daily generation budget. Each run frees up 24 hours later." };
    } else if (publicDaily.length >= 20)
      return { refused: "The shared daily generation budget is used up. Please try again tomorrow." };
    else if (
      publicDaily.filter((job) => job.owner === owner || job.ip === ip).length >= 3
    )
      return { refused: "You have used your three generations for today. Please try again tomorrow." };
    const job: Job = {
      id: crypto.randomUUID(),
      owner,
      ip,
      key,
      created: Date.now(),
      status: "uploading",
      result: null,
      message: null,
      lane: judge ? "judge" : "public",
      kind,
    };
    this.ctx.storage.sql.exec(
      "INSERT INTO jobs (id,owner,ip,key,created,status,result,message,lane,kind) VALUES (?,?,?,?,?,?,NULL,NULL,?,?)",
      job.id,
      owner,
      ip,
      key,
      job.created,
      job.status,
      job.lane,
      job.kind,
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
    // Recovery never reruns a possibly completed trainer after a crash. The summary lives in
    // its own object because an exercise summary exceeds R2's 2 KB custom-metadata budget.
    for (const job of this.rows("SELECT * FROM jobs WHERE status='running'")) {
      const saved = await this.env.UPLOADS.head(`${job.id}/environment.zip`);
      await getContainer(this.env.GENERATOR, "weft-video-generation")
        .stop()
        .catch(() => {});
      const summary =
        saved?.customMetadata?.status === "complete"
          ? ((await this.env.UPLOADS.get(`${job.id}/result.json`)
              .then((object) => object?.text())
              .catch(() => null)) ?? null)
          : null;
      this.finish(
        job,
        saved?.customMetadata?.status ?? "failed",
        summary,
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
        `${job.id}/result.json`,
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
      const pipeline = PIPELINES[pipelineOf(job.kind)];
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
              "X-Weft-Pipeline": pipelineOf(job.kind),
            },
            signal: AbortSignal.timeout(pipeline.deadline),
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
        if (result.length > MAX_SUMMARY) throw new Error("result_too_large");
        if (status === "complete") JSON.parse(result);
        // A refusal the pipeline explained is reported as explained. Only the closed code
        // selects advice; the detail is the pipeline's own words, length-bounded.
        let refusal: string = pipeline.refusal;
        if (status === "failed") {
          const encoded = response.headers.get("X-Weft-Failure");
          if (encoded && encoded.length <= 4096)
            try {
              const reported = JSON.parse(atob(encoded)) as {
                code?: unknown;
                detail?: unknown;
              };
              const advice =
                typeof reported.code === "string" ? ADVICE[reported.code] : undefined;
              const detail =
                typeof reported.detail === "string"
                  ? reported.detail.slice(0, 400)
                  : "";
              if (advice || detail)
                refusal = [advice, detail && `Reported: ${detail}`]
                  .filter(Boolean)
                  .join(" ");
            } catch {
              // Unparseable diagnostics fall back to the pipeline-wide refusal.
            }
        }
        // The Container SDK wraps bodies in an unknown-length stream; R2 requires an exact length.
        const transfer = new FixedLengthStream(size);
        const abort = new AbortController();
        const pumping = response.body.pipeTo(transfer.writable, { signal: abort.signal });
        try {
          await Promise.all([
            pumping,
            this.env.UPLOADS.put(`${job.id}/environment.zip`, transfer.readable, {
              httpMetadata: { contentType: "application/zip" },
              customMetadata: { status },
            }),
          ]);
        } catch (error) {
          abort.abort();
          await pumping.catch(() => {});
          throw error;
        }
        // Durable copy of the summary for crash recovery; the row below is what visitors read.
        if (result)
          await this.env.UPLOADS.put(`${job.id}/result.json`, result, {
            httpMetadata: { contentType: "application/json" },
          });
        this.finish(job, status, result || null, status === "failed" ? refusal : null);
      } catch (error) {
        console.error("weft_generation_failed", error instanceof Error ? error.message.slice(0,240) : "unknown");
        this.finish(job, "failed", null, pipeline.error);
      }
      // Deliberately not stopping the container here. The original "no idle paid container
      // between visitor jobs" was written for the ball lane, where a job took four seconds
      // and provisioning dominated nothing; the exercise lane pays ~22 s for it. sleepAfter
      // above bounds the exposure instead, so a queued follow-up finds the instance warm.
      // A crashed or wedged instance is still reclaimed by the recovery path above, which
      // stops it explicitly before reading the saved archive.
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

const tooLarge = (limit: number) =>
  new Rejected(
    `Choose an MP4 smaller than ${Math.round(limit / (1024 * 1024))} MB.`,
    413,
  );

async function limitedBody(request: Request, limit: number, read: number) {
  const length = Number(request.headers.get("Content-Length"));
  if (!(length > 0 && length <= limit) || !request.body) throw tooLarge(limit);
  const reader = request.body.getReader();
  const bytes = new Uint8Array(length);
  let size = 0;
  const timer = setTimeout(() => {
    void reader.cancel();
  }, read);
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

// Predefined avatar appearances, pre-imported at build time under the lab's own
// /api/avatar/<key>/render.json contract so the existing client needs only a base path.
// Keys are content digests, so these are immutable and need no session: a visitor can
// pick and substitute an appearance before, during or without any submission.
const AVATAR_KEY = /^[a-f0-9]{64}$/;
const AVATAR_FILE = /^(render\.json|texture-\d{1,2})$/;

async function avatars(request: Request, env: Env, url: URL) {
  if (!["GET", "HEAD"].includes(request.method))
    throw new Rejected("Not found.", 404);
  let asset: string;
  let type = "application/json";
  if (url.pathname === "/api/avatars") asset = "/website/avatars/index.json";
  else {
    const match = url.pathname.match(/^\/api\/avatar\/([^/]+)\/([^/]+)$/);
    if (!match || !AVATAR_KEY.test(match[1]) || !AVATAR_FILE.test(match[2]))
      throw new Rejected("Avatar resource does not exist.", 404);
    asset = `/website/avatars/${match[1]}/${match[2]}`;
    if (match[2] !== "render.json") type = "application/octet-stream";
  }
  const response = await env.ASSETS.fetch(new Request(url.origin + asset, { method: "GET" }));
  if (!response.ok) throw new Rejected("Avatar resource does not exist.", 404);
  const headers = new Headers({
    "Content-Type": type,
    // Digest-keyed, so a cached copy can never be the wrong avatar.
    "Cache-Control": "public, max-age=31536000, immutable",
  });
  // The client sizes a render payload before fetching it, and a HEAD carries no body to
  // measure, so the asset store's length has to be forwarded rather than dropped.
  const length = response.headers.get("Content-Length");
  if (length) headers.set("Content-Length", length);
  return new Response(request.method === "HEAD" ? null : response.body, { status: 200, headers });
}

async function api(request: Request, env: Env) {
  const url = new URL(request.url);
  if (url.pathname === "/api/avatars" || url.pathname.startsWith("/api/avatar/"))
    return avatars(request, env, url);
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
      // The client validates size before uploading instead of discovering a 413.
      pipelines: Object.fromEntries(
        Object.entries(PIPELINES).map(([name, limits]) => [
          name,
          { max_upload_bytes: limits.upload, constraints: limits.constraints },
        ]),
      ),
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
  const accessCode = request.headers.get("X-Weft-Access-Code") ?? "";
  const judgeSecret = (env as { JUDGE_ACCESS_CODE?: string }).JUDGE_ACCESS_CODE;
  let judge = false;
  if (accessCode) {
    if (!judgeSecret || (await hash(accessCode)) !== (await hash(judgeSecret)))
      throw new Rejected("Access code not recognized.", 403);
    judge = true;
  }
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
    const requested = request.headers.get("X-Weft-Pipeline") ?? "ball";
    if (!(requested in PIPELINES))
      throw new Rejected("Unknown pipeline requested.");
    const kind = requested as Pipeline;
    const pipeline = PIPELINES[kind];
    const length = Number(request.headers.get("Content-Length"));
    if (!(length > 0 && length <= pipeline.upload)) throw tooLarge(pipeline.upload);
    const ip = await hash(
      (request.headers.get("CF-Connecting-IP") ?? "unknown") +
        ":" +
        Math.floor(Date.now() / DAY),
    );
    const reserved = await queue.reserve(owner, ip, key, judge, kind);
    if ("refused" in reserved) throw new Rejected(reserved.refused, 429);
    if (reserved.fresh) {
      try {
        const body = await limitedBody(request, pipeline.upload, pipeline.read);
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
    const kind = pipelineOf(job.kind);
    const result = job.result ? JSON.parse(job.result) : null;
    if (result)
      Object.assign(result, {
        video: `/api/jobs/${job.id}/source`,
        download: `/api/jobs/${job.id}/download`,
        title: PIPELINES[kind].title,
      });
    return json({
      id: job.id,
      status: job.status,
      kind,
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
