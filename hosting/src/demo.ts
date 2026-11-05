// SPDX-License-Identifier: AGPL-3.0-or-later
// The /api/demo surface the repository's own demo page calls. The page is served unchanged
// from src/weft/assets, so this implements its contract rather than the page implementing
// ours. What the hosted pipeline cannot do, it says so instead of simulating.
//
// Security model, deliberately reusing the session the rest of the API already issues:
//   * the bearer token is hash(session cookie + ":demo"), so it is unforgeable without the
//     256-bit HttpOnly cookie, needs no server-side store, and dies with the session;
//   * showcase listings and their media are public, and are the only public reads. They can
//     only ever name curated, committed, licensed files — an id is matched against the seed
//     list, never used to build a path;
//   * every other read requires the token, and anything naming a visitor's own run is
//     scoped to hash(cookie) so one session cannot read another's;
//   * private responses are no-store; public showcase media is immutable.
import seeds from "../../src/weft/assets/demo_examples.json";

type Seed = {
  id: string;
  title: string;
  caption?: string;
  metrics?: string;
  category?: string;
  video: string;
  source_video?: string;
  source_credit?: string;
  source_page?: string;
  input_label?: string;
  result_label?: string;
  input_note?: string;
  result_note?: string;
  scope?: string;
  comparison_note?: string;
};

const SEEDS = seeds as Seed[];
const SEED_BY_ID = new Map(SEEDS.map((seed) => [seed.id, seed]));
const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

const hex = async (text: string) =>
  Array.from(
    new Uint8Array(
      await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)),
    ),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");

const privately = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "private, no-store",
    },
  });

/** A fresh session value, in the same shape the rest of the API issues and reads. */
const newSession = () =>
  Array.from(crypto.getRandomValues(new Uint8Array(32)), (b) =>
    b.toString(16).padStart(2, "0"),
  ).join("");

export const demoCookie = (request: Request) =>
  request.headers
    .get("Cookie")
    ?.match(/(?:^|;\s*)weft_session=([a-f0-9]{64})(?:;|$)/)?.[1] ?? "";

/** The page's bearer token for a session: derived, never stored, never guessable. */
export const demoToken = (cookie: string) => hex(cookie + ":demo");

/**
 * Bearer header or ?token= (media elements cannot send headers). Compared as digests so a
 * mismatch costs the same time whatever the prefix, and an absent cookie can never match.
 */
async function authorized(request: Request, url: URL) {
  const cookie = demoCookie(request);
  if (!cookie) return false;
  const offered =
    request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "") ??
    url.searchParams.get("token") ??
    "";
  if (!offered) return false;
  const [a, b] = await Promise.all([
    hex(offered + ":compare"),
    hex((await demoToken(cookie)) + ":compare"),
  ]);
  return a === b;
}

/** Curated examples only: committed media with recorded credit, never a visitor upload. */
function showcaseItems() {
  return SEEDS.map((seed) => {
    const item: Record<string, unknown> = {
      id: seed.id,
      title: seed.title,
      caption: seed.caption ?? "",
      metrics: seed.metrics ?? "",
      kind: "verified",
      readonly: true,
      video_url: `/api/demo/showcase/${seed.id}/video`,
    };
    if (seed.source_video) {
      item.source_video_url = `/api/demo/showcase/${seed.id}/source`;
      item.source_credit = seed.source_credit ?? "";
      item.source_page = seed.source_page ?? "";
      for (const key of [
        "category",
        "input_label",
        "result_label",
        "input_note",
        "result_note",
        "scope",
        "comparison_note",
      ] as const)
        if (seed[key] !== undefined) item[key] = seed[key];
    }
    return item;
  });
}

/** Media for a curated example. The id selects a seed; the seed supplies the path. */
async function showcaseMedia(env: Env, url: URL, id: string, which: "video" | "source") {
  const seed = ID.test(id) ? SEED_BY_ID.get(id) : undefined;
  const path = which === "video" ? seed?.video : seed?.source_video;
  if (!path) return privately({ error: "Not found." }, 404);
  const asset = await env.ASSETS.fetch(new Request(`${url.origin}/${path}`));
  if (!asset.ok) return privately({ error: "Not found." }, 404);
  const response = new Response(asset.body, asset);
  response.headers.set("Content-Type", "video/mp4");
  // Curated, licensed, content-addressed by release: safe to cache hard.
  response.headers.set("Cache-Control", "public, max-age=86400");
  return response;
}

/**
 * The stages the hosted pipeline actually runs. The local portal's walkthrough has seven,
 * including a Meta Muse review and live character generation; this deployment runs neither,
 * so they are absent rather than shown as steps that never progress.
 */
const STEPS = [
  {
    id: "intake",
    title: "Read the video as evidence",
    explainer:
      "Local pose tracking turns your clip into joint trajectories with per-sample confidence, competing hypotheses and explicit gaps — no API calls, nothing discarded.",
    matters: "Technical depth: evidence is retained and inspectable, never asserted.",
    impact:
      "Motion capture normally means a studio booking at roughly $500–$5,000 a day. This reads an ordinary phone clip.",
    estimate_seconds: 25,
  },
  {
    id: "world",
    title: "Build an executable world",
    explainer:
      "That evidence becomes a running MuJoCo simulation: a scaled articulated body, joint limits, contacts and an authored ground plane, every assumption labelled.",
    matters: "Impact: this is the artifact reinforcement-learning teams wait weeks for.",
    impact: "Hand-authoring one task environment is engineer-weeks of loaded cost.",
    estimate_seconds: 8,
  },
  {
    id: "task",
    title: "Freeze the task and its gates",
    explainer:
      "A reinforcement-learning task is attached with success gates frozen before the run: thresholds the pipeline cannot edit to flatter its own result.",
    matters: "Technical depth: immutable scoring, decided before the evidence arrives.",
    impact:
      "Movable goalposts are why most simulation results never transfer. Freezing them first is what makes the number worth reading.",
    estimate_seconds: 6,
  },
  {
    id: "rollout",
    title: "Train and score an agent in it",
    explainer:
      "A tabular policy trains against the frozen task, then it and two baselines are scored over a hundred episodes each, with the comparison recorded.",
    matters: "Demo: working software end to end, not a mock or a scripted animation.",
    impact:
      "An environment you cannot run is a diagram. This is where the video becomes something an agent trains against.",
    estimate_seconds: 35,
  },
  {
    id: "avatar",
    title: "Build an avatar-ready scene",
    explainer:
      "The same body is rebuilt as a shared-world scene, the only kind a replacement character can be retargeted onto, with its proportions preserved rather than stretched onto the recovered body.",
    matters:
      "Appearance is decoupled from the physics it rides on. Measured on this path: bone lengths move by at most 1.7e-15 relative, with no non-finite vertex.",
    impact:
      "One captured behaviour can be re-skinned for any product or study without re-capturing or re-simulating.",
    estimate_seconds: 10,
  },
] as const;


// Mirrors the exercise pipeline's bound in index.ts. Kept local so this module needs no
// import from the router that calls it.
const MAX_UPLOAD = 40 * 1024 * 1024;

/**
 * Where a refusal belongs, from the wording the pipeline's own classifier produces.
 * Everything except a budget or deadline overrun is decided while reading the evidence, so
 * those two are the only ones belonging to a later stage. No step is failed on a guess.
 */
function failedStep(message: string) {
  const text = message.toLowerCase();
  if (text.includes("took too long") || text.includes("budget") || text.includes("deadline"))
    return "rollout";
  return "intake";
}

type JobRow = {
  id: string;
  status: string;
  created: number;
  message: string | null;
  result: string | null;
};

/** The run as the page's renderer expects it: a status, an id, and a status per step. */
function runState(job: JobRow) {
  const terminal = ["complete", "failed", "expired"].includes(job.status);
  let summary: Record<string, unknown> = {};
  try {
    if (job.result) summary = (JSON.parse(job.result).summary ?? {}) as Record<string, unknown>;
  } catch {
    summary = {};
  }

  const detail: Record<string, string> = {};
  if (job.status === "complete") {
    const pose = (summary.pose ?? {}) as Record<string, number>;
    const task = (summary.task ?? {}) as Record<string, string>;
    const rates = (summary.baselines ?? {}) as Record<string, Record<string, number>>;
    detail.intake = pose.sampled_frames
      ? `${pose.tracked_frames} of ${pose.sampled_frames} sampled frames tracked`
      : "Evidence retained and inspectable";
    detail.world = "World built and loaded";
    detail.task = task.joint
      ? `Gates frozen before the run on ${task.joint}`
      : "Gates frozen before the run";
    detail.rollout =
      rates.q_learning && rates.random
        ? `Learned ${rates.q_learning.successes}/${rates.q_learning.episodes} against random ${rates.random.successes}/${rates.random.episodes}`
        : "Policy trained and scored against baselines";
    detail.avatar = "Avatar-ready scene built; physics untouched";
  }

  const blame = job.status === "failed" ? failedStep(job.message ?? "") : "";
  const steps = STEPS.map((step) => {
    if (job.status === "complete")
      return { id: step.id, status: "done", detail: detail[step.id] ?? "" };
    if (job.status === "failed")
      return step.id === blame
        ? {
            id: step.id,
            status: "failed",
            error: job.message ?? "The pipeline declined this clip.",
          }
        : { id: step.id, status: "skipped", detail: "Not reached" };
    // While it runs the Worker cannot see inside the container, so no step is claimed
    // finished. The elapsed clock against the estimate is the honest signal.
    return { id: step.id, status: "pending" };
  });

  return {
    status: terminal ? (job.status === "complete" ? "complete" : "failed") : "running",
    run: job.id,
    steps,
    elapsed_s: Math.max(0, Math.round((Date.now() - job.created) / 100) / 10),
    message: job.message ?? "",
  };
}

/** Startable clips: the curated showcase sources, which are committed and credited. */
function clipList() {
  return SEEDS.filter(
    (seed) => seed.source_video && seed.category !== "Rolling ball",
  ).map((seed) => {
    const credit = (seed.source_credit ?? "").split("\u00b7").map((part) => part.trim());
    return {
      id: seed.id,
      title: seed.title,
      creator: credit[0] ?? "",
      licence: credit[1] ?? "",
      why: seed.caption ?? "",
    };
  });
}

export async function demo(request: Request, env: Env, url: URL): Promise<Response> {
  const path = url.pathname;

  // --- Public: curated examples and their media. The only unauthenticated reads. ---
  if (path === "/api/demo/showcase" && request.method === "GET")
    return new Response(
      JSON.stringify({ items: showcaseItems(), home_url: "/" }),
      {
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": "public, max-age=300",
        },
      },
    );

  const media = /^\/api\/demo\/showcase\/([^/]{1,64})\/(video|source)$/.exec(path);
  if (media && request.method === "GET")
    return showcaseMedia(env, url, media[1], media[2] as "video" | "source");

  if (path === "/api/demo/featured" && request.method === "GET")
    return new Response(
      JSON.stringify({ featured: SEEDS.slice(0, 3).map((seed) => seed.id) }),
      {
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Cache-Control": "public, max-age=300",
        },
      },
    );

  // --- The access gate. Issues the derived token; never echoes the code. ---
  if (path === "/api/demo/unlock" && request.method === "POST") {
    const body = (await request.json().catch(() => ({}))) as { code?: unknown };
    const offered = typeof body.code === "string" ? body.code.trim() : "";
    const secret = (env as { JUDGE_ACCESS_CODE?: string }).JUDGE_ACCESS_CODE ?? "";
    if (!offered || !secret)
      return privately({ error: "Access code not recognized." }, 403);
    const [a, b] = await Promise.all([hex(offered), hex(secret)]);
    if (a !== b) return privately({ error: "Access code not recognized." }, 403);

    // Unlock is the page's first call, so it establishes the session rather than demanding
    // one: this page never fetches /api/session, and telling a visitor to reload could not
    // have produced a cookie either.
    const existing = demoCookie(request);
    const cookie = existing || newSession();
    const response = privately({ token: await demoToken(cookie) });
    if (!existing)
      response.headers.set(
        "Set-Cookie",
        `weft_session=${cookie}; Path=/; Max-Age=86400; Secure; HttpOnly; SameSite=Strict`,
      );
    return response;
  }

  // --- Everything below is session-scoped. ---
  if (!(await authorized(request, url)))
    return privately({ error: "Enter your access code to continue." }, 403);

  if (path === "/api/demo/steps" && request.method === "GET")
    return privately({ steps: STEPS });

  if (path === "/api/demo/clips" && request.method === "GET")
    return privately({ clips: clipList() });

  // Owner is the session, so one visitor can never read another's runs.
  const cookie = demoCookie(request);
  const owner = await hex(cookie);
  const queue = env.SUBMISSIONS.getByName("weft-admission-v1");

  if (path === "/api/demo/state" && request.method === "GET") {
    const latest = (await queue.latest(owner)) as JobRow | null;
    if (!latest) return privately({ status: "idle", run: null, steps: [] });
    return privately(runState(latest));
  }

  // This visitor's own finished runs. The curated showcase is listed separately and
  // publicly; nothing here exposes another session's work or a project id.
  if (path === "/api/demo/gallery" && request.method === "GET") {
    const latest = (await queue.latest(owner)) as JobRow | null;
    const demos =
      latest && latest.status === "complete"
        ? [
            {
              id: latest.id,
              title: "Your run",
              caption: "Built in this session from your clip.",
              metrics: "",
              kind: "studio",
              source: "your run",
              video_url: `/api/demo/gallery/${latest.id}/video`,
            },
          ]
        : [];
    return privately({ demos, items: demos, home_url: "/" });
  }

  // The normalised clip the pipeline actually read back, for this session's own run only.
  const galleryVideo = /^\/api\/demo\/gallery\/([a-f0-9-]{36})\/video$/.exec(path);
  if (galleryVideo && ["GET", "HEAD"].includes(request.method)) {
    const job = (await queue.getJob(galleryVideo[1], owner)) as JobRow | null;
    if (!job || job.status !== "complete") return privately({ error: "Not found." }, 404);
    const object = await env.UPLOADS.get(`${job.id}/source.mp4`, {
      range: request.headers,
    });
    if (!object) return privately({ error: "Not found." }, 404);
    const headers = new Headers({
      "Content-Type": "video/mp4",
      "Cache-Control": "private, no-store",
      "Accept-Ranges": "bytes",
    });
    if (object.size) headers.set("Content-Length", String(object.size));
    return new Response(object.body, { headers });
  }

  // One archive per run, not one per step: this pipeline packages the whole run, and
  // offering the same bytes under five step links would misrepresent what it produced.
  const artifact = /^\/api\/demo\/artifact\/([a-z_]{1,32})(?:\/([a-f0-9-]{36}))?$/.exec(path);
  if (artifact && ["GET", "HEAD"].includes(request.method)) {
    const job = (
      artifact[2]
        ? await queue.getJob(artifact[2], owner)
        : await queue.latest(owner)
    ) as JobRow | null;
    if (!job || job.status !== "complete") return privately({ error: "Not ready." }, 404);
    const object = await env.UPLOADS.get(`${job.id}/environment.zip`, {
      range: request.headers,
    });
    if (!object) return privately({ error: "Not found." }, 404);
    const headers = new Headers({
      "Content-Type": "application/zip",
      "Cache-Control": "private, no-store",
      "Content-Disposition": 'attachment; filename="weft-environment.zip"',
    });
    if (object.size) headers.set("Content-Length", String(object.size));
    return new Response(object.body, { headers });
  }

  if (path === "/api/demo/start" && request.method === "POST") {
    // Same-origin only. The token already proves this session entered the access code, so
    // it stands in for the CSRF header the main upload route requires.
    if (request.headers.get("Origin") !== env.PUBLIC_ORIGIN)
      return privately({ error: "Reload this page before submitting." }, 403);
    if (env.SUBMISSIONS_ENABLED !== "true")
      return privately({ error: "Submissions are not open yet." }, 503);

    const clipId = url.searchParams.get("clip") ?? "";
    let source: ArrayBuffer | null = null;
    if (clipId) {
      const seed = ID.test(clipId) ? SEED_BY_ID.get(clipId) : undefined;
      if (!seed?.source_video) return privately({ error: "Unknown clip." }, 404);
      const asset = await env.ASSETS.fetch(new Request(`${url.origin}/${seed.source_video}`));
      if (!asset.ok) return privately({ error: "Unknown clip." }, 404);
      source = await asset.arrayBuffer();
    } else {
      const declared = Number(request.headers.get("Content-Length") ?? "0");
      if (declared > MAX_UPLOAD)
        return privately({ error: "Choose a video up to 40 MB." }, 413);
      source = await request.arrayBuffer();
    }
    if (!source || source.byteLength === 0)
      return privately({ error: "Choose a video, or start from one of the clips." }, 400);
    if (source.byteLength > MAX_UPLOAD)
      return privately({ error: "Choose a video up to 40 MB." }, 413);

    const ip = await hex(
      (request.headers.get("CF-Connecting-IP") ?? "unknown") +
        ":" +
        Math.floor(Date.now() / 86_400_000),
    );
    // The token proves the access code, so this runs in the reviewer lane with its own
    // daily budget and the same one-at-a-time admission as every other submission.
    const reserved = (await queue.reserve(owner, ip, crypto.randomUUID(), true, "exercise")) as
      | { job: JobRow; fresh: boolean }
      | { refused: string };
    if ("refused" in reserved) return privately({ error: reserved.refused }, 429);
    if (reserved.fresh) {
      try {
        await env.UPLOADS.put(`${reserved.job.id}/source.mp4`, source, {
          httpMetadata: { contentType: "video/mp4" },
        });
        await queue.uploaded(reserved.job.id, owner);
      } catch (error) {
        await queue.failed(reserved.job.id, owner);
        throw error;
      }
    }
    const fresh = (await queue.getJob(reserved.job.id, owner)) as JobRow | null;
    return privately(fresh ? runState(fresh) : { status: "running", run: reserved.job.id, steps: [] }, 202);
  }

  return privately({ error: "Not found." }, 404);
}
