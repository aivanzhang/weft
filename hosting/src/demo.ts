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
    const cookie = demoCookie(request);
    if (!cookie)
      return privately({ error: "Reload this page before entering a code." }, 403);
    const body = (await request.json().catch(() => ({}))) as { code?: unknown };
    const offered = typeof body.code === "string" ? body.code.trim() : "";
    const secret = (env as { JUDGE_ACCESS_CODE?: string }).JUDGE_ACCESS_CODE ?? "";
    if (!offered || !secret)
      return privately({ error: "Access code not recognized." }, 403);
    const [a, b] = await Promise.all([hex(offered), hex(secret)]);
    if (a !== b) return privately({ error: "Access code not recognized." }, 403);
    return privately({ token: await demoToken(cookie) });
  }

  // --- Everything below is session-scoped. ---
  if (!(await authorized(request, url)))
    return privately({ error: "Enter your access code to continue." }, 403);

  if (path === "/api/demo/steps" && request.method === "GET")
    return privately({ steps: STEPS });

  return privately({ error: "Not found." }, 404);
}
