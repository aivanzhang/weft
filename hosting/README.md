# Hosted video submissions

Cloudflare account: **Kintsugi Software**, `73c2bbd14f7b9025d9fd9eb56ad54609`.
Worker: `weft`. Private R2 bucket: `weft-private-submissions`.
Public URL: **https://weft.kintsugi.software** (custom domain; live for hackathon judging).

## Restored November 4, 2026

Restored with the owner's authorization for judge access, on the custom domain
`weft.kintsugi.software` (release `video-environments-20261104-judges`). The
`workers.dev` address and preview URLs remain disabled (`workers_dev: false`,
`preview_urls: false`); the custom domain is the only public entry. Verified
unauthenticated end to end at restore time: landing page and gallery assets,
`/health`, a fresh-session MP4 submission through the container pipeline to a
complete result with a downloadable archive, cross-session result privacy
(404), and the dark `workers.dev` address. `OPENAI_API_KEY`, `META_API_KEY`
and `MESHY_API_KEY` are stored as encrypted Worker secrets; current Worker
code does not read them, so no hosted path can spend them until a reviewed
feature does.

## Boundaries

The static site stays buildless and opens from `website/index.html`. `build-assets.mjs` copies an explicit public allowlist; repository files, uploaded videos, and saved runs are never included implicitly. The browser calls the same-origin `/api/` service only when the submission panel opens.

The Worker owns a random, HttpOnly, Secure, SameSite session cookie, CSRF checks, input limits, and private result access. A SQLite Durable Object coordinates this service's single generation slot and admission budget. Original MP4s and complete successful/failed run archives live in private R2, not ephemeral container storage. The container receives no credentials and has outbound networking disabled. It runs only the fixed Python pipeline, never uploaded scripts or generated code.

## Two pipelines, one image

A request header, `X-Weft-Pipeline`, selects which fixed pipeline runs. It defaults to `ball`, so a client that does not send it behaves exactly as before. Both live in one container image to avoid a second build and a second idle instance.

`ball` takes one isolated yellow ball moving horizontally under a fixed camera. This is not arbitrary-video or multi-object reconstruction. Push/brake controls and the parking task are authored additions.

`exercise` takes one person, head to feet, against a plain background with a fixed camera in a single continuous shot. Camera cuts, wall mirrors and bystanders each read as a competing person and are refused; a long occlusion breaks the root reference. It returns the artifacts a browser needs to show the result and to retarget a substituted avatar client-side: pose evidence and quality, the articulated body with its authored ground, the executable task, the trained policy with its rollout, the pose-derived reference rollout, and both evaluations. `rl-evaluation.json` is the learned policy against random and no-control baselines; `human-evaluation.json` is the authored reference servo's tracking error. They are different measurements and must not be presented as one.

Admission is strict and sensitive near its own threshold: the same footage can be admitted at one encoding and refused at another. Verify a clip against this image, not a developer machine.

Failed inputs never fall back to the sample environment. Gallery initial/repaired models are stages of one source video, not different environments.

## Limits and recovery

- Ball: MP4, 10 MiB, 15 seconds, 450 frames, at most 1920 × 1080.
- Exercise: MP4, 40 MiB, at most 480 decoded frames after normalization, long side up to 1920, 1-240 fps. `/api/session` reports both caps so a client validates before a 413.
- One active generator, at most three admitted uploads/jobs. Twenty attempts globally and three per visitor in a rolling day; the IP guard also limits three attempts in its daily bucket. Failed uploads count.
- Ball child: 150 CPU seconds, 1.5 GiB address space, 128 MiB per-file writes, 180 seconds wall time; container request ceiling 210 seconds.
- Exercise child: 420 CPU seconds, 3 GiB address space, 256 MiB per-file writes, 390 seconds wall time (the lab worker's own 360-second limit reports first); container request ceiling 420 seconds. Result archives max 32 MiB for both.
- One `standard-3` container: 2 vCPU, 8 GiB RAM, 16 GB disk. Stop after each job; five-minute idle fallback. No cron or locally running service. CPU bills by active use, so the faster instance costs about the same per job as `standard-1` and finishes sooner.
- The result summary lives in its own R2 object, not in custom metadata, which caps at 2 KiB; an exercise summary is about 2 KiB on its own.
- Upload idempotency keys and active-owner lookup prevent duplicate jobs. Browser polling is nonoverlapping and stops in hidden tabs. Reload resumes the owned result.
- Interrupted running jobs recover a persisted archive or fail explicitly; they are never automatically retrained. Queued jobs persist across restarts. A worker update must not race a running generation; check the queue/production job before deploying again.
- Access expires 24 hours after admission. One-shot Durable Object expiry alarms delete new submission files; R2's two-day lifecycle is a backstop for outages. Expired metadata is pruned after seven days on subsequent queue activity. Local research artifacts are unrelated and never pruned.
- Results stay private to the submitting browser; there is no public gallery auto-publication, email delivery, or third-party inference.

## Development and release

Use `npm ci --ignore-scripts` in this directory. Dependencies and the Linux image base are pinned. `requirements.lock` is generated from root `pyproject.toml` with the existing simulation lock as constraints:

```sh
uv pip compile pyproject.toml --extra host --constraint requirements.lock --python-version 3.14 --python-platform linux --no-annotate --no-header -o hosting/requirements.lock
cd hosting
npm run types
npm run check
npm run build
npm run deploy
```

The user manages local servers; none is needed to review the static page. `npm run types` disables runtime generation and uses installed official Workers types, avoiding a local listener. There are no new automated tests.

Deployment bundles the static page and Linux image. Create the dedicated private R2 bucket and its lifecycle once; never make it public. The supplied configuration pins the Kintsugi account and does not touch Pinnacle or any existing application. Deploy with the Weft-specific release lock; do not overlap releases. No account upgrade, new credential, paid model, or YC redemption is part of deployment.

After deployment, immediately submit the CC0 sample through the normal session, consent, CSRF, and quota path. Verify terminal status, source-video range response, private-access refusal, download integrity, and packaged `reset`/`step` example. Never bypass admission limits for verification. Cloudflare can take several minutes to provision the first container; that is infrastructure readiness, not a reason to start duplicate jobs.

## Cost envelope — October 28, 2026 (PDT)

One-time local preparation uses existing hardware; no external inference. For 600 submissions/month (20/day), assuming every job reaches its 180-second limit plus 30 seconds startup, provisioned `standard-1` memory costs about $1.26, active CPU at most $1.08, and disk $0.071 before shared allowances. Stopping after jobs avoids idle billing. Even a five-minute fallback after every job adds about $0.76 in provisioned memory/disk. Up to 19.2 GB/month of container result egress adds at most $0.96 at the highest listed regional rate before included bandwidth. Actual representative local Linux job: 37.9 seconds under a 0.5-CPU limit; this is not a production latency guarantee.

R2 retains at most about 0.84 GiB of input/archive payload per normal 24-hour window, or roughly twice that during delayed expiry. About 1,200 Class A writes/month and a few thousand Class B reads are expected, excluding abusive request traffic. R2 bills rounded million-operation units: if shared allowances are exhausted, allow $4.50 for Class A, $0.36 for Class B, and rounded GB-month storage. Result polling reads the small admission ledger, not R2. Static assets total about 1.3 MB and are not container-served. Allow an additional small amount for Workers/Durable Object requests, SQLite operations, and job-duration billing. Logs are disabled; errors remain in private run evidence.

Conservative expected incremental envelope at these limits: approximately **$10/month or less**, plus the **$5/month Workers paid plan if not already active**. This is a projection, not a billing guarantee or a newly approved dollar budget. Existing billing-plan details were not readable through the current OAuth grant (403); do not upgrade a plan automatically. No additional inference charges. Provider-wide free tiers/credits are shared and are not assumed available.

YC's current Cloudflare offer lists $350,000 in startup credits, with separate product caps and eligibility/application requirements; no claim or redemption was made. Credits are not deducted from this estimate. See [YC offer](https://bookface.ycombinator.com/deals/1512), [container pricing](https://developers.cloudflare.com/containers/platform/pricing/), [R2 pricing](https://developers.cloudflare.com/r2/pricing/), and [Durable Objects pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/).

## Verification evidence

The real fixture was rebuilt locally with 15,000 training episodes and 300 evaluation episodes; its full artifact manifest verified. A second, resource-limited Linux generation completed in the deployment image without network access. Decode/platform differences produce different fitted parameters; the public gallery retains the provenance of its actual export, not substituted numbers. Both records are preserved under local `artifacts/` after installation.

JavaScript syntax, Worker type checking, route imports, image build, dependency audit, and allowlisted packaging are checked. Browser layout/playback and real-device behavior are not inspected without the user's browser authorization. Production verification status is recorded separately after deployment.


### Production verification

Verified **October 28, 2026, 11:50:30 a.m. PDT** on Worker version `5b58b959-5601-4811-b27f-7526e9dde652` and container image digest `sha256:df99379a0fba430f0fce7822506a6dd89953f39288901cd7365edfa547c96cbb`.

The guarded visitor flow completed in 42.9 seconds (34.9 seconds generation), produced a 2,356,227-byte environment archive, and verified all 52 full-run artifact hashes. Private access without the session was refused, and authenticated video byte ranges returned correctly. No quota bypass, paid model call, or public upload exposure was used. Two admitted diagnostic jobs failed before the container-stream handoff was corrected; the third completed within the normal three-per-visitor limit. Earlier admission failures occurred before creating any job.

The deployed handoff uses `FixedLengthStream`: the Container SDK wraps the response in an unknown-length stream, which R2 cannot store directly. Both the copy and storage promise are awaited before marking a job complete or stopping the container. Keep this invariant during changes.

Non-secret receipt: `production-verification.json`. The downloaded evidence archive and original local generation runs are preserved under ignored `artifacts/`, not public website assets. Cloudflare's default filtering rejects Python's generic user agent; the diagnostic used its own explicit Weft client identifier without changing access rules.
