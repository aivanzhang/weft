# API-backed product portal

**Follow-up:** [Muse physics review is now integrated](portal-muse.md), using the separate existing Meta key/ledger. The acceptance history below describes the earlier OpenAI portal milestone.

## Direction and ownership

November 2, 2026 (PST): prioritize the user-facing web product over additional
standalone reconstruction studies. Preserve the full generalized-video goal and
both tracks. The user explicitly requires project API keys, not ChatGPT login or
coding-session authentication, and approved **$5 total OpenAI portal development
spend**. Meta's separate $5 ledger and avatar work are not modified.

Root owns `portal.py`, `portal_pipeline.py`, `portal_worker.py`, `openai_access.py`,
portal assets, CLI/mounted-lab integration and their tests. Preserve concurrent
changes under `tools/avatar-assets/`, `avatar_replay.js` and its test. No hosted
deployment, browser automation, new dependencies or persistent server was started.

## Implementation

- `weft portal`: loopback bearer-authenticated upload and project APIs, bounded
  single-worker queue, one owner lease, saved-project reopening, cancellation,
  private artifacts and mounted versions of the existing lab.
- `openai_access.py`: fixed HTTPS Responses endpoint, owner-only project key,
  strict output model/version, durable SQLite reservations, no automatic retry,
  known-usage settlement and exact-source/request reuse of settled responses.
  No ChatGPT/OAuth credential discovery or agent CLI is used.
- `portal_pipeline.py`: retain original media and six sampled-frame identities;
  structured scene planning; unchanged reconstruction/task gates; existing
  `prepare_exercise_scene` person-only full-reference body/world/learning pipeline;
  object localization, planar tracking and spatial construction/training/lab.
- Exercise defaults are the established **lite / video / raw** pose settings,
  explicit root assistance, two generations / four candidates, three evaluation
  starts. A ready lab means executable, not a learned improvement or physical truth.
- Color-region localization is preferred only when distinct source-pixel regions
  match the AI's object count and shape types. AI boxes, labels and roles are not
  silently treated as pixel measurements. Otherwise appearance hypotheses retain
  two source pixels of edge context and must pass existing tracking gates.
- Bounds: 100 MB input, 60 seconds / 1,800 decoded frames; 420-second worker wall
  time, 360 CPU seconds, 250ms sampled 1.5 GiB peak RSS, 650 MB / 6,000 members per
  job; 50 projects / 2 GB storage. Preserve outputs instead of automatic deletion.
  Hosted multi-tenant security and remote job ownership are not implemented here.

## Evidence and preserved failures

All acceptance servers were temporary loopback HTTP tests, not user dev servers.
Native runs use pinned runtime copies; see each control directory's plan and
runtime manifest rather than the concurrently edited working tree.

- `.local/portal-acceptance-01`: harness rejected its invalid 33-character request
  ID before creating a job or calling the API. Corrected in attempt02.
- `.local/portal-acceptance-02`: actual project-key OpenAI request completed in
  2.462 seconds for **$0.001736 upper estimated cost**. The initial full/image/raw
  pose choice rejected competing hypotheses. Job
  `7653b6f294f241e78fbe0968254a3b00` remains failed; no thresholds were changed.
- The successful response was registered for reuse only after reconstructing and
  verifying its exact request from all pinned JPEGs, source hash, ledger identity,
  response ID and usage. Receipt: `analysis-reuse-migration.json` in attempt02.
- `.local/portal-acceptance-03`: actual HTTP upload → reused OpenAI analysis →
  established shared exercise pipeline → training/evaluation → mounted sandbox →
  recording/export completed in **29.5865 seconds**. Worker **26.5790 seconds,
  560,218,112 B peak RSS**. The initial in-sandbox bind denial occurred before job
  creation; the narrowly authorized loopback test then ran successfully.
- Ready job: `.local/portal/4e0f1fe859664bb5ba788214e129451d`;
  lab: `build/learning_workflow/lab`; recording: `session-b3a8f083138e`.
  **Nine actions / ten events replay exactly**. All **1,325 run members** verified,
  and recording did not change run manifest
  `d8cfe07cc44e3d36efafcd6039c49435188e684c763e28fb9ce11e5cc8d1fe26`.
  Original source SHA-256:
  `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
- Exercise learned and authored controllers each pass **3/3 simulator starts**;
  coast/random pass **0/3**. Selection retained the original feasible controller;
  no learned-policy improvement, unassisted balance or physical accuracy claim.
- `.local/portal-objects-01`: clearly synthetic three-shape test, second actual
  OpenAI call **$0.001516**; inaccurate AI regions failed texture/contrast checks.
- `.local/portal-objects-02`: source-pixel localization fixed that issue. The
  source/archive adapter then rejected a **347-member** ordinary run because its
  old limit was256. Preserve that failed package and original error; downstream
  training must not mask construction failure with a missing-scene error.
- Updated bounded archive capacities: input256→512, output320→576. Source/output
  byte limits remain64 MiB. **96 construction/boundary checks pass**; current-source
  snapshots remain complete, rather than omitting implementation files.
- `.local/portal-objects-03`: final object-route HTTP acceptance completed in
  **2.8923 seconds**, worker2.0672s /148,783,104 B peak RSS. Ready job
  `90ee5fd0201843feab3196b062c936ed`, lab `lab`, recording
  `session-4d9024478992`: **nine actions /ten events replay exactly**, all719 run
  members verified. Run manifest
  `2d9c1cf5c7ffd61d88eb64b19c6f013e26856257c58eebd183269c68d5579b18`.
  Reuses paid analysis, no new call. This is a synthetic software control.
- Final saved-only audit: `.local/portal-saved-audit-01.json`, **2,044 run members**,
  both complete recording exports, both runtime snapshots (326 files each), original
  media and fresh ordinary recording receipts verified. No new physics or inference.
  **151 distinct Python cases pass** across the listed receipts.
- Last verified portal ledger: **two calls, $0.003252 committed, $4.996748 remaining,
  zero unsettled**. Recheck before further paid work; no budget history was reset.

## Verification and current limits

Receipts: `.local/portal-tests-01.xml`, `portal-integration-tests-02.xml`,
`portal-pipeline-tests-01.xml`, `portal-pipeline-tests-02.xml`,
`portal-localization-tests-01.xml`, `portal-spatial-capacity-tests-01.xml`.
The earlier pipeline test failure was a test's incorrect return-shape assumption;
the corrected test passes. Portal UI lifecycle, mounted URL isolation and existing
lab/appearance UI suites pass without browser automation. These checks are not
live-browser visual or production multi-user qualification.

API model/pricing and strict image/schema contract were checked against official
[model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-mini),
[vision guidance](https://developers.openai.com/api/docs/guides/images-vision) and
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
Raw official pages are preserved under `.local/openai-*-20261102.html`.

## Next safe work

1. Reopen with `.venv-lab/bin/python -m weft portal`; the user manages the server.
   Keep ready run manifests and all rejected attempts unchanged.
2. Prioritize useful portal progression: expose missing-input assistance and
   source-qualified equipment workflows through the same jobs and sandbox, then
   independent unfamiliar-video acceptance. Do not hide unsupported scenes.
3. Before hosted access: implement user authentication, per-user job/artifact
   isolation and quotas, external durable job/storage ownership, and secure worker
   credentials. The existing hosted website stays offline until explicitly restored.
4. Continue the inherited motion/identity/contact/native-clock work where it
   directly improves this product. Do not revert to unrelated study wrappers or
   narrow the generalizable goal to these two demonstration inputs.

Publication and the final saved-run audit are recorded in
`.local/portal-publication-01.json` and `.local/portal-saved-audit-01.json` once
complete. Do not infer publication from this note alone.
