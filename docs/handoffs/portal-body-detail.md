# Detailed body attempts and collision-first contact search

Recorded November 2, 2026 (PST). The complete Weft goal remains active and incomplete. This milestone follows published `d1a13faa5ddb84f3f0d960f77d89931ba455ae64`, which added source-linked spatial portal interaction and the task-support advisory. Preserve unrelated avatar edits and every previous failed result.

## Parallel ownership and implementation

- A (`native_clock_contract`) owns the closed `full-dense-region-v1` recipe and `exercise_perception_pipeline.py` options. Exact baseline evidence/raw poses, parent/source/model identities, all-hypothesis region and sampling settings are retained. Every frame uses the fixed local Full model; original identity, body, floor and clock checks remain unchanged. New body seeds must bind the exact newly generated body manifest. Legacy producer compatibility is independently rederived, not accepted merely because its hash is listed.
- C (`native_clock_observations`) owns `portal_body_detail.py`: small immutable child metadata under the existing portal lease, one attempt per exact parent/recipe, deduplication before quota checks, and heavy source/model/raw preparation only in the worker. The child is a normal video project and uses the existing supervised queue. Original failure, paid consent and equipment outputs cannot be rebound or overwritten. Partial work is retained and never silently retried.
- B (`native_clock_runtime`) owns `moving_equipment_contact_restore.py` and the optional `separate-collision-support-v1` search strategy. It separates collision restoration from support restoration, retaining every attained collision constraint and all original hard bounds. A fresh native nonlinear audit remains authoritative; linear feasibility is not task success. The prior strategy stays the default. No new controller or learned policy is installed.
- Root owns authenticated portal routes, worker forwarding, UI, artifact downloads, integration verification, this handoff and publication. No other lane edits those shared files.

## Product behavior

`POST /api/projects/{parent}/perception/body-detail` accepts only a recipe ID and 32-hex request ID under existing authentication. It queues a separate child; pose/model parsing and video copying do not occur on the request thread. The UI never chooses one competing person, automatically retries, or sends media to providers. On local success, the existing OpenAI-image and Meta-video consent flow remains separate. Child lineage and the exact request/region recipe are inspectable. [Usage guide](../portal-body-detail.md).

The original upload must have a sealed failed **body** preflight, bounded raw estimates and the existing MP4 envelope. Completed bodies with failures in later stages are not relabeled eligible. Parent and child share source identity, not mutable state. Cancellation/restart preserves partial evidence; the start marker prevents another inference pass. The existing per-job and 2 GB portal bounds remain. The previously demonstrated portal root has insufficient room to reserve another full-size child; new acceptance uses isolated bounded storage, not deletion or a raised cap.

## Verification

`.local/portal-body-detail-checkpoint-01/tests.json` aggregates **221 current Python checks and 21 UI groups**, all passing. Counts are deduplicated by testcase identity: 117 recipe/legacy, 23 child custody, 47 contact restoration/search and 34 root integration/transport/legacy-routing cases. A superseded initial-anchor rejection test remains in history, giving 222 historical identities, not 222 current checks. Repeated receipts add no new checks. Independent reviews cleared all lanes and root integration. These are scoped tests, not a full-repository or live-browser claim.

The complete worker orchestration tests explicitly stub inference. They prove request/queue/preparation/option forwarding, failure preservation and the transition to separate paid consent—not a newly reconstructed real body. The saved-source staging acceptance must stop before repeating the already-proven 458-frame Full-model inference. Its role is to verify actual source/model/region custody and HTTP behavior, not manufacture unfamiliar-video coverage.

Release records: `.local/portal-body-detail-recipe-01/`, `.local/portal-body-detail-child-01/`, `.local/moving-contact-restoration-01/`. Source acceptance locations: `.local/portal-body-detail-source-01/` (staging only) and `.local/moving-contact-restoration-source-01/` (one changed contact method). Publication evidence belongs in `.local/portal-body-detail-checkpoint-01/publication.json` after remote verification.

### Actual saved-source staging

`.local/portal-body-detail-source-01/result.json` records **passed_staging_only**: **0.469 seconds**, **190,300,160 bytes peak RSS**. The real authenticated create/dedup requests took 5.15/3.02 ms, with request/recipe downloads at 1.78/0.62 ms. This small bounded run is not a sustained responsiveness or browser-render benchmark.

Original failed project `a85537e09e284facb10de22cf2a7dcbb` remains unchanged in the existing portal. Its exact source SHA is `c289fc1afe0b60a418cc6ed9e4554c8414a4d970d2dfcaea1f6f82839204dac3`. An isolated thin input clone retained all required metadata and original raw input bytes; it does not pretend to copy the entire old package. The recipe preserves **153 sampled rows /155 hypotheses**, deriving region `[262,121,449,434]` and verifying the actual installed Full-model bytes. Recipe SHA `d3b39a13c436b0315ef4a7338f52a36e6780a771cb24f0bd47ba36f49434723a`.

The isolated child `afd2a55afabd4bd7b4a17582aa919ea5` is explicitly **interrupted before inference**, not a ready body or a demonstration to resume. The queue never started; the temporary HTTP server closed. Original parent, clone, runtime and canonical budget ledgers stayed unchanged. All 38 retained members total 46,570,937 bytes before the seal; zero API, inference, model-compilation or physics calls. The initial harness bootstrap refusal is retained: MuJoCo import requested the exact read-only macOS translation lookup, which the corrected audit allowed once; no generic subprocess permission was added.

### Preserved contact integration failure

The first changed-strategy actual-source run is retained at `.local/moving-contact-restoration-source-01/result.json`: **9.735 seconds**, **861,536,256 bytes peak RSS**, original source/runtime/budgets unchanged. It exposed a `zero direction must satisfy original hard bounds` integration error, not a completed no-witness search. Initial source-only interpolation checks found no original arm-bound violation; the failed run did not save enough intermediate residuals to identify the exact later row. The helper/caller used inconsistent feasibility assumptions. The correction must retain row/time/residual diagnostics and establish a valid hard-bound starting point without clamping or widening original gates. A new run requires that corrected implementation and a new output directory.

The corrected implementation uses a bounded Phase-I feasibility search, then independently checks the original hard rows before collision/support restoration. Unsupported anchors return no direction and retain exact row/time/residual diagnostics. Temporary feasibility slack never relaxes final acceptance. Both independent reviews cleared the fix; the initial failure and its hash-verified runtime bytes remain retained.

### Corrected actual contact run

`.local/moving-contact-restoration-source-02/result.json` records successful **execution and custody**, not successful contact: **11.665 seconds**, **862,961,664 bytes peak RSS**, with **2.339 seconds** in the new search. One model performs **8,954 native forward evaluations and zero physics steps**. Both support alternatives complete but still yield **no geometric/static witness**. Original source signatures, manifest, runtime and canonical provider ledgers remain unchanged. Root handle **9253** exited successfully; no model/server/job remains running. The first failed handle **3932** is also terminal.

Do not repeat this unchanged search or call it task success. The saved-only residual comparison belongs in `.local/moving-contact-restoration-source-02/REVIEW.md`; a subsequent method must address the actual remaining constraints. The default search, world, target, force limits, scorer and portal controller selection remain unchanged.

Saved comparison shows a concrete geometric improvement: the right-forearm overlaps fell from **150/242 to 0/242 native samples** for both alternatives, with all six nonlinear updates accepted per alternative. This is zero-step geometry evidence, not a successful contact rollout. Upward support normals still fail **242/242** samples, and support gaps fail 242/242 on the left and 241/242 on the right. Corrections approach the restricted 0.5-radian search cap (0.486/0.494 rad); the previous inactive-arm stall is no longer the observed failure. Invalid-contact static-load ratios do not establish actuator impossibility. Preserve these distinctions before choosing another representation or a separate user-authored task.

## Remaining work

Qualify this acquisition recipe on a distinct eligible video under a frozen local resource plan; do not rerun the unchanged 458-frame inference or relabel inspected development data as fresh evaluation. Contact geometry alone cannot establish approach continuity, sustained support, full-task learning or physical accuracy. Reliable unfamiliar-video identity/camera/geometry/contact, useful independent RL success and end-to-end Muse repair remain parts of the complete goal. The portal is testable now, but those generalization claims remain unresolved.
