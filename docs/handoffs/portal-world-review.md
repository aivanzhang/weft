# Muse review of the coupled world

## Verified locally — November 2, 2026 (PST)

New exercise reviews now include the actual saved world's construction assumptions,
not just source-body observations. This completes the missing review-context link,
**not** automatic reconstruction repair or physical validation. Existing reviews
remain immutable and their narrower scope stays visible.

## Implementation

- `portal_scene_context.py`: no-follow, bounded reads of the sealed scene/task,
  lab configuration and exact support priors when applicable. Existing validators
  check source identity, scene/task binding, fixed gates and authored support
  agreement. The closed 12 KiB projection excludes policies, scores, trajectories
  and evaluator definitions; the combined advisor context stays under 20 KB.
- Includes coordinate units/scale/clock assumptions, gravity, geometry, materials,
  body dimensions/motor settings, root assistance and authored load windows.
  No missing equipment is invented; source-qualified is not physical validation.
- `scene_advisor.py`: optional validated `simulation_world`, separately qualified
  `world:human` / `world:<object>` references and context hash in the API receipt.
  Observation IDs remain unchanged. Same model, request/token/response limits,
  canonical ledger and no-retry policy. No generated code or applied proposals.
- `portal_review.py`: new review binding pins both construction observations and
  current world; rechecks after transport. An input mutation rejects completion
  while preserving settled/unknown provider charges and saved response.
- `portal_checks.py`: verifies the exact reviewed world context before local
  comparisons; old bindings still use the old construction-only path. No physics,
  task, scorer, prior review or recording is rewritten.
- Portal distinguishes legacy body-only reviews from new world-context reviews.
  Objects-only reviews keep their existing observation scope; coupled exercise
  context is not silently claimed for those projects.

## Verification

70 distinct Python checks pass: `.local/portal-world-review-unit-01.xml` (47),
`.local/portal-scene-context-01.xml` (23). Six DOM UI groups pass with
`node tests/portal_ui_smoke.mjs`. Mocked transport tests cover world-object
references, private result exclusion, post-call tamper rejection with retained
charge and rejection of context changes before checks.

Read-only actual-input check `.local/portal-scene-context-real-01.json`: original
exercise projection 2,244 bytes  / 48.4 ms; support child 2,935 bytes  / 52.2 ms. These
are measured development samples, not sustained responsiveness guarantees.

Real authenticated HTTP acceptance `.local/portal-world-muse-01/` reviewed child
`639d135203db457ead1bb720cead115c`, using its existing source video and world.
No reinference, training or new physics steps. Complete runtime/dependency
snapshot, prepared context, binding, plan, harness and result are retained.

- One real `muse-spark-1.3` Standard request, zero retries; cost **$0.01732575**.
- HTTP journey 11.1332 s; worker 9.4826 s / 138,231,808 bytes peak RSS.
  27 HTTP requests mean 49.4 ms/max 1.2723 s including native sandbox loading.
- Three hypotheses. Muse explicitly identifies `world:support_box` as authored
  and distant, and its forearm-load requirement as a task assumption rather than
  a measurement. Other motion/scene descriptions remain unverified model claims.
- Requested contact/calibration/EMG measurements were **not executed**. No hypothesis
  was accepted and no repair applied. Do not use this critique as physical truth.
- All 1,325 run members verify unchanged; manifest
  `2a0652b8b1eedec6952f22fe74c252c62662fc47f7566d6662d5b3751ceb0085`.
  Original ready-project status is unchanged and the sandbox remains usable.
- World context hash
  `1318927bcfd260da7a45b4e90330ae2db98bae9892017af615ec146d51a176a9`;
  sent context exactly matches prepared context. New binding also passes local
  check admission without executing comparisons.
- `legacy-binding-audit.json` verifies the original project's saved review still
  binds unchanged. No duplicate paid review or old-run rewrite.
- Canonical Meta ledger: 11 reservations, $0.266662 committed/$4.733338 remaining,
  zero unsettled. Current pricing rechecked at the official
  [pricing page](https://dev.meta.ai/docs/pricing-rate-limits): input $1.25/M,
  output $4.25/M; reservation $0.058368. No new recurring service or deployment.

## Approval-review incident

Initial automatic approval review rejected transport, treating source images as
private. No process/request/charge occurred. `payload-provenance.json` documents
that this is RickyBennison's public CC BY-SA
[Basic single leg squat](https://commons.wikimedia.org/wiki/File:Basic_single_leg_squat.webm),
SHA256 `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
All three thumbnail hashes exactly match the prior authorized Meta review.
New fields are authored box/load settings, not private patient measurements.
The same action was re-reviewed with that evidence and approved; no alternate
route, payload hiding or approval bypass was used. Preserve this incident.

## Handoff

All acceptance servers/workers are terminal and reaped. Root owns only advisor,
portal context/integration/UI/tests and these docs. Parallel avatar files remain
separately owned; do not stage or reset them. Publication is recorded in
`.local/portal-world-muse-01/publication.json`; verify Git/remote on resume.
Hosted site stays offline. The overall goal remains active.

- [ ] Connect typed Muse proposals to bounded executable candidates and independent
  acceptance; current natural-language measurement requests require unavailable
  physical evidence, not reinterpretation as passed simulator checks.
- [ ] Finish source-qualified ground/equipment and unfamiliar-video reconstruction.
- [ ] Implement the coordinated [native-clock integration](native-clock-integration.md),
  preserving raw pose/detector history and old recordings. This is the next concrete
  reconstruction change; a read-only review identified exact nominal substitutions.
