# Dense source poses and constrained contact targets

**Continuation:** [Compact body and portal/API handoff](compact-body-portal.md)
records completed compaction, actual queued preflight, real provider calls and
the remaining saved-bridge publication/recovery work. The active-stage notes
below describe the earlier checkpoint, not a reason to repeat those jobs.

## Active checkpoint — November 2, 2026, 5:53 PM PST

The complete video-to-interactive-world goal remains active. Three independent
lanes handle dense source reconstruction, original-task contact control, and
portal diagnostics/recovery review. Root integrates, audits and publishes.
Separate avatar work is not owned by this checkpoint. No paid API call,
deployment, browser session or persistent development server was started.

## Unfamiliar-video reconstruction

Source: `data/exercise/kettlebell-flip-juggling.mp4`, SHA256
`c289fc1afe0b60a418cc6ed9e4554c8414a4d970d2dfcaea1f6f82839204dac3`.
All 458 original frames remain. This is exposed development evidence, not a
fresh final evaluation. The original full-frame Lite/Full failures are retained.

`person_region_proposal.py` derives one fixed region from every confident
landmark in every retained hypothesis across both original histories. It never
chooses a convenient competing pose. The result `[260, 121, 449, 434]` exactly
matches the frozen source-study recipe. Outside-region people remain unknown;
this is a detector hint, not recovered segmentation or verified identity.
Twenty focused checks pass. Independent saved-only comparison takes 0.181 s /
55 MB; see `.local/region-proposal-review-01/REVIEW.md` and `comparison.json`.

The regional Full-model study `.local/flip-fixed-region-01` observes 147/153
usable poses with no competing detections, but the unchanged 0.25-second motion
gap limit rejects its 0.30-second gap. No body, seed or world is promoted.

The explicit `pose_sampling="all_frames"` option observes every original frame,
with a 900-frame metadata/decoded cap. Default sampling is unchanged; the CLI
option is `exercise-body --pose-sampling all_frames`. Capture stops before
inference on decoded frame 901 and preserves its honest decoder row.

One dense study, `.local/flip-dense-region-01`, finishes all 458 inferences:
442 usable poses, zero competing detections, complete native timing, and no
remaining long root-motion gap. It then hits the fixed 1 GiB RSS watchdog during
body finalization: 21.893 seconds, 1,085,177,856-byte sampled peak. The partial
79.40 MB output is not a completed body package. Do not repeat inference or
raise the budget to hide this failure.

Recovery is now complete: duplicate durable buffers are released before body
construction, and a fresh bounded process recovers the exact saved capture into
a separate pose-only package. Independent review and 16 recovery tests pass.
Historical captured code is never executed. Original raw/native bytes and the
failed study remain unchanged; ordinary construction gates are preserved.

`.local/flip-dense-recovery-01/body_source` is a sealed construction-only mobile
body, with all 458 source slots and 442 references. Body manifest SHA256:
`8bb1fb6977d2a5c6cf97c9fe9543d4e8c93c2ee2c36d852aadf8db50110beeb7`.
Recovery/body/attempted preflight takes 12.755 s with a 1,055,834,112-byte sampled
peak, below the original 1 GiB bound. No repeated inference or native capture.
The attempted preflight performs one ordinary video metadata inspection; its
worker's zero `source_decodes` counter was success-only and must not be read as
zero inspections. Preserve the original receipt and separate correction.

The first seed attempt exposes missing recovery ancestry in downstream adapters.
Both adapters now retain all proof files and the reviewed validator; six focused
custody/tamper tests pass. One fresh saved-body preflight then gets past that
failure, but rejects the **500 KB scene metadata limit** in 5.709 s / 749 MB.
See `.local/flip-preflight-recovery-01`. All 49 body members remain unchanged;
no inference, body reconstruction or API call is repeated. Its normal video
metadata inspection count is one.

**Active dependency:** compact, externally hash-pinned provenance. The dense
human document is 1.62 MB compact JSON, mostly reconstruction evidence, while
the full 442-reference trajectory is about 319 KB. Preserve every reference and
source slot; do not blindly widen the generic scene limit. The source agent owns
the design, the runtime agent independently reviews, and root owns integration.
No qualified preflight, new playable world or physical ground/contact result is
claimed. The original failed portal project is not overwritten.

## Portal failure explanation

The background local-preflight worker now adds a bounded explanation beside a
failed run: competing-pose timestamps, pose counts, raw alternatives and window
screens. It never joins identities, retries inference or offers a screened
window as a playable full task. The request path reads only a small sealed
summary, not the raw pose history. Optional diagnostic failure cannot replace
the original failure.

Actual saved-project attachment: `.local/portal-pose-diagnostic-02`, job
`a85537e09e284facb10de22cf2a7dcbb` beneath `.local/portal-kettlebell-flip-01/portal`.
All 47 prior files remain unchanged; exactly three diagnostic files were added.
The original 143/153 usable samples and two identity breaks remain a failure.
Attachment takes 0.452 s / 159 MB; compact summary read takes 0.000579 s on this
fixture, not a general UI performance guarantee. No inference/decode/API call.
Report SHA256 `1a5d44663e80b3f6ffc70064ed644260a370faa3d796422317c90888507f0eaf`.

Check receipts: `.local/portal-pose-diagnostic-01-tests.xml`,
`-02-tests.xml`, `-03-tests.xml` (37 distinct Python cases); 17 portal DOM groups.
The last focused case requires the decoder timestamp field while permitting
explicit null. No browser visual QA is claimed. Detailed ownership, source pins,
preserved import-guard failure and release hashes are in
`.local/portal-pose-diagnostic-02/HANDOFF.md`.

## Original-task contact comparison — rejected

`source_contact_trajectory.py` proposes bounded right-arm/root target corrections
without changing source observations, the original scored reference, camera,
equipment, friction or force limits. A separate fresh geometry audit checks
source-pixel nonregression, floor/negative-gap bounds and parameter/rate limits.
This is authored control, not a learned policy or recovered physical contact.

The source study `.local/source-contact-trajectory-01` completes in 1.257 s /
211 MB with 948 geometric evaluations and no physics steps. One correction
passes all six constraint groups. Squared penetration improves only 0.1731%;
the worst 0.099668-unit overlap is unchanged. No cleared-contact claim follows.

One full original-task candidate rollout is saved in
`.local/source-contact-trajectory-trial-01`: 254 controls, 4,064 substeps,
3.887 s / 203 MB. It uses the exact saved 127-reference body's original evaluator,
seed 19, reset ordinal 10, and no jitter. The first 40 control boundaries match
the saved baseline exactly; the first action difference is control 41.
The baseline is reused, not rerun. Original source/reference/task/model arrays
are asserted unchanged at every control boundary.

Both baseline and candidate fail `equipment_stability:sam-2`. Working-object
spin improves 5.402 → 5.080 rad/s, but displacement worsens 0.1604 → 0.1695,
orientation 0.3078 → 0.4219 rad, and speed 1.050 → 1.089 units/s. Small body/floor
improvements do not override the failed original task. Candidate is rejected and
unpromoted; no sweep or training is authorized by this result.

Original evaluator SHA256:
`c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.
Trial plan SHA256:
`c3d8c324d06151e87dcc255d46c5997344c8f8bb37c94ba40a41fb2790548f76`.
Preserve the pre-import harness failure and all 254 action/state journal rows.
Final review adds strict manifest membership, bounded immutable correction
samples and precomputed slopes, and reserves the final geometric audit within
the existing evaluation budget. Twenty-six distinct focused cases pass. A
saved-only check verifies all 254 correction samples bit-exact and unchanged
action arithmetic; no fit or rollout is repeated. Full lane handoff and final
test/source pins: `.local/source-contact-trajectory-01/HANDOFF.md` and
`release.json`.

Root's `.local/dense-contact-root-audit-01.json` independently verifies all 47
original portal files, three new attachments, 409 contact source members and 306
original runtime members. This hash audit performs no inference or physics.
The deduplicated checkpoint receipt
`.local/dense-contact-checkpoint-verification-01.json` records **352 distinct
passing Python cases**, 17 portal DOM groups, and an independent readback of all
49 sealed body members (98,011,052 bytes). These are scoped software checks, not
the complete repository or independent physical validation.

## Resume and publication

Next: finish compact scene provenance and its independent review; then carry the
saved body into the ordinary preflight/portal path without rewriting old failures.
Contact work needs a materially different evidence-backed hypothesis, not
another sweep of the rejected correction. Full-task learning and broader
physical/generalization evidence remain unfinished.

The ordinary portal reuse design is retained in
`.local/portal-body-reuse-design-01/REVIEW.md`: an explicit stopped-portal CLI
creates a new source-bound child, copies the verified body, and queues the normal
preflight/quote/consent path. A sealed reuse receipt must distinguish copied body
custody from new inference. This is a design, not an implemented intake feature;
it depends on compact scene qualification and must not rewrite failed parents.

Checkpoint **`f7a2030b54adc89bb7c74407fb7028e334ed322f`** is pushed to private
`origin/main`; independent `git ls-remote` matches exactly. Publication receipt:
`.local/dense-contact-publication-01.json`. Owned paths/index were clean and the
stash list empty after publication; separate avatar changes remain untouched.
No local video, study, provider ledger, key or model was committed.

### Next parallel stage — November 2, 2026, 6:05 PM PST

Recovery and seed-only workers are terminal/reaped. Compact scene work is active:

- Source agent: versioned `shared_human_projection.py`, full-body admission and
  core scene/seed/human/packaging/source-proof integration. Exact original
  `human.json` stays sealed; all runtime reference/root rows and physical fields
  stay unchanged. No increase to the generic 500 KB limit.
- Runtime agent: strict full-sidecar/projection joins in Muse attachment and
  build reporting, plus independent core packaging review.
- Contact agent: strict joins in equipment proposal, reconstruction and learning
  intake, plus a full consumer audit for omitted-confidence fallbacks. The
  completed rejected contact trial is not rerun.
- Root: confidence-weighted joint-task guard, integration and publication.
  Nine focused task tests pass in `.local/compact-human-task-tests-01.xml`.

The projection marker cannot stand in for complete body evidence. Consumers
needing source confidence/audits must read the pinned full sidecar or explicitly
reject the projection. Required verification includes exact physical/XML/action
parity, source/sidecar tamper rejection and one fresh saved-body preflight after
integration. No inference, body reconstruction or paid call is needed for this
stage. Preserve every previous failure and separate avatar edit.
