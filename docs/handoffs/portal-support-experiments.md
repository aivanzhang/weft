# Portal support experiments

## Verified locally — November 2, 2026 (PST)

**Later update:** [coupled-world Muse context](portal-world-review.md) is now
live-verified. New reviews include authored support geometry/load/control inputs;
the original body-only scope below describes this support-feature checkpoint.
Existing reviews are not rewritten or automatically resubmitted.

Completed exercise projects can create a separate authored-static-support world,
optionally train, and use the existing lab's controls and recording. This does
**not** reconstruct equipment/contact or complete the generalizable video goal.
The hosted site remains offline. Start the local portal yourself:

```sh
.venv-lab/bin/python -m weft portal
```

Choose the original exercise project, expand **Create a separate support
experiment**, enter explicit geometry/load assumptions, and open the resulting
sandbox. The saved negative-control example is already ready; do not rebuild it.

## Implementation and boundaries

- `portal_supports.py` pins/copies the complete body manifest and all members,
  verifies original-video identity, and calls the existing exercise builder.
  No pose inference, API request, generated code or new simulator.
- `portal.py`: authenticated `POST /api/projects/<id>/supports`, strict 128 KB
  request, exact source hash, request-id deduplication, original exercise parents
  only, existing job/storage/queue bounds. Original project/run/task remain intact.
- `portal_worker.py`: same single supervised worker; no provider key forwarded
  for support construction. Worker wall bound 420 s, sampled RSS 1.5 GiB,
  job output 650 MB/6,000 members; construction/training cooperative bound 240 s.
- Form in `assets/portal.*`: one axis-aligned box, named body segment, upward load,
  explicit half-open step window, root assistance and optional training. Coordinates
  are body-scaled world units, not metres. Static material/scale defaults and
  unsupported contact claims stay visible. The API accepts the existing bounded
  1–12-support schema. No automatic placement or silent floor adjustment.
- A child defines a new authored task, not an improvement against the parent task.
  Training uses two generations/four candidates, three changed evaluation starts,
  training seed 1729/evaluation seed 938401/jitter 0.1. These are exposed development
  seeds, not fresh final evidence. Existing scorers and task gates are unchanged.
- `portal_review.py`: support-child reviews explicitly cover original body evidence
  only, not added supports, load windows or child-world controls. Binding and UI
  before/during/after review expose that limitation. No new paid review was run.
- Static-support avatar recording remains unqualified and rejected by the existing
  lab. Source-derived equipment and full Muse-driven repair are still required.

## Native evidence and preserved failures

Parent `4e0f1fe859664bb5ba788214e129451d`, original single-leg squat,
source SHA-256 `ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.
Body manifest `383b32a7bad914b967d672c8bc030af8a2e4833d878d181ad9ee3d36f8e8665a`.
All 37 body members/18,961,850 bytes copied without reinference. The negative-control
box at `[1.9,1.9,0.2]`, size `[0.2,0.2,0.2]`, deliberately cannot supply the left
forearm's requested 5% upward body-weight load during steps `[0,30)`. Explicit
root assistance is enabled; this is not unassisted balance or observed equipment.

1. `.local/portal-support-01/`: failed child
   `b16cc265ebc8486dbef71bcc9c97b778` retained. World construction passed; learning
   custody rejected the incomplete verification runtime, which omitted root
   dependency manifests. `diagnosis.json` pins that cause. Workspace preflight
   reached the backend; frozen preflight identified missing `pyproject.toml`.
   No training/simulation gate was changed to fix this harness error.
2. `.local/portal-support-02/`: complete frozen runtime plus dependency manifests;
   child `639d135203db457ead1bb720cead115c` constructed/trained/evaluated successfully.
   Worker 13.4538 s, peak 392,970,240 bytes; portal observed ready at 14.16 s.
   The harness then used invalid `controller:"learned"` instead of `policy:true`.
   Preserve its failure and partial `session-25072c43a70d`; the ready project was
   neither relabeled failed nor rebuilt.
3. `.local/portal-support-03/`: corrected authenticated interaction check reused
   that completed project; no construction/training repeat. New recording
   `session-58cca8caa40f`, nine actions/ten events, exact state/task replay including
   all three learned decisions. Export receipt is available. Harness 2.5713 s;
   nine HTTP requests mean 150 ms/max 1.154 s including first native lab load. These
   are bounded workload observations, not sustained-load or visual-browser QA.

Ready lab: `.local/portal/639d135203db457ead1bb720cead115c/build/learning_workflow/lab`.
Its 1,325 members verify, manifest
`2a0652b8b1eedec6952f22fe74c252c62662fc47f7566d6662d5b3751ceb0085`.
Parent 1,325 run members verify with unchanged manifest
`d8cfe07cc44e3d36efafcd6039c49435188e684c763e28fb9ce11e5cc8d1fe26`.
`portal-support-03/audit.json` checks the copied body, unchanged ledgers across all
three attempts, replay receipt and all 12 evaluation trials. Every trial fails the
support gate with peak support load 0; authored, learned and coast each finish 3/3
intervals, random 0/3. Selected controller remains `authored_unchanged`: no learned
improvement. The original-body tracking success does not override support failure.

No new API calls: last observed Meta $0.249336 committed/$4.750664 remaining,
OpenAI $0.003252 committed/$4.996748 remaining, zero unsettled for both. All acceptance
servers/workers are terminal and reaped; no root-owned live job remains.

## Checks and publication

35 distinct Python cases pass across:
`.local/portal-support-unit-01.xml`, `portal-support-http-01.xml`,
`portal-support-idempotency-01.xml`, `portal-support-review-scope-01.xml`.
Five DOM UI groups pass via `node tests/portal_ui_smoke.mjs`.
Native acceptance uses the complete frozen runtime in `portal-support-02/runtime`;
post-freeze edits only clarify body-only Muse scope and are separately offline
verified. Do not rerun paid reviews or native training for these label changes.

Worktree ownership: root owns portal code/assets/tests and this milestone's docs.
The parallel avatar session owns dirty `avatar_replay.js`, its test and
`tools/avatar-assets/`; do not stage/reset/overwrite those. Main was refreshed
with `git pull --ff-only` (already current); concurrent avatar checkpoint
`5f2f4be` was present. Stash list empty. Publication receipt will be recorded in
`.local/portal-support-03/publication.json`; recheck remote/state on resume.

## Remaining work and next safe action

- [x] Supply validated authored support context to new Muse reviews, clearly
  separated from recovered evidence. [Live verification](portal-world-review.md).
- [ ] Execute typed Muse proposals as bounded candidate changes and independent
  comparisons, preserving original task/scorer/source and rejected proposals.
- [ ] Finish source-derived ground/equipment, native clock and unfamiliar-video
  reconstruction gaps. Manual boxes are not completion evidence for those items.
- [ ] Qualify support-profile avatars and richer visual placement separately.

Next: extend the evidence-bound Muse proposer/independent-test connection without
new inference or repeated paid calls where saved proposals suffice. Reuse saved
successful labs and failures. The full goal remains active, not blocked/completed.
