# Active goal: deliver the Weft hackathon demo

**Quality issue reopened:** [Continuous motion verification](avatar-motion-quality.md) exposes defective Studio skin weights. Import/replay plumbing passes, but custom-avatar visual motion must not be called complete; the continuous robot clip is the current usable preview.

Updated November 3, 2026 (PST), at the user's explicit direction. This is the
current delivery goal. The broader physical-reconstruction research remains a
backlog, not a prerequisite for this rough end-to-end demo.

**Demo journey verified November 3, 2026 (PST):** [Acceptance evidence and remaining visual limits](hackathon-demo.md). Generalized avatar motion, saved-video sandbox, normal portal recording/export, Studio delivery and a real Muse API review pass. This does not complete arbitrary-video physical reconstruction.

## User journey

1. A user uploads a video in the local web portal.
2. The portal shows real processing stages and inspectable video, pose/body,
   reconstruction, assumptions and failure artifacts.
3. The result opens as a usable 3D RL environment: reset, step, manual/reference
   controls, visible task/reward and available bounded learning/comparison.
   Exercise videos are the primary demo path. Explicit body+ground approximation
   is acceptable; never call omitted equipment reconstructed or claim physical
   accuracy. Do not require perfect motion tracking or successful training just
   to permit a valid manual sandbox.
4. The user can keep the original body, choose a compatible built-in avatar, or
   create an avatar through the existing Avatar Studio and import its delivery.
   Integrate the work from session `01a18ebe-a17c-7cc0-9117-06ca476fb31c`, now in
   `tools/avatar-assets/`; do not build a competing asset pipeline.
5. A generalized motion adapter maps the RL body's joints, root, orientation,
   scale and simulation clock to the selected rig. The avatar follows actual
   environment state, not its unrelated baked animation. Qualify Muse panda,
   robot and a compatible custom rig. Retain a skeleton overlay and immediate
   return to the default body. Clearly reject missing/incompatible mappings.
6. Recording, playback and export preserve actions, simulation state and avatar
   selection/mapping. Changing appearance must not change physics, rewards or
   controls.
7. Muse Spark participates meaningfully in the same workflow through the real
   bounded project API: review source/world assumptions and explain suggested
   checks. Preserve actual request/result/cost; label advice as a proposal, not
   verified physics. The Muse avatar and Muse Spark API are distinct features.

## Motion priority clarification

The user explicitly corrected the request: **do not focus on individually
repairing flight/hopping clips.** Fix the generalized avatar motion/retargeting
path relative to the RL environment. Keep those authoring failures preserved,
but they are not blockers to a correctly mapped humanoid demo. Do not claim that
an arbitrary animal topology can automatically follow a human skeleton.

## Delivery and verification

- Finish the smallest coherent implementation using the existing portal,
  simulator, Studio, importer, renderer and recorders.
- Provide one documented launch workflow and a licensed known-good example
  while retaining ordinary video upload. The user manages persistent servers.
- Test a real end-to-end journey: source video → completed simulation → manual
  actions → avatar substitution → recording/export/replay → Muse response.
  Verify state/reward parity for identical actions across appearance choices.
- Verify the Studio-to-Weft delivery path with real assets. Distinguish newly
  generated output from reused previously generated assets. Show unsupported
  jobs honestly; never substitute a fixture for a real generation claim.
- Run focused checks, inspect the actual outputs and record exact commands,
  artifact identities, limitations and remaining non-demo research. No new
  research sweeps, appearance perfection work or broad physical-validation gate.
- Wrap up the parallel agents after their product tasks; checkpoint and push
  relevant verified work. Preserve unrelated dirty files and all prior evidence.

## Unchanged boundaries

API keys only, never ChatGPT-login or coding-session inference in the product.
Use the existing separate project ledgers and authorizations; no budget reset,
cross-project credential borrowing, credit purchase or new subscription. Estimate
any new paid call and preserve unknown charges. No hosted deployment, hackathon
submission or external messages are authorized. No browser interaction unless
explicitly requested; use direct HTTP/CLI and offscreen artifact verification.
No `npm run dev`, automatically opened goal files or clipboard changes.

Keep source/scored runs immutable, saved failures truthful and raw observations
separate from authored approximations. No weaker physics/scoring checks merely
to label a demo successful. Completion means the demo journey actually works,
not that arbitrary-video reconstruction or physical accuracy has been solved.
