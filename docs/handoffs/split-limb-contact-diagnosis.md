# Saved equipment failure and next control experiment

Updated November 1, 2026, 10:10 AM PST. The complete video-to-world goal remains
active. This is exposed development evidence, not a new holdout or physical
contact validation.

## Completed

The existing `inspect_recording_failure` inspected the closed recording at
`artifacts/split-limb-feedback-play-05/lab/sessions/session-1dcf521d278e`.
The exact frozen lab runtime was reused; no new trajectory, replay, media decode,
training, API call, or source modification occurred. The current remote was
fetched and matched local `main` at `deeabdc` before implementation.

The recording has two episodes. Do not confuse the first mixed manual/coast/
authored/learned demonstration with the second full learned trial:

| Episode | Controls | First equipment limit crossing | Co-occurring saved contact |
| --- | --- | --- | --- |
| 1 | 12 mixed controls | Step 10, 0.300–0.333 seconds | Right shin / `right_knee_geom` |
| 2 | 254 learned controls | Step 49, 1.600–1.633 seconds | Right forearm / `right_elbow_geom` |

In episode 2, `sam-2` speed and spin first cross their frozen limits at step 49;
orientation crosses at 50, displacement at 51. The declared reference clock
places the first crossing between source reference frames 48 and 52, with
49–51 missing. This is not native-video synchronization. A boundary contact
co-occurring with a cumulative peak does not prove the exact causal substep.

The full episode uses seed 29, `jitter=false`, `jitter_scale=0`, and completes
254 controls / 8.466667 simulated seconds. Only `equipment_stability:sam-2`
fails at the end; the transient running body error recovers. Final equipment
maxima: displacement 0.105749 model units, rotation 0.650340 radians,
speed 0.722086 model units/s, and spin 5.624470 radians/s.

## Reproducible evidence

- Result: `artifacts/split-limb-contact-diagnosis-02/` (three files, 67,243 bytes).
- Timeline: `2988d86eb36f7818a81e60e9d575a0b5cb791a972cab75dbc4ed628c8e7a5b73`.
- Verification: `6bdefce34aed06e55ba9b8ba79a89cd984b4fd824b9bb342494fc75cee6102d0`.
- Plan: `.local/split-limb-contact-diagnosis-02/plan-frozen-01.json`,
  `7006ddbc937c44dea68d224975c0f70234c61b96bd87d2174225a5908024a243`.
- Worker: `6d6a6fe8d834d76205f979ace0702cfb5cdc9735d9b96d3eff996dcb289a602a`.
- Runtime: `.local/split-limb-feedback-play-04/runtime`, manifest
  `6acc810d058c7bf422110e7f411738265d259ff4c6d11728705670318fa034f5`.
- Recording manifest:
  `94ee100c5093ad51880578972aa123cf01b11f75a791362379ea23a053db91cd`.
- The supervisor completed in 0.983026 seconds, child high-water RSS
  166,592,512 bytes, exit 0, process reaped. No job remains live.
- Seven model compilations validate the existing mass/action contract;
  126 quaternion-matrix conversions, zero data constructions, steps, controls,
  replays or decodes. The known MuJoCo startup subprocess probe was denied.
  All 295 pinned runtime/input identities remained unchanged.

Attempt 01 is preserved as failed. Its harness mistakenly blocked quaternion
matrix arithmetic needed by ordinary geometry validation; it stopped before
model construction or simulation. Attempt 02 allows only this mathematical
operation in addition to bounded model compilation. No stepping restriction,
scorer, source, task or success limit changed.

## Next implementation and ownership

- Root owns the bounded native comparison and eventual lab integration.
- The equipment-control agent owns a new explicitly authored, joint-only
  clearance correction and analytical tests. This reads current simulated
  geometry/relative velocity, unlike v5's joint-only position/velocity features.
  It is not another learned policy or a collision-avoidance guarantee.
- Test the correction around the exact pinned v5 base policy against the saved
  nominal baseline. Preserve the initial state, body/root references, geometry,
  scorer, equipment and task limits. Never apply forces to equipment or modify
  root actions. Reject required-contact tasks until separately supported.
- One fixed-config full-interval development trial comes before any expanded
  learning search. Retain failures and compare every gate, not reward alone.
- A separate reconstruction agent owns conditional finite-support envelopes
  and their existing public request/report path. No overlapping controller
  edits, new acquisition, agreement acceptance or paid work is authorized.

The original arm/depth/geometry failures remain relevant. Do not rerun them or
invent a hidden equipment cavity. Independent unfamiliar-video calibration and
physical contact accuracy remain unresolved; they do not block this local work.
