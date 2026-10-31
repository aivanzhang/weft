# Foot admission gap

Owner: `/root/exercise_reconstruction`. Updated 2026-10-31 14:03 PDT.

## Completed diagnostic scope

Diagnose the preserved initial-floor rejection in
`artifacts/coupled-feet-lab-acceptance-01` using saved first-pose data only.
Root owns publication/shared consumers; this lane owns this note and a new scoped
`.local/` arithmetic receipt/helper. No body, root, floor, limb, admission-gate,
source-threshold, `foot_body` or `shared_world` edits. No acceptance retry,
inference, media decode, acquisition, paid/network call or native run is planned.

The candidate CLI succeeded, but the lab CLI rejected initial
`right_foot_geom` / `mobile_floor` intersection. The terminal failure records
zero controls, zero native substeps and one reset. This is exposed development
data, not unfamiliar/final evaluation. The original 8/118 source-pose and 16/235
control-grid floor failures remain unchanged.

## Initial evidence and bounded check

- Candidate: `artifacts/coupled-feet-lab-acceptance-01/candidate/scene.json`.
- Reference: `artifacts/coupled-feet-source-input-01/foot-reference.json`.
- Frozen implementation: `.local/feet-lab-combined-runtime-01/src/weft/`; the five
  inspected/used modules match the failed acceptance's sealed implementation.
- Existing body `floor_fit` says the first old collision surfaces were placed
  0.005 model units above the plane. The foot capsules were not in that fit.
- Saved first right-toe offset is −0.12889564019615427 model units in z relative
  to the unchanged ankle; the added capsule radius is 0.018 model units.
- Initial ankle code computes `knee_rotation.T @ desired_world_foot_rotation`.
  No obvious extra body-frame rotation appears in this inspected path.

One native-free calculation reused frozen `spatial_pose.rotation_zyx` and
`rotation_to_zyx` for the exact first body/reference pose. It compared derived
XML-local capsule endpoints with the saved offsets and calculated signed plane
gaps for original shins and added feet. Bounds were one frame/two feet, ≤16 MiB
input, ≤1 MiB receipt and ≤5 s computation; $0. Native/model/media/network and
subprocess paths were denied. No later pose or held-out outcome was inspected.

## Verified outcome

Command:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-lab/bin/python .local/foot_admission_analytic_01.py
```

Completed once in **0.055343 s**, reading **2,681,500 bytes**. Every source/body/
reference and inspected runtime pin matched. No native model/forward/step/reset,
media decode, inference, API/provider call, source edit or acceptance retry.

| Initial geometry | Left | Right |
| --- | ---: | ---: |
| Ankle height above supplied plane | 0.0805880024 | 0.0430000000 |
| Original shin capsule gap | +0.0425880024 | +0.0050000000 |
| Added foot capsule gap | −0.0143141240 | **−0.1038956402** |
| Beyond unchanged −0.025 limit | No | **Yes** |

All distances are **body-scaled world/model units**, not independently calibrated
metres. The scale bridge is 1. The original right-shin gap agrees with the saved
0.005 floor fit. The right-foot result follows directly from
`0.043 − 0.12889564019615427 − 0.018 = −0.10389564019615427`.
The left foot overlaps the plane slightly but remains within the existing
tolerance; it was not relabeled contact-free.

The knee-local ZYX rotation round-trip differs from the saved desired orientation
by at most **1.67e−15**, and the derived endpoint offsets differ from the saved
reference by at most **1.95e−16**. No reversed knee axis, doubled body-frame
rotation, duplicated shin offset or capsule scaling mismatch appears in this
inspected arithmetic. This predicts the actual failing foot without a native
rerun. It is **not** a compiled-engine equivalence test or physical contact proof.

## Evidence pins

- Receipt `.local/foot-admission-analytic-01.json`:
  `229fd1c39057df96b10e373ac6d804a16d6716f4c768e801bfe26249372d5067`.
- Helper `.local/foot_admission_analytic_01.py`:
  `43db67ae4fd1bdc6ef84f3ba778afd5569d4e83bced51a733ab083dd12a284b4`.
- Failed acceptance manifest:
  `8df1993d1962b4892533b8da7a7244e4583710f2b4773d645d891e15845b1fc9`.
- Candidate scene bytes:
  `b966f5353a47300058bf7e6d1357c7af90934e2678d29128df5a4da23c858419`.
- Body canonical hash:
  `324fcc2ebda4b5144dafa531db1e4018ecf420b2ff88b56337097f30bfe05e37`.
- Original foot-reference bytes:
  `79c43517dc98224f40877a2caac4967cfc0a75454045aa680fcf0976f0880729`.

The receipt lists all exact runtime/member hashes. Original source identity is
`ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`,
source frame 0 / time 0. Body and foot clock/identity bindings agree.

## Method-level issue versus this clip

The general issue is that adding collision feet to a body whose initial floor fit
excluded those feet does not guarantee a feasible combined body. Preserving the
old geometry/gates correctly exposes that mismatch. Prefix-fixed foot shape,
monocular foot directions, old limb lengths/root placement and an uncalibrated
horizontal floor are separate assumptions, not jointly measured support geometry.

For this exposed clip, those exact retained estimates place the right toe far
below the floor. The saved values explain the rejection; there is no supported
transform/attachment code repair to make here. Do not raise the root, lower the
floor, shrink the foot/limbs, rotate the source direction or loosen the gate to
make the lab launch pass. The previous body feasibility report is not a physical-
feet admission result. The separate two-view jogging adapter is different source
evidence and cannot substitute for this clip's missing calibration.

## Next safe action

- [x] Explain this first-pose rejection without changing geometry or rerunning it.
- [ ] Root may expose a foot-aware pre-lab rejection report, preserving the base
  scorer and failed candidate. This improves reviewability, not reconstruction.
- [ ] Reconstruction improvement needs independent source-bound camera/scale/
  ground or support-surface evidence, then a separately identified construction
  hypothesis checked against unchanged gates. Existing `source_surface_anchors`
  can express conditional source-to-surface coordinates, but this saved clip has
  no qualified calibration/contact measurements; authored input stays authored.

**No static native audit is recommended now:** saved geometry already explains
the observed rejection. If later integration produces a conflicting compiled
transform, propose one bounded zero-step comparison before doing it. No live
process or Git mutation. This note is released to root; shared consumer files
remain untouched. General unfamiliar-video reconstruction/contact remains open.
