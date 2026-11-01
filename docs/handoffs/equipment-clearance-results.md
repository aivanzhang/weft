# Joint-only clearance comparison

Completed November 1, 2026, 10:19 AM PST. **Rejected for promotion:** the new
correction did not pass the task and worsened equipment displacement and speed.
It remains an experimental API, not a new lab default or learned policy.

## Matched comparison

One fixed-config 254-control trial wraps the pinned v5 policy with the
[authored clearance controller](equipment-clearance-controller.md). The baseline
is the already-recorded full learned episode, not a repeated rollout.
The initial snapshot matches exactly: seed 29, no jitter, same scene, task,
source references, equipment, physics and evaluator. The controller's module is
the only addition to the frozen runtime; all previous 291 files remain exact.

| Metric | Saved v5 | V5 + authored correction | Frozen limit |
| --- | ---: | ---: | ---: |
| Full interval completed | Yes | Yes | 254 controls |
| Task success | No | No | Every gate must pass |
| `sam-2` displacement, model units | 0.105749 | 0.285829 | 0.03 |
| `sam-2` rotation, radians | 0.650340 | 0.064600 | 0.15 |
| `sam-2` speed, model units/s | 0.722086 | 1.341094 | 0.15 |
| `sam-2` spin, radians/s | 5.624470 | 3.726445 | 0.75 |
| Body-angle RMS error, radians | 0.149185 | 0.145349 | Unchanged |
| Maximum floor penetration, model units | 0.003571 | 0.010387 | Unchanged |
| Mean reward | 0.745387 | 0.743459 | Not the success definition |

Both trials pass the final body/floor and `sam-3` gates; both fail only
`equipment_stability:sam-2`. Better rotation alone is not a successful repair.
All 34 changed actions respect the 0.25 joint-correction cap. All 254 root
commands equal the base controller evaluated at the **same current state**;
the full root trajectory need not equal the saved baseline because the changed
joint dynamics affect ordinary root feedback. No equipment-force channel is used.

## Verification and bounds

The prepared controller makes no new model/data, forward or integration call
and does not modify live joint positions/velocities during action selection.
One detached scratch-state central-difference check validates an active
positive-gap derivative: analytic `0.7642962694647119`, numerical
`0.764296269634146`, absolute error `1.6943e-10`. This checks one geometric
gradient, not the controller's ability to avoid collisions.

The bounded supervisor finished in **1.704560 seconds**, exit 0, process reaped;
child high-water RSS **134,938,624 bytes**. Counts: 14 model compilations,
two data objects (one detached derivative check), 258 forwards, one reset,
254 controls, 4,064 physics substeps, 27,571 distance queries and 608 Jacobians.
Zero decoding, rendering, replay, inference, training or paid requests. The known
MuJoCo startup subprocess probe was denied. No process remains live.

- Output: `artifacts/equipment-clearance-native-01/`, three files /1,111,575 bytes.
- Result hash: `818b75dbf611468dc1fc5fe854cc40c9e76b3b76133d6477351d5f5fbfe5a7fb`.
- Trace hash: `3d5aff57f981604c40469620a4df4c3d3e6f672903710e46c2cb0cee0345dc90`.
- Plan: `.local/equipment-clearance-native-01/plan-frozen-01.json`,
  `5b28aa3dc967f4f3a9822a67e245583c9adb74dbab875c9da7855d0f159e5abb`.
- Worker: `18cfc20b01d7c33d3b41453ee9b6e4f5c73c1e999494ac39c452700307611741`.
- Runtime: `.local/equipment-clearance-native-01/runtime`, 292 files /6,326,469
  bytes, manifest `13fc911251f537609b8727999484bd39f26e2c852c7f14119e4379d9a9ca885d`.
- Base policy: `47e4d2034233606d46bb1a94052a7ea35812bccf77d7b3449d3d3bb1dcc5c015`.
- Source scene: `d8e3e6df7bf640f0bcc1d9088cd96edd70bfad00ac0e161d08b3d0a80a36564d`.
- Task: `031105ec0c2deb0f633211f3f4dfa58ed789b4a2f181957f0b6b1dbd5e0b12da`.
- Protected evaluator remains
  `c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb`.

The native point-distance/Jacobian usage follows the
[MuJoCo API](https://mujoco.readthedocs.io/en/latest/APIreference/APIfunctions.html#mj-geomdistance);
the executed environment remains pinned to MuJoCo 3.17.0. The successful software
experiment is separate from its failed task outcome and unverified physics.

## Saved evidence changes the next action

Saved trace inspection finds speed/spin fail at control 44, five controls earlier
than v5; displacement fails at 45. The correction is neither absent nor blocked:
32 of its 34 active controls saturate the joint cap; 21 controls reach a pair
force cap. No unavailable, unactuated or truncated pair rows occur.
Before step 44 the right forearm is 0.0072013 model units away and closing at
2.021923 model units/s. The controller alternates inactive and saturated commands
around the impact. See `.local/equipment-clearance-controller-01/trace-analysis-01.json`
(`a414cb540ea42683f3838170cc91894bcae7221d7ef821e9006016c630b71604`).

Do not repeat this fixed trial, increase a gain-search budget, relax gates or
enable this as a default. The next distinct hypothesis is a bounded,
dynamics-aware minimum-change joint-action projection: account for the base
torque, joint inertia and braking distance before choosing a correction.
First establish its detached-state prediction and action bounds; a proximity
force alone is insufficient on this case. Required-contact tasks need separate
handling, not blanket repulsion. Reconstruction uncertainty and independently
measured unfamiliar-video contact remain unresolved.
