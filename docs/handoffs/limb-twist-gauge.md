# Source-bounded limb orientation

## Verified local proposal — November 2, 2026, PST

Optional joint-image schema 3 now preserves the complete schema-2 motion fit,
then selects a source-bounded axial orientation. It improves source image fit
and some native force probes, but **does not make the motion force-feasible**.
Keep it opt-in. Original-world controls have mixed results; no default,
learned-policy or physical-accuracy claim is supported.

Root owns the full goal, comparison/recording, publication and combined handoff.
This lane owns only the seven implementation/test/example/handoff paths below.
No commit, push, browser, server, deployment, inference, paid request, key or
budget-ledger access was performed in this lane. Source-only fitting excludes
published-reference/assessment coordinates. The $5 shared Meta budget is untouched.

## Implementation and boundaries

`limb_twist_gauge.py` rotates the local limb frame around its upper-segment
axis. During this second stage, the upper endpoint, upper direction, hinge
bend, lengths, proximal origin, parent frame, source clocks and cameras remain
fixed. A nearly straight limb can have poorly identified axial orientation;
this is an explicit hypothesis, not recovered anatomy.

Frozen safeguards:

- At most 90 degrees of local axial rotation, 0.20 rad per update, 16 iterations
  and six backtracks. Maximum distal displacement is 20 mm from the **preceding
  motion fit**, not the earlier raw construction.
- Retain every original image observation. Complete image loss cannot increase;
  no individual residual can increase by more than one assumed image standard
  deviation. All rejected proposals remain in the fit record.
- Missing observations freeze that pose. Missing neighbors or gaps over 0.25 s
  break temporal coupling. Initially singular Euler charts are held fixed.
- Keep the existing uncalibrated 0.0025 image fraction and 60 rad/s² source-time
  acceleration assumption. The gauge objective uses the three proximal Euler
  coordinates; hinge acceleration is constant during this stage. Schema 3's
  total report includes all four coordinates.
- Keep the original initial-stage 0.30-rad correction bound. Larger total
  corrections require a validated axial reconstruction and its geometric
  envelope; they are not unrestricted Euler changes or relaxed task gates.
- Roots, floor, body shape, contacts, motor caps, task and evaluator remain
  unchanged. No native force result feeds back into source fitting or settings.

The scalar two-neighbor solver uses linear storage. All stage inputs, outputs,
trials and assumptions remain recorded. Restoration checks validate both stages
before restoring the original constructor for its existing integrity checks.
Schemas 1 and 2 retain exact output bytes on eight frozen-runtime cases.

Owned paths:

- `src/weft/limb_twist_gauge.py`
- `src/weft/joint_pixel_trajectory.py`
- `src/weft/multiview_joint_pixels.py`
- `tests/test_limb_twist_gauge.py`
- `tests/test_joint_twist_gauge.py`
- `examples/joint-twist-gauge-uncertainty.json`
- `docs/handoffs/limb-twist-gauge.md`

## Frozen source and native results

The unchanged ASPset `04ac-0006` source retains 47 poses, 555 source slots and
853 qualified joint-image constraints. The gauge changes 91 limb poses on the
left hip/shoulder. Both right limbs reject every image-regressing proposal and
remain unchanged from schema 2. No unseen joints are filled.

| Matched source check | Motion fit, schema 2 | Axial stage, schema 3 |
| --- | ---: | ---: |
| Complete weighted image loss | 648.375783 | 636.150600 |
| Four-axis acceleration loss | 1,114.352 | 898.297988 |
| Common constraint-free midpoint probes over motor caps | 14/44 | 11/44 |
| Maximum common midpoint force/limit | 2.129317× | 1.681514× |
| RMS of common midpoint maximum force/limit | 0.896600 | 0.781276 |
| Constraint-free interior transition probes over caps | 42/42 | 42/42 |
| Maximum common interior-probe force/limit | 9.865684× | 9.865684× |
| Failed source/grid floor checks at original 0.025 tolerance | 0/47; 0/111 | 0/47; 0/111 |

31/44 common midpoint samples and 26/42 common interior probes improve. Native
transition probes remain finite-window velocity-jump tests, not exact path
derivatives. Contact-constrained samples are retained separately in the raw
diagnostic; no contact assistance establishes physical validity.

Upper endpoints stay fixed within 2.24e-16 m; maximum distal displacement is
0.0200000000000023 m including floating-point roundoff. Maximum gauge-stage
Euler change is 0.464969 rad. Total correction from the earlier pre-motion
constructor reaches 0.764969 rad. Minimum sampled floor distance becomes
−0.01796994 model units; passing the original floor tolerance does not qualify
contact or remove the small penetrations.

### Separate original-world control comparison

Root completed four starts using the original constructed task/targets and
unchanged forces, reusing the four saved schema-2 baseline trials. No baseline
rerun or source retuning occurred. Root's saved summary is
`.local/twist-gauge-comparison-01/preliminary-saved-summary.json`; its independent
audit is root-owned. These are simulated errors against original constructed
targets, **not measured physical/reference coordinates**.

| Four-start mean | Schema-2 commands | Schema-3 commands |
| --- | ---: | ---: |
| Original-task successes | 4/4 | 4/4 |
| Twelve-joint endpoint RMSE, model units | 0.121054817 | 0.115872172 |
| Angular RMSE, radians | 0.183876116 | 0.189469880 |
| Root RMSE, model units | 0.027472468 | 0.027314620 |
| Peak floor penetration, model units | 0.001991448 | 0.001902797 |
| Mean reward | 0.748026828 | 0.737682482 |

Endpoint error improves about 4.28%; angular error worsens about 3.04% and
reward falls. This is a **mixed outcome**, not an overall controller improvement.
Root's new 184-action recording/export replays all 185 events exactly. See
`artifacts/twist-gauge-comparison-01`; do not duplicate its completed replay.

## Verification and resumable artifacts

139 distinct focused tests pass, including analytic axial geometry/derivatives, a
900-element banded residual, missing/gap boundaries, source-image guards,
schema-3 native geometry, resealed-input tampering and the real CLI's
record/export/exact-replay path. Receipt:
`.local/limb-twist-gauge-tests-01.xml` plus the two added singular-chart/strongly
observed rejection cases in `.local/limb-twist-gauge-boundary-tests-01.xml`.

Eight schema-1/2 cases are byte-identical to the prior frozen runtime: ordinary,
missing/gapped, conflicting-view and two-pose inputs in each version. Receipt:
`.local/limb-twist-gauge-legacy-parity-01.json`.

| Completed lane job | Elapsed / peak RSS | Bounds: wall / CPU / sampled RSS |
| --- | --- | --- |
| Four-limb source screen | 0.415 s / 52,281,344 B | 60 s / 45 s / 512 MiB |
| Native force/floor check | 0.203 s / 84,787,200 B | 30 s / 25 s / 384 MiB |
| Full original-source constructor | 2.752 s / 187,973,632 B | 60 s / 45 s / 512 MiB |
| Synthetic 900-frame/four-view fit | 2.236 s fit / 51,691,520 B | 30 s / 25 s / 192 MiB |
| Saved-only arithmetic audit | 1.118 s / 126,763,008 B | Completed local arithmetic; no native execution |

RSS checks are cooperative samples, not hard operating-system memory limits.
The large analytic case retains all 7,200 images exactly and moves no endpoint;
it is a responsiveness/storage check, not another source/video qualification.
Native work uses one model, two data objects, 92 inverse calls, 158 forward
calls and zero physics steps. Construction uses four model compilations,
53 data objects, 253 forwards and zero physics steps. These measurements do not
establish browser performance or full physical accuracy.

The full constructor accepts `human.json`; all four output-angle arrays match
the frozen standalone source screen **exactly**, and shape/roots/frame/mobility/
force limits equal the prior body. The saved-only audit verifies 401 package
members, all 853 image slots, 188 limb poses, 39 retained proposal evaluations,
92 native samples and 2,024 force-axis balances without new inverse calls.

| Artifact under `artifacts/` | Members | Manifest SHA-256 |
| --- | ---: | --- |
| `limb-twist-gauge-body-01/run` | 385 | `a85a6decd7d2f4849ef3944c28679b9de5c8741c198056693809d6e978d24448` |
| `limb-twist-gauge-source-02` | 7 | `2f6e5b567fe228529dcc91fd89c1f1286128b919f48e00a916a76e21d3d398d6` |
| `limb-twist-gauge-native-02` | 7 | `7d98626397efa6a745a9a5a82e611580ab211912d3bd84c8fdb03300c0a96626` |
| `limb-twist-gauge-performance-01` | 2 | `2616e470e7c8071f0cdbfaa2c84b0c5db7a2a97b7902f3abc0edb12bdaaa496c` |
| `limb-twist-gauge-saved-audit-01` | 3 | `4824f918c50d38863ef96abf18d09673c1c13770f7f2298603394cb10b5717fd` |

Controls, frozen runtimes, exact plans, launch hashes, process receipts and
logs are under `.local/limb-twist-gauge-{source-02,native-02,body-01,performance-01}`.
PIDs 8298, 8728, 9878 and 10279 are terminal/reaped. The body plan preserves the
original source/camera seals. The example is byte-identical to the body's frozen
schema-3 settings. No live job from this lane remains.

Completed construction command; **do not rerun into this existing output**:

```sh
PYTHONPATH="$PWD/.local/limb-twist-gauge-body-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft camera-body \
  --run artifacts/aspset-next-original-body-01 --control-joint right_knee \
  --camera artifacts/aspset-next-camera-input-01/camera-left.json \
  --ground artifacts/aspset-next-camera-input-01/ground-left.json \
  --root-view artifacts/aspset-next-camera-input-01/root-view.json \
  --additional-view artifacts/aspset-next-camera-input-01/right-view.json \
  --body-view both --fit-limbs --partial-segments \
  --fit-joint-pixels .local/limb-twist-gauge-body-01/joint-twist-gauge-uncertainty.json \
  --out artifacts/limb-twist-gauge-body-01
```

### Preserved harness failures

- `.local/limb-twist-gauge-source-01`: the parent RSS probe attempted `/bin/ps`,
  which the sandbox denied. No worker result or numerical output was produced.
  A narrow exact-command process check found no matching worker before retry.
  The failed parent has **no child-reaping receipt**; preserve that limitation.
  Attempt 02 uses in-child RSS samples plus parent wall/output limits.
- `.local/limb-twist-gauge-native-01`: the frozen older native function does not
  accept the new optional `check_budget` argument. It stopped after one model
  compilation, before any data allocation or inverse/forward call. PID 8680
  is terminal/reaped. Attempt 02 removes only that unsupported argument;
  pre-call native wrappers still enforce counts/time/sampled memory.

Neither failure caused settings changes, discarded evidence or an unchanged
duplicate job. Original source, native and scored runs remain intact.

## Remaining work and completion boundary

- [x] Root completed the independent original-world comparison audit and shared
  sandbox integration below. Publication is tracked by the root receipt; this
  lane did not commit or push independently.
- [ ] Resolve the remaining 42/42 interior force-probe failures using new
  source-qualified evidence/methods, without enlarging motor caps, relaxing task
  gates or tuning from published assessment coordinates.
- [ ] Qualify additional subjects/cameras/visibility patterns and independent
  ground/equipment contact; this exposed development clip is not a final holdout.
- [ ] Demonstrate full-task learned behavior and fresh independent physical
  evaluation. These authored control trials and simulated targets do not satisfy
  those completion gates.

The complete video-to-3D RL goal remains active. Preserve the separate root,
ground/equipment and avatar work; no blanket claim of generalized reconstruction,
force feasibility, physical transfer or completion is authorized by this result.

## Root integration and playable shared sandbox

Verified November 2, 2026, PST. The three delegated lanes finished their scoped
work. A later agent turn hit the usage limit before runtime integration; root
completed that work locally. No live delegated or native jobs remain.

```sh
PYTHONPATH="$PWD/.local/twist-gauge-shared-01/runtime/src" \
  .venv-lab/bin/python -B -P -m weft lab-console \
  --resume artifacts/twist-gauge-shared-01/lab --seed 29
```

Use `controls`, `mode reference`, `record start`, `step 3`, `mode manual`,
`motor left_knee 0.1`, `step 2`, `mode coast`, `step 1`, `record stop`, `files`,
`quit`. These exact controls ran successfully. Recording/export replays all six
actions and seven events exactly; the ordinary saved recording receipt is
available. All fourteen evidence categories load. This ready lab uses the new
candidate's targets; it does not replace the original-task comparison.

Runtime repair: schema 3 now pins `limb_twist_gauge.py` through candidate snapshots,
manual labs and both policy-receipt dependency sets. Schema 1/2 native dependency
lists remain unchanged. Candidate snapshots also retain their existing
`camera_shared_receipt.py` import. A resealed lab missing the new dependency
rejects. The first test attempt incorrectly mocked a snapshot function the loader
does not call; corrected contract-removal test passes. No product guard was weakened.

**321 distinct scoped checks pass** across the lane checks, 47 existing
silhouette regressions and 101 runtime/candidate/lab/receipt cases; use the latest
result per test. Exact receipt list:
`.local/twist-gauge-shared-01/saved-verification.json`. No unchanged full suite rerun.

Actual shared journey: PID13565, **7.595 s /386,007,040 B**, 34 model compilations,
four data objects, 69 forwards, 96 physics substeps, eight video read calls and
seven written frames. External 60-wall/45-CPU-second cap; 768 MiB sampled RSS;
256 MiB/1,200-file output bounds. Source fitting, inference, training and API
calls: zero. All six recorded actions replay exactly. Terminal/reaped.

| New artifact | Members | Manifest SHA-256 |
| --- | ---: | --- |
| `twist-gauge-shared-01/candidate` | 69 | `d227e06c9d9ee7786db54a1bd0a81811848a5577b62d09ad857b50ce7252e22b` |
| `twist-gauge-shared-01/lab/run` | 398 | `c292a7cd6b007e8835f44b8d60c5f7d5897068c0e54155b93118bc2045ef5585` |
| `twist-gauge-shared-01` | 482 | `1a73d7f0af8cd20c07fd23c6d273bd5e6dd174f3e15ef43624a49280a17cdead` |

The last package captures lifecycle/recordings at acceptance; only completed
`run/` is immutable during later interaction. Recording:
`lab/sessions/session-6be3ca3c0193`. Saved audit checks 867 source/shared members
and all317 frozen runtime files. Original body remains exact. Resources/plans:
`.local/twist-gauge-shared-01/`.

### Original-task comparison: completed, not repeated

Four newly executed gauge trials reused four saved schema-2 baselines with exact
matching initial states. Original scene, task, scorer, limits and root commands
at the same current state stayed unchanged. Mean endpoint RMSE improves
.121054817→.115872172 model units (4.28%); angular RMSE worsens
.183876116→.189469880 rad (3.04%); reward worsens .748026828→.737682482.
Root error and mean peak penetration improve slightly. Both pass4/4; no added
success, learned policy, physical reference claim or default promotion.

One184-action recording has185events, exact export replay and an available
ordinary receipt. No duplicate replay or baseline rollout ran. PID10161:
10.074 s /402,964,480 B, 8,832 native substeps, terminal/reaped. Output:
`artifacts/twist-gauge-comparison-01`, session `session-e08ce85650b8`.
Saved-only independent audit verifies1,200members,736actions and185events:
`artifacts/twist-gauge-comparison-audit-01`, seal
`982a224fe2d921606d8448b569e636aea1836716ffcb5309a2f3345474c66a35`.
Verifier: `.local/twist-gauge-comparison-01/verify_saved.py`.
