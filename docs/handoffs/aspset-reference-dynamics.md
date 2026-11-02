# Source-only motion-force diagnostic

## Verified result — November 2, 2026, 5:51 AM PST

Determine whether the frozen `04ac-0006` constructed trajectory asks for forces
outside its unchanged motor limits, separately from feedback lag. This study
does not fit a body/controller, change task gates, advance physics, or consume
the published joint-reference assessment. Source, floor and force assumptions
stay unchanged. It cannot establish measured biomechanics or physical accuracy.

Root owns `reference_dynamics.py`, its study script/tests and this handoff.
The avatar work remains separately owned. Standing approval covers publication
of verified owned paths only; no new deployment, spend or destructive cleanup.

The frozen target itself exceeds the current motor budget at sampled states;
feedback lag is not the sole cause of the recorded tracking gap. This does not
exclude useful approximate tracking improvements or prove a human cannot make
the movement: model geometry, inertia, damping and force limits are authored.

| Sample | Without active constraints | Above a motor limit | Largest force/limit without constraints |
| --- | ---: | ---: | ---: |
| Constant-speed interval midpoint | 45 | 23 | 4.095× |
| Interior 20 ms velocity-transition probe | 43 | 43 | 10.560× |
| Starting from rest, 20 ms probe | 1 | 1 | 3.158× |

There are 46 midpoints and 45 interior probes in total. The remaining one
midpoint/two knots have active constraints and also exceed limits. Their much
larger demands (up to 1,120×) include forces required to impose a penetrating
target under the unchanged soft-contact model; they are not inferred real
contact forces. No ground or equipment-contact claim is added.

The largest constraint-free midpoint demand is left-hip lateral at source
2.20 s: 122.862 authored torque units against 30. Its coupled velocity-dependent
bias dominates despite that individual axis moving slowly. The largest
constraint-free transition demand is right-hip lateral at .56 s:
−316.789 against ±30, with a probe acceleration of −386.732 rad/s².

Piecewise-linear targets have discontinuous velocity. The 20 ms probes use the
average adjacent velocity and velocity difference divided by the control
interval, at the unchanged knot position. These are diagnostics, not an
executed smooth trajectory. The reported mass-matrix impulse and motor-only
impulse time freeze configuration and omit other forces; neither is a minimum
transition-time certificate. No terminal braking beyond the source is invented.

### Inputs, execution and retained failure

Completed native plan: `.local/aspset-next-dynamics-02/plan.json`; input
`artifacts/aspset-next-shared-01/lab/run`, seal
`d998da1621c5d70a935d5492f2b07df326fc8e25d69194d25e56f06b2f94691c`.
Output: `artifacts/aspset-next-reference-dynamics-02`, **54 members**, seal
`ac3490fb6a8c481af4a7a80165deaeed41fc565d23aa756ba8ff5896caa884ca`.
Frozen runtime reuses the previously verified source runtime, with the new
diagnostic added; source copies/hashes are in the control directory.

Bounds: 30 wall/20 CPU seconds, 512 MiB RSS, 32 MiB/512 output files,
24 model compilations, 32 data allocations, 512 forward calls, 100 inverse calls,
**zero physics steps, inference and paid calls**. Complete source samples:
46 interval midpoints, 45 interior velocity changes and one reset transition.
The explicit transition-probe window is the unchanged 20 ms control interval.

Actual worker: **.568814 s /117,882,880 B peak RSS**, three model compilations,
two data allocations and 92 inverse calls. Supervisor .623915 s, exit 0,
PID 89069 reaped; 55 files /2,020,826 B including the seal. No live root worker.
The direct diagnostic section is .318588 s; these are one local run's timings,
not a sustained interactive-performance claim. MuJoCo 3.17.0.

Attempt01 (`.local/aspset-next-dynamics-01`) stopped with zero inverse calls,
zero steps and zero output after three compilations: its overly strict guard
required the runtime cap to equal the engine clamp. The root runtime cap is
850.517984, inside the unchanged ±2000 engine clamp. The corrected guard
accepts a stricter runtime cap, retains it for every ratio and rejects a lower
engine/joint clamp. Three additional analytical cases cover this distinction.
Attempt01's logs/result remain; no existing output or source was overwritten.

### Verification and implementation

**41 distinct tests pass:** 30 kernel and 11 source/report boundary cases.
The first test run exposed a NumPy/native-enum membership comparison; explicit
integer enum comparison fixes it. Its failed receipt is retained. Current
qualification reuses unchanged input tests and reruns changed native guards:

- `.local/reference-dynamics-tests-02.xml` (27 initial passing cases).
- `.local/reference-dynamics-native-guards-03.xml` (15 affected/new cases).
- `.local/reference-dynamics-study-tests-01.xml` (11 input/report cases).
- `.local/reference-dynamics-report-delta-02.xml` (one changed report case).

Tests cover analytic mass/damping/gravity, inverse-to-forward acceleration,
contact/constraint separation, axis permutation, live-state/model preservation,
malformed arrays, unsupported actuation, tighter runtime caps and source seals.
The module allocates separate native state and never advances physics. The
existing controller, independent scorer, evaluator and all task gates are untouched.

Independent saved-only verifier `.local/aspset-next-dynamics-02/verify_saved.py`
does not import the new kernel or simulate. It checks all **448 source/output
members**, derives the targets directly from the sealed scene, and independently
recalculates 92 time/velocity/acceleration/force decompositions, **2,024 axis
samples**, cap ratios, impulses and summaries. Maximum force-balance residual:
7.276e−12; actual .071809 s. Receipt: `saved-verification.json` beside the script.

Force accounting follows MuJoCo's [inverse-dynamics definition](https://mujoco.readthedocs.io/en/stable/computation/index.html#inverse-dynamics):
required applied force includes inertia and bias, minus passive and constraint
forces. The native model supplies those terms; no gain or force is fitted.

Owned implementation: `src/weft/reference_dynamics.py`,
`scripts/studies/audit_camera_reference_dynamics.py`,
`tests/test_reference_dynamics.py`, `tests/test_reference_dynamics_study.py`.
No default UI/control mode or saved lab is replaced.

### Inspect or reproduce

[Readable report](../../artifacts/aspset-next-dynamics-report-01/index.html),
seven members, seal
`014aa5ac847a878377b789b3211d2f2efee679be133d5b9edb5d685479531272`.
It reuses the exact native results and adds an explicit constraint-free maximum
column, without native recomputation. Two tables/17 rows and all three local
links checked; no browser interaction or visual QA.

For a new diagnostic output (do not repeat the unchanged completed study):

```sh
.venv-lab/bin/python -B scripts/studies/audit_camera_reference_dynamics.py \
  --run artifacts/aspset-next-shared-01/lab/run \
  --out artifacts/my-reference-dynamics
```

The previous playable lab remains unchanged; its command is in
[the different-motion handoff](aspset-next-clip.md). No learned policy is added
to that clip by this diagnostic.

### Next safe action and Git ownership

Prepare one bounded temporal reconstruction proposal using only original
source-image constraints and uncertainty. Preserve raw poses, clocks, dimensions,
ground and original source/evaluation targets. Compare image residuals and force
demands before admitting it, then test full-interval/changed-start control against
the unchanged original task as well as reporting candidate tracking separately.
Do not replace independent assessment with a smoother self-target, raise force
limits, shorten/retime the source, or fit the published C3D reference.

Ground/equipment contact, unfamiliar subjects/cameras, meaningful learning and
independent physical accuracy remain required; the complete goal stays active.
External cost is $0; no API/key/ledger access, acquisition, server, browser or
deployment. Both attempts are terminal/reaped. Preserve parallel avatar edits.
Base at preparation was `6ae2397`; publication uses exact owned paths, refreshes
remote state and records the verified result in
`.local/aspset-next-dynamics-publication-01.json`. Stashes were empty.
