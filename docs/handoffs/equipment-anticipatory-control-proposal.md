# Two-control anticipatory correction

## Decision and current scope

Root approved implementation of **one experimental two-control projector**, not
another control-50 finite-difference scale sweep. At the exact state before
control 49, vary only the first action's 19 joint commands and forecast a supplied,
unchanged second action. Implementation and independent review are complete;
**37 guarded mocked checks pass**. No native search has run. Owned files are new
`src/weft/shared_exercise_sequence_projection.py` and
`tests/test_shared_exercise_sequence_projection.py`; scratch is
`.local/equipment-anticipatory-control-proposal-01/`. Existing predictor,
projector, policy, task and scorer files remain frozen; no public/default route.
Root owns Git and any separately frozen native execution. External cost is $0.

## Why this changes the experiment

The saved one-control search at 49 moved the equipment's endpoint speed/spin from
0.392155 / 3.642698 to approximately zero, within the original limits 0.15 / 0.75.
Its largest joint change already reached 0.25. That successful immediate correction
did not avoid the next control's failure; repeated pre-50 local searches abstained.

Root's completed two-control predictor qualification now forecasts
`[saved selected49, saved base50]` from before 49. Both full endpoints and every
substep contact match the isolated actual 49/50 steps exactly. Before taking 49,
the forecast identifies 50's endpoint speed **0.4798370892 > 0.15** and spin
**2.867588606 > 0.75**. Actual control 50 fails `equipment_stability:sam-2`, with
larger accumulated peaks **0.5976082792 / 3.516482254**.

This supports testing an earlier action with a longer prediction horizon. It does
not establish that a feasible correction exists. The accumulated peaks also prove
that endpoint constraints are **not complete task-gate coverage**.

## One fixed search contract

Implemented API:

```python
from weft.shared_exercise_sequence_projection import (
    SequenceProjectionConfig, prepare_action_sequence_projector,
)

projector = prepare_action_sequence_projector(env, config=SequenceProjectionConfig())
pair, diagnostic = projector.project(first_base_action, second_fixed_action)
```

- Exactly two explicitly supplied float32/float64 actions. Copy both at admission.
  First-action changes affect only joints 0–18, inside original base ±0.25 and
  [-1, 1]. Preserve original first-action root bytes and dtype. Preserve the entire
  second action's bytes and dtype. Return a tuple of separate arrays, not a stacked
  array that could coerce either dtype.
- The saved second action is a **fixed development input**, previously produced
  by v5 after the old selected49. After changing 49 it is not a recomputed v5
  feedback action. No callback, future-policy inference or closed-loop claim.
- Reuse one frozen `ActionProjector` measurement context without calling its
  `.project()` or one-step predictor. Freeze reference targets once at `t+dt` and
  `t+2dt`; measure each sequence row at its own validated initial time using the
  corresponding target. Keep reset-relative equipment coordinates, every limit,
  sampling label and the existing `1e-12` acceptance tolerance unchanged.
- Stack the existing 12 measurements at each boundary into 24 separate rows for
  this two-equipment case. Do not collapse them into a nonsmooth across-time max.
  General count remains twice `(4 × equipment_count + 4)`, under the existing
  maximum of 12 equipment objects. This is not a new scorer.
- One fixed **0.1** finite-difference model, all 19 first-action coordinates,
  direction toward zero as in the existing single-round search. Record the actual
  representable difference. Preserve the original-base trust box, root identity
  and inward float rounding through `_bounded_trial`.
- Reuse the existing normalized linear solver with at most 64 aggregate sweeps;
  its output is only a bounded proposal. Test scales 1, 1/2 and 1/4 with actual
  nonlinear sequence predictions. Keep any passing finite-difference candidate
  if the solve produces no candidate. Choose the smallest measured passing joint
  correction, retaining encounter order for ties.
- Repeat the complete selected pair from the unchanged live state, comparing both
  full endpoints and all substep contacts before returning it. Repeat even an
  unchanged base pair. If no measured candidate passes, or any prediction, clock,
  identity, budget or repeat check fails, return no action pair and preserve the
  reason. No fallback, second search round or retries.

## Bounds and evidence

One lifetime decision; a single five-second active budget includes validation,
prediction, solving and final reporting. A single sequence predictor owns one
scratch data object and performs one live-state copy per two-control sequence.

| Work | Maximum |
| --- | ---: |
| Base + 19 differences + 3 scales + selected repeat | 24 sequences |
| Reserved forecast controls | 48 |
| Forecast substeps (16/control) | 768 |
| Forecast boundary forwards / copies / scratch objects | 48 / 24 / 1 |
| Solver sweeps | 64 |
| Whole returned diagnostic, including accepted proof | 2 MiB |

Retain all evaluated actions, both constraint vectors, rejected outcomes, response
errors and attempted/returned native counters. Keep only the needed winning full
sequence while searching; diagnostics retain hashes of other endpoint/contact
proofs. Unknown native counts, partial forecasts or omitted required proof cannot
be accepted. A report-budget failure returns no pair, with explicit bounded
failure evidence. Caller serialization and the existing live/task identity guards
remain required.

## What may be claimed and how actual metrics are checked

A returned pair is a **repeated detached proposal that passes the measured
boundary/floor constraints**. It is not a gate-safe prefix, predicted task success
or validated controller. The sequence report has substep contacts but no equipment
velocity/pose or tracking values at every substep, so those accumulated task peaks
cannot be certified from it. Do not replace them with endpoint maxima.

Any later native pilot must reconstruct the same 48 saved controls once, then:

1. Run the bounded search from that fixed state. If it abstains, take no action.
2. If it returns a pair, apply the first fixed action once in the isolated
   validation environment. Compare the entire actual endpoint and substep contacts
   with its prediction; separately save the unchanged scorer's accumulated metrics.
3. Stop on parity failure, engine failure or a failed monotonic protected gate.
   Record finite prefix mean-angular/root crossings as provisional under the
   already-reviewed Rollout02 semantics; do not rewrite raw `failed_gates`.
4. Only if the first step remains eligible, apply the unchanged second action once
   and make the same comparisons. Do not replan or substitute a different action.

Full pilot upper bound, including initialization used in the existing qualification:
48 prefix + two accepted live controls and 48 forecast controls = **1,568 substeps**;
11 models, two data objects, 100 forwards, one reset, 24 copies. Proposed outer
envelope stays 40 CPU / 60 wall seconds, 512 MiB RSS and 4 MiB output; compact
diagnostic/proof plus result/manifest must reserve space before starting. Root must
review and freeze the actual plan; this note is not execution approval. A two-step
acceptance is still an incomplete 254-control task, and any original gate failure
must remain visible.

## Evidence and implementation checks

Bounded saved-only reads, no new geometry or simulation:

- Saved 49 projection `artifacts/equipment-action-controller-native-02/projection.json`,
  SHA `26902044ed86759550b1ba21d2c11ca0033dfaabab2e2d3097f4c795b36f4696`.
  Relevant values/constraints are retained in scratch `saved-evidence-01.json`.
- Completed sequence qualification plan
  `.local/equipment-sequence-prediction-native-01/plan-frozen-02.json`,
  SHA `fa9a96faf625ffc91b771c6602e212298330f649b4a27186b61c1536108511eb`;
  runtime inventory SHA
  `a71c264d135b0dce3af67301fad87151413c99c749c5e79722d8ad32818cf5e1`.
- Root verified the completed native output once: seal
  `60a9abdff611d2b1728c7ff3d12a3c48446e0dd4d07e4ab4406692765eb3fd53`;
  saved audit `956eac8fde3e6d95f2066581d8fcafc6ead258c85915efc3d9af1b230596024a`.
  Endpoint/accumulated numbers above are root's reported values, not another replay.

## Implementation release

All 37 focused mocked checks pass: 0.615591 seconds, 118,095,872-byte peak RSS,
zero native attempts or executed network/child processes. The reused guard blocked
one expected import-time platform probe. Coverage includes distinct frozen target
times; both-boundary rejection; fixed-second/root/differing-dtype bytes; whole-pair
counts and mandatory repeat; a retained passing probe after solver conflict;
partial/unknown counters; near-cap proof omission retaining all 24 measurements;
shared time/solver/prediction budgets; and unchanged live/task state. These tests
exercise the real measurement context with authored sequence forecasts, not native
physics. Existing passing suites were not rerun.

- Source SHA256:
  `083e629aec3eebc9a22098dfd9c3e8af1b8d1f3e70a36c2710a6c5eb913432ef`.
- Tests SHA256:
  `6347149e95abe276ff7f2d76380880098782c26337f25e688c88bf5e1c4517a9`.
- Scratch `process-01.json`:
  `13b74199ec340cbc7306cbd14ce74a425ad01904f0424c194f1753f6be23c96e`.
- All 28 old source files remain byte-identical: 25 policy/task files, the old
  one-action predictor/projector and the new qualified sequence predictor.
  `unchanged-source-after-01.json`:
  `ec6eebb3f781982b58b032c0298e248397b89ca053d06dd4dcda9c754fce2974`.

Independent review is closed on these exact pins; the reviewer inspected the
source, tests and receipt without duplicate execution. No remaining finding.
Source/tests are frozen. The next safe action is root review and freeze of one
bounded native search at pre49, followed by at most the two verified actual
actions under the rules above. No native process is owned by this lane.

Completed command, preserved for reproduction rather than an unchanged rerun:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin TZ=America/Los_Angeles \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=disable \
  .venv-lab/bin/python -B -P .local/equipment-anticipatory-control-proposal-01/run_checks.py \
  .local/equipment-anticipatory-control-proposal-01/tests-01.xml \
  .local/equipment-anticipatory-control-proposal-01/process-01.json \
  tests/test_shared_exercise_sequence_projection.py
```

No full-episode search, parameter sweep, paid call, Git mutation or source/scorer
change is part of this lane. Endpoint/accumulated-gate limitations remain explicit.
