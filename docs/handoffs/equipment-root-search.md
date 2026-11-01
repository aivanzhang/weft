# Nonlinear root-assistance search

## Current state and ownership

Implemented an **experimental, non-default** measured search in
`src/weft/shared_exercise_root_search.py`. The first guarded run passes **50
focused mocked cases**: 0.652983 seconds, 118,259,712-byte peak RSS, zero native
or forbidden operations. One import-time OS probe was blocked before execution.
Independent source/test review is closed with no blocking findings; the reviewer
read the receipt and cases without duplicating execution.
No native qualification, full-task trial or new holdout query has run.

This lane owns only the new module, `tests/test_shared_exercise_root_search.py`,
this handoff and `.local/equipment-root-search-01/`. Root owns native execution,
Git and integration. No live handles; $0 external or recurring spend. Existing
predictors, projectors, scorer, policies and public defaults remain unchanged.

## Why this hypothesis is different

The [prior two-action search](equipment-sequence-control-native.md) abstained:
all 23 pairs failed measured constraints. Its affine model predicted negative
equipment speed/spin for a candidate whose actual forecast was positive and
failed. It also reused exactly the old 23 first-action joint vectors, kept roots
fixed and could not change the second action. More scales or solver sweeps are
not a demonstrated remedy.

V5 learns ten shared joint-feedback coefficients and preserves authored roots;
the previous clearance correction also preserves roots. Original v1 could learn
root-feedback gains, so root actuation is **not newly introduced authority**.
The distinct hypothesis here is current-state nonlinear polling of legal root
assistance at **both** forecast boundaries, with no fitted response model or
training. The task already permits these three channels. Sparse polls do not
establish that a feasible root correction exists or does not exist.

## API and fixed bounds

```python
from weft.shared_exercise_root_search import prepare_root_assistance_search

search = prepare_root_assistance_search(env)
actions, diagnostic = search.search((first_base, second_base))
```

`RootSearchConfig` defaults and hard ceilings are 26 sequences, 52 forecast
controls and five active seconds; the root residual/poll step is fixed at 0.25.
One helper permits one decision, uses one sequence-predictor lifetime and at most
one scratch data object. The whole report is bounded to 2 MiB. No policy callback,
solver, training, live step or hidden horizon shortening is allowed.

- Input is an explicit, copied list/tuple/2-D array of **one or two** finite
  float32/float64 action vectors. Different dtypes/bases remain separate.
- Only channels 19–21 (`root_x/y/z`) may change. Each action retains its exact 19
  joint bytes and dtype. Every root stays in its **original supplied base ±0.25**
  intersected with `[-1, 1]`, including inward representable rounding.
- Existing physical motor limits are unchanged; no equipment force is added.
  The saved task's per-axis root limit is 773.578233 in its prior force units;
  0.25 corresponds to at most 193.394558 of those units, not calibrated anatomy
  or a new measured force claim.
- Evaluate the baseline, then each legal ±0.25 root-coordinate poll across all
  supplied controls (at most 12). A second poll is centered only on a strictly
  improved **measured** maximum/mean normalized violation. Both polls remain in
  the original box. Deduplicate unchanged/clipped action bytes before prediction.
- Rank actual measured maximum positive normalized excess, then mean excess,
  then squared root change, retaining encounter order for exact ties. Original
  `limit + 1e-12` pass tolerance is unchanged; passing candidates therefore rank
  by minimum change. The best full prediction is retained, not every large proof.
- Reserve the complete next unique poll **and** a repeat before that poll starts.
  Baseline +12 +12 +repeat is at most 26 sequences /52 controls /832 scratch
  substeps. Failed attempts consume reservations; unknown native counters remain
  explicitly unknown. No budget reset or extra predictor is used.
- Return a candidate only after a full sequence repeat matches measured values
  and every control row's endpoint, contacts, actions and clocks—not hashes from
  different encoding domains. Even an unchanged baseline must repeat.

Every measured candidate, rejection, rank and skipped poll is reported. A failed
forecast retains its attempted actions, counts and available partial rows. If a
full accepted/partial proof would exceed the total report cap, the helper
abstains and explicitly records omitted proof counts; measured evaluations and
the failed-attempt record remain. Live-state/task changes are reported, never
silently restored. Deadline accounting includes measurement and report encoding.

## Whole-task use and honest gate scope

The helper contains no step49 special case. A later caller can recompute the
current v5 base each control, supply `(base, base)` as an explicitly held-base
two-control assumption, apply **only the returned first action**, then replan.
For the final remaining control it supplies exactly one action. This is not a
forecast of recomputed v5 feedback at the second boundary. A caller must enforce
an independent aggregate episode work/time/output budget before preparing each
new one-decision helper; a full rollout is not authorized by these pure checks.

The frozen measurement context checks original endpoint equipment/tracking
limits and sampled floor contacts. It does **not** predict the scorer's accumulated
substep equipment peaks, contact-duration criteria or final mean tracking gates.
Actual control50 previously showed larger accumulated speed/spin than its
endpoint, so passing this screen is not task-gate safety.

Only actual `env.step` and unchanged task metrics can establish progress. Stop
on engine/metric/nonrecoverable gate failure; keep incomplete-interval mean
tracking crossings provisional. Full success still requires all 254 controls
and original `completed_interval` plus `success`. No equipment result, gate,
source/evaluation partition or public route changes here.

## Verification and next safe action

Focused tests cover a discontinuous response, both-action correction, actual
max/mean/minimum-change ordering, copied joint/dtype identity, original-box
clipping/deduplication, complete-repeat mismatch at either boundary, full-poll
admission, partial/unknown counts, horizon refusal and explicit final one-control
handling, environment/task changes, deadline and report-cap abstention.
The frozen dependency tests were not rerun.
One bounded hash comparison confirms all 28 previously frozen dependencies plus
the prior sequence projector remain exact: **29 unchanged files**. Receipt
`unchanged-sources-01.json` has SHA256
`3774d5cfe65cf917df7618dad7cff893a51e789a5ed1f02ccabf93799aec27ca`.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-lab/bin/python \
  .local/equipment-root-search-01/run_checks.py \
  .local/equipment-root-search-01/tests-01.xml \
  .local/equipment-root-search-01/process-01.json \
  tests/test_shared_exercise_root_search.py
```

The output paths above already exist; do not rerun unchanged checks.

| Evidence | SHA256 |
| --- | --- |
| New module | `f4cedb0c8cbc160542c9a1f11a5d5c07041a57015d745a0cbdd07e434592bdb7` |
| New tests | `06faa4c9eea7e2e1cf0120af5cf0ddca549f738cc762aef5726def619432c35d` |
| `process-01.json` | `491836e72e68ac87ae802383f356a2ab5148266520c07b4c1ee38bc4bafde019` |

Next: prepare one bounded native decision for root's review/execution before any
full-task validation. The held-base second forecast must remain
separate from later replanned actions. A two-action qualification with the saved
48-control prefix would cap at 11 models, two data objects, 104 forwards, one
reset, 26 copies and 1,632 substeps (768 prefix +832 forecast +at most32 actual).
Those are a **proposed** qualification envelope, not executed work or permission
for another episode. No change to the $5 shared Meta ledger is needed.
