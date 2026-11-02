# Local experiments beside Muse review

## Completed — November 2, 2026 (PST)

The portal runs fixed, bounded control comparisons beside a completed Muse review.
This is not arbitrary model-written test execution or automatic repair/acceptance.
The saved review requests real calibration/contact observations; those remain
unresolved and are not replaced by simulator agreement.

## Implementation

- `portal_checks.py` verifies construction/advice/provider-response bindings,
  source identity and scene/task manifest hashes. Changed or linked inputs reject.
- Exercise: unchanged `SharedExerciseTaskEnv` and gates; identical seeded resets
  for authored control, root channels zero, and all controls zero. Already-unassisted
  tasks skip the redundant root condition. Rigid: existing `SpatialRigidEnv`,
  matched coast / positive-x / negative-x force interventions.
- At most three trials / 900 control steps each; 45-second cooperative execution,
  60-second supervised wall bound, sampled 1.5 GiB RSS guard and 32 MiB traces.
  Shares the portal's single worker; at most four queued check jobs. No API call.
- `POST /api/projects/<id>/muse/checks`: authenticated, bodyless, deduplicated;
  requires a ready simulation and proposed review, not a provider key. Cancellation
  retains partial traces; interruption does not requeue. Original terminal states
  remain unchanged.
- `muse/checks/`: input bindings, complete action/state/reward JSONL traces,
  report and manifest; lifecycle metadata remains outside the seal. Reports label
  exposed development simulation, `accepted:false`, `applied:false`, and physical
  hypotheses unresolved. Current reports include runtime/package fingerprints.
- UI: **Run local comparisons · no API charge**, status, original-task success,
  steps/reward, report and downloadable traces. No policy selection, repair,
  source rewrite, scorer change, browser automation or deployment.

## Actual acceptance

`.local/portal-checks-01/` contains the bounded plan, HTTP harness, frozen runtime,
`result.json`, `replay-verification.json` and `audit.json`. Do not rerun it: the
saved project already has completed check results and the harness refuses duplicates.

Project `4e0f1fe859664bb5ba788214e129451d`; source
`ab4602823ad00de3409eb50d2c65c6b1ce5566c5dca6753555ff2b19107db88e`.

| Fixed condition | Steps | Original task success | Angular RMSE (rad) | Root RMSE (model units) |
| --- | ---: | --- | ---: | ---: |
| Authored reference controller | 234/234 | Yes | 0.112517 | 0.020017 |
| Same controller, root forces zero | 234/234 | No | 0.318212 | 0.505983 |
| All controls zero | 234/234 | No | 1.386819 | 0.369076 |

This exposes dependence on virtual root assistance in this authored model, not
real balance, contact, loads or muscle forces. No policy was trained or selected.

- HTTP acceptance **2.237585 s**; worker **1.467167 s / 119,717,888 B peak RSS**.
  Nine HTTP requests: mean 1.876 ms / max 4.191 ms. Not sustained-load proof.
- **702 actions replay exactly**, including every reward and full state; three
  identical initial-state hashes. Traces total 15,649,649 bytes.
- All **1,325 original run members** reverified; manifest unchanged:
  `d8cfe07cc44e3d36efafcd6039c49435188e684c763e28fb9ce11e5cc8d1fe26`.
  Ready/proposed statuses and advice hashes unchanged. Both API budgets unchanged.
- Duplicate HTTP submission made no second run. Worker exited 0 and was reaped;
  temporary HTTP server shut down. No live acceptance job remains.
- **22 distinct Python / four deterministic UI checks pass**: actual dynamics and
  replay, task/gates, tampering/links, bounds, cancellation/restart, deduplication,
  auth/origin and rendering. A sandbox bind denial in `portal-checks-binding-01.xml`
  was resolved by narrow loopback escalation in `portal-checks-http-01.xml`.
- After acceptance, code review corrected rigid-direction labels, rejected linked
  lab paths and added runtime metadata. Focused tests cover those changes; exercise
  dynamics were not changed or rerun. The original frozen report is untouched;
  the audit records observed package versions. Object checks have synthetic test
  coverage, not a real-video Muse/portal acceptance claim.

## Continue

Implement typed, bounded repair candidates and independent acceptance, plus
source-qualified equipment/missing-input assistance. Keep real measurement
requests unresolved until supplied. The full goal remains active.

Use `.venv-lab/bin/python -m weft portal --port 0`; the saved exercise review and
comparisons are available without another charge or experiment. Preserve concurrent
avatar work and publish only portal-owned paths. Hosted site remains offline.
