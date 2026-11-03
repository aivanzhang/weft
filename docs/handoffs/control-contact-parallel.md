# Parallel controller and contact work

## Parallel checkpoint — November 3, 2026 (PST)

Three non-overlapping agents implemented and checked the lanes below against
the same saved sources. The integrator owns review, end-to-end verification,
documentation and publication. No branch,
deployment, new provider request or default-controller change is planned. Keep
the separate avatar files and `tools/avatar-assets/` changes untouched.

| Lane | Owned files | Required evidence |
| --- | --- | --- |
| Controller timing | `moving_equipment_preview_control.py`, its tests and local study | One fixed-weight counterfactual, original task and native duration unchanged; reuse the saved baseline |
| Two-arm contact | `moving_equipment_bilateral_contact.py`, `contact_force_balance.py`, their tests and local study | Complete original terminal geometry/force constraints, every refusal retained; static feasibility is not dynamic task success |
| Pose continuity | `spatial_twist_continuity.py`, its tests and local study | Change only bounded interior zero-confidence twist estimates; preserve source slots, clocks, observed anchors and uncertainty |

## Verified starting evidence

The diagnostic in `.local/moving-body-control-diagnosis-01/result.json` reproduces
the saved failed controller metrics exactly: 457 controls /7,312 native substeps,
11.594 seconds, 861,765,632 B peak RSS. All source/runtime/policy/provider-budget
checks pass. `moving_body_control_diagnostic.py` has 11 focused checks and
independent review. No training, API call or source inference was repeated.

The maximum angular error is 1.06335 rad at 15.033333 simulated seconds. The
three left-shoulder axes contribute 89.27% of squared error and saturate their
original motor commands. There is no immediate human contact at that solve.
At the preceding control boundary, a source-reference velocity changes sign
0.333 ms later, but its old command is held for the full 33.333 ms action.
This motivates exact-interval-average velocity feedback, not higher forces or
an easier scorer. The saved trace does not alone establish causality.

Source pose ambiguity at 4.033 seconds is separate: near-straight limbs lose
observable twist, and the current fallback can jump on re-entry. The optional
continuity proposal must not claim to fix the later global peak, whose source
twist estimates have nonzero confidence.

## Source custody and bounds

- Controller source: `.local/portal-task-source-01/portal/cbc6e2600f8e4058abab4a220de38df6/lab`.
  Manifest `378004dcd1fbbaebcae00c80b695589cb960db8ed86c3fda1135fa6795c00e5d`.
  Frozen policy `e89f4bebc4d423ba9e1c8cef53b6d4412bd943cacd52dae8c4572f71dcf5c81f`.
- Contact source: `.local/moving-placement-source-acceptance-01/task-child`.
  Manifest `a64fd1a63bf3aca3dddfe9aeda818c4525a1c281f43e6113c5d8f1a08c0c2021`.
  This is the original hover task, not the separately authored floor target.
- Each actual trial uses a new directory, a closed plan, original hashes,
  bounded runtime/memory/output and explicit terminal results. Never rerun an
  unchanged failed search or execute archived source code.
- Shared provider totals remain last verified at Meta $0.466027/$5 and OpenAI
  $0.007886/$5, zero unsettled. These lanes require no paid calls.

The different single-leg source retains 79 references, 77 partially tracked,
with the right arm mostly uncertain. Its saved ground screen has zero usable
stationary planes. `.local/singleleg-world-preparation-01/HANDOFF.md` records
the exact construction path and missing scope choice. Do not convert its
unknown inventory into a fabricated user declaration or move its floor to
make construction pass.

## Completed source checks

**Controller timing:** `.local/moving-preview-control-01/result.json` completes
one 457-control /7,312-substep trial in 11.707 seconds /861,913,088 B. It reuses
the exact saved baseline without rerunning it. Original source, policy, task,
runtime and budgets remain unchanged. The only altered feature is human/root
target velocity averaged over the exact next action; position targets, gains,
bias, object-feedback formula, force limits and scorer are unchanged.

| Original-task metric | Saved baseline | Interval-preview candidate |
| --- | ---: | ---: |
| Angular RMS error (rad) | 0.325057 | 0.268086 |
| Peak angular error (rad; limit 0.70) | 1.063347 | 0.978511 |
| Root RMS error (model units) | 0.024761 | 0.017887 |
| Maximum floor penetration (model units) | 0.009146 | 0.004247 |
| Final object distance (world units) | 0.173219 | 0.012527 |
| Final object speed (world units/s) | 0.182968 | 0.022554 |
| Terminal placement hold (seconds) | 0 | 2.405917 |
| Mean reward | 0.583336 | 0.495295 |
| Full task success | No | No |

The candidate clears the terminal placement requirement but still fails peak
angular error. Saved-only scorer decomposition explains the reward regression:
mean tracking reward improves 0.596405→0.677523, while placement reward over
the **whole episode** falls 0.570267→0.313067. The object finishes better but
spends less earlier time close to the goal and moving slowly. Exact reward
reconstruction differs by less than 2.3e-14; no scorer change is warranted.
The new peak occurs at 10.166667 seconds, dominated by left-hip twist and torso
flexion, not the old late shoulder peak. Angular-limit exceedances fall from
301 to 188 native samples, but saturated angular-axis samples rise from 22,144
to 33,104. This is a tradeoff, not an unqualified improvement.
No changed-start evaluation, gain search, learned-policy claim or installation
has followed this mixed result. This uses privileged simulated reference
preview, not physical observation of future intent.

**Pose continuity:** `.local/spatial-twist-source-01/` retains all 442 references
and 458 source slots. One uncertain left-shoulder reference at 4.033 simulated
seconds changes; the two observed anchors, hinge bend, confidence and clocks
remain exact. Peak Euler-coordinate speed over that short span falls from
101.142 to 5.297 rad/s. Distal unit direction changes by 0.04130, within the
explicit 0.06 hypothesis bound; this is not exact distal-position preservation.
The pure proposal takes 0.209 seconds /101,449,728 B, with no native model,
physics, inference or API call. No derived body has been admitted or installed.

**Two-arm contact:** `.local/moving-bilateral-contact-source-01/` completes in
11.980 seconds /862,404,608 B. It retains all 17 terminal knots and 242 native
boundaries, uses one geometry model /4,114 forward evaluations and finds no
valid complete-window contact configuration. Only 9/242 audited poses pass all
geometry checks. Remaining support gaps affect 208 left and 216 right samples;
two right-forearm overlaps exceed the unchanged tolerance. All floor, joint,
reference, correction and containment checks pass, but seven knots reach the
0.5-rad correction bound. No force-balance calls occur because geometry fails:
the explicit 10-unit per-forearm force hypothesis is **untested**, not disproved.
All 114 source-file hashes, runtime and provider ledgers remain unchanged.
This is a bounded negative result, not global impossibility; no physics steps
or dynamic grasp were tested. Do not repeat the same local search unchanged.

## Verification and publication

There are **117 distinct focused passing checks**: 32 force-balance, 23 bilateral
geometry, 11 unchanged-controller diagnostic, 32 twist-continuity and 19 preview
controller cases. Reused final passing XML records cover 92 cases; the initial
25 twist cases retain their earlier root tool output. Do not synthesize XML for
those historical checks or rerun passing work merely to change its format.

The root saved-only audit `.local/control-contact-checkpoint-01/verification.json`
verifies all ten current source/test release pins and 19 result/trace members.
It makes no simulation, inference or API calls. Its first attempt used a wrong
contact-report filename; the corrected path verifies the unchanged report.
Initial test-only and numeric-input failures remain in their lane receipts.

Remote refs were refreshed before publication; `main` and `origin/main` matched
`92e0458be00b5d1617583287de50de1d62d21c7a`, with no stashes. Publication outcome
will be recorded in `.local/control-contact-checkpoint-01/publication.json`.
Only the ten released code/test files and this checkpoint's documentation are
owned here. Concurrent avatar changes are intentionally excluded.

## Completion gate

Check each lane's terminal report before publication. Preserve failures and
partial results. A promising controller requires matched changed-start
evaluation before any default promotion. A contact witness requires separate
dynamic acquisition/retention tests. Source continuity requires a separate
derived-body custody path before product use. None alone completes unfamiliar
video reconstruction, physical accuracy or meaningful general RL learning.

Next, address the remaining source-reference discontinuity near 10.167 seconds
and the controller's whole-episode placement tradeoff. Keep the original
targets/scorer for comparison; the isolated zero-confidence shoulder proposal
does not explain that high-confidence torso/leg event. Do not repeat this same
controller or tune contact-force caps while geometry remains invalid.

Current playable portal instructions remain in [the portal guide](../portal.md).
The user starts the server; this milestone does not authorize hosted deployment.
