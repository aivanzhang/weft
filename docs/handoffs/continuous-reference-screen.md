# Continuous source-reference screen

## Rejected hypothesis — November 2, 2026, PST

Previous turn made verified progress at `3c666c7`: optional motion-prior
construction, original-task comparisons and exact recorded replay. Both tracks
and the full goal stay active. Root owns this bounded source-only diagnostic;
preserve separate avatar edits and all existing labs/scored runs.

The source search checked two public archive metadata pages. The directory
still offers large compressed archives, not individual videos; the archive
member viewer returned 503. No new subject/media, test-partition data or reference
coordinates were acquired. Details: `.local/aspset-cross-subject-metadata-01`.
No unbounded archive download or unchanged request retry is planned.

Independent hypothesis: straight-line source-pose interpolation makes velocity
jump at every knot. Test scalar monotone cubic Hermite interpolation with
continuous velocity, unchanged source poses/times and zero starting velocity.
Terminal braking is not inferred. No angle unwrapping, source refit, floor shift,
gain/force changes, scorer changes, learning, API or physics steps.

The implementation is study-only in `.local/continuous-reference-screen-01`.
Known-input checks cover 2/3/47/900 poses ×22 axes, exact knots/continuous
velocities within roundoff, endpoint-range preservation, analytic derivatives
against finite differences, constants, gap rejection and no extrapolation.
They pass in .041 s /43,368,448 B; no product runtime has changed.

Compared actual continuous inverse dynamics at five points per interval
for both frozen original and motion-prior bodies. Reuse saved linear midpoint
diagnostics, keeping common constraint-free subsets explicit. Endpoint
acceleration can jump; C1 does not mean continuous force or physical validity.
Admission requires every sampled force within unchanged caps and the
unchanged .025 floor tolerance, not merely lower average error.

Frozen bounds: one local worker, 30 s wall/25 s CPU/384 MiB sampled RSS,
eight model compilations/eight data allocations/1,000 forwards/500 inverse
calls/zero physics steps, 32 MiB/16 output files. Two sealed source packages,
460 inverse calls completed in **.721 s /92,438,528 B**: six model compilations,
four data allocations and zero forward calls/physics steps. PID 99728 is
terminal/reaped, exit 0. External/recurring cost $0. Do not repeat this hypothesis
or install it as a controller/default.

| Same source targets | Original image-only poses | Motion-prior poses |
| --- | ---: | ---: |
| Continuous samples exceeding a motor cap | 203/230 | 182/230 |
| Constraint-free samples exceeding a cap | 191/218 | 166/214 |
| Largest constraint-free force/limit | 10.914× | 10.011× |
| Common constraint-free midpoints improved | 0/44 | 1/44 |
| RMS of largest per-midpoint force/limit, linear → continuous | 1.463→2.990 | .897→1.661 |

Velocity continuity alone is not a repair: it redistributes acceleration while
retaining problematic source poses. The continuous path has one-sided finite
acceleration at each endpoint; these are not the old linear path's 20 ms
velocity-jump probes. The largest contact-constrained force ratios reach
1,385×/2,604× and remain conditional model responses, not physical forces.
Maximum sampled floor penetrations stay .020439/.018524 model units, below the
unchanged .025 tolerance; that does not qualify contact or force feasibility.

The derivative rule follows the weighted harmonic-mean construction described
in the [SciPy PCHIP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html),
with an explicitly different zero starting velocity. No SciPy installation,
reference-coordinate fit, angle unwrapping or terminal braking was used.

## Integrity and preserved correction

- `artifacts/continuous-reference-screen-01`: seven sealed members,
  manifest SHA `b3572a05cd129c8ec1d7597cff1d9e2e611213ba9be348c48295efda46cc73ce`.
  Retains both complete sample tables, fixed plan, worker and interpolator.
- `artifacts/continuous-reference-audit-01`: two sealed members,
  manifest SHA `ae0297743bcf331a7206921f055aa1265e0d48ecf5c2efbf541ae1b211497aa2`.
  Saved-only arithmetic verifies 131 members, all 460 samples/10,120 scalar
  force balances, exact source poses, continuous velocities and paired subsets
  in .090 s, without new native calls.
- Use the audit's `corrected-report.json`: the screen summary incorrectly
  defaulted the **absent original-linear floor-distance field** to zero. It is
  now explicitly unavailable in the new companion, not a measured zero.
  Original artifacts remain unchanged; force values/conclusions are unaffected.

The original runs, working controls, original task and scorers remain unchanged.
Remaining work needs source-qualified geometry or force-aware trajectory
construction, not another interpolation/gain sweep. New subjects/cameras,
ground/equipment, independent physical evidence and full-task learning remain
required. During product integration, a separate ordinary evidence-reader bug
was found and fixed; see [camera lab evidence](camera-lab-evidence.md).
