# Explicit equipment depth prior

## Ownership and status — November 1, 2026, PST

Implemented; guarded analytical checks pass. Root integration/review pending. Owner:
`/root/reconstruction_next_step`. Owned product files are
`equipment_proposal_config.py`, `equipment_proposal.py`,
`shared_world_reconstruction.py`, plus root-authorized opt-in dependency
archiving in `shared_world_pipeline.py`. New focused tests:
`tests/test_equipment_depth_prior.py`. Scratch and exact pre-edit copies:
`.local/equipment-depth-prior-core-01/`.

The workflow agent owns guided configuration/reporting. Root owns integration,
TODO/combined handoff and any separately planned native/source acceptance.
Controller, loader and avatar work stay untouched. No Git, native geometry,
media processing, API/key/ledger operation, acquisition or paid work is allowed
in this lane. All three guard processes exited zero; no active job remains.

## Approved opt-in contract

Retain `source-equipment-proposal-config-v1`. Existing
`geometry_prior: floor_box_depth_equals_width` and two-field equipment rows
remain unchanged. The new mode is:

```json
{
  "geometry_prior": "floor_box_explicit_depth_ratio",
  "equipment": [{
    "object_id": "explicit-source-object",
    "source_frame": 12,
    "depth_prior": {
      "depth_to_width_ratio": 0.5,
      "provenance": "Explicit supplied depth assumption; not verified physical geometry."
    }
  }]
}
```

This is a fragment of the complete ordinary config, not a standalone config.
Every selected object requires its own exact prior. Ratio is dimensionless,
finite, numeric but not boolean, between `1/120` and `120`. These bounds derive
from the existing admitted half-extents `[.005, .6]`; actual dimensions,
projection, floor and initial-collision checks remain independently required.
Provenance is a nonempty string of at most 2,000 characters, preserved exactly.
Neither the ratio nor its explanation is authenticated measurement/contact.

New proposal method:
`explicit-fixed-pelvis-floor-box-depth-ratio-proposal-v1`; new per-object
geometry status: `authored_floor_box_ratio_candidate`. The exact `depth_prior`
survives into object provenance. No default ratio, search, fit, measured label,
hidden cavity or force channel is introduced. Ratio one must retain the old
numerical geometry, with deliberately distinct explicit-prior provenance.

## Verified state

**101 distinct analytical cases pass:** 59 new repository tests, 41 existing
validator-compatibility cases, and one saved-code comparison case. The latter
checks 81 independently projected default boxes plus one complete default
proposal and one constructed-scene result byte-for-byte against the exact
pre-edit code. It is not 83 additional pytest cases.

- `checks-01.xml` / `checks-01.json`: 54 new and 41 existing cases, 0.638091 s
  complete guard runtime.
- `checks-02.xml` / `checks-02.json`: two new extreme-ratio inversion cases,
  0.300549 s. Its explicit `-k ratio_limits` filtered the other pending checks;
  it did not rerun the earlier passing cases.
- `checks-03.xml` / `checks-03.json`: three new projection-domain rejections and
  the saved-code comparison, 0.305701 s. The comparison detail is
  `baseline-comparison.json`.

All receipts are under `.local/equipment-depth-prior-core-01/`. Each guard
records zero native calls. The known MuJoCo import-time `subprocess.Popen`
probe was denied in each fresh process; no guard was loosened or process run.
Bounds were 10 seconds CPU and 4 MiB per output file, with no child processes
or network. These times are scoped checks, not product/UI throughput claims.

The tests use independent homogeneous eight-corner projections across
left/right/straddling branches and depth ratios `.25`, `.5`, `1`, `2`, `4`,
plus the existing extent-derived ratio extremes. The tests reject nonfinite,
boolean, absent/out-of-range and unexplained priors; fake ratio/dimension/floor/
clock/projection values; and unchanged projection-domain violations. A mocked
initial-overlap rejection still propagates, with the original start and common
source frames retained.

Actual producer/package/constructor composition runs on complete authored
metadata: six source slots, three body references and 18 person/equipment
slots. Source registries, original-body mapping, projection, package hashes,
source rechecks and the new conditional dependency archive are real. Only
model-schema, native initial-collision and native reference-feasibility
boundaries are mocked. This is **not native admission**. Legacy construction
does not gain the optional module; explicit-depth construction archives exact
`equipment_proposal_config.py` bytes. All source parents remain unchanged.

Reproducible guard prefix:

```sh
.venv-lab/bin/python -B .local/split-limb-feedback-actor-01/run_pure.py XML RECEIPT TESTS...
```

`checks-01` selected `tests/test_equipment_depth_prior.py` and
`tests/test_equipment_proposal_config.py` before the five additional edge cases
were added. `checks-02` selected `test_equipment_depth_prior.py -k ratio_limits`.
`checks-03` selected
`tests/test_equipment_depth_prior.py::test_ratio_mode_retains_projection_domain_rejections_without_fallback`
and `.local/equipment-depth-prior-core-01/check_baseline.py`.
Do not rerun these unchanged passing checks merely to inspect their evidence.

## Remaining work

- [ ] Root/workflow integration and independent review. The workflow owner has
  the approved fields, public validator and reusable model-free test fixture.
- [ ] Any native/source acceptance requires a separate root-owned frozen plan;
  none is implied by these mocked model boundaries.

The already-saved real configuration still declares equal-depth boxes for
`sam-2` and `sam-3`. No evidence currently justifies selecting a new real ratio.
The authored fixture cannot establish unfamiliar-video, contact, grasp or
physical accuracy. Do not tune geometry to the protected controller failure,
rerun old failed searches, alter source clocks or rewrite sealed artifacts.

This lane completes when its code, explicit opt-in contract, legacy numerical
parity and guarded core checks are integrated without changing protected gates.
It does not complete generalized or physically accurate reconstruction.
