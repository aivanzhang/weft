# Calibrated foot consumer

Updated 2026-10-31 14:28 PDT. Owner: `/root/support_ui`. Status: source-geometry consistency fix tested and frozen.

## Scope and interface

Own only `src/weft/calibrated_foot_reference.py`,
`tests/test_calibrated_foot_reference.py`, and this note. Root owns existing
consumers/CLI/lab/engine/recording/Git. Earlier reconstruction and admission
diagnosis are published at `12cd7a59ca3cc0f6c4fa0897971a659f892069e0`; this lane
remains excluded until reviewed.

`load_calibrated_foot_reference(body_bytes, reference_bytes)` prepares an
immutable adapter for **calibrated-two-view-foot-reference-v1**, never relabeling
it as the monocular profile. It preserves exact raw bytes/hashes, all original
slots/statuses and source/body/camera/ground/world-clock bindings. A bounded
cached start view exposes local heel/toe centerlines and reference rotations.
`initial_ankle_angles(knee_rotations, coordinate_frame=...)` requires the exact
bound body frame. Missing/incompatible start evidence remains unavailable; no
later pose selection or interpolation.

Reuse camera/root/ground/world-alignment validators and existing ZYX rotation
math without importing native `foot_body`. The full evidence is retained once as
immutable bytes, not deep-copied on state/angle requests. Numeric source scale
does not turn the camera/body prior into measured anatomy. Capsule mass/radius,
physics, contacts and body placement are not supplied or changed by this adapter.

## Bounds and verification

At most 8 MiB per input / 16 MiB combined, 1,800 original source slots, 900 body
poses and a 5-second cooperative preparation deadline. Only bounded metadata
(≤32 KiB) and two-foot start geometry (≤8 KiB) are cached. The angle-query path
never reparses/deep-copies full evidence; oversized knee matrices reject before
array allocation. Pure analytical tests block native physics/models,
network/media/subprocesses. No source study or native job ran.

First command:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_calibrated_foot_reference.py --junitxml=.local/calibrated-foot-consumer-tests-01.xml
```

**65 passed, 1 failed**, 0.597 s. The failure was a test expectation: incompatible
body/source metadata returned unavailable, while the test incorrectly expected
an exception. The test now verifies unavailable status, retained original bytes
and retained mismatched counts. The original receipt is preserved, SHA
`8bf26f1b4655dace93ad9415a65742d8537cb484f53fcf5aab7de3344692de6c`.

The same narrow validation review added strict numeric clock/count checks and a
cheap 3×3 knee-matrix envelope before NumPy allocation. Focused command:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_calibrated_foot_reference.py -k 'body_source_metadata or boolean_clock or oversized_knee or exact_body_frame_ankle or improper_knee or control_queries' --junitxml=.local/calibrated-foot-consumer-focused-02.xml
```

**14 passed**, 0.094 s; SHA
`3a8ee972913849aa381b1095dfbab167b4113eabfc4cc4361af9727729fea2ae`.
The prior 69 distinct cases are covered across these runs; 55 unchanged passing
cases were not repeated. Coverage includes 1,002 original source slots, all
missing/excluded rows, nonzero original start, refusal to choose a later available
foot, exact source/camera/body/ground/clock pins, negative source gaps retained,
singular/nonsingular proper-rotation round-trips, immutable snapshots, and final
cooperative deadline enforcement. The analytical body fixture validates binding
math only; it intentionally is not a complete physics-valid human model.

## Source-geometry consistency correction

Read-only review found that coordinated axes/endpoints could disagree with the
retained source points, and a signed gap could be erased while retaining the
same world point and ground. Preparation now checks each available foot's exact
three-point world/body slots against the already-bound translation and ground.
Saved dimensions and orthonormal axes must reproduce the saved heel→toe and
heel→ankle vectors (absolute arithmetic tolerance `1e-10`, no relative tolerance).
World coordinates retain the source ±2,000 m bound; mapped coordinates are bounded
by ±4,000 model units. Shapes, axes, gaps and raw evidence are never refitted,
normalized or replaced. No triangulation or original-source acquisition occurs.
These are internal saved-evidence consistency checks, not new physical validation
or an independent revalidation of the pinned camera/pose input documents.

The five new cases reject a coordinated 90° axis/endpoint forgery, an erased
negative source gap, changed body/world points that disagree with the mapping,
and a changed inferred source dimension. Red command, against pre-fix module SHA
`fc725656076739e55f1fd26fc037ee58a1c9a3d6d9472d9936c366674bf3c0ce`:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_calibrated_foot_reference.py::test_saved_geometry_must_match_retained_source_triplet --junitxml=.local/calibrated-foot-consumer-source-geometry-before-01.xml
```

**5 failed as expected**, 0.109 s; every forgery was accepted before the fix.
Receipt SHA `895348db456de633c56949cf8695ac877a028d46ee8e2c477a6999be07f5d0a0`.
Focused green command:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_calibrated_foot_reference.py -k 'saved_geometry_must_match or signed_source_gaps or distinct_consumer or exact_body_frame_ankle or control_queries or endpoint_geometry or preserve_more_original_slots' --junitxml=.local/calibrated-foot-consumer-source-geometry-after-01.xml
```

**21 passed**, 0.449 s (5 new + 16 affected prior cases); receipt SHA
`efb84bd6e7ff0ef757b1cb7846ec22c9e8d74d0e425dbd52e0d834c00789e4e4`.
The remaining 53 prior passes are reused, giving 74 distinct covered cases; all
earlier receipts remain unchanged. Valid negative gaps, immutable bytes, the
1,002-slot envelope and cached angle queries remain covered. Additional work is
constant per available foot during bounded preparation only; the query path and
cache sizes are unchanged. Test-suite elapsed time is not a browser/native or
maximum-envelope performance claim. No native/model/media/network work ran.

## Usage and frozen files

```python
from pathlib import Path
from weft.calibrated_foot_reference import load_calibrated_foot_reference

prepared = load_calibrated_foot_reference(
    Path("human.json").read_bytes(), Path("calibrated-foot-reference.json").read_bytes()
)
summary = prepared.describe()  # Small cached metadata, not the full pose history.
if prepared.ready:
    geometry = prepared.initial_geometry()
    angles = prepared.initial_ankle_angles(
        knee_rotations, coordinate_frame=summary["body_coordinate_frame"]
    )
# prepared.reference_bytes retains every original source row and uncertainty.
```

`knee_rotations` must contain proper 3×3 left/right knee matrices in the **exact**
declared body frame. No normalization, body rotation, scaling, outer-world
translation or source-clock conversion is inferred. Local centerlines do not
include an authored capsule radius/mass. `initial_geometry()` includes the exact
original source signed gaps and marks physical admission false. Missing or
incompatible binding/start evidence returns unavailable; malformed envelopes,
geometry or limits reject. Both raw input byte objects are retained unchanged.

| File | SHA256 |
| --- | --- |
| `src/weft/calibrated_foot_reference.py` | `501b682b1d607f4c32546995cd65f55004c620e72f2d3dc4bf16bd154871bb85` |
| `tests/test_calibrated_foot_reference.py` | `9352c02226128a177039fdeb4d52c353d0d3057db8e47101e35355f402b9071f` |

## Explicit integration boundary

The currently published shared-world profiles reject camera-reference bodies. Adapter readiness is
**not coupled-world admission**. Root must separately preserve/validate trajectory,
scaling, placement, collisions and task gates before any future consumer wiring.
No existing foot consumer accepts the new profile yet. Root/support_learning are
separately implementing a versioned camera-body shared-world boundary with unit
scale, no outer placement and source-fps scene clock; this adapter does not alter
that work or the old rejection. It requires the exact original human bytes and
30-Hz `human_spec.dt`, while retaining source-fps row timestamps unchanged.
Single-leg right-foot gap
−0.10389564 versus −0.025 remains rejected; the separate jogging study's six
negative source gaps remain unchanged. Contact/load/grasp and physical accuracy
are unverified.

Next integration boundary: explicitly wire the distinct calibrated profile only
after a compatible camera-world scene has independently validated its unchanged
body, ground, trajectory and source clock. Then validate capsule construction and
original-start collisions without changing source estimates/gates. This adapter
does not perform physical admission, recover contact or make the original failed
single-leg candidate usable. No saved real-source or native acceptance was added.

All three owned paths are frozen/released to root. No existing consumer edits,
live job or Git mutation; root owns publication. Source and runtime identities
must be rechecked before any later source/native study. No repeat of these
unchanged passing checks is needed.
