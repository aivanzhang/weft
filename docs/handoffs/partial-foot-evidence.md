# Partial calibrated foot evidence

Updated 2026-10-31 14:50 PDT. Owner: `/root/exercise_reconstruction`.
**Implemented and analytically tested; frozen for review. No saved-source run.**
Own only `src/weft/partial_foot_evidence.py`, `tests/test_partial_foot_evidence.py`
and this note. Root owns integration/Git; no shared consumer, engine, task, lab or
avatar files changed. Root reports reviewed consumer/importer publication at
`d1b461fad68974b67828d2eb71e691072a8ee1c2`; these three new paths were excluded.

## What this adds

The complete-foot producer discards usable ankle/toe coordinates when a heel
fails. This separate `partial-calibrated-foot-evidence-v1` adapter exposes exact
saved **ankle/toe estimates** independently, only in the construction prefix.
It does not make the strict calibrated consumer permissive or attach feet.

Every heel coordinate/orientation remains null, with
`unsupported_by_partial_profile`. Original heel status is unchanged, including
`estimated` on valid rows. Missing/excluded/unsampled/body-unavailable slots retain
their original identities, clocks and status. Numeric confidence is null with
`not_stored_in_saved_triangulation`: stored triangulation has observation statuses
and ray diagnostics, not numeric image confidence. The original confidence rule
is provenance, never an invented measurement or probability.

All positions remain camera/model-derived estimates. Supplied camera scale,
ground, cross-view timing and body mapping are priors, not measured anatomy.
Qualification means exact pinned stored evidence, not independent image/camera
validation. Contact/load/grasp remain unknown and physical admission is false.

## Public API and bounds

```python
from weft.partial_foot_evidence import load_partial_foot_evidence

partial = load_partial_foot_evidence(body_bytes, reference_bytes, triangulation_bytes)
partial.describe()            # bounded identity/coverage/strict-consumer metadata
partial.frame(0)              # original frame, never a later available substitute
partial.iter_frame_bytes()    # immutable cached rows, streamed without full copying
```

The object retains the exact three input byte objects. Preparation reuses the
strict consumer, existing byte/envelope checks, saved-point validator and explicit
world-alignment mapping. Individual queries do not parse/revalidate/copy complete
histories. The original gap series survives verbatim in `reference_bytes`; new
partial-endpoint gaps have a separately named series and denominator.

Limits: three inputs ≤8 MiB each / 24 MiB total; ≤1,800 source slots / 900 body
poses; five-second cooperative preparation (not process isolation); ≤8 MiB cached
output, ≤16 KiB per row and ≤64 KiB metadata. No I/O, ray reconstruction, fitting,
heel hypothesis/family, interpolation, shape scaling or native installation API.

## Known source gap and exact future inputs

The saved jogging acceptance remains **unavailable at frame 0**: right heel
`inconsistent_image_rays`, right ankle/toe `estimated`, left foot conditional.
All 240 source / 480 foot / 1,440 endpoint slots and six negative gaps remain.
Previous acceptance seal:
`c85f39905ecf8b99fe8bdff24c971f4d7fa4fb4cfbda4fc3031216b6004d3fbd`.
See [saved acceptance](calibrated-foot-consumer-jog.md); it must not be rerun.

A future separately approved partial-evidence study would use exactly:

- `artifacts/two-view-root-jog-lab-01/run/human.json`, raw SHA256 `849ece268cc6b091cf85460f3e22ecee172aa5323c4db88dac3e221e367d0867`.
- `artifacts/multiview-foot-reference-jog-01/result.json`, raw SHA256 `f35cbedc99b319b33bf73813059065483481a31ec920e9d8ad8a148d9547fdb2`.
- `artifacts/two-view-root-jog-lab-01/run/two-view-triangulation.json`, raw SHA256 `0ed173522834b43cbd0af4c683e3bf424acb9d7f6418c463a31260856d5e93c9`, body-bound canonical SHA256 `fceb752b2c1f6978cfe7460681944f2a26cf8bea4ca3cf7686351f2c3129841f`.

No saved landmark payload was supplied to this new adapter. Its source-specific
result remains unverified. Two endpoints still leave one free rotation about
their connecting axis. Fixed triangle priors also need not match their estimated
separation: expected distance is `sqrt((length-along)^2 + height^2)`. A heel needs
another source-backed original-frame constraint with explicit uncertainty, not
ground-up orientation, a later start or a lower-penetration choice. A heel point
alone would not establish sole/collision geometry or support/contact.

## Frozen verification

- Module: `adbbc7361d59c50c115d7921c7c82ec7c8e49045fa95038937f687a889f6a0ea` (12,820 bytes).
- Tests: `bd2f0f9348be82386706c2234813d553bcce99ec8bf10c38358ccb894e3d9182` (15,265 bytes).
- `PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_partial_foot_evidence.py --junitxml=.local/partial-foot-evidence-tests-01.xml`: **26 passed**, 0.597 s; receipt SHA256 `23780f44529a89887d79ec9093b0c105661e0e0b7d49f65a312ce10447f27f5b`.
- New larger case only: same command with `-k large_original_clock --junitxml=.local/partial-foot-evidence-large-02.xml`: **1 passed**, 1.552 s; receipt SHA256 `3fd6e8ce61102aeb528514e012c092a4061316188cf13e7ec28f9f845a38e6ba`.
- **27 distinct cases**, no passing suite repeated. All are authored analytical fixtures with native/model/media/network/subprocess and ray reconstruction forbidden. Checks cover original rejection, exact joins, missing binding, later-coordinate nonuse, unchanged negative gaps, unknown confidence, malformed formerly-skipped endpoints, deadlines/output bounds and immutable small queries.
- Larger fixture: 8,206,000 input bytes; 1,002 source / 2,004 foot / 6,012 endpoint slots; preparation **0.565911 s**; 20 row queries **0.000224 s**, largest query 3,277 bytes, complete cache 2,779,688 bytes. These are local analytical timings, not UI or real-video performance guarantees.
- Initial JUnit is 8,392,662 bytes because pytest expanded the synthetic oversized-input parameter name. Preserve it. Compact IDs were assigned afterward without changing assertions or rerunning passed cases. Future receipts avoid this artifact overhead. Scoped diff validation passed.

## Next safe action

Independent read-only review, then root integration. No source study, native
foot admission, authored-heel experiment or product wiring is authorized here.
If a saved study is approved later, freeze exact inputs/runtime/limits and retain
all outcomes without retries or threshold changes. The existing first-frame heel
rejection, six negative gaps and floating/support uncertainty remain unresolved.

Live handles: none. No Git mutations or paid work. All three files are released
for review; no edits or jobs queued after freeze.
