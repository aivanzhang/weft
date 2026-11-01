# Camera-body reference input custody

## Status — October 31, 2026 (PDT)

Implemented a separate metadata-only loader for the opt-in camera/surface fit.
The legacy `whole_body_reference_inputs.py` is unchanged. This lane touched only
the new loader, its authored-fixture tests and this note; root owns pipeline,
worker/CLI integration and any later native acceptance. No real source/media
reads, native imports/execution, reconstruction, API/network calls or Git
mutations ran. No live handles remain.

**21 distinct pure checks passed.** This does not establish that a saved real
source has all required bindings or that a camera candidate passes native
admission. The analytical bodies are metadata fixtures, not physics acceptance.

## Public API and accepted sources

```python
prepare_camera_reference_inputs(source, *, expected_manifest_sha256=None, deadline)
recheck_camera_reference_inputs(snapshot, *, deadline)
```

`deadline` is a finite monotonic deadline. Preparation returns exactly
`{files, identity, pins}`; `files` maps output aliases to bytes, `identity` is
JSON-compatible, and `pins` contains absolute `Path`/raw-SHA pairs for terminal
source recheck. Neither function writes outputs or constructs a model.

Accepted packages are narrowly distinguished:

1. The existing camera-body producer's sealed `schema:1` package, with its
   camera/root/ground proof, source-copy provenance and producer inventory.
2. A sealed `camera-shared-candidate-v1`, status `constructed`, whose retained
   `source-run/manifest.json` and `source-verification.json` bind the same selected
   bytes to the original producer package.

Both require an unchanged grounded `two-view-root-body-v1` body and at least
three original reference poses. Other camera methods, stacked top-level
refinement/alignment markers, failed packages and active policy/lab/outcome
markers reject. No generic lab or arbitrary JSON body is reinterpreted as a
camera source.

The loader reuses the existing no-follow `_Reader`/manifest/root-identity custody
helpers and `camera_shared_candidate._qualified_inputs`. The latter calls the
pure camera coupling/root/ground validators and verifies both views, canonical
evidence/camera bindings, original triangulation/root slots and source clocks.
It does not call `_scene`, shared-world/native admission or reconstruction.

## Frozen copied-member contract

| Selected source member | Output alias |
| --- | --- |
| `human.json` | `inputs/human.json` |
| `evidence.json` | `inputs/evidence-original.json` and identical `inputs/evidence-derived.json` |
| `camera.json` | `inputs/source-camera.json` |
| `ground.json` | `inputs/ground.json` |
| `two-view-triangulation.json` | `inputs/two-view-triangulation.json` |
| `two-view-root.json` | `inputs/two-view-root.json` |
| `inputs/two-view-root/evidence.json` | `inputs/secondary-evidence.json` |
| `inputs/two-view-root/camera.json` | `inputs/secondary-camera.json` |
| `inputs/two-view-root/config.json` | `inputs/two-view-config.json` |
| `source.json` | `inputs/source.json` |
| `source-input-manifest.json` | `inputs/source-manifest.json` |
| `camera-provenance.json` | `inputs/camera-provenance.json` |
| Actual supplied `manifest.json` | `inputs/body-manifest.json` |
| Generated selected-only custody proof | `inputs/camera-inputs-proof.json` |

A constructed shared candidate additionally preserves `source-run/manifest.json`
as `inputs/camera-source-manifest.json` and `source-verification.json` as
`inputs/camera-source-verification.json`.

The body, both evidence aliases and source camera are **raw-byte exact**. No
dimension field is added. `dimension_derivation` is
`none; exact original evidence bytes`; there is no authenticated-decoder claim.
Root compares `inputs/source-camera.json` byte-for-byte with the supplied surface
camera before output creation. Only the explicit camera fitter branch qualifies
camera/evidence dimensions directly; legacy dimension checks remain unchanged.

## Identities and proof scope

`identity.profile` and the generated proof profile are
`camera-body-reference-inputs-v1`, status `metadata_verified`.

- `body_manifest_sha256`: raw seal of the actual supplied package.
- `camera_body_manifest_sha256`: original camera-body producer seal; differs
  from the above when the supplied package is a constructed shared candidate.
- `source_manifest_sha256`: raw upstream original source-input seal.
- `human_sha256`, `evidence_original_sha256`, `evidence_derived_sha256` and
  `source_camera_sha256`: raw file hashes. Separate `*_canonical_sha256` fields
  identify parsed body/evidence/camera objects; never substitute one for another.
- `copied_files`: every output alias → raw SHA, including the encoded proof.
  Parent embeds the identity at `plan.source`; report joins selected aliases
  against this map and the sealed outer inventory. The proof has no self-hash.
- Coupling/root/ground/shape/source-clock pins, original body timestep and the
  complete original source clock are retained. Body timestep is not forced to
  equal fractional source-frame duration.

The proof contains `input_manifest_sha256`, `selected_members` with raw
hash/byte-count/copied-alias records, `attribution`, the validated
`camera_coupling`, both `views`, source-slot counts/statuses, dimension policy and
scope. Original complete slots remain in the exact root/triangulation documents,
not newly filled or retimed rows. Its historical original-body/root hashes are
retained declarations from the validated binding; historical body bytes are not
loaded or falsely claimed verified.

Primary source rights remain a retained declaration. Secondary media/rights are
unavailable. No source video/image, scene/task, model XML, policy or prior outcome
is copied or read. Unselected historical manifest members are not claimed
verified. No anatomy, calibration accuracy, contact or simulator admission is
established by internal metadata consistency.

## Bounds, verification and resumption

At most 15 selected source files are read (including manifests), each metadata
member≤8 MiB and each manifest≤256 KiB, with the existing 32-MiB aggregate reader
limit. Copies are≤18 files/32 MiB: 15 members for a producer and 17 for a shared
candidate. Parent still enforces its combined input≤64 files/32 MiB and complete
output≤96 files/64 MiB after adding surface inputs/runtime/report. Overflow
rejects; no proof or source slots are dropped. Root identity, no-follow paths,
exact selected hashes and prepared output copies are rechecked before publishing.

Tests run with native/model/media/network/process entrypoints denied:

```sh
PYTHONPATH=src .venv-lab/bin/python -m pytest -q tests/test_camera_body_reference_inputs.py -o junit_family=legacy --junitxml=.local/camera-body-reference-inputs-tests-01.xml
```

First 20 checks passed in 0.232 seconds. A later malformed-provenance rejection
and one affected positive path passed in 0.061 seconds in
`.local/camera-body-reference-inputs-tests-02.xml`; unchanged checks were reused.
Coverage includes raw/canonical separation, fractional clocks/nonzero original
starts, retained failed/unsampled slots, exact selected reads, both source routes,
deletion/reseal attacks, unsupported bodies, changed bytes/pins/aliases, symlinks,
deadlines and copy-budget exhaustion. No native acceptance was run.

Frozen loader SHA:
`e22d2a46a5ed1b3e564751f697cec2441da4a7795a09bad2ef88df7b53f3b317`.
Frozen test SHA:
`e01c60e6dce16f4e3fd0a72c625eece3bdb68890a6078f3791a8d040d1a702fc`.
Command/results/receipt pins are in
`.local/camera-body-reference-inputs-checks-01.json`.

Next safe action: root/reviewer checks the released adapter and integrates only
the explicit camera route; UI/report validates selected proof aliases without
claiming full source-package verification. Freeze any later authored native plan
separately. Do not alter the legacy loader, auto-create real surface assumptions,
read media to fill gaps or rerun earlier failed studies.
