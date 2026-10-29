# Saved tracking as uncertain scene evidence

`tracking-evidence` bridges saved Meta tracking into the shared scene representation without making another request or asserting that a simulator can be built safely.

```sh
.venv-lab/bin/python -m weft tracking-evidence \
  --run artifacts/meta-three-ball-01/tracking \
  --source-mapping artifacts/meta-three-ball-input-01 \
  --out artifacts/my-ball-evidence
```

Output contains `segmentation-evidence.json`, `scene-observations.json`, `report.json`, and a manifest. The first file preserves full raw proposals and anonymous per-frame location sets. The shared file represents boxes and centers as **inferred**, with unknown physical identity, object kind, role, camera calibration, mass, and forces. Both retain missing and pre-initialization observations; neither fills gaps or promotes a world.

## Time and source identity

Shared schema version 3 records the exact decoder presentation-time vector instead of inventing a constant frame rate. Versions 1 and 2 retain their existing frame/fps contracts. Validation rejects inconsistent times, unordered timestamps, mismatched source dimensions, foreign frame IDs, and relationships outside the decoded interval.

The submitted MP4 and its original source are different identities. An optional source-mapping bundle must be sealed and contain both media identities plus complete frame/timestamp correspondence. Unknown formats or missing original media are rejected. The old toy-van preparation lacked its original WebM copy; it was not weakened or edited. A new verified bundle, `artifacts/meta-toy-van-mapping-01`, adds that exact original and preserves the earlier manifest. Its 22-file seal is `69eccb22283dc50e62c05fce1e00f1d35aeb98e8ddb181087a473c91d8637340`.

## Association uncertainty

A frozen geometry rule flags possible ambiguity when the gap between two returned boxes is at most half the smaller box diagonal. Groups persist and merge; separating again does not prove identities recovered. A missing post-initialization proposal also leaves continuity unverified. Raw provider IDs, confidences, masks, and the onset frame remain unchanged.

This is a conservative proximity heuristic. It can flag visually distinguishable objects because it performs no appearance analysis. It is not a collision detector, a calibrated error probability, or proof that an ID is wrong. It never reassigns IDs using evaluation labels. Encoded masks remain opaque in this export, with their hashes and original payloads retained. The separate offline [`tracking-masks` command](decoded-masks.md) decodes those unchanged payloads for silhouette inspection without changing identity assignments or promoting physics.

## Verified saved cases

| Case | Frames | Object/frame slots | Persistent ambiguity groups | Physics ready |
|---|---:|---:|---:|---|
| Three balls | 181 | 543 | 2 | No |
| Toy van | 354 | 354 | 0 | No |

The ball groups begin at relative frame 40 for IDs 1/2, then merge IDs 1/2/3 at frame 61, corresponding to original frames 2620 and 2641. These are automatically generated construction warnings, not fitted to the later identity labels. The toy van's zero groups do not clear its camera/edit or physical-model limitations.

Saved exports:

- `artifacts/segmentation-evidence-01/three-ball`: three-file seal `a9272d4a29ef82049146a4c63e2e3f760085b41fb2baeb0ea6ac4754721c653d`.
- `artifacts/segmentation-evidence-01/toy-van`: three-file seal `5b26c995fc8ec6e4999fa24e6014d9160e9bfd5c72ebe78659e29f7445ffab01`.

Thirty-four adapter tests and sixty shared-schema/converter tests passed before real exports. Bounds are 500 frames, eight objects, 14,000 pair checks, 64 MB per loaded JSON, and 256 MB per verified bundle. Media hashing streams bytes. No network request, training, source relabeling, default tracker change, or simulator-gate relaxation occurs in this export.

## Remaining reconstruction work

The scene generator still needs independent camera/support assumptions and a way to carry alternative identity assignments into dynamics. The current export is the evidence bridge, not an executable-world adapter. Accurate anonymous locations must not be discarded merely because identity is ambiguous, but they also cannot establish which physical object a later control command should affect.
