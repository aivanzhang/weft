# Three- and four-camera body candidates

Weft can now use **two or more supporting cameras per joint** from three or four
supplied views. This is an optional manual sandbox, not automatic camera recovery
from one video. Camera calibration, synchronized clocks, shared coordinates and
ground must be provided explicitly. Muscles remain illustrative; no learned
controller or physical-accuracy claim is added.

## Try the saved exercise candidate

```sh
.venv-lab/bin/python -m weft lab-console \
  --resume artifacts/aspset-third-camera-body-lab-01
```

Enter `controls`, choose an `action`, then `step`. Use `record start`, `record stop`
and `files` to save and inspect actions. Nothing moves until you step. To start the
visual lab yourself instead:

```sh
.venv-lab/bin/python -m weft lab \
  --resume artifacts/aspset-third-camera-body-lab-01
```

This candidate **fails the floor check**. It is useful for inspection and manual
interaction, not a successful exercise reconstruction. The existing equipment
sandbox and all defaults are unchanged.

- [Recorded manual interaction](../artifacts/aspset-third-camera-body-lab-01/sessions/session-c3c8e7838e80/replay.html)
  and [video](../artifacts/aspset-third-camera-body-lab-01/sessions/session-c3c8e7838e80/video.mp4).
- [Joint support and conflicts](../artifacts/aspset-third-camera-body-lab-01/run/two-view-triangulation.json),
  [body geometry](../artifacts/aspset-third-camera-body-lab-01/run/two-view-body.json),
  [floor failures](../artifacts/aspset-third-camera-body-lab-01/run/reference-feasibility.json).
- [Saved comparison audit](../artifacts/aspset-third-view-audit-01/report.json) and
  [model versus image-estimated joints](../artifacts/aspset-third-view-audit-01/model-image-joints.json).

These local artifacts are not included in a fresh Git clone.

## Add another calibrated view

Build from the same sealed original mobile body and [two-view root inputs](two-view-root.md#build-a-separate-candidate).
Add `--additional-view third-view.json`; repeat once for a fourth camera.

```json
{
  "schema_version": 1,
  "name": "third-camera",
  "evidence": "third-evidence.json",
  "camera": "third-camera.json",
  "clock_provenance": "Describe how these exact source frames align with the other views"
}
```

Paths resolve relative to this JSON. The original root configuration's
`world_provenance` must describe the shared coordinate system for **all** views.
Matching frame rate alone does not prove physical synchronization.

```sh
.venv-lab/bin/python -m weft camera-body \
  --run path/to/original-mobile-run \
  --camera first-camera.json --ground ground.json \
  --root-view root-view.json --additional-view third-view.json \
  --body-view both --out artifacts/my-three-camera-lab
```

Omit `--body-view` for root position only, or choose `directions`/`dimensions`
separately. No model call, training or server starts. Additional views cannot yet
be combined with `--partial-segments`. Shared-world/learning adapters that require
the older two-view method remain unavailable; they are not silently bypassed.

## What is retained

- Every original source/frame/joint slot, including unsampled and conflicting
  observations. Every original retained body pose still requires both hips.
- Exact evidence and camera hashes, declared clocks, original ground and input
  files, per-joint contributing views and rejected estimates.
- Existing bounded pixel-error solver: every qualifying view participates; no
  favorable-pair selection, outlier removal or guessed missing depth.
- Explicit nearest-proper solver rotations for decimal-rounding error, bounded
  to 1e-5 per matrix element. Original camera matrices stay unchanged. This is
  numerical preparation, not calibration fitting.
- Missing complete limb directions retain labeled single-view priors. Fixed
  body-length, angle, floor and optimization limits remain unchanged.

New methods are `multi-view-root-body-v1` and `multi-view-body-v1`. Historical
`two-view-*.json` artifact filenames remain for compatibility; their contents
identify the actual method and all three/four views. The legacy two-view path is
unchanged.

## Real-source result: improved coverage, worse floor consistency

ASPset clip `04ac-0003` retains all 188 source slots and 47 body poses. The same
frozen manually labeled ground is used for both candidates.

| Check | Two views | Three views |
| --- | ---: | ---: |
| Complete left-arm estimates | 26/47 | 47/47 |
| Complete left-leg estimates | 19/47 | 22/47 |
| Failing source floor poses | 1/47 | 4/47 |
| Failing interpolated floor poses | 3/111 | 7/111 |
| Maximum floor penetration, model units | 0.066662 | 0.092148 |

Frame 80's missing knee/ankle now have mid/right image support, but its model
ankles still differ from those image estimates by about 0.112/0.140 m under the
supplied scale. The model and image-estimated joints are different artifacts;
neither establishes measured anatomy or contact. All regressions remain saved,
and the candidate is not promoted.

The actual lab recorded **39 actions, one reset and 41 frames**, with every state,
reward and completion flag replayed exactly. Construction took 1.687 s and
150.8 MB peak RSS; interaction/export/replay took 1.050 s and 160.8 MB. Across 60
in-process action/state/JSON calls, median latency was 0.832 ms and p95 0.880 ms;
these are not browser, HTTP or sustained-use measurements.

The saved-only audit checked 776 manifest members, all 6,204 joint slots and 4,247
projections. It retained 84 conflicting joint estimates and all missing slots.
No paid API call was used. Exact source, runtime, test and process evidence is in
the [handoff](handoffs/aspset-third-view.md).
