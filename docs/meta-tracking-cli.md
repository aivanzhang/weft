# Inspect video tracking locally

The `segment` command prepares a Meta tracking request by default. It does not train or generate physics. An explicit `--execute` sends one bounded paid request and writes an offline frame-by-frame viewer.

## Open the saved example without another call

Open [the prepared viewer](../artifacts/meta-tracking-view-01/index.html) in a browser yourself. Use **Previous frame**, **Next frame**, or the slider. Boxes and confidence are model output; missing observations remain missing. The links below the video expose the complete report and raw model output. This viewer preserves encoded masks; the separate [`tracking-masks` viewer](decoded-masks.md) displays decoded silhouettes on exact source frames, including a completed exercise example.

The viewer is local and makes no API calls. Browser video seeking is approximate; it is not a frame-exact visual scorer. Browser/device interaction has not been tested by the agent. The input's timestamp mapping and the exported JavaScript syntax were checked separately.

## Use your own MP4

Choose a source-pixel rectangle around each object at a known frame. Each `--box` contains six integers: **object ID, frame, left, top, width, height**. IDs must be distinct integers 1–8; frames start at zero. Do not use later evaluation annotations as initialization.

```sh
# Example using the saved toy-van derivative; prepares locally, costs $0.
.venv-lab/bin/python -m weft segment \
  --video artifacts/meta-toy-van-input-03/toy-van-crf18.mp4 \
  --box 1 0 970 538 108 68 \
  --out artifacts/my-tracking-preparation
```

Inspect `request.json`, `video.json`, and `report.json`. To execute, repeat with a **new output directory**, add `--execute --meta-budget 0.15`, and retain the same input and boxes. The shared $5 development ledger also applies. The per-request allowance must cover the conservative reservation shown by preparation; more objects or frames can require a larger allowance.

This path accepts MP4 files up to 37 MB, 500 decoded frames, eight objects, 4096 pixels per dimension, and 8,847,360 pixels per frame. It verifies dimensions, decoded count, increasing decoder timestamps, and copied-video identity. It never silently trims, rescales, changes formats, adds boxes, retries a paid request, or replaces missing detections with predictions. It does not independently certify encoded container completeness for arbitrary inputs.

Successful execution writes `index.html` alongside the source copy and sealed evidence. Each object can have its own initialization frame. Tracking propagates forward; pre-initialization slots are explicitly unobserved, not scored as detection failures. A returned ID is the model's association, not proof of physical identity after contact or occlusion.

## View an existing sealed result

```sh
.venv-lab/bin/python -m weft tracking-view \
  --run artifacts/meta-toy-van-live-01 \
  --video artifacts/meta-toy-van-input-03/toy-van-crf18.mp4 \
  --out artifacts/my-tracking-view
```

This verifies the sealed run, requires the exact MP4 used for the call, and creates a separate viewer without altering the original. No request is sent.

Use [`tracking-evidence`](segmentation-evidence.md) to export these saved proposals into the shared scene format, preserving exact decoder times and possible identity ambiguity. This export is also offline and does not generate or certify physics.

Then use [`tracking-masks`](decoded-masks.md#decode-another-saved-result) with that evidence export and the exact submitted video to inspect silhouettes, areas and centers. It preserves all raw proposals and identity warnings, and makes no additional request.

The separate [`tracking-scene` workflow](segmentation-planar-candidates.md) tests explicitly authored planar assumptions and preserves unsupported outcomes. Executable scenes can be controlled and recorded with `scene-sandbox`; passing tracking alone is not enough to establish a physical reconstruction.

## Limits that remain

The network exchange runs in a dedicated process with a 120-second deadline and at most two seconds of termination cleanup. Secrets are not saved in artifacts; the shared ledger conservatively retains uncertain charges after failure. Local decode has a 30-second between-frame check, not a hard process deadline. These are application limits, not provider-side spending caps.

Tracking is one input to reconstruction. Camera motion, edited time, depth, contacts, gravity, and body/muscle dynamics still require separate evidence and checks. This command never bypasses the existing simulation rejection gates or automatically enables paid processing in the regular sandbox.
