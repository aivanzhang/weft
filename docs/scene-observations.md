# Shared scene-observations contract

`src/weft/observations.py` provides a versioned evidence interchange format for the existing rolling, planar and exercise pipelines. It does **not** replace their simulators, add segmentation, infer new objects, or establish generalized-video reconstruction.

## API

```python
from weft.observations import adapt_run, load_observations, validate_observations
from weft.io import write_json

# Read an existing lab directory or its run/ directory without modifying it.
observations = adapt_run("artifacts/exercise-lab-03")
write_json("/tmp/exercise-observations.json", observations)
validated = load_observations("/tmp/exercise-observations.json")
```

In-memory adapters are `adapt_tracks(evidence, profile="rolling" | "planar", tracks=None, scene=None, graph=None)` and `adapt_exercise(evidence, scene=None)`. The planar adapter requires the **full** `tracked-objects.json`, not only the fitting partitions, so excluded-frame measurements are retained. `validate_observations(document)` returns a validated independent copy. Loading is read-only, bounded to 64 MB per input artifact, and refuses named input files resolving outside the run directory. No model or network request occurs.

Run adaptation off a UI/event-loop thread; the exercise document retains large original model outputs and normalized landmarks.

## Schema version 1

Top-level profile is `scene-observations-v1` with `schema_version: 1`.

| Field | Contract |
| --- | --- |
| `source` | SHA-256 source identity, original image dimensions, FPS, declared and decoded frame counts |
| `time` | Seconds since video start, explicitly assumed frame-index/FPS basis; decoder timestamps remain in original payloads |
| `image` | Dimensions of the coordinate plane used by observations; this can differ from original dimensions after resizing |
| `coordinates` | Fixed units/axes for image pixels, relative pose depth, uncalibrated model-world coordinates and joint angles |
| `camera` | Motion and calibration claims, plus the original background-motion screen if present; “fixed” is an assumption, not recovered calibration |
| `objects` | Persistent object IDs, labels, property claims and explicit unknowns |
| `evidence` | Unique source-prefixed IDs, object/frame/time identity, evidence status and unchanged original record payload |
| `observations` | One record per represented object/frame, visibility, partition, evidence references and typed measurements |
| `relationships` | Participant IDs, evidence status, references, optional time interval and attributes for contact/support/joint hypotheses |
| `provenance` | Canonical input-evidence hash, original source metadata, optional scene/graph, and file hashes when loaded from a run |
| `limits` | Adapter scope and retained source limitations |

Every property claim contains `value`, `status`, and `evidence_ids`. Status is one of **observed**, **inferred**, **assumed**, or **unknown**. Unknown claims use `value: null`; observed/inferred claims require actual registry references. Active/passive/static roles are separate from evidence status. A role absent from the input stays unknown.

The evidence registry preserves original record IDs and payloads; the normalizer does not silently rewrite inconsistent timestamps. Observation IDs are deterministic: `<source hash>:observation:<frame>:<object ID>`. Hashes detect changes but do not authenticate an author who can rewrite both data and hashes.

### Geometry and units

- **Image centers:** measured tracker outputs in the observation plane's pixels, not physical positions.
- **Radius:** inferred circle approximation from image geometry.
- **Box:** inferred extent centered on the measured centroid. Original tracker widths/heights do not retain exact contour-bound origins, so the adapter does not claim exact segmentation boxes or fabricate masks.
- **Image landmarks:** inferred model coordinates converted from normalized x/y to pixels. Out-of-image landmarks remain unmodified; visibility/presence confidence remains attached.
- **Pose depth:** separate model-relative values in image-width units.
- **Model-world landmarks:** separate `model_meters`, explicitly `metric_calibrated: false`; these must not be treated as calibrated scene meters.
- **Joint angles:** inferred radians, with original confidence and model metadata retained in the raw pose payload.

Mixed unit labels, incompatible axes/calibration flags, coordinate dimension errors, nonfinite values and observed-status upgrades for inferred geometry/model outputs are rejected.

### Visibility and time

Visibility is `detected`, `partial`, `not_detected`, `not_sampled`, or `unknown`. Missing detections are not proof that an object was absent. Exercise frames not sent through pose inference are `not_sampled`, distinct from failed pose detections. Partial poses preserve all supplied estimates and confidence, including uncertain limbs.

For the legacy rolling format, visibility may report a detection whose geometry was not retained in a selected episode. That row becomes **unknown with no measurements**, not fabricated geometry. The adapter cannot restore evidence an earlier producer discarded.

Observations must be ordered by source frame and cannot duplicate an object/frame. Frame numbers must lie inside the decoded source. All declared observation/evidence times must match the fixed frame-index/FPS basis. A future variable-frame-rate timebase needs an explicit versioned extension, not relabeling existing seconds.

### Relationships and evidence boundaries

Planar projected-contact candidates remain **inferred, unverified**. Rolling support relationships remain hypotheses grounded in their cited observations. Source scene relationships without referenced observations are preserved as **unknown**, with their original claim in attributes; no plausible relationship becomes an observed fact merely because a simulator contains it.

Exercise hinge joints and anchored pelvis/torso are **assumed simulator topology/support**, not recovered joint mechanics or observed ground contacts. There are no fabricated muscle forces, anatomy scans, or contact measurements.

Generation, development validation, same-video audit, construction, unassigned and unsampled partitions remain separate. **This complete interchange artifact includes audit evidence; it is not an authorized repair-model payload.** Downstream generators/repair advisors must filter by their permitted evidence partitions before receiving records. Converting format does not promote a development set to a fresh holdout.

## Verification

### Selected-source intervals: schema 2

The profile name remains `scene-observations-v1`; `schema_version: 2` explicitly marks selected-interval sources. Schema 1 remains supported. Schema 2 adds `source.decode_interval` (inclusive absolute frame bounds), `source.prefix_advanced_frames` and `source.source_complete: false`. `source.decoded_frames` is the selected count, **not** the absolute last-frame index or a fabricated whole-source count. Full metadata frames remain separately bounded and labeled metadata.

All evidence IDs, frame times and relationship intervals must stay within the selected absolute interval. The adapter requires successful selected-frame consumption; it does not turn that into independently verified encoded-frame or whole-source completeness. Before initialization, visibility is `not_sampled`; initialized authored regions are `unknown`, with no measured positions. The original payload remains retained.

The extended contract has **51 passing tests**, including large absolute frame IDs, bounds, incomplete intervals, rebased IDs/times, relationship timing, pre-initialization coverage and read-only failed-run export. `adapt_appearance(tracks)` and `adapt_run` can export completed raw appearance tracking even when reconstruction failed before producing a scene. Completed evidence takes precedence; malformed evidence does not silently fall back. No physics or contact relationships are invented. [Selected-interval workflow and evidence](source-intervals.md).

### Preserved original contract check

`tests/test_observations.py`: **23 checks passed** across the core suite and targeted additions. Checks cover all three adapters; active/passive/static/unknown roles; excluded tracks; missing versus unsampled poses; uncertain/out-of-frame landmarks; inferred geometry; source identity; duplicate and dangling references; frame/time conflicts; relationship timing; unrelated-object citations; mixed units; no invented detection; and read-only file loading.

`artifacts/scene-observations-contract-01` preserves the adapter source plus normalized outputs from three actual saved runs:

| Input | Objects | Raw evidence records | Frame/object observations | Relationships | Adaptation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rolling-lab-01` | 3 | 188 | 157 | 1 | 0.004 s |
| `exercise-lab-03` | 1 person | 118 | 235 | 9 assumed | 0.150 s |
| Analytic elastic-contact run | 2 | 216 | 108 | 0 | 0.007 s |

These are single-run local timings, not sustained throughput claims. Every output round-tripped through validation; all input file hashes stayed unchanged. The output manifest verifies **5 files**, SHA-256 `af86d1f380b1520563576e5d48ecfd8e50f0bfa2018ed18c7192aabe20cb4abb`.

Remaining: foundation-model segmentation/detection, generic object association, calibrated camera/3D recovery, inferred physical relationships, shared executable-world adapters, and diverse independently labeled perception benchmarks. This schema supplies a common place for such evidence; it does not implement those capabilities.
