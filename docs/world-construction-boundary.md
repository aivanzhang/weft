# Shared-world construction: evidence and coordinate boundary

Verified October 29, 2026, **6:29 PM PDT**. These A-owned helpers give B a checked evidence/coordinate interface for one eventual person-and-object simulation. They **do not** implement the mixed simulator or recover new geometry. Existing scene profiles, actions, scorers, recordings and lab defaults are unchanged.

## What is implemented

### Source-qualified, complete evidence

`world_evidence.load_world_evidence({sha256: raw_json_bytes})` pins the **exact observation-document bytes**, not just a video's hash. An entity is identified by `(document_sha256, object_id)`. Two sources containing `sam-1` do not become the same object.

The registry uses the existing observation validator and additionally requires every decoded object/frame slot. This includes unavailable detections, unsampled frames, v2 absolute intervals and v3 decoder timestamps. It rejects omitted slots instead of interpreting them as perfect tracking or physical disappearance. Inputs remain byte-identical; internal state is immutable and inspection results are defensive copies.

```python
from hashlib import sha256
from pathlib import Path
from weft.world_evidence import load_world_evidence

raw = Path("scene-observations.json").read_bytes()
registry = load_world_evidence({sha256(raw).hexdigest(): raw})
inspection = registry.summary()  # Complete inspection metadata; not a model input.
construction = registry.construction_bytes()  # Immutable filtered JSON bytes.
```

At most eight documents / 64 MiB / two million JSON nodes are admitted. Duplicate keys, nonfinite values, malformed data and excessive nesting reject. Existing source formats remain unchanged; this is an opt-in adapter, not a schema migration.

### Separate construction from reserved evaluation

The default export includes only `construction`. Callers may explicitly select `generation` or `development_validation`; **`audit` can never be exported through this API**. Exposure to development evidence still makes it development, not a fresh holdout.

Every source slot remains, including withheld slots. A withheld slot retains its ID, original frame/time, entity and partition, but has null visibility, empty measurements and no evidence references. The original observation stays intact in the registry. Missing included observations retain their actual unavailable status.

The export omits raw payloads, provenance/scene blobs, labels, arbitrary relationship attributes and unqualified global claims. It includes typed measurements and only authorized qualified references. Relationships spanning an excluded participant interval are withheld, even when an opening reference is allowed. Hidden relationship counts are not exposed. Explicit authored physical priors belong in the candidate contract, not smuggled through a model-facing evidence export.

### Explicit coordinate and clock hypotheses

`world_alignment.load_world_alignment(document, evidence_ids_by_document=...)` loads the coordinated `world-alignment-v1` format. [Exact fields and ownership](parallel-workstreams.md#shared-world-boundary-review--october-29-2026).

- Each frame carries its name, length unit and right-handed axes. Supported axis labels are `right_handed_xyz`, `x_right_y_away_z_up`, `x_right_y_depth_z_up`, and `x_right_y_down_z_forward`; adding another convention requires an explicit extension.
- A positive uniform scale, unit `wxyz` quaternion and translation define the spatial map. Reflection/shear cannot enter through a proper rotation.
- A positive clock rate and explicit source/target origins define the clock map. Original source time remains in the returned state.
- Spatial and clock claims independently retain observed/inferred/assumed/unknown status, method, uncertainty and qualified evidence references. Referenced IDs must come from the caller's authorized construction registry; traceability alone does not verify the claim.
- Unknown mappings remain null and cannot transform a state. An assumed mapping can be inspected/applied but stays physically unverified.

For scale `s`, rotation `R` and clock rate `r`:

| Quantity | Conversion |
| --- | --- |
| Position | `s * R * position + translation` |
| Time | `r * (source_time − source_origin) + target_origin` |
| Linear velocity | `(s / r) * R * velocity` |
| Linear acceleration | `(s / r²) * R * acceleration` |
| Angular velocity | `(1 / r) * R * angular_velocity` |
| Orientation | Frame quaternion × source quaternion |

`alignment.map_state(from_frame=..., to_frame=..., time_s=..., position=..., ...)` requires exact frame/unit/axis agreement at use time. It returns a coordinate-conversion result, not a simulator state certified for execution. The rotation is prepared once; inspection JSON is not reparsed per state.

No mass, contact law, actuator limit, reward tolerance, implicit floor or physics engine is migrated. Rigid `scene_width` and human model lengths do not become measured metres. Camera-body origin removal must be reversed explicitly from its saved provenance; no origin is guessed by this helper.

## Saved real-source study

`artifacts/world-binding-01` contains the original observation copies, registry summary, filtered construction export, controlled reserved-data variants, authored/unknown alignment examples, mapped state, source snapshots and report. These files are local-only.

| Input | Source frames | Object/frame slots | Construction included / withheld |
| --- | ---: | ---: | ---: |
| Ramp segmentation | 44 | 88 | 88 / 0 |
| Three-ball segmentation | 181 | 543 | 543 / 0 |
| Toy-van segmentation | 354 | 354 | 354 / 0 |
| Person silhouette | 68 | 68 | 68 / 0 |
| Heavy squat pose adaptation | 213 | 213 | 107 / 106 |
| **Total** | **860** | **1,266** | **1,160 / 106** |

The 106 withheld Heavy squat slots are originally unsampled. Other construction-scoped unavailable slots remain explicitly unavailable, not removed. Five documents contain eight qualified entities but only four distinct bare IDs. **These unrelated sources remain separate scenes**; this study does not combine their bodies into a physical world.

An explicitly injected control changes half the ramp rows to reserved audit evidence. It is a software test, not a new physical holdout. The export retains 44 included / 44 withheld slots. Changing hidden visibility, measurements, provenance and relationship count leaves the construction export identical except for its pinned input-document hash. Removing one source row rejects `87 slots, expected 88`; requesting audit access rejects.

The authored alignment control checks position, clock, velocity, acceleration, angular velocity and quaternion conversion. A null spatial mapping rejects instead of receiving an identity/default scale. No real calibration or geometry was inferred.

## Verification, performance and preserved corrections

- **23 evidence tests pass**: `.local/world-evidence-tests-03.xml` (0.04 s).
- **29 alignment tests pass**: `.local/world-alignment-tests-03.xml` (0.03 s).
- Review corrected omitted-slot acceptance at the new boundary, withheld-visibility/relationship-count leakage, left-handed-frame acceptance and substring evidence authorization. Prior reports remain; there was no change to source/evaluator thresholds.
- Real-source study: **0.546 seconds**, one worker, six frozen source/test snapshots, 6,574,225 input bytes; 60-second / 64-MiB study bounds. Output before final report/manifest: 9,774,160 bytes.
- Five-source load: **193.812 ms**; inspection summary **0.381 ms**; construction export **39.224 ms**. Keep load/export off request and UI threads; these are not interactive UI measurements.
- One thousand state conversions: median **0.00917 ms**, p95 **0.01029 ms**, maximum **0.03554 ms**. No sustained UI or peak-memory claim is made.
- Read-only audit `.local/world-binding-verification-01.json` checks **22 sealed files**, six used snapshots, exact source slots/partitions/clocks, qualified identities, omitted raw payloads and independent saved coordinate/time arithmetic.
- Manifest: `7d1a3ea836c53afe80581cf37a039670b2cc2cfa8234ca8bbc0fe4f0e1252c87`.

No inference, download, video decode, paid request, simulator run or deployment was needed. All study jobs are terminal. Reproduction uses a **new** directory; do not rerun the unchanged passing study:

```sh
.venv-lab/bin/python -m pytest tests/test_world_evidence.py tests/test_world_alignment.py
.venv-lab/bin/python scripts/studies/check_world_binding.py --out artifacts/world-binding-new
```

## Still required for one usable mixed world

1. A must bind a person and relevant objects from the **same recording/physical scene**, reconcile source-qualified identities, and supply evidence-backed camera/shape/root alignment or explicit unresolved alternatives. A coordinate helper does not supply those missing values.
2. B must compile articulated and rigid components into one model/clock, with explicit shape, unit, control and task migration. B's in-progress mixed-world capability is not verified by this construction study.
3. Verify coupled contact with a non-contact control, initial-penetration handling without hidden shifts, passive-body freedom from source-position scripts, and exact combined recording/replay.
4. Retain prior geometry/contact failures and independently evaluate new observations and changed actions before claiming physical transfer. The full Track A and Weft goals remain incomplete.
