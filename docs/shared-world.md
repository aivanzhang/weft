# Coupled human and object sandbox

Weft can now place a mobile articulated body and rigid objects in **one MuJoCo simulation**, with one clock and contact solver. You can control the body or a dynamic object, inspect three views, and record/replay actions. This is a runtime capability, not proof that equipment geometry, forces or anatomy were recovered correctly from a video.

For separate whole-interval body/equipment checks with **human-only actions**, use the current `artifacts/shared-exercise-policy-lab-02` via [exercise tasks and recordings](shared-exercise.md). That lab keeps equipment passive and displays its unchanged authored policy plus separate failed evaluation. It does not replace this sandbox's existing score/object controls or enable recovered grasp.

## Try the source-linked equipment sandbox

Run this yourself from the repository root; no inference or paid call is repeated:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/source-equipment-integration-04/lab
```

Open the full token-bearing local URL printed by the command. No server was left running by the integration study.

1. In **Video**, inspect the source. Its body poses are estimates, not anatomical measurements.
2. In **Coupled 3D world**, compare the front, top and angled views. The two yellow boxes are **source-linked equipment collision proxies**, not recovered kettlebell shapes. Their image bounds come from model masks; depth, solid shape, mass and friction are assumptions. Camera controls do not change physics.
3. Select **Human** and a motor, then use manual controls. The 19 joint and three virtual root actions are explicit; the body is not demonstrating unassisted balance. **Reference** is a programmed controller; **Learned** stays disabled without a compatible trained policy.
4. Select a dynamic object to use its directional-force controls. Human/object contact markers are simulated contacts, not observed contact measurements.
5. Start recording, step or play, reset if needed, then stop/save. Actor changes are locked while recording. Exported HTML, MP4 and JSONL retain the states, actions and resets.

This candidate preserves the original body, floor, source slots and clocks. An explicit +0.2-unit horizontal coordinate translation fits the equipment inside the numerical world; it moves the entire frame together, not the body relative to the equipment. Its [migration receipt](../artifacts/source-equipment-integration-04/candidate/frame-migration.json) records the changed reset/escape envelope and eight matched-control checks.

**The exercise is not solved.** Both 60-step baselines failed the authored combined task: coast failed body tracking; reference control tracked the body but moved the working equipment, failing object stability. The body still lacks hands/grasp constraints, and the fitted starting wrist-to-box gap is about 0.182 model units. Simulated contacts are not measured hand support.

For a fully authored software fixture instead:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/combined-world-integration-02/authored-lab
```

Follow the saved stages:

- [Person/equipment tracking](../artifacts/kettlebell-shared-evidence-01/index.html): original imagery, masks and missing/pre-initialization slots.
- [Camera alternatives and projected boxes](../artifacts/equipment-camera-perspective-01/perspective-0008.png): inspect the source fit; the chosen camera is an authored hypothesis, not recovered calibration.
- [Construction assumptions](../artifacts/source-equipment-integration-04/candidate/geometry-proposal.json) and [verified source mapping](../artifacts/source-equipment-integration-04/candidate/source-binding-verification.json).
- [Interactive-world preview](../artifacts/source-equipment-integration-04/interactive-world.png).
- [Runtime and baseline report](../artifacts/source-equipment-integration-04/results.json): three exact recorded replays, one for the body and each box; each has 12 actions, one reset and 14 exported frames. Recording paths are included.

The older source-body sandbox with an authored ball/pedestal remains at `artifacts/combined-world-integration-02/source_body_authored_props-lab`; its five reference/static conflicts are preserved. It is not the equipment-proxy candidate above.

These artifact links are local-only and will not exist in a fresh clone without the saved studies.

## Build another bounded local candidate

Use new output names; completed outputs are never overwritten.

```sh
.venv-lab/bin/python -m weft shared-world-fixture --out artifacts/my-coupled-fixture
.venv-lab/bin/python -m weft shared-world-lab --run artifacts/my-coupled-fixture --out artifacts/my-coupled-lab
.venv-lab/bin/python -m weft lab --resume artifacts/my-coupled-lab
```

`shared-world-adapt --human-run <sealed-mobile-run> --out <new-run>` accepts the supported mobile `exercise-v3` input with sealed `human.json`, `scene.json` and complete `evidence.json`. It exports exact observation documents and all source slots; metadata-only pose counts cannot establish source qualification. It currently adds authored props and rejects camera-reference bodies. It does not reconstruct arbitrary equipment automatically.

For a direct camera/shape configuration or one-command human-only assembly, use [the equipment proposer](equipment-proposals.md); it no longer needs chained camera studies. For separate stage control, the constructor also consumes a sealed geometry proposal and exact equipment evidence. This older example reuses completed studies without inference or paid calls:

```sh
.venv-lab/bin/python -m weft shared-world-reconstruct \
  --run artifacts/combined-world-integration-02/source_body_authored_props-lab/run \
  --proposal-run artifacts/equipment-camera-perspective-01 \
  --evidence-run artifacts/kettlebell-shared-evidence-01 \
  --source-mapping artifacts/kettlebell-tracking-input-02 \
  --target-object sam-2 --frame-translation 0.2 0 \
  --out artifacts/my-source-equipment
.venv-lab/bin/python -m weft shared-world-lab \
  --run artifacts/my-source-equipment --out artifacts/my-source-equipment-lab
```

The frame translation is explicit and optional; omitting it for this input rejects the old bounds. Construction never silently shifts the floor, body or objects. Proposal/human inputs have separate bounded sizes; the scene remains bounded to 500 KB of compact JSON. This is an inspectable intermediate workflow, not yet one-click arbitrary-video reconstruction.

`shared-world-train --run <sealed-run> --out <new-policy-run> --generations 2 --population 4 --eval-episodes 2` is a small actual CEM learning experiment, not a guarantee of useful task success. `shared-world-lab --learning <policy-run>` verifies that the policy belongs to the scene. Training, test fixtures, source fit and physical validation remain separate evidence levels.

## Authored compound objects

The runtime accepts `shape: compound` with `rigid-compound-v1` geometry: **one object/body/control ID**, containing 1–8 local box, sphere or capsule parts, with at most 32 rigid primitives across the scene. Shape, centered diagonal inertia and assumptions are authored, not measured from a video. Part positions/rotations compose with the parent pose; capsule length denotes its straight centerline, with hemispherical ends added.

Collision, Python previews, the live front/top/angled views and offline replay use the actual transformed parts, not a replacement bounding box. Parts inherit the parent's role/color and receive no independent force controls. Invalid or excessive geometry is rejected; primitive-only scenes retain their previous behavior.

This support is verified on analytical fixtures and a source integration that represents the **same original boxes** as one-part compounds. No source handle, cavity or grasp has been recovered. The trained-policy lab remains unchanged; runtime/integration acceptance is not source fit or physical validation.

### Prepare an authored geometry alternative

`shared-world-geometry` replaces only the geometry/inertia of named existing objects in a complete sealed shared-world run. It does not fit shapes to imagery. Body/reference poses, object IDs/order/poses/velocities/mass/materials, camera/floor/clocks, source evidence and task parameters remain unchanged. Old fit reports and any policies/evaluations are archived under `source-run/`, not inherited as validation of the new shape. The alternative has a new scene identity and no installed policy.

Create a configuration like this, replacing the hash, object ID and **illustrative, unmeasured** geometry values with your intended hypothesis:

```json
{
  "schema_version": 1,
  "profile": "shared-world-geometry-hypothesis-config-v1",
  "base_scene_sha256": "REPLACE_WITH_CANONICAL_BASE_SCENE_SHA256",
  "assumptions": ["This replacement is authored; source fit is unverified."],
  "replacements": [{
    "object_id": "equipment",
    "geometry": {
      "profile": "rigid-compound-v1",
      "origin": "authored_hypothesis",
      "assumptions": ["Dimensions and centered inertia are authored priors."],
      "inertia": {"mode": "authored_centered_diagonal", "diagonal": [0.004, 0.004, 0.004]},
      "parts": [{
        "id": "body",
        "shape": "box",
        "position": [0, 0, 0],
        "quaternion": [1, 0, 0, 0],
        "size": [0.2, 0.2, 0.2]
      }]
    }
  }]
}
```

`base_scene_sha256` is the canonical hash of the validated `scene.json` object (`weft.io.canonical_hash`), **not** the raw file hash. Each unique `object_id` must already exist; replacements cannot add actors, move them or change mass. Supply complete geometry, not a patch. Parts use body-local positions and `[w,x,y,z]` unit quaternions. Boxes use full `size`; spheres use `radius`; capsules use `radius` and centerline `length` along local Z. Inertia remains an explicit centered diagonal prior, subject to rigid-body bounds; it is not computed from recovered mass distribution.

```sh
.venv-lab/bin/python -m weft shared-world-geometry \
  --run artifacts/my-base-run \
  --config /absolute/path/to/geometry-config.json \
  --out artifacts/my-geometry-01 \
  --max-seconds 30
```

Use the sealed run directory containing `manifest.json` and `scene.json` (a lab's `run/`, not its mutable outer directory). `--out` must be new and outside both inputs. `--max-seconds` defaults to **30**, accepts **1–60**, and is cooperative: it checks stages/readback/publication but cannot interrupt a native call. Configuration is bounded to 64 KiB; input/output packages to 512 files/128 MiB and 1,024 files/192 MiB respectively. No fitting, video decoding, inference or API request occurs.

Inspect `geometry-hypothesis.json` and the final manifest before using the output:

| Output | Meaning |
| --- | --- |
| `input-config.json`, `previous-geometry.json` | Exact choices and archived prior geometry/provenance |
| `initial-geometry-check.json` | Existing clearance checks at an unperturbed reset; **zero physics steps** |
| `scene.json`, `candidate.xml` | Ready alternative and compiled compound parts |
| `exercise-task.json` | Input task, if supplied, with only its scene hash rebound |
| `source-verification.json`, `source-run/` | Exact source proof and preserved original package |
| `scene.candidate.rejected.json` | Non-ready candidate when one could be formed; not a playable scene |

**Ready means initial clearance only.** Full-reference clearance, silhouette fit, contact/support, task success and physical accuracy remain unverified. A preserved exercise task keeps all controls, gates, interval/contact requirements and source scope; rebinding its scene hash also changes its task identity. Old policies/evaluations cannot be attached to the changed scene as if still current.

Feed only a ready result into the [human-only lab](shared-exercise.md#prepare-a-lab-from-a-geometry-alternative). Its sealed task is selected by default; preparation does not silently replace custom contact requirements or assistance with defaults. `silhouette-exercise-lab-01` is the current source-shaped world with its own evaluated policy. The representation-only integration below has no policy.

### Verified representation-only source example

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/shared-world-geometry-integration-01/interaction/lab
```

This saved run converts the two existing boxes to **one box part each with exact box inertia**, preserving body/source/task parameters and original video attribution. It does not change collision outlines, fit imagery or claim improved shape. Geometry inspection links the authored configuration, prior geometry and initial-clearance report; old fit panels are not reused as validation.

`artifacts/shared-world-geometry-integration-01` completes in **6.523 s**, 856 files/170,610,628 bytes before the root manifest. It verifies **69 actions, one reset, 71 frames**, resume and exact own replay. The full declared 60-step interval finishes but **fails `equipment_stability:sam-2`**. No older-baseline comparison, policy training or paid call was requested. [Report](../artifacts/shared-world-geometry-integration-01/report.json); [exact seals and scoped checks](handoffs/combined.md#current-authored-geometry-integration-checkpoint).

Both this lab and the unchanged policy lab pass actual saved-payload UI checks. This is source-custody and interaction acceptance, not a recovered kettlebell, full-video evaluation or solved exercise.

### Propose a shape from saved silhouettes

`shared-world-shape` connects saved mask evidence to a conditional solid-shape proposal, initial admission and a separate full-reference collision comparison. A **mask** is the inferred pixel region belonging to an object. It does not measure hidden depth or empty space.

```sh
.venv-lab/bin/python -m weft shared-world-shape \
  --run artifacts/source-equipment-integration-04/candidate \
  --masks-run artifacts/kettlebell-shared-evidence-01 \
  --task artifacts/shared-exercise-human-jitter-01/training/exercise-task.json \
  --object-id sam-2 --object-id sam-3 \
  --source-frame 8 --conditional-exterior \
  --out artifacts/my-silhouette-01
```

The explicit `--conditional-exterior` option permits clear silhouette rows to narrow up to eight solid horizontal bands per object. Camera, depth, object pose/mass/materials, body motion and original box inertia remain fixed assumptions. Other foreground, uncertain rows and empty rows retain the original envelope; holes are filled, never interpreted as recovered handles. The method makes one proposal per requested object and combines all image-qualified proposals once. It does not search camera settings, contact outcomes, frames or subsets until something passes.

| Output | Inspect |
| --- | --- |
| `shape-comparison.html` | Offline before/after silhouette outlines, uncertainty and separate fit/collision results |
| `objects/<ID>/result.json` | Original and proposed geometry, native-cell fit, missing/unknown slots and rejected/no-improvement outcomes |
| `settings.json`, `inputs/`, `geometry-input/` | Frozen rule, source identities/clocks, selected masks and exact baseline/task |
| `geometry/` | Separate initially admitted candidate, if one qualified; earlier inputs remain unchanged |
| `clearance-comparison.json` | Every source slot/reference, full-span 30 Hz grid and declared task-control boundary |
| `report.json`, `manifest.json` | Stage outcomes and file integrity; completion is not a physical-success label |

The collision comparison fixes equipment at its declared pose and examines every human/rigid-part, rigid-part/floor and human/floor pair. It does not run physics or establish grasp, stability, continuous-time clearance, self-collision or rigid–rigid clearance. Missing source/reference coverage stays unavailable; interpolated grid samples are labeled separately. Every positive overlap increase remains in the report, including numerical-scale changes.

Use a new output directory. Bounds are four requested objects, 128 MiB/2,048 input files, 384 MiB/2,048 output files and a default **60-second cooperative limit**, with bounded failure preservation. No API, inference, video decode or training runs. A ready `geometry/` may be inspected in a new lab only after reviewing its separate reports; old policies and source-fit panels do not validate the changed scene. The original task keeps its interval and gates, with only the scene identity rebound.

#### Verified source-shaped world

The saved [before/after report](../artifacts/silhouette-shared-world-03/shape-comparison.html) retains the occluded working object unchanged. The idle object's conditional silhouette overlap improves **0.593595 → 0.666873** with eight solid bands. It still covers **1,075/1,130** foreground cells: 55 were already outside the original projection, and none are newly lost. Depth, inertia and hidden shape remain assumptions; these are not recovered handles.

The separate full-reference comparison retains **696 scheduled samples: 558 checked and 138 unavailable**, with no increased or new overlap. Existing worst overlap remains **0.104829 model units**; this does not resolve later body/equipment contact. Earlier loader and floating-point construction failures remain preserved beside the completed run.

`artifacts/silhouette-exercise-control-01` then trains and selects a policy for this **new scene identity**, with no mask, camera, task or gate changes. On three fresh starts, selected control passes **3/3**, while coast/random/authored control each pass **0/3**. The [current playable lab](shared-exercise.md#try-it), `artifacts/silhouette-exercise-lab-01`, verifies manual/controller interaction, reset, **68 actions/70 recorded frames**, and exact replay. Its two-second root-assisted task requires tracking and equipment stability, not grasp or full-exercise completion. Old-world policies and fit panels were not carried over as validation.

## Evidence and runtime boundaries

- `shared_world_evidence.py` uses `world_evidence.py` to bind exact document SHA256s and actual entity IDs. Foreign media require a verified original/derivative mapping. Conflicting canonical-frame partitions reject across documents, entities and aliases; an empty unsampled slot remains withheld but does not reserve an unseen measurement as a holdout.
- Raw pose outputs retain inferred status and their original assumed frame/FPS clock. Missing/unsampled/withheld values are not converted into measurements. Authored objects have null observation references and zero observed slots.
- Packaging, lab loading and training verify pinned documents. Outer manifests must cover every required observation, mapping manifest and bundle child. Policy outputs preserve complete source evidence; link-safe copying and final verification reject missing files, altered bytes and collisions. Aggregate admission is bounded to 64 MiB/256 pinned files and a 30-second preflight deadline, not an end-to-end runtime promise.
- The physics core rejects unsupported calibration claims and initial cross-entity intersections. Reference inspection uses separate simulation state, and required human contact must involve the task's target object.
- Force/torque units derive from the declared uncalibrated scene units; the interface does not relabel them measured newtons or metres. Muscles remain illustrative command-intensity bands.

## Source equivalence and remaining physical gaps

One SAM request tracked a person and two kettlebells across all 257 frames. The first eight frames have no initialized objects: all **24 pre-initialization slots** remain. All **747 later masks** decoded; this establishes complete returned proposals, not correct identity or physical shape. The conservative unresolved identity group expands to all three proposals at frame 139. Hand/equipment overlap is not measured contact.

The input preparation preserves every native YUV pixel and timestamp. The original WebM and lossless MP4 both have native timestamps 3–8536 ms; their decoder-relative clocks are 0–8533 ms. The explicit 3 ms origin relationship is retained. Packet/frame durations differ on 86 frames and are recorded separately.

The SAM observation document identifies the submitted MP4; body observations identify the original WebM. **Neither hash is replaced.** `source_equivalence.py` checks all sealed recorded pixel/frame/clock evidence before admitting the alias. It performs no new decode and cannot authenticate a producer or establish physical truth. Pose frame/FPS clocks remain assumed; they are not silently converted into observed decoder timestamps.

The current proposal improves eligible source/body projection RMSE from 16.66 to 11.09 pixels. Box edges stay within two pixels of later mask bounds, but median silhouette overlap is only 0.471/0.573: solid boxes do not reconstruct handles or cavities. Unseen depth, material properties, body/equipment identity and actual support remain unverified. Reconstruction uses zero object-force channels; manual object forcing is a separate authored intervention.

The request settled at **$0.0498**; the shared Meta ledger then held nine settled requests, **$0.235952 of $5**, zero unsettled. This is a historical cost snapshot, not permission to create another budget. Price reference: [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits), checked October 29, 2026.

## Verification scope

The earlier construction checkpoint covers **285 distinct passing cases**; three source-equipment recordings verify exact replay. Candidate construction took **1.016 s**, lab packaging **0.720 s**, and local three-step control dispatch plus JSON had p95 **3.923 ms** across 30 samples.

Current reset/runtime integration has **143 passing cases** (`.local/shared-exercise-reset-integration-04.xml`) plus six study-driver checks. Five analytical JS checks cover compound transforms, outlines, small parts, rejection and the 32-part bound; exercise and legacy replay checks retain lifecycle/task behavior. The new human-only source lab verifies 68 actions/one reset/70 frames and exact replay while preserving failure. [Current receipts and identities](handoffs/combined.md#current-reset-assembly-and-compound-checkpoint). Counts overlap with earlier targeted checks and must not be added into a repository-wide total. These are scoped local measurements, not browser latency or sustained-use guarantees.

No live browser/device QA, independently measured biomechanics, recovered equipment geometry, physical changed-action validation or general arbitrary-video completion is claimed.
