# Coupled human and object sandbox

Weft can now place a mobile articulated body and rigid objects in **one MuJoCo simulation**, with one clock and contact solver. You can control the body or a dynamic object, inspect three views, and record/replay actions. This is a runtime capability, not proof that equipment geometry, forces or anatomy were recovered correctly from a video.

For separate whole-interval body/equipment checks with **human-only actions**, see [exercise tasks and recordings](shared-exercise.md). That task does not replace this sandbox's existing score or enable recovered grasp.

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

The reusable next stage consumes a sealed geometry proposal and exact equipment evidence. This example reuses completed studies without inference or paid calls:

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

Latest targeted receipts cover **285 distinct passing cases**; see the [combined handoff](handoffs/combined.md) for scope and identities. Three new source-equipment recordings verify exact replay. New-candidate construction took **1.016 s**, lab packaging **0.720 s**, and local three-step control dispatch plus JSON had p95 **3.923 ms** across 30 samples. These are scoped local measurements, not browser latency or sustained-use guarantees.

No live browser/device QA, independently measured biomechanics, recovered equipment geometry, physical changed-action validation or general arbitrary-video completion is claimed.
