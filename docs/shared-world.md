# Coupled human and object sandbox

Weft can now place a mobile articulated body and rigid objects in **one MuJoCo simulation**, with one clock and contact solver. You can control the body or a dynamic object, inspect three views, and record/replay actions. This is a runtime capability, not proof that equipment geometry, forces or anatomy were recovered correctly from a video.

## Try the saved local sandbox

Run this yourself from the repository root; no inference or paid call is repeated:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/combined-world-integration-02/source_body_authored_props-lab
```

Open the full token-bearing local URL printed by the command. No server was left running by the integration study.

1. In **Video**, inspect the source. Its body poses are estimates, not anatomical measurements.
2. In **Coupled 3D world**, compare the front, top and angled views. Camera controls do not change physics. The ball and pedestal are **authored props**, not the kettlebells from this video.
3. Select **Human** and a motor, then use manual controls. The 19 joint and three virtual root actions are explicit; the body is not demonstrating unassisted balance. **Reference** is a programmed controller; **Learned** stays disabled without a compatible trained policy.
4. Select a dynamic object to use its directional-force controls. Human/object contact markers are simulated contacts, not observed contact measurements.
5. Start recording, step or play, reset if needed, then stop/save. Actor changes are locked while recording. Exported HTML, MP4 and JSONL retain the states, actions and resets.

The source-linked candidate has **five reference/static conflict checks** (two source poses and three control-grid poses). Keep the warning visible. A playable manual sandbox is not a passed physical reconstruction or a successful learned exercise.

For a fully authored software fixture instead:

```sh
.venv-lab/bin/python -m weft lab --resume artifacts/combined-world-integration-02/authored-lab
```

Saved evidence and previews:

- [Integration/recording report](../artifacts/combined-world-inspection-03/report.json): four exact replays, each with 12 actions, one reset and 14 exported frames.
- [Source-linked body preview](../artifacts/combined-world-inspection-03/source_body_authored_props-human.png).
- [Same-source person/equipment tracking inspection](../artifacts/kettlebell-shared-evidence-01/index.html): separate evidence workflow; not yet used as the sandbox's rigid geometry.

These artifact links are local-only and will not exist in a fresh clone without the saved studies.

## Build another bounded local candidate

Use new output names; completed outputs are never overwritten.

```sh
.venv-lab/bin/python -m weft shared-world-fixture --out artifacts/my-coupled-fixture
.venv-lab/bin/python -m weft shared-world-lab --run artifacts/my-coupled-fixture --out artifacts/my-coupled-lab
.venv-lab/bin/python -m weft lab --resume artifacts/my-coupled-lab
```

`shared-world-adapt --human-run <sealed-mobile-run> --out <new-run>` accepts the supported mobile `exercise-v3` input with sealed `human.json`, `scene.json` and complete `evidence.json`. It exports exact observation documents and all source slots; metadata-only pose counts cannot establish source qualification. It currently adds authored props and rejects camera-reference bodies. It does not reconstruct arbitrary equipment automatically.

`shared-world-train --run <sealed-run> --out <new-policy-run> --generations 2 --population 4 --eval-episodes 2` is a small actual CEM learning experiment, not a guarantee of useful task success. `shared-world-lab --learning <policy-run>` verifies that the policy belongs to the scene. Training, test fixtures, source fit and physical validation remain separate evidence levels.

## Evidence and runtime boundaries

- `shared_world_evidence.py` uses `world_evidence.py` to bind exact document SHA256s and actual entity IDs. It recomputes counts from complete construction-only slots and rejects foreign sources, missing documents, altered bytes, invented IDs and unsafe paths.
- Raw pose outputs retain inferred status and their original assumed frame/FPS clock. Missing/unsampled/withheld values are not converted into measurements. Authored objects have null observation references and zero observed slots.
- Packaging, lab loading and training verify pinned documents. Policy outputs preserve their required evidence, so copied policies do not become metadata-only source claims.
- The physics core rejects unsupported calibration claims and initial cross-entity intersections. Reference inspection uses separate simulation state, and required human contact must involve the task's target object.
- Force/torque units derive from the declared uncalibrated scene units; the interface does not relabel them measured newtons or metres. Muscles remain illustrative command-intensity bands.

## Same-source tracking result and remaining bridge

One SAM request tracked a person and two kettlebells across all 257 frames. The first eight frames have no initialized objects: all **24 pre-initialization slots** remain. All **747 later masks** decoded; this establishes complete returned proposals, not correct identity or physical shape. The conservative unresolved identity group expands to all three proposals at frame 139. Hand/equipment overlap is not measured contact.

The input preparation preserves every native YUV pixel and timestamp. The original WebM and lossless MP4 both have native timestamps 3–8536 ms; their decoder-relative clocks are 0–8533 ms. The explicit 3 ms origin relationship is retained. Packet/frame durations differ on 86 frames and are recorded separately.

The SAM observation document correctly identifies the submitted MP4, while body observations identify the original WebM. Both identities and their complete mapping remain preserved. **Do not replace one hash with the other.** The next construction step needs a validated original/derivative equivalence and clock bridge, then uncertainty-aware person/equipment geometry and contact hypotheses. The current same-source binder deliberately rejects different source hashes until that bridge is implemented.

The request settled at **$0.0498**; the shared Meta ledger then held nine settled requests, **$0.235952 of $5**, zero unsettled. This is a historical cost snapshot, not permission to create another budget. Price reference: [Meta pricing](https://dev.meta.ai/docs/pricing-rate-limits), checked October 29, 2026.

## Verification scope

The checkpoint includes source-binding, mixed-physics, recovery, HTTP and mocked UI tests; see the [combined handoff](handoffs/combined.md) for current counts and exact receipts. Four saved recordings verify exact state replay. Fresh three-step-plus-snapshot core timings were p95 **2.643 ms** for the authored fixture and **1.654 ms** for the source body (30 samples each). These are not browser latency or sustained-use measurements.

No live browser/device QA, independently measured biomechanics, recovered equipment geometry, physical changed-action validation or general arbitrary-video completion is claimed.
