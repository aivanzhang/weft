# Build an exercise scene from saved evidence

Use a verified mobile-body run and a sealed equipment tracking export to build **one shared world and a human-only lab**. This path no longer requires chained camera-study directories. It still requires explicit camera/shape assumptions; it does not automatically recover arbitrary equipment from a video.

For an immediately playable example, use the [exercise sandbox](shared-exercise.md). For a new video, first prepare its mobile body and tracking evidence using the [shared-world guide](shared-world.md). Keep the original video hash and any original/derivative mapping bundles.

## One assembly command

Create a JSON configuration with these fields, using your own original video hash, observation IDs and original source-frame numbers:

```json
{
  "schema_version": 1,
  "profile": "source-equipment-proposal-config-v1",
  "source_sha256": "REPLACE_WITH_ORIGINAL_VIDEO_SHA256",
  "camera": {
    "mode": "fixed_pelvis_perspective_prior",
    "distance_model_units": 2.0
  },
  "geometry_prior": "floor_box_depth_equals_width",
  "equipment": [{"object_id": "YOUR_EQUIPMENT_ID", "source_frame": 12}],
  "target_object_id": "YOUR_EQUIPMENT_ID"
}
```

The distance above is an **authored example**, not a measured camera distance. The current shape prior makes solid boxes resting on the existing floor, with unseen depth equal to width. It rejects incompatible projections, missing/withheld selected observations and initial collisions. It does not fill missing frames, move the floor or search for a different camera to force admission.

```sh
.venv-lab/bin/python -m weft shared-exercise-build \
  --run artifacts/my-shared-body \
  --evidence-run artifacts/my-tracking-evidence \
  --config /absolute/path/to/equipment-config.json \
  --out artifacts/my-shared-exercise-01

.venv-lab/bin/python -m weft lab --resume artifacts/my-shared-exercise-01/lab
```

The first command prints stage progress and does not start a server. The second command is user-started. Add `--root-assistance` to assembly only if you want the three explicit virtual root controls; otherwise it uses 19 human joint controls. Equipment remains passive in both modes.

If tracking used a verified derivative, add `--source-mapping <bundle>`; repeat for distinct derivatives. Canonical root `scene-observations.json` and the older nested layout are supported without moving files. Conflicting sealed copies reject.

An optional `"frame_translation_xy": [0.2, 0.0]` explicitly translates the entire frame and invokes the existing bounded nominal dynamics check. It is not a body-only adjustment; it changes numerical bounds and must not be described as complete task equivalence. Omit it unless that coordinate migration is intended. The assembly command passes the resolved choice to construction; it does not invent or apply a second translation.

## Inspect each stage

| Output | Meaning |
| --- | --- |
| `settings.json`, `stages.json`, `result.json` | Requested inputs, completed stages and outcome |
| `proposal/input-config.json` | Exact user choices |
| `proposal/report.json` | Resolved assumptions, exact source bindings and admission/rejection reasons |
| `proposal/geometry-proposal.json` | Initially admitted proposal; absent on rejection |
| `proposal/geometry-candidate.rejected.json` | Preserved non-ready geometry, when a candidate could be formed |
| `candidate/` | Standard shared-world construction, complete source proof and existing reference checks |
| `lab/run/` | Immutable task, scene, evidence and runtime snapshot |
| `lab/sessions/` | New interaction recordings, separate from construction evidence |

Every output directory must be new. Unsupported input stops before construction or lab creation. Later failures preserve completed stages. If a lab was already successfully prepared before an outer deadline or finalization failure, the failed assembly result explicitly retains `lab_status: ready` and its launch path; it never relabels that completed lab. A failure-only manifest identifies unsealed partial outputs rather than pretending they were verified.

The default assembly deadline is 90 seconds, checked around stages and during final hashing. Native calls and bounded source-verification calls already underway cannot be preempted. Each next stage reserves its complete output allowance before starting: proposal 256 files/64 MiB, construction 512 files/128 MiB, lab 1,024 files/192 MiB, within the shared 1,600-file/384 MiB envelope. No inference, training, provider call or deployment occurs in assembly.

For separate stage control, use `shared-world-propose`, then `shared-world-reconstruct`, then `shared-exercise-lab`. The proposal report includes the exact construction arguments. Use [human-only training/evaluation](shared-exercise.md) separately; a ready geometry package is not a successful policy.

## Verified scope

`artifacts/shared-equipment-proposal-integration-01` verifies the three public stage commands on the existing 257-frame kettlebell evidence. All physical scene settings match the previous candidate exactly; only equipment provenance differs. Proposal preparation took 0.805 s and lab preparation 0.900 s. Its 365 files total 88.15 MB. The overall run took 131.1 s including a source-write coordination wait, missing its separately declared 120-second wall bound; that overrun remains recorded. No stage was retried, and no inference, fitting or paid call was repeated.

This is integration evidence for an exposed development clip, not a new camera calibration or generalization result. Unknown depth, approximate shape, body–equipment overlap, missing hands/grasp and the current two-second exercise interval remain limitations. Broader-video automation and independent physical accuracy are still open.
