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
  --full-reference \
  --out artifacts/my-shared-exercise-01

.venv-lab/bin/python -m weft lab --resume artifacts/my-shared-exercise-01/lab
```

The first command prints stage progress and does not start a server. The second command is user-started. Add `--root-assistance` to assembly only if you want the three explicit virtual root controls; otherwise it uses 19 human joint controls. Equipment remains passive in both modes.

`--full-reference` builds one continuous episode from the first retained body pose to the last. It uses the existing `shared-world-v2` contract: at most 1,800 actions / 60 seconds, unchanged timestep and no chunk resets. Missing poses and source prefixes stay missing; the selected interval is not necessarily the entire original video. Unsupported endpoints reject instead of being rounded or retimed. Omit the flag to keep the original short-window behavior.

The full-reference choice is made before task/lab packaging, so it creates only one candidate and one lab. It does not attach a short-window policy, run training, change success limits or require a temporary lab. `candidate/reconstruction.json` records the original window and the selected interval; `lab/run/exercise-task.json` binds the new task to that exact scene.

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

For separate stage control, use `shared-world-propose`, then `shared-world-reconstruct --full-reference`, then `shared-exercise-lab`. The proposal report includes the exact construction arguments. The reconstruction flag is optional and has the same interval semantics. Use [human-only training/evaluation](shared-exercise.md) separately; a ready geometry package is not a successful policy.

## Verified scope

**Full-reference assembly:** `artifacts/full-reference-assembly-01` runs the public command with `--full-reference` in **3.148 s** (3.407 s including CLI startup), producing **374 files / 88,686,504 bytes including its manifest**, seal `2b7a5954682099a6a06216362f24ad5df6b475cf265bf10b1b2d8f3534ad1965`. All physical settings and source documents match the earlier box assembly; only the scene profile and task interval change to 254 continuous actions. All 121 poses / 257 slots remain. Three reference-control actions and one reset work through the lab API; the sealed run stays unchanged. This new lab is **untrained**, not a replacement for the trained example above. Root verification: `.local/full-reference-assembly-verification-01.json`. No inference, fitting, paid calls, server or browser was used.

**One-command acceptance:** `artifacts/shared-exercise-assembly-01` completes the public assembler on the existing 257-frame kettlebell evidence in **3.225 s**, within its 90-second bound. Its 362 files total about 88.3 MB; root seal `4028bb4e36176252d4eb3de4f14b9c81839fc979192a17289583ba37b0d5c931`. The prepared lab is `artifacts/shared-exercise-assembly-01/lab`. Independent verification checks all 361 sealed members; the root manifest is the remaining file. No inference, fitting, training or API call was needed. The existing 31 assembly/bounds checks remain passing.

The earlier `artifacts/shared-equipment-proposal-integration-01` verifies the three stage commands separately. All physical scene settings match the previous candidate exactly; only equipment provenance differs. Proposal preparation took 0.805 s and lab preparation 0.900 s. Its 365 files total 88.15 MB. The overall run took 131.1 s including a source-write coordination wait, missing its separately declared 120-second wall bound; that overrun remains recorded. No stage was retried, and no inference, fitting or paid call was repeated.

The proposer still uses **`floor_box_depth_equals_width`**, and these source scenes remain solid boxes. [Compound collision/rendering](shared-world.md#authored-compound-objects) is separately tested with analytical shapes; it does not infer source handles or admit compound equipment through this configuration. Strict source-compound proposal admission is in progress.

These earlier integration results use a two-second task on an exposed development clip, not a new camera calibration or generalization result. The full-reference option removes that interval restriction; unknown depth, approximate shape, body–equipment overlap and missing hands/grasp remain. Broader-video automation and independent physical accuracy are still open.
