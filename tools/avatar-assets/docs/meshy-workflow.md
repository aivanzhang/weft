# Repeatable generated-avatar workflow

Implemented on2026-10-31 in `tools/avatar-assets`. Meshy generates an actual mesh, textures and optional humanoid skin; local tools inspect, preserve and finish it. Quality is reviewed separately from provider/validator success. This is not Weft's importer, simulator, physics or replay implementation.

## Inputs and presets

Copy a config to a **new ID/path**. `examples/meshy/muse-panda-v1.json` is an image-conditioned example; `workshop-robot-v1.json` is genuinely text-only. Both pin Meshy7.1 and request4K geometry/textures. No subject is hard-coded in the backend.

Required fields: subject, style, exact prompt, rights/provenance, generation mode/resolution/target triangles/rest pose, authored height, rig policy, named state prompts/durations and a finite credit ceiling. Image mode takes one selected local PNG/JPEG/WebP≤8MiB/16MP with an exact SHA256; it sends those bytes only. Text mode sends no files. Multiple-reference conversion is unsupported in this bounded adapter; don't silently upload a folder.

- `rig: humanoid`: Meshy auto-rig plus optional2–10s text motions/retargets, up to8 states.
- `rig: none`: generated geometry for creatures/objects; no automatic animation claim. Use the existing local scene route or an explicit finishing/rig-authoring step for non-humanoid articulation.
- A-pose/T-pose are requests, not guarantees. Inspect actual output. Target triangle count is not a vertex budget.

## Paid generation and resumption

From `tools/avatar-assets`, with the already designated owner-only Weft-root key `.local/credentials/meshy-api-key`:

```sh
.venv/bin/python -m avatar_assets meshy account
.venv/bin/python -m avatar_assets meshy ledger
.venv/bin/python -m avatar_assets meshy plan examples/meshy/workshop-robot-v1.json --out outputs/my-new-robot-job
.venv/bin/python -m avatar_assets meshy run outputs/my-new-robot-job --max-seconds 900
```

The example config ID must also be changed for a genuinely new job. A ledger ID cannot be rebound to another folder/plan. `plan` is local/free. `run` makes paid submissions only for missing configured stages; repeat it on the **same folder** to resume. It does not automatically regenerate failed/poor outputs. A `checkpointed` result is unfinished work with a saved task ID, not a completed avatar. The watch window is bounded; an already-started request/download retains its separate deadline, so this is not an instantaneous process-kill deadline.

Individual controls remain available:

```sh
.venv/bin/python -m avatar_assets meshy status outputs/my-new-robot-job --stage mesh
.venv/bin/python -m avatar_assets meshy fetch outputs/my-new-robot-job --stage mesh
.venv/bin/python -m avatar_assets meshy submit outputs/my-new-robot-job --stage texture
```

Unknown/lost POST responses preserve the reservation and stop resubmission. Reconcile the existing task before continuing; do not clear the ledger. A failed partial download is retained and stops automatic overwriting. Provider URLs can expire; do not manufacture a replacement URL. Status polling is capped120/stage, at least15s apart; terminal responses are cached.

### Cost of this completed provider trial

Official pricing checked2026-10-31: [pricing](https://docs.meshy.ai/en/api/pricing), [text-to-motion](https://docs.meshy.ai/en/api/text-to-motion), [animation retargeting](https://docs.meshy.ai/en/api/animation).

| Work | Credits |
| --- | ---: |
| Panda image→4K textured mesh |35|
| Panda humanoid rig |5|
| Four Prime motions plus four retargets |52|
| Robot text mesh + texture + rig |40|
| **Total13 paid submissions, all settled** |**132**|

The independent journal `.local/meshy/ledger.sqlite3` retains the original140-credit base ceiling. The user subsequently approved35 additional credits for the standalone dragon: one25-credit mesh and one10-credit texture, both completed. The append-only grant is bound to that job and its plan; it cannot fund unrelated jobs. Aggregate authorization is175, actual reported/reserved spend167 across15 submissions, with no unknown submissions or pricing overruns. The earlier132 includes rejected panda motions; refunds do not reopen allowance. Do not reset history or invent a new allowance from the account balance. USD purchase price is not exposed by the account endpoint, so **dollar cost is unknown**, not zero. No new account, subscription, paid compute/storage service or Meta-budget use.

Only after explicit additional spending authorization, record `decision: approved`, the exact `user_reply`, `job_id`, canonical `plan_digest` and `additional_credits` equal to the prepared job limit. `examples/meshy/emerald-dragon-approval-v1.json` preserves the actual authorization. Apply it with `meshy authorize JOB --approval-record PATH`. This local command is idempotent for identical approval; changed grants, mismatched plans and retroactive grants are rejected. It does not buy credits or submit generation. Local finishing/reloads/browser checks make no provider calls.

## Local finishing and verification

Originals stay untouched. Normalize only if split normals/UVs push a model above the100k local profile:

```sh
.venv/bin/python -m avatar_assets.normalize_generated outputs/JOB/stages/rig/downloads/model.glb --out outputs/rig-clean-v1
.venv/bin/python -m avatar_assets.provider_model outputs/rig-clean-v1/model.glb --out outputs/rig-preview-v1
```

Cleanup averages normals at coincident positions and preserves geometry positions, UV seams, weights, clips and texture bytes. It accepts at most200k input vertices/100k triangles and still requires≤100k output vertices. It is a recorded authored derivative, not recovery of the original unseen model.

`assemble_generated` takes a hash-pinned source and named source clips. Exact named rest hierarchies must match. It rejects constant clips and incompatible rigs rather than pretending to retarget them:

```sh
.venv/bin/python -m avatar_assets.assemble_generated examples/meshy/muse-panda-assemble-v1.json --out outputs/panda-assembly-next
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/workshop-robot-finish-v1.json --out outputs/robot-finish-next
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/muse-panda-finish-v4.json --out outputs/panda-finish-next
.venv/bin/python -m avatar_assets reload3d outputs/panda-finish-next --out outputs/panda-reload-next
.venv/bin/python -m avatar_assets verify outputs/panda-finish-next
```

These finishing examples use their pinned local source paths. Change the source/hash and tailor the design for a new subject; do not call changing only the ID a new character. Finishing accepts one bounded textured mesh with an existing skin, or an unrigged mesh with explicit `authored_rig`. Its JSON controls optional smoothing, normalized spatial masks, constrained bone weights, a rigid accessory anchor, native hair, brightness-banded portable fur, and explicit local gesture curves. The script contains no character identity constants. Configuration authoring and final visual review remain assisted steps—not guaranteed automatic production-quality rig repair.

### Locally authored creature rigs

The [dragon recipe](../examples/meshy/emerald-dragon-finish-v3.json) exercises this route with an actual text-generated mesh, not an analytic fixture. Meshy supplied shape and textures only; our local code constructs the19-bone skeleton, bounded skin weights and four real skeletal clips. See [delivery and limitations](dragon-delivery-v1.md).

`authored_rig` requires these fields:

| Field | Meaning and bounds |
| --- | --- |
| `space` | `normalized_bounds_blender_z_up`: positions relative to the actual mesh bounds after native import |
| `height_meters` | Authored final height,0.05–10m; uniform scale, centered horizontally and grounded at minimum Z |
| `bones` |1–64 unique named bones, one root, parents before children; `head`/`tail` finite triples in[-1,2], nonzero lengths |
| `base_bones` | Named controls used for initial nearest-segment blending |
| `layers` | Up to32 ordered spatial masks with `bones`, `min`, `max`, `feather`0–0.3 and `min_vertices`1–100000 |

Later layers blend into earlier weights. Distance is measured to finite bone segments; only four influences survive and are normalized. Coincident UV-seam positions receive equal weights. A missed minimum-coverage mask fails rather than silently leaving a limb unbound. This is a bounded geometric weighting method—not an anatomical detector, collision solver or universal auto-rigger.

`authored_actions` supplies2–32 time keys per state, Euler rotation triples in degrees within±60, using only configured bone names. Each state retains its own prompt/meaning; curves are baked into real GLB animations. Configure `camera_view` as `front`, `three_quarter` or `three_quarter_left` to reveal tails/accessories; explicit views fit projected bounds. Inspect the deformed mesh at action peaks before full media rendering. The dragon's original arm-raise celebration distorted its belly; the preserved third version uses a bow/wing/tail flourish instead. Large arm articulation remains unverified.

Rebuild locally from the preserved raw model, without provider calls:

```sh
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/emerald-dragon-finish-v3.json --out outputs/dragon-finish-next
.venv/bin/python -m avatar_assets reload3d outputs/dragon-finish-next --out outputs/dragon-reload-next
.venv/bin/python -m avatar_assets export2d outputs/dragon-finish-next --preset preview --size 720 --portrait 1600 --seconds 4 --samples 8 --denoise-portrait --out outputs/dragon-media-next
.venv/bin/python -m avatar_assets verify outputs/dragon-media-next
```

For a genuinely new subject, first create a new generation config/ID and prepare its local plan; inspect the resulting geometry and author new bone positions/masks/actions. Reusing the pipeline is supported; blindly reusing dragon anatomy for a different creature is not. An object can instead use the existing procedural scene route for precise hinges and controls.

The optional dense native profile is capped600k configured parent/child fibers, with the same600s/2GiB/512MiB job limits. The90-child trial exceeded2GiB and is rejected; the current45-child/299k-fiber panda completed below the guard. Browser cards budget12 exported triangle corners per ribbon because tangents can split vertices. Up to16 zero/nonorthogonal tangents and at most0.1% of vertices may receive an explicit orthogonalization/fallback; the untouched native export and exact repaired indices remain separate. Other invalid tangents fail.

### Outputs

`model.glb`, `source.blend`, `portrait.png`, five sampled PNG poses/state with first-pose static fallback, named real skeletal clips, `rig-contract.json`, `exported-nodes.json`, skin/motion/attribute proofs, exact design/backend source snapshots, native job/log/resource reports and an artifact-hashed manifest. Existing folders are never overwritten. The GLB has texture-alpha masks for cards; the portrait is opaque. No seamless loop is certified. Native strands are editable in Blender; browser cards are a separate approximation, not strand export or simulated hair.

The raw Meshy rig is allowed to contain a constant clip, but that is never advertised as animation. Robot received explicit local actions. Panda's paid generated motions are preserved but rejected: bad provider weights assigned16,185 predominantly weighted vertices to `RightHand`, including much of the belly. Local weight constraints and smaller authored gestures are explicitly distinguished from those rejected clips. Crafting/working are currently pantomimes without a visible workpiece or articulated fingers—not completed tool interaction.

## Bounds and provenance

- Network:45s JSON request deadline,2MiB responses; reference8MiB; downloads64MiB/file and120s each; no redirects, no credentials sent to asset hosts. At most three recognized downloads/stage. Poll/call/state counts and aggregate512MiB raw-job output are bounded. No automatic paid retries.
- Native: disabled script auto-execution, embedded local GLB only, no remote/data URLs or compressed/morph features;≤100k vertices/100k triangles, finite attributes,16MP/texture. One job uses≤4CPU threads; at most two heavy jobs ran concurrently. Wall/memory/output monitoring samples every250ms/about2s; it is not an instantaneous OS/descendant-memory cap. Cleanup is separately bounded7s.
- Public receipts strip signed URL query strings. Full provider responses remain only in the owner-only ledger. Exact requests are reconstructable from preserved prompt/config/reference bytes and hashes; the key never belongs in a manifest.
- Generated hidden surfaces, rest pose, scale and skeleton are assumptions. Original reference media lack an independent license/source-model verification. Metadata is provenance, not rights clearance. Do not upload unrelated files or borrow sibling-project credentials.
- No byte-identical generative rerun guarantee. Paid regeneration is stochastic; local manifests/configs/hashes make existing results auditable. Blender exports can vary in floating-point details/index order. Reuse hashed originals rather than needlessly regenerating.

## Acceptance and unsupported results

Current generated models remain candidates below the Muse target. Inspect face/proportions/fur at ordinary display size and close up, state meaning, feet, shoulder attachment and motion clipping. A validator proves structure and sampled movement—not likeness, good deformation, independent anatomy or human approval.

The HD models exceed Weft's frozen20k-vertex/40k-triangle/diffuse profile; no faithful unchanged conversion or new mapping is claimed. Keep the earlier matte Weft bundles. Do not duplicate its importer or change physics/actions/rewards. A separate optimized version needs a new mapping, explicit fidelity losses and consumer validation.

Live ownership, exact task IDs, current candidate paths, costs, measurements, rejected attempts and next action are in [TODO](../TODO.md). No task-owned persistent gallery server is left running; users start it themselves.

## Media exports and bounded continuation

`export2d` now accepts generated packs as well as authored scenes. It renders actual complete skeletal curves, not repeated stills. For the current four-second previews:

```sh
.venv/bin/python -m avatar_assets export2d outputs/workshop-robot-meshy-refined-v2 --preset preview --size 720 --portrait 1600 --seconds 4 --samples 8 --out outputs/robot-media-next
```

The Muse preset remains1600px portrait/720px/24fps/eight-second clips; shorter previews must keep their actual duration labels. Native media has a1200s/2GiB/512MiB worker guard. A time-limited generated render with completed, hash-bound frames can be continued **once**, using a new output and `--resume-frames PREVIOUS_FAILED_FOLDER`; the extra worker is capped600s (aggregate1800s). SourceBLEND/manifest/scene/spec must match. No second continuation, no memory-failure replay, no changed-input reuse, and no missing frame is called complete. Partial files remain in the failed source; valid PNGs are copied byte-for-byte.

Completed media includes native frames, silentH.264 videos, exact first-frame PNG fallbacks, contact sheets, original/optionally filtered portrait and hashes. Identical endpoints are reported separately from seamlessness; no velocity-continuous loop is certified.

### Measured example exports

- Panda: `outputs/muse-panda-meshy-media-v2`. The original `-v1` was stopped at1200.074s with365 completed video frames; the one continuation reused those plus the portrait and rendered the missing23 frames in67.726s. Aggregate native wall time1267.800s; original peak2,079,604,736B and continuation peak1,707,753,472B, sampled direct-child RSS. The original failed output remains unchanged.
- Robot: `outputs/workshop-robot-meshy-media-v1`, native612.815s and1,199,587,328B sampled peak. No continuation needed.
- Both:1600px opaque portrait, four720px silentH.264 clips,96frames each at24fps/four seconds. Per-state static PNGs are exact copies of the first native frame. All96 decoded thumbnails differ in each clip; this proves changing images, not meaningful tasks or seamlessness.
- Compact `*-portable-v1` derivatives retain identical portrait/video/fallback bytes and source-manifest hashes, omitting only raw frames and verbose render logs. They occupy about11.45MB for panda and7.52MB for robot; omitted data remain in the full packs. Catalog: `browser/gallery-meshy-v2.json`.

No further provider charge occurred during finishing, media rendering, packaging or browser verification. The panda still has visible coat/dragon artifacts and a flatter muzzle; a1600px export is not Muse-quality acceptance. Both crafting/working gestures need actual task interaction before semantic acceptance.
