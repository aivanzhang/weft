# Emerald dragon delivery

Standalone text-generated dragon with a locally authored creature rig and four skeletal motions. Meshy supplied the mesh and textures only. Its green scales, amber eyes, cream belly plates, orange horns/spines, paired wings and long tail remain consistent across all states. No reference image was uploaded and no fur was requested for this scaled character.

## View and edit

From Weft:

```sh
./tools/avatar-assets/run gallery --catalog browser/gallery-meshy-v3.json
```

Open `http://127.0.0.1:8765/` yourself. The dragon is first; earlier panda/robot/reference entries remain. Stop an older gallery process before reusing its port. Playback starts paused and supports rotation, state selection, seeking, static portraits and existing videos. This is local asset playback, not physical simulation.

| Deliverable | Immutable path under `tools/avatar-assets` |
| --- | --- |
| Real animated GLB and editable Blender source | `outputs/emerald-dragon-finished-v3/model.glb`, `source.blend` |
| Full portrait, videos, fallbacks and native frames | `outputs/emerald-dragon-media-v1` |
| Compact, byte-preserving gallery media | `outputs/emerald-dragon-media-v1-portable-v1` |
| Independent native reload | `outputs/emerald-dragon-reload-v1` |
| Part-motion proof, corrected rig metadata and review | `outputs/emerald-dragon-review-v1` |
| Original provider model and receipts | `outputs/emerald-dragon-meshy-v1` |
| Generation request / finishing design | `examples/meshy/emerald-dragon-v1.json`, `emerald-dragon-finish-v3.json` |

Final GLB SHA256: `b87ee5f5fe0c3b9ab6b84d9a310104a75408ccc247d68c27fbb1d85685840d14`.
Original textured GLB SHA256: `a5e034586961d2999846d185321e6b1b4289e8138ce4b11a4a19746700109efb`.

## What moves

All clips last four seconds and are baked skeletal animation, not repeated stills or provider stock clips.

| State | Visible meaning | Sampled exported-mesh evidence |
| --- | --- | --- |
| `idle` | Small body/head movement and relaxed tail sway | Head1.95cm, tail3.71cm |
| `look_around` | Looks to both sides with a curious tilt | Head12.13cm, tail8.65cm |
| `wing_flap` | Three paired wing flaps while standing | Wings19.85cm; head unchanged |
| `milestone_level_up` | Bow, raised head, wing flourish and tail wag | Wings11.49cm, tail14.76cm |

These are maximum displacements over 33 sampled times and up to 32 strongly weighted vertices per part, not full-body contact/collision or physical-lift measurements. Feet stayed fixed within 0.000001m in those samples. First/end poses return to rest, but **no visually seamless or velocity-continuous loop is certified**. Do not label wing flapping as simulated flight.

## Mesh and rig contract

- 36,322 exported vertices, 36,501 triangles, 19 joints, real `JOINTS_0`/`WEIGHTS_0`, explicit normals/UV0/tangents. Embedded 4K basecolor/normal maps and 2K metallic-roughness. Opaque surface materials; no alpha hair cards or native groom.
- Authored height 1.2m. Native authoring uses Z-up, facing −Y; GLB uses meters, +Y up, +Z forward and right-handed coordinates. Mesh node 19 and armature node 20 have identity object transforms. Respect bone rotations, the full parent hierarchy and inverse bind matrices; bone-local translations are not world-space joint coordinates.
- Rest pose is upright, feet planted, arms low/bent, wings partially spread and tail extending behind. It is authored from generated geometry, not observed anatomy. Scale and hidden surfaces are assumptions.
- The nineteen controls are **not** Weft's nineteen joint angles. There is no Weft mapping, no invented ten-frame neutral alignment and no attachment to a panda; this is a standalone dragon. The existing panda shoulder attachment is unchanged.

| Exported node IDs | Controls |
| --- | --- |
|18 /10 /1 /0|Root / Body / Neck / Head|
|3 /2;7 /6|UpperArm_L / Forearm_L; UpperArm_R / Forearm_R|
|5 /4;9 /8|WingRoot_L / WingTip_L; WingRoot_R / WingTip_R|
|12 /11;14 /13|Thigh_L / Foot_L; Thigh_R / Foot_R|
|17 /16 /15|TailBase / TailMid / TailTip|

Exact local transforms and skin order are in the model's `exported-nodes.json`; world rest matrices and an explicitly local creature contract are in `outputs/emerald-dragon-review-v1/authored-rig-metadata.json` and `authored-rig-contract.json`, bound to the GLB hash. The latter corrects inherited provider/humanoid wording in the original pack's contract without changing that pack or its media binding. The exporter now writes correct authored-rig wording for future builds.

## Repeat the process

Read [the reusable workflow and schema](meshy-workflow.md#locally-authored-creature-rigs). Existing provider outputs are cached; do not submit another generation merely to view or rebuild this dragon. From `tools/avatar-assets`, using new output folders:

```sh
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/emerald-dragon-finish-v3.json --out outputs/dragon-next
.venv/bin/python -m avatar_assets reload3d outputs/dragon-next --out outputs/dragon-next-reload
.venv/bin/python -m avatar_assets export2d outputs/dragon-next --preset preview --size 720 --portrait 1600 --seconds 4 --samples 8 --denoise-portrait --out outputs/dragon-next-media
.venv/bin/python -m avatar_assets verify outputs/dragon-next-media
```

For another subject, author a new prompt/config/ID, prepare a local `meshy plan`, inspect costs and existing authorization, then run only the authorized stages. Inspect the new geometry before authoring its bone positions, spatial weight masks and meaningful action keys. The rig builder, validators and export pipeline are reusable; bone placement and final visual acceptance remain assisted work. Paid regeneration is stochastic, not byte-reproducible; preserved inputs, originals and manifests make existing outputs reproducible/auditable locally.

## Cost, provenance and retained failures

The user approved 35 additional credits. Meshy7.1 geometry 25 + texture 10 succeeded with no paid retry, rig or animation call. No recurring service or new subscription. Dollar purchase cost per credit is unknown. The persistent ledger retains its base 140 plus the explicit dragon-scoped 35 grant: 175 aggregate allowance, 167 spent/reserved across 15 settled submissions including historical panda/robot work. No Meta credential/budget use.

- Mesh task `01a1954b-d585-72d0-b0eb-26b636356569`, seed536030566.
- Texture task `01a1954d-4400-77c2-b0bc-c3c7cd1e424f`, seed3041861271.
- Generation 143.406s; native final finishing 13.941s / 1,314,603,008B sampled peak RSS; independent reload 1.803s / 569,196,544B. Full media finished in 527.181s / 1,179,238,400B without continuation. Compact media occupies 10,222,767 artifact bytes; all original frames remain in the full folder.
- Native workers are limited to 4 threads and 2GiB; finish 600s/512MiB, reload 180s/512MiB, media 1200s/512MiB with at most one separately bounded continuation. RSS is sampled direct-child memory, not an OS-enforced whole-process-tree guarantee.
- `finished-v1` hid the full tail at its camera angle. `finished-v2` exposed belly distortion under raised-arm celebration. Both remain intact; `finished-v3` keeps the full tail in view and removes that arm gesture. Current large arm ranges, fingers, eye blinking and independent mouth controls are **not verified/supported delivery features**.
- Fine scales and several silhouette areas remain coarse/faceted at close range.4K maps and1600px portraits do not establish Muse-quality equivalence. Local rig/motion validation is not human visual approval.

Rights/provenance records preserve the exact user request and generated hidden-surface assumptions. Provider account terms apply; commercial rights are not independently verified. The original Muse ZIP is unchanged. This high-detail PBR mesh exceeds Weft's 20k-vertex diffuse-only profile; no unchanged-fidelity Weft conversion is delivered. The explicit result is `outputs/emerald-dragon-weft-unsupported-v1`. Do not change Weft's importer, actions or physics to accept it.

## Verification status

Completed: 261 full-suite tests before final metadata/camera edits; then all 16 generated-workflow and 24 gallery tests passed for those changes. Skill validation, corrected-contract CPU validation, exported part-motion checks and independent native reload passed. The final read-only delivery audit took 2.667s / 201,850,880B process peak and verified all model/media hashes, canonical manifests, unchanged originals, 1600px opaque portrait, four silent 720px/24fps/96-frame videos and exact first-frame fallbacks. Every clip had 96 distinct decoded thumbnails; this is motion evidence, not a quality score.

Final browser proof: `outputs/browser-dragon-media-v2/browser-test.json`, 14 grouped checks across all four assets and 12 actual skeletal clips, no external request/page error. It exercised camera reset, playback, seeking, video, mobile layout, reduced-motion static default and load-error fallback. Initial ready was 4.923s; warm asset selections were 90–467ms. These are local test observations, not sustained FPS guarantees. The new bounded `initial_view` setting reveals the dragon's full tail without changing earlier assets' default cameras. Initial camera proof v1 is preserved.

Portrait, all four contact sheets, independent back view and final browser screenshots were inspected. The selected standing gestures are accepted at that sampled visual scope; no human approval, Muse equivalence, facial animation or physical validity is implied. [Current handoff](../TODO.md) and [evidence index](../evidence/dragon-delivery-v1.json) preserve exact commands and continuation state. Analytic test fixtures are not counted as generated characters; the real dragon joins the earlier generated panda/robot and locally authored fox/lamp examples.
