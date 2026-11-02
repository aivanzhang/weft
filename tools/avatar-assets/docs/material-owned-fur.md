# Material-owned fur

The opt-in `material_fur` profile keeps each selected surface's own color image
or constant color when adding native hair and portable tufts. The old single-image
path and approved panda/dragon assets are unchanged. This is a compatibility and
source-preservation improvement, not dense-fur, Muse-quality or fresh Studio acceptance.
The original animated fur-attachment check **fails** on the fox; this candidate
is not an accepted animated fur delivery. A separate opt-in
[skin-consistent root revision](skin-consistent-fur.md) now passes all-root sampled
attachment without changing the source body. Its appearance and source-motion
limitations remain; the original failed candidate and evidence below are unchanged.

## Inputs and reusable command

Use exact material names from the inspected source, not a species template:

```json
{
  "material_fur": {"version": 1, "materials": ["coat", "cream", "dark"]},
  "normal_strength": "preserve"
}
```

These fields extend a complete `finish-generated` design. A native `groom` and
shared regional `groom_flow` are required. Portable fur, when supplied, must use
`profile: "tuft-v1"` and `sample_root_color: true`. AI authoring can opt in using
observed names; the coordinator sets `normal_strength` to `preserve` after quality
defaults. Native/portable fibers, action intent and final-media review remain
required by the high-quality policy. The profile is not selected automatically.

The complete [fox example](../examples/generated/fox-material-fur-v1.json) uses an
existing local authored model. From `tools/avatar-assets`, choose a **new** output:

```sh
.venv/bin/python -m avatar_assets finish-generated \
  examples/generated/fox-material-fur-v1.json --out outputs/my-material-fox-v1
.venv/bin/python -m avatar_assets verify outputs/my-material-fox-v1
```

This command performs local finishing, not fresh text generation. It needs the
hash-matching source at the example's relative path and the existing local Blender.
Its 256px pose samples are development drafts, not final-HD action acceptance.
The actual run below used a stricter prepared controller instead of the general
600-second / 2-GiB / 512-MiB finishing ceiling. Do not rerun its sealed command.

### Admitted sources

- One bounded triangular skinned emitter, applied identity meter frame, one
  linear armature, at most 64 bones/groups and 100k vertices/faces.
- One to 32 explicitly selected materials. Every referenced vertex must have
  unambiguous material ownership and, for textured surfaces, identical UVs on
  all its loops. Split seams explicitly upstream; averaging them is unsupported.
- Direct sRGB image **Color** to Principled Base Color, Linear/REPEAT/FLAT,
  coherent active/render UV0, or an unlinked constant linear RGBA color.
  Selected emitting surfaces must be opaque. Linked alpha, procedural graphs,
  mapped coordinates and alternate texture routes are rejected.
- At most 4096px per image edge, 16 Mi-pixels in aggregate across unique live
  color images, and 64 MiB packed bytes per image. Unused slots are excluded.
  One CPU image buffer is held per sampling/build phase.
- Global budgets: at most 10k native parents, at least 16 per selected material,
  240k configured fibers including render-child overrides, 150k stored hair keys,
  20 seconds aggregate comb work, and 5k portable tufts. Material count never
  multiplies the requested totals; the ordinary 100k exported-vertex ceiling remains.

This initial profile does not combine with material/normal/topology/weight repair
controls, attachment, adaptive undercoat or experimental portable fringe. Existing
nonzero normal-texture UV routes still need the normal-export guard's explicit
authored receipt; the new color adapter does not authorize them.

## Outputs and preservation

The native groom uses source-indexed temporary subsets, retaining every source
vertex, weight and coordinate for consistent whole-body flow bounds. Only selected
faces emit. Original body geometry/materials are not edited. Each retained native
hair object records its real source/material; temporary objects are removed only
after their own copies are made.

Portable tufts share a global area-weighted budget across material/band regions.
Textured roots use their own triangle UV and linear-light bilinear sampling;
constant-color roots retain the original linear RGB. The report records sampled
root coordinates, source IDs and colors. Native hair is not exported as GLB strands.

In addition to the standard model, editable `source.blend`, portrait, draft state
frames, exported node IDs and manifests, preserve:

- `material-groom/report.json` and `correspondence.npz`: exact source face/loop/
  vertex and normal correspondence, budgets and original-body signatures.
- `material-coat.json`: aggregate per-material tuft allocations.
- `finish-report.json`: bound copies of both reports, sampling and motion findings.
- `sources/`: exact design and helper snapshots. Prepared pipeline jobs pin all
  four new helpers, and native reuse checks/copies the material receipts.

`native_root_texture_sampling_verified` remains false: a bound image/UV graph and
portrait inspection do not independently prove every strand's renderer sample.
Model alpha-masked tufts and opaque rendered backgrounds are different facts.
Use `portrait.png` or a state's first PNG for reduced motion; no seamless loop is
certified by this increment.

## Actual local fox check

Recorded November 2, 2026, PST. This uses the existing authored fox, not Meshy or
a new AI-authored acceptance recipe. Its unseen geometry, metric scale, rest pose,
rig and four actions remain authored assumptions. Procedural textures are existing
local sources; third-party commercial rights are not independently verified.

| Artifact | Immutable path / SHA256 |
| --- | --- |
| Source | `outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb` — `f34aa798252355fc6e44a03f78cdc7f31eab682a20b591a93c9dcb35ed82f8a3` |
| Finished GLB | `outputs/fox-material-fur-finish-v1/model.glb` — `5f944c6d5a3c63b2b9ccbd6d965651e6367805f64aa37e82ff54ea3e0a71c7fb` |
| Finish manifest | `outputs/fox-material-fur-finish-v1/manifest.json` — `8bbad9c6db6263d4fae88b3f8548704af4b8e7f7f0b7caff643d18e6541e055d` |
| Correspondence | `outputs/fox-material-fur-finish-v1/material-groom/correspondence.npz` — `8b7b59269e10cbe83eb146cc30168f9acc6c8a85e0ccde94316c416f90b78d13` |
| Exported rest views | `outputs/fox-material-fur-export-preview-v1/` — front, side, back and three-quarter, each 720px |
| Visual review | `outputs/material-fur-fox-appearance-review-v1/manifest.json` — `458f4fe0bbe92224980597288202d968728a03a730fb8673503d4bb117ff0c7a` |

Actual one-shot command:

```sh
.venv/bin/python -B outputs/material-fur-fox-preparation-v2/execute.py --execute
```

The public finisher completed: native PID 87042 exited 0, 26.123 seconds,
1,553,629,184 B sampled direct-child RSS; controller 27.027 seconds, CPU worker
87037 exited. Native limits were 120 seconds / 2 GiB / 128 MiB output; CPU phases
30 seconds / 512 MiB. The finish produced 62,618 exported vertices, 20 bones,
four existing four-second actions, 5,000 native parents / 155,000 configured
fibers, and 3,000 portable tufts. The three source color images total 9,437,184 pixels.

Native parent allocation: coat 3347, cream 885, dark 768. Portable allocation:
coat 2017, cream 527, dark 456. Comb work took 0.791 seconds with 20,000 stored keys.
Producer source-normal error is zero; independent evidence is recorded separately.

Independent native GLB loading/rest views used:

```sh
.venv/bin/python -B outputs/material-fur-fox-export-preview-preparation-v1/execute.py --execute
```

PID 87603 exited 0 in 2.345 seconds, 709,033,984 B sampled direct-child RSS,
62,677,971 B output, under the same 120-second / 2-GiB / 128-MiB limits.
This verifies imported rig/action presence and four static renders, not continuous
playback or browser compatibility. Measurements exclude descendant RSS.

Independent CPU audit `outputs/fox-material-fur-independent-audit-v2` passes:
original-body positions, UVs and named weights match exactly under source
correspondence; all 47,776 oriented triangles and 16 material/image bindings are
preserved. Twenty-bone rest error is at most 2.384e-7; four clips at nine all-body
samples differ by at most 3.433e-7 m. All 384 recorded tuft roots retain their
source triangle/material/UV ownership. Independent linear-light bilinear color
error is at most 1.328e-7, exported RGB error 1.571e-7, root-center error 1.116e-7 m.
These are sampled roots, not all 3,000 roots or GPU/native-strand sampling.

Audit PID 89140 exited 0 in 3.123 seconds, 337,477,632 B sampled RSS, bounded at
30 seconds / 512 MiB / 16 MiB. Manifest SHA:
`2fe038fb9bbc65e2959ea6835ae9ced13e5847ce3b7f9e749abd441a6bf8680e`.
Normals/tangents are not byte-identical: measured lower-bound maximum is about
0.000400, with 2,075 body vertices having ambiguous source-normal identity.
The audit does not establish shading equality, new-coat motion or action quality.

The separate coat-only skin audit then found a real attachment limitation:
`outputs/fox-material-coat-skin-audit-v1`, manifest SHA
`8a8e4f11af8c1817a67e676d206a9b9cffed166b43f74bd11875c510914319a5`.
All 384 sampled roots match expected normalized named weights (maximum error
6.068e-8), rest anchors (1.116e-7 m) and declared skin animation (3.862e-7 m).
However, barycentrically blended weights at a new point do not exactly reproduce
the deformation of its supporting triangle when its vertices have different
weights. No influences were lost to the four-weight export limit in these samples.

After compensating only the authored root inset, 21 of 384 sampled roots exceed
the unchanged 1e-5 m attachment gate. Per-axis maxima are 0.000160 mm for idle,
0.328997 mm for making_something, 2.142180 mm for milestone_level_up, and
0.770125 mm for working. Maximum total distance is 2.495361 mm. Exact times,
root/triangle IDs and influences are retained. This is an **attachment rejection**,
not a checker failure or permission to widen tolerances. The single bounded CPU
job (PID 89764) exited 1 in 0.783 seconds, 181,223,424 B sampled RSS; no retry,
native call, asset edit or quality promotion followed.

The documented public `verify` command also passes for the actual finished pack:
`outputs/material-fur-public-verification-v1`, PID 89018 exited 0, 0.522 seconds,
142,409,728 B sampled RSS. This rechecks the saved inventory and rig, not appearance.

### Critical exported IDs and limits

The final GLB is right-handed, meters, +Y up and +Z forward. Its exported rest
node transforms and inverse bind matrices are authoritative, not assumed neutral
Weft positions. Skeleton root is node 19, torso 18, head 2, `AvatarRig` 20;
body mesh node 21; portable coat nodes 22–24. Skin 0 joint order is:

```text
19 18 2 0 1 3 7 6 5 4 10 9 8 14 13 12 11 17 16 15
```

Sketchbook node 4 follows hand 5; pencil 11 follows hand 12; tail is 3.
Full hierarchy/weights live in `exported-nodes.json` and the GLB. There is no
dragon or new attachment in this case. The validator's name-to-name local map
is **not a Weft mapping**. This HD model exceeds the frozen 20k-vertex consumer
profile and uses richer materials; Weft runtime compatibility is unsupported.

### What the images establish

Seventeen original images were inspected: native portrait, four exported views
and twelve 256px draft state frames. Identity, colors and props remain coherent.
The coat reads as short plush with isolated outline hairs, not dense Muse fur.
Tail color borders and lower-leg patches remain abrupt. The raised-arm state is
visible; small drawing/reading gestures need final-resolution continuous review.

The retained `milestone_level_up` sample reports up to 2.543x edge stretch and
33.009 mm excess length at two seconds. Existing motion preservation does not
accept that deformation or action quality. No full video, loop, new rig quality,
native/portable visual equivalence, automatic prompt success or Studio delivery
is claimed. Approved panda/dragon assets and gallery defaults are unchanged.

## Software checks, failures and next action

Component checks: 14 allocation/sampling tests, eight material-adapter tests,
five config tests, ten lazy-color tests, sixteen existing generated-workflow
tests, forty pipeline/policy/source-custody tests, and seventeen native-reuse
custody tests. These are software/analytic
checks, not newly generated character examples. Per-run commands, snapshots,
logs and process bounds are preserved under `outputs/material-*-tests-v*`.

Retained failures:

- `material-fur-fox-preparation-v1`: inherited actions do not meet the explicit
  authored-action requirement for motion camera fit. Rejected before native work;
  v2 uses rest framing and makes no continuous framing claim.
- `material-finish-contract-tests-v1`: the workflow suite's bare sibling import
  failed because the harness omitted `tests/` from `PYTHONPATH`. Its fifteen other
  checks passed. v2 corrects the harness and passes 21 config/workflow checks;
  unchanged lazy-color checks were not rerun.
- Reuse review identified that two absent/null coat receipts could compare equal.
  Reuse now requires a dictionary with the exact `material-owned-tufts-v1` policy
  before matching/copying. This post-run guard does not change native geometry.
- Independent audit v1 expected VEC4 vertex color, but the actual GLB uses valid
  VEC3 RGB. It stopped with a checker-schema failure after the body checks. V2
  admits VEC3 or VEC4 without changing RGB, geometry, skin tolerances or assets.
  The v1 job and checker-only diff remain saved.
- Reuse-test preparation initially named a nonexistent `tests/__init__.py`;
  tests use a namespace package. It failed before any test child launched.
  An append-only corrected preparer preserved/verified the existing snapshots.

Reuse tests exercise real file copying and resealed bad receipts with analytic
model/media bytes, explicitly not a native rebuild. Their one child (PID 89398)
exited 0 in 0.522 seconds, 46,268,416 B sampled RSS. Actual fox dedicated/embedded
receipts and correspondence hash also match. Evidence:
`outputs/material-native-reuse-tests-v1`, manifest SHA
`bebfe74dadc9a2dd8a864b9be7024f79e90a735e6a6970e02588bbb2775bcdd6`.

No new provider or generation-pipeline role calls occurred. The saved Meshy
ledger remains 202 used/reserved out of 210 authorized credits. The separate
13-call AI study is exhausted; the prepared 105-credit / maximum 15-role-call
fresh-example request is still pending, not granted by this local work.

Next: test a genuinely different bounded root-attachment policy, such as roots
anchored to actual source vertices, before another animated-coat acceptance.
Keep the correct weight/color evidence separate from failed surface attachment;
do not repeat this unchanged experiment or compensate by widening the gate.
Save the current compatibility work and its rejected attachment result privately.
Do not rerun sealed native jobs, promote the coat as dense hair, restart terminal
Studio jobs or spend pending allowances. Fresh generated/action/Studio acceptance
remains governed by [the active completion criteria](hands-off-studio.md).

## Saved evidence and ownership

`evidence/material-owned-fur-v1` preserves 441 origin records / 293 exact snapshots /
7,693,029 B, including the rejected attachment result and all failed preparations.
Manifest SHA: `12704910e5e41ad3c144fa2c9c0c2d3efb6ddb8781d921a8a23c7009e269dda9`.
Large GLB/Blend/NumPy files remain local by recorded path/hash; this is not their
complete remote backup. The original reference ZIP and approved panda dance GLB
hashes still match. Packaging PID89950 exited0 in0.264s; RSS was unsampled, not zero.

Root owned native execution, integration, review and save. Agent lanes owned
material kernels/pipeline contracts, native-groom/reuse custody, and independent
CPU audits. All owned current jobs have exited; no native, provider, Studio or
browser handle needs resuming. Private save status is recorded separately in
`.local/avatar-quality-publication-v15/receipt.json`. No Weft consumer, physics,
actions, rewards, credentials or its Meta budget were changed by this work.
