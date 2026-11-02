# Actual-source UV2 control

This local experiment continues private checkpoint `dbdd7e2`. It uses an existing
generated panda, not a fresh text-generation or accepted Studio job. No provider
or pipeline-role calls, browser, server, public-fringe activation or deployment.
The original ZIP, approved assets and previous outputs are unchanged.

## What changed and why

The control starts with `outputs/panda-shoulder-uv-repair-v2/model.glb`, SHA256
`736ad72d408a43295098f44a29919c6436b37f741b7335f35a2c3f5f2007fb3e`.
Its 51,376 vertices and 62,581 triangles already contain the bounded shoulder
color repair. The new control keeps the previous rig, actions, grooming, cameras
and sampling settings, with only these declared changes:

- Update the source hash and exact whole-pack vertex budget.
- Omit the previously rejected `normal_continuity` operation. This retains the
  original-normal control; no normal guard or threshold is disabled. It is not
  a geometry repair or proof of byte-exact native normal import/export.
- Explicitly exclude `body_cap__contact_uv_dark_v1` from fringe roots, alongside
  `body_cap`. These clone names do not inherit exclusions automatically.
- Bake the optional undercoat on UV2 while preserving source UV0/UV1 and linked
  roughness/metallic maps. The selected normal response is intentionally replaced.

The assisted local control is not exposed as the automatic pipeline's default.
Public `portable_fringe` remains unsupported. Exact source/config comparisons are
in `outputs/panda-latest-source-control-audit-v1` and
`outputs/panda-uv2-control-reconciliation-v1`.

## Export hazard and the exact scope of the pass

Installed Blender 4.5.13 imports source normals but not the standard glTF tangent
attribute. Its exporter recalculates tangents from active UV0. The contact clone
uses a source normal map on UV1, whose directions differ from the repaired color
UV0: all 438 corners have different handedness; direction difference p95 is
171.70°. A unit/orthogonal tangent check alone cannot establish correct shading.
The earlier analytic UV2 fixture used co-oriented UV0/UV1 and did not expose this.

Actual native preflight independently verifies that **all 146 affected faces**
are selected for a newly authored UV2 normal map. It also selects all 103 dark
cap-clone faces, 265 other cap faces and 31,208 body faces. Four regions total;
the conservative export bound is **99,154 vertices**, including 482 additional
seam vertices, within the unchanged 100,000 bound. The independent audit joins
all 62,581 native triangles to the exact source and checks the selection hashes.

The staged wrapper rejects unless every old UV1-normal face is intentionally
replaced, its new native receipt binds the face count/material/UV2, and no **used**
polygon or raw-export primitive retains the old UV1 normal route. Unused original
material slots are harmless. This is **not general preservation of an original
UV1 normal map**. Roughness/metallic maps can stay on UV1 because they do not use
the tangent frame.

## Executed commands and results

All commands ran from `tools/avatar-assets` into fresh immutable output folders.
Do not rerun a sealed preparation or overwrite any result.

```sh
.venv/bin/python outputs/panda-uv2-source-preflight-preparation-v1/execute.py
.venv/bin/python outputs/panda-uv2-staged-control-preparation-v1/execute.py
# CPU derivative controller invoked this frozen child:
.venv/bin/python outputs/panda-uv2-portable-control-v1/process.py
.venv/bin/python outputs/panda-uv2-portable-preview-preparation-v1/execute.py
```

| Stage | Actual result | Observed wall time / sampled child RSS |
| --- | --- | --- |
| Native preflight | Exact masks, UV frames and adaptive budget pass | 3.646 s / 915,570,688 B; PID 61897 exited |
| Staged native preparation | Editable scene, fur, rig, actual four-second action and raw GLB; no rendered images | 16.339 s / 1,440,186,368 B; PID 62772 exited |
| CPU derivatives | Exact authored skin restoration, unused fringe tangent omission, three receipt-bound UV2 corrections, structural/motion checks | 1.034 s / 280,461,312 B; PID 63087 exited |
| Independent fidelity audit | Exact source IDs/topology/weights; nine sampled exported poses | 0.781 s / 308,805,632 B; PID 63721 exited |
| Independent exported preview | Four 720px views and four 720px actual poses | 4.151 s / 1,055,621,120 B; PID 63447 exited |

Bounds were predeclared: preflight 45 s / 1,536 MiB / 32 MiB; native preparation
120 s / 2 GiB / 512 MiB; CPU derivatives 45 s / 512 MiB / 128 MiB; preview
120 s / 2 GiB / 128 MiB. Sampled child RSS is not whole-process-tree memory.
All processes are terminal. No repeated native full finish or paid retry occurred.

Blender logged divide/overflow/invalid warnings from matrix operations, retained
in the logs. Native contracts still require finite normalized weights and finite
render-task matrices. A separate actual-point replay of the 19-bone weight field
is bit-exact to a non-BLAS calculation, with maximum sum error 2.22e-16. The warning
cause is **not proved**; do not suppress the logs or claim that replay diagnoses it.

## Exact artifacts and verified boundaries

- Editable scene: `outputs/panda-uv2-staged-control-v1/prepared-source.blend`.
  Thirteen native still tasks are saved, **not rendered**.
- Raw GLB: `outputs/panda-uv2-staged-control-v1/model.glb`, SHA256
  `6e02cc29c26872096765618de6354afa10f2da24dd181b9ca38c907b6a4efd16`.
- Final experimental GLB: `outputs/panda-uv2-portable-control-v1/model.glb`, SHA256
  `82c05f20956226b90358ba67d582fe3964623d5ff146f12ef76c9e5247ac8f87`;
  94,909 exported vertices, 76,981 triangles, authored 19-bone rig.
- Native IDs cover 94,576 source vertices: 51,376 body plus 43,200 fringe corners.
  All material-oriented triangles and body/fringe weights match the native
  snapshot exactly. Every raw/final normal byte is unchanged. The fringe-only
  helper-normal comparison has max error 0.00005883, below unchanged 0.0002;
  no whole-body native-normal snapshot proof is claimed.
- Nine exported-skeleton samples from 0–4 s have source-skin error zero and
  maximum fringe-root gap 1.60502e-9 m, below unchanged 1e-6. This is not continuous
  collision/flight/balance validation or independent native pose equivalence.
- Independent byte audit permits only 56,763 joint/weight bytes, 16,197 bytes in
  the three authored UV2 tangent ranges, ten temporary-ID binding removals and
  one unused fringe-tangent binding removal. All other JSON/binary bytes match.
  Evidence: `outputs/panda-uv2-independent-fidelity-v1/review.json`.

Units, authored rest pose, exact bone IDs/parents, mappings and action data remain
in the frozen finish config, source-skin report, final rig contract and GLB.
This HD result exceeds Weft's separate 20k-vertex/40k-triangle diffuse contract;
no Weft integration, physics change or recovered anatomy is claimed.

## Appearance rejected; next work

All eight actual exported images were visually reviewed and checked as 720×720
opaque RGB PNGs. `front.png` is a valid candidate static fallback, not a new
approved gallery/default. Identity and palette remain recognizable; the sampled
dragon travels away and back while the panda's pose changes.

`outputs/panda-uv2-appearance-review-v1/review.json` rejects this candidate:
large folded dark shoulder patches and raised white cheek shards remain exposed
after takeoff. Portable contours still read as sparse bristles/short plush, not
dense fine Muse-like fur. The four phase-spaced stills do not prove continuous
wing flapping, full action meaning or a seamless loop. No loop is claimed.

Do not spend on full video for this known-defective control or replace approved
assets. Preserve the stopped cheek-geometry branch. A different justified geometry
method or an authorized better source is still needed, together with fresh accepted
Studio runs. The separate pending paid allowance is unchanged.

## Reusable export guard

The finisher now calls `assert_normal_uv_export` before any model export or temporary
skin-ID writes. Its two source-pinned helpers are `scripts/normal_uv_export.py` and
`scripts/blender_normal_uv_export.py`. It follows only used material slots and the
live output → Principled → NormalMap → image/UV path; unused slots and disconnected
old nodes are not render inputs. Coherent UV0 routes pass. UV1/UV2 passes only for an
exact current authored undercoat receipt: material, triangle count, UV layer, bake
policy and clean packed PNG hash. Other live nonzero normal routes return explicit
unsupported evidence in `normal-uv-export.json`; there is no general source-frame
restoration. Existing post-export authored tangent correction remains mandatory.

The guard does not read texture pixels, resolve image paths or modify geometry,
UVs, normals, materials or images. It bounds objects/materials/triangles, graph
size, hashing, wall time and report bytes. Dirty authored images are rejected
because their packed bytes no longer attest the pixels an exporter could encode.

Twenty analytic/mocked-graph checks pass in
`outputs/normal-uv-export-guard-tests-v2`: 0.264 s, PID 64557 exited; sampled RSS
unknown (no samples). The prior 19-test run and its receipt-finalizer recovery
remain in v1; these are not another 19 unique tests or a native acceptance.

One actual Blender check exercises both sides of the admission boundary:

```sh
.venv/bin/python outputs/normal-uv-export-native-preparation-v1/execute.py
```

`outputs/normal-uv-export-native-v1` passes: the saved authored panda scene admits
94,576 source vertices / 76,981 triangles / ten used material slots and three exact
UV2 targets. Its original UV-repaired source rejects with
`nonzero_normal_uv_without_receipt` on `body_surface__contact_uv_dark_v1`.
Before/after positions, vertex/corner normals, topology, material ownership, UV
arrays/defaults, node graphs and packed image hashes match. Source files are also
unchanged. The guard itself takes 0.035 s for the authored scene and 0.026 s for
the source; the whole owned native process takes 5.467 s / 442,433,536 B sampled
RSS, PID 65577 exited, within 45 s / 2 GiB / 16 MiB. No render or export is repeated.
Manifest SHA256: `e638000241a4a26970338e3f7432e8edb5fc68051a0ade79fd3047e005de7643`.
This component pass does not change the appearance rejection above.

## Pipeline and resume custody

Both helpers are copied into finishing jobs and included in pipeline source pins.
A changed/missing helper blocks unfinished-job reuse. A successful native reuse
also retains the exact dedicated `normal-uv-export.json` receipt and requires it
to match the embedded finish report, supported policy and immutable prior pack.

The pipeline validates the bounded, hash-bound receipt before another repair role.
Only an unsupported receipt from a failed stage becomes a concise affected-material
finding; supported receipts are not rejection evidence. Changed, missing-listed,
malformed, oversized, duplicate-key or symlink receipts stop for inspection. An
older pack or a failure before this guard can legitimately have no receipt.

`outputs/normal-uv-rejection-pipeline-tests-v1` records 22 passing checks (12 new,
ten adjacent regression): 0.777 s / 69,304,320 B sampled child RSS. A separate
CPU-only check of both exact native reports also passes in
`outputs/normal-uv-diagnostic-actual-v1`: 0.264 s, too short for an RSS sample.
It retains the original 146-face source-material finding and report hash, while
returning no rejection for the authored scene. No originals or native results
are modified, and no new model/role call occurs.

Thirteen distinct integration contracts also pass: both helper pins and pre-native
copies, stale/missing source refusal, dedicated receipt retention, invalid resealed
receipt refusal, exact repair evidence and unchanged adjacent guards. Evidence:
`outputs/normal-uv-custody-tests-v1` through `v3`. The first run passes 12/13 and
fails only on the new fixture's `/var` versus canonical `/private/var` path text.
The path-fixed case passes alone; a separate case is renamed and rechecked after
removing prose-matching assertions. Final accounting is eleven unchanged initial
passes plus two changed-case passes, not fifteen unique contracts. Owned wall
times are 0.523 / 0.263 / 0.263 s; first sampled RSS 46,956,544 B, reruns unsampled.
PIDs 66518, 66671 and 66798 exited. Production inputs remain exact.

In total, **55 distinct focused checks** pass (20 guard + 22 diagnostic + 13
integration), plus the actual native and saved-report checks. These are component
contracts, not 55 generated avatars or whole-project/full-pipeline acceptance.

## Saved evidence and continuation

`evidence/source-uv-control-v1` preserves 575 source records as 306 byte-exact
snapshots / 11,425,366 B. Identical files share one snapshot; logs are not normalized.
Manifest SHA256: `5f627a0a1f28bcf0b67935a7e9cd15410b1951445fff518d406d594e92ff3181`.
It includes eight actual preview images, recipes, source versions, reports, test
failures and recoveries. Large GLB/Blend/NumPy files remain at their immutable local
paths with hashes; the compact bundle alone cannot recreate the whole model.
The reference ZIP and previously approved dance/flight GLB match their original
hashes. Packaging took 0.532 s / 23,052,288 B sampled RSS, PID 67038 exited, within
30 s / 256 MiB / 32 MiB. The initial inline preparation syntax failure occurred
before execution or file writes and is recorded separately, not a native retry.

All owned jobs and review tasks are terminal. The commit containing this handoff
saves only avatar tooling/evidence; its private-push receipt is
`.local/avatar-quality-publication-v12/receipt.json`. Preserve all unrelated Weft
work. No provider or pipeline-role call, deployment, browser or server was started.
The allowance remains 210 Meshy credits, 202 used/reserved, eight remaining; the
separate 105-credit and 9-expected/15-maximum-role-call request remains unanswered.
Do not treat this checkpoint as spending approval.

Next safe action: choose a justified new geometry approach or prepare an authorized
better source, preserving the stopped cheek branch and this rejected control.
Do not repeat its failed selection or spend full-media rendering on it. Fresh
text-driven Studio acceptance and meaningful final action/media review remain
required; do not promote this component result to the gallery or complete the goal.
