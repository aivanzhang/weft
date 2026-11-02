# Compound avatar motion

The follow-up is **the existing small shoulder dragon takes off, flaps and lands
while the panda dances**. It does not replace the separately approved standalone
dragon. Approved fur assets at checkpoint `4ade45a` remain unchanged.

## Current candidate and evidence

`outputs/muse-panda-dance-flight-v2` is a **tested authored-motion preview**:
HD media, independent review and real browser playback pass. It is not polished
final quality or Muse parity. Cheek scuffs, mottled shoulder shading and a small
bright fleck remain; browser coat also shows colored flecks. The native video
is the fuller fur representation, not a promise of matching browser hair.
[Current handoff and save status](../TODO-history-2026-11-02.md#active-shoulder-dragon-flight-and-panda-dance).

| Item | Immutable path or result |
| --- | --- |
| Recipe | `examples/meshy/muse-panda-dance-flight-v2.json` |
| Model, editable source, prompts and backend snapshots | `outputs/muse-panda-dance-flight-v2` |
| Source partition | `outputs/companion-separation-v7/diagnostic-model.glb` |
| Independent exported-GLB reload | `outputs/muse-panda-dance-flight-reload-v2` |
| All-frame native measurement | `outputs/muse-panda-dance-flight-audit-v2/audit.json` |
| Full verified HD media | `outputs/muse-panda-dance-flight-media-v2` |
| Verified compact media | `outputs/muse-panda-dance-flight-media-v2-portable-v1` |
| Focused browser-tested catalog | `browser/gallery-compound-motion-v1.json` |
| Current tests | `outputs/complex-motion-v1/full-tests-v2.txt`: 444 passed in 12.307 s |

Final-model SHA256:
`693f8af27d9d18b5beeb344e6168f9c79471f82b9660780867b8d71956ff2c67`.
Source-partition SHA256:
`734ebdbbd06c0428a57f817d1a5deed8ff7078016ab262abb313513d94f293ca`.

The exported skin has 75,318 vertices and 19 joints. One 4 s action uses 21 LINEAR
keys: alternating toy hops, 4° toe lifts and 8° root turns; the companion moves
outward before rising, flaps its actual asymmetric wings and returns to its perch.
These are authored gestures—not physical balance, collision avoidance or flight.

Native build: 26.997 s/1,831,534,592 B sampled direct-child RSS. Independent reload:
2.041 s/668,614,656 B. The 97-frame native audit completed in 2.555 s/336,297,984 B:
base skin and the 54 mm configured fur envelope remain framed; worst measured edge
extension is 6.36 mm body/6.34 mm companion. After removing companion-root travel,
right/left wing coverage moves up to 34.48/19.83 mm. Counts are measurements, not
visual quality scores or proof that every strand, surface intersection or
intermediate-time pose is correct.

HD rendering completed in 219.116 s/1,690,763,264 B sampled direct-child RSS. The
opaque 1600 px portrait and silent 720×720/24 fps/4 s H.264 clip have 96 distinct decoded
frames and an exact first-frame fallback. Endpoint image MAE 0 is not seamless-loop
certification. Compact media retains 31 artifacts/5,496,723 B, manifest SHA256
`7ecc83314dda1644f361dc6de8613b2b9a6715cc423cfc09708bed32b21006a2`.

Actual browser verification passed in 9.516 s with 2,688,057,344 B summed owned-process
RSS (shared pages can double-count), initial load 1861 ms, no unexpected errors or
external requests. Model/clip/video playback, seek, reduced motion, injected
failures and mobile layout passed; owned browser/server exited. This is not
sustained FPS evidence. The default gallery now places this preview first and
retains all seven earlier entries; its previous bytes are saved in
`browser/gallery-before-compound-motion-v1.json`.

The [sealed delivery manifest](../outputs/compound-motion-delivery-v1/manifest.json)
binds the summary, visual review, media audit, rig delivery, unsupported Weft
result, preservation/cost and gallery-promotion receipts. Manifest SHA256:
`ee1512e11ff534fc2d01955a2320bc50be4546dc6bbef8cb7c20342b63d213b0`.
The [evidence index](../evidence/compound-motion-v1.json) provides the concise entry
point. The new entry's browser proof is not a rerun of the unchanged older models.

### Actual controls and consumer boundary

| Exported node | ID | Parent |
| --- | ---: | --- |
| `CreatureRig` | 20 | scene root |
| `Root` | 18 | `CreatureRig` 20 |
| `CompanionRoot` | 17 | `Root` 18 |
| `DragonWingRight` | 15 | `CompanionRoot` 17 |
| `DragonWingLeft` | 16 | `CompanionRoot` 17 |
| `PartitionedBodyAndPart` | 19 | scene root; mesh 0 / skin 0 |

Exact local transforms, inverse binds and every exported ID are in the model's
`exported-nodes.json`/GLB. Rest height is authored 1.2 m; native space is +Z up / -Y front,
exported glTF is +Y up / +Z front in meters. Pivots and unseen anatomy are authored.
The local identity `bone_to_simulated_joint` map tests structure only. It is **not
Weft's 19-angle mapping**. This 75k-vertex PBR/fur model is not an unchanged-fidelity
Weft-compatible deliverable: the explicit result is
`outputs/compound-motion-delivery-v1/weft-unsupported.json` (75,318 vertices,
72,181 triangles, PBR/4K maps and no verified target mapping). Do not change its
importer, physics, actions or rewards to accommodate it.

## Reusable controls

These data-only `finish_generated` controls also guide the AI author:

- `interpolation: "LINEAR"` avoids spline overshoot. Each key may contain
  `translations_m: {"Bone": [x,y,z]}`, components ±2 m, at most 64 controls/32 keys.
  Translations require LINEAR; omitted values return to zero at that key. Values
  use rest-armature axes in meters **relative to the animated parent**, not
  absolute world paths. Legacy rotation-only actions are unchanged.
- For a stable world path, compensate the parent's full XYZ rotation, not just
  one lean axis. With root pivot P, translation T, rotation R and child rest pivot C:
  `child_offset = inverse(R) * (target_world - P - T) - (C - P)`.
  The exact launch/landing blend is preserved in
  `outputs/complex-motion-v1/choreography-design-v2.json`. That non-executable
  draft is superseded by the pinned final recipe; it is not a second delivery.
- `camera_fit: "motion"` fits one fixed camera to every authored key and
  midpoint plus a configured fur margin. This is sampled framing, not a
  continuous-time guarantee; run the full-frame audit and inspect playback.
- `part_rigs` accepts 1–4 already separated, material-defined parts with exact
  `name`, `materials`, `root_bone`, `layers`, `min_vertices`, `max_vertices`.
  A part root and its descendants are its only influences; existing spatial
  layers use ORIGINAL full-mesh bounds. Shared boundary vertices, missing
  materials and insufficient/final-zero layer coverage fail before assignment.
  Requires `authored_rig`; mutually exclusive with old rigid `attachment`.
- `groom_excluded_parts` names configured non-furry parts. It excludes their
  actual weighted vertices from strands/cards without leaving a spatial bald
  patch on the body when they depart. Eye/muzzle exclusions remain separate.
- `normal_continuity` is an optional list of at most 4 records with exact fields
  `name`, `materials`, `reference_materials`, `max_distance_m`, `min_vertices`,
  `max_vertices`. Target materials cannot overlap or serve as references;
  distance 0–.03 m and aggregate ≤10k target vertices are bounded. It transfers
  reference corner normals to nearby target corners and stores authored point/
  face attributes for strand/card direction. It does **not** repair positions,
  UVs, weights, topology, hidden geometry or surface intersections.

The v2 normal repair changed 1,095 target normals across 369 faces and reoriented
35 native parent strands while protected geometry/UV/weight hashes stayed equal.
Native 6500-parent/30-child hair and 2400 portable cards remain distinct looks;
regional `groom_flow` still affects native hair only. More fibers or a higher
resolution cannot substitute for fixing a diagnosed mesh/shading defect.

```json
{
  "id": "takeoff",
  "interpolation": "LINEAR",
  "keys": [
    {"second": 0, "rotations": {}},
    {"second": 2, "rotations": {"Wing": [0, 15, 0]},
     "translations_m": {"CompanionRoot": [-0.2, 0, 0.3]}},
    {"second": 4, "rotations": {}}
  ]
}
```

This is a schema example, not a generated model or universal wing rig. Inspect
actual surfaces, pivots and weights. Never translate a fused accessory, borrow
another creature, relabel root travel as wing articulation or claim a seamless
loop merely because endpoint poses match.

## Commands and bounds

Run from `tools/avatar-assets`. Existing outputs are immutable. These first
commands only verify saved artifacts; they do not regenerate anything:

```sh
.venv/bin/python -m avatar_assets verify outputs/muse-panda-dance-flight-v2
.venv/bin/python -m avatar_assets verify outputs/muse-panda-dance-flight-reload-v2
```

For a changed/repeated recipe, choose fresh folders and reuse the pinned source:

```sh
.venv/bin/python -m avatar_assets.finish_generated examples/meshy/muse-panda-dance-flight-v2.json --out outputs/dance-flight-next
.venv/bin/python -m avatar_assets reload3d outputs/dance-flight-next --out outputs/dance-flight-next-reload
```

Finishing stays within 600 s/2 GiB sampled RSS/512 MiB output, four CPU threads.
The native audit measures all vertices at 24 fps, at most 241 frames total across
requested actions, 100k base vertices/300k edges, 60 s and 8 MiB report. It excludes
hair/cards/custom-shape helpers and requires one actual body mesh. Final part
membership uses root/descendant weights plus finishing counts; material clones
are reported, not mistaken for missing parts. Root-relative weighted movement
separates actual articulation from whole-part travel.

Use the bounded runner and snapshot dependencies when repeating the audit:

```sh
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import shutil
from avatar_assets.runner import run_bounded
from avatar_assets.safety import new_directory, require, write_json
root = Path.cwd()
model = root / 'outputs/dance-flight-next'
out = new_directory(root / 'outputs/dance-flight-next-audit')
sources = out / 'sources'
sources.mkdir()
for name in ('blender_verify_compound_motion.py', 'blender_verify.py', 'creature_rig_math.py'):
    shutil.copyfile(root / 'scripts' / name, sources / name)
shutil.copyfile(root / 'avatar_assets/part_rigs.py', sources / 'part_rigs.py')
command = [str(root / '.tools/Blender.app/Contents/MacOS/Blender'), '--background',
           '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
           '--threads', '2', '--python', str(sources / 'blender_verify_compound_motion.py'),
           '--', str(model / 'source.blend'), str(model / 'finish-config.json'),
           str(out / 'audit.json')]
job = run_bounded(command, out / 'audit.log', out, timeout=60, memory_mb=1024, output_mb=64)
write_json(out / 'job.json', job)
require(job['exit_code'] == 0 and not job['failure'], 'Inspect preserved audit failure')
PYTHON
```

These are the exact completed media/packaging commands. Existing output folders
are immutable; **choose new names for a changed recipe**, not a duplicate of this
finished run. The bounded export retains raw and denoised portraits plus every
source frame. HD handle 55110 finished; no task-owned native/browser/server job
is left running.

```sh
.venv/bin/python -m avatar_assets export2d outputs/muse-panda-dance-flight-v2 --out outputs/muse-panda-dance-flight-media-v2 --preset preview --size 720 --portrait 1600 --seconds 4 --samples 8 --denoise-portrait
# Completed after full media and appearance review:
.venv/bin/python scripts/package_portable_media.py outputs/muse-panda-dance-flight-media-v2 --out outputs/muse-panda-dance-flight-media-v2-portable-v1
.venv/bin/python -m avatar_assets verify outputs/muse-panda-dance-flight-media-v2-portable-v1
```

Media bounds are 1200 s/2 GiB/512 MiB; the current plan allows one render/no retry.
Compact packaging copies bytes; it omits full frames from the copy, not the
original. Require 720×720 silent H.264 at 24 fps/4 s, 1600 px portrait, actual first-frame
fallback, truthful opacity/loop declarations and parent/source hash continuity.
The completed local browser suite used this exact command; use a new folder
only after a changed model/catalog or unresolved failure:

```sh
node browser/tests/gallery.mjs outputs/browser-compound-motion-v1 browser/gallery-compound-motion-v1.json
```

No provider call, cloud render or recurring service is added. Existing AI-session
cost is not exposed. The user starts any persistent gallery themselves:

```sh
# From Weft: all eight entries, or just the new preview.
./tools/avatar-assets/run gallery
./tools/avatar-assets/run gallery --catalog browser/gallery-compound-motion-v1.json
```

Open `http://127.0.0.1:8765/` yourself. No owned server remains running. Local
verification does not establish a Git push: require successful remote verification
in Weft-root `.local/avatar-compound-motion-v1/receipt.json`.

## Preserved failures and source errata

- 12° toe kicks pulled belly skin; tighter foot regions worsened ankle extension
  to 25.32 mm. Combined v2 uses the original smooth weights and 4° kicks.
- Combined-v1 was rejected for angular shoulder appearance despite passing
  reload and 97-frame measurement. Its incomplete media was stopped and retained;
  receipt `outputs/complex-motion-v1/media-v1-stopped.json`. Do not resume it.
- Source-v7 retains diagnostic `needs_repair`. Its 62,117 source faces plus 497
  authored closure triangles retain original positions/image bytes, with 209
  explicitly inventoried contact-face UV corrections. Zero cross-part edges and
  no unrepaired cut loops do not make it manifold: 21 valence-four edges and
2 cap/cap plus 208 cap/source screen hits remain. No watertight/physical claim.
- Source-v7's inherited “pinched loops remain open” sentence is stale: inspect
  its actual empty `unrepaired_loops` and remaining topology diagnostics. “UVs
  retained” excludes the documented contact/closure UV authoring. Keep the sealed
  manifest and report these errata separately.
- Combined-v2's rights prose inherits “null source until parent…” from the design
  draft. Its actual source path/hash are pinned as listed above; do not rewrite
  sealed model/source manifests. The local identity rig map is not Weft alignment.

The analytic two-part native fixture independently tests meter offsets under a
rotated/scaled rig and parent/child motion. Its sampled export/reload error was
0.000000361 m; this is coordinate evidence, not this character's quality. All
original references, previous assets and rejected strategies remain preserved.
