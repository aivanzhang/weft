# Skin-consistent portable fur

The opt-in root policy fixes a measured animation defect without editing source
body geometry, textures, weights or clips. The actual fox passes **3,000 roots ×
36 sampled poses**; maximum attachment error is **0.000374 mm** against the
unchanged **0.01 mm** gate. This is an attachment component pass, not Muse-level
fur, source-motion repair or accepted hands-off text generation.

## Reusable input and command

Add this field to a valid `finish-generated` design's portable settings:

```json
{"profile": "tuft-v1", "root_placement": "skin-consistent-v1", "sample_root_color": true}
```

The complete [example](../examples/generated/fox-skin-consistent-fur-v1.json)
uses the existing authored fox, not a new Meshy generation. From `tools/avatar-assets`:

```sh
.venv/bin/python -m avatar_assets finish-generated \
  examples/generated/fox-skin-consistent-fur-v1.json --out outputs/my-skin-consistent-fox-v1
.venv/bin/python -m avatar_assets verify outputs/my-skin-consistent-fox-v1
```

Choose a new output directory; the source must match the example's SHA256.
The command needs the existing local Blender and environment. Default/legacy/
combed placement is unchanged; explicit root placement is supported only for
`tuft-v1`. AI authoring instructions describe this option but do not enable it
automatically. The example's 256px state stills are development drafts, not final
HD animation acceptance. `verify` checks inventory/rig structure, not appearance
or the independent attachment audit below.

### Why two placement domains

Blending skin weights at a triangle point does not generally match interpolating
the three already-deformed triangle corners. The original coat therefore moved
as declared yet separated from its supporting surface. Its failed evidence is
preserved at `outputs/fox-material-coat-skin-audit-v1` (21/384 sampled roots).

- Where all three normalized named skin-weight tuples are exactly equal, their
  deformation agrees. Keep seeded area-stratified roots inside those triangles.
- Where weights differ, use an actual source vertex with its unchanged weights.
  Sample unique source IDs without replacement, weighted by incident face area.
- Allocate the unchanged total between domains by area and largest remainder;
  uniform wins ties. When the count permits, give each nonempty domain one root
  and record any one-root transfer. Insufficient varying-domain capacity returns
  unsupported, never fewer roots or reallocation to easier regions.

Complete positive named influences are sorted and normalized with `math.fsum`;
there is no equality tolerance or cutoff. Zero entries do not affect the tuple.
More than four positive influences or normalization underflow is unsupported.
Existing normals, source color sampling, flow and style settings remain intact.
Source seam IDs are not silently welded; coincident anchors remain reportable.

Pure vertex-only placement was screened and stopped before rendering: it would
underfill two coarse fox parts and relocate some roots by 135 mm. The hybrid
retains interior sampling there. The unexecuted draft and both screens remain
in `outputs/vertex-only-integration-draft-v1` and the vertex-root preflights.

Current new finishes additionally record exact source vertex IDs in version-2
archives and export per-mesh IDs. The [automatic attachment audit](portable-fur-attachment.md)
now checks those final GLBs through a reusable pipeline stage. The version-1
archive and independent fox result below remain historical evidence; their files
are unchanged and are not silently upgraded.

## Outputs and limits

Normal finisher outputs remain: real skinned GLB, editable `source.blend`, portrait,
state frames, node/bone inventory, exact design/source snapshots and manifest.
Each portable mesh additionally has a receipt in
`finish-report.json → portable_fur[].root_summary.binding_archive` for:

```text
portable-root-bindings/PortableCoat_0[.NNN].npz
  triangle_index  int32[N]
  barycentric    float64[N,3]
  root_inset     float64[N,3]
```

Records cover every root in generation order, not just the maximum 128 sampled
color/flow records. Coordinates are Blender-local metres, +Z up; inset is
**surface root minus bottom-edge midpoint**. Triangle IDs refer to that portable
part's ordered emitter triangles, not guessed global GLB IDs. Preserve the
material correspondence needed to reconstruct them. A hash-bound copy is byte
custody, not semantic validation of arrays or skin.

Limits: 100k source vertices/triangles, 5k total roots, four positive influences
per referenced vertex, 512 KiB per archive, 1 MiB/128 files per binding folder.
The writer validates numeric shapes before copying, rejects symlinks/traversal/
overwrite, and writes without pickle. Reuse requires matching helper snapshots,
exact file inventory, receipts, aggregate count, size and SHA256 before copying.
The public 1 MiB JSON read ceiling remains unchanged. Existing material/groom,
exported-vertex, native runtime/memory/output limits still apply.

Native hair remains a separate editable render representation; GLB contains
alpha-masked tufts, not exported native strands. Rendered backgrounds are opaque.
Use `portrait.png` or the first state PNG for reduced motion. No loop is certified.

## Actual fox result

Recorded November 2, 2026, PST. Existing locally authored fox geometry, rig,
procedural textures and four actions are preserved, not a new generated example.
Unseen surfaces, neutral pose and scale are authored assumptions. No additional
provider content or independent third-party commercial rights verification.

| Artifact | Immutable path / SHA256 |
| --- | --- |
| Source | `outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb` — `f34aa798252355fc6e44a03f78cdc7f31eab682a20b591a93c9dcb35ed82f8a3` |
| Model | `outputs/fox-skin-consistent-fur-finish-v1/model.glb` — `e08bd6e0e548e1e89c6a46b811ab9d5fa11b8c9e15844d50f5ec480455171d80` |
| Finish manifest | `outputs/fox-skin-consistent-fur-finish-v1/manifest.json` — `be6ee85e53a06f28531bec2b4a94d31ca5078b38efeeeceded7d1a0c793f23de` |
| Full-root audit | `outputs/fox-skin-consistent-root-audit-v1/manifest.json` — `688e68215b776ee96737418911aa0fede8f5bb3f9c47a51960e8541cf0712256` |
| Coverage release | `outputs/fox-skin-root-selected-coverage-v1/release-manifest.json` — `4abed1faadf947d84db5a115099e9e5c337f9c5ba1591f10dcd25d141b47f612` |
| Appearance record | `outputs/skin-consistent-fox-appearance-review-v1/manifest.json` |

Actual one-shot commands, already finished; do not rerun sealed outputs:

```sh
.venv/bin/python -B outputs/skin-consistent-fox-preparation-v1/execute.py --execute
.venv/bin/python -B outputs/skin-consistent-fox-preview-preparation-v1/execute.py --execute
```

The first wraps the public finisher with stricter limits: 120 seconds / 2 GiB /
128 MiB for native work; 30 seconds / 512 MiB per CPU phase; 128 MiB combined
output. Native PID 94785 exited 0 in 26.348s at 1,561,853,952 B sampled direct-child
RSS. CPU 94770 exited; controller took 27.277s and final combined output was
106,818,971 B. Full-finish output alone is 106,381,024 B. No paid call or retry.

Independent GLB reload/render PID 95318 exited 0 in 2.080s at 712,376,320 B sampled
RSS under the same native limits. Four 720px/16-sample EEVEE rest views use the
previous model's recorded bounds for frozen camera/light framing. The preparation
pins the 32 reconstructed bounding-box corners and historical model/inspection;
they agree within 1.01e-7m. This comparison-only renderer snapshot does not alter
the production renderer. Metrics measure the direct child, not descendants.

The body/color audit preserves 47,776 oriented body triangles, source positions,
UVs, named weights and 16 material/image semantics. All 384 sampled root colors
pass. It does not prove exact shading: prior normal/tangent differences around
0.000400 and 2,075 ambiguous coincident normal identities remain.

The independent root audit checks all 3,000 archived roots, their exported bottom
edges, exact insets and named weights at nine times in each of four clips:
108,000 root-time checks, zero over the fixed gate. Full numeric measurements
are retained in `sampled-motion.npz`, not only extrema. Maximum declared-skin
motion error is 3.757e-7m and named-weight error 2.627e-8. This finite CPU check
does not certify continuous playback, hair tips, native strands or GPU rendering.

Counts remain coat 2017 (1535 interior/482 vertex), cream 527 interior and dark
456 interior. Torso/tail receive 419/268 roots against area quotas 418.973/267.683.
Every indexed component expecting at least one root has a root. There are 236
coat and 52 cream tiny indexed components without roots, each expected below
one; indexing/seams are not a visible-hole measurement. Two seam IDs (5723/5751)
share one selected root position. No roots were dropped, merged or relocated to
hide this. Coverage and attachment checks do not accept coat appearance.

### Appearance and motion remain limited

Eight images were directly inspected: four new exported rest views, new native
portrait, one new raised-arm draft, and the old exported back/side views. Identity,
palette, scarf and drawing props remain coherent. The coat still reads as short
plush with sparse outline hairs; tail color borders, back-of-leg patches and
hard body-part transitions remain conspicuous. The comparison does not show a
density upgrade. The raised-arm state is legible in one draft, not full action
acceptance. No continuous clip was reviewed in this increment.

The retained raised-arm source clip still has 2.543× edge stretch and 33.009mm
excess length at its sampled peak. Root attachment does not repair body stretch.
No new accepted fox, Muse parity, fresh Studio delivery, physical motion or
loop is claimed. Approved panda/dragon assets and gallery defaults are unchanged.

### Rig and integration boundary

GLB units are metres, right-handed +Y up/+Z forward; Blender basis is `(x,-z,y)`.
Use exported node transforms/inverse bind matrices as the rest/bind definition.
There are 20 authored joints and four inherited four-second clips. Root node 19,
torso 18, head 2, armature 20, body 21, portable coat 22–24. Skin 0 joint order:

```text
19 18 2 0 1 3 7 6 5 4 10 9 8 14 13 12 11 17 16 15
```

Sketchbook node 4 follows hand 5; pencil 11 follows hand 12; tail is 3. There is
no dragon attachment in this example. `exported-nodes.json` is authoritative.
The local validator map is not a Weft joint mapping. This HD model exceeds the
frozen 20k-vertex/diffuse-only consumer profile; Weft compatibility is unsupported.
No importer, rendering, simulation, physics, action or reward code changed.

## Verification, custody and next action

Focused checks passed: 11 vertex-selector, 12 hybrid-selector, 21 archive and
47 integration checks. The latter cover actual geometry/skin math, full binding
records, source pins and native reuse; exact default card bytes match the saved
pre-change implementation. The tests use analytic fixtures, not generated assets.
The public `verify` command passes on the actual new pack. Per-run requests,
source hashes, logs, bounds and receipts are retained under `outputs/skin-root-*-v1`.

Independent body audit: 3.123s / 337,428,480 B sampled RSS, PID 95132 exited.
Full-root audit: 3.644s / 221,954,048 B, PID 95277 exited. Public verify: 0.522s /
135,249,920 B, PID 95397 exited. Selected coverage: 0.259s, PID 95459 exited;
97,959,936 B process self-peak, no monitor RSS sample (not zero memory).
No native/CPU/Studio/browser/server job remains owned or running.

Compact evidence in `evidence/skin-consistent-fur-v1` includes exact binding and
measurement NPZ files. Large GLB/Blend sources stay at immutable local paths with
hashes; this is not a full remote model backup. All stopped vertex-only proposals,
the original failed attachment audit and approved assets remain preserved.

Final read-only review found the card builder missing from prepared-job source
pins. `blender_fur_cards.py` is now pinned alongside the selectors and archive
helper. The one changed resume-custody test passes in 0.038s (bounded 0.263s,
PID 96575 exited, monitor RSS unsampled): changing only that source rejects
before work or prepared artifacts change. The prior 47-check receipt is unchanged;
this is a focused rerun, not an extra claimed unique test. Historical native
build source pins were complete and remain valid.

The evidence bundle preserves 521 origin records / 345 exact snapshots /
20,275,657 B; manifest SHA
`3a1ca77bc102a55a0b003709e8255a3235edcdbf3ab0affeb9dca8c8fbea9c3f`.
Packaging initially failed before launching because its output folder did not
exist. The original request/script/error remain; creating the required new empty
folder fixes the harness. The sole actual package child (PID 96919) exited in
0.531s / 23,887,872 B sampled RSS. Original ZIP and approved panda hashes match.

No new provider or generation-pipeline role calls. The Meshy ledger remains
202/210 credits; the 13-call AI study is exhausted. The fresh-case 105-credit /
maximum-15-role allowance remains pending, not granted by this local work.

Next safe work: make the independent root/coverage acceptance reusable across
new exports and test it without weakening its gates; then address final-visible
fur/source defects with a distinct measured change. Do not rerender this unchanged
comparison, promote the opt-in profile by default, or claim Studio completion.
Fresh prompt/custom-action acceptance and final-HD media remain separate gates.
