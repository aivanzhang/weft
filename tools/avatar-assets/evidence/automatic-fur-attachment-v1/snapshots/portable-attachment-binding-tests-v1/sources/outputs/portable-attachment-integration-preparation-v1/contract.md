# Portable attachment integration contract

Local component work; no provider/role calls or automatic quality acceptance.
Root owns native/card recording and public finish plumbing. Runner owns archive
v2 and reader. Review owns numeric auditor. Contracts owns pipeline/Studio wiring.
No shared-file edits outside these assignments without coordination.

## Binding format

Extend `fur_bindings.write_bindings(..., *, correspondence=None)` compatibly.
None writes existing schema1 unchanged. A correspondence dictionary supplies:
`source_node` (native body object name), `coat_node` (native card object name),
`source_vertex_count` (1..100000), `source_triangle_count` (1..100000), and
`source_vertex_ids` (N by3 actual native body vertex IDs).

With correspondence, write schema2 and one additional array:
`source_vertex_ids`: little-endian int32[N,3]. Existing arrays retain their
meaning/encoding. Reject repeated corner IDs and out-of-range IDs. Existing
`triangle_index` remains the per-band ordinal, not an invented global face ID.
Receipt retains all existing keys plus `export_correspondence`:

```json
{"policy":"portable-fur-source-ids-v1","attribute":"_AVATAR_FUR_VERTEX_ID",
 "source_node":"BODY","coat_node":"COAT","source_vertex_count":3,
 "source_triangle_count":1,"coat_vertex_count":6}
```

`coat_vertex_count` is exactly6*N. Names are nonempty, bounded255UTF8 bytes,
no control characters; coat name must equal writer name. Preserve existing
512KiB/file,1MiB/folder,128file,5000aggregate root limits and append-only behavior.
Support v1/v2 copy; v2 supports legacy/combed/tuft reports, while existing v1
requires its historical skin-consistent placement. Copy remains hash custody.

New `load_bindings(pack, receipt)` verifies exact size/hash/receipt/ZIP entries,
NPY shapes/dtypes/layout/decoded sizes before numeric allocation, no pickle or
remote inputs. Returns only the arrays. Writer-generated NPZ uses ZIP_STORED;
other compression is unsupported. Enforce current barycentric/inset limits.

## Native identities

Default `card_geometry` calls keep old results. An optional keyword
`record_bindings=False` captures all root triangle/barycentric/inset records for
all three geometry profiles without extra RNG draws or geometry changes. Existing
skin-consistent calls still record as before. The native public finisher always
requests records for new portable cards and writes v2 with actual selected faces.

Stamp `_AVATAR_FUR_VERTEX_ID` INT/POINT on the final body and each card mesh:
body values0..body_vertex_count-1; card values0..6*N-1. Attribute names are local
to each mesh. ID0/1 of each6vertex group are the bottom endpoints. Export these
custom attributes only for portable-card output. Reject collisions, renamed
attributes, nonidentity meter frames or ambiguous object ownership. No source
weight restoration, welding, geometry edit or style/default placement change.
Keep copied helpers and archive receipts pinned for reuse and pipeline resumes.

## Numeric audit API

`avatar_assets.fur_attachment`:
`audit_finish_pack(pack, intent) -> report` and
`validate_report(pack, intent, report) -> report` (custody/schema validation,
no full numeric rerun). CLI `python -m avatar_assets.fur_attachment PACK INTENT REPORT`.
INTENT exact fields: schema_version1, source_sha256, design_sha256, model_sha256,
state_ids (1..8 exact unique expected IDs). Use public1MiBJSON/64MiBGLB ceilings.

Report fields include schema_version1, policy`portable-fur-attachment-v1`,
status`measured`/`not_applicable`/`unsupported`, passed boolean, copied intent,
hash/byte-bound input inventory, implementation hashes, root count, expected
state IDs, sampled states/times, fixed gates, extrema/counts and bounded witnesses.
Absent portable fur is honestly not_applicable. Historical schema1 lacks exact
export correspondence: explicitly unsupported, never silently reconstructed.
Malformed/missing/hash-changed inputs raise; numeric threshold failure returns
a successfully measured report with passed=false. Every reported pass must be
derivable from validated metrics; no trusting a cached boolean alone.

Match exact named nodes and custom IDs in the final GLB, not nearest coordinates,
UV guesses or assumed primitive order. Prove every declared source triangle
exists in the actual body by its oriented ID corners; seam-duplicate rows for
the same source ID must agree in position and named skin weights. Match every
coat source ID; bottom endpoints are IDs6*r and6*r+1. Require appropriate real
skin, supported linear animation and bounded attributes. Compare the posed
exported root midpoint to barycentrically posed actual body corners minus the
declared inset transformed by root skin. Compare named weights/rest midpoint too.
Coordinates use the existing Blender-to-glTF (x,z,-y) conversion; explicit
applied rest meter frame only. This is attachment to the final body, not proof
that source body skin/shading or selected mask/color regions were preserved.

Gates unchanged: named weights/rest midpoint1e-6, animated attachment1e-5m,
authored inset<=0.0005m. Bound at100kvertices/100ktriangles,64joints,5000roots,
8states. Sample actual clips at24fps including exact endpoints, max97times/state
(4seconds),776total. No continuous/loop/physics/appearance claim. Keep report
under1MiB and bounded witness rows; hash all measured root errors if not retaining
full arrays. Child outer bound60s/512MiB/16MiB. Minimal inherited environment.

## Pipeline

Add `portable_attachment` after deformation gate and before expensive reload/media.
Create `portable-root-audit-{attempt}` with intent/report/job and sealed manifest.
Pin all new source dependencies. Child failures/uncertain exits, schema/custody
or missing evidence become EvidenceCustodyError; only validated measured numeric
failure enters existing bounded author repair. Valid unsupported mapping must
stop without purchasing a repair call. Validate caches without repeating work.
Provide bounded numeric findings to repair/critic and bind audit pack in delivery.
Studio label: Check portable fur attachment. Project events; no work in HTTP/UI.
Existing completed historical delivery semantics stay unchanged; source changes
reject unfinished old jobs. No provider test or paid allowance expansion.

## Verification

Archive invalid/oversized/traversal/shape preallocation tests; pure geometry bytes
unchanged except opt-in metadata; generic audit positive/motion-detachment and
numeric/custody failures; different node/material/action names, multiple bands,
seams and transforms. One changed-contract native analytic fixture, then actual
new-version saved-fox finish/automatic audit if analytic and integration pass.
Preserve all stopped/failed runs and approved outputs. No browser/server or UI
automation unless explicitly authorized anew; API/Studio contract tests only.
