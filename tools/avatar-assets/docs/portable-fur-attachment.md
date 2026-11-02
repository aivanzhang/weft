# Automatic portable-fur attachment audit

The local pipeline now checks whether every recorded portable-fur root stays
attached to the **actual exported body** during sampled animation. It runs before
exported previews, final media and critic calls. This is a numerical attachment
gate, not approval of fur appearance or a completed hands-off Studio run.

The implementation passes analytic tests, real Blender positive/negative controls,
and a real pipeline attachment-stage run on the complete saved fox. The latter
checks 3,000 roots across 388 sampled poses without an attachment failure. It is
not a fresh text generation or complete Studio delivery. No provider call,
paid retry or spending allowance was added by this work.

## Supported inputs and exact identities

New public `finish-generated` outputs record all roots for `legacy`, `combed-v1`
and `tuft-v1` portable-card geometry. The opt-in
[`skin-consistent-v1` root placement](skin-consistent-fur.md) remains a separate
`tuft-v1` setting. Recording does not change placement, style, random draws,
geometry, skin weights or source assets. Single-image and admitted
[material-owned](material-owned-fur.md) paths retain their existing masks and
color contracts. Experimental edge fringe is not part of this card audit and
remains disabled in public finishing.

| Identifier | Meaning |
| --- | --- |
| `portable-fur-attachment-v1` | Numeric audit policy; report schema version 1 |
| `portable-fur-source-ids-v1` | Export-correspondence policy |
| `_AVATAR_FUR_VERTEX_ID` | Local point ID on the named body and each named coat |
| `portable-root-bindings/PortableCoat_*.npz` | Append-only, hash-bound all-root archive |
| `finish-report.json → portable_fur[].root_summary.binding_archive` | Archive receipt, schema version 2 for new exports |
| `finish-report.json → portable_fur_identity` | Exact body/coat ownership and exported-ID receipt |

Each archive contains little-endian `triangle_index:int32[N]`,
`barycentric:float64[N,3]`, `root_inset:float64[N,3]` and, in version 2,
`source_vertex_ids:int32[N,3]`. Triangle ordinals refer to the **selected band's
emitter list**. The new source IDs resolve that list after masks, material ownership
and brightness selection; they are not guessed global GLB triangle numbers.

The receipt binds path, byte length, SHA256, root count and the named source/coat
meshes. Its `export_correspondence` records the policy, attribute, source vertex
and triangle counts, and `coat_vertex_count == 6*N`. Native IDs are INT/POINT;
exported IDs are exact integer-valued FLOAT/SCALAR. IDs are local to each named
mesh. The two bottom endpoints of root `r` are coat IDs `6*r` and `6*r+1`.
Export seam duplicates may share a source ID, but must agree in rest position
and normalized named skin weights. No nearest-point matching or welding occurs.

Archive coordinates are `blender_local_z_up_meters`; the inset convention is
`surface_root_minus_bottom_edge_midpoint`. The audit converts Blender `(x,y,z)`
to glTF `(x,z,-y)`. It requires applied identity metre frames and a real skin;
arbitrary transforms or ambiguous ownership are not inferred.

## What is measured

For every root, the auditor proves that its oriented source-ID triangle exists
in the exported body, checks the coat's normalized named weights against the
complete expected barycentric blend, and checks its rest midpoint. It then skins
the actual exported body corners and card endpoints at each sample time. The
comparison includes the declared inset transformed by the root skin. This catches
the case where apparently correct blended weights still detach from the moving
triangle. Expected influences are not truncated to the easiest four bones.

| Fixed gate | Bound |
| --- | ---: |
| Named-weight error | `1e-6` |
| Rest midpoint error, maximum absolute coordinate component | `1e-6 m` |
| Animated attachment error, maximum absolute coordinate component | `1e-5 m` (0.01 mm) |
| Authored inset length | `0.0005 m` maximum |

The auditor accepts actual `LINEAR` or `STEP` clips longer than zero and at most
four seconds. It samples at 24 fps from each clip's actual start, includes the
exact endpoint, and supports 1–8 expected states: at most 97 times per state,
776 total. Longer clips and unsupported interpolation return `unsupported`; the
audit does not shorten a requested action to obtain a pass.

Resource admission occurs before numeric array allocation: 64 MiB GLB, 100,000
aggregate selected exported vertices and triangles, 64 joints and 5,000 roots.
Repeated primitive occurrences count toward the limits even if they reference
the same accessors. JSON is bounded to 1 MiB; archives are at most 512 KiB each,
1 MiB/128 files collectively. Exact ZIP/NPY member names, types, shapes, decoded
sizes and values are validated without pickle. Symlinks, traversal, duplicate
IDs with conflicting data and changed hashes fail closed.

The pipeline child has a 60-second / 512 MiB / 16 MiB output envelope and a
minimal environment. Reports retain aggregate errors, counts, exact state times,
up to 32 witnesses and hashes of the complete error streams. A sampled RSS peak
is reported only when the monitor actually observed one.

## Pipeline, Studio and reuse

The new `portable_attachment` stage follows the deformation gate and precedes
native reload, action audit, exported review images, full media and critic calls.
Studio displays **Check portable fur attachment** from saved stage events; no
audit work runs in the UI or status request.

Each attempt writes:

```text
portable-root-audit-N/
  intent.json       exact source, design, model hashes and expected state IDs
  report.json       measured / not_applicable / unsupported; passed boolean
  job.json          command, limits, timing, exit and resource receipt
  audit.log         bounded child output; monitor/handle files are also retained
  manifest.json     bound input/output inventory and implementation identity
portable-root-audit-N.json   separate controller custody receipt
```

Only an authenticated `measured` result with `passed:false` enters the existing
bounded author-repair loop. The author and critic receive a compact hash-bound
summary and at most four witnesses; the full report remains preserved.
`unsupported` stops without another repair-role call. Missing, malformed, changed
or unsealed evidence, worker failure and uncertain exit require inspection, not
automatic retries. No threshold or action meaning is weakened.

A genuinely absent portable coat returns `not_applicable`, not an attachment
measurement. Historical packs with absent/version-1 export correspondence return
`unsupported`; the tool never reconstructs IDs from positions or relabels old
evidence. An existing completed historical pipeline delivery remains resumable
under its original contract. Unfinished jobs with stale source pins are rejected.

Native reuse checks the exact design, successful prior job, inventory and helper
hashes, then copies binding archives and receipts unchanged. It does not rerender
or upgrade historical evidence. Prepared-only scenes remain a separate
experimental route and cannot be substituted for a completed public finish.
Validated audit caches are reused without repeating numeric work; controller
receipts detect report/manifest replacement. Critic receipts and new deliveries
bind the audit pack and its controller receipt.

## Commands and outputs

Run from `tools/avatar-assets`. A normal new `finish-generated` pack now records
version-2 identities automatically; it does **not** by itself claim this numeric
audit passed. The [prompt pipeline](prompt-pipeline.md) invokes the gate.

For a standalone diagnostic, prepare a fresh audit directory and an `intent.json`
with exactly these fields, using the reviewed finish-design's bytes and expected
state order:

```json
{
  "schema_version": 1,
  "source_sha256": "<source SHA256>",
  "design_sha256": "<sources/finish-design.json SHA256>",
  "model_sha256": "<model.glb SHA256>",
  "state_ids": ["idle", "look_around"]
}
```

The CLI is `python -m avatar_assets.fur_attachment PACK INTENT REPORT`. Use the
existing bounded runner rather than treating that CLI alone as an OS resource
limit. This example assumes the fresh directory and reviewed intent already exist:

```sh
.venv/bin/python -B - <<'PY'
from pathlib import Path
import sys
from avatar_assets.runner import run_bounded
from avatar_assets.safety import read_json, write_json

pack = Path('outputs/my-finished-pack-v1').resolve()
out = Path('outputs/my-attachment-audit-v1').resolve()
cmd = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin',
       'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1',
       'OMP_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1',
       'PYTHONPATH='+str(Path.cwd()), sys.executable, '-B', '-m',
       'avatar_assets.fur_attachment', str(pack), str(out/'intent.json'),
       str(out/'report.json')]
job = run_bounded(cmd, out/'audit.log', out, timeout=60, memory_mb=512, output_mb=16)
write_json(out/'job.json', job)
if job['exit_code'] != 0 or job['failure'] or not job['process_exit_verified']:
    raise SystemExit('Audit failed; preserve this attempt and inspect its receipts')
print(read_json(out/'report.json'))
PY
```

Exit code zero means the measurement completed; inspect `status` and `passed`.
A numeric rejection deliberately exits zero with `passed:false`. The standalone
example does not create a pipeline delivery or its controller custody seal.
Programmatic APIs are `audit_finish_pack(pack, intent)` and
`validate_report(pack, intent, report)`: the latter validates custody and report
consistency, not a second full numeric run.

## Preserved verification

- [Archive tests](../outputs/portable-attachment-binding-tests-v1/release.json):
  32 analytic checks; prior version-1 archive bytes preserved.
- [Recording/identity tests](../outputs/portable-attachment-recording-tests-v1/result.json):
  34 checks, including comparison with a pinned pre-change geometry helper.
- [Numeric auditor tests](../outputs/portable-attachment-audit-tests-v1/review.json)
  and [metadata admission checks](../outputs/portable-attachment-admission-tests-v1/review.json):
  positive motion, genuine detachment, seams/IDs and malformed/oversized inputs.
  The later admission change preserves the numeric scorer and thresholds; earlier
  native evidence remains bound to its original source hash.
- [Real native analytic export](../outputs/portable-attachment-native-check-v1/result.json):
  three profiles, two bands each. Legacy and combed controls pass rest/weight
  checks but fail animated attachment at **17.005 mm**; the skin-consistent tuft
  control passes at **9.107e-9 m**. Each has 100 root-pose checks. These are analytic
  meshes exported by Blender, not generated avatars or appearance approval.
- Pipeline/Studio fixtures: **18 unique methods verified**, across the preserved
  [initial run](../outputs/portable-attachment-pipeline-tests-v1/receipt.json)
  and [five corrected-fixture reruns](../outputs/portable-attachment-pipeline-tests-v2/receipt.json).
  The initial run had five analytic-fixture errors; those failures remain saved.
  No native, provider or role calls ran in these orchestration tests.

The earlier [actual fox all-root audit](skin-consistent-fur.md) is separate
historical evidence. It is not a substitute for a new version-2 export or proof
that the new automatic pipeline stage has run on that asset.

## Actual saved-fox integration

Recorded November 2, 2026, PST. The existing locally authored fox, procedural
textures, native groom, 3,000 skin-consistent portable tufts, rig and four inherited
4-second clips were finished again solely to test the new exact-ID export. This
is not a new Meshy example. The source's hidden surfaces, units and pose remain
authored assumptions; no independent third-party rights verification is claimed.

| Artifact | Immutable path / SHA256 |
| --- | --- |
| Full finish | `outputs/fox-automatic-fur-attachment-finish-v1/manifest.json` — `a460c38fe267f560f5cda46d615e25b1a8e20e91eef7370ab50e230bdbcd2636` |
| Actual GLB | `outputs/fox-automatic-fur-attachment-finish-v1/model.glb` — `d863546e70aa921ac381bf5a1199ffddfb64d12904b29e765e60d0f84794a129` |
| Numeric report | `outputs/fox-automatic-fur-stage-v1/portable-root-audit-0/report.json` — `21bd42aced27c13dcef1781e5d7885882dd7ed0b22793b861c259dffdd564734` |
| Audit manifest | `outputs/fox-automatic-fur-stage-v1/portable-root-audit-0/manifest.json` — `f0d28abbeba03e69e78c7db58163f18d2991e9f8ae5f7692873e73f0011c59bf` |
| Controller receipt | `outputs/fox-automatic-fur-stage-v1/portable-root-audit-0.json` — `88d5071884f2c512a4fbace86117065a7734be0f6d36d71053cd1879ca251f10` |

The real gate checked **1,164,000 root-pose pairs**: all 3,000 roots at 97 times
in each of four clips. Maximum per-coordinate separation was **1.151203e-7 m**
(0.115 micrometres), below the unchanged 1e-5 m threshold. Rest midpoint error
was 1.151203e-7 m; named-weight error was 2.220446e-16. All failure counts are zero.
There are 30,418 native body IDs, 18,000 native card IDs, 62,609 exported vertices,
59,776 triangles and 20 joints. The different counts include export seam splits.

This component check uses real `pipeline.prepare`, source pins, a `Pipeline`
instance, its lock/guard/events, the actual bounded audit child and custody seals.
A fail-on-use AI-role object prevents planning/authoring/critic calls. The check
invokes only this stage, not the complete pipeline or deformation/action gates;
it is not a Studio acceptance or a claim those other gates passed. Cache
validation took 0.0304s and preserved the report and every audit-pack byte without
another numeric run. The prepared request's three-call API minimum is not a new
AI allowance; this component run makes zero role calls.

Exact one-shot commands, already finished; never rerun these sealed directories:

```sh
.venv/bin/python -B outputs/automatic-fur-fox-preparation-v1/execute.py --execute
.venv/bin/python -B outputs/automatic-fur-fox-audit-preparation-v1/execute.py --execute
```

The public finish's owned Blender child PID4480 exited0 in26.714s at
1,558,937,600B sampled RSS, under120s/2GiB/128MiB. CPU4478 exited; the controller
took27.671s with107,610,111B combined output. The audit child PID4940 exited0
in1.303s at198,410,240B sampled RSS under60s/512MiB/16MiB; isolated controller
PID4939 exited0 in1.554s at43,745,280B under90s/512MiB/64MiB. These are sampled
direct-child readings, not whole-process-tree or instantaneous memory ceilings.
No owned process remains running.

The initial CPU-stage preparation was revised **before any launch** to propagate
SIGTERM into owned-child cleanup. Both preparation versions and the exact initial
driver are retained; there was one numeric execution, not a retry. Native export
warnings are retained in `finish.log`; the independent exact-ID audit, not a
warning-free log, establishes the attachment result.

The new native720px portrait was directly inspected. It retains the fox, teal
scarf and drawing props but still reads as short plush, with hard part transitions
and a sharp tail-color edge. No density upgrade, full-action/loop review, improved
source stretch or new quality approval follows from this instrumentation change.
The20state PNGs remain256px development poses, not completed animation exports.
The prior source's raised-arm deformation failure remains relevant and unchanged
in scope; attachment does not repair it. Approved panda/dragon and gallery defaults
were not replaced. This HD model exceeds the frozen Weft20k-vertex/diffuse profile.

### Preservation and integration inventory

The separate bounded comparison at `outputs/automatic-fur-fox-preservation-v1`
passes exact typed primitive attributes/indices (excluding the new IDs), node,
skin, animation, material and embedded-image semantics against the previous
skin-consistent fox. All nine shared binding arrays are exact. The portrait,
20 draft poses and three atlases are all **pixel-identical**. Only the three
atlases also match encoded bytes; the21native PNG byte streams differ.
Only the configuration ID/prompt changed among compared configuration fields.
The initial sealing summary overstated PNG byte equality; its failed assertion
is retained and the summary corrected without repeating the comparison.
This proves preservation relative to
that previous output, not fidelity to an unknown original model or nicer fur.

One public `verify_pack` checks all102artifact hashes plus the local rig/motion
contract. Comparison PID5614 exited0 in0.768s at162,955,264B sampled RSS under
30s/384MiB/16MiB. No attachment/native rerun was needed.

`integration-inventory.json` records actual node IDs, bone palette order, parents,
local/world rest matrices, units, bounds and clip timings. Critical nodes: rig20,
root19, torso18, head2, body21 and portable coats22–24; tail3. Sketchbook4 follows
left hand5; pencil11 follows right hand12. Skin0 palette order is
`[19,18,2,0,1,3,7,6,5,4,10,9,8,14,13,12,11,17,16,15]`.
The explicit glTF rest matrices/inverse binds define the authored bind pose,
not an assumed T-pose or recovered anatomy. glTF is right-handed +Y up, metres
by producer convention; physical scale is unmeasured. No dragon is attached to
this fox. Local identity maps are not Weft simulated-joint mappings. Neither
Weft physics nor its importer was changed.

This gate does not establish convincing fur, source-body/shading fidelity,
native-strand behavior, hair-tip attachment, continuous animation, seamless loops,
collision/physics, action meaning, Weft compatibility or Muse-quality parity.
It does not complete the broader hands-off text-to-avatar goal. Required visual,
action, final-media and delivery checks remain in force.

## Saved compact evidence

[`evidence/automatic-fur-attachment-v1`](../evidence/automatic-fur-attachment-v1/manifest.json)
records751origins and358deduplicated exact snapshots totaling11,013,685B. Its
manifest SHA256 is `83c560d25d82535142ada2355506dc3a0334b83d83074f7f5fde9aa654416cde`.
It includes the small editable analytic controls, numeric archives, source pins,
observed failures and receipts. Large avatar GLB/Blend intermediates stay local
at their recorded hashes; this is not a complete remote asset backup. Packaging
PID6493 exited0 in0.260s under30s/256MiB/40MiB; RSS was unsampled, not zero.

The user subsequently accepted the prior appearance and prioritized a real
hands-off Studio run. That is the next workstream; this gate is complete and
should not trigger more appearance-upgrade experiments.
