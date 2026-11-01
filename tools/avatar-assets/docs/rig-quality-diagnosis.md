# Automatic rig spike diagnosis

**Updated 2026-11-01, 2:36 PM PST.** Read-only diagnosis of the two preserved attempts in
`outputs/studio-8a7e0d76b8b8427f93f52157`. No source geometry, existing design,
worker, prompt or pipeline was changed by this task. The host-side gate and
focused tests are implemented; the parent integrated its controller hook and
generic author guidance. A new automatic rig-quality result remains unverified.

## Finding: incomplete arm coverage, not a successful repair

The exact action remains: “Wave the right hand side to side in a clear greeting
while keeping the torso and feet steady.” The source is the existing generated
robot, SHA256 `5f00602dd727429277ae53c0e8940ba9162ea119e643ba554f081c00cc1b7e8c`.
This is development evidence, not fresh text-to-mesh acceptance.

Both agent designs use `base_bones: ["torso"]`, then independent rectangular
single-bone regions. Any point outside those appendage boxes stays fully attached
to the torso. Minimum selected-vertex counts prove that a box hits *some* geometry;
they do not prove complete limb coverage, correct anatomy or continuous weights.

| Evidence | Attempt 0 | Attempt 1 |
| --- | --- | --- |
| Numeric torso stability | Fails: 228.288 mm maximum world displacement | Passes: reported world displacement 0 |
| Largest retained edge extension | 515.592 mm | 325.472 mm |
| Rest → posed length for that edge | 8.466 → 524.058 mm | 3.739 → 329.211 mm |
| Stretch ratio for that edge | 61.90× | 88.05× |
| Highest ratio among four retained outliers | 124.17× | 116.08× |
| Endpoint ownership | Torso or torso-heavy feather → hand | Hand 100% → torso 100% |

The [actual 720px attempt-1 pose](../outputs/studio-8a7e0d76b8b8427f93f52157/model-1/renders/right_hand_wave/0003.png)
shows a long narrow strip extending down from the raised right arm and stretched
panels near the shoulder. The [original inspection view](../outputs/studio-8a7e0d76b8b8427f93f52157/inspection/front.png)
does not contain that extended strip. This is visibly unacceptable deformation;
neither material polish nor a valid moving skeleton resolves it.

### Exact mechanism

`scripts/creature_rig_math.py` computes each box's influence from its distance
**inside** the box. Positive feather has zero influence at the boundary and full
influence only farther inside. Attempt 0 put the hand box boundary exactly at
normalized X=0. Its outermost hand vertices therefore retained torso weights;
adjacent hand-weighted vertices traveled away. Similar outer-boundary feathering
also left distant vertices in the nominal torso role, explaining its erroneous
full-height extent and stability failure.

The repair agent received the actual outlier coordinates, endpoint weights and
pose sheet. Its saved explanation correctly targeted the outer-boundary leakage.
Attempt 1 extended outer boxes beyond 0..1 and set all feathers to zero, but left
internal gaps between appendage boxes. For the worst reported edge at 1.2 seconds:

- Vertex **33828**, normalized `[0.18768649, 0.43307271, 0.42581616]`, belongs to
  the hand box and is fully weighted to `right_hand`.
- Vertex **33863**, normalized `[0.19023999, 0.42689472, 0.42712215]`, lies just
  beyond the hand box's X maximum **0.19**, below the forearm box's Z minimum
  **0.435**, and outside the upper-arm box. It matches only the global torso box.
- The original 3.739 mm connecting edge becomes 329.211 mm long. The stationary
  endpoint correctly passes a numerical stillness test, but that test cannot
  determine that it is anatomically an arm vertex.

This identifies a concrete mask hole. It does not prove every joint in the source
is usable, that all remaining pose defects share this cause, or that unobserved
surfaces can be repaired automatically. Both attempts retain the same substantial
greeting curves; reducing them to a tiny movement is not a valid fix.

## Minimal reusable implementation

New helper: [deformation_gate.py](../avatar_assets/deformation_gate.py).

```python
audit_finish_pack(
    pack,
    expected_source_sha256=source_hash,
    expected_design_sha256=finish_config_hash,
    expected_state_ids=planned_state_ids,
    authored_height=planned_height,
)
```

The controller supplies the expected bindings; the helper verifies the native
manifest's exact report/design hashes, source identity, unchanged-source flag,
rig/design/height equality where authored, state order and authored sample times. Evidence is
bounded to 1 MiB per input JSON, eight states, eight poses per state, four retained
outliers per state, 64 bone names and a 64 KiB result. Malformed/nonfinite lengths,
locations, indices or weights, inconsistent ratios, missing evidence and changed
bindings fail closed. It does not load the mesh or invoke a renderer/model.

**Fixed gross-spike criterion:** reject any retained edge with both
`stretch_ratio > 8` and `excess_length > 0.03 * diagnostic_height`. For this 1.2 m
robot the absolute threshold is 36 mm. The conjunction avoids treating a large
ratio on a microscopic edge alone as a gross failure, while catching these
hundreds-of-millimeters spikes. This is a conservative authoring regression gate,
not a calibrated perceptual tolerance. It has no per-job threshold override.

Version 2 uses the native rest measurement's height where available; it never
substitutes the plan's requested height for an imported skin's measured scale.
The diagnostic height is the actual rest Z span, or the largest span only when
Z is degenerate. Native array hashes, meter units, applied coordinate space,
vertex/edge counts and actual bone names are bound to the finish report. Bone
names must also match the hash-bound exported rig verification and its model hash.
Every sampled state must identify the same rest-array hash and exact measured
sample count; totals must agree. This validates the trusted native measurement
receipt, not a new host-side reconstruction of those arrays.

Imported skins require this complete `deformation_measurement` marker. A nominal
1.2 m request for an actual 0.2 m mesh therefore gets a 6 mm gross-extension
threshold, not 36 mm. New authored markers must also match the bound authored
height/bones. Historical authored packs keep an explicitly labeled
`legacy_authored_samples` path using their bound authored height; a partially
present new marker cannot downgrade to that path. The API remains unchanged.

Place the gate immediately after successful native build/source-design binding,
before independent reload, exported renders and full media. Save the result with
the attempt; on failure, feed its exact edge IDs, sample times, normalized locations
and endpoint weights into the existing bounded author repair. Preserve the action
contract and fixed budgets. Continue the existing motion and visual checks only
after this gate passes; do not replace them.

### Scope limits

- A passing result is **`no_reported_severe_spike`**, not skin or visual approval.
  Existing native diagnostics retain only four worst-extension edges across at
  most eight selected poses. They can miss intermediate-time defects or another
  edge meeting the conjunction outside the retained four. Later native diagnostics
  should count severe edges over all already-evaluated edges before truncating the
  illustrative rows; full-frame checks are a separate bounded extension.
- The separate native adapter now captures rest/edge evidence before grooming on
  both authored and source-skin routes, checking unchanged vertex counts, edge
  counts and endpoint ordering at each sampled pose. Its actual-helper tests
  cover both routes; no real imported-rig native run has yet been verified for
  this addition. Old imported packs without the marker remain unsupported, not
  silently accepted from an empty list. This gate covers the pipeline's four-second
  authored actions on either rig route, not arbitrary provider clip formats.
- Small tears, collapses, self-intersection, implausible joints, metal bending,
  clipping and wrong action meaning still require independent review. A 35 mm
  defect may be unacceptable even though it is below this gross rejection bound.
- The helper assumes the controller has verified the full successful pack; it
  independently binds only the files it consumes, not every model/media artifact.

## Generic author guidance and remaining diagnostics

The earlier instruction to start with torso/root coverage was incomplete. The
parent has added base-field, whole-chain and inside-feather coverage guidance to
the reusable prompt. Preserve these principles instead of this robot's recipe;
the per-endpoint layer-membership diagnostics below remain proposed:

1. Explain inside-only feathering and last-layer precedence explicitly. Extend
   masks past object extrema where appropriate; do not leave the outer surface
   as a torso-weighted feather boundary.
2. Cover the **whole appendage chain**, including joint neighborhoods, before
   refining individual segments. Overlapping chain fields can blend adjacent
   intended bones; rigid overrides should follow real mechanical boundaries.
   A torso-only base is permissible only with complete appendage coverage.
3. Do not blindly switch every vertex to all bones or globally enlarge boxes:
   that can move the belly with an arm or assign one overlapping limb to another.
   Choose the base field and layer subsets from inspected geometry. Hard-surface
   bodies need joint-specific rigid regions, not broad blended metal panels.
4. Diagnose more than the first worst edge. For each retained endpoint, report
   matched layers and influence alpha, final overriding layer, closest excluded
   box face, and endpoint bone pair. This makes a mask gap distinguishable from
   a real joint boundary or fused contact. These additions are proposed, not yet
   implemented; the new gate already retains the existing actionable data.
5. Audit all interfaces in a repaired moving chain. Zeroing every feather can
   eliminate mixed-body motion while creating hard weight discontinuities. Never
   count disappearance of the original warning as complete repair without the
   other action/deformation checks. Unrepairable fused geometry stays unsupported.

## Verification and next regression cases

[Original focused tests](../tests/test_deformation_gate.py): **10 passed in 0.027 seconds**
(0.134460 seconds for the observed test runner). They cover strict two-part
thresholds, exact bindings, sampling, malformed/nonfinite/oversized evidence,
duplicate edges and explicit missing pre-rigged support. These are analytic report
fixtures, not an avatar-quality demonstration.

The version-2 suite passes **17 tests in 0.046 seconds**, adding real-versus-planned
scale mismatch, flat-mesh scale fallback, measured bone/export bindings, marker
schema/units/count/hash validation, per-state sample bindings and failed legacy
downgrade. These are analytic native-report fixtures. The separate adapter owner
reports **27 focused helper tests passed in 1.809 seconds**, including compilation
and testing of the actual native baseline helper on both routes; this is still
not a Blender execution or finished imported avatar. Read-only version-2 checks
also preserve the existing outcomes: both robot attempts reject, and
`outputs/muse-panda-dance-flight-v2` has no reported severe spike. Original report
hashes remain unchanged; no existing receipts were rewritten.

Read-only replay on the **actual preserved reports** rejected both attempts:
four offending retained edges each. The helper took **1.030 ms / 0.845 ms** for
attempts 0/1 in this run; RSS was not measured, and these are not maximum-size
performance claims. Original inputs' hashes were checked before and after.

Durable evidence: [verification receipt](../outputs/robot-deformation-gate-v1/verification.json),
[attempt 0 result](../outputs/robot-deformation-gate-v1/model-0-deformation-gate.json),
[attempt 1 result](../outputs/robot-deformation-gate-v1/model-1-deformation-gate.json),
[artifact manifest](../outputs/robot-deformation-gate-v1/manifest.json).
This version-1 proof folder retains exact original finish reports, designs, manifests, native
job receipts and prior action audits, plus implementation/test snapshots. Its
`retained-evidence` directories are evidence subsets, **not complete model packs**.
Its source snapshots stay immutable and describe the version-1 gate. No native,
AI or provider calls and no new credits were used for either gate revision.

Before claiming a general skin-quality improvement, add bounded geometry-level
regressions for: an interior chain-mask hole despite passing minimum coverage;
positive feather on an outer bound; a static torso endpoint adjacent to a moving
hand; proper rigid translation/rotation with no edge extension; a credible small
joint bend; tiny edges with high ratios but negligible absolute extension; seam
duplicate consistency; and proportionally scaled versions of the same defect.
Then run a new automatic development job with unchanged greeting semantics.
Preserve these failed attempts. This diagnosis/gate does not itself repair the rig.
