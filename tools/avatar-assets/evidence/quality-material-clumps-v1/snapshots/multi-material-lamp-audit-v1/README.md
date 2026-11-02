# Independent multi-material lamp audit

**Material/image and sampled rig/motion integration checks pass. This is not full shading-frame fidelity, visual quality approval, fresh generation or a Studio delivery.**

Source: `outputs/lamp-hd-build-v2/model.glb`, SHA256
`774eae69f51ae1a0cd49c61dd19abeab19b847ba8b72f393ad409b4582500949`.
Final: `outputs/multi-material-lamp-finish-v1/model.glb`, SHA256
`fc3815510f6d436cd6bec470fd57a7cd976da4f2528cec30151eb8fc832195a0`.
The source is an existing agent-authored procedural lamp, not a neural-generation
result. The public finisher produced this local derivative without provider or
AI-role calls.

## Verified against actual GLB data

- All four embedded images are byte-identical: two different base-color images
  and two normal images. All eleven named materials retain base-color constants,
  metallic/roughness, emissive, normal scale, UV/texture/sampler routes, alpha
  and culling semantics. Normal scales remain 1. No shared texture was fabricated.
- Positions, UV0 and named bone weights compare exactly. Every oriented
  position/UV/weight triangle and source point group is accounted for in each
  material. Export splitting changes 37,762 vertices to 38,063, while all 44,754
  triangles remain. This is explicit correspondence, not equality inferred from
  counts. The saved correspondence map is `correspondence.npz`.
- All eight bone names and their named parents remain. Node IDs reorder, and the
  mesh moves from an identity armature parent to scene-root. Named world-rest
  matrices differ by at most 1.45223e-7, below the declared 1e-6 component bound.
  Units/axes remain the authored meter / glTF Y-up convention, not measured reality.
- The four original skeletal clips retain names and 0–4-second duration. All
  38,063 final vertices were replayed at nine times per clip. Maximum source/final
  weighted position error is 4.03784e-7 m, below the unchanged 1e-5 m comparison
  bound. Maximum observed displacements are 0.07585 m idle, 0.35973 m crafting,
  0.38071 m celebration and 0.27740 m working. These are actual skin movement,
  not an unrigged transform or repeated stills.
- `source_color_sampling` reports skipped, no consumers, zero selected vertices
  and zero selection calls. No groom/cards/fringe/undercoat was added. The exact
  dedicated normal-UV receipt is supported, with coherent UV0 on brass and steel.
- Twenty-one manifest-bound PNGs verify as single-frame RGB, 256×256: one
  portrait plus five poses for each clip. These are integration drafts. No MP4
  or full-media quality acceptance was produced by this finish.

## Preserved limits

Normals and tangents are **not byte-preserved**. Measured minimum-candidate
component differences reach 0.000413805 and 0.000300005 respectively. Several
coincident source points have different corner normals, so those comparisons
are lower bounds, not a unique corner-frame match. The audit does not certify
complete shading-frame equivalence or hide these differences as a pass. No
normal/tangent repair or threshold change was attempted.

Nine samples do not prove every interpolated frame, seamless loops, collisions,
mechanical physics, semantic action quality or Weft compatibility. The raw public
rig report also retains its generic generated/humanoid assumption text; this
audit preserves the correct procedural-lamp provenance rather than adopting it.
Root's separate three-image review remains a limited visual check, not a 720px
or final-media acceptance.

## Cost, bounds and execution

One CPU-only audit ran under 30 s / 512 MiB / 16 MiB bounds: 3.377 s observed
runner wall time, 191,004,672 B sampled child RSS, exit 0 verified. The child's
own high-water RSS was 194,854,912 B. Exact script, sources, declared tolerances,
input hashes, runner receipt and reports are retained. No native, provider,
browser, model-role call or unchanged passing test rerun occurred here.

The separate public finish took 6.846 s overall; its native child took 6.230 s
with 561,233,920 B sampled RSS. Its CPU preflight receipt contains only the
initial loop sample, not complete preflight timing/peak; do not report that
sample as full preflight resource measurement. Both original controller/native
processes exited. No model, source recipe, production code or shared document
was changed by this audit.
