# Protected-skin repairs

A finishing repair must not rewrite a skin that already passed its checks.
This mode limits an AI repair to an explicit patch over an authenticated,
fixed baseline. It does **not** weaken visual, deformation or action requirements.
See [the current handoff](../TODO.md) for live ownership and final acceptance.

**Current user direction:** retain the already accepted appearance and finish the
hands-off website/backend workflow. This safeguard prevents regressions; it is
not authorization for more fur tuning. A new request may explicitly freeze that
appearance, but prior failed reviews remain failed—user preference does not
retroactively turn an old critic result into a pass.

## Eligibility and protected fields

The latest eligible review must pass identity, framing, deformation and motion.
Its remaining failures must concern appearance or a whole-body root trajectory
(`travel`, `rhythm` or `return_pose` tied to the sole skeleton root). A failed
limb/wing articulation, identity or deformation check does not qualify.
The baseline also needs source-bound passing numerical deformation, action and
portable-attachment evidence, exact paid author/reviewer records and image hashes.
Source, plan, intent and continuation ancestry must match the current job.

The coordinator protects:

- Rest geometry, rig, bone hierarchy, skin-weight layers and constraints.
- Attachments, part rigs and action-role bindings.
- Every joint rotation, non-root translation, key time and interpolation mode.
- Existing topology/smoothing and other fields excluded from the finishing schema.

Permitted fields come from the closed finishing schema, not arbitrary JSON keys.
For an eligible failed root-trajectory requirement, the author may supply metre
translation vectors for the existing keys, in their exact order. No new keys,
bones or states are allowed. **Appearance-only repairs cannot change root motion.**
A repair needing protected changes must return `unsupported`; a later rejected
candidate cannot unlock the fixed baseline's skin or articulation.

“Fixed” means the **skin and non-root articulation**, not a forever-unchanged
finishing seed. The baseline may advance only after a new authenticated review
qualifies, numerical and visual deformation checks pass, and the protected skin/
non-root projection exactly matches the prior baseline. This retains verified
finishing and allowed root-motion improvements. Once a hop/trajectory requirement
passes, a subsequent appearance-only patch must preserve those root values.
A numerically failed patch never becomes the baseline. The latest qualifying
critique remains the active repair target; older reviews remain provenance, not
instructions to undo newer verified improvements.

## Transport and saved outputs

The AI returns schema version 3 with `status`, `reason` and `patch`:

- `finish_updates`: every allowed property is present; `null` inherits the fixed
  baseline, not a previous rejected patch.
- `remove_finish_fields`: explicit allowed removals; duplicate or conflicting
  update/removal requests fail validation.
- `root_translations`: `null` preserves motion; otherwise only schema-listed
  state IDs and exact existing key counts are valid.

The pure composer creates an effective version-2 author answer without modifying
raw responses. Existing version-2 and legacy answers remain readable. Each ready
patch records these additive files under `composition/author-<attempt>/`:

| File | Purpose |
|---|---|
| `baseline.json` | Exact fixed answer, allowed scope and authenticated evidence |
| `effective-answer.json` | Deterministically composed answer sent to normal validation/build |
| `receipt.json` | Request, raw answer/paid receipt, baseline, schema, effective-answer and protected-field hashes |

Readback recomposes and verifies these bindings, including the baseline supplied
in the paid prompt. The patch is bounded to 128 KiB; digest input to 256 KiB.
Root vectors retain finite schema bounds of −2 to +2 metres; normal semantic,
geometry, native resource and final-quality checks still apply. There is no
extra provider call for composition and no new spending or retry authorization.

## Why this was needed

[The terminal summary](../outputs/6c2a-terminal-summary-v1/manifest.json) records
three successfully built candidates that all failed the unchanged deformation
gate before media or review. Relative to the previous passing skin, the authors
changed weight layers/constraints while adjusting coat and root motion. Bone
rest definitions and rotation keys stayed unchanged. The
[weight diagnosis](../outputs/6c2a-weight-regression-diagnosis-v1/manifest.json)
retains concrete adjacent-vertex weight discontinuities; this is not permission
to relax the rejection threshold or blame root translation alone.

[The accepted-appearance trace](../outputs/accepted-panda-export-trace-v1/manifest.json)
also distinguishes the user-accepted native-first demonstration from its lighter,
sparse/flecked browser coat. There is no proven Muse-quality exported recipe to
silently restore. Finishing dimensions should be compared relative to body height,
not copied blindly from a differently scaled model.

## Local proof and remaining limits

[The native proof](../outputs/protected-skin-native-proof-v1/result.json) is an
**analytic derivative**, not a fresh AI-generated asset: it combines the fixed
passing baseline with copied finishing/root fields from a rejected child.
The actual native build exited successfully in **43.641 s**, sampled
**1,786,314,752 bytes** direct-child peak RSS and wrote **89,330,732 bytes**.
The complete check took **44.235 s** according to its result receipt. Limits stayed
600 s / 2 GiB direct-child RSS / 512 MiB output. Inputs remained unchanged.

[The comparison](../outputs/protected-skin-native-proof-v1/comparison.json) confirms
matching rest-position/edge hashes, bounds, vertices, bone names and sample counts.
The unchanged deformation gate passed: maximum reported excess 40.575 mm,
maximum ratio 6.1235, no retained offending edge. This is a bounded gross-spike
check, not zero deformation, all-frame correctness or recovered anatomy.

No appearance, visible-hop, collision, physics or final delivery acceptance follows
from that proof. No provider call was used.

## Frozen test evidence

[Composition/integration evidence](../outputs/protected-repair-composition-tests-v1/manifest.json)
records **77 distinct passing analytic checks**: 13 new and 64 affected. They cover
protected-field rejection, appearance-only root immutability, failed-patch baseline
retention, verified baseline advancement, current-review preservation and the real
coordinator's full-answer-to-patch transition. Native/provider calls were mocked;
these tests are not generated assets. Fixture setup failures remain recorded.

Manifest SHA:
`6d0bb8b100194412fd533333a2e7c115f3bb94d1a729baf2054d8b0683d9cc29`.
Baseline evidence remains bounded to 48 files / 128 MiB, baseline JSON to 256 KiB,
composition ancestry to 24 levels and each receipt to 16 KiB. The three-file
composition record does not copy large models. Actual consumer/preflight checks
and any future website run remain separately recorded in [the handoff](../TODO.md);
these unit results do not claim their completion or grant more paid work.
