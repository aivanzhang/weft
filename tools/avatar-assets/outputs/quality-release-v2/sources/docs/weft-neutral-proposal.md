# Authored visual-neutral proposal v1

**This is a user-authorized design proposal, not an observed or canonical Weft neutral pose.** The user asked the asset-authoring session to define a neutral convention. The numerical proposal is [neutral-proposal-v1.json](../examples/weft/neutral-proposal-v1.json).

Weft's supplied contract names ten body frames but does not provide their actual neutral coordinates, rotations, joint-angle vector or calibrated scale. This proposal fills the **authoring assumption** needed for provisional visual bindings. It does not establish actual Weft alignment or change physics, actions, rewards, imports or recording behavior.

## Explicit convention

Use a right-handed authoring frame: **+Z up, −Y front**. At the front view, +X is viewer-right/anatomical left; −X is viewer-left/anatomical right. Coordinates are authored meters with an uncalibrated numerical scale of 1 into the proposed target units.

Every frame below has identity rotation `[0, 0, 0, 1]` in xyzw quaternion order. Positions are **world-space**, not relative to the preceding row or a simulated parent.

| Body frame | Proposed position `[x, y, z]` |
| --- | --- |
| `mobile_root` | `[0, 0, 0]` |
| `torso` | `[0, 0, 1.0]` |
| `left_shoulder_body` | `[0.25, 0, 1.4]` |
| `right_shoulder_body` | `[-0.25, 0, 1.4]` |
| `left_elbow_body` | `[0.45, 0, 1.1]` |
| `right_elbow_body` | `[-0.45, 0, 1.1]` |
| `left_hip_body` | `[0.15, 0, 0.9]` |
| `right_hip_body` | `[-0.15, 0, 0.9]` |
| `left_knee_body` | `[0.15, 0, 0.5]` |
| `right_knee_body` | `[-0.15, 0, 0.5]` |

The landmarks suggest a standing figure with arms angled outward and downward. They neither recover anatomy nor force the authored mesh into a new pose. Identity target-frame rotations do **not** mean that Weft's 19 joint angles are zero. The actual neutral joint vector, ordering and root translation remain unknown and are `null` in the proposal.

## Exact axis conversion

The source GLB uses right-handed **+Y up, +Z front**. Convert it to the proposed frame with a positive quarter-turn around X:

```text
orientation_xyzw = [sqrt(0.5), 0, 0, sqrt(0.5)]
                ≈ [0.7071067811865476, 0, 0, 0.7071067811865476]
O = Rx(+π/2)
O · [x, y, z] = [x, -z, y]
scale = 1
G = O · S(scale)
```

This maps source up `[0, 1, 0]` to `[0, 0, 1]`, and source front `[0, 0, 1]` to `[0, -1, 0]`, without reflection. This quaternion is exact for the **proposal's** axes; it is not evidence that actual Weft uses those axes. Matrices act on column vectors; serialize matrices column-major and quaternions xyzw.

## Bind-preserving offsets

For each bound node, define:

- `B_world`: its actual exported glTF **global** default/bind transform, accumulated through its ancestors.
- `F0`: the proposed world transform for the selected target body frame.
- `C = T(binding.translation) · R(binding.rotation_xyzw)`: the binding's rigid offset.
- `G = O · S(scale)`: the common axis/scale conversion. The proposal chooses the overall model alignment `M = G`.

The supplied consumer equation and its required offset are:

```text
J(q) = F_body(q) · C · G
C    = inverse(F0) · G · B_world · inverse(G)
```

The order is intentional:

```text
J(q0_proposed)
  = F0 · [inverse(F0) · G · B_world · inverse(G)] · G
  = G · B_world
```

Extract the translation from `C` and convert its 3×3 rotation to a unit xyzw quaternion. Check that the rotation is orthonormal with determinant +1 and that no residual scale/shear remains; the binding schema has no per-bone scale or shear. Do not replace `B_world` with a bone's local transform. Do not simply copy its local quaternion or set every offset to zero.

Because the proposed `F0` rotations are identity and scale is 1, the same computation simplifies to:

```text
binding.translation = O · B_world.translation - F0.translation
binding.rotation    = O · B_world.rotation · inverse(O)
```

Confirm `B_world · inverse_bind_matrix ≈ identity` for source bind geometry. Then neutral skin matrices are `G`, so the original skinned shape is preserved under the common axis conversion. If the authored default pose differs from its bind pose, preserve and disclose that distinction rather than assuming this identity.

The JSON records a finite matrix-algebra check using all ten proposed frames and an independently constructed translated/rotated source transform. That check verifies the equation and axis conversion—not actual Weft forward kinematics.

## Node roles and descendants

Use actual exported glTF node indices. The `AvatarRig` ancestor containing the mesh and all skin joints is the retarget root and maps to `mobile_root`; the source root bone alone may not contain the mesh.

Map torso, upper arms, forearms, thighs and shins to the corresponding torso, shoulder, elbow, hip and knee frames. Check anatomical sides from the source transforms and geometry, not names alone. In this proposal anatomical left is +X.

Unmapped head, hand, foot and accessory descendants retain authored local rest transforms. A shoulder dragon follows its real attachment hierarchy; it does not gain a physical body or independently inferred motion. The 19 live simulated joint angles and root translation remain consumer-owned. Authored clips are provenance, not replacements for the live pose.

## Correction with actual consumer evidence

When the Weft owner supplies actual neutral world matrices `F_actual(q0)`, recompute only the metadata offsets:

```text
C_actual = inverse(F_actual(q0)) · G · B_world · inverse(G)
```

Keep geometry, hierarchy, skin weights and inverse-bind matrices unchanged. This preserves the same authored neutral model alignment `M = G`. If verified axes, units or an explicitly chosen global placement require a different alignment `M`, use the general form:

```text
C_actual = inverse(F_actual(q0)) · M · B_world · inverse(G)
```

The consumer must then verify the actual neutral pose, individual joint perturbations, root translation, left/right agreement and attachment behavior. Neutral agreement alone does not prove moving joints remain visually connected when source proportions differ from the simulated body. Record remaining alignment limits explicitly. No actual Weft compatibility or calibration is claimed until those consumer checks pass.
