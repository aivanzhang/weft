"""Resolve a skinned rig's bones to Weft's ten semantic body roles, or refuse with reasons.

Generated rigs are usually bipedal but inconsistently named: the same elbow appears as
`arm_l_lower`, `elbow_L` or `forearm_l`. Hand-written mappings per asset are what kept
custom avatars from importing automatically. This resolver scores bone names against
per-role synonyms, requires an unambiguous winner, and then checks the skeleton's actual
structure (distinct sides, proximal bone is an ancestor of its distal bone, a torso above
both limb pairs). A rig that does not satisfy those checks is refused, never guessed:
Weft drives mapped nodes with absolute body frames, so a wrong mapping silently deforms
the mesh instead of failing.

Role resolution is naming and tree structure only. It is not anatomy recognition, and it
does not establish that the mesh is skinned well enough to pose cleanly.
"""

import re

ROLES = ("mobile_root", "torso",
         "left_shoulder_body", "left_elbow_body", "left_hip_body", "left_knee_body",
         "right_shoulder_body", "right_elbow_body", "right_hip_body", "right_knee_body")
LIMB_ROLES = tuple(role for role in ROLES if role not in ("mobile_root", "torso"))
# Ordered most to least specific; a later term never overrides an earlier match.
_LIMB_TERMS = {
    # "arm"/"leg" are deliberate last resorts for Mixamo-style rigs, where `LeftArm` is the
    # humerus and `LeftLeg` the shin; more specific terms always win over them.
    "shoulder": ("upperarm", "upper_arm", "arm_upper", "humerus", "shoulder", "clavicle", "upperlimb", "arm"),
    "elbow": ("forearm", "fore_arm", "lowerarm", "lower_arm", "arm_lower", "elbow", "radius"),
    "hip": ("upperleg", "upper_leg", "leg_upper", "upleg", "thigh", "femur", "hip", "upperhind"),
    "knee": ("lowerleg", "lower_leg", "leg_lower", "shin", "calf", "knee", "tibia", "leg"),
}
_TORSO_TERMS = ("chest", "torso", "upperbody", "upper_body", "ribcage", "thorax", "spine2", "spine_02")
_ROOT_TERMS = ("weftavatarroot", "root", "hips", "pelvis", "body_root", "armature", "rig")
_LEFT = ("left", "_l", "l_", ".l", "lft")
_RIGHT = ("right", "_r", "r_", ".r", "rgt")


class RigMappingError(ValueError):
    """Raised when a rig cannot be mapped unambiguously to the ten roles."""


def _normalize(name):
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _tokens(name):
    return re.split(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])", name or "")


def _side_of(name):
    """Return 'left', 'right' or None from side markers, preferring whole tokens."""
    lowered = [token.lower() for token in _tokens(name) if token]
    if "l" in lowered and "r" not in lowered:
        return "left"
    if "r" in lowered and "l" not in lowered:
        return "right"
    for token in lowered:
        if token in ("left", "lft"):
            return "left"
        if token in ("right", "rgt"):
            return "right"
    flat = _normalize(name)
    left = any(marker.strip("_.") in flat for marker in ("left", "lft"))
    right = any(marker.strip("_.") in flat for marker in ("right", "rgt"))
    if left != right:
        return "left" if left else "right"
    return None


def _without_side(name):
    """Drop side tokens so `arm_l_lower` compares as `armlower`, not `armllower`."""
    keep = [token for token in _tokens(name)
            if token and token.lower() not in ("l", "r", "left", "right", "lft", "rgt")]
    return _normalize("".join(keep))


def _limb_kind(name):
    """Return the most specific limb term matched, or None."""
    flat = _without_side(name)
    best = None
    for kind, terms in _LIMB_TERMS.items():
        for rank, term in enumerate(terms):
            if _normalize(term) in flat:
                if best is None or rank < best[0]:
                    best = (rank, kind)
                break
    return best[1] if best else None


def _ancestors(parents, node):
    chain, current = [], parents.get(node)
    while current is not None and current not in chain:
        chain.append(current)
        current = parents.get(current)
    return chain


def resolve_body_roles(bone_names, *, parents=None, root_name=None):
    """Return {role: bone name} for a bipedal-compatible rig, or raise RigMappingError.

    bone_names maps each bone's node index to its name. parents maps node index to parent
    index for the structural checks; without it only naming is validated.
    """
    if not isinstance(bone_names, dict) or not 1 <= len(bone_names) <= 256:
        raise RigMappingError("Rig mapping requires a bounded bone inventory")
    if len(set(bone_names.values())) != len(bone_names):
        raise RigMappingError("Rig bone names must be unique")
    candidates = {role: [] for role in LIMB_ROLES}
    for node, name in bone_names.items():
        kind = _limb_kind(name)
        side = _side_of(name)
        if kind and side:
            candidates[f"{side}_{kind}_body"].append(node)
    mapping, reasons = {}, []
    # Distal joints first: they disambiguate their own proximal bone below.
    for role in ("left_elbow_body", "right_elbow_body", "left_knee_body", "right_knee_body",
                 "left_shoulder_body", "right_shoulder_body", "left_hip_body", "right_hip_body"):
        found = candidates[role]
        if not found:
            reasons.append(f"no bone matched {role}")
            continue
        if len(found) == 1:
            mapping[role] = found[0]
            continue
        distal = role.replace("_shoulder_", "_elbow_").replace("_hip_", "_knee_")
        resolved_distal = mapping.get(distal) if distal != role else None
        if resolved_distal is not None and parents:
            # A clavicle and a humerus both read as "shoulder"; the driven bone is the one
            # directly above the elbow, so prefer the closest ancestor of the distal joint.
            chain = _ancestors(parents, resolved_distal)
            ranked = sorted((chain.index(node), node) for node in found if node in chain)
            if ranked:
                mapping[role] = ranked[0][1]
                continue
        names = ", ".join(sorted(bone_names[node] for node in found))
        reasons.append(f"{role} is ambiguous between {names}")
    if reasons:
        raise RigMappingError("Rig does not map to Weft's body roles: " + "; ".join(sorted(reasons)))

    def pick(terms, exclude):
        for rank, term in enumerate(terms):
            matched = [node for node, name in bone_names.items()
                       if _normalize(term) in _normalize(name) and node not in exclude]
            if len(matched) == 1:
                return matched[0]
            if len(matched) > 1:
                # Prefer the shallowest match so `spine`/`chest` beats a nested duplicate.
                if parents:
                    matched.sort(key=lambda node: len(_ancestors(parents, node)))
                    if len(_ancestors(parents, matched[0])) < len(_ancestors(parents, matched[1])):
                        return matched[0]
                raise RigMappingError(f"Ambiguous bone for {terms[0]}: "
                                      + ", ".join(sorted(bone_names[node] for node in matched)))
        return None

    assigned = set(mapping.values())
    torso = pick(_TORSO_TERMS, assigned)
    if torso is None:
        raise RigMappingError("Rig does not map to Weft's body roles: no torso bone matched")
    mapping["torso"] = torso
    if root_name is not None:
        mapping["mobile_root"] = root_name
    else:
        root = pick(_ROOT_TERMS, assigned | {torso})
        if root is None:
            raise RigMappingError("Rig does not map to Weft's body roles: no root bone matched")
        mapping["mobile_root"] = root

    resolved = {role: (bone_names[node] if node in bone_names else node) for role, node in mapping.items()}
    if len(set(resolved.values())) != len(resolved):
        raise RigMappingError("Rig mapping assigned one bone to several roles")
    if parents:
        for side in ("left", "right"):
            for proximal, distal in (("shoulder", "elbow"), ("hip", "knee")):
                upper, lower = mapping[f"{side}_{proximal}_body"], mapping[f"{side}_{distal}_body"]
                if upper not in _ancestors(parents, lower):
                    raise RigMappingError(
                        f"{resolved[f'{side}_{distal}_body']} is not below {resolved[f'{side}_{proximal}_body']}; "
                        "the rig's structure contradicts its names")
        # Arms hang off the torso; legs hang off the root/pelvis, so only the arms are
        # required below the torso. Legs must not sit under an arm, which is what a
        # confused arm/leg naming would produce.
        for side in ("left", "right"):
            shoulder = mapping[f"{side}_shoulder_body"]
            if torso not in _ancestors(parents, shoulder):
                reasons.append(f"{resolved[f'{side}_shoulder_body']} is not below the torso")
        shoulders = {mapping["left_shoulder_body"], mapping["right_shoulder_body"]}
        for side in ("left", "right"):
            hip = mapping[f"{side}_hip_body"]
            if shoulders & set(_ancestors(parents, hip)):
                reasons.append(f"{resolved[f'{side}_hip_body']} sits under an arm")
        if reasons:
            raise RigMappingError("Rig structure contradicts its names: " + "; ".join(sorted(reasons)))
    return resolved


def resolve_from_glb(glb, *, root_name=None):
    """Resolve roles from a parsed GLB's first skin; returns (mapping, report)."""
    skins = glb.doc.get("skins", [])
    if len(skins) != 1:
        raise RigMappingError("Automatic rig mapping expects exactly one skin")
    joints = skins[0].get("joints") or []
    nodes = glb.doc.get("nodes", [])
    bone_names = {node: nodes[node].get("name") for node in joints
                  if isinstance(nodes[node].get("name"), str) and nodes[node].get("name")}
    if len(bone_names) != len(joints):
        raise RigMappingError("Every skin joint requires a unique name for automatic mapping")
    parents = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = index
    mapping = resolve_body_roles(bone_names, parents=parents, root_name=root_name)
    return mapping, {"profile": "automatic-rig-role-mapping-v1", "bones": len(bone_names),
                     "mapping": dict(mapping),
                     "scope": "Bone-name synonyms checked against skeleton structure; not anatomy "
                              "recognition, skin quality or posed-deformation qualification"}
