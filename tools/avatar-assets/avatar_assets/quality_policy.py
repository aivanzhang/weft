"""Subject-specific finishing and full-resolution evidence policy, not a quality score."""

from copy import deepcopy
import math
import re

from avatar_assets.safety import require, slug


MATERIALS = ("fur", "hair", "scales", "metal", "ceramic", "skin", "fabric", "glass", "plastic", "other")
VIEWS = ("front", "side", "back", "three_quarter")
REPRESENTATIONS = ("native", "exported")


def validate_surface_intent(value, *, ready=True):
    require(isinstance(value, list) and len(value) <= 8 and (value or not ready),
            "Surface intent requires one to eight observed/intended regions")
    names = []
    for region in value:
        require(isinstance(region, dict) and set(region) == {"region", "material", "finish"},
                "Invalid surface intent fields")
        for key, maximum in (("region", 200), ("finish", 600)):
            require(isinstance(region[key], str) and 0 < len(region[key]) <= maximum and region[key].strip(),
                    "Invalid surface intent " + key)
        require(region["material"] in MATERIALS, "Invalid surface material")
        names.append(region["region"])
    require(len(set(names)) == len(names), "Duplicate surface region")
    return deepcopy(value)


def build_quality_policy(request, plan, inspection=None):
    """Require measured geometry before masks; never supply species-specific coordinates."""
    surfaces = validate_surface_intent(plan.get("surface_intent", []), ready="surface_intent" in plan and plan["status"] == "ready")
    materials = {region["material"] for region in surfaces}
    fibers = bool(materials & {"fur", "hair"})
    mixed_coat = fibers and bool(materials - {"fur", "hair"})
    observed = {"status": "missing", "source_sha256": None, "bounds_blender_z_up": None}
    if inspection is not None:
        require(isinstance(inspection, dict), "Invalid geometry inspection")
        native, preflight = inspection.get("native", {}), inspection.get("preflight", {})
        require(isinstance(native, dict) and isinstance(preflight, dict), "Invalid geometry inspection reports")
        if "bounds_blender_z_up" in native:
            bounds = native["bounds_blender_z_up"]
            require(isinstance(bounds, dict) and set(bounds) == {"min", "max"}, "Invalid inspected bounds")
            for vector in bounds.values():
                require(isinstance(vector, list) and len(vector) == 3 and all(type(x) in (int, float) and math.isfinite(x) for x in vector),
                        "Invalid inspected bounds")
            require(all(a < b for a, b in zip(bounds["min"], bounds["max"])), "Degenerate inspected bounds")
            digest = native.get("source_sha256")
            require(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest)
                    and preflight.get("sha256") == digest, "Inspection source hashes do not match")
            require(native.get("source_unchanged") is True and preflight.get("source_unchanged") is True,
                    "Inspection did not preserve source")
            observed = {"status": "observed", "source_sha256": digest, "bounds_blender_z_up": deepcopy(bounds)}
    states = plan.get("states", [])
    require(isinstance(states, list) and len(states) <= 8, "Invalid policy states")
    ids = [slug(state["id"]) for state in states]
    require(len(set(ids)) == len(ids), "Duplicate policy state")
    return {
        "schema_version": 1, "mode": "high_quality", "surface_intent": surfaces,
        "inspection": observed,
        "candidate_settings": {"resolution": 720, "samples": 16, "normal_strength": .3 if fibers else .6,
                               "pose_resolution": 720, "review_views": True},
        "required_features": (["native_groom", "portable_fur", "regional_groom_flow", "source_colored_cards"] if fibers else [])
                             + (["measured_non_fur_masks"] if mixed_coat else []),
        "fur_requested": fibers, "mixed_coat": mixed_coat,
        "review": {"minimum_side": 720, "portrait_minimum_side":
                   512 if request.get("preset") == "draft" else 1600 if request.get("preset") == "muse" else 720,
                   "views": list(VIEWS), "representations": list(REPRESENTATIONS), "state_ids": ids,
                   "state_pose_minimum_side": 720, "full_motion_audit_required": True},
        "authoring_rules": [
            "Choose masks, bone locations and coat directions from actual source geometry and coordinate guides; no subject templates.",
            "Use clean neutral geometry with space around moving parts; coarse carved fur and fused contacts are not solved by denser grooming.",
            "Keep scale relief, metal highlights, glass and mechanical creases; do not add fibers or globally smooth every subject.",
            "For fur/hair, preserve silhouette, patch colors and directional length changes; inspect native strands and exported cards independently.",
            "Only diagnose-and-repair color, normals or geometry with their bounded controls; retained textures or high counts are not added detail.",
            "A departing fused part needs a separately reviewed geometry derivative; hidden separation/cap repair is not automatic.",
        ],
        "limitations": ["Necessary finishing/evidence coverage, not a visual quality guarantee or human approval.",
                        "Long hair, cloth dynamics, arbitrary prop interaction and facial performance remain unsupported by this backend.",
                        "No new provider call, paid retry, native size limit or repair allowance is authorized by this policy."],
    }


def evaluate_finish_policy(config, policy):
    """Structural prerequisites cannot prove coat density, likeness or material appearance."""
    require(isinstance(config, dict) and isinstance(policy, dict) and policy.get("mode") == "high_quality", "Invalid finishing policy input")
    require(all(field not in config or isinstance(config[field], dict) for field in ("groom", "groom_flow", "portable_fur")),
            "Finishing policy requires validated finishing fields")
    failures = []
    if type(config.get("pose_resolution")) is not int or config["pose_resolution"] < 720 or config.get("review_views") is not True:
        failures.append("High-quality review requires actual 720px state poses and native multiview renders")
    if not policy["surface_intent"]:
        failures.append("No structured surface intent; material-specific finishing is unclassified")
    if policy["inspection"]["status"] != "observed":
        failures.append("Actual source bounds and hash-bound inspection are required before region authoring")
    elif config.get("source_sha256") != policy["inspection"]["source_sha256"]:
        failures.append("Finishing and inspection source hashes differ")
    if policy["fur_requested"]:
        for field in ("groom", "portable_fur", "groom_flow"):
            if not config.get(field):
                failures.append("Requested fur/hair requires " + field + "; painted texture is not a substitute")
        if config.get("groom") and config["groom"].get("parents", 0) * (config["groom"].get("children", 0)+1) < 1000:
            failures.append("Sparse diagnostic groom cannot stand in for the required coat")
        settings = config.get("portable_fur", {}).get("settings", {})
        if settings.get("sample_root_color") is not True or settings.get("profile") != "combed-v1":
            failures.append("Portable fur needs source-colored combed cards; inspect their actual fidelity separately")
        if policy["mixed_coat"] and not (config.get("groom_exclusions") or config.get("groom_excluded_parts")):
            failures.append("Mixed furry/non-furry surfaces need measured exclusions or exact part masks")
    elif config.get("groom") or config.get("portable_fur"):
        failures.append("Finishing added fibers absent from the intended materials")
    return {"passed": not failures, "failures": failures, "visual_acceptance": False,
            "scope": "Structural material/representation prerequisites only; appearance still needs full-resolution review."}


def evaluate_quality_evidence(policy, evidence):
    """Check actual per-pose resolution, not the total dimensions of a large montage."""
    require(isinstance(policy, dict) and policy.get("mode") == "high_quality", "Invalid quality evidence policy")
    require(isinstance(evidence, list) and len(evidence) <= 128, "Too much quality evidence")
    permitted = {"id", "kind", "representation", "view", "width", "height", "state_ids", "pose_width", "pose_height"}
    seen, views, states, portraits = set(), set(), set(), set()
    review = policy["review"]
    for item in evidence:
        require(isinstance(item, dict) and set(item) <= permitted
                and {"id", "kind", "representation", "width", "height"} <= set(item), "Invalid quality evidence fields")
        identifier = item["id"]
        require(isinstance(identifier, str) and 0 < len(identifier) <= 200 and identifier not in seen, "Invalid or duplicate quality evidence ID")
        seen.add(identifier)
        kind, representation = item["kind"], item["representation"]
        require(kind in ("view", "portrait", "state") and representation in REPRESENTATIONS, "Invalid quality evidence kind")
        for field in ("width", "height", "pose_width", "pose_height"):
            if field in item:
                require(type(item[field]) is int and 1 <= item[field] <= 16384, "Invalid evidence image dimensions")
        side = min(item["width"], item["height"])
        if kind == "view":
            require(item.get("view") in VIEWS, "Invalid evidence view")
            if side >= review["minimum_side"]:
                views.add((representation, item["view"]))
        elif kind == "portrait":
            if side >= review["portrait_minimum_side"]:
                portraits.add(representation)
        else:
            names = item.get("state_ids")
            require(isinstance(names, list) and 1 <= len(names) <= 8 and all(isinstance(name, str) for name in names) and len(set(names)) == len(names)
                    and set(names) <= set(review["state_ids"]), "Invalid state evidence names")
            require({"pose_width", "pose_height"} <= set(item)
                    and item["pose_width"] <= item["width"] and item["pose_height"] <= item["height"],
                    "State evidence needs actual per-pose dimensions")
            if min(item["pose_width"], item["pose_height"]) >= review["state_pose_minimum_side"]:
                states.update((representation, name) for name in names)
    missing = []
    for representation in review["representations"]:
        missing.extend(representation + " view: " + view for view in review["views"] if (representation, view) not in views)
        missing.extend(representation + " full-resolution state: " + name for name in review["state_ids"] if (representation, name) not in states)
    if "native" not in portraits:
        missing.append("native full-resolution portrait")
    return {"passed": not missing, "missing": missing, "evidence_ids": sorted(seen), "visual_acceptance": False,
            "scope": "Required full-resolution multiview/state coverage, not inspection of image content or motion meaning."}
