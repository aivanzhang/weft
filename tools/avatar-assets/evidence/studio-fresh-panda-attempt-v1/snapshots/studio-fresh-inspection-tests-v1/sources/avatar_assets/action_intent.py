"""Exact action requests and bounded evidence gates, never a semantic/physics oracle."""

from copy import deepcopy
import math
import re

from avatar_assets.safety import require, slug


KINDS = ("articulation", "travel", "rhythm", "takeoff", "landing", "stability", "interaction", "expression")
METRICS = frozenset({"scale_m", "weighted_vertices", "moving_vertices",
                     "max_root_relative_displacement_m", "max_world_displacement_m", "max_travel_m", "max_vertical_rise_m",
                     "descent_from_peak_m", "rhythmic_reversals", "active_control_count"})
COUNTS = frozenset({"weighted_vertices", "moving_vertices", "rhythmic_reversals", "active_control_count"})


def _text(value, label, maximum=600):
    require(isinstance(value, str) and 0 < len(value) <= maximum and value.strip(), "Invalid " + label)


def validate_custom_actions(value):
    """Empty/omitted requests retain the legacy state-ID route; never normalize user text."""
    require(isinstance(value, list) and len(value) <= 8, "custom_actions must contain at most eight strings")
    for item in value:
        _text(item, "custom action", 600)
    return list(value)


def required_kinds(text):
    """Conservative common-intent guard; arbitrary prose still needs independent review."""
    text = text.casefold()
    kinds = {"stability"} if _stationary_language(text) else set()
    # Negative clauses must not turn 'wave without flying' into a flight request.
    text = re.sub(r"\b(?:not|never|without|no)\b[^,.;]*", "", text)
    if re.search(r"\b(?:dance|dances|dancing)\b", text):
        kinds.add("rhythm")
    if re.search(r"\b(?:fly|flies|flying|flight|walk|walks|walking|run|runs|running|travel|travels)\b", text):
        kinds.add("travel")
    if re.search(r"\b(?:flap|flaps|flapping|wave|waves|waving|bend|bends|bending)\b", text):
        kinds.add("articulation")
    if re.search(r"\b(?:take[ -]?off|takes off|taking off|launch|launches)\b", text):
        kinds.update(("takeoff", "travel"))
    if re.search(r"\b(?:land|lands|landing)\b", text):
        kinds.add("landing")
    return kinds


def _stationary_language(text):
    """Only explicit hold/stay clauses; 'steady rhythm' and upright travel are not stillness."""
    text = text.casefold()
    if re.search(r"\b(?:keep|keeps|keeping|remain|remains|remaining|stay|stays|staying|hold|holds|holding)\b"
                 r"[^.;]{0,100}\b(?:still|stationary|motionless|planted|"
                 r"steady(?!\s+(?:rhythm|pace|tempo|speed|beat|motion|wave|cadence))|"
                 r"fixed(?!\s+(?:rate|pace|tempo|speed|direction)))\b", text):
        return True
    return bool(re.search(r"\b(?:do not|does not|don't|doesn't|without)\s+mov(?:e|ing)\s+"
                          r"(?:(?:the|its|their|both|either|your)\s+)?(?:feet|foot|torso|body|base|pedestal|legs)\b", text))


def validate_requirements(value, roles):
    require(isinstance(roles, list) and 1 <= len(roles) <= 4, "Action requires one to four roles")
    role_ids = []
    for role in roles:
        require(isinstance(role, dict) and set(role) == {"id", "description"}, "Invalid action role fields")
        role_ids.append(slug(role["id"]))
        _text(role["description"], "role description")
    require(len(set(role_ids)) == len(role_ids), "Duplicate action role")
    require(isinstance(value, list) and 1 <= len(value) <= 8, "Action requires one to eight requirements")
    ids = []
    for item in value:
        require(isinstance(item, dict) and set(item) == {"id", "kind", "role", "description"},
                "Invalid action requirement fields")
        ids.append(slug(item["id"]))
        require(item["kind"] in KINDS and item["role"] in role_ids, "Invalid requirement kind or role")
        _text(item["description"], "requirement description")
        require(not _stationary_language(item["description"]) or item["kind"] == "stability",
                "Stationary requirement must use stability, not a motion requirement")
    require(len(ids) == len(set(ids)), "Duplicate action requirement")
    require({item["role"] for item in value} == set(role_ids), "Every action role needs a requirement")
    for role in role_ids:
        kinds = {item["kind"] for item in value if item["role"] == role}
        require("stability" not in kinds or kinds == {"stability"},
                "Stability cannot share a whole-clip role with movement requirements; phases are unsupported")
    return deepcopy(value)


def validate_state_intents(states, custom_actions, requested_ids=(), *, ready=True):
    """Validate exact ordered intent; legacy states may omit all three metadata fields."""
    actions = validate_custom_actions(custom_actions)
    require(isinstance(states, list) and len(states) <= 8, "Invalid action states")
    require(isinstance(requested_ids, (list, tuple)) and len(requested_ids) <= 8, "Invalid requested state IDs")
    for name in requested_ids:
        slug(name)
    require(len(set(requested_ids)) == len(requested_ids), "Duplicate requested state ID")
    require(not actions or not requested_ids or len(actions) == len(requested_ids),
            "Custom actions and explicit state IDs must have matching counts")
    names = []
    for index, state in enumerate(states):
        require(isinstance(state, dict) and {"id", "meaning"} <= set(state)
                and set(state) <= {"id", "meaning", "user_request", "roles", "requirements"},
                "Invalid planned state fields")
        names.append(slug(state["id"]))
        _text(state["meaning"], "state meaning", 2000)
        supplied = state.get("user_request", "")
        require(isinstance(supplied, str) and len(supplied) <= 600, "Invalid state user_request")
        if actions:
            require(index < len(actions) and supplied == actions[index], "Planner changed or reordered a custom action")
        else:
            require(not supplied, "Planner invented a custom action request")
        structured = "roles" in state or "requirements" in state
        require(not actions or structured, "Custom actions require structured intent")
        if structured:
            validate_requirements(state.get("requirements"), state.get("roles"))
            kinds = {item["kind"] for item in state["requirements"]}
            require(required_kinds(supplied or state["meaning"]) <= kinds,
                    "Planner omitted a required action capability; do not relabel the action")
            if ready:
                require(not kinds & {"interaction", "expression"},
                        "Ready plan contains unsupported interaction or expression requirements")
    require(len(names) == len(set(names)), "Duplicate planned state")
    if ready:
        expected = len(actions) if actions else (len(requested_ids) if requested_ids else 4)
        require(len(names) == expected and (not requested_ids or names == list(requested_ids)),
                "Planner changed requested states or omitted the four-state default")
    return deepcopy(states)


def validate_action_bindings(value, states, bone_parents):
    """Bind semantic roles to observed bones; caller supplies actual exported hierarchy."""
    require(isinstance(bone_parents, dict) and 1 <= len(bone_parents) <= 64, "Invalid observed bone hierarchy")
    for name, parent in bone_parents.items():
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name), "Invalid observed bone")
        require(parent is None or (isinstance(parent, str) and parent in bone_parents), "Unknown observed bone parent")
        seen, node = set(), name
        while node is not None:
            require(node not in seen, "Cyclic observed bone hierarchy")
            seen.add(node)
            node = bone_parents[node]
    expected = [state for state in states if state.get("requirements")]
    require(isinstance(value, list) and len(value) == len(expected), "Bindings must cover every structured action")
    for item, state in zip(value, expected):
        require(isinstance(item, dict) and set(item) == {"state_id", "roles"} and item["state_id"] == state["id"],
                "Action binding state mismatch")
        roles = item["roles"]
        require(isinstance(roles, list) and len(roles) == len(state["roles"]), "Action binding role mismatch")
        for role, intent in zip(roles, state["roles"]):
            require(isinstance(role, dict) and set(role) == {"id", "root_bone", "bones"}
                    and role["id"] == intent["id"], "Action binding role mismatch")
            root, bones = role["root_bone"], role["bones"]
            require(isinstance(root, str) and root in bone_parents, "Unknown action root bone")
            require(isinstance(bones, list) and 1 <= len(bones) <= 64
                    and all(isinstance(name, str) and name in bone_parents for name in bones), "Unknown action target bone")
            require(len(set(bones)) == len(bones), "Duplicate action target bone")
            for name in bones:
                node = name
                while node is not None and node != root:
                    node = bone_parents[node]
                require(node == root, "Action target is outside its bound root hierarchy")
    return deepcopy(value)


def evaluate_action_evidence(requirements, evidence, height_meters):
    """Trusted measurements in meters; missing metrics fail closed, with no visual acceptance."""
    require(type(height_meters) in (int, float) and math.isfinite(height_meters)
            and .05 <= height_meters <= 10, "Invalid authored height")
    require(isinstance(evidence, dict) and set(evidence) == {"state_id", "evidence_id", "roles"},
            "Invalid action evidence envelope")
    slug(evidence["state_id"])
    _text(evidence["evidence_id"], "measurement evidence ID", 200)
    roles = evidence["roles"]
    require(isinstance(roles, dict) and len(roles) <= 4, "Invalid measured roles")
    require(isinstance(requirements, list) and 1 <= len(requirements) <= 8, "Invalid measured requirements")
    require(all(isinstance(item, dict) and isinstance(item.get("role"), str) for item in requirements),
            "Invalid measured requirement role")
    role_ids = list(dict.fromkeys(item["role"] for item in requirements))
    validate_requirements(requirements, [{"id": name, "description": "Measured action role"} for name in role_ids])
    require(set(roles) <= set(role_ids), "Measured unknown action role")
    for role, values in roles.items():
        require(isinstance(values, dict) and set(values) <= METRICS, "Invalid action measurement fields")
        for key, number in values.items():
            require(type(number) in (int, float) and math.isfinite(number) and number >= 0, "Invalid action measurement")
            if key in COUNTS:
                require(type(number) is int and number <= (100000 if "vertices" in key else 1024), "Invalid measurement count")
            else:
                require(number <= 100, "Action measurement exceeds world bounds")
        if "scale_m" in values:
            require(.0001 <= values["scale_m"] <= 100, "Invalid actual role scale")
        if {"weighted_vertices", "moving_vertices"} <= set(values):
            require(values["moving_vertices"] <= values["weighted_vertices"], "Moving coverage exceeds weighted coverage")
    needed = {
        "articulation": {"scale_m", "weighted_vertices", "moving_vertices", "max_root_relative_displacement_m"},
        "travel": {"scale_m", "max_travel_m"},
        "rhythm": {"scale_m", "rhythmic_reversals", "active_control_count", "max_travel_m", "max_root_relative_displacement_m"},
        "takeoff": {"scale_m", "max_vertical_rise_m", "max_travel_m"},
        "landing": {"scale_m", "descent_from_peak_m"},
        "stability": {"scale_m", "weighted_vertices", "max_world_displacement_m"},
    }
    rows = []
    for item in requirements:
        kind, values = item["kind"], roles.get(item["role"], {})
        row = {"id": item["id"], "kind": kind, "role": item["role"],
               "evidence_ids": [evidence["evidence_id"]], "status": "missing", "reason": ""}
        if kind not in needed:
            row.update(status="unsupported", reason="These measurements/backend cannot establish " + kind)
        elif missing := needed[kind] - set(values):
            row["reason"] = "Missing actual measurements: " + ", ".join(sorted(missing))
        else:
            scale = values["scale_m"]
            if kind == "articulation":
                passed = (values["weighted_vertices"] >= 8 and values["moving_vertices"] >= max(3, math.ceil(values["weighted_vertices"]*.01))
                          and values["max_root_relative_displacement_m"] >= .005*scale)
                reason = "Root-relative skin displacement and meaningful weighted coverage"
            elif kind == "travel":
                passed, reason = values["max_travel_m"] >= .03*scale, "Actual role travel at least 3% of measured role scale"
            elif kind == "rhythm":
                passed = (values["rhythmic_reversals"] >= 3 and values["active_control_count"] >= 2
                          and max(values["max_travel_m"], values["max_root_relative_displacement_m"]) >= .02*scale)
                reason = "Three rhythmic reversals, two independently active controls, and readable displacement"
            elif kind == "takeoff":
                passed = values["max_vertical_rise_m"] >= .05*scale and values["max_travel_m"] >= .05*scale
                reason = "Actual rise and travel at least 5% of measured role scale"
            elif kind == "stability":
                tolerance = min(.005*scale, .005)
                passed = values["weighted_vertices"] >= 8 and values["max_world_displacement_m"] <= tolerance
                reason = "World-space weighted-surface displacement at most 0.5% of measured role scale, capped at 5mm; not ground-contact proof"
                row["maximum_displacement_m"] = tolerance
            else:
                passed, reason = values["descent_from_peak_m"] >= .05*scale, "Actual descent from peak; contact still needs visual review"
            row.update(status="passed" if passed else "failed", reason=reason)
        rows.append(row)
    return {"schema_version": 1, "state_id": evidence["state_id"], "requirements": rows,
            "passed": bool(rows) and all(row["status"] == "passed" for row in rows),
            "semantic_approval": False, "physics_verified": False,
            "scope": "Necessary bounded motion measurements, not sufficient evidence of action meaning or visual quality."}
