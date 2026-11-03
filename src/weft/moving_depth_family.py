"""Complete source-bound depth factors with explicit, unscored assumption selections."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType

from weft.io import canonical_hash
from weft.moving_equipment_proposal import (_check, _deadline, _family, _plain, _track_proposal,
    validate_moving_settings, SETTINGS_PROFILE)

PROFILE = "moving-depth-family-v1"
SETTINGS = "moving-depth-family-priors-v1"
SELECTION = "moving-depth-selection-v1"
BUNDLE = "moving-depth-selected-proposals-v1"
MAX_BYTES = 16 << 20
SCOPE = "Complete factorized raw-identity and conditional depth alternatives; explicit selections are authored assumptions, not recovered identity, shape, prefix, motion or task success."
_TOKEN = object()


def _require(condition, message):
    if not condition:
        raise ValueError("Moving depth family: " + message)


def _unique(pairs):
    value = {}
    for name, item in pairs:
        _require(name not in value, "duplicate saved JSON key")
        value[name] = item
    return value


def _invalid_constant(value):
    raise ValueError("Moving depth family: nonfinite saved JSON constant")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _encoded(value, deadline):
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop(); nodes += 1
        _require(nodes <= 1_000_000 and depth <= 32, "document structural bound exceeded")
        if isinstance(item, dict):
            _require(len(item) <= 256 and all(isinstance(k, str) and len(k) <= 256 for k in item), "bounded document keys required")
            pending.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            _require(len(item) <= 21600, "document list bound exceeded")
            pending.extend((v, depth + 1) for v in item)
        elif isinstance(item, str):
            _require(len(item) <= 16384, "document string bound exceeded")
        else:
            _require(item is None or type(item) in (bool, int, float), "JSON values required")
            if type(item) in (int, float):
                _require(abs(item) <= 1e12 and math.isfinite(item), "finite bounded document number required")
        if nodes % 1024 == 0:
            _check(deadline)
    output = bytearray()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        encoded = chunk.encode()
        _require(len(output) + len(encoded) <= MAX_BYTES, "complete document exceeds byte bound")
        output.extend(encoded)
        _check(deadline)
    return bytes(output)


def _legacy_settings(settings):
    return {key: value for key, value in settings.items() if key != "depth_domain_model_units"} | {
        "profile": SETTINGS_PROFILE, "depth_offsets_model_units": [0.]}


def validate_moving_depth_settings(settings):
    keys = {"schema_version", "profile", "camera", "depth_domain_model_units", "shape", "mass_kg", "friction", "initialization"}
    _require(isinstance(settings, dict) and set(settings) == keys and settings.get("profile") == SETTINGS,
             "closed explicit depth-family priors required")
    domain = settings["depth_domain_model_units"]
    _require(isinstance(domain, list) and len(domain) == 2
             and all(type(v) in (int, float) and abs(v) <= 100 and math.isfinite(v) for v in domain), "bounded depth domain required")
    legacy = validate_moving_settings(_legacy_settings(settings))
    distance = legacy["camera"]["distance_model_units"]
    _require(.05 < domain[0] <= domain[1] and distance - 2 <= domain[0] <= domain[1] <= distance + 2,
             "depth domain must preserve original positive camera/offset limits")
    _require(legacy["initialization"] == {"method": "first_initialized_box_zero_velocity"},
             "depth intervals require the explicit source-ray zero-velocity prefix prior")
    return {key: _plain(value) for key, value in settings.items()}


@dataclass(frozen=True, init=False, slots=True)
class PreparedMovingDepthFamily:
    _token: object
    _inputs: object
    _value: object
    document_bytes: bytes

    @property
    def document(self):
        return _plain(self._value)


def validate_prepared_family(prepared, inputs=None, *, deadline=None):
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs

    stop = _deadline(deadline)
    _require(type(prepared) is PreparedMovingDepthFamily and getattr(prepared, "_token", None) is _TOKEN,
             "an internally prepared immutable family is required")
    validate_moving_equipment_inputs(prepared._inputs, deadline=stop)
    _require(inputs is None or prepared._inputs is inputs, "family must belong to the exact admitted source packet")
    return prepared


def _source_parts(packet, settings):
    binding, body = packet.source_binding, packet.body_binding
    entities = {row["world_entity_id"]: row for row in binding["entity_bindings"] if row["world_entity_id"] != "human"}
    ids = sorted(entities)
    _require(1 <= len(ids) <= 12, "complete bounded raw identity inventory required")
    distance = settings["camera"]["distance_model_units"]
    scale = body["root_scale_unchanged"]["model_units_per_pixel"]
    camera = {"distance_model_units": distance, "focal_px": distance / scale,
        "pixel_origin": body["root_origin_unchanged"]["pelvis_image_px"],
        "original_scale_at_anchor_model_units_per_pixel": scale,
        "world_anchor_model_units": [0., 0., packet.base_scene["human_spec"]["mobility"]["base_height"]],
        "world_axes": ["x_right", "y_depth", "z_up"], "status": "authored_uncalibrated_fixed_camera"}
    index = {(source["document_sha256"], track["entity_ref"]["object_id"]): track
             for source in packet.motion_evidence["sources"] for track in source["tracks"]}
    tracks = {name: index[(entity["observation_document_sha256"], entity["observation_entity_id"])]
              for name, entity in entities.items()}
    return ids, camera, tracks


def _pins(packet):
    return {"admission_receipt_sha256": sha256(packet.admission_receipt_bytes).hexdigest(),
        "full_human_sha256": sha256(packet.human_bytes).hexdigest(),
        "motion_evidence_sha256": sha256(packet.motion_evidence_bytes).hexdigest(),
        "body_binding_sha256": canonical_hash(packet.body_binding),
        "base_scene_sha256": canonical_hash(packet.base_scene), "source_sha256": packet.source_binding["source_sha256"]}


def prepare_moving_depth_family(inputs, settings, *, deadline=None):
    """Extract original geometry once; retain all raw identity/depth factors without selection."""
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs
    from weft.moving_depth_geometry import extract_depth_geometry
    from weft.moving_depth_intervals import solve_depth_intervals

    stop = _deadline(deadline)
    packet = validate_moving_equipment_inputs(inputs, deadline=stop)
    settings = validate_moving_depth_settings(settings)
    ids, camera, tracks = _source_parts(packet, settings)
    identity = _family(packet, ids)
    identity.update(representation="factorized_nonempty_subsets_and_independent_depth_components",
                    materialized_assignment_count=0, joint_object_collision_status="not_evaluated")
    first = packet.body_references[0]
    geometry = extract_depth_geometry(packet, deadline=stop)
    bridge = packet.base_scene["coordinate_frame"]["scale_bridge"]
    scale, origin = bridge["body_height_to_world_scale"], bridge["origin_world"]
    distance, focal = camera["distance_model_units"], camera["focal_px"]
    legacy = _legacy_settings(settings)
    factors = []
    for name in ids:
        _check(stop)
        track = tracks[name]
        reference = _track_proposal(track, name, 0, 0., camera, legacy, first["time_s"], first["source_frame"], stop)
        factor = {"object_id": name, "entity_ref": _plain(track["entity_ref"]), "reference_track": reference,
                  "ray": None, "depth_intervals": None, "status": "unsupported_input", "reasons": list(reference["reasons"])}
        if reference["status"] == "conditional_free_body_hypothesis":
            basis = reference["initial_state_hypothesis"]["basis_frame"]
            slot = next(row for row in track["slots"] if row["source_frame"] == basis)
            x, y, width, height = slot["measurements"]["box"]["value"]
            intercept = [0., -distance, camera["world_anchor_model_units"][2]]
            slope = [(x + width / 2 - camera["pixel_origin"][0]) / focal, 1.,
                     -(y + height / 2 - camera["pixel_origin"][1]) / focal]
            ray = {"domain": settings["depth_domain_model_units"],
                "center_intercept": [origin[i] + scale * intercept[i] for i in range(3)],
                "center_slope": [scale * value for value in slope], "radius_intercept": 0.,
                "radius_slope": scale * reference["shape_prior"]["radius_model_units"] / distance,
                "radius_bounds": [.01, .5]}
            report = solve_depth_intervals(ray, geometry, deadline=stop)
            factor.update(ray=ray, depth_intervals=report, status=report["status"], reasons=list(report["reasons"]))
        factors.append(factor)
    document = {"schema_version": 1, "profile": PROFILE, "settings": settings, "input_pins": _pins(packet),
        "identity_family": identity, "camera_hypothesis": camera, "geometry": geometry, "factors": factors,
        "status": "ready_for_explicit_selection" if any(row["status"] == "feasible_intervals" for row in factors) else "abstained",
        "scope": SCOPE}
    raw = _encoded(document, stop)
    result = object.__new__(PreparedMovingDepthFamily)
    for key, value in {"_token": _TOKEN, "_inputs": packet, "_value": _freeze(document), "document_bytes": raw}.items():
        object.__setattr__(result, key, value)
    return result


def admit_moving_depth_family(inputs, family_document, *, deadline=None):
    """Re-extract and rederive once; hashes on a supplied geometry receipt are not authority."""
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs

    stop = _deadline(deadline)
    packet = validate_moving_equipment_inputs(inputs, deadline=stop)
    if isinstance(family_document, bytes):
        _require(len(family_document) <= MAX_BYTES, "saved family exceeds byte bound")
        family_document = json.loads(family_document, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    _encoded(family_document, stop)
    _require(isinstance(family_document, dict) and family_document.get("profile") == PROFILE
             and family_document.get("input_pins") == _pins(packet), "saved family differs from original admitted inputs")
    rebuilt = prepare_moving_depth_family(packet, family_document.get("settings"), deadline=stop)
    _require(canonical_hash(family_document) == canonical_hash(rebuilt.document),
             "saved family differs from exact source, geometry and interval rederivation")
    return rebuilt


def validate_depth_selection(selection, family):
    """Check explicit choices against a document; source authority requires a prepared capability."""
    _require(isinstance(selection, dict) and set(selection) == {"schema_version", "profile", "family_sha256", "choices"}
             and type(selection["schema_version"]) is int and selection["schema_version"] == 1
             and selection["profile"] == SELECTION, "closed explicit selection required")
    _require(isinstance(family, dict) and family.get("profile") == PROFILE
             and selection["family_sha256"] == canonical_hash(family), "selection must pin the complete family")
    choices = selection["choices"]
    _require(isinstance(choices, list) and 1 <= len(choices) <= 3, "one to three explicit choices required")
    factors = {row["object_id"]: row for row in family["factors"]}
    seen = set()
    for choice in choices:
        _require(isinstance(choice, dict) and set(choice) == {"equipment"}, "closed equipment assignment required")
        equipment = choice["equipment"]
        _require(isinstance(equipment, list) and 1 <= len(equipment) <= 12, "explicit nonempty bounded subset required")
        ids = []
        for row in equipment:
            _require(isinstance(row, dict) and set(row) == {"object_id", "interval_index", "depth_model_units"}, "closed depth assumption required")
            name, index, depth = row["object_id"], row["interval_index"], row["depth_model_units"]
            _require(isinstance(name, str) and name in factors, "selection names an unknown raw identity")
            factor = factors[name]
            intervals = (factor.get("depth_intervals") or {}).get("certified_intervals", [])
            _require(type(index) is int and 0 <= index < len(intervals)
                     and type(depth) in (int, float) and abs(depth) <= 100 and math.isfinite(depth)
                     and intervals[index][0] <= depth <= intervals[index][1], "selection must lie inside its certified interval")
            ids.append(name)
        _require(ids == sorted(set(ids)), "selected raw identities must be sorted and unique")
        key = canonical_hash(equipment)
        _require(key not in seen, "duplicate selected assumptions are not separate candidates")
        seen.add(key)
    return _plain(selection)


def materialize_moving_depth_selection(prepared, selection, *, deadline=None):
    """Materialize only explicit subsets; keep the complete factorized family in every bundle."""
    stop = _deadline(deadline)
    prepared = validate_prepared_family(prepared, deadline=stop)
    packet, family = prepared._inputs, prepared.document
    selection = validate_depth_selection(selection, family)
    settings = _legacy_settings(family["settings"])
    ids, camera, tracks = _source_parts(packet, settings)
    first = packet.body_references[0]
    bundle = {"schema_version": 1, "profile": BUNDLE, "settings": settings,
        "maximum_candidates": len(selection["choices"]), "input_pins": _plain(family["input_pins"]),
        "identity_family": _plain(family["identity_family"]), "camera_hypothesis": camera,
        "track_proposals": [], "candidates": [], "status": "conditional_candidates", "reasons": [], "scope": SCOPE,
        "depth_family": family, "depth_selection": selection}
    distance = camera["distance_model_units"]
    for index, choice in enumerate(selection["choices"]):
        equipment_ids = [row["object_id"] for row in choice["equipment"]]
        refs = []
        for selected in choice["equipment"]:
            name, depth = selected["object_id"], selected["depth_model_units"]
            track = _track_proposal(tracks[name], name, index, depth - distance, camera, settings,
                first["time_s"], first["source_frame"], stop, depth_model_units=depth)
            _require(track["status"] == "conditional_free_body_hypothesis", "selected source support became unavailable")
            track["depth_assumption"] = {**selected, "status": "explicit_authored_depth_not_measured",
                                         "family_sha256": selection["family_sha256"]}
            bundle["track_proposals"].append(track)
            refs.append({"object_id": name, "track_proposal_id": track["id"]})
        bundle["candidates"].append({"id": f"candidate-{index + 1}",
            "identity_assignment": {"equipment_ids": equipment_ids, "unknown_complement_ids": sorted(set(ids) - set(equipment_ids)),
                                    "physical_identity_verified": False},
            "depth_choices": _plain(choice["equipment"]), "objects": refs,
            "status": "conditional_free_body_hypothesis", "reasons": []})
    _encoded(bundle, stop)
    return bundle
