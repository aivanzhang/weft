"""Selected camera angle-overlay custody; no scene admission or geometry work."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from weft import whole_body_reference_report as saved_report
from weft.camera_shared_receipt import IDENTITIES
from weft.shared_exercise_reference_inputs import (
    MAX_INPUT_BYTES, MAX_CAMERA_INPUT_BYTES, MAX_MANIFEST_BYTES, PROPOSAL_GATES as LEGACY_GATES,
    _complete_geometry, _manifest, _require, load_reference_inputs,
)
from weft.source_surface_fit_inputs import surface_fit_member_names, validate_surface_fit_snapshot


PROFILE = "camera-reference-inputs-v1"
MULTIVIEW_PROFILE = "camera-multiview-reference-inputs-v1"
MULTIVIEW_METHOD = "blocked-multiview-camera-body-surface-interval-reference-v1"
MULTIVIEW_OBJECTIVE = "camera_multiview_surface_interval_v1"
PROPOSAL_GATES = LEGACY_GATES | frozenset({
    "surface_interval_nonregression", "camera_projection_domain", "camera_endpoint_depth_change"})
_ORIGINAL_ALIASES = {"human.json": "inputs/human.json", "evidence.json": "inputs/evidence-original.json",
                     "camera.json": "inputs/source-camera.json", "ground.json": "inputs/ground.json"}
_SECONDARY_ALIASES = {"inputs/two-view-root/camera.json": "inputs/secondary-camera.json",
                      "inputs/two-view-root/evidence.json": "inputs/secondary-evidence.json"}


@dataclass(frozen=True, slots=True)
class CameraReferenceInputs:
    """The proposal remains separate from the exact original body bytes."""

    proposal_directory: Path
    source_directory: Path
    proposal_manifest_bytes: bytes
    scene_manifest_bytes: bytes
    original_human_bytes: bytes
    angle_proposal_bytes: bytes
    report_bytes: bytes
    evidence_bytes: bytes
    scene_bytes: bytes
    task_bytes: bytes
    _proposal_members: tuple
    _receipt_json: str

    def receipt(self):
        return json.loads(self._receipt_json)

    @property
    def identities(self):
        return self.receipt()["identities"]

    @property
    def proposal_members(self):
        return MappingProxyType(dict(self._proposal_members))


def load_camera_reference_inputs(proposal, scene_run, task_path, *, expected_proposal_manifest_sha256=None,
                                 expected_scene_manifest_sha256=None, expected_task_sha256=None,
                                 max_seconds=10, max_input_bytes=MAX_CAMERA_INPUT_BYTES,
                                 multiview_max_input_bytes=None):
    """Reuse bounded selected reads and require the distinct camera route."""
    _require(type(max_input_bytes) is int and 1024 <= max_input_bytes <= MAX_CAMERA_INPUT_BYTES,
             "camera aggregate read bound must be 1 KiB–64 MiB, including rechecks")
    value = load_reference_inputs(proposal, scene_run, task_path, allow_camera=True,
        expected_proposal_manifest_sha256=expected_proposal_manifest_sha256,
        expected_scene_manifest_sha256=expected_scene_manifest_sha256, expected_task_sha256=expected_task_sha256,
        max_seconds=max_seconds, max_input_bytes=min(max_input_bytes, MAX_INPUT_BYTES),
        camera_max_input_bytes=max_input_bytes, multiview_max_input_bytes=multiview_max_input_bytes)
    _require(isinstance(value, CameraReferenceInputs), "camera angle-overlay inputs required")
    return value


def _load_camera(proposal, source, task_path, proposal_manifest_raw, proposal_files, *, read, check, finish, pins,
                 select_multiview_budget=None):
    proposal_pin, scene_pin, task_pin = pins
    _require("candidate-human.json" not in proposal_files,
             "camera angle overlay cannot also publish a replacement human")
    raw = {}

    def selected(name, maximum=8 << 20):
        _require(name in proposal_files, "camera proposal lacks required sealed input: " + name)
        if name not in raw:
            raw[name] = read(proposal, name, proposal_files[name], maximum)
        return raw[name]

    report = saved_report._json(selected("report.json"))
    multiview = report.get("profile") == MULTIVIEW_METHOD
    _require(report.get("profile") in {saved_report.CAMERA_METHOD, MULTIVIEW_METHOD},
             "angle overlay requires the actual camera producer report")
    _require(report.get("status") == "unadmitted_construction_proposal" and report.get("promoted") is False
             and report.get("geometry_admission") == "unsupported_angle_overlay_not_a_human_spec"
             and report.get("failures") == [] and report.get("budget_reason") is None
             and type(report.get("gates")) is dict and set(report["gates"]) == PROPOSAL_GATES
             and all(value is True for value in report["gates"].values()),
             "complete uninstalled camera proposal with all fourteen saved gates passing required")
    # Only a sealed, passing two-view producer can select its explicit read allowance.
    if multiview and select_multiview_budget is not None:
        select_multiview_budget()
    documents = {"report.json": report}
    for name in ("angle-proposal.json", "inputs/human.json", "inputs/evidence-derived.json"):
        documents[name] = saved_report._json(selected(name))
    plan = saved_report._json(selected("plan.json", 128 << 10))
    documents["plan.json"] = plan
    _require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
             and plan.get("profile") == "whole-body-reference-candidate-plan-v1" and plan.get("status") == "frozen"
             and plan.get("proposal_objective") == (MULTIVIEW_OBJECTIVE if multiview else "camera_surface_interval_v1"),
             "frozen camera input plan required")
    descriptor = plan.get("surface_constraints")
    names = surface_fit_member_names(descriptor)
    secondary = plan.get("secondary_view")
    if multiview:
        from weft.camera_multiview_inputs import camera_multiview_member_names, validate_camera_multiview_snapshot
        names = names | camera_multiview_member_names(secondary)
    else:
        _require("secondary_view" not in plan, "single-view proposal cannot silently consume a secondary view")
    _require(type(plan.get("input_files")) is dict
             and all(name in proposal_files and plan["input_files"].get(name) == proposal_files[name]
                     for name in names | {"inputs/human.json", "inputs/evidence-derived.json"}),
             "camera plan input pins differ from study seal")
    for name in sorted(names):
        selected(name, 64 << 10 if name.endswith(("surface-config.json", "camera-secondary-config.json")) else 8 << 20)
    check()
    validate_surface_fit_snapshot(raw, descriptor, source_sha256=documents["inputs/human.json"].get("source_sha256"))
    if multiview:
        validate_camera_multiview_snapshot(raw, secondary,
            source_sha256=documents["inputs/human.json"].get("source_sha256"))
    check()
    source_identity = plan.get("source")
    original_aliases = _ORIGINAL_ALIASES | (_SECONDARY_ALIASES if multiview else {})
    source_extras = tuple(saved_report._CAMERA_EXTRAS) + (tuple(_SECONDARY_ALIASES.values()) if multiview else ())
    _require(type(source_identity) is dict and source_identity.get("profile") == "camera-body-reference-inputs-v1"
             and type(source_identity.get("copied_files")) is dict
             and all(name in proposal_files and source_identity["copied_files"].get(name) == proposal_files[name]
                     for name in {*source_extras, "inputs/human.json", "inputs/evidence-derived.json"}),
             "camera original-copy pins differ from plan and study seal")
    for name in source_extras:
        selected(name, 256 << 10 if name.endswith("manifest.json") else 8 << 20)
    _require(raw["inputs/source-camera.json"] == raw.get("inputs/surface-camera.json")
             and raw["inputs/evidence-original.json"] == raw["inputs/evidence-derived.json"],
             "camera proposal must preserve exact original camera and evidence bytes")
    # Large saved reports stay parsed once; each raw alias is still independently pinned.
    documents.update({name: saved_report._json(value) for name, value in raw.items() if name not in documents})
    proof = documents["inputs/camera-inputs-proof.json"]
    _require(type(proof.get("selected_members")) is dict, "camera original-source proof required")
    for original, alias in original_aliases.items():
        _require(type(proof["selected_members"].get(original)) is dict
                 and proof["selected_members"][original].get("bytes") == len(raw[alias])
                 and (not multiview or original not in _SECONDARY_ALIASES
                      or proof["selected_members"][original].get("sha256") == proposal_files[alias]
                      == documents["inputs/body-manifest.json"]["files"].get(original)
                      and proof["selected_members"][original].get("copied_member") == alias),
                 "camera selected-source byte count differs")
    model = saved_report._model(documents, {name: proposal_files[name] for name in raw}, check)
    _complete_geometry(documents, model)

    scene_manifest_raw = read(source, "manifest.json", scene_pin, MAX_MANIFEST_BYTES)
    scene_manifest, scene_files = _manifest(scene_manifest_raw)
    status = scene_manifest.get("status")
    _require(status == "ready" or status == "constructed"
             and scene_manifest.get("profile") == "camera-shared-candidate-v1", "sealed original camera scene required")
    _require({"scene.json", "camera-coupling.json", *original_aliases} <= scene_files.keys(),
             "original camera scene lacks its exact source views")
    scene_raw = read(source, "scene.json", scene_files["scene.json"])
    scene, original = saved_report._json(scene_raw), documents["inputs/human.json"]
    canonical = saved_report._canonical
    coupling = report["camera_fit"]["coupling"]
    _require(type(scene.get("schema_version")) is int and scene["schema_version"] == 1
             and scene.get("profile") == "shared-world-v6" and scene.get("controlled_id") == "human"
             and scene.get("physical_validation") == "unverified" and scene.get("learned_policy_available") is False
             and scene.get("objects") == [] and scene.get("target") is None
             and canonical(scene.get("human_spec")) == canonical(original)
             and canonical(scene.get("camera_coupling")) == canonical(coupling),
             "original camera scene/body/coupling identity differs")
    _require(type(scene.get("source_binding")) is dict and scene["source_binding"].get("mode") == "source_qualified"
             and scene["source_binding"].get("source_sha256") == model["source_sha256"]
             and scene["source_binding"].get("total_source_frames") == coupling["clock"]["total_source_frames"],
             "original camera scene source or frame denominator differs")
    for original_name, alias in original_aliases.items():
        _require(read(source, original_name, scene_files[original_name]) == raw[alias],
                 "original camera scene view differs from proposal input: " + original_name)
    _require(canonical(saved_report._json(read(source, "camera-coupling.json", scene_files["camera-coupling.json"])))
             == canonical(coupling), "original scene coupling copy differs")
    task_raw = read(task_path.parent, task_path.name, task_pin, 64 << 10)
    task = saved_report._json(task_raw)
    if "exercise-task.json" in scene_files:
        _require(read(source, "exercise-task.json", scene_files["exercise-task.json"], 64 << 10) == task_raw,
                 "supplied task differs from sealed original camera task")
    interval = {"start_offset_s": 0., "horizon_steps": coupling["clock"]["horizon_steps"]}
    _require(type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == "camera-shared-task-v1" and task.get("scene_sha256") == canonical(scene)
             and task.get("interval") == interval and scene["task"].get("object_goal") is None
             and all(scene["task"].get(k) == v for k, v in interval.items())
             and scene.get("dt") == coupling["clock"]["control_dt_s"]
             and all(type(task.get(key)) is str and saved_report._SHA.fullmatch(task[key]) for key in IDENTITIES)
             and task["camera_coupling_sha256"] == canonical(coupling)
             and task["body_sha256"] == canonical(original) and task["source_sha256"] == model["source_sha256"]
             and task["source_clock_sha256"] == coupling["source_clock_sha256"],
             "supplied camera task must bind the unchanged original full source interval")
    read_receipt = finish()
    identities = {"proposal_manifest_sha256": sha256(proposal_manifest_raw).hexdigest(),
        "scene_manifest_sha256": sha256(scene_manifest_raw).hexdigest(), "source_sha256": model["source_sha256"],
        "original_human_sha256": sha256(raw["inputs/human.json"]).hexdigest(),
        "original_human_canonical_sha256": canonical(original),
        "angle_proposal_sha256": sha256(raw["angle-proposal.json"]).hexdigest(),
        "angle_proposal_canonical_sha256": canonical(documents["angle-proposal.json"]),
        "proposal_report_sha256": sha256(raw["report.json"]).hexdigest(),
        "input_evidence_sha256": sha256(raw["inputs/evidence-derived.json"]).hexdigest(),
        "input_camera_sha256": sha256(raw["inputs/source-camera.json"]).hexdigest(),
        "scene_file_sha256": sha256(scene_raw).hexdigest(), "scene_sha256": canonical(scene),
        "task_file_sha256": sha256(task_raw).hexdigest(), "task_sha256": canonical(task),
        **{key: task[key] for key in IDENTITIES}}
    if multiview:
        identities.update(additional_view_sha256=canonical(documents["angle-proposal.json"]["additional_view"]),
            multiview_qualification_sha256=canonical(report["multiview_images"]),
            secondary_view_config_sha256=secondary["config_sha256"], secondary_source_sha256=secondary["secondary_source_sha256"],
            secondary_camera_sha256=secondary["camera_sha256"], secondary_evidence_sha256=secondary["evidence_sha256"],
            secondary_world_alignment_sha256=secondary["world_alignment_sha256"],
            secondary_observation_sha256s=list(secondary["observation_sha256s"]))
    receipt = {"schema_version": 1, "profile": MULTIVIEW_PROFILE if multiview else PROFILE, "status": "selected_metadata_verified",
        "identities": identities, "scene_completion": {"route": "camera_manifest_" + status, "status": status},
        "external_pins_supplied": {"proposal_manifest": proposal_pin is not None,
            "scene_manifest": scene_pin is not None, "task": task_pin is not None}, **read_receipt,
        "selected_proposal_members": {name: proposal_files[name] for name in raw},
        "saved_gates": dict(report["gates"]), "promoted": False, "physical_validation": "unverified",
        "installation": "unsupported_angle_overlay_not_a_human_spec", "body_installed": False,
        "preparation": {"source_directory": str(source), "expected_manifest_sha256": identities["scene_manifest_sha256"],
            "required_next_checks": ["Full independent world-source admission", "Original camera scene/task validation",
                "Exact current runtime and trusted velocity helper verification", "Numerical camera-controller validation"]},
        "scope": "Selected saved metadata and complete producer report checks only; no source-fit recomputation, native admission, body replacement or physical validation. Unselected members/media/runtime were not read. Local seals without supplied pins are frozen identities, not external authentication."}
    return CameraReferenceInputs(proposal, source, proposal_manifest_raw, scene_manifest_raw, raw["inputs/human.json"],
        raw["angle-proposal.json"], raw["report.json"], raw["inputs/evidence-derived.json"], scene_raw, task_raw,
        tuple(sorted(raw.items())), json.dumps(receipt, sort_keys=True, allow_nan=False))
