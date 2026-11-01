"""Authored metadata custody only; these packets are not native or physical evidence."""

from copy import deepcopy
from hashlib import sha256
import json
import time

import pytest

from weft import exercise_geometry_inputs as inputs
from weft.io import canonical_hash
from weft.shared_world_geometry import CONFIG_PROFILE, _build


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def digest(value):
    return sha256(value if isinstance(value, bytes) else encoded(value)).hexdigest()


class Packet:
    def __init__(self, root):
        self.root = root
        source_sha = digest(b"authored source bytes, not a decodable video")
        objects = []
        for name, frame in (("weight", 4), ("support", 6)):
            objects.append({"id": name, "shape": "box", "size": [.4, .3, .5], "mass": 2.,
                "position": [1., 0., .25], "quaternion": [1., 0., 0., 0.], "velocity": [0., 0., 0.],
                "angular_velocity": [0., 0., 0.], "friction": .8, "role": "equipment",
                "provenance": {"source_prefix": "observations/", "source_bounds_px": [1., 2., 5., 7.],
                    "source_reference": {"document_sha256": "a" * 64, "object_id": name, "frame": frame},
                    "camera_hypothesis": {"profile": "authored-camera", "scale": 1.},
                    "mode": "source_inferred", "uncertainty": {"mass": "authored"}}})
        base = {"profile": "authored-metadata-fixture", "objects": objects, "assumptions": ["fixture, not physical evidence"],
            "human_spec": {"frames": [0, 4, 6], "source_times": [0., .1, .2]},
            "source_binding": {"mode": "source_qualified", "source_sha256": source_sha},
            "coordinates": {"floor_z": 0., "axes": "unchanged-authored"},
            "task": {"object_goal": "weight", "threshold": .1}, "learned_policy_available": False}
        geometry = {"profile": "rigid-compound-v1", "parts": [{"id": "band", "shape": "box",
            "size": [.4, .3, .3], "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.]}],
            "inertia": {"profile": "authored-centered-diagonal-v1", "diagonal": [.1, .1, .1]}}
        config = {"schema_version": 1, "profile": CONFIG_PROFILE, "base_scene_sha256": canonical_hash(base),
            "assumptions": ["Conditional exterior prior, not measured empty space"],
            "replacements": [{"object_id": "weight", "geometry": geometry}]}
        candidate, previous = _build(base, config, digest(config))
        task = {"profile": "authored-task-fixture", "scene_sha256": canonical_hash(base),
            "source_scope": {"frames": [0, 4, 6]}, "timing": {"seconds": 2., "max_controls": 60},
            "gates": {"object_speed": .15}, "control": {"root_assistance": "disabled"}}
        self.docs = {"geometry-input/scene.json": base, "geometry/scene.json": candidate,
            "geometry-config.json": config, "geometry/input-config.json": deepcopy(config),
            "geometry-input/exercise-task.json": task,
            "geometry/exercise-task.json": {**deepcopy(task), "scene_sha256": canonical_hash(candidate)},
            "geometry/previous-geometry.json": {"objects": previous},
            "inputs/base-manifest.json": {"schema_version": 1, "files": {"source.mp4": source_sha}},
            "inputs/masks-manifest.json": {"schema_version": 1, "files": {"mask.png": "b" * 64}}}
        settings = {"profile": inputs.STUDY_PROFILE, "scene_sha256": canonical_hash(base),
            "conditional_exterior": True, "task_file_sha256": digest(task),
            "source_manifest_sha256": digest(self.docs["inputs/base-manifest.json"]),
            "mask_manifest_sha256": digest(self.docs["inputs/masks-manifest.json"]), "selections": []}
        report = {"profile": inputs.STUDY_PROFILE, "status": "completed", "stage": "finished",
            "original_inputs_unchanged": True, "candidate_directory": "geometry",
            "geometry_admission": {"status": "ready"}, "clearance": {"status": "complete"},
            "physical_validation": "unverified", "objects": []}
        for obj in objects:
            name, ref = obj["id"], obj["provenance"]["source_reference"]
            qualified = name == "weight"
            status = "conditional_proposal" if qualified else "no_qualified_improvement"
            before, after = {"iou": .5}, {"iou": .6} if qualified else None
            detail = {"status": status, "source_reference": deepcopy(ref),
                "camera_hypothesis": deepcopy(obj["provenance"]["camera_hypothesis"]),
                "input_hashes": {"scene_sha256": canonical_hash(base)}, "baseline": {"scores": before}}
            if qualified:
                detail.update(candidate={"scores": after, "geometry": deepcopy(geometry)},
                    qualification={key: True for key in inputs._QUALIFICATION})
            self.docs[f"objects/{name}/result.json"] = {"status": status, "report": detail,
                "geometry_config": deepcopy(config) if qualified else None,
                "candidate_geometry": deepcopy(geometry) if qualified else None}
            settings["selections"].append({"world_object_id": name, "source_frame": ref["frame"],
                "document_sha256": ref["document_sha256"], "observation_entity_id": name})
            report["objects"].append({"object_id": name, "status": status, "report": f"objects/{name}/result.json",
                "before_conditional_mask_fit": before, "after_conditional_mask_fit": after})
        self.docs.update({"settings.json": settings, "report.json": report,
            "geometry/geometry-hypothesis.json": {"profile": "shared-world-geometry-hypothesis-v1", "status": "ready",
                "initial_geometry_admitted": True, "base_scene_sha256": canonical_hash(base),
                "candidate_scene_sha256": canonical_hash(candidate), "input_config_sha256": digest(config),
                "selected_objects": ["weight"], "exercise_task": {"status": "metadata_only_scene_rebind",
                    "changed_fields": ["scene_sha256"], "parameters_source_scope_and_gates_unchanged": True,
                    "previous_file_sha256": digest(task), "new_file_sha256": digest(self.docs["geometry/exercise-task.json"])}}})
        self.root_status, self.candidate_status = "completed", "ready"
        self.root_extra = {}
        self.seal()

    def seal(self):
        self.docs["geometry-input/manifest.json"] = {"schema_version": 1, "status": "ready", "files": {
            name.removeprefix("geometry-input/"): digest(doc) for name, doc in self.docs.items()
            if name.startswith("geometry-input/") and name != "geometry-input/manifest.json"}}
        self.docs["geometry/geometry-hypothesis.json"]["base_manifest_sha256"] = digest(self.docs["geometry-input/manifest.json"])
        self.docs["geometry/manifest.json"] = {"schema_version": 1, "status": self.candidate_status, "files": {
            name.removeprefix("geometry/"): digest(doc) for name, doc in self.docs.items()
            if name.startswith("geometry/") and name != "geometry/manifest.json"}}
        self.docs["report.json"].update(settings_sha256=canonical_hash(self.docs["settings.json"]),
            candidate_manifest_sha256=digest(self.docs["geometry/manifest.json"]))
        raw = {name: encoded(doc) for name, doc in self.docs.items()}
        raw["unread/source.mp4"] = b"authored source bytes, not a decodable video"
        manifest = {"schema_version": 1, "status": self.root_status,
            "files": {**{name: digest(value) for name, value in raw.items()}, **self.root_extra}}
        raw["manifest.json"] = encoded(manifest)
        for name, value in raw.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        return digest(raw["manifest.json"])

    def read(self, **kwargs):
        return inputs.read_geometry_resume_inputs(self.root, deadline=time.monotonic() + 5., **kwargs)


@pytest.fixture
def packet(tmp_path):
    return Packet(tmp_path / "study")


def test_metadata_snapshot_preserves_two_alternatives_task_and_claims_without_media(packet, monkeypatch):
    original = inputs.read_package_file
    read = []

    def metadata_only(root, name, **kwargs):
        assert not name.endswith((".mp4", ".png"))
        read.append(name)
        return original(root, name, **kwargs)

    monkeypatch.setattr(inputs, "read_package_file", metadata_only)
    selected = packet.read(expected_manifest_sha256=digest((packet.root / "manifest.json").read_bytes()))
    receipt, snapshot = selected.receipt(), selected.snapshot_files()
    assert selected.candidate_run == packet.root / "geometry"
    assert receipt["metadata_files_read"] == len(read) == len(set(read)) == 17
    assert receipt["metadata_bytes_read"] < inputs.MAX_BYTES
    assert receipt["selected_objects"] == ["weight"]
    assert all(receipt["preserved"].values()) and not any(receipt["work"].values())
    assert receipt["source_fit"] == "retained_conditional_study_claims"
    assert len(receipt["retained_source_fit"]) == 2
    assert "manifest.json" not in snapshot and snapshot["study-manifest.json"] == (packet.root / "manifest.json").read_bytes()
    for name in ("geometry-input/scene.json", "geometry/scene.json", "geometry/exercise-task.json"):
        assert snapshot[name] == encoded(packet.docs[name])
    assert packet.docs["geometry/exercise-task.json"]["gates"] == packet.docs["geometry-input/exercise-task.json"]["gates"]


@pytest.mark.parametrize("case", ["root_failed", "candidate_failed", "profile", "unfinished", "clearance", "no_opt_in",
    "path", "camera", "frame", "body", "floor", "mass", "task_gate", "task_clock", "config_bytes",
    "compound", "prior_box", "qualification", "scores", "failed_object"])
def test_semantic_changes_rejected_even_when_the_modified_metadata_is_resealed(packet, case):
    docs = packet.docs
    if case == "root_failed": packet.root_status = "failed"
    elif case == "candidate_failed": packet.candidate_status = "failed"
    elif case == "profile": docs["settings.json"]["profile"] = "different"
    elif case == "unfinished": docs["report.json"]["stage"] = "clearance"
    elif case == "clearance": docs["report.json"]["clearance"]["status"] = "partial"
    elif case == "no_opt_in": docs["settings.json"]["conditional_exterior"] = False
    elif case == "path": docs["report.json"]["candidate_directory"] = "../elsewhere"
    elif case == "camera": docs["objects/weight/result.json"]["report"]["camera_hypothesis"]["scale"] = 2.
    elif case == "frame": docs["settings.json"]["selections"][0]["source_frame"] = 7
    elif case == "body": docs["geometry/scene.json"]["human_spec"]["frames"] = [0, 6]
    elif case == "floor": docs["geometry/scene.json"]["coordinates"]["floor_z"] = 1.
    elif case == "mass": docs["geometry/scene.json"]["objects"][0]["mass"] = 3.
    elif case == "task_gate": docs["geometry/exercise-task.json"]["gates"]["object_speed"] = .2
    elif case == "task_clock": docs["geometry/exercise-task.json"]["timing"]["seconds"] = 3.
    elif case == "config_bytes": docs["geometry/input-config.json"]["assumptions"] = ["changed"]
    elif case == "compound": docs["geometry/scene.json"]["objects"][0]["geometry"]["parts"][0]["size"][0] = .1
    elif case == "prior_box": docs["geometry/previous-geometry.json"]["objects"][0]["previous_geometry"]["size"][0] = .1
    elif case == "qualification": docs["objects/weight/result.json"]["report"]["qualification"]["uncertain_envelope_preserved"] = False
    elif case == "scores": docs["report.json"]["objects"][0]["after_conditional_mask_fit"] = {"iou": 1.}
    elif case == "failed_object":
        docs["objects/support/result.json"]["status"] = "failed"
        docs["objects/support/result.json"]["report"]["status"] = "failed"
        docs["report.json"]["objects"][1]["status"] = "failed"
    packet.seal()
    with pytest.raises(ValueError):
        packet.read()


@pytest.mark.parametrize("pin", ["bad", "f" * 64, 1])
def test_manifest_pin_rejected(packet, pin):
    with pytest.raises(ValueError):
        packet.read(expected_manifest_sha256=pin)


def test_selected_member_tampering_and_later_mutation_rejected(packet):
    selected = packet.read()
    path = packet.root / "geometry/scene.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed after qualification"):
        inputs.recheck_geometry_resume_inputs(selected, deadline=time.monotonic() + 5.)
    with pytest.raises(ValueError):
        packet.read()


@pytest.mark.parametrize("bound,value", [("MAX_FILES", 2), ("MAX_BYTES", 100)])
def test_metadata_budget_stops_before_unbounded_reads(packet, monkeypatch, bound, value):
    monkeypatch.setattr(inputs, bound, value)
    with pytest.raises(ValueError):
        packet.read()


def test_expired_deadline_and_unsafe_manifest_path_rejected(packet):
    with pytest.raises(TimeoutError):
        inputs.read_geometry_resume_inputs(packet.root, deadline=0.)
    packet.root_extra["../escape.json"] = "a" * 64
    packet.seal()
    with pytest.raises(ValueError):
        packet.read()


def test_cached_manifest_reuse_cannot_bypass_its_tighter_byte_bound(tmp_path):
    (tmp_path / "manifest.json").write_bytes(b" " * 100)
    reader = inputs._Reader(tmp_path, time.monotonic() + 5.)
    assert len(reader.take("manifest.json", maximum=100)) == 100
    with pytest.raises(ValueError, match="per-file bound"):
        reader.take("manifest.json", maximum=99)
