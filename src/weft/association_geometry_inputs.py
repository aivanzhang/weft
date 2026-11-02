"""Reuse pinned conditional association studies without refitting or promoting identity."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile

from weft.exercise_geometry_inputs import GeometryResumeInputs, _json, _signature
from weft.io import canonical_hash
from weft.mask_track_hypotheses import CONTRACT as ASSOCIATION_CONTRACT
from weft.shared_exercise_task import GATES_SHA256
from weft.shared_world_geometry import _invariants, validate_geometry_hypothesis_config
from weft.shared_world_packaging import read_package_file, read_package_manifest
from weft.world_evidence import load_world_evidence


PROFILE = "conditional-association-geometry-admission-v1"
MAX_ARCHIVE_BYTES = 48 << 20
MAX_ARCHIVE_MEMBERS = 1600
MAX_METADATA_BYTES = 16 << 20
MAX_INTAKE_BYTES = 112 << 20
QUALIFICATION = {"explicit_conditional_exterior", "image_stationarity_screen", "no_new_foreground_misses",
                 "all_positive_row_hulls_preserved", "within_original_envelope", "strict_conditional_exterior_reduction"}
SCOPE = ("Pinned prior source-image study claims, not newly fitted or independently measured. "
         "Initial identity, static camera/shape, depth and inertia remain unverified assumptions; no physical contact claim.")


def _require(condition, message):
    if not condition:
        raise ValueError("Association geometry inputs: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Association geometry receipt deadline")


def _sha(raw):
    return sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _name(name):
    _require(type(name) is str and 1 <= len(name) <= 512 and "\\" not in name
             and not PurePosixPath(name).is_absolute() and all(part not in ("", ".", "..") for part in name.split("/")),
             "unsafe archive member name")


def validate_association_archive(raw, *, expected_sha256, deadline):
    """Read exact stored members in memory; never extract or execute archive contents."""
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_ARCHIVE_BYTES and _sha(raw) == expected_sha256,
             "archive bytes differ from their trusted receipt")
    files, total = {}, 0
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        members = archive.infolist()
        _require(1 <= len(members) <= MAX_ARCHIVE_MEMBERS, "archive member bound")
        for item in members:
            _check(deadline)
            _name(item.filename)
            _require(item.filename not in files and not item.is_dir()
                     and item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1
                     and item.file_size == item.compress_size and 0 <= item.file_size <= MAX_METADATA_BYTES
                     and not stat.S_ISLNK(item.external_attr >> 16), "duplicate, compressed or nonregular archive member")
            total += item.file_size
            _require(total <= MAX_ARCHIVE_BYTES, "archive expanded byte bound")
            files[item.filename] = archive.read(item)
    _require("inventory.json" in files, "archive inventory missing")
    inventory = _json(files["inventory.json"])
    _require(set(inventory) == {"profile", "files"} and inventory["profile"] == "association-proof-archive-v1"
             and inventory["files"] == {name: _sha(data) for name, data in files.items() if name != "inventory.json"},
             "archive inventory differs")
    _check(deadline)
    return files


def _archive(files, deadline):
    inventory = _bytes({"profile": "association-proof-archive-v1", "files": {name: _sha(raw) for name, raw in files.items()}})
    _require(len(files) + 1 <= MAX_ARCHIVE_MEMBERS and sum(map(len, files.values())) + len(inventory) <= MAX_ARCHIVE_BYTES,
             "proof archive bound")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted({**files, "inventory.json": inventory}.items()):
            _check(deadline); _name(name)
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(item, raw)
    result = stream.getvalue()
    _require(len(result) <= MAX_ARCHIVE_BYTES, "serialized archive byte bound")
    return result


def _validate_study(files):
    seal = _json(files["study/manifest.json"])
    _require(seal.get("status") == "complete" and seal.get("files") == {
        name.removeprefix("study/"): _sha(raw) for name, raw in files.items()
        if name.startswith("study/") and name != "study/manifest.json"}, "archived study seal differs")
    plan, summary = _json(files["study/plan.json"]), _json(files["study/result.json"])
    for root in ("source", "scene"):
        original_seal = _json(files[root + "/manifest.json"])
        _require(_sha(files[root + "/manifest.json"]) == plan[root]["manifest_sha256"], "archived source seal differs")
        for name, raw in files.items():
            if name.startswith(root + "/") and name != root + "/manifest.json":
                _require(original_seal["files"].get(name[len(root) + 1:]) == _sha(raw), "archived source member differs")
    association, proof = _json(files["study/association.json"]), _json(files["study/encoded-mask-proof.json"])
    source = _json(files["source/evidence/scene-observations.json"])
    masks = _json(files["source/masks/mask-evidence.json"])
    base = _json(files["scene/scene.json"])
    _require(summary.get("status") == "completed" and summary.get("all_source_hashes_unchanged") is True
             and summary.get("geometry_installed") is False and summary.get("physical_identity_verified") is False
             and summary.get("contact") == "unknown", "source study is incomplete or promoted")
    _require(association.get("contract") == ASSOCIATION_CONTRACT
             and proof.get("association_sha256") == canonical_hash(association)
             and proof.get("encoded_to_raster_derivation_verified") is True
             and proof.get("source_document_sha256") == _sha(files["source/evidence/scene-observations.json"])
             and proof.get("mask_document_sha256") == _sha(files["source/masks/mask-evidence.json"])
             and association.get("source_document_sha256") == proof["source_document_sha256"]
             and association.get("mask_document_sha256") == proof["mask_document_sha256"], "association/source proof identity differs")
    for name, digest in plan["runtime"].items():
        _require(_sha(files["study/runtime/" + name]) == digest, "frozen source-study runtime differs")
    _require(_sha(files["study/worker.py"]) == plan["worker_sha256"], "source-study executor differs")
    _require(proof["codec_hashes"] == _json(files["source/masks/report.json"])["codec_hashes"], "original codec provenance differs")
    construction = json.loads(load_world_evidence({proof["source_document_sha256"]: files["source/evidence/scene-observations.json"]}).construction_bytes())["sources"][0]
    if "identity_groups" in proof:
        # Historical receipts copied global groups; reuse only when no partition was withheld.
        _require(all(row["admission"] == "included" for row in construction["observations"])
                 and proof["identity_groups"] == masks["identity_groups"], "historical global groups cannot expose withheld evidence")
    else:
        _require(proof.get("construction_relationships") == construction["relationships"], "partition-filtered relationships differ")
    rows = {row["id"]: row for row in masks["observations"]}
    slots = {(row["frame"], row["object_id"]): row for row in construction["observations"]}
    records = proof["records"]
    _require(type(records) is list and len(records) == proof["decoded_masks"] == summary["counts"]["encoded_decodes"]
             and len({row["id"] for row in records}) == len(records), "decode receipt coverage differs")
    loaded = association["loaded_rasters"]
    _require({row["id"]: (row["sha256"], row["bytes"]) for row in loaded} ==
             {row["id"]: (row["raster_sha256"], row["pixels"]) for row in records}, "association and encoded decode coverage differ")
    _require(sum(row["pixels"] for row in records) == proof["decoded_pixels"] <= 50_000_000, "encoded pixel count differs")
    for record in records:
        row = rows[record["id"]]
        _require(record["encoded_sha256"] == row["encoded_sha256"] and record["raster_sha256"] == row["raster_sha256"]
                 and slots[row["frame"], row["object_id"]]["admission"] == "included", "decoded record is changed or withheld")
    expected_frames = len(source["source"]["frame_times_s"])
    _require(len(association["frames"]) == expected_frames, "complete association frame registry required")
    indexed = {}
    for frame in association["frames"]:
        for row in frame["slots"]:
            original = rows[row["id"]]
            _require(original["frame"] == frame["frame"] and original["object_id"] == row["raw_object_id"], "raw association identities changed")
            indexed[frame["frame"], row["raw_object_id"]] = row
    _require(len(indexed) == len(masks["observations"]), "complete raw association slots required")
    selections, proposals, replacements, assumptions = plan["objects"], [], [], []
    _require(type(selections) is list and 1 <= len(selections) <= 2 and len(set(selections)) == len(selections), "bounded fixed object selection required")
    original_objects = {obj["id"]: obj for obj in base["objects"]}
    for name in selections:
        proposal = _json(files["study/" + name + ".json"])
        detail = proposal["report"]
        _require(detail["profile"] == "conditional-association-equipment-geometry-v1"
                 and proposal["status"] == detail["status"] and detail["object_id"] == name
                 and detail["input_hashes"]["scene_sha256"] == canonical_hash(base)
                 and detail["association_proof"] == proof and detail["construction_frames"] == plan["construction_frames"]
                 and detail["static_prior"] == plan["static_prior"] and detail["initial_association_prior"] == plan["initial_association_prior"]
                 and detail["raw_identity_reclassified"] is False and detail["physical_identity_verified"] is False
                 and detail["contact"] == "unknown" and detail["physical_validation"] == "unverified"
                 and detail["source_slots_filled"] == 0, "proposal changed fixed source/prior/identity scope")
        ref = original_objects[name]["provenance"]["source_reference"]
        _require(detail["source_reference"] == ref, "original scene reference changed")
        chain = detail["conditional_source_chain"]
        _require(chain and [row["frame"] for row in chain] == list(range(chain[0]["frame"], plan["construction_frames"][-1] + 1)), "incomplete conditional chain")
        for link in chain:
            original, linked = rows[link["evidence_id"]], indexed[link["frame"], link["raw_object_id"]]
            _require(link["raw_object_id"] == ref["object_id"] and linked["id"] == link["evidence_id"]
                     and linked["previous_id"] == link["previous_id"] and linked["anchor_id"] == chain[0]["evidence_id"]
                     and link["identity"] == original["identity"] and link["original_source"] == original.get("original_source")
                     and slots[link["frame"], link["raw_object_id"]]["admission"] == "included", "conditional chain lost raw uncertainty or source custody")
        _require(set(detail["qualification"]) == QUALIFICATION, "unknown source-image qualification")
        if proposal["status"] == "conditional_proposal":
            _require(all(v is True for v in detail["qualification"].values()), "unqualified candidate cannot be admitted")
            config = validate_geometry_hypothesis_config(proposal["geometry_config"])
            _require(config["base_scene_sha256"] == canonical_hash(base) and len(config["replacements"]) == 1
                     and config["replacements"][0] == {"object_id": name, "geometry": proposal["candidate_geometry"]}
                     and detail["candidate_geometry"] == proposal["candidate_geometry"], "candidate config differs from qualified geometry")
            replacements.extend(config["replacements"])
            assumptions.extend(value for value in config["assumptions"] if value not in assumptions)
        else:
            _require(proposal["status"] == "no_qualified_improvement" and proposal["geometry_config"] is None,
                     "failed proposal cannot be silently omitted")
        proposals.append({"object_id": name, "status": proposal["status"], "source_fit": detail["source_fit"],
                          "scope": "Retained conditional study claims, not newly measured"})
    _require(replacements, "no qualified source-image candidate")
    config = validate_geometry_hypothesis_config({"schema_version": 1, "profile": "shared-world-geometry-hypothesis-config-v1",
        "base_scene_sha256": canonical_hash(base), "assumptions": assumptions, "replacements": replacements})
    return plan, config, proposals


def prepare_association_study_archive(study, *, expected_manifest_sha256, deadline):
    """Pin complete prior study and every consumed source member; do not repeat source fits."""
    root = Path(study).absolute()
    manifest_raw, pins = read_package_manifest(root)
    _require(_sha(manifest_raw) == expected_manifest_sha256 and _json(manifest_raw).get("status") == "complete", "complete pinned study seal required")
    files = {"study/manifest.json": manifest_raw}
    size = len(manifest_raw)
    def reserve(name, amount):
        _require(type(amount) is int and 0 <= amount <= MAX_METADATA_BYTES
                 and len(files) + int(name not in files) + 1 <= MAX_ARCHIVE_MEMBERS
                 and size - len(files.get(name, b"")) + amount <= MAX_ARCHIVE_BYTES - (1 << 20),
                 "source archive reservation exceeds file/byte bound")
    def retain(name, data):
        nonlocal size
        reserve(name, len(data))
        size += len(data) - len(files.get(name, b""))
        files[name] = data
    for name, digest in pins.items():
        _check(deadline)
        reserve("study/" + name, 0)
        data = read_package_file(root, name, expected_sha256=digest,
            max_bytes=min(MAX_METADATA_BYTES, MAX_ARCHIVE_BYTES - (1 << 20) - size))
        retain("study/" + name, data)
    plan, result = _json(files["study/plan.json"]), _json(files["study/result.json"])
    roots, root_pins = {}, {}
    for name, key in (("source", "source"), ("scene", "scene")):
        roots[name] = Path(plan[key]["path"]).absolute()
        reserve(name + "/manifest.json", 8 << 20)
        raw, root_pins[name] = read_package_manifest(roots[name])
        _require(_sha(raw) == plan[key]["manifest_sha256"], "original source manifest changed")
        retain(name + "/manifest.json", raw)
    for path, row in result["source_members"].items():
        _check(deadline)
        target = Path(path).absolute()
        matches = [(name, target.relative_to(directory).as_posix()) for name, directory in roots.items() if target.is_relative_to(directory)]
        _require(len(matches) == 1, "consumed source member outside declared roots")
        name, relative = matches[0]
        pin = plan[name]["manifest_sha256"] if relative == "manifest.json" else root_pins[name].get(relative)
        _require(pin == row["sha256"], "source member lost original manifest custody")
        reserve(name + "/" + relative, row["bytes"])
        data = read_package_file(roots[name], relative, expected_sha256=pin, max_bytes=max(1, row["bytes"]))
        _require(len(data) == row["bytes"], "source receipt byte count differs")
        retain(name + "/" + relative, data)
    reserve("scene/exercise-task.json", 64 << 10)
    retain("scene/exercise-task.json", read_package_file(roots["scene"], "exercise-task.json",
        expected_sha256=root_pins["scene"]["exercise-task.json"], max_bytes=64 << 10))
    resolved, config, proposals = _validate_study(files)
    archive = _archive(files, deadline)
    receipt = {"schema_version": 1, "profile": "association-geometry-source-receipt-v1", "scope": SCOPE,
        "study_manifest_sha256": expected_manifest_sha256, "archive_sha256": _sha(archive), "archive_members": len(files) + 1,
        "base_manifest_sha256": plan["scene"]["manifest_sha256"], "base_scene_sha256": config["base_scene_sha256"],
        "selected_objects": [row["object_id"] for row in config["replacements"]], "retained_source_fit": proposals,
        "original_task_sha256": _sha(files["scene/exercise-task.json"]), "physical_validation": "unverified",
        "work": {"fits": 0, "mask_decodes": 0, "inference_calls": 0, "api_calls": 0}}
    _check(deadline)
    return archive, receipt, config, Path(resolved["scene"]["path"])


def read_association_geometry_inputs(run, *, expected_manifest_sha256, deadline):
    """Product intake for a sealed admitted candidate with its complete conditional proof."""
    root = Path(run).absolute()
    manifest_signature = _signature(root / "manifest.json")
    manifest_raw, pins = read_package_manifest(root)
    _require(_signature(root / "manifest.json") == manifest_signature, "admission manifest changed during read")
    _require(_sha(manifest_raw) == expected_manifest_sha256 and _json(manifest_raw).get("profile") == PROFILE
             and _json(manifest_raw).get("status") == "completed", "complete pinned admission study required")
    loaded = {"manifest.json": manifest_raw}
    signatures = {"manifest.json": manifest_signature}
    total = len(manifest_raw)
    def member(name, maximum=MAX_METADATA_BYTES):
        nonlocal total
        _check(deadline)
        _require(name in pins, "admission member missing")
        if name in loaded:
            return loaded[name]
        _require(len(loaded) < 32 and total < MAX_INTAKE_BYTES, "intake aggregate metadata bound")
        signature = _signature(root / name)
        data = read_package_file(root, name, expected_sha256=pins[name], max_bytes=min(maximum, MAX_INTAKE_BYTES - total))
        _require(_signature(root / name) == signature, "admission member changed during read")
        total += len(data)
        loaded[name], signatures[name] = data, signature
        return data
    receipt = _json(member("association-context.json"))
    archive = member("association-context.zip", MAX_ARCHIVE_BYTES)
    archived = validate_association_archive(archive, expected_sha256=receipt["archive_sha256"], deadline=deadline)
    _, config, proposals = _validate_study(archived)
    selected = [row["object_id"] for row in config["replacements"]]
    _require(_sha(archived["study/manifest.json"]) == receipt["study_manifest_sha256"]
             and receipt["retained_source_fit"] == proposals and receipt["selected_objects"] == selected
             and receipt["base_manifest_sha256"] == _sha(archived["scene/manifest.json"])
             and receipt["base_scene_sha256"] == config["base_scene_sha256"], "retained source study receipt differs")
    _require(_json(member("geometry-config.json")) == config, "installed config differs from retained source proposal")
    candidate_manifest = _json(member("geometry/manifest.json"))
    _require(candidate_manifest.get("status") == "ready", "candidate lacks ordinary geometry admission")
    for name, digest in candidate_manifest["files"].items():
        _require(pins.get("geometry/" + name) == digest, "candidate/study seals differ")
    base, candidate = _json(member("baseline/scene.json")), _json(member("geometry/scene.json"))
    _require(canonical_hash(base) == config["base_scene_sha256"] and loaded["baseline/scene.json"] == archived["scene/scene.json"]
             and set(base) == set(candidate) and candidate.get("learned_policy_available") is False, "baseline scene differs")
    preserved = _invariants(base, candidate, set(selected))
    _require(member("geometry/source-run/association-context.zip", MAX_ARCHIVE_BYTES) == archive
             and member("geometry/source-run/association-context.json") == loaded["association-context.json"],
             "candidate package lost its complete conditional proof")
    task_raw, changed_task_raw = member("baseline/exercise-task.json"), member("geometry/exercise-task.json")
    task, changed_task = _json(task_raw), _json(changed_task_raw)
    _require(task_raw == archived["scene/exercise-task.json"] and _sha(task_raw) == receipt["original_task_sha256"]
             and task["scene_sha256"] == canonical_hash(base)
             and changed_task == {**task, "scene_sha256": canonical_hash(candidate)}, "original task gates or interval changed")
    admission = _json(member("geometry/geometry-hypothesis.json"))
    _require(admission.get("profile") == "shared-world-geometry-hypothesis-v1"
             and admission.get("status") == "ready" and admission.get("initial_geometry_admitted") is True
             and admission.get("selected_objects") == selected and admission.get("base_scene_sha256") == canonical_hash(base)
             and admission.get("candidate_scene_sha256") == canonical_hash(candidate)
             and admission.get("input_config_sha256") == _sha(loaded["geometry-config.json"])
             and member("geometry/input-config.json") == loaded["geometry-config.json"], "ordinary admission receipt differs")
    rebind = admission.get("exercise_task", {})
    _require(rebind.get("status") == "metadata_only_scene_rebind" and rebind.get("changed_fields") == ["scene_sha256"]
             and rebind.get("parameters_source_scope_and_gates_unchanged") is True
             and rebind.get("previous_file_sha256") == _sha(task_raw)
             and rebind.get("new_file_sha256") == _sha(changed_task_raw), "ordinary task rebind receipt differs")
    for change in config["replacements"]:
        obj = next(row for row in candidate["objects"] if row["id"] == change["object_id"])
        _require(obj["shape"] == "compound" and obj["geometry"] == change["geometry"], "admitted candidate geometry differs")
    comparison = _json(member("matched-trials.json"))
    _require(comparison.get("profile") == "association-geometry-matched-trials-v1"
             and comparison.get("controller") == "authored" and comparison.get("seed") == 1729
             and comparison.get("reset_options") == {"jitter": False, "jitter_scale": 0.}
             and comparison.get("base_scene_sha256") == canonical_hash(base)
             and comparison.get("candidate_scene_sha256") == canonical_hash(candidate)
             and comparison.get("gates_sha256") == GATES_SHA256
             and comparison.get("horizon_steps") == task["interval"]["horizon_steps"], "matched original-task comparison differs")
    trials = comparison.get("trials")
    _require(type(trials) is list and [row.get("mode") for row in trials] == ["baseline", "candidate"], "both original-task trials required")
    for row, expected_task in zip(trials, (task, changed_task)):
        trial_raw = member("trials/" + row["mode"] + ".json")
        trial = _json(trial_raw)
        _require(row.get("task_sha256") == canonical_hash(expected_task) and row.get("trial_sha256") == _sha(trial_raw)
                 and trial.get("seed") == comparison["seed"] and trial.get("reset_options") == comparison["reset_options"]
                 and all(row.get(key) == trial.get(key) for key in ("status", "control_steps", "metrics"))
                 and type(trial.get("control_steps")) is int and 0 <= trial["control_steps"] <= comparison["horizon_steps"],
                 "retained trial/task receipt differs")
        metrics = trial.get("metrics")
        if metrics is None:
            _require(trial["status"] == "reset_failed" and trial["control_steps"] == 0,
                     "missing trial metrics require an explicit pre-reset failure")
        else:
            _require(type(metrics) is dict and metrics.get("task_sha256") == canonical_hash(expected_task)
                     and metrics.get("scene_sha256") == expected_task["scene_sha256"]
                     and metrics.get("gates_sha256") == GATES_SHA256, "nested trial metric identities differ")
    summary = {"schema_version": 1, "profile": "exercise-geometry-resume-inputs-v1", "status": "selected_metadata_verified",
        "scope": SCOPE, "study_manifest_sha256": _sha(manifest_raw), "candidate_manifest_sha256": _sha(loaded["geometry/manifest.json"]),
        "baseline_scene_sha256": canonical_hash(base), "candidate_scene_sha256": canonical_hash(candidate),
        "baseline_task_sha256": canonical_hash(task), "task_sha256": canonical_hash(changed_task),
        "source_sha256": base["source_binding"]["source_sha256"], "selected_objects": receipt["selected_objects"],
        "preserved": preserved, "retained_source_fit": proposals, "source_fit": "retained_conditional_study_claims",
        "saved_clearance_status": "not_run_matched_original_task_trials_retained", "physical_validation": "unverified",
        "complete_proof": {"candidate_member": "source-run/association-context.zip", "sha256": receipt["archive_sha256"],
                           "scope": "Complete archive remains in the ordinary candidate package; metadata snapshot does not duplicate it."},
        "work": {"fits": 0, "media_decodes": 0, "physics_steps": 0, "inference_calls": 0, "api_calls": 0}}
    _check(deadline)
    for name, signature in signatures.items():
        _require(_signature(root / name) == signature, "admission metadata changed after validation")
    metadata = {name: raw for name, raw in loaded.items() if not name.endswith(".zip")}
    _require(len(metadata) <= 20 and sum(map(len, metadata.values())) <= 16 << 20,
             "ordinary geometry metadata snapshot bound")
    return GeometryResumeInputs(root, root / "geometry", tuple(metadata.items()), tuple(signatures.items()), _bytes(summary))
