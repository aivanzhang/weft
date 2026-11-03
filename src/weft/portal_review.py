"""One budgeted Muse review of sealed construction evidence, outside scored runs."""

import json
from pathlib import Path
import time

from weft.io import digest, local_time, write_json
from weft.lab_worker import atomic_json
from weft.meta_access import access_status
from weft.scene_advisor import ESTIMATE, MODEL, advise_run
from weft.portal_supports import body_path
from weft.portal_scene_context import construction_context
from weft.openai_access import PROJECT_ROOT

TERMINAL = {"proposed", "failed", "cancelled", "interrupted"}


def availability(project):
    status = access_status(project)
    return {**status, "model": MODEL, "estimated_one_call_usd": ESTIMATE,
            "available": bool(status["credential_present"] and status["budget_configured"]
                              and status.get("remaining_usd", 0) >= ESTIMATE)}


def _saved_body_source(root, config, status):
    from hashlib import sha256
    from weft.person_inventory import make_conditional_person_inventory
    from weft.portal_saved_body import PROFILE, MODELING_SCOPE, validate_request
    from weft.shared_world_packaging import read_package_file, read_package_manifest
    from weft.spatial_scene_inputs import _json

    def read(directory, name, maximum, pin=None):
        raw = read_package_file(directory, name, max_bytes=maximum, expected_sha256=pin)
        return _json(raw, maximum), sha256(raw).hexdigest()

    if status.get("route") != "exercise" or status.get("lab") != "lab":
        raise ValueError("Saved-body review requires the original ready body lab")
    receipt, receipt_pin = read(root, "saved-body/receipt.json", 128 << 10, status.get("receipt_sha256"))
    request, request_pin = read(root, "saved-body/request.json", 128 << 10, config.get("saved_body_request_sha256"))
    if (receipt_pin != status.get("receipt_sha256") or request_pin != config.get("saved_body_request_sha256")
            or receipt.get("profile") != PROFILE or request.get("profile") != PROFILE
            or request.get("child_id") != config.get("id") or config.get("id") != root.name
            or receipt.get("parent_id") != request.get("parent_id") or request.get("parent_id") != config.get("parent_id")
            or receipt.get("source_sha256") != config.get("source_sha256")
            or receipt.get("parent_files") != request.get("parent_files")
            or request.get("identity") != config.get("saved_body_identity")):
        raise ValueError("Saved-body review receipt or original source binding differs")
    payload = validate_request(request.get("payload"))
    if payload["modeling_scope"] != MODELING_SCOPE:
        raise ValueError("Saved-body review requires explicit authored modeling scope")
    inventory = make_conditional_person_inventory(config["source_sha256"], request.get("original_exercise_intent"))
    source = root / "saved-body/world"
    manifest_raw, files = read_package_manifest(source)
    required = {"source.mp4", "human.json", "evidence.json", "inventory.json", "scene.json", "inputs/body-source-manifest.json"}
    if not required <= files.keys():
        raise ValueError("Saved-body review is missing sealed construction inputs")
    if sha256(manifest_raw).hexdigest() != receipt.get("world_manifest_sha256"):
        raise ValueError("Saved-body construction seal differs from its receipt")
    original, original_pin = read(source, "inputs/body-source-manifest.json", 1 << 20,
                                  files.get("inputs/body-source-manifest.json"))
    if (original_pin != receipt.get("body_manifest_sha256") or original_pin != request.get("body_manifest_sha256")
            or original.get("profile") != "exercise-body-source-v1" or original.get("status") != "construction_only"
            or original.get("simulation_ready") is not False or not isinstance(original.get("files"), dict)):
        raise ValueError("Saved-body review original construction-body seal differs")
    for name in ("source.mp4", "human.json", "evidence.json"):
        if name not in files or files[name] != original["files"].get(name):
            raise ValueError("Saved-body review must retain the exact original video and body evidence")
    for name in ("human.json", "evidence.json"):
        value, _ = read(source, name, 32_000_000, files[name])
        if value.get("source_sha256") != config["source_sha256"]:
            raise ValueError("Saved-body review body/evidence source differs")
    actual, inventory_pin = read(source, "inventory.json", 4096, files.get("inventory.json"))
    if actual != inventory or receipt.get("inventory") != inventory:
        raise ValueError("Saved-body review cannot relabel original inventory or authored ground")
    lab_manifest, lab_files = read_package_manifest(root / "lab/run")
    lab_config, _ = read(root / "lab", "lab-config.json", 64 << 10)
    if (lab_config.get("manifest_sha256") != sha256(lab_manifest).hexdigest()
            or "scene.json" not in files or files["scene.json"] != lab_files.get("scene.json")):
        raise ValueError("Saved-body review world differs from the current lab")
    return source, {"profile": "portal-saved-body-review-v1", "receipt_sha256": receipt_pin,
                    "request_sha256": request_pin, "body_manifest_sha256": original_pin,
                    "inventory_sha256": inventory_pin, "inventory": inventory,
                    "world_manifest_sha256": sha256(manifest_raw).hexdigest(),
                    "lab_manifest_sha256": sha256(lab_manifest).hexdigest()}


def construction_input(root, *, project=PROJECT_ROOT):
    root = Path(root).resolve()
    config = json.loads((root / "job.json").read_bytes())
    status = json.loads((root / "status.json").read_bytes())
    if status.get("status") != "ready":
        raise ValueError("Finish reconstruction before requesting a Muse review")
    route = status.get("route")
    if route not in {"exercise", "objects"}:
        raise ValueError("This project's construction format is not supported by Muse review")
    attachment = None
    saved_body = config.get("kind") == "saved_body"
    if saved_body:
        source, attachment = _saved_body_source(root, config, status)
    elif status.get("equipment_candidate_id"):
        from weft.portal_perception_review import validate_attachment

        source, attachment = validate_attachment(root, project=project)
    elif config.get("kind") == "imported_lab":
        from weft.portal_evidence_attachment import validate_attachment

        source, attachment = validate_attachment(root)
    else:
        source = body_path(root, config) if route == "exercise" else root / "planar/run"
    source_name = "source.mp4" if saved_body else config["source_file"]
    names = ["evidence.json", source_name] + (["human.json"] if route == "exercise" else [])
    manifest_path = source / "manifest.json"
    if any(p.is_symlink() for p in (manifest_path, *manifest_path.parents)) or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("Unsafe construction manifest")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest["files"]
    pins = {}
    for name in names:
        path = source / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size > (100_000_000 if name == source_name else 32_000_000)):
            raise ValueError("Unsafe construction evidence")
        pins[name] = digest(path)
        if files.get(name) != pins[name]:
            raise ValueError("Construction evidence no longer matches its saved manifest")
    if pins[source_name] != config["source_sha256"]:
        raise ValueError("Construction video differs from the uploaded source")
    binding = {"construction_directory": source.relative_to(root).as_posix(),
               "manifest_sha256": digest(manifest_path), "input_sha256": pins,
               "scope": "Construction observations and authored body assumptions only; no scores, policies or final evaluation"}
    if attachment is not None:
        binding["saved_body_attachment" if saved_body else "perception_attachment" if status.get("equipment_candidate_id") else "imported_attachment"] = attachment
    if saved_body:
        from weft.scene_advisor import DEFAULT_QUESTIONS
        intent = attachment["inventory"]["original_exercise_intent"]
        binding.update(review_scope="conditional_body_authored_ground", scope="Original video and pose evidence with an explicitly authored body/ground-only model. Omitted equipment, obstacles, other people and contacts are not verified absent. No scores, policies, recovered anatomy or physical validation.")
        binding["review_questions"] = [*DEFAULT_QUESTIONS,
            "Original uploaded inventory is " + intent["inventory"] + " (basis: " + intent["basis"] + "). The current model intentionally omits equipment, obstacles, other people and their contacts; omission is not absence. Ground and anatomy are authored unverified priors. Identify missing interactions or alternative support hypotheses without treating this body/ground approximation as a complete reconstructed scene."]
    if status.get("equipment_candidate_id"):
        from weft.portal_perception_review import CONDITIONAL_QUESTION
        from weft.scene_advisor import DEFAULT_QUESTIONS

        binding["review_questions"] = [*DEFAULT_QUESTIONS, CONDITIONAL_QUESTION]
    if config.get("kind") == "support_experiment":
        binding.update(review_scope="original_body_only",
                       scope="Original body construction evidence only. Added supports, load windows and child-world control settings are not reviewed; no scores, policies or final evaluation.")
    return source, binding


def review_input(root, *, project=PROJECT_ROOT):
    source, binding = construction_input(root, project=project)
    world, world_binding = construction_context(root)
    if world is not None:
        binding.update(review_scope="coupled_world_assumptions", simulation_world=world_binding,
                       scope="Source observations plus current coupled-world construction assumptions; authored supports are not recovered evidence. No scores, policies or final evaluation.")
    if "saved_body_attachment" in binding:
        if world is None or world.get("objects") != []:
            raise ValueError("Saved-body review requires its explicit body/ground-only world context")
        binding.update(review_scope="conditional_body_authored_ground",
                       scope="Source observations plus an explicitly authored body/ground-only world. Original scene inventory remains unchanged; omitted entities and contacts are not verified absent. No scores, policies or physical validation.")
    return source, binding, world


def execute_review(root, *, project):
    root = Path(root)
    folder = root / "muse"
    started = time.monotonic()
    status_path = folder / "status.json"
    if status_path.exists():
        saved = json.loads(status_path.read_bytes())
        if saved.get("status") in TERMINAL:
            return 0 if saved["status"] == "proposed" else 1
    try:
        source, binding, world = review_input(root, project=project)
        write_json(folder / "binding.json", binding)
        coupled = world is not None
        if (folder / "cancel-request.json").exists():
            atomic_json(folder / "status.json", {"status": "cancelled", "message": "Cancelled before any Muse request"})
            return 0
        atomic_json(folder / "status.json", {"status": "running", "message":
            "Muse is reviewing source observations and current world assumptions, including authored supports where present."
            if coupled else "Muse is reviewing construction evidence and proposing tests",
            "review_scope": binding.get("review_scope", "construction_observations")})
        receipt = advise_run(source, folder / "advice", execute=True, max_cost_usd=ESTIMATE, project_root=project,
                             **({"questions": binding["review_questions"]} if "review_questions" in binding else {}),
                             **({"scene_context": world} if coupled else {}))
        _, after, _ = review_input(root, project=project)
        if after != binding:
            raise ValueError("Construction inputs changed during Muse review")
        atomic_json(folder / "status.json", {"status": "proposed", "message":
            "World review ready. Authored supports and control assumptions are not physical observations. Proposals have not been tested or applied."
            if coupled else "Review ready. These proposals have not been tested or applied.",
            "review_scope": binding.get("review_scope", "construction_observations"),
            "seconds": time.monotonic() - started, "cost_usd": receipt["cost_usd"], "completed_at": local_time()})
        return 0
    except Exception as error:
        # API errors can contain provider data; publish the saved adapter's sanitized failure class.
        atomic_json(folder / "status.json", {"status": "failed", "message": "Muse review failed; evidence and uncertain charges are preserved. No automatic retry.",
            "error_type": type(error).__name__, "seconds": time.monotonic() - started})
        return 1
