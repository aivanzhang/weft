"""Explicit support experiments reuse a pinned body without altering its source project."""

import json
from pathlib import Path
import shutil
import time

from weft.exercise_support_inputs import validate_support_priors
from weft.io import digest, local_time, write_json
from weft.lab_worker import atomic_json
from weft.shared_world_packaging import read_package_file, read_package_manifest


def body_path(root, config):
    name = config.get("body_source_directory", "build/body_source")
    if name not in {"build/body_source", "body_source"}:
        raise ValueError("Unsupported saved body location")
    path = Path(root) / name
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Saved body paths cannot contain links")
    return path


def copy_body(source, target, *, expected_manifest, deadline):
    raw, files = read_package_manifest(source)
    if digest(source / "manifest.json") != expected_manifest or len(files) > 512:
        raise ValueError("Parent body identity or file bound changed")
    if json.loads(raw).get("profile") != "exercise-body-source-v1":
        raise ValueError("Support experiments require an existing source-bound body")
    target.mkdir()
    total = len(raw)
    for name, expected in files.items():
        if time.monotonic() >= deadline:
            raise TimeoutError("Body reuse exceeded its time allowance")
        data = read_package_file(source, name, max_bytes=64 * 1024**2, expected_sha256=expected)
        total += len(data)
        if total > 128 * 1024**2:
            raise ValueError("Saved body exceeds 128 MiB")
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    (target / "manifest.json").write_bytes(raw)
    if read_package_file(source, "manifest.json", max_bytes=1_000_000) != raw:
        raise ValueError("Parent body manifest changed during reuse")
    return {"manifest_sha256": expected_manifest, "copied_files": len(files), "copied_bytes": total}


def execute_support_job(root):
    from weft.exercise_scene_pipeline import prepare_exercise_scene

    root = Path(root).resolve()
    started, events = time.monotonic(), []
    def progress(stage, message):
        events.append({"stage": stage, "message": message, "time": local_time()})
        atomic_json(root / "status.json", {"status": "running", "stage": stage, "events": events})
    try:
        status = json.loads((root / "status.json").read_bytes())
        if status.get("status") in {"ready", "failed", "cancelled", "interrupted"}:
            return 0 if status["status"] == "ready" else 1
        config = json.loads((root / "job.json").read_bytes())
        parent_id = config["parent_id"]
        if len(parent_id) != 32 or any(c not in "0123456789abcdef" for c in parent_id):
            raise ValueError("Invalid parent project")
        parent = root.parent / parent_id
        parent_config = json.loads((parent / "job.json").read_bytes())
        if parent_config["source_sha256"] != config["source_sha256"]:
            raise ValueError("Parent video differs from this experiment")
        priors_raw = read_package_file(root, "support-priors.json", max_bytes=128_000,
                                      expected_sha256=config["support_priors_sha256"])
        priors = validate_support_priors(priors_raw)
        if priors["source_sha256"] != config["source_sha256"]:
            raise ValueError("Support assumptions refer to a different video")
        progress("reconstruct", "Reusing the complete pinned body evidence; no new pose inference or AI request")
        receipt = copy_body(body_path(parent, parent_config), root / "body_source",
                            expected_manifest=config["parent_body_manifest_sha256"], deadline=started + 30)
        source = root / "body_source" / config["source_file"]
        if digest(source) != config["source_sha256"]:
            raise ValueError("Copied original video differs from the source pin")
        shutil.copyfile(source, root / config["source_file"])
        write_json(root / "body-reuse.json", receipt)
        (root / "analysis").mkdir()
        write_json(root / "analysis/plan.json", {"route": "exercise",
            "summary": "User-authored static support experiment using the original reconstructed body",
            "warnings": ["Support geometry and load windows are user assumptions, not recovered equipment or measured contact.",
                         "This experiment defines a new support task; its score is not an improvement against the parent task."]})
        def stage(row):
            progress("train" if row["stage"].startswith("learning") else "reconstruct",
                     row["stage"].replace("_", " ") + ": " + row["status"].replace("_", " "))
        settings = {"training_seed": 1729, "evaluation_seed": 938401, "generations": 2,
                    "population": 4, "trials": 3, "jitter_scale": .1} if config["train"] else None
        result = prepare_exercise_scene(None, None, None, root / "build", human_run=root / "body_source",
            scene_scope="human_static_supports", static_supports=root / "support-priors.json", full_reference=True,
            train_settings=settings, max_seconds=240 if settings else 120, progress=stage)
        if result.get("status") != "ready" or not result.get("lab_directory"):
            raise ValueError(result.get("error", {}).get("message", "Support construction failed its existing checks"))
        lab = Path(result["lab_directory"]).resolve()
        if not lab.is_relative_to(root):
            raise ValueError("Support lab is outside this experiment")
        atomic_json(root / "status.json", {"status": "ready", "stage": "ready", "events": events,
            "lab": lab.relative_to(root).as_posix(), "route": "exercise", "seconds": time.monotonic() - started,
            "scope": "Source-bound body with user-authored static supports and a new authored support task; physical validation unverified"})
        return 0
    except Exception as error:
        message = str(error)[:1500]
        write_json(root / "failure.json", {"type": type(error).__name__, "message": message, "time": local_time()})
        atomic_json(root / "status.json", {"status": "failed", "stage": "failed", "events": events, "error": message})
        return 1
