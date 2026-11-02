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

TERMINAL = {"proposed", "failed", "cancelled", "interrupted"}


def availability(project):
    status = access_status(project)
    return {**status, "model": MODEL, "estimated_one_call_usd": ESTIMATE,
            "available": bool(status["credential_present"] and status["budget_configured"]
                              and status.get("remaining_usd", 0) >= ESTIMATE)}


def construction_input(root):
    root = Path(root).resolve()
    config = json.loads((root / "job.json").read_bytes())
    status = json.loads((root / "status.json").read_bytes())
    if status.get("status") != "ready":
        raise ValueError("Finish reconstruction before requesting a Muse review")
    route = status.get("route")
    if route not in {"exercise", "objects"}:
        raise ValueError("This project's construction format is not supported by Muse review")
    source = body_path(root, config) if route == "exercise" else root / "planar/run"
    names = ["evidence.json", config["source_file"]] + (["human.json"] if route == "exercise" else [])
    manifest_path = source / "manifest.json"
    if any(p.is_symlink() for p in (manifest_path, *manifest_path.parents)) or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("Unsafe construction manifest")
    manifest = json.loads(manifest_path.read_bytes())
    files = manifest["files"]
    pins = {}
    for name in names:
        path = source / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file()
                or path.stat().st_size > (100_000_000 if name == config["source_file"] else 32_000_000)):
            raise ValueError("Unsafe construction evidence")
        pins[name] = digest(path)
        if files.get(name) != pins[name]:
            raise ValueError("Construction evidence no longer matches its saved manifest")
    if pins[config["source_file"]] != config["source_sha256"]:
        raise ValueError("Construction video differs from the uploaded source")
    binding = {"construction_directory": source.relative_to(root).as_posix(),
               "manifest_sha256": digest(manifest_path), "input_sha256": pins,
               "scope": "Construction observations and authored body assumptions only; no scores, policies or final evaluation"}
    if config.get("kind") == "support_experiment":
        binding.update(review_scope="original_body_only",
                       scope="Original body construction evidence only. Added supports, load windows and child-world control settings are not reviewed; no scores, policies or final evaluation.")
    return source, binding


def review_input(root):
    source, binding = construction_input(root)
    world, world_binding = construction_context(root)
    if world is not None:
        binding.update(review_scope="coupled_world_assumptions", simulation_world=world_binding,
                       scope="Source observations plus current coupled-world construction assumptions; authored supports are not recovered evidence. No scores, policies or final evaluation.")
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
        source, binding, world = review_input(root)
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
                             **({"scene_context": world} if coupled else {}))
        _, after, _ = review_input(root)
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
