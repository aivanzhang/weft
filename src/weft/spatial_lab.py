"""Package a verified 3D rigid scene into a new inspectable local lab without altering inputs."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.spatial_rigid import ACTIONS, PROFILE, UNITS, SpatialRigidEnv, validate_scene


REJECTED_SOURCE_STATUSES = frozenset({"failed", "error", "unsupported", "rejected", "cancelled"})
REQUIRED_POLICY_FILES = frozenset({"scene.json", "policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"})


def authored_spatial_scene():
    """Return a deterministic 3D rigid-body fixture without importing test helpers."""
    base = {
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "friction": [0.3, 0.001, 0.0001],
    }
    scene = {
        "schema_version": 1,
        "profile": PROFILE,
        "units": deepcopy(UNITS),
        "bounds": [1.0, 0.8, 0.8],
        "dt": 1.0 / 120.0,
        "gravity": [0.0, 0.0, -9.81],
        "contact": {"time_constant": 0.01, "damping_ratio": 1.0},
        "objects": [
            base
            | {
                "id": "block",
                "label": "Controlled block",
                "role": "active",
                "shape": "box",
                "position": [0.24, 0.36, 0.14],
                "size": [0.06, 0.05, 0.05],
                "mass": 1.0,
                "color": "#965644",
                "provenance": {"scope": "authored_analytic_fixture", "physical_validation": "unverified"},
            },
            base
            | {
                "id": "companion",
                "label": "Companion sphere",
                "role": "passive",
                "shape": "sphere",
                "position": [0.50, 0.42, 0.075],
                "radius": 0.035,
                "mass": 0.45,
                "color": "#C7795D",
                "provenance": {"scope": "authored_analytic_fixture", "physical_validation": "unverified"},
            },
            base
            | {
                "id": "floor",
                "label": "Support table",
                "role": "static",
                "shape": "box",
                "position": [0.50, 0.40, 0.02],
                "size": [0.96, 0.76, 0.04],
                "mass": 1.0,
                "color": "#D8C9BA",
                "provenance": {"scope": "authored_support_surface", "physical_validation": "unverified"},
            },
        ],
        "controlled_id": "block",
        "target": [0.78, 0.56, 0.065],
        "task": {"position_tolerance": 0.035, "speed_tolerance": 0.06, "acceleration": 8.0, "max_steps": 360},
        "assumptions": [
            "Analytic 3D rigid-body simulation fixture; not reconstructed from a real video.",
            "Virtual translational forces act on the controlled object center of mass; contacts and spin come from MuJoCo rigid-body dynamics.",
            "Physical validation remains unverified.",
        ],
    }
    validate_scene(scene)
    return scene


def write_sealed_spatial_fixture(out):
    """Write a standalone sealed spatial-rigid-v1 run suitable for lab import or sandbox control."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new fixture directory")
    scene = authored_spatial_scene()
    env = SpatialRigidEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
    finally:
        env.close()
    out.mkdir(parents=True)
    report = {
        "created_at": local_time(),
        "profile": PROFILE,
        "status": "authored_analytic_fixture",
        "physical_validation": "unverified",
        "physics_ready": False,
        "scope": "Authored 3D rigid-body fixture for local simulation, control and recording checks; not video reconstruction.",
    }
    write_json(out / "scene.json", scene)
    write_json(out / "report.json", report)
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "created_at": local_time(),
            "files": {name: digest(out / name) for name in ("report.json", "scene.json")},
        },
    )
    return {"directory": str(out), "verification": verify_artifacts(out), "scene_sha256": canonical_hash(scene)}


def _read_sealed_bundle(root, *, label):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 4_000_000:
        raise ValueError(f"{label} requires a bounded sealed manifest.json")
    verified = verify_artifacts(root)
    manifest_bytes = manifest_path.read_bytes()
    if sha256(manifest_bytes).hexdigest() != verified["manifest_sha256"]:
        raise ValueError(f"{label} manifest changed after verification")
    manifest = json.loads(manifest_bytes)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files or len(files) > 4096:
        raise ValueError(f"{label} manifest must list 1–4096 sealed files")
    total_bytes = 0
    for name in files:
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() != name or name == "manifest.json":
            raise ValueError(f"{label} manifest paths must be canonical relative artifact names")
        target = root / name
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"{label} file escapes its sealed directory: {name}")
        size = target.stat().st_size
        if size > 128 * 1024 * 1024:
            raise ValueError(f"{label} file exceeds the 128-MiB per-artifact bound: {name}")
        total_bytes += size
    if total_bytes > 400 * 1024 * 1024:
        raise ValueError(f"{label} exceeds the 400-MiB bundle copy bound")

    def read_json(name, *, limit=32 * 1024 * 1024):
        if name not in files:
            raise ValueError(f"Required {label} file is not sealed: {name}")
        path = root / name
        if path.stat().st_size > limit:
            raise ValueError(f"Sealed {label} JSON exceeds its byte bound: {name}")
        raw = path.read_bytes()
        if len(raw) > limit or sha256(raw).hexdigest() != files[name]:
            raise ValueError(f"Sealed {label} input changed after verification: {name}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"Sealed {label} JSON must be an object: {name}")
        return value

    return verified, manifest_bytes, files, read_json


def _probe_controls(scene):
    env = SpatialRigidEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        initial = env.snapshot()
        controlled_id = scene["controlled_id"]
        initial_body = next(item for item in initial["objects"] if item["id"] == controlled_id)
        trials = []
        for action_index, action_name in enumerate(ACTIONS):
            env.reset(seed=0, options={"jitter": False})
            _, reward, terminated, truncated, _ = env.step(action_index)
            after = env.snapshot()
            body = next(item for item in after["objects"] if item["id"] == controlled_id)
            trials.append(
                {
                    "action": action_index,
                    "name": action_name,
                    "controlled_id": controlled_id,
                    "position": body["position"],
                    "position_delta": [body["position"][axis] - initial_body["position"][axis] for axis in range(3)],
                    "velocity": body["velocity"],
                    "angular_velocity": body["angular_velocity"],
                    "virtual_force": after["virtual_force"],
                    "contact_count": after["contact_count"],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
        # Also verify braking from nonzero motion so coast vs brake is explicit.
        env.reset(seed=0, options={"jitter": False})
        env.step(2)
        moving_before = next(item for item in env.snapshot()["objects"] if item["id"] == controlled_id)
        env.step(7)
        after_brake = next(item for item in env.snapshot()["objects"] if item["id"] == controlled_id)
        env.reset(seed=0, options={"jitter": False})
        env.step(2)
        env.step(0)
        after_coast = next(item for item in env.snapshot()["objects"] if item["id"] == controlled_id)
        braking_check = {
            "pre_step_action": "right",
            "speed_before_mps": float(sum(v * v for v in moving_before["velocity"]) ** 0.5),
            "speed_after_brake_mps": float(sum(v * v for v in after_brake["velocity"]) ** 0.5),
            "speed_after_coast_mps": float(sum(v * v for v in after_coast["velocity"]) ** 0.5),
        }
    finally:
        env.close()
    return {
        "schema_version": 1,
        "profile": PROFILE,
        "controlled_id": controlled_id,
        "seed": 0,
        "trials": trials,
        "moving_brake_check": braking_check,
        "scope": "Simulated single-step control probes from reset(seed=0); verifies MuJoCo actuator response, not real-world physical transfer.",
    }


def _validate_policy_bundle(files, read_json, scene):
    if not REQUIRED_POLICY_FILES <= set(files):
        missing = sorted(REQUIRED_POLICY_FILES - set(files))
        raise ValueError("Spatial policy bundle is missing required sealed artifacts: " + ", ".join(missing))
    policy_scene = read_json("scene.json")
    if canonical_hash(policy_scene) != canonical_hash(scene):
        raise ValueError("The spatial policy bundle and source 3D scene must seal the exact same scene.json")
    policy = read_json("policy.json")
    from weft.spatial_learning import validate_spatial_policy

    validate_spatial_policy(policy, scene)
    for name in ("training.json", "rl-evaluation.json", "policy-rollout.json"):
        read_json(name)
    return policy


def prepare_spatial_lab(source_run, out, *, policy_run=None):
    """Import a verified spatial-rigid-v1 run into a new local lab directory."""
    started = time.monotonic()
    source_run, out = Path(source_run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new spatial lab directory")
    if out == source_run or out.is_relative_to(source_run):
        raise ValueError("Keep the new lab outside the immutable source run")
    if policy_run is not None:
        policy_run = Path(policy_run).resolve()
        if out == policy_run or out.is_relative_to(policy_run):
            raise ValueError("Keep the new lab outside the immutable policy run")

    source_verified, source_manifest_bytes, source_files, read_source_json = _read_sealed_bundle(
        source_run, label="Source 3D run"
    )
    if "scene.json" not in source_files:
        raise ValueError("scene.json must belong to the sealed 3D run")
    for sidecar in ("report.json", "candidate.json"):
        if (source_run / sidecar).exists() and sidecar not in source_files:
            raise ValueError(f"Unsealed {sidecar} present in source run")

    scene = read_source_json("scene.json")
    if scene.get("profile") != PROFILE:
        raise ValueError(f"Spatial lab requires profile {PROFILE!r}, got {scene.get('profile')!r}")
    validate_scene(scene)

    source_report = read_source_json("report.json") if "report.json" in source_files else None
    source_candidate = read_source_json("candidate.json") if "candidate.json" in source_files else None
    if source_report is not None:
        status_value = source_report.get("status")
        if isinstance(status_value, str) and status_value in REJECTED_SOURCE_STATUSES:
            raise ValueError(f"Cannot import failed or rejected 3D run ({status_value})")
        if source_report.get("physical_validation", "unverified") != "unverified" or source_report.get("physics_ready") is True:
            raise ValueError("Spatial lab requires imported 3D scenes to remain explicitly physically unverified")
        if ("candidate_sha256" in source_report or "assumptions_sha256" in source_report) and source_candidate is None:
            raise ValueError("Candidate export is missing its sealed candidate.json")
    if source_candidate is not None:
        if source_candidate.get("physical_validation", "unverified") != "unverified" or source_candidate.get("physics_ready") is True:
            raise ValueError("Candidate 3D export must remain explicitly physically unverified")
        if source_report is not None and "candidate_sha256" in source_report:
            if source_report["candidate_sha256"] != canonical_hash(source_candidate):
                raise ValueError("Candidate and export report hashes differ")
        if "scene" in source_candidate and source_candidate["scene"] != scene:
            raise ValueError("Candidate scene differs from sealed scene.json")

    policy_verified = policy_manifest_bytes = policy_files = None
    learned_policy_available = False
    if policy_run is not None:
        policy_verified, policy_manifest_bytes, policy_files, read_policy_json = _read_sealed_bundle(
            policy_run, label="Spatial policy run"
        )
        _validate_policy_bundle(policy_files, read_policy_json, scene)
        learned_policy_available = True
    elif "policy.json" in source_files:
        _validate_policy_bundle(source_files, read_source_json, scene)
        learned_policy_available = True

    source_meta = read_source_json("source.json") if "source.json" in source_files else None
    source_video_artifact = None
    if source_meta is not None:
        candidate_artifact = source_meta.get("artifact")
        if isinstance(candidate_artifact, str) and Path(candidate_artifact).name == candidate_artifact and candidate_artifact in source_files:
            if source_meta.get("sha256") is not None and source_files[candidate_artifact] != source_meta.get("sha256"):
                raise ValueError("Sealed source video digest does not match source.json")
            source_video_artifact = candidate_artifact

    interventions = _probe_controls(scene)
    implementation = _implementation_files()
    implementation_hashes = {name: digest(path) for name, path in implementation.items()}

    out.mkdir(parents=True)
    run = out / "run"
    run.mkdir()

    shared_optional = {"source.json", "evidence.json", "observations.json", "tracks.gif", "tracks-contact-sheet.png"}
    if source_video_artifact:
        shared_optional.add(source_video_artifact)
    copied_source = {}
    for name, expected in sorted(source_files.items()):
        archive_target = run / "inputs" / "source-run" / name
        archive_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_run / name, archive_target)
        if digest(archive_target) != expected:
            raise ValueError("Source run changed during copying; lab remains unsealed")
        copied_source[name] = f"inputs/source-run/{name}"
        if name in shared_optional:
            top_target = run / name
            shutil.copyfile(archive_target, top_target)
            if digest(top_target) != expected:
                raise ValueError("Shared source evidence changed during copying")
    (run / "source-input-manifest.json").write_bytes(source_manifest_bytes)

    if policy_run is not None:
        for name in ("policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"):
            target = run / name
            shutil.copyfile(policy_run / name, target)
            if digest(target) != policy_files[name]:
                raise ValueError("Policy artifact changed during copying")
        if "settings.json" in policy_files:
            target = run / "settings.json"
            shutil.copyfile(policy_run / "settings.json", target)
            if digest(target) != policy_files["settings.json"]:
                raise ValueError("Policy settings changed during copying")
        (run / "learning-input-manifest.json").write_bytes(policy_manifest_bytes)
    elif learned_policy_available:
        for name in ("policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"):
            shutil.copyfile(source_run / name, run / name)
        if "settings.json" in source_files:
            shutil.copyfile(source_run / "settings.json", run / "settings.json")

    for name, path in implementation.items():
        target = run / "implementation-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if digest(target) != implementation_hashes[name]:
            raise RuntimeError("Implementation changed during preparation; lab remains unsealed")
    if _implementation_files() != implementation:
        raise RuntimeError("Implementation file set changed during preparation; lab remains unsealed")
    write_json(run / "implementation.json", implementation_hashes)

    write_json(run / "scene.json", scene)
    write_json(
        run / "task.json",
        {
            "profile": PROFILE,
            "controlled_id": scene["controlled_id"],
            "target": scene["target"],
            **scene["task"],
            "scope": "Authored 3D rigid-body target parking task; simulation success does not establish physical transfer.",
        },
    )
    write_json(run / "interventions.json", interventions)
    write_json(
        run / "repair-ledger.json",
        {
            "proposals": [],
            "accepted": [],
            "rejected": [],
            "scope": "Imported 3D rigid scene; no automatic physical repair was executed or accepted.",
        },
    )

    reconstruction = {
        "profile": PROFILE,
        "physical_validation": "unverified",
        "physics_ready": False,
        "scene_sha256": canonical_hash(scene),
        "source_status": source_report.get("status", "sealed_spatial_scene") if source_report else "sealed_spatial_scene",
        "source_video_present": source_video_artifact is not None,
        "assumptions": list(scene["assumptions"]),
        "scope": "3D rigid-object hypothesis imported for inspection, bounded control and recording. Depth, shape, materials, controls and task remain assumptions unless independently measured.",
    }
    if source_report is not None:
        for key in (
            "source_sha256",
            "source_initialization_frames",
            "depth_provenance",
            "depth_before",
            "depth_after",
            "factor",
            "lateral_offset",
            "front_parent_scene_sha256",
            "parent_material_scene_sha256",
        ):
            if key in source_report:
                reconstruction[key] = source_report[key]
        if isinstance(source_report.get("camera_proposals"), dict):
            reconstruction["camera_proposals_summary"] = {
                name: {
                    k: proposal[k]
                    for k in ("method", "available", "metric_geometry_verified", "physical_validation", "reason")
                    if isinstance(proposal, dict) and k in proposal
                }
                for name, proposal in source_report["camera_proposals"].items()
            }
    write_json(run / "reconstruction.json", reconstruction)

    provenance = {
        "created_at": local_time(),
        "source_run": str(source_run),
        "source_verification": source_verified,
        "scene_sha256": canonical_hash(scene),
        "copied_source_files": copied_source,
        "source_video_artifact": source_video_artifact,
        "learned_policy_available": learned_policy_available,
        "new_training": False,
        "new_inference": False,
        "paid_requests": 0,
        "scope": "Copied verified 3D scene artifacts into a new local lab. Original input directories remain unchanged.",
    }
    if policy_run is not None:
        provenance["policy_run"] = str(policy_run)
        provenance["policy_verification"] = policy_verified
    write_json(run / "lab-provenance.json", provenance)

    runtime_seconds = round(time.monotonic() - started, 6)
    report = {
        "created_at": local_time(),
        "profile": PROFILE,
        "lab_profile": "spatial-rigid",
        "status": "ready_spatial_lab",
        "source_status": reconstruction["source_status"],
        "physical_validation": "unverified",
        "physics_ready": False,
        "accepted": False,
        "learned_policy_available": learned_policy_available,
        "source_video_present": source_video_artifact is not None,
        "controlled_id": scene["controlled_id"],
        "object_count": len(scene["objects"]),
        "paid_requests": 0,
        "runtime_seconds": runtime_seconds,
        "scope": reconstruction["scope"],
    }
    write_json(run / "report.json", report)
    write_json(
        run / "manifest.json",
        {
            "schema_version": 1,
            "created_at": local_time(),
            "files": {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob("*")) if path.is_file()},
        },
    )

    if verify_artifacts(source_run) != source_verified:
        raise RuntimeError("Immutable source run changed during preparation; no ready lab published")
    if policy_run is not None and verify_artifacts(policy_run) != policy_verified:
        raise RuntimeError("Immutable policy run changed during preparation; no ready lab published")

    lab_config = {
        "schema_version": 1,
        "profile": "spatial-rigid",
        "control_profile": "spatial_rigid",
        "created_at": local_time(),
        "source_run": str(source_run),
        "learned_policy_available": learned_policy_available,
        "source_video_present": source_video_artifact is not None,
        "physical_validation": "unverified",
        "episodes": 0,
        "eval_episodes": 0,
        "bounds": {"wall_seconds": 360, "external_spend_usd": 0, "recurring_services": 0},
    }
    if policy_run is not None:
        lab_config["learning_source"] = str(policy_run)
    write_json(out / "lab-config.json", lab_config)

    events = [
        {"index": 0, "stage": "source", "time": local_time(), "elapsed_seconds": 0.0, "message": "Verified sealed 3D scene run; no new perception or API call."},
        {"index": 1, "stage": "world", "time": local_time(), "elapsed_seconds": runtime_seconds, "message": f"Validated {PROFILE} scene ({len(scene['objects'])} objects, controlled: {scene['controlled_id']}) and 8-action MuJoCo probes."},
        {
            "index": 2,
            "stage": "ready",
            "time": local_time(),
            "elapsed_seconds": runtime_seconds,
            "message": (
                "3D rigid scene ready with verified learned policy; physical validation remains unverified."
                if learned_policy_available
                else "3D rigid scene ready for manual inspection, control and recording; no learned policy trained."
            ),
        },
    ]
    write_json(out / "status.json", {"status": "ready", "events": events, "result": report})
    return {
        "directory": str(out),
        "verification": verify_artifacts(run),
        "scene_sha256": canonical_hash(scene),
        "learned_policy_available": learned_policy_available,
        "physical_validation": "unverified",
        "runtime_seconds": runtime_seconds,
        "inspect": f".venv-lab/bin/python -m weft lab --resume {out}",
    }
