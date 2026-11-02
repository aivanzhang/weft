"""Fixed local experiments related to Muse proposals; never model-generated tests."""

from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np

from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab_worker import atomic_json
from weft.portal_review import construction_input
from weft.scene_advisor import validate_advice

TERMINAL = {"complete", "failed", "cancelled", "interrupted"}
MAX_TRACE_BYTES = 32 * 1024**2


def read_json(path, maximum=4 * 1024**2):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Check input is missing, linked or too large")
    raw = path.read_bytes()
    if len(raw) > maximum:
        raise ValueError("Check input grew beyond its bound")
    return json.loads(raw), sha256(raw).hexdigest()


def inputs(root):
    root = Path(root).resolve()
    _, binding = construction_input(root)
    stored, _ = read_json(root / "muse/binding.json")
    if binding != stored:
        raise ValueError("Construction changed since Muse review")
    proposal, proposal_sha = read_json(root / "muse/advice/hypotheses.json", 140_000)
    context, context_sha = read_json(root / "muse/advice/context.json", 32_000)
    response, response_sha = read_json(root / "muse/advice/response.json", 140_000)
    if proposal.get("accepted") is not False or proposal.get("source") != "model_proposal":
        raise ValueError("Checks require an unaccepted model proposal")
    advice = validate_advice(proposal["advice"], [r["id"] for r in context["observations"]], [r["id"] for r in context["objects"]])
    if advice != json.loads(response["choices"][0]["message"]["content"]):
        raise ValueError("Saved advice differs from the provider response")
    status, _ = read_json(root / "status.json")
    config, _ = read_json(root / "job.json")
    if context["source_sha256"] != config["source_sha256"]:
        raise ValueError("Advice refers to a different video")
    selected = root / status["lab"] / "run"
    if any(p.is_symlink() for p in (selected, *selected.parents)):
        raise ValueError("Simulation paths cannot contain links")
    run = selected.resolve()
    if not run.is_relative_to(root):
        raise ValueError("Simulation is outside this project")
    manifest, manifest_sha = read_json(run / "manifest.json")
    names = ["scene.json"] + (["exercise-task.json"] if status["route"] == "exercise" else ["source.json"])
    values, pins = {}, {}
    for name in names:
        values[name], pins[name] = read_json(run / name)
        if manifest["files"].get(name) != pins[name]:
            raise ValueError("Simulation or task differs from its immutable manifest")
    scene = values["scene.json"]
    if status["route"] == "exercise" and scene["source_binding"]["source_sha256"] != config["source_sha256"]:
        raise ValueError("Simulation source does not match the reviewed video")
    if status["route"] == "objects" and values["source.json"]["sha256"] != config["source_sha256"]:
        raise ValueError("Rigid simulation source does not match the reviewed video")
    return scene, values.get("exercise-task.json"), advice, {
        "source_sha256": config["source_sha256"], "run_manifest_sha256": manifest_sha,
        "run": run.relative_to(root).as_posix(), "run_inputs": pins,
        "proposal_sha256": proposal_sha, "context_sha256": context_sha, "response_sha256": response_sha,
        "construction": binding}


def experiments(scene, task, output, *, deadline):
    output = Path(output)
    exercise = task is not None
    if exercise:
        from weft.shared_exercise_task import SharedExerciseTaskEnv
        factory = lambda: SharedExerciseTaskEnv(scene, task)
        conditions = ("authored", "root_unforced", "coast") if task["control"]["root_assistance"] == "explicit_3_axis" else ("authored", "coast")
    else:
        from weft.spatial_rigid import SpatialRigidEnv
        factory = lambda: SpatialRigidEnv(scene)
        conditions = ("coast", "positive_x", "negative_x")
    results, initial_identity, total_bytes = [], None, 0
    for condition in conditions:
        env = factory()
        try:
            env.reset(seed=1729, options={"jitter": False})
            initial = env.snapshot()
            initial_sha = canonical_hash(initial)
            if initial_identity is not None and initial_identity != initial_sha:
                raise ValueError("Paired experiments must use identical initial states")
            initial_identity = initial_sha
            trace = output / (condition + ".jsonl")
            count, reward_sum, terminated, truncated = 0, 0., False, False
            horizon = env.horizon_steps if exercise else scene["task"]["max_steps"]
            if not 1 <= horizon <= 900:
                raise ValueError("Experiment horizon exceeds 900 control steps")
            with trace.open("x") as stream:
                def save(row):
                    nonlocal total_bytes
                    raw = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
                    total_bytes += len(raw.encode())
                    if total_bytes > MAX_TRACE_BYTES:
                        raise ValueError("Experiment traces exceed 32 MiB")
                    stream.write(raw)
                save({"kind": "reset", "seed": 1729, "state": initial})
                for step in range(horizon):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Local checks exceeded their time allowance")
                    if (output / "cancel-request.json").exists():
                        raise InterruptedError("Local checks cancelled; partial traces preserved")
                    if exercise:
                        action = env.authored_action() if condition != "coast" else np.zeros(env.action_count)
                        if condition == "root_unforced":
                            for index, name in enumerate(env.action_names):
                                if name in {"root_x", "root_y", "root_z"}:
                                    action[index] = 0
                    else:
                        action = {"coast": 0, "positive_x": 2, "negative_x": 1}[condition]
                    _, reward, terminated, truncated, info = env.step(action)
                    state = env.snapshot()
                    save({"kind": "step", "step": step + 1, "action": action.tolist() if exercise else action,
                          "reward": float(reward), "state": state})
                    count += 1
                    reward_sum += float(reward)
                    if terminated or truncated:
                        break
            results.append({"condition": condition, "steps": count, "horizon_steps": horizon,
                "terminated": bool(terminated), "truncated": bool(truncated), "reward_sum": reward_sum,
                "success": bool(info.get("success", False)), "metrics": info,
                "initial_state_sha256": initial_sha, "final_state_sha256": canonical_hash(state),
                "trace": trace.name, "trace_sha256": digest(trace)})
        finally:
            env.close()
    return results


def execute_checks(root):
    root = Path(root)
    output = root / "muse/checks"
    started = time.monotonic()
    try:
        saved, _ = read_json(output / "status.json")
        if saved.get("status") in TERMINAL:
            return 0 if saved["status"] == "complete" else 1
        scene, task, advice, binding = inputs(root)
        write_json(output / "inputs.json", binding)
        atomic_json(output / "status.json", {"status": "running", "message": "Comparing fixed control interventions under the original task"})
        results = experiments(scene, task, output, deadline=started + 45)
        _, _, _, after = inputs(root)
        if after != binding:
            raise ValueError("Inputs changed during local experiments")
        report = {"schema_version": 1, "profile": "portal-local-checks-v1", "created_at": local_time(),
            "protocol": "exercise-control-ablation-v1" if task else "rigid-force-intervention-v1",
            "evidence_partition": "exposed_development_simulation",
            "implementation_sha256": digest(Path(__file__)),
            "runtime": {"python": platform.python_version(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "gymnasium")},
                        "physics_files": {name: digest(Path(__file__).with_name(name)) for name in
                                          ("shared_world.py", "shared_exercise_task.py", "spatial_rigid.py")}},
            "seconds": time.monotonic() - started, "external_api_usd": 0, "accepted": False, "applied": False,
            "scope": "Related simulated intervention diagnostics, not execution of Muse's proposed physical measurements or independent physical validation",
            "hypotheses": [{"id": row["id"], "status": "unresolved", "required_tests": row["required_tests"],
                            "reason": "These requested tests have not been executed; local control probes cannot validate the full physical claim"}
                           for row in advice["hypotheses"]],
            "input_binding": binding, "trials": results}
        write_json(output / "report.json", report)
        write_json(output / "manifest.json", {"files": {p.name: digest(p) for p in output.iterdir()
            if p.name in {"report.json", "inputs.json"} or p.suffix == ".jsonl"}})
        atomic_json(output / "status.json", {"status": "complete", "message": "Local comparisons are ready. Muse's physical hypotheses remain unresolved.", "seconds": report["seconds"]})
        return 0
    except Exception as error:
        atomic_json(output / "status.json", {"status": "cancelled" if isinstance(error, InterruptedError) else "failed",
            "message": str(error)[:500], "error_type": type(error).__name__, "seconds": time.monotonic() - started})
        return 1
