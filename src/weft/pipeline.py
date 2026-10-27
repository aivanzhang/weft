import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

from weft.config import RunConfig
from weft.env import Task
from weft.evaluate import audit_episode, evaluator_identity, physics_checks, replay_metrics
from weft.io import canonical_hash, digest, local_time, write_json
from weft.learning import changed_action_trials, evaluate_policies, train
from weft.meta import propose_meta, status
from weft.render import replay_video, write_viewer
from weft.repair import repair
from weft.source import record_source
from weft.vision import TrackingConfig, analyze
from weft.world import generate


def _implementation_files():
    package = Path(__file__).parent
    files = {str(path.relative_to(package)): path for path in package.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts}
    for name in ("pyproject.toml", "requirements.lock"):
        path = package.parent.parent / name
        if path.is_file():
            files["project/" + name] = path
    return files


def verify_artifacts(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Artifact integrity requires a nonempty file manifest")
    errors = []
    for name, expected in manifest["files"].items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ValueError("Artifact integrity requires string file names and hashes")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            errors.append(name)
    if errors:
        raise ValueError("Artifact integrity failed: " + ", ".join(errors))
    return {"verified_files": len(manifest["files"]), "manifest_sha256": digest(root / "manifest.json")}


def run(video, out, episodes=15000, eval_episodes=300, seed=1729, provider="local", meta_budget=None,
        *, eval_seed=938401, repair_enabled=True, tracking=TrackingConfig(), experiment=None):
    config = RunConfig(episodes, eval_episodes, seed, eval_seed, provider, repair_enabled, meta_budget, tracking).validate()
    access = status()
    if provider == "meta" and not access["credential_present"]:
        raise ValueError("Meta mode requires META_API_KEY or MODEL_API_KEY; no fallback executed")
    video, out = Path(video).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Output directory already exists; use a new run directory")
    if not video.is_file():
        raise ValueError("Input video does not exist")
    out.mkdir(parents=True)
    started = time.perf_counter()
    evaluator = evaluator_identity()
    implementation_files = _implementation_files()
    implementation = {name: digest(path) for name, path in implementation_files.items()}
    timings = {}
    try:
        write_json(out / "run-config.json", {"schema_version": 1, "run": config.to_dict(), "experiment": experiment})
        source = record_source(video, out)
        shutil.copyfile(video, out / "source.mp4")
        stage = time.perf_counter()
        evidence = analyze(video, out, tracking)
        initial, graph = generate(evidence)
        timings["analysis_seconds"] = time.perf_counter() - stage
        write_json(out / "graph.json", graph)
        write_json(out / "world-initial.json", initial.to_dict())
        write_json(out / "generation-payload.json", {"world": initial.to_dict(), "observations": evidence["generation"]})
        extra = []
        if provider == "meta":
            extra.append(propose_meta(video, evidence["generation"], initial, out, meta_budget))
        stage = time.perf_counter()
        selected, repairs = (repair(initial, evidence["generation"], evidence["development_validation"], extra)
                             if repair_enabled else (initial, []))
        timings["repair_seconds"] = time.perf_counter() - stage
        write_json(out / "repair-ledger.json", repairs)
        write_json(out / "world.json", selected.to_dict())
        write_json(out / "task.json", Task().to_dict())
        origin = evidence["generation"][0]["time_s"]
        before = replay_metrics(initial, evidence["development_validation"], origin)
        after = replay_metrics(selected, evidence["development_validation"], origin)
        physical = physics_checks(selected)
        if not physical["passes"] or not after["passes"]:
            raise RuntimeError("Selected world did not pass required independent gates")
        interventions = changed_action_trials(selected)
        write_json(out / "interventions.json", interventions)
        q, training = train(selected, out, episodes=episodes, seed=seed)
        stage = time.perf_counter()
        evaluation, examples = evaluate_policies(selected, q, count=eval_episodes, seed=eval_seed)
        timings["rl_evaluation_seconds"] = time.perf_counter() - stage
        write_json(out / "rl-evaluation.json", evaluation)
        write_json(out / "rl-example-traces.json", examples)
        # The audit runs after selection and learning; its result cannot change the selected world.
        audit = [{"before": audit_episode(initial, rows), "after": audit_episode(selected, rows)}
                 for rows in evidence["audit_episodes"]]
        write_json(out / "audit.json", {"episodes": audit, "tuned_on_audit": False,
                                       "scope": "Withheld from automatic fitter; same video and development-inspected tracking profile"})
        stage = time.perf_counter()
        replay = replay_video(video, evidence, initial, selected, out)
        timings["render_seconds"] = time.perf_counter() - stage
        if evaluator_identity() != evaluator:
            raise RuntimeError("Evaluator changed during the run")
        costs = {"external_compute_usd": 0.0, "inference_usd": 0.0, "recurring_usd": 0.0}
        if provider == "meta":
            costs["inference_usd"] = json.loads((out / "meta-call.json").read_text())["cost_usd"]
            access = access | {"live_access_verified": True, "note": "One recorded Meta request completed"}
        report = {
            "created_at": local_time(), "source": source,
            "run_config": config.to_dict(),
            "provider": "Local computer vision + numerical fitting; Meta not used" if provider == "local" else "Local CV + numerical fitting + one Meta proposal",
            "meta_access": access, "evaluator": evaluator,
            "world_hash": canonical_hash(selected.to_dict()),
            "task_hash": canonical_hash(Task().to_dict()),
            "reconstruction": {"before": before, "after": after}, "physics": physical,
            "repairs": repairs, "audit": audit, "replay": replay,
            "training": training,
            "rl": {"scope": evaluation["scope"], "seed": evaluation["seed"],
                   "policies": {k: {name: value for name, value in v.items() if name != "trials"}
                                for k, v in evaluation["policies"].items()}},
            "timings": timings | {"training_seconds": training["seconds"], "total_seconds": time.perf_counter() - started},
            "cost": costs,
            "limits": evidence["limits"] + graph["assumptions"] + [
                "No external real-world intervention trials or demonstrated sim-to-real policy transfer",
                "No arbitrary-video semantics, 3D reconstruction, object extraction, or learned general physics",
                "No live Meta verification in local mode; not a submission-ready Meta integration",
            ],
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "Pillow", "gymnasium")}},
        }
        write_json(out / "report.json", report)
        write_viewer(out, report, selected, graph, interventions)
        # Hashes identify a version; source copies let later iterations reproduce it without a commit.
        if _implementation_files() != implementation_files:
            raise RuntimeError("Implementation files changed during the run")
        for name, path in implementation_files.items():
            destination = out / "implementation-source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if digest(destination) != implementation[name]:
                raise RuntimeError("Implementation changed during the run")
        write_json(out / "implementation.json", implementation)
        write_json(out / "manifest.json", {"schema": 1, "created_at": local_time(),
                   "files": {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file()}})
        verification = verify_artifacts(out)
        return {"output": str(out), "initial_error_px": before["rmse_px"], "selected_error_px": after["rmse_px"],
                "accepted_repairs": sum(r["accepted"] for r in repairs),
                "rl_success_rate": evaluation["policies"]["q_learning"]["success_rate"],
                "random_success_rate": evaluation["policies"]["random"]["success_rate"],
                "audit_episodes": len(audit), "seconds": report["timings"]["total_seconds"],
                "verification": verification, "provider": report["provider"]}
    except Exception as error:
        write_json(out / "failure.json", {"error_type": type(error).__name__, "message": str(error),
                                         "elapsed_seconds": time.perf_counter() - started, "time": local_time()})
        raise
