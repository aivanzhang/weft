"""Exercise source-linked proxy construction, live controls and replay without API calls."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from weft.io import digest, local_time, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.recording import verify_recording
from weft.shared_world import SharedWorldEnv, render_snapshot
from weft.shared_world_lab import prepare_shared_world_lab
from weft.shared_world_pipeline import construct_source_equipment_run


def baselines(scene, out):
    results = []
    for name in ("coast", "authored_reference"):
        env = SharedWorldEnv(scene)
        frames = []
        try:
            env.reset(seed=1729)
            for _ in range(min(60, scene["task"]["horizon_steps"])):
                action = env.authored_action() if name == "authored_reference" else np.zeros(env.action_space.shape)
                assert np.array_equal(action[-3 * len(scene["objects"]):], np.zeros(3 * len(scene["objects"])))
                _, reward, terminated, truncated, info = env.step(action)
                frames.append({"action": action.tolist(), "reward": reward, "state": env.snapshot(),
                               "terminated": terminated, "truncated": truncated, "info": info})
                if terminated or truncated:
                    break
            write_json(out / f"{name}-rollout.json", {"seed": 1729, "frames": frames})
            results.append({"controller": name, "steps": len(frames), "final_info": frames[-1]["info"],
                            "scope": "Authored task baseline, not learned policy or physical exercise success."})
        finally:
            env.close()
    return results


def inspect(candidate, destination):
    preparation = prepare_shared_world_lab(candidate, destination)
    lab = Lab(destination)
    try:
        session = lab.load_session()
        scene = session.scene
        timings = []
        for _ in range(30):
            session.reset(1729)
            started = time.perf_counter()
            result = lab.control("/api/step", {"controller": "reference", "steps": 3})
            json.dumps(result, allow_nan=False)
            timings.append((time.perf_counter() - started) * 1000)
        recordings = []
        for actor in ("human", *(obj["id"] for obj in scene["objects"])):
            lab.control("/api/actor", {"actor": actor})
            lab.control("/api/reset", {})
            lab.control("/api/record/start", {})
            action = {"controller": "reference"} if actor == "human" else {"action": 4}
            for count in (6, 3):
                lab.control("/api/step", {**action, "steps": count})
            lab.control("/api/reset", {})
            lab.control("/api/step", {**action, "steps": 3})
            stopped = lab.control("/api/record/stop", {})
            lab.export_future.result(timeout=30)
            recording = destination / "sessions" / stopped["directory"]
            recordings.append({"actor": actor, "directory": str(recording),
                               "verification": verify_recording(recording)})
        Image.fromarray(render_snapshot(session.snapshot(), session.scene)).save(destination.parent / "interactive-world.png")
        return {"preparation": preparation, "recordings": recordings,
                "control_dispatch_plus_json": {"samples": len(timings), "p50_ms": float(np.percentile(timings, 50)),
                    "p95_ms": float(np.percentile(timings, 95)), "maximum_ms": max(timings),
                    "scope": "Local three-step batches, not browser/device or sustained resource verification."}}
    finally:
        lab.close()


def main():
    parser = argparse.ArgumentParser()
    for name in ("base-run", "proposal-run", "evidence-run", "source-mapping", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--target-object", required=True)
    parser.add_argument("--frame-translation", type=float, nargs=2, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = {name: str(getattr(args, name)) for name in ("base_run", "proposal_run", "evidence_run", "source_mapping")}
    identities = {name: digest(Path(path) / "manifest.json") for name, path in inputs.items()}
    write_json(args.out / "pre-run.json", {"created_at": local_time(), "inputs": inputs, "input_manifests": identities,
        "hypothesis": "A declared coordinate migration admits source-linked proxy geometry; one simulator supports all actor controls and exact recordings.",
        "frame_translation_xy": args.frame_translation, "task_target": args.target_object,
        "bounds": {"construction_candidates": 2, "recordings": 3, "baseline_rollouts": 2,
                   "baseline_max_steps": 60, "timed_three_step_batches": 30, "export_timeout_seconds": 30,
                   "api_calls": 0, "learning_runs": 0, "external_cost_usd": 0},
        "scope": "Initial geometry plus software interaction; no recovered grip, hand support, full exercise or physical validation."})
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    result = {"status": "running", "physical_validation": "unverified"}
    try:
        kwargs = {"target_object_id": args.target_object, "source_mappings": [args.source_mapping]}
        try:
            construct_source_equipment_run(args.base_run, args.proposal_run, args.evidence_run,
                                           args.out / "original-frame-candidate", **kwargs)
        except ValueError as error:
            result["original_frame_rejection"] = {"type": type(error).__name__, "reason": str(error)}
            write_json(args.out / "original-frame-rejection.json", result["original_frame_rejection"])
            if "exceeds shared-world bounds" not in str(error):
                raise
        else:
            raise AssertionError("Expected the preserved original-frame placement to exceed its old numerical bounds")
        candidate = args.out / "candidate"
        result["construction"] = construct_source_equipment_run(args.base_run, args.proposal_run, args.evidence_run,
            candidate, frame_translation=args.frame_translation, **kwargs)
        scene = json.loads((candidate / "scene.json").read_text())
        result["baseline_rollouts"] = baselines(scene, args.out)
        result["lab"] = inspect(candidate, args.out / "lab")
        result["status"] = "verified_source_proxy_runtime_not_physical_reconstruction"
    except Exception as error:
        result.update(status="failed_preserved", error_type=type(error).__name__, reason=str(error))
    finally:
        result["input_manifest_identities_unchanged"] = all(digest(Path(path) / "manifest.json") == identities[name] for name, path in inputs.items())
        if not result["input_manifest_identities_unchanged"]:
            result.update(status="failed_preserved", reason="An input manifest changed during the study")
        write_json(args.out / "results.json", result)
        # New sessions and lifecycle updates stay outside the immutable study seal.
        sealed = {}
        for path in sorted(args.out.rglob("*")):
            if not path.is_file() or path == args.out / "manifest.json":
                continue
            rel = path.relative_to(args.out)
            if rel.parts[0] == "lab" and (len(rel.parts) < 3 or rel.parts[1] not in {"run", "sessions"}):
                continue
            sealed[rel.as_posix()] = digest(path)
        write_json(args.out / "manifest.json", {"created_at": local_time(), "files": sealed})
    print(json.dumps({"status": result["status"], "reason": result.get("reason"),
                      "verification": verify_artifacts(args.out),
                      "input_manifests_unchanged": result["input_manifest_identities_unchanged"]}))


if __name__ == "__main__":
    main()
