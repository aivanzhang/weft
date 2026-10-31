"""Bounded, local-only pipeline process for the interactive lab."""

import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time

from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, _runtime_packages, run, verify_artifacts


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    os.replace(temporary, path)


def execute(root):
    root = Path(root).resolve()
    settings = json.loads((root / "lab-config.json").read_text())
    out = root / "run"
    started = time.monotonic()
    events = []

    def progress(stage, detail=None):
        detail = detail if isinstance(detail, dict) else {"message": str(detail or stage)}
        if not detail.get("message"):
            detail = dict(detail, message=(f"Episode {detail['episode']}/{detail.get('requested_episodes', settings['episodes'])}"
                         if "episode" in detail else f"{detail['policy']}: {detail.get('successes', '?')}/{detail.get('episodes', '?')} successes"
                         if "policy" in detail else json.dumps(detail, allow_nan=False)))
        event = {"index": len(events), "stage": stage, "time": local_time(),
                 "elapsed_seconds": round(time.monotonic() - started, 3), **detail}
        events.append(event)
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        atomic_json(root / "status.json", {"status": "running", "stage": stage,
                                          "events": events[-300:]})
        print(f"[{stage}] {detail.get('message', '')}", flush=True)

    try:
        training_enabled = settings.get("training_enabled", True)
        if type(training_enabled) is not bool or (not training_enabled and settings["profile"] not in ("planar", "appearance")):
            raise ValueError("Training opt-out requires a planar or appearance video lab")
        if not training_enabled and (settings.get("episodes") is not None or settings.get("eval_episodes") is not None):
            raise ValueError("A manual-only lab cannot request training or evaluation episodes")
        if settings["profile"] == "rolling":
            result = run(settings["video"], out, episodes=settings["episodes"],
                         eval_episodes=settings["eval_episodes"], seed=settings["seed"],
                         progress=progress)
        elif settings["profile"] == "exercise":
            result = exercise_pipeline(out, settings, progress)
        else:
            from weft.scene import SceneEnv, validate_scene
            from weft.scene_vision import fit_scene, inspect_video

            out.mkdir()
            implementation = _implementation_files()
            hashes = {name: digest(path) for name, path in implementation.items()}
            progress("source", {"message": "Copying and hashing the local video"})
            source_video = out / ("source" + Path(settings["video"]).suffix.lower())
            shutil.copyfile(settings["video"], source_video)
            write_json(out / "source.json", {"sha256": digest(source_video),
                       "original_name": Path(settings["video"]).name,
                       "rights": settings["source_note"], "scope": "User-provided local input"})
            write_json(out / "run-config.json", settings)
            if settings["profile"] == "appearance":
                from weft.generic_vision import inspect_generic
                evidence, scene = inspect_generic(source_video, out, settings["scene_config"], progress=progress)
            else:
                evidence, scene = inspect_video(source_video, out,
                                               config=settings.get("scene_config"), progress=progress)
            progress("repair", {"message": "Comparing bounded dynamics hypotheses against development tracks"})
            scene = fit_scene(evidence, scene, out, progress=progress)
            reconstruction = json.loads((out / "reconstruction.json").read_text())
            if not reconstruction["replay_gate"]["passes"]:
                raise RuntimeError("Generated scene failed the fixed per-object reconstruction gate; inspect preserved tracks and hypotheses")
            scene = validate_scene(scene)
            write_json(out / "scene.json", scene)
            write_json(out / "task.json", {"controlled_id": scene["controlled_id"],
                       "target": scene["target"], **scene["task"],
                       "provenance": "Authored movement controls and parking objective; not recovered intent"})
            progress("evaluate", {"message": "Checking executable state and changed actions"})
            interventions = {}
            for action in range(6):
                env = SceneEnv(scene)
                env.reset(seed=0, options={"jitter": False})
                states = [env.snapshot()]
                for _ in range(30):
                    _, _, terminated, truncated, _ = env.step(action)
                    states.append(env.snapshot())
                    if terminated or truncated:
                        break
                interventions[str(action)] = states
                env.close()
            write_json(out / "interventions.json", {"scope": "Simulated actions, not physical trials",
                                                     "trials": interventions})
            if training_enabled:
                from weft.scene_learning import train_scene
                progress("train", {"message": f"Training and evaluating on one fixed world ({settings['episodes']} episodes)"})
                learning = train_scene(scene, out, episodes=settings["episodes"],
                                       eval_episodes=settings["eval_episodes"], seed=settings["seed"], progress=progress)
            else:
                learning = {"status": "not_requested", "learned_policy_available": False,
                            "evaluation": None,
                            "scope": "Manual controls only; no policy training or policy evaluation was requested"}
                progress("learning_skipped", {"message": "Learning not requested. Manual controls and recording are available after the reconstruction checks."})
            progress("render", {"message": ("Preserving the world, evidence, policy, and source snapshots" if training_enabled else
                                               "Preserving the world, evidence, and source snapshots; no policy was created")})
            result = {"profile": settings["profile"], "learning": learning,
                      "objects": len(scene["objects"]), "assumptions": scene["assumptions"],
                      "source_sha256": digest(source_video),
                      "runtime_packages": _runtime_packages(),
                      "cost": {"external_api_usd": 0, "cloud_usd": 0},
                      "seconds": time.monotonic() - started,
                      "limits": [("Explicit source regions tracked by appearance; authored 2D shapes and physics"
                                  if settings["profile"] == "appearance" else
                                  "Fixed-camera colored rigid objects in a 2D image plane"),
                                 "Same-video development evidence, not an independent final holdout",
                                 "No metric 3D recovery, inferred intent, or demonstrated real-world transfer"]}
            write_json(out / "report.json", result)
            for name, path in implementation.items():
                target = out / "implementation-source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                if digest(target) != hashes[name]:
                    raise RuntimeError("Implementation changed during the run; use a new output after edits finish")
            write_json(out / "implementation.json", hashes)
            write_json(out / "manifest.json", {"schema": 1, "created_at": local_time(),
                       "files": {str(path.relative_to(out)): digest(path)
                                 for path in sorted(out.rglob("*")) if path.is_file()}})
            result["verification"] = verify_artifacts(out)
        progress("ready", {"message": "Environment ready. Inspect the artifacts, then play and record."})
        atomic_json(root / "status.json", {"status": "ready", "stage": "ready", "events": events[-300:],
                                          "result": result, "finished_at": local_time()})
        return 0
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error), "time": local_time()}
        write_json(root / "failure.json", failure)
        progress("failed", {"message": str(error)})
        atomic_json(root / "status.json", {"status": "failed", "events": events[-300:], "error": failure})
        return 1


def exercise_pipeline(out, settings, progress):
    from weft.exercise_source import describe_source
    from weft.exercise_vision import inspect_exercise
    from weft.human import HumanEnv, build_human, evaluate_reference
    from weft.human_learning import train_human
    from weft.human_task import propose_joint_task
    from weft.recording import render_snapshot
    from PIL import Image

    started = time.monotonic()
    out.mkdir()
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    progress("source", {"message": "Preserving the exercise video and local pose model identity"})
    source_video = out / ("source" + Path(settings["video"]).suffix.lower())
    shutil.copyfile(settings["video"], source_video)
    source_hash = digest(source_video)
    source = describe_source(source_hash) or {"original_name": Path(settings["video"]).name,
                                             "rights": settings["source_note"]}
    write_json(out / "source.json", {**source, "sha256": source_hash, "artifact": source_video.name,
                                     "derived_overlays": "Pose annotations are Weft-generated adaptations; retain source attribution/license"})
    write_json(out / "run-config.json", settings)
    progress("observe", {"message": "Estimating 33 body landmarks locally; retaining missing and uncertain joints"})
    evidence = inspect_exercise(source_video, out, settings["pose_model"], progress=progress,
                                pose_mode=settings.get("pose_mode", "video"),
                                pose_identity=settings.get("pose_identity", "raw"))
    write_json(out / "pose-quality.json", evidence["quality"])
    write_json(out / "model.json", evidence["model"])
    progress("world", {"message": "Fitting an articulated body with explicit joint and support assumptions"})
    spec = build_human(evidence, out, body_model=settings.get("body_model", "hinge"))
    task_proposal = propose_joint_task(evidence, spec)
    write_json(out / "task-selection.json", task_proposal)
    env = HumanEnv(spec)
    env.select_joint(task_proposal["selected_joint"])
    env.reset(seed=0)
    write_json(out / "scene.json", env.scene)
    write_json(out / "task.json", {"scope": "Authored single-joint target control with virtual support; not whole-body balance",
                                   **env.scene["task"],
                                   "selected_joint": env.selected_joint,
                                   "assumptions": spec["assumptions"]})
    progress("repair", {"message": "Recording model constraints; no claim of recovering true muscle parameters"})
    write_json(out / "repair-ledger.json", {"proposals": [], "scope":
               "Body proportions are fitted to pose estimates and joint angles bounded. Automated physical human-model repair is not implemented."})
    progress("evaluate", {"message": "Checking executable joint controls and reference-pose projection"})
    reference_indices = list(range(0, len(spec["reference"]), max(1, math.ceil(len(spec["reference"]) / 120))))
    references = [env.reference_snapshot(i) for i in reference_indices]
    write_json(out / "human-reference.json", {"scope": "Kinematic pose projection, not a dynamic rollout or measured muscle activation",
                                             "source_indices": reference_indices, "frames": references})
    images = [Image.fromarray(render_snapshot(snapshot, env.scene, width=480))
              for snapshot in references[::max(1, len(references) // 40)][:48]]
    images[0].save(out / "reconstruction.gif", save_all=True, append_images=images[1:], duration=150, loop=0)
    interventions = {}
    for action in range(6):
        env.reset(seed=0)
        states = [env.snapshot()]
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(action)
            states.append(env.snapshot())
            if terminated or truncated:
                break
        interventions[str(action)] = states
    write_json(out / "interventions.json", {"scope": "Authored joint motors and stabilization, not measured human muscle forces",
                                             "trials": interventions})
    if env.mobility:
        mobile_trials = {}
        for action in (6, 7, 8, 9):
            env.reset(seed=0, options={"jitter": False})
            env.advance_physics(6)
            states = [env.snapshot()]
            for _ in range(30):
                states.append(env.advance_physics(action))
            mobile_trials[str(action)] = states
        write_json(out / "mobile-interventions.json", {"scope": "Simulated release/push/support under authored gravity, masses and floor; no physical intervention evidence",
                                                       "trials": mobile_trials})
    env.close()
    reference_evaluation = evaluate_reference(spec, out)
    write_json(out / "reconstruction.json", {"reference_frames": len(references),
               "scope": f"Simplified {len(env.joint_names)}-rotation-axis body; " + ("mobile pelvis with authored floor/support." if env.mobility else "anchored pelvis.") + " Kinematic projection is not exact full-body reconstruction.",
               "body_frame": spec.get("body_frame"),
               "physical_validation": "unavailable", "muscle_activation_ground_truth": "unavailable"})
    progress("train", {"message": "Learning joint control, then comparing against random and no-control baselines"})
    learning = train_human(spec, out, episodes=settings["episodes"], eval_episodes=settings["eval_episodes"],
                           seed=settings["seed"], progress=progress, selected_joint=task_proposal["selected_joint"])
    result = {"profile": "exercise", "source": source, "learning": learning, "task_selection": task_proposal,
              "pose_identity": evidence.get("pose_identity"),
              "reference_frames": len(references), "assumptions": spec["assumptions"],
              "pose_quality": evidence["quality"], "reference_tracking": reference_evaluation,
              "seconds": time.monotonic() - started, "runtime_packages": _runtime_packages(),
              "cost": {"external_api_usd": 0, "cloud_usd": 0},
              "limits": ["Pose and body shape are estimates from one camera, not anatomical measurements",
                         "Muscle bands and intensity are illustrative actuator proxies, not measured activation/force",
                         ("Mobile pelvis uses image-derived displacement and authored floor/virtual forces; no recovered physical balance" if env.mobility else "Pelvis position anchored; authored rotation axes; no full-body balance or biomechanical validation"),
                         "Single-joint target learning is not learned reproduction of the entire exercise"]}
    progress("render", {"message": "Saving rotatable body poses, learning results, and reproducible artifacts"})
    write_json(out / "report.json", result)
    for name, path in implementation.items():
        destination = out / "implementation-source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if digest(destination) != hashes[name]:
            raise RuntimeError("Implementation changed during the run; repeat in a new directory after edits finish")
    write_json(out / "implementation.json", hashes)
    write_json(out / "manifest.json", {"schema": 1, "created_at": local_time(),
               "files": {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    result["verification"] = verify_artifacts(out)
    return result


def _tighten_limits():
    # A supervised parent can impose tighter caps; the child must never raise either inherited limit.
    for kind, requested in ((resource.RLIMIT_CPU, (300, 305)),
                            (resource.RLIMIT_FSIZE, (512 * 1024 * 1024, 512 * 1024 * 1024))):
        old_soft, old_hard = resource.getrlimit(kind)
        hard = requested[1] if old_hard == resource.RLIM_INFINITY else min(requested[1], old_hard)
        soft = min(requested[0], hard) if old_soft == resource.RLIM_INFINITY else min(requested[0], hard, old_soft)
        resource.setrlimit(kind, (soft, hard))


def main():
    _tighten_limits()
    raise SystemExit(execute(sys.argv[1]))


if __name__ == "__main__":
    main()
