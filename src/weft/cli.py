import argparse
import json
from pathlib import Path

from weft.compare import compare_runs, save_comparison
from weft.env import RollingEnv
from weft.experiments import run_experiment
from weft.meta import status
from weft.pipeline import run, verify_artifacts
from weft.source import fetch_sample
from weft.world import World


def main():
    parser = argparse.ArgumentParser(description="Weft: bounded video-to-environment evidence pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-sample", help="Download the pinned CC0 video once")
    fetch.add_argument("--out", default="data/rolling-ball.mp4")
    execute = commands.add_parser("run", help="Run A–E locally; output directory must be new")
    execute.add_argument("--video", default="data/rolling-ball.mp4")
    execute.add_argument("--out", required=True)
    execute.add_argument("--episodes", type=int, default=15000)
    execute.add_argument("--eval-episodes", type=int, default=300)
    execute.add_argument("--seed", type=int, default=1729)
    execute.add_argument("--eval-seed", type=int, default=938401)
    execute.add_argument("--no-repair", action="store_true")
    execute.add_argument("--provider", choices=("local", "meta"), default="local")
    execute.add_argument("--max-meta-cost-usd", type=float)
    experiment = commands.add_parser("experiment", help="Run a hash-pinned experiment config")
    experiment.add_argument("config")
    experiment.add_argument("--out", required=True)
    compare = commands.add_parser("compare", help="Compare verified runs and disclose incompatible benchmarks")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--out", help="Optional new JSON file outside both runs")
    verify = commands.add_parser("verify", help="Verify the complete artifact manifest")
    verify.add_argument("directory")
    commands.add_parser("meta-status", help="Report credential presence without making a request")
    interact = commands.add_parser("interact", help="Step the Python simulator manually")
    interact.add_argument("--world", required=True)
    lab = commands.add_parser("lab", help="Watch video-to-RL stages, interact, and record in a local lab")
    lab.add_argument("--video", help="Local video, at most 100 MB and 60 seconds")
    lab.add_argument("--out", help="New lab directory (runs and recordings stay separate)")
    lab.add_argument("--profile", choices=("planar", "appearance", "rolling", "exercise"), default="exercise")
    lab.add_argument("--pose-model", help="Local pinned MediaPipe model for the exercise profile")
    lab.add_argument("--pose-mode", choices=("video", "image"), default="video",
                     help="Exercise inference: tracked video (default) or independent frames; no automatic fallback")
    lab.add_argument("--pose-identity", choices=("raw", "temporal"), default="raw",
                     help="Keep raw anatomy labels or explicitly repair strongly supported short identity swaps")
    lab.add_argument("--body-model", choices=("hinge", "spatial", "mobile"), default="hinge",
                     help="Exercise body: hinge, spatial, or mobile with explicit root/floor assumptions")
    lab.add_argument("--config", help="Tracking/task JSON; required for appearance region tracking")
    lab.add_argument("--episodes", type=int, default=1500)
    lab.add_argument("--eval-episodes", type=int, default=100)
    lab.add_argument("--seed", type=int, default=1729)
    lab.add_argument("--source-note", default="User-provided local video; rights not independently verified")
    lab.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects a free port")
    lab.add_argument("--no-serve", action="store_true", help="Run stages in the terminal without a web server")
    lab.add_argument("--resume", help="Inspect a saved lab without rerunning its pipeline")
    lab.add_argument("--scene-run", help="Opt-in import of a verified spatial-rigid-v1 run into a new --out lab")
    lab.add_argument("--shared-world-run", help="Opt-in import of a verified shared-world-v1 run into a new --out lab")
    lab.add_argument("--learning", help="Optional verified policy directory paired with --scene-run or --shared-world-run")
    replay = commands.add_parser("replay", help="Verify, export, or non-destructively recover an interaction recording")
    replay.add_argument("directory")
    replay.add_argument("--export", action="store_true", help="Create video and offline replay from a closed journal")
    replay.add_argument("--recover-incomplete", action="store_true",
                        help="Recover complete journal events from an interrupted recording into a new --out directory")
    replay.add_argument("--out", help="New output directory required with --recover-incomplete")
    replay.add_argument("--verify-actions", action="store_true",
                        help="Re-execute a human-only exercise journal and compare exact task/engine outcomes")
    demo = commands.add_parser("demo", help="Create a clearly labeled synthetic tabletop test video")
    demo.add_argument("--out", required=True, help="New directory for video, config, and provenance")
    pose_model = commands.add_parser("fetch-pose-model", help="Download/verify the pinned local pose model (no inference upload)")
    pose_model.add_argument("--out")
    pose_model.add_argument("--variant", choices=("lite", "full", "heavy"), default="lite", help="Explicit model choice; existing lab default remains lite")
    exercise_sample = commands.add_parser("fetch-exercise-sample", help="Download/verify an attributed exercise video")
    exercise_sample.add_argument("--out", default="data/exercise")
    exercise_sample.add_argument("--name", choices=("single-leg-squat", "squat", "kettlebell-pushup", "lunge-twist", "jumping-jacks-burpees"), default="single-leg-squat")
    benchmark = commands.add_parser("benchmark", help="Run a pinned diverse-video suite with explicit evidence gates")
    benchmark.add_argument("manifest")
    benchmark.add_argument("--out", required=True)
    benchmark.add_argument("--workers", type=int, default=2, choices=(1, 2))
    advisor = commands.add_parser("advise", help="Prepare evidence-grounded Meta interaction questions; never apply suggestions automatically")
    advisor.add_argument("--run", required=True, help="Completed lab run/ directory")
    advisor.add_argument("--out", required=True, help="New directory outside the completed run")
    advisor.add_argument("--question", action="append", help="Optional question; repeat up to four times")
    advisor.add_argument("--execute", action="store_true", help="Explicitly send the prepared evidence to Meta")
    advisor.add_argument("--meta-budget", type=float, help="Maximum estimated USD for this single request")
    segment = commands.add_parser("segment", help="Prepare or explicitly execute bounded Meta video tracking, not physics")
    segment.add_argument("--video", required=True, help="MP4, at most 500 frames and 37 MB")
    segment.add_argument("--box", action="append", nargs=6, type=int, required=True,
                         metavar=('ID','FRAME','X','Y','WIDTH','HEIGHT'), help="Source-pixel initialization; repeat for at most eight objects")
    segment.add_argument("--out", required=True, help="New artifact directory")
    segment.add_argument("--execute", action="store_true", help="Send one paid request; otherwise prepare offline")
    segment.add_argument("--meta-budget", type=float, help="Per-request reservation allowance; aggregate budget also applies")
    tracking_view = commands.add_parser("tracking-view", help="Create an offline viewer from sealed tracking; no API calls")
    tracking_view.add_argument("--run", required=True, help="Sealed study containing call/call.json and call/segmentation.json")
    tracking_view.add_argument("--video", required=True, help="Exact MP4 used for that request")
    tracking_view.add_argument("--out", required=True)
    tracking_masks = commands.add_parser("tracking-masks", help="Decode saved Meta silhouettes and inspect exact source frames; no API calls")
    tracking_masks.add_argument("--run", required=True, help="Sealed original tracking-evidence export")
    tracking_masks.add_argument("--video", required=True, help="Exact submitted MP4")
    tracking_masks.add_argument("--out", required=True, help="New decoded-mask and frame-viewer directory")
    tracking_evidence = commands.add_parser("tracking-evidence", help="Export saved model proposals as uncertain scene evidence; no API or physics")
    tracking_evidence.add_argument("--run", required=True, help="Sealed segment tracking directory")
    tracking_evidence.add_argument("--out", required=True, help="New sealed evidence directory")
    tracking_evidence.add_argument("--source-mapping", help="Optional sealed, verified original-to-submitted video mapping bundle")
    tracking_scene = commands.add_parser("tracking-scene", help="Test an explicitly authored planar hypothesis from saved tracking; no API calls")
    tracking_scene.add_argument("--run", required=True, help="Sealed tracking-evidence export")
    tracking_scene.add_argument("--config", required=True, help="Explicit physical, identity, task and partition assumptions JSON")
    tracking_scene.add_argument("--out", required=True, help="New candidate/replay or rejection directory")
    tracking_stabilize = commands.add_parser("tracking-stabilize", help="Test fixed-reference image registration without changing raw tracking or making API calls")
    tracking_stabilize.add_argument("--run", required=True, help="Sealed original tracking-evidence export")
    tracking_stabilize.add_argument("--video", required=True, help="Exact submitted MP4")
    tracking_stabilize.add_argument("--reference-frame", required=True, type=int, help="Latest object initialization frame")
    tracking_stabilize.add_argument("--out", required=True, help="New registration evidence directory")
    scene_console = commands.add_parser("scene-sandbox", help="Control a saved planar or 3D rigid scene and record every action")
    scene_console.add_argument("--run", required=True, help="Verified directory containing scene.json")
    scene_console.add_argument("--record", required=True, help="New recording directory outside the source run")
    scene_console.add_argument("--seed", type=int, default=1729)
    meta_configure = commands.add_parser("meta-configure", help="Configure a persistent total Meta budget and optional hidden local key")
    meta_configure.add_argument("--budget", required=True, type=float, help="Total USD budget; preserves all prior reservations and charges")
    meta_configure.add_argument("--key", action="store_true", help="Read the Meta key through a hidden terminal prompt")
    observations = commands.add_parser("observations", help="Export a validated shared scene-observation graph from an existing run")
    observations.add_argument("--run", required=True)
    observations.add_argument("--out", required=True, help="New JSON file outside the immutable run")
    imitate = commands.add_parser("imitate", help="Learn explicit whole-body motor control from a verified mobile exercise run")
    imitate.add_argument("--run", required=True, help="Existing verified exercise-v3 run directory")
    imitate.add_argument("--out", required=True, help="New learning artifact directory")
    imitate.add_argument("--seed", type=int, default=1729)
    imitate.add_argument("--generations", type=int, default=8)
    imitate.add_argument("--population", type=int, default=16)
    imitate.add_argument("--horizon", type=int, default=60, help="Control steps per trial, at most 180")
    body_console = commands.add_parser("body-sandbox", help="Control all body motors in a terminal and save a visual replay")
    body_console.add_argument("--run", required=True, help="Completed imitate learning directory")
    body_console.add_argument("--record", required=True, help="New recording directory outside the learning run")
    body_console.add_argument("--seed", type=int, default=1729)
    feet_console = commands.add_parser("feet-sandbox", help="Test an opt-in ankle/foot candidate and record explicit motor actions")
    feet_console.add_argument("--body", required=True, help="Existing mobile human.json; kept unchanged")
    feet_console.add_argument("--feet", required=True, help="Matching kinematic foot-reference.json")
    feet_console.add_argument("--record", required=True, help="New recording directory outside the source artifacts")
    feet_console.add_argument("--seed", type=int, default=1729)
    feet_build = commands.add_parser("feet-build", help="Build an optional foot candidate from a completed mobile exercise run")
    feet_build.add_argument("--run", required=True, help="Completed mobile exercise lab run/ directory")
    feet_build.add_argument("--out", required=True, help="New candidate directory outside the source run")
    feet_build.add_argument("--image-directions", action="store_true", help="Opt into bounded image-direction correction; remains unverified")
    body_lab = commands.add_parser("body-lab", help="Prepare a visual whole-body lab from verified source and learning runs")
    body_lab.add_argument("--source-lab", required=True, help="Completed mobile exercise lab directory")
    body_lab.add_argument("--learning", required=True, help="Completed imitate learning directory")
    body_lab.add_argument("--out", required=True, help="New lab directory; does not start a server")
    camera_body = commands.add_parser('camera-body', help='Build a manual camera-aware body lab using supplied calibration; no model calls')
    camera_body.add_argument('--run', required=True, help='Completed original mobile exercise run/ directory')
    camera_body.add_argument('--camera', required=True, help='Source-pinned camera JSON with one transform per original frame')
    camera_body.add_argument('--ground', help='Optional source/calibration-pinned horizontal ground JSON; no pose-based floor fit')
    camera_body.add_argument('--root-view', help='Optional second-view config for hip-midpoint root; requires --ground, preserves monocular shape')
    camera_body.add_argument('--body-view', choices=['directions', 'dimensions', 'both'],
                             help='Optional partial two-view body geometry; requires --root-view, retains explicit priors for missing limbs')
    camera_body.add_argument('--partial-segments', action='store_true',
                             help='Use supported upper segments when wrists/ankles are missing; twist and distal bends remain explicit priors')
    camera_body.add_argument('--out', required=True, help='New candidate lab; no server or training is started')
    spatial_fixture = commands.add_parser("spatial-fixture", help="Write a verified analytic spatial-rigid-v1 fixture run; no API calls")
    spatial_fixture.add_argument("--out", required=True, help="New sealed 3D scene directory")
    spatial_lab = commands.add_parser("spatial-lab", help="Package a verified spatial-rigid-v1 run into a new local lab; no server started")
    spatial_lab.add_argument("--run", required=True, help="Existing verified spatial-rigid-v1 run directory")
    spatial_lab.add_argument("--learning", help="Optional verified spatial policy directory for the same scene")
    spatial_lab.add_argument("--out", required=True, help="New lab directory outside the immutable inputs")
    spatial_train = commands.add_parser("spatial-train", help="Train and evaluate a bounded 3D rigid policy on a verified spatial-rigid-v1 run")
    spatial_train.add_argument("--run", required=True, help="Existing verified spatial-rigid-v1 run directory")
    spatial_train.add_argument("--out", required=True, help="New policy training directory outside the source run")
    spatial_train.add_argument("--seed", type=int, default=1729)
    spatial_train.add_argument("--eval-seed", type=int, default=938401)
    spatial_train.add_argument("--generations", type=int, default=8)
    spatial_train.add_argument("--population", type=int, default=16)
    spatial_train.add_argument("--eval-episodes", type=int, default=12)
    body_eval = commands.add_parser("body-evaluate", help="Evaluate a sealed whole-body policy across changed initial conditions and matched baselines")
    body_eval.add_argument("--learning", required=True, help="Existing verified whole-body learning directory")
    body_eval.add_argument("--out", required=True, help="New evaluation output directory")
    body_eval.add_argument("--seed", type=int, default=938401)
    body_eval.add_argument("--trials-per-window", type=int, default=3)
    capabilities_cmd = commands.add_parser("capabilities", help="Inspect shared environment, action, observation, task, and evidence capabilities")
    capabilities_cmd.add_argument("--run", required=True, help="Existing verified run directory or ready lab directory")
    capabilities_cmd.add_argument("--out", help="Optional new JSON path outside the inspected directory")
    sw_fixture = commands.add_parser("shared-world-fixture", help="Write a verified analytic shared-world-v1 coupled human+object fixture; no API calls")
    sw_fixture.add_argument("--out", required=True, help="New sealed shared-world-v1 directory")
    sw_adapt = commands.add_parser("shared-world-adapt", help="Adapt a verified mobile exercise-v3 run into a source-qualified shared-world-v1 run")
    sw_adapt.add_argument("--human-run", required=True, help="Existing verified mobile exercise-v3 run or lab directory")
    sw_adapt.add_argument("--out", required=True, help="New sealed shared-world-v1 directory outside the source")
    sw_adapt.add_argument("--scale", type=float, default=1.0, help="Explicit body-height-to-world scale prior in [0.5, 2.0]")
    sw_construct = commands.add_parser("shared-world-reconstruct", help="Construct source-linked initial equipment proxies; no API calls or recovered grasp claim")
    sw_construct.add_argument("--run", required=True, help="Sealed source-qualified body/shared-world base")
    sw_construct.add_argument("--proposal-run", required=True, help="Sealed geometry-proposal study")
    sw_construct.add_argument("--evidence-run", required=True, help="Sealed equipment observation export")
    sw_construct.add_argument("--source-mapping", action="append", default=[], help="Verified original/derivative bundle; repeat for distinct derivatives")
    sw_construct.add_argument("--target-object", required=True, help="Authored task target ID from the observation document")
    sw_construct.add_argument("--frame-translation", nargs=2, type=float, metavar=("DX", "DY"),
                              help="Explicit positive XY coordinate-frame translation; preserves body/floor but changes numerical bounds")
    sw_construct.add_argument("--out", required=True, help="New sealed construction output outside the inputs")
    sw_train = commands.add_parser("shared-world-train", help="Train and evaluate a bounded CEM policy on a coupled shared-world-v1 run")
    sw_train.add_argument("--run", required=True, help="Existing verified shared-world-v1 run directory")
    sw_train.add_argument("--out", required=True, help="New policy training directory outside the source run")
    sw_train.add_argument("--seed", type=int, default=1729)
    sw_train.add_argument("--generations", type=int, default=8)
    sw_train.add_argument("--population", type=int, default=16)
    sw_train.add_argument("--eval-episodes", type=int, default=8)
    sw_lab = commands.add_parser("shared-world-lab", help="Package a verified shared-world-v1 run into a new local lab; no server started")
    sw_lab.add_argument("--run", required=True, help="Existing verified shared-world-v1 run directory")
    sw_lab.add_argument("--learning", help="Optional verified shared-world policy directory for the same scene")
    sw_lab.add_argument("--out", required=True, help="New lab directory outside the immutable inputs")
    exercise_eval = commands.add_parser("shared-exercise-evaluate",
        help="Evaluate and record human-only controls with passive equipment; no training or API calls")
    exercise_eval.add_argument("--run", required=True, help="Sealed shared-world candidate or ready lab")
    exercise_eval.add_argument("--out", required=True, help="New evaluation directory outside the immutable input")
    exercise_choice = exercise_eval.add_mutually_exclusive_group()
    exercise_choice.add_argument("--root-assistance", action="store_true", help="Allow three explicit virtual root controls")
    exercise_choice.add_argument("--task", help="Validated shared-exercise-task-v1 JSON, with frozen gates and contact intervals")
    exercise_eval.add_argument("--seed", type=int, default=938401)
    exercise_eval.add_argument("--jitter", action="store_true", help="Use the same seeded reset perturbation for both baselines")
    exercise_eval.add_argument("--max-seconds", type=float, default=60)
    args = parser.parse_args()
    try:
        if args.command == "fetch-sample":
            result = {"path": str(fetch_sample(args.out))}
        elif args.command == "run":
            result = run(args.video, args.out, args.episodes, args.eval_episodes, args.seed,
                         args.provider, args.max_meta_cost_usd, eval_seed=args.eval_seed,
                         repair_enabled=not args.no_repair)
        elif args.command == "experiment":
            result = run_experiment(args.config, args.out)
        elif args.command == "compare":
            result = compare_runs(args.baseline, args.candidate)
            if args.out:
                save_comparison(args.out, result)
        elif args.command == "verify":
            result = verify_artifacts(args.directory)
        elif args.command == "benchmark":
            from weft.benchmark import run_suite
            result = run_suite(args.manifest, args.out, args.workers)
        elif args.command == "advise":
            from weft.scene_advisor import advise_run
            result = advise_run(args.run, args.out, args.question,
                                execute=args.execute, max_cost_usd=args.meta_budget)
        elif args.command == "meta-configure":
            from weft.meta_access import configure_access
            result = configure_access(total_budget_usd=args.budget, prompt=args.key)
        elif args.command == "segment":
            from weft.segmentation_lab import track_video
            result = track_video(args.video, args.out, args.box, execute=args.execute, max_cost_usd=args.meta_budget)
        elif args.command == "tracking-view":
            from weft.segmentation_lab import prepare_tracking_view
            result = prepare_tracking_view(args.run, args.video, args.out)
        elif args.command == "tracking-masks":
            from weft.segmentation_masks import export_masks
            result = export_masks(args.run, args.video, args.out)
        elif args.command == "tracking-evidence":
            from weft.segmentation_observations import export_tracking_evidence
            result = export_tracking_evidence(args.run, args.out, source_mapping_bundle=args.source_mapping)
        elif args.command == "tracking-scene":
            from weft.segmentation_scene import export_candidate
            path = Path(args.config)
            if path.stat().st_size > 65536:
                raise ValueError("Tracking scene assumptions must be at most 64 KiB")
            result = export_candidate(args.run, json.loads(path.read_text()), args.out)
        elif args.command == "scene-sandbox":
            from weft.scene_console import console
            result = console(args.run, args.record, seed=args.seed)
        elif args.command == "tracking-stabilize":
            from weft.stabilized_segmentation import stabilize_tracking
            result = stabilize_tracking(args.run, args.video, args.out, reference_frame=args.reference_frame)
        elif args.command == "observations":
            from weft.io import write_json
            from weft.observations import adapt_run
            source, output = Path(args.run).resolve(), Path(args.out).resolve()
            if output.exists() or output == source or source in output.parents:
                raise ValueError("Use a new output file outside the source run")
            document = adapt_run(source)
            write_json(output, document)
            result = {"path": str(output), "scope": "Shared observations and explicit unknowns; not generalized reconstruction"}
        elif args.command == "meta-status":
            result = status()
        elif args.command == "imitate":
            from weft.whole_body_learning import train_whole_body
            verify_artifacts(args.run)
            spec = json.loads((Path(args.run) / "human.json").read_text())
            result = train_whole_body(spec, args.out, seed=args.seed, generations=args.generations,
                                      population=args.population, horizon_steps=args.horizon)
        elif args.command == "body-sandbox":
            from weft.whole_body_console import console
            result = console(args.run, args.record, seed=args.seed)
        elif args.command == "feet-sandbox":
            from weft.foot_sandbox import console
            result = console(args.body, args.feet, args.record, seed=args.seed)
        elif args.command == "feet-build":
            from weft.foot_candidate import build_candidate
            result = build_candidate(args.run, args.out, image_directions=args.image_directions)
        elif args.command == "body-lab":
            from weft.whole_body_lab import prepare_whole_body_lab
            result = prepare_whole_body_lab(args.source_lab, args.learning, args.out)
        elif args.command == 'camera-body':
            from weft.camera_body_lab import prepare_camera_body_lab
            result = prepare_camera_body_lab(args.run, args.camera, args.out, ground_path=args.ground,
                                             root_view_path=args.root_view, body_view=args.body_view, partial_segments=args.partial_segments)
        elif args.command == "spatial-fixture":
            from weft.spatial_lab import write_sealed_spatial_fixture
            result = write_sealed_spatial_fixture(args.out)
        elif args.command == "spatial-lab":
            from weft.spatial_lab import prepare_spatial_lab
            result = prepare_spatial_lab(args.run, args.out, policy_run=args.learning)
        elif args.command == "spatial-train":
            from weft.spatial_learning import train_spatial_policy
            result = train_spatial_policy(
                args.run,
                args.out,
                seed=args.seed,
                eval_seed=args.eval_seed,
                generations=args.generations,
                population=args.population,
                eval_episodes=args.eval_episodes,
            )
        elif args.command == "body-evaluate":
            from weft.whole_body_learning import evaluate_changed_conditions
            result = evaluate_changed_conditions(
                args.learning,
                args.out,
                seed=args.seed,
                trials_per_window=args.trials_per_window,
            )
        elif args.command == "capabilities":
            from weft.capabilities import describe_run_capabilities
            result = describe_run_capabilities(args.run, out=args.out)
        elif args.command == "shared-world-fixture":
            from weft.shared_world_lab import write_sealed_shared_world_fixture
            result = write_sealed_shared_world_fixture(args.out)
        elif args.command == "shared-world-adapt":
            from weft.shared_world_lab import adapt_to_shared_world
            result = adapt_to_shared_world(args.human_run, args.out, scale=args.scale)
        elif args.command == "shared-world-reconstruct":
            from weft.shared_world_pipeline import construct_source_equipment_run
            result = construct_source_equipment_run(args.run, args.proposal_run, args.evidence_run, args.out,
                target_object_id=args.target_object, source_mappings=args.source_mapping,
                frame_translation=args.frame_translation)
        elif args.command == "shared-world-train":
            from weft.shared_world_learning import train_shared_world_policy
            result = train_shared_world_policy(
                args.run,
                args.out,
                seed=args.seed,
                generations=args.generations,
                population=args.population,
                eval_episodes=args.eval_episodes,
            )
        elif args.command == "shared-world-lab":
            from weft.shared_world_lab import prepare_shared_world_lab
            result = prepare_shared_world_lab(args.run, args.out, policy_run=args.learning)
        elif args.command == "shared-exercise-evaluate":
            from weft.shared_exercise_run import evaluate_shared_exercise_run
            result = evaluate_shared_exercise_run(args.run, args.out, root_assistance=args.root_assistance,
                task_path=args.task, seed=args.seed, jitter=args.jitter, max_seconds=args.max_seconds)
        elif args.command == "lab":
            from weft.lab import launch_lab
            result = launch_lab(args)
        elif args.command == "replay":
            from weft.recording import export_recording, recover_incomplete_recording, verify_recording
            if args.recover_incomplete:
                if not args.out:
                    raise ValueError("--recover-incomplete requires --out <new-directory> to preserve the interrupted source")
                result = recover_incomplete_recording(args.directory, args.out, export=True)
            else:
                if args.out:
                    raise ValueError("--out is supported only with --recover-incomplete")
                result = export_recording(args.directory) if args.export else verify_recording(args.directory)
            if args.verify_actions:
                from weft.shared_exercise_recording import verify_action_replay
                result["action_replay"] = verify_action_replay(args.out if args.recover_incomplete else args.directory)
        elif args.command == "demo":
            from weft.demo import create_demo
            result = create_demo(args.out)
        elif args.command == "fetch-pose-model":
            from weft.exercise_vision import fetch_pose_model
            result = {"path": str(fetch_pose_model(args.out, variant=args.variant))}
        elif args.command == "fetch-exercise-sample":
            from weft.exercise_source import fetch_exercise_sample
            result = fetch_exercise_sample(args.out, args.name)
        else:
            env = RollingEnv(World.from_dict(json.loads(Path(args.world).read_text())))
            print("l = left, c = coast, r = right, reset = new trial, q = quit")
            print(env.reset(seed=0))
            while True:
                command = input("> ").strip()
                if command == "q":
                    break
                if command == "reset":
                    print(env.reset(seed=0)); continue
                if command not in {"l", "c", "r"}:
                    print("Use l, c, r, reset, or q"); continue
                try:
                    observation, reward, terminated, truncated, info = env.step({"l": 0, "c": 1, "r": 2}[command])
                    print(json.dumps({"observation": observation.tolist(), "reward": reward,
                                      "terminated": terminated, "truncated": truncated, **info}))
                except RuntimeError as error:
                    print(error)
            result = {"status": "closed"}
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        parser.exit(1, f"weft: {error}\n")


if __name__ == "__main__":
    main()
