import argparse
import json
from pathlib import Path
import shlex
import sys

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
    portal = commands.add_parser("portal", help="Open the local video-to-simulation web portal")
    portal.add_argument("--root", default=".local/portal", help="Persistent private project and recording directory")
    portal.add_argument("--port", type=int, default=8765)
    portal_import = commands.add_parser("portal-import", help="Copy a verified exercise lab into the stopped portal")
    portal_import.add_argument("--lab", required=True)
    portal_import.add_argument("--root", default=".local/portal")
    portal_import.add_argument("--name", help="Project filename with the original video extension")
    portal_attach = commands.add_parser("portal-attach-evidence", help="Verify original body evidence for an imported portal world's Muse review")
    portal_attach.add_argument("--root", default=".local/portal")
    portal_attach.add_argument("--project-id", required=True)
    portal_attach.add_argument("--body-source", required=True, help="Exact sealed body package used by the imported world")
    portal_attach.add_argument("--max-seconds", type=float, default=30)
    portal_equipment = commands.add_parser("portal-equipment-register", help="Register exact prepared equipment evidence for matching portal uploads; no scan or API call")
    portal_equipment.add_argument("--root", default=".local/portal")
    portal_equipment.add_argument("--source-sha256", required=True)
    portal_equipment.add_argument("--run", required=True, help="Source-qualified body/world package")
    portal_equipment.add_argument("--evidence-run", required=True)
    portal_equipment.add_argument("--source-mapping", action="append", default=[])
    portal_equipment.add_argument("--label", required=True)
    openai_setup = commands.add_parser("openai-budget", help="Set a lifetime portal API allowance without resetting charges")
    openai_setup.add_argument("--total-usd", type=float, required=True)
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
    lab.add_argument("--episodes", type=int, help="Training episodes (default: 1500)")
    lab.add_argument("--eval-episodes", type=int, help="Evaluation episodes (default: 100)")
    lab.add_argument("--no-train", action="store_true",
                     help="Build a manual planar/appearance video lab without training or policy evaluation")
    lab.add_argument("--seed", type=int, default=1729)
    lab.add_argument("--source-note", default="User-provided local video; rights not independently verified")
    lab.add_argument("--port", type=int, default=0, help="Loopback port; 0 selects a free port")
    lab.add_argument("--avatar", action="append", help="Sealed local avatar import; repeat up to four times")
    lab.add_argument("--no-serve", action="store_true", help="Run stages in the terminal without a web server")
    lab.add_argument("--resume", help="Inspect a saved lab without rerunning its pipeline")
    lab.add_argument("--scene-run", help="Opt-in import of a verified spatial-rigid-v1 run into a new --out lab")
    lab.add_argument("--shared-world-run", help="Opt-in import of a verified shared-world-v1 run into a new --out lab")
    lab.add_argument("--learning", help="Optional verified policy directory paired with --scene-run or --shared-world-run")
    lab_console = commands.add_parser("lab-console", help="Interact with a saved lab and record in the terminal; no server, browser or training")
    lab_console.add_argument("--resume", required=True, help="Ready lab matching the installed runtime; completed run stays unchanged")
    lab_console.add_argument("--seed", type=int, default=1729)
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
    pose_recovery = commands.add_parser("exercise-pose-recover", help="Verify saved source-only crop proposals for body preparation; no new inference or simulation")
    pose_recovery.add_argument("--run", required=True, help="Sealed selective-crop proposal package with its original whole-image evidence")
    pose_recovery.add_argument("--video", required=True, help="Exact original source video; bytes verified without decoding")
    pose_recovery.add_argument("--out", required=True, help="New sealed pose-evidence directory outside the inputs")
    pose_recovery.add_argument("--max-seconds", type=float, default=30)
    mode_recovery = commands.add_parser("exercise-pose-recover-modes", help="Compare saved VIDEO/IMAGE poses using source consistency; no new inference or simulation")
    mode_recovery.add_argument("--primary-run", required=True, help="Sealed original single-person VIDEO pose run")
    mode_recovery.add_argument("--alternative-run", required=True, help="Sealed original two-hypothesis IMAGE run using the same model and source")
    mode_recovery.add_argument("--video", required=True, help="Exact original video; hash-checked without decoding")
    mode_recovery.add_argument("--out", required=True, help="New sealed evidence directory outside the inputs")
    mode_recovery.add_argument("--max-seconds", type=float, default=30)
    pose_proposals = commands.add_parser("exercise-pose-prepare", help="Generate bounded source-only crop proposals from video using explicit Full/IMAGE/raw inference")
    pose_proposals.add_argument("--video", required=True, help="Local exercise video; source bytes and clocks stay unchanged")
    pose_proposals.add_argument("--pose-model", required=True, help="Pinned local Full model; no model download or automatic alternative")
    pose_proposals.add_argument("--source-note", default="Rights not independently verified; user-provided local video")
    pose_proposals.add_argument("--out", required=True, help="New sealed proposal directory outside the inputs; not a playable lab")
    pose_proposals.add_argument("--max-seconds", type=float, default=180)
    body_source = commands.add_parser("exercise-body", help="Prepare body evidence for a shared world without creating or training a body-only lab")
    body_input = body_source.add_mutually_exclusive_group(required=True)
    body_input.add_argument("--video", help="Local exercise video, at most 64 MiB and 60 seconds")
    body_input.add_argument("--evidence-run", help="Sealed pose run; reuse exact source/model/options without inference")
    body_clock = body_source.add_mutually_exclusive_group()
    body_clock.add_argument("--native-clock", help="With --evidence-run: exact source-native-clock-v1 JSON; preserve raw detector times")
    body_clock.add_argument("--native-timing", action="store_true", help="With --video: capture native presentation timing while preserving detector timestamps")
    body_source.add_argument("--pose-model", help="Required with --video: local pinned pose model")
    body_source.add_argument("--pose-mode", choices=("video", "image"), help="Raw-video inference mode; default video")
    body_source.add_argument("--pose-identity", choices=("raw", "temporal"), help="Raw-video anatomy handling; default raw")
    body_source.add_argument("--pose-sampling", choices=("bounded", "all_frames"),
                             help="Default bounded sampling; all_frames observes every original frame, at most 900")
    body_source.add_argument("--max-pose-hypotheses", type=int, choices=(1, 2),
                             help="Default 2; 1 explicitly assumes one person and can miss competing people")
    body_source.add_argument("--person-region", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                             help="Explicit fixed person region in original pixels; preserves full-frame coordinates and competing detections")
    body_source.add_argument("--source-note", help="Raw-video provenance note; never overrides a saved source")
    body_source.add_argument("--max-seconds", type=float, help="Cooperative bound: default 180 for video, 30 for saved evidence")
    body_source.add_argument("--out", required=True, help="New sealed construction-only output; not a playable lab")
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
    perception = commands.add_parser("exercise-perceive", help="Bind saved AI equipment regions to original MP4 tracking and body evidence; offline by default")
    perception.add_argument("--video", required=True, help="Exact analyzed MP4, at most 500 frames and 37 MB")
    perception.add_argument("--analysis", required=True, help="Saved portal analysis with source frames and original provider request/response")
    perception.add_argument("--out", required=True)
    perception.add_argument("--pose-model", help="Local pose model; required for execution unless --body-source is supplied")
    perception.add_argument("--body-source", help="Reuse a sealed exact-source body package without pose inference")
    tracking_source = perception.add_mutually_exclusive_group()
    tracking_source.add_argument("--execute-tracking", action="store_true", help="Make one bounded Meta tracking request, after local body checks")
    tracking_source.add_argument("--saved-tracking", help="Exact-source sealed tracking with identical initialization prompts; no API call")
    perception.add_argument("--meta-budget", type=float, help="Explicit single-request allowance; existing shared project ledger still applies")
    perception.add_argument("--max-seconds", type=float, default=420)
    tracking_view = commands.add_parser("tracking-view", help="Create an offline viewer from sealed tracking; no API calls")
    tracking_view.add_argument("--run", required=True, help="Sealed study containing call/call.json and call/segmentation.json")
    tracking_view.add_argument("--video", required=True, help="Exact MP4 used for that request")
    tracking_view.add_argument("--out", required=True)
    tracking_masks = commands.add_parser("tracking-masks", help="Decode saved Meta silhouettes and inspect exact source frames; no API calls")
    tracking_masks.add_argument("--run", required=True, help="Sealed original tracking-evidence export")
    tracking_masks.add_argument("--video", required=True, help="Exact submitted MP4")
    tracking_masks.add_argument("--out", required=True, help="New decoded-mask and frame-viewer directory")
    tracking_masks.add_argument("--retain-binary", action="store_true",
                                help="Keep already-decoded binary masks for later pose/equipment checks; no extra decode")
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
    stereo_ground = commands.add_parser("stereo-ground", help="Estimate conditional ground from manual points in two calibrated source images")
    stereo_ground.add_argument("--camera", action="append", required=True, help="Two source-bound camera JSON files, in annotation-view order")
    stereo_ground.add_argument("--image", action="append", required=True, help="Two original-raster JPEG/PNG source frames, in the same order")
    stereo_ground.add_argument("--annotations", required=True, help="Explicit point pairs, source/frame identities, partitions and pixel uncertainty")
    stereo_ground.add_argument("--out", required=True, help="New calibration/report directory under an existing parent")
    floor_calibration = commands.add_parser("floor-calibrate", help="Check measured camera/floor references and retain nominal calibration alternatives")
    floor_calibration.add_argument("--observation", action="append", required=True, help="Exact source observation JSON; repeat up to eight times")
    floor_calibration.add_argument("--source-sha256", required=True, help="Original video SHA256 shared by the source observations")
    floor_calibration.add_argument("--request", required=True, help="Source-pinned calibration mode, measurement references and explicit construction frames")
    floor_calibration.add_argument("--measurement", action="append", default=[], help="Exact measurement JSON with provenance and uncertainty; repeat up to eight times")
    floor_calibration.add_argument("--camera", help="Original camera sidecar for the explicitly requested supplied mode")
    floor_calibration.add_argument("--ground", help="Original matching ground sidecar; never applied to a body automatically")
    floor_calibration.add_argument("--out", required=True, help="New output directory under an existing parent")
    floor_calibration.add_argument("--max-seconds", type=float, default=30., help="Packaging limit of 10–30 seconds; numerical work runs in one bounded child")
    surface_anchors = commands.add_parser("surface-anchors", help="Map annotated video points onto supplied calibrated surfaces; save conditional coordinates and a report")
    surface_anchors.add_argument("--observation", action="append", required=True, help="Exact qualified observation JSON; repeat up to eight times")
    surface_anchors.add_argument("--source-sha256", required=True, help="Original source video SHA256 shared by the observations")
    surface_anchors.add_argument("--camera", help="Supplied source-pinned camera calibration JSON; no fitting or default camera")
    surface_anchors.add_argument("--surfaces", help="Explicit horizontal ground/static-support surface JSON")
    surface_anchors.add_argument("--annotations", help="Source-frame-bound authored pixel/surface annotations JSON")
    surface_anchors.add_argument("--out", required=True, help="New output directory under an existing parent")
    surface_anchors.add_argument("--max-seconds", type=float, default=30., help="Cooperative packaging deadline; not a hard process or memory limit")
    surface_inputs = commands.add_parser("surface-inputs", help="Collect source-linked surface inputs and inspect missing fields",
        description="Collect source-linked surface inputs and export the existing fit config; never fit or install a body")
    surface_source = surface_inputs.add_mutually_exclusive_group(required=True)
    surface_source.add_argument("--observation", action="append", help="Exact saved observation JSON; repeat up to eight times")
    surface_source.add_argument("--resume", help="Sealed prior surface-inputs output; originals remain unchanged")
    surface_inputs.add_argument("--source-sha256", help="Optional original video SHA256; otherwise require matching observation declarations, not a new media hash")
    surface_inputs.add_argument("--camera", help="Exact supplied camera JSON for a new collection; no inferred/default camera")
    surface_inputs.add_argument("--body", help="Exact original human.json for a new collection; not a replacement body")
    surface_edit = surface_inputs.add_mutually_exclusive_group()
    surface_edit.add_argument("--selection", help="Explicit edited entity/channel selection downloaded from the prior report")
    surface_edit.add_argument("--request", help="Explicit edited collection-request.json; requires --resume")
    surface_inputs.add_argument("--manifest-sha256", help="Optional exact prior output-manifest SHA256; requires --resume")
    surface_inputs.add_argument("--out", required=True, help="New output directory; supplied inputs stay unchanged")
    surface_inputs.add_argument("--max-seconds", type=float, default=30., help="Cooperative metadata/packaging deadline; no native model, fitting or media decoding")
    equipment_compose = commands.add_parser("exercise-equipment-evidence",
        help="Join saved pose, equipment and masks into inspectable image evidence; no new inference")
    equipment_compose.add_argument("--pose-run", required=True, help="Sealed complete pose evidence; body admission is separate")
    equipment_compose.add_argument("--equipment-run", required=True, help="Sealed original tracking-evidence export")
    equipment_compose.add_argument("--masks-run", required=True, help="Sealed tracking-masks export with --retain-binary")
    equipment_compose.add_argument("--human-id", required=True, help="Explicit original person identity")
    equipment_compose.add_argument("--equipment-id", action="append", required=True, help="Explicit tracked object ID; repeat for each selected object")
    equipment_compose.add_argument("--minimum-confidence", type=float, required=True, help="Authored pose-confidence threshold, not measured accuracy")
    equipment_compose.add_argument("--uncertainty-px", type=float, required=True, help="Authored image-point range in pixels")
    equipment_compose.add_argument("--uncertainty-method", required=True, help="How the authored pixel range was chosen")
    equipment_compose.add_argument("--uncertainty-note", required=True, help="Limits of the chosen range; not contact or calibration")
    equipment_compose.add_argument("--source-equivalence-run", help="Verified original-to-submitted mapping when source bytes differ")
    for source_kind in ("pose", "equipment", "masks"):
        equipment_compose.add_argument("--" + source_kind + "-manifest-sha256", help="Optional exact " + source_kind + " seal SHA256")
    equipment_compose.add_argument("--out", required=True, help="New evidence package outside the original runs")
    equipment_compose.add_argument("--max-seconds", type=float, default=30., help="Cooperative saved-evidence preparation bound")
    equipment_report = commands.add_parser("equipment-evidence-report", help="Inspect saved wrist/equipment image evidence without rerunning reconstruction")
    equipment_report.add_argument("--run", required=True, help="Completed sealed study with silhouette metadata and full frame records")
    equipment_report.add_argument("--out", required=True, help="New offline report directory outside the original study")
    equipment_report.add_argument("--manifest-sha256", help="Optional exact study seal SHA256")
    equipment_report.add_argument("--max-seconds", type=float, default=10., help="Cooperative saved-metadata/report deadline; no video, mask or physics processing")
    equipment_support = commands.add_parser("equipment-support", help="Check saved equipment candidates against image evidence; never infer physical contact")
    equipment_support.add_argument("--run", required=True, help="Completed pose/equipment silhouette study")
    equipment_support.add_argument("--out", required=True, help="New comparison directory; originals remain unchanged")
    equipment_support.add_argument("--fixed-camera", required=True, action="store_true",
                                   help="Explicitly assume a fixed source camera; not measured calibration")
    equipment_support.add_argument("--manifest-sha256", help="Optional exact source-manifest hash")
    equipment_support.add_argument("--hand-points", action="store_true",
                                   help="Include saved wrist/finger estimates using cached binary masks; not proof of contact")
    equipment_support.add_argument("--max-seconds", type=float, default=20.)
    body_lab = commands.add_parser("body-lab", help="Prepare a visual whole-body lab from verified source and learning runs")
    body_lab.add_argument("--source-lab", required=True, help="Completed mobile exercise lab directory")
    body_lab.add_argument("--learning", required=True, help="Completed imitate learning directory")
    body_lab.add_argument("--out", required=True, help="New lab directory; does not start a server")
    body_reference_report = commands.add_parser("body-reference-report", help="Inspect a saved body-fitting proposal without rerunning inference or physics")
    body_reference_report.add_argument("--run", required=True, help="Sealed whole-body reference proposal study")
    body_reference_report.add_argument("--out", required=True, help="New offline report directory")
    body_reference_report.add_argument("--manifest-sha256", help="Optional exact source-manifest hash")
    body_reference_report.add_argument("--max-seconds", type=float, default=10)
    body_reference_propose = commands.add_parser("body-reference-propose", help="Fit one saved body in a bounded worker and save an offline comparison; never install it")
    body_reference_propose.add_argument("--run", required=True, help="Sealed construction-only exercise-body source")
    body_reference_propose.add_argument("--out", required=True, help="New proposal and inspection directory")
    body_reference_propose.add_argument("--manifest-sha256", help="Optional exact body-source manifest hash")
    body_reference_propose.add_argument("--objective", choices=("image-first", "floor-first", "camera-root", "surface-intervals", "camera-surface-intervals", "camera-multiview-surface-intervals"), default="image-first",
                                        help="Image-first default; camera-root keeps body angles and shape fixed")
    body_reference_propose.add_argument("--surface-config", help="Exact source/camera/static-surface interval inputs and authored weight; required for surface-interval objectives")
    body_reference_propose.add_argument("--camera-image-scale", type=float,
                                        help="Explicit [1e-6,1] authored loss units/pixel; required for camera surface fitting, which saves an uninstalled angle proposal")
    body_reference_propose.add_argument("--secondary-view-config",
                                        help="Exact second-view observations, explicit world/clock mapping and authored weight; only for camera-multiview-surface-intervals")
    body_reference_propose.add_argument("--camera", help="Explicit frozen camera-hypothesis JSON; required only for camera-root")
    body_reference_propose.add_argument("--camera-focal-px", type=float,
                                        help="Alternative to --camera: explicitly authored pinhole focal length in pixels, not calibration")
    body_reference_propose.add_argument("--fit-seconds", type=float, help="Fitting bound; default/max 30 seconds, or 10 for camera-root")
    body_reference_propose.add_argument("--max-seconds", type=float, default=60, help="Workflow time bound, 20–60 seconds plus owned cleanup")
    reference_trial = commands.add_parser("body-reference-trial", help="Compare a fitted reference controller with authored control on an unchanged task; record both")
    reference_trial.add_argument("--proposal", required=True, help="Completed body-reference-propose proposal/ directory; passing fit gates required")
    reference_trial.add_argument("--run", required=True, help="Sealed original shared-world source directory; never modified")
    reference_trial.add_argument("--task", required=True, help="Exact original exercise-task JSON; never regenerated from the proposal")
    reference_trial.add_argument("--out", required=True, help="New trial, recordings and replay directory")
    reference_trial.add_argument("--seed", type=int, default=1729, help="Same reset seed for both controllers; exposed development, not final evaluation")
    reference_trial.add_argument("--max-seconds", type=float, default=60, help="Cooperative workflow bound, at most 120 seconds")
    reference_trial.add_argument("--proposal-manifest-sha256", help="Optional exact proposal-study seal")
    reference_trial.add_argument("--scene-manifest-sha256", help="Optional exact original-world seal")
    reference_trial.add_argument("--task-sha256", help="Optional exact original-task byte hash")
    camera_body = commands.add_parser('camera-body', help='Build a manual camera-aware body lab using supplied calibration; no model calls')
    camera_body.add_argument('--run', required=True, help='Completed original mobile exercise run/ directory')
    camera_body.add_argument('--camera', required=True, help='Source-pinned camera JSON with one transform per original frame')
    camera_body.add_argument('--control-joint', help='Initial manual joint; required for construction-only bodies without a task/policy selector')
    camera_body.add_argument('--ground', help='Optional source/calibration-pinned horizontal ground JSON; no pose-based floor fit')
    camera_body.add_argument('--root-view', help='Optional second-view config for hip-midpoint root; requires --ground, preserves monocular shape')
    camera_body.add_argument('--additional-view', action='append', default=[],
                             help='Third/fourth calibrated view config; requires --root-view, retains per-joint contributors and conflicts')
    camera_body.add_argument('--body-view', choices=['directions', 'dimensions', 'both'],
                             help='Optional partial two-view body geometry; requires --root-view, retains explicit priors for missing limbs')
    camera_body.add_argument('--partial-segments', action='store_true',
                             help='Use supported upper segments when wrists/ankles are missing; twist and distal bends remain explicit priors')
    camera_body.add_argument('--fit-limbs', action='store_true',
                             help='Bounded angle fit to supported joint positions; preserves fixed lengths/root/ground, requires --body-view directions or both')
    camera_body.add_argument('--fit-partial-pixels', action='store_true',
                             help='Constrain unseen wrists/ankles from available images; depth remains inferred, requires --fit-limbs and --partial-segments')
    camera_body.add_argument('--partial-pixel-uncertainty',
                             help='Explicit assumed image/angle error JSON for the opt-in v2 fit; requires --fit-partial-pixels')
    camera_body.add_argument('--partial-depth-branches', action='store_true',
                             help='Assess nominal depth alternatives using bracketing source poses; requires --partial-pixel-uncertainty')
    camera_body.add_argument('--fit-joint-pixels', metavar='SETTINGS.json',
                             help='Fit knee/elbow and ankle/wrist images across time with explicit uncertainty; requires --fit-limbs and --partial-segments')
    camera_body.add_argument('--out', required=True, help='New candidate lab; no server or training is started')
    spatial_fixture = commands.add_parser("spatial-fixture", help="Write a verified analytic spatial-rigid-v1 fixture run; no API calls")
    spatial_fixture.add_argument("--out", required=True, help="New sealed 3D scene directory")
    spatial_scene = commands.add_parser("spatial-scene", help="Build a 3D hypothesis from a sealed planar scene and explicit depth/material assumptions; no simulation")
    spatial_scene.add_argument("--run", required=True, help="Sealed planar reconstruction directory; preserved unchanged")
    spatial_scene.add_argument("--config", required=True, help="Explicit source-pinned 3D assumptions JSON")
    spatial_scene.add_argument("--scene-artifact", default="scene.json", help="Exact sealed relative scene path; no automatic candidate selection")
    spatial_scene.add_argument("--out", required=True, help="New 3D candidate directory outside the input")
    spatial_scene.add_argument("--manifest-sha256", help="Optional exact source-manifest hash")
    spatial_scene.add_argument("--max-seconds", type=float, default=20, help="Cooperative construction bound, at most 20 seconds")
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
    spatial_train.add_argument("--max-seconds", type=float, default=90, help="Cooperative training/evaluation limit, 1–180 seconds")
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
    sw_propose = commands.add_parser("shared-world-propose", help="Prepare explicit equipment priors directly from verified body/tracking evidence; no API or fitting")
    sw_propose.add_argument("--run", required=True, help="Sealed source-qualified body/shared-world base")
    sw_propose.add_argument("--evidence-run", required=True, help="Sealed equipment observation export")
    sw_propose.add_argument("--config", required=True, help="Source-pinned camera, equipment and coordinate-frame choices")
    sw_propose.add_argument("--source-mapping", action="append", default=[], help="Verified original/derivative bundle; repeat for distinct derivatives")
    sw_propose.add_argument("--out", required=True, help="New sealed proposal directory outside the inputs")
    sw_propose.add_argument("--max-seconds", type=float, default=30)
    exercise_config = commands.add_parser("shared-exercise-config",
        help="Choose explicit equipment/camera settings from saved inputs; prepare a configuration, not a world")
    config_source = exercise_config.add_mutually_exclusive_group(required=True)
    config_source.add_argument("--human-run", help="Sealed saved mobile body or ready lab; never pose-only evidence")
    config_source.add_argument("--resume", help="Prior sealed configuration package; reuse only its copied inputs")
    exercise_config.add_argument("--evidence-run", help="Original sealed equipment observations; required with --human-run")
    exercise_config.add_argument("--source-mapping", action="append", default=[],
                                 help="Complete verified source mapping for discovery; repeat for distinct derivatives")
    exercise_config.add_argument("--request", help="Edited request downloaded from the setup page; requires --resume")
    exercise_config.add_argument("--manifest-sha256", help="Optional exact prior configuration-package seal; requires --resume")
    exercise_config.add_argument("--out", required=True, help="New immutable setup package and offline page")
    exercise_config.add_argument("--max-seconds", type=float, default=30., help="Cooperative saved-input preparation limit; no inference or physics")
    exercise_auto = commands.add_parser("shared-exercise-auto-build", help="Propose equipment roles from saved evidence and build a conditional world; abstain on ambiguity")
    exercise_auto.add_argument("--run", required=True, help="Sealed source-qualified shared-world base")
    exercise_auto.add_argument("--evidence-run", required=True, help="Sealed equipment observation export")
    exercise_auto.add_argument("--source-mapping", action="append", default=[])
    exercise_auto.add_argument("--out", required=True, help="New output directory outside the inputs")
    exercise_auto.add_argument("--max-seconds", type=float, default=90)
    exercise_conditional = commands.add_parser("shared-exercise-alternatives", help="Build bounded conditional equipment-role alternatives without resolving raw identities")
    exercise_conditional.add_argument("--run", required=True)
    exercise_conditional.add_argument("--evidence-run", required=True)
    exercise_conditional.add_argument("--source-mapping", action="append", default=[])
    exercise_conditional.add_argument("--out", required=True)
    exercise_conditional.add_argument("--max-candidates", type=int, choices=(2, 3, 4), default=3)
    exercise_conditional.add_argument("--max-seconds", type=float, default=90)
    exercise_build = commands.add_parser("shared-exercise-build", help="Build one exercise world and lab from video or saved body evidence")
    exercise_source = exercise_build.add_mutually_exclusive_group(required=True)
    exercise_source.add_argument("--run", help="Sealed source-qualified shared-world base")
    exercise_source.add_argument("--human-run", help="Sealed exercise-body output, mobile exercise-v3 run or ready lab")
    exercise_source.add_argument("--video", help="Local exercise video; requires --pose-model and person-only, static-support priors or saved equipment evidence")
    exercise_source.add_argument("--geometry-run", help="Reuse a completed saved conditional silhouette alternative and its exact task; no fresh-video reconstruction or refit")
    exercise_build.add_argument("--geometry-manifest-sha256", help="With --geometry-run: pin the exact completed source-study manifest")
    exercise_build.add_argument("--pose-model", help="Raw video only: pinned local pose model; no download or upload")
    exercise_build.add_argument("--pose-mode", choices=("video", "image"), help="Raw video only; default video")
    exercise_build.add_argument("--pose-identity", choices=("raw", "temporal"), help="Raw video only; default raw")
    exercise_build.add_argument("--native-timing", action="store_true", help="Raw video only: use captured native presentation timing, including selective-crop recovery")
    exercise_build.add_argument("--pose-recovery", choices=("selective-crop",),
        help="Raw video only: explicit Full/IMAGE/raw baseline and bounded source-only crops; requires the local Full model")
    exercise_build.add_argument("--source-note", help="Raw video provenance; default rights not independently verified")
    exercise_build.add_argument("--body-scale", type=float,
                                help="With --human-run or --video: body-height-to-world scale prior in [0.5, 2.0] (default 1.0)")
    exercise_scope = exercise_build.add_mutually_exclusive_group()
    exercise_scope.add_argument("--person-only", action="store_true",
        help="Declare one selected person with ground-only support and no task-relevant equipment/obstacles; absence is not verified, full reference retained")
    exercise_scope.add_argument("--static-supports", help="Source-pinned authored static-support priors, including scale/root controls; no observed-equipment claim")
    exercise_build.add_argument("--evidence-run", help="Sealed equipment observation export; required unless declaring person-only or static supports")
    exercise_build.add_argument("--config", help="Source-pinned equipment configuration; required unless declaring person-only or static supports")
    exercise_build.add_argument("--source-mapping", action="append", default=[])
    exercise_build.add_argument("--root-assistance", action="store_true", help="Enable three explicitly assisted root controls")
    exercise_build.add_argument("--full-reference", action="store_true",
                                help="Run every retained body reference in one continuous episode; preserve missing source poses")
    exercise_build.add_argument("--train", action="store_true", help="Run bounded joint-policy search, training-only selection and separate evaluation before preparing one lab")
    exercise_build.add_argument("--training-seed", type=int, help="With --train; default 1729")
    exercise_build.add_argument("--evaluation-seed", type=int, help="Required with --train: explicit separate seed stream; not automatically a fresh holdout")
    exercise_build.add_argument("--training-generations", type=int, help="With --train; default 2, never shorten the task to fit")
    exercise_build.add_argument("--training-population", type=int, help="With --train; default 8")
    exercise_build.add_argument("--training-policy-family", choices=("phase-residual", "feedback-residual", "feedback-residual-split-limbs"),
        help="With --train: phase corrections (default), joint feedback, or separate arm/leg feedback")
    exercise_build.add_argument("--evaluation-trials", type=int, help="With --train; default 3 changed starts")
    exercise_build.add_argument("--evaluation-jitter", type=float, help="With --train; human-only start variation, default 0.10")
    exercise_build.add_argument("--out", required=True, help="New directory for all inspectable stages and the lab")
    exercise_build.add_argument("--max-seconds", type=float,
                                help="Cooperative total bound: defaults 90s saved, 270s raw or training, 450s both; selective-crop uses 450s or 600s with training")
    exercise_inspect = commands.add_parser("shared-exercise-inspect", help="Create an offline step-by-step report from a saved exercise build, including rejections")
    exercise_inspect.add_argument("--run", required=True, help="Existing sealed shared-exercise-build output; never modified")
    exercise_inspect.add_argument("--out", required=True, help="New self-contained inspection directory outside the source")
    exercise_inspect.add_argument("--max-seconds", type=float, default=30)
    exercise_inspect.add_argument("--pixel-support", help="Optional sealed local image-patch diagnostic; does not rerun or repair source poses")
    pixel_check = commands.add_parser("shared-exercise-pixel-check", help="Compare saved source PNG motion with original pose estimates; no video decode or inference")
    pixel_check.add_argument("--pose-run", required=True, help="Sealed original raw pose evidence directory")
    pixel_check.add_argument("--extraction-run", required=True, help="Sealed source-frame-pixels-v1 receipt and two to four original PNGs")
    pixel_check.add_argument("--out", required=True, help="New diagnostic companion; never modifies the original evidence")
    pixel_check.add_argument("--max-seconds", type=float, default=5, help="Cooperative total bound, at most five seconds")
    source_frames = commands.add_parser("shared-exercise-frames", help="Extract two to four original frames for saved-pose inspection; no pose inference")
    source_frames.add_argument("--pose-run", required=True, help="Sealed original raw pose evidence with its exact source video")
    source_frames.add_argument("--frames", nargs="+", type=int, required=True, help="Two to four increasing original frame IDs")
    source_frames.add_argument("--out", required=True, help="New source-frame-pixels-v1 extraction directory")
    source_frames.add_argument("--max-seconds", type=float, default=30)
    sw_construct = commands.add_parser("shared-world-reconstruct", help="Construct source-linked initial equipment proxies; no API calls or recovered grasp claim")
    sw_construct.add_argument("--run", required=True, help="Sealed source-qualified body/shared-world base")
    sw_construct.add_argument("--proposal-run", required=True, help="Sealed ready geometry proposal")
    sw_construct.add_argument("--evidence-run", required=True, help="Sealed equipment observation export")
    sw_construct.add_argument("--source-mapping", action="append", default=[], help="Verified original/derivative bundle; repeat for distinct derivatives")
    sw_construct.add_argument("--target-object", required=True, help="Authored task target ID from the observation document")
    sw_construct.add_argument("--frame-translation", nargs=2, type=float, metavar=("DX", "DY"),
                              help="Explicit positive XY coordinate-frame translation; preserves body/floor but changes numerical bounds")
    sw_construct.add_argument("--full-reference", action="store_true",
                              help="Construct a continuous full-reference v2 task instead of the initial window")
    sw_construct.add_argument("--out", required=True, help="New sealed construction output outside the inputs")
    sw_geometry = commands.add_parser("shared-world-geometry", help="Admit an explicitly authored compound-shape alternative; no recovered geometry or physical-accuracy claim")
    sw_geometry.add_argument("--run", required=True, help="Complete sealed shared-world candidate to preserve")
    sw_geometry.add_argument("--config", required=True, help="Versioned compound replacements pinned to the canonical base scene hash")
    sw_geometry.add_argument("--out", required=True, help="New sealed hypothesis directory outside both inputs")
    sw_geometry.add_argument("--max-seconds", type=float, default=30)
    sw_shape = commands.add_parser("shared-world-shape", help="Propose conditional solid shapes from saved silhouettes and compare full-motion clearance")
    sw_shape.add_argument("--run", required=True, help="Sealed source-qualified shared-world candidate")
    sw_shape.add_argument("--masks-run", required=True, help="Sealed segmentation and decoded-mask evidence for the same source")
    sw_shape.add_argument("--object-id", action="append", required=True, help="Rigid world ID; repeat for up to four objects")
    sw_shape.add_argument("--source-frame", type=int, required=True, help="Exact original construction-frame index; no frame search")
    sw_shape.add_argument("--conditional-exterior", action="store_true", help="Explicitly assume clear-row exterior may narrow solid bands; not measured empty space")
    sw_shape.add_argument("--task", help="Frozen exercise task to preserve apart from the candidate scene hash")
    sw_shape.add_argument("--out", required=True, help="New directory for original geometry, proposals, admission and clearance comparison")
    sw_shape.add_argument("--max-seconds", type=float, default=60)
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
    exercise_fullspan = commands.add_parser("shared-exercise-fullspan",
        help="Prepare an explicit full-reference task without clipping, retiming, or training")
    exercise_fullspan.add_argument("--run", required=True, help="Sealed shared-world-v1 scene with its exercise task")
    exercise_fullspan.add_argument("--out", required=True, help="New full-reference candidate; original files remain unchanged")
    exercise_fullspan.add_argument("--max-seconds", type=float, default=30)
    exercise_lab = commands.add_parser("shared-exercise-lab",
        help="Prepare an immutable human-only exercise lab with passive equipment; no server or training")
    exercise_lab.add_argument("--run", required=True, help="Sealed shared-world candidate or ready lab")
    exercise_lab.add_argument("--out", required=True, help="New lab directory outside the immutable input")
    exercise_lab.add_argument("--learning", help="Optional sealed human-only policy for this exact scene and task")
    exercise_lab.add_argument("--evaluation", help="Optional separate comparison for the exact installed policy; requires --learning")
    exercise_lab.add_argument('--reference-trial', help='Completed camera fitted-reference trial; attach its exact non-learned controller to a new lab')
    exercise_lab.add_argument('--reference-manifest-sha256', help='Optional exact completed reference-trial seal')
    exercise_lab_choice = exercise_lab.add_mutually_exclusive_group()
    exercise_lab_choice.add_argument("--root-assistance", action="store_true", help="Enable three explicit virtual root controls")
    exercise_lab_choice.add_argument("--task", help="Exact task JSON for this scene profile; otherwise preserve a sealed input task before using defaults")
    exercise_lab.add_argument("--max-seconds", type=float, default=60)
    shared_feet = commands.add_parser("shared-feet-build",
        help="Add source-bound feet to one existing exercise world; preserve its original task and evidence")
    shared_feet.add_argument("--run", required=True, help="Sealed v2/v3/v4 shared world with an explicit exercise task")
    shared_feet.add_argument("--foot-reference", required=True, help="Exact body-bound foot-reference JSON")
    shared_feet.add_argument("--task", help="Explicit original task; otherwise retain the sealed input task")
    shared_feet.add_argument("--out", required=True, help="New candidate directory; no input is overwritten")
    shared_feet.add_argument("--max-seconds", type=float, default=60)
    camera_shared = commands.add_parser("camera-shared-build",
        help="Import an unchanged sealed two-view camera body into the shared simulator; no lab or training")
    camera_shared.add_argument("--run", required=True, help="Sealed grounded two-view-root camera-body run")
    camera_shared.add_argument("--out", required=True, help="New immutable candidate directory")
    camera_shared.add_argument("--max-seconds", type=float, default=30)
    force_check = commands.add_parser("motion-forces", help="Explain source-motion motor limits in a new offline report; no simulation steps or training")
    force_check.add_argument("--run", required=True, help="Sealed assisted human-only camera-world run, usually LAB/run")
    force_check.add_argument("--out", required=True, help="New report directory outside the immutable run")
    force_check.add_argument("--max-seconds", type=float, default=30, help="Cooperative processing limit, at most 60 seconds")
    support_build = commands.add_parser("shared-support-build",
        help="Bind a sealed video body to explicitly authored static supports and load requirements")
    support_build.add_argument("--body-source", required=True)
    support_build.add_argument("--config", required=True, help="Source-pinned shared-static-support-construction-v1 JSON")
    support_build.add_argument("--out", required=True, help="New immutable candidate directory")
    support_build.add_argument("--max-seconds", type=float, default=30)
    support_fixture = commands.add_parser("shared-support-fixture",
        help="Create an unexecuted analytical support fixture, not a reconstructed video")
    support_fixture.add_argument("--out", required=True)
    support_fixture.add_argument("--lowered", action="store_true", help="Predeclared support-height counterfactual")
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
    exercise_train = commands.add_parser("shared-exercise-train", help="Train a bounded human-only policy; freeze it before separate evaluation")
    exercise_train.add_argument("--run", required=True, help="Sealed shared-world candidate or ready lab")
    exercise_train.add_argument("--out", required=True, help="New immutable training directory")
    exercise_train_choice = exercise_train.add_mutually_exclusive_group()
    exercise_train_choice.add_argument("--root-assistance", action="store_true", help="Enable three explicitly assisted root channels")
    exercise_train_choice.add_argument("--task", help="Exact compatible exercise task JSON; preserve its source, interval, controls and gates")
    exercise_train.add_argument("--policy-family", choices=["feedback", "phase-residual", "feedback-residual", "feedback-residual-split-limbs", "camera-feedback"], default="feedback",
                                help="Feedback gains (default), phase/joint corrections, separate arm/leg feedback, or camera-task corrections")
    exercise_train.add_argument("--seed", type=int, default=1729)
    exercise_train.add_argument("--generations", type=int, help="Default: 3 for feedback, 6 for phase-residual, 2 for other families")
    exercise_train.add_argument("--population", type=int, help="Default: 12 for phase-residual, 8 otherwise")
    exercise_train.add_argument("--max-seconds", type=float, default=60)
    exercise_refine = commands.add_parser("shared-exercise-refine", help="Refine verified residual training by prioritizing unchanged task limits; no evaluation data")
    exercise_refine.add_argument("--learning", required=True, help="Sealed compatible two-generation residual training package")
    exercise_refine.add_argument("--out", required=True, help="New versioned feasibility-first policy and training evidence")
    exercise_refine.add_argument("--max-seconds", type=float, default=60)
    exercise_select = commands.add_parser("shared-exercise-select", help="Select a saved residual candidate meeting every training-task gate; no retraining")
    exercise_select.add_argument("--learning", required=True, help="Complete sealed residual-policy training package")
    exercise_select.add_argument("--out", required=True, help="New selected-policy package; original candidates and policy stay unchanged")
    exercise_select.add_argument("--max-seconds", type=float, default=15)
    exercise_policy_eval = commands.add_parser("shared-exercise-policy-evaluate", help="Compare a frozen human-only policy with coast/random/authored on separate changed starts")
    exercise_policy_eval.add_argument("--learning", required=True, help="Verified immutable human-only training directory")
    exercise_policy_eval.add_argument("--out", required=True, help="New evaluation directory outside training inputs")
    exercise_policy_eval.add_argument("--seed", type=int, default=924017)
    exercise_policy_eval.add_argument("--trials", type=int, default=3)
    exercise_policy_eval.add_argument("--jitter-scale", type=float, default=.10,
                                     help="Human-only start perturbation; equipment retains its scene pose and velocity")
    exercise_policy_eval.add_argument("--max-seconds", type=float, default=30)
    avatar_import = commands.add_parser("avatar-import", help="Validate and seal a local rigged avatar without changing physics")
    avatar_source = avatar_import.add_mutually_exclusive_group(required=True)
    avatar_source.add_argument("--source", help="Factory package directory containing avatar.json")
    avatar_source.add_argument("--model", help="Local rigged GLB/glTF; requires explicit --metadata")
    avatar_import.add_argument("--metadata", help="JSON with identity, rights, provenance and explicit body-bone mapping")
    avatar_import.add_argument("--out", required=True, help="New sealed output directory; never overwritten")
    qualify = commands.add_parser("qualify", help="Check sealed environments: legal actions, complete intervals and exact replay")
    qualify_source = qualify.add_mutually_exclusive_group(required=True)
    qualify_source.add_argument("--run", help="One existing sealed run or ready lab; never modified")
    qualify_source.add_argument("--cases", help="JSON list of up to five cases; relative inputs use this file's directory")
    qualify.add_argument("--out", required=True, help="New qualification directory outside the inputs")
    qualify.add_argument("--controller", choices=("coast", "authored"), help="Single-run baseline; default coast, never a learned policy")
    qualify.add_argument("--seed", type=int, help="Single-run deterministic reset seed; default 1729")
    qualify.add_argument("--probe-steps", type=int, help="Single-run changed-action prefix; default 6")
    qualify.add_argument("--probe-channel", type=int, help="Single-run motor channel from capabilities; default 0")
    qualify.add_argument("--manifest-sha256", help="Require this exact single-run manifest identity before execution")
    qualify.add_argument("--max-seconds", type=int, default=60)
    qualify.add_argument("--max-steps", type=int, default=12000, help="Total simulation steps including exact replay")
    qualify.add_argument("--max-output-mib", type=int, default=64)
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
        elif args.command == "exercise-perceive":
            from weft.exercise_perception_pipeline import prepare_exercise_perception
            result = prepare_exercise_perception(args.video, args.analysis, args.out, pose_model=args.pose_model,
                body_source=args.body_source, saved_tracking=args.saved_tracking, execute_tracking=args.execute_tracking,
                max_meta_cost_usd=args.meta_budget, max_seconds=args.max_seconds)
        elif args.command == "tracking-view":
            from weft.segmentation_lab import prepare_tracking_view
            result = prepare_tracking_view(args.run, args.video, args.out)
        elif args.command == "tracking-masks":
            from weft.segmentation_masks import export_masks
            result = export_masks(args.run, args.video, args.out,
                                  **({"retain_binary": True} if args.retain_binary else {}))
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
        elif args.command == "stereo-ground":
            from weft.stereo_ground_package import prepare_stereo_ground
            result = prepare_stereo_ground(args.camera, args.image, args.annotations, args.out)
        elif args.command == "floor-calibrate":
            from weft.floor_target_calibration_pipeline import prepare_floor_target_calibration_package
            labels = {"inputs": "Read source and measurements", "calibration": "Check camera and floor alternatives",
                      "report": "Explain geometry and uncertainty", "seal": "Save calibration evidence"}
            result = prepare_floor_target_calibration_package(args.observation, args.out,
                source_sha256=args.source_sha256, request=args.request, measurements=args.measurement,
                camera=args.camera, ground=args.ground, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "surface-anchors":
            from weft.source_surface_pipeline import prepare_source_surface_anchors
            labels = {"inputs": "Preserve source and assumptions", "anchors": "Resolve conditional 3D points", "report": "Save evidence and limits"}
            result = prepare_source_surface_anchors(args.observation, args.out, source_sha256=args.source_sha256,
                camera=args.camera, surfaces=args.surfaces, annotations=args.annotations, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "surface-inputs":
            if args.resume and any(value is not None for value in (args.source_sha256, args.camera, args.body)):
                raise ValueError("--resume keeps exact saved inputs; do not mix --source-sha256, --camera or --body")
            if not args.resume and (args.request is not None or args.manifest_sha256 is not None):
                raise ValueError("--request and --manifest-sha256 require --resume of a sealed surface collection")
            if args.resume and args.selection is None and args.request is None:
                raise ValueError("--resume requires an explicit edited --selection or --request; saved results are not rerun")
            from weft.exercise_surface_pipeline import prepare_exercise_surface_collection
            labels = {"inputs": "Read saved source evidence", "selection": "Prepare source-linked choices",
                      "qualification": "Check supplied surface inputs", "report": "Save input worksheet and report"}
            result = prepare_exercise_surface_collection(args.observation or [], args.out,
                source_sha256=args.source_sha256, camera=args.camera, body=args.body,
                selection=args.selection, request=args.request, resume=args.resume,
                expected_manifest_sha256=args.manifest_sha256, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "exercise-equipment-evidence":
            from weft.exercise_equipment_evidence import prepare_exercise_equipment_evidence
            labels = {"inputs": "Read original saved pose and equipment", "pose": "Qualify original pose estimates",
                      "silhouette": "Compare source points with cached masks", "seal": "Save complete image evidence"}
            result = prepare_exercise_equipment_evidence(args.pose_run, args.equipment_run, args.masks_run, args.out,
                human_id=args.human_id, equipment_ids=args.equipment_id,
                minimum_confidence=args.minimum_confidence, uncertainty_px=args.uncertainty_px,
                uncertainty_method=args.uncertainty_method, uncertainty_note=args.uncertainty_note,
                source_equivalence_run=args.source_equivalence_run,
                expected_pose_manifest_sha256=args.pose_manifest_sha256,
                expected_equipment_manifest_sha256=args.equipment_manifest_sha256,
                expected_masks_manifest_sha256=args.masks_manifest_sha256, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "equipment-evidence-report":
            from weft.equipment_silhouette_inspection import prepare_equipment_silhouette_inspection
            labels = {"inputs": "Read saved image evidence", "report": "Build frame-by-frame inspection",
                      "seal": "Save report and uncertainty"}
            result = prepare_equipment_silhouette_inspection(args.run, args.out,
                expected_manifest_sha256=args.manifest_sha256, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "equipment-support":
            from weft.equipment_support_workflow import prepare_equipment_support
            labels = {"inputs": "Read saved pose and equipment evidence", "selection": "Check original equipment candidates",
                      "hands": "Compare named hand points with equipment",
                      "report": "Explain kept, excluded and unknown candidates", "seal": "Save the comparison"}
            result = prepare_equipment_support(args.run, args.out, fixed_camera=args.fixed_camera,
                expected_manifest_sha256=args.manifest_sha256, max_seconds=args.max_seconds,
                **({"hand_points": True} if args.hand_points else {}),
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "body-lab":
            from weft.whole_body_lab import prepare_whole_body_lab
            result = prepare_whole_body_lab(args.source_lab, args.learning, args.out)
        elif args.command == "body-reference-report":
            from weft.whole_body_reference_report import prepare_whole_body_reference_report
            result = prepare_whole_body_reference_report(args.run, args.out,
                expected_manifest_sha256=args.manifest_sha256, max_seconds=args.max_seconds)
        elif args.command == "body-reference-propose":
            if (args.objective in {"surface-intervals", "camera-surface-intervals", "camera-multiview-surface-intervals"}) != (args.surface_config is not None):
                raise ValueError("Surface-interval objectives require --surface-config; other objectives do not accept it")
            if (args.objective in {"camera-surface-intervals", "camera-multiview-surface-intervals"}) != (args.camera_image_scale is not None):
                raise ValueError("Camera surface objectives require --camera-image-scale; other objectives do not accept it")
            if (args.objective == "camera-multiview-surface-intervals") != (args.secondary_view_config is not None):
                raise ValueError("camera-multiview-surface-intervals requires --secondary-view-config; other objectives do not accept it")
            if args.objective == "camera-root":
                if bool(args.camera) == (args.camera_focal_px is not None):
                    raise ValueError("camera-root requires exactly one of --camera or --camera-focal-px")
                from weft.body_camera_root_pipeline import prepare_camera_root_proposal
                def camera_progress(stage, detail):
                    print("weft body proposal: " + json.dumps({"stage": stage, **detail}, allow_nan=False),
                          file=sys.stderr, flush=True)
                result = prepare_camera_root_proposal(args.run, args.camera, args.out,
                    expected_manifest_sha256=args.manifest_sha256,
                    fit_seconds=10 if args.fit_seconds is None else args.fit_seconds,
                    camera_focal_px=args.camera_focal_px,
                    max_seconds=args.max_seconds, progress=camera_progress)
            else:
                if args.camera or args.camera_focal_px is not None:
                    raise ValueError("Camera options are only supported with --objective camera-root")
                from weft.whole_body_reference_pipeline import prepare_whole_body_reference_proposal
                result = prepare_whole_body_reference_proposal(args.run, args.out,
                    expected_manifest_sha256=args.manifest_sha256,
                    proposal_objective={"image-first": "image_v1", "floor-first": "floor_first_v1",
                                        "surface-intervals": "surface_interval_v1",
                                        "camera-surface-intervals": "camera_surface_interval_v1",
                                        "camera-multiview-surface-intervals": "camera_multiview_surface_interval_v1"}[args.objective],
                    fit_seconds=30 if args.fit_seconds is None else args.fit_seconds, max_seconds=args.max_seconds,
                    **({"surface_config": args.surface_config} if args.surface_config is not None else {}),
                    **({"camera_image_scale": args.camera_image_scale} if args.camera_image_scale is not None else {}),
                    **({"secondary_view_config": args.secondary_view_config} if args.secondary_view_config is not None else {}))
        elif args.command == "body-reference-trial":
            from weft.shared_exercise_reference_trial import prepare_reference_trial

            def reference_progress(stage, detail):
                print("weft reference trial: " + json.dumps({"stage": stage, **detail}, allow_nan=False),
                      file=sys.stderr, flush=True)

            result = prepare_reference_trial(args.proposal, args.run, args.task, args.out,
                seed=args.seed, max_seconds=args.max_seconds,
                expected_proposal_manifest_sha256=args.proposal_manifest_sha256,
                expected_scene_manifest_sha256=args.scene_manifest_sha256,
                expected_task_sha256=args.task_sha256, progress=reference_progress)
        elif args.command == 'camera-body':
            from weft.camera_body_lab import prepare_camera_body_lab
            result = prepare_camera_body_lab(args.run, args.camera, args.out, ground_path=args.ground,
                root_view_path=args.root_view, body_view=args.body_view, partial_segments=args.partial_segments,
                additional_view_paths=args.additional_view, limb_fit=args.fit_limbs, partial_pixel_fit=args.fit_partial_pixels,
                pixel_uncertainty_path=args.partial_pixel_uncertainty, partial_depth_branches=args.partial_depth_branches,
                joint_pixel_uncertainty_path=args.fit_joint_pixels, control_joint=args.control_joint)
        elif args.command == "spatial-fixture":
            from weft.spatial_lab import write_sealed_spatial_fixture
            result = write_sealed_spatial_fixture(args.out)
        elif args.command == "spatial-scene":
            from weft.spatial_scene_pipeline import prepare_spatial_scene
            result = prepare_spatial_scene(args.run, args.config, args.out,
                scene_artifact=args.scene_artifact, expected_manifest_sha256=args.manifest_sha256,
                max_seconds=args.max_seconds)
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
                max_seconds=args.max_seconds,
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
                frame_translation=args.frame_translation, full_reference=args.full_reference)
        elif args.command == "shared-world-propose":
            from weft.equipment_proposal import prepare_equipment_proposal
            result = prepare_equipment_proposal(args.run, args.evidence_run, args.config, args.out,
                source_mappings=args.source_mapping, max_seconds=args.max_seconds)
        elif args.command == "shared-world-geometry":
            from weft.shared_world_geometry import prepare_shared_world_geometry
            result = prepare_shared_world_geometry(args.run, args.config, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-world-shape":
            from weft.silhouette_pipeline import prepare_silhouette_world
            result = prepare_silhouette_world(args.run, args.masks_run, args.out, task_path=args.task,
                object_ids=args.object_id, source_frame=args.source_frame,
                conditional_exterior=args.conditional_exterior, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-config":
            if args.resume is not None:
                if args.evidence_run is not None or args.source_mapping or args.request is None:
                    raise ValueError("--resume requires --request and reuses its copied sources; omit --evidence-run and --source-mapping")
            elif args.evidence_run is None or args.request is not None or args.manifest_sha256 is not None:
                raise ValueError("Discovery requires --human-run and --evidence-run; --request and --manifest-sha256 require --resume")
            from weft.shared_exercise_config import prepare_shared_exercise_config
            labels = {"inputs": "Read saved body and equipment", "availability": "Explain usable and missing frames",
                      "request": "Check explicit setup choices", "report": "Prepare the setup page", "seal": "Save the setup package"}
            result = prepare_shared_exercise_config(args.human_run, args.evidence_run, args.out,
                source_mappings=args.source_mapping, resume=args.resume, request=args.request,
                expected_manifest_sha256=args.manifest_sha256, max_seconds=args.max_seconds,
                progress=lambda stage, detail: print(f"{labels.get(stage, stage)}: {detail}", file=sys.stderr, flush=True))
        elif args.command == "shared-exercise-auto-build":
            from weft.equipment_autoproposal import prepare_automatic_equipment_scene
            result = prepare_automatic_equipment_scene(args.run, args.evidence_run, args.out,
                source_mappings=args.source_mapping, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-alternatives":
            from weft.equipment_conditional_assembly import prepare_conditional_equipment_scene
            result = prepare_conditional_equipment_scene(args.run, args.evidence_run, args.out,
                source_mappings=args.source_mapping, max_seconds=args.max_seconds, max_candidates=args.max_candidates)
        elif args.command == "shared-exercise-build":
            from weft.exercise_scene_pipeline import prepare_exercise_scene
            if args.native_timing and args.video is None:
                raise ValueError("--native-timing requires --video; saved bodies and worlds retain their own clocks")
            if args.geometry_run is not None:
                overrides = [name for name in ("evidence_run", "config", "human_run", "video", "pose_model", "pose_mode",
                    "pose_identity", "pose_recovery", "source_note", "body_scale", "static_supports", "training_seed",
                    "evaluation_seed", "training_generations", "training_population", "training_policy_family",
                    "evaluation_trials", "evaluation_jitter") if getattr(args, name) is not None]
                overrides.extend(name for name in ("source_mapping", "root_assistance", "full_reference", "person_only", "train")
                                 if getattr(args, name))
                if overrides:
                    raise ValueError("--geometry-run preserves its saved scene/task; remove supplied overrides: " +
                                     ", ".join("--" + name.replace("_", "-") for name in overrides))
            elif args.geometry_manifest_sha256 is not None:
                raise ValueError("--geometry-manifest-sha256 requires --geometry-run")
            elif args.static_supports is not None:
                if (args.run is not None or args.evidence_run is not None or args.config is not None
                        or args.source_mapping or args.body_scale is not None or args.root_assistance):
                    raise ValueError("--static-supports owns scale/root assumptions and requires --video or --human-run, without equipment/config/mapping overrides")
            elif args.person_only:
                if args.run is not None or args.evidence_run is not None or args.config is not None or args.source_mapping:
                    raise ValueError("--person-only requires --video or a sealed exercise-body --human-run, without --run, equipment, config or mappings")
            elif args.evidence_run is None or args.config is None:
                raise ValueError("Provide --person-only, --static-supports or both --evidence-run and --config")
            labels = {"body_seed": "Prepare source body", "equipment_priors": "Resolve equipment assumptions", "shared_world": "Build shared world",
                      "interactive_lab": "Prepare interactive lab", "body_source": "Read exercise video and prepare body",
                      "body_source_progress": "Observe body", "learning_workflow": "Train and compare controllers",
                      "learning_training": "Train joint controller", "learning_training_selection": "Select using training results",
                      "learning_independent_evaluation": "Compare frozen controller on separate starts",
                      "learning_interactive_lab": "Prepare trained interactive lab",
                      "source_pose_proposals": "Observe full frames and selected crops",
                      "source_pose_recovery": "Verify source-only pose recovery", "source_body_source": "Prepare recovered body evidence",
                      "support_priors": "Bind authored supports to the body", "geometry_source": "Reuse saved conditional geometry"}
            train_options = {key: value for key, value in (
                ("training_seed", args.training_seed), ("evaluation_seed", args.evaluation_seed),
                ("generations", args.training_generations), ("population", args.training_population),
                ("policy_family", args.training_policy_family),
                ("trials", args.evaluation_trials), ("jitter_scale", args.evaluation_jitter)) if value is not None}
            if train_options and not args.train:
                raise ValueError("Training settings require --train; no training is enabled by default")
            if args.train and args.evaluation_seed is None:
                raise ValueError("--train requires an explicit --evaluation-seed, separate from training")
            result = prepare_exercise_scene(args.run, args.evidence_run, args.config, args.out,
                human_run=args.human_run, body_scale=args.body_scale,
                video=args.video, pose_model=args.pose_model, pose_mode=args.pose_mode,
                pose_identity=args.pose_identity, source_note=args.source_note, pose_recovery=args.pose_recovery,
                **({"native_timing": True} if args.native_timing else {}),
                train_settings=train_options if args.train else None,
                source_mappings=args.source_mapping, root_assistance=args.root_assistance,
                full_reference=args.full_reference,
                max_seconds=args.max_seconds,
                **({"geometry_run": args.geometry_run, "geometry_manifest_sha256": args.geometry_manifest_sha256}
                   if args.geometry_run is not None else {}),
                **({"scene_scope": "human_static_supports", "static_supports": args.static_supports}
                   if args.static_supports is not None else {"scene_scope": "human_only"} if args.person_only else {}),
                progress=lambda row: print(f"{labels.get(row['stage'], row['stage'].replace('_', ' '))}: {row['status'].replace('_', ' ')}",
                                           file=sys.stderr, flush=True))
        elif args.command == "shared-exercise-inspect":
            from weft.exercise_build_report import prepare_exercise_build_report
            result = prepare_exercise_build_report(args.run, args.out, max_seconds=args.max_seconds,
                **({"pixel_support": args.pixel_support} if args.pixel_support is not None else {}))
        elif args.command == "shared-exercise-pixel-check":
            from weft.pose_pixel_support_package import prepare_pose_pixel_support
            result = prepare_pose_pixel_support(args.pose_run, args.extraction_run, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-frames":
            from weft.source_frame_pixels import prepare_source_frame_pixels
            result = prepare_source_frame_pixels(args.pose_run, args.out, frames=args.frames, max_seconds=args.max_seconds)
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
        elif args.command == "shared-exercise-fullspan":
            from weft.shared_exercise_fullspan import prepare_fullspan_shared_exercise
            result = prepare_fullspan_shared_exercise(args.run, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-evaluate":
            from weft.shared_exercise_run import evaluate_shared_exercise_run
            result = evaluate_shared_exercise_run(args.run, args.out, root_assistance=args.root_assistance,
                task_path=args.task, seed=args.seed, jitter=args.jitter, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-lab":
            from weft.shared_exercise_lab import prepare_shared_exercise_lab
            result = prepare_shared_exercise_lab(args.run, args.out, root_assistance=args.root_assistance,
                task_path=args.task, policy_run=args.learning, evaluation_run=args.evaluation,
                max_seconds=args.max_seconds,
                **({'reference_trial': args.reference_trial, 'reference_manifest_sha256': args.reference_manifest_sha256}
                   if args.reference_trial is not None or args.reference_manifest_sha256 is not None else {}))
        elif args.command == "shared-feet-build":
            from weft.shared_exercise_feet_candidate import prepare_shared_feet_candidate
            result = prepare_shared_feet_candidate(args.run, args.foot_reference, args.out,
                                                  task_path=args.task, max_seconds=args.max_seconds)
        elif args.command == "camera-shared-build":
            from weft.camera_shared_candidate import prepare_camera_shared_candidate
            result = prepare_camera_shared_candidate(args.run, args.out, max_seconds=args.max_seconds)
        elif args.command == "motion-forces":
            from weft.reference_dynamics_run import run_motion_force_check
            result = run_motion_force_check(args.run, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-support-build":
            from weft.shared_world_support import construct_static_support_run
            result = construct_static_support_run(args.body_source, args.config, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-support-fixture":
            from weft.shared_support_fixture import write_static_support_fixture
            result = write_static_support_fixture(args.out, lowered=args.lowered)
        elif args.command == "shared-exercise-train":
            if args.policy_family == "camera-feedback":
                if args.root_assistance:
                    raise ValueError("Camera learning preserves an existing task; supply its exact --task or sealed task instead of --root-assistance")
                from weft.camera_shared_learning import train_camera_shared_policy as train_policy
                generations, population = 2, 8
            elif args.policy_family in ("feedback-residual", "feedback-residual-split-limbs"):
                from weft.shared_exercise_feedback_learning import train_shared_exercise_feedback_policy as train_policy
                generations, population = 2, 8
            elif args.policy_family == "phase-residual":
                from weft.shared_exercise_residual_learning import train_shared_exercise_residual_policy as train_policy
                generations, population = 6, 12
            else:
                from weft.shared_exercise_learning import train_shared_exercise_policy as train_policy
                generations, population = 3, 8
            assistance = {} if args.policy_family == "camera-feedback" else {"root_assistance": args.root_assistance}
            if args.policy_family == "feedback-residual-split-limbs":
                from weft.shared_exercise_feedback_residual import SPLIT_POLICY_PROFILE
                assistance["policy_profile"] = SPLIT_POLICY_PROFILE
            result = train_policy(args.run, args.out, task_path=args.task, **assistance, seed=args.seed,
                generations=generations if args.generations is None else args.generations,
                population=population if args.population is None else args.population, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-refine":
            from weft.shared_exercise_feasibility_learning import train_shared_exercise_feasibility_policy
            result = train_shared_exercise_feasibility_policy(args.learning, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-select":
            from weft.shared_exercise_policies import select_policy
            result = select_policy(args.learning, args.out, max_seconds=args.max_seconds)
        elif args.command == "shared-exercise-policy-evaluate":
            from weft.shared_exercise_lab import _json
            from weft.shared_world_packaging import read_package_file

            policy = _json(read_package_file(Path(args.learning), "policy.json", max_bytes=64 * 1024))
            if isinstance(policy, dict) and policy.get("profile") == "camera-shared-policy-v1":
                from weft.camera_shared_learning import evaluate_camera_shared_policy as evaluate_policy
            else:
                # Preserve the runtime-pinned legacy policy dispatcher unchanged.
                from weft.shared_exercise_policies import evaluate_policy
            result = evaluate_policy(args.learning, args.out, seed=args.seed,
                trials=args.trials, jitter_scale=args.jitter_scale, max_seconds=args.max_seconds)
        elif args.command == "portal":
            from weft.portal import launch_portal
            result = launch_portal(args)
        elif args.command == "portal-import":
            from weft.portal_import import import_lab
            result = import_lab(args.lab, args.root, name=args.name)
        elif args.command == "portal-attach-evidence":
            from weft.portal_import import attach_existing_import
            result = attach_existing_import(args.root, args.project_id, args.body_source, max_seconds=args.max_seconds)
        elif args.command == "portal-equipment-register":
            from weft.portal_equipment_candidates import register_equipment_source
            result = register_equipment_source(args.root, source_sha256=args.source_sha256, base_run=args.run,
                evidence_run=args.evidence_run, source_mappings=args.source_mapping, label=args.label)
        elif args.command == "openai-budget":
            from weft.openai_access import OpenAIBudget
            result = OpenAIBudget().configure(args.total_usd, note="Administrator explicitly configured the portal lifetime allowance through the CLI")
        elif args.command == "lab":
            from weft.lab import launch_lab
            result = launch_lab(args)
        elif args.command == "lab-console":
            from weft.lab_console import console
            result = console(args.resume, seed=args.seed)
        elif args.command == "avatar-import":
            from weft.avatar_assets import import_avatar_model, import_avatar_package
            if args.source:
                if args.metadata:
                    raise ValueError("--source already contains metadata; do not combine it with --metadata")
                result = import_avatar_package(args.source, args.out)
            else:
                if not args.metadata:
                    raise ValueError("A custom model needs --metadata with an explicit body-bone mapping")
                with Path(args.metadata).open("rb") as stream:
                    raw = stream.read(128 * 1024 + 1)
                if len(raw) > 128 * 1024:
                    raise ValueError("Avatar metadata exceeds 128 KiB")
                result = import_avatar_model(args.model, json.loads(raw), args.out)
        elif args.command == "qualify":
            from weft.runtime_qualification import _json, qualify_runtimes
            options = {key: getattr(args, key) for key in ("controller", "seed", "probe_steps", "probe_channel", "manifest_sha256")
                       if getattr(args, key) is not None}
            if args.cases:
                if options:
                    raise ValueError("With --cases, put each case's controller, seed, probe and manifest settings in that file")
                path = Path(args.cases)
                with path.open("rb") as stream:
                    raw = stream.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("Qualification case plan exceeds 1 MiB")
                cases = _json(raw)
                if not isinstance(cases, list):
                    raise ValueError("Qualification case plan must be a JSON list")
                for case in cases:
                    if isinstance(case, dict) and isinstance(case.get("input"), str):
                        case["input"] = str(path.parent / case["input"])
            else:
                cases = [{"id": "environment", "input": args.run, **options}]
            result = qualify_runtimes(cases, args.out, max_seconds=args.max_seconds,
                max_steps=args.max_steps, max_output_bytes=args.max_output_mib * 1024 * 1024)
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
                from weft.shared_exercise_lab import verify_shared_exercise_action_replay
                result["action_replay"] = verify_shared_exercise_action_replay(args.out if args.recover_incomplete else args.directory)
        elif args.command == "demo":
            from weft.demo import create_demo
            result = create_demo(args.out)
        elif args.command == "fetch-pose-model":
            from weft.exercise_vision import fetch_pose_model
            result = {"path": str(fetch_pose_model(args.out, variant=args.variant))}
        elif args.command == "exercise-pose-prepare":
            from weft.pose_crop_pipeline import prepare_pose_crop_proposals
            destination = prepare_pose_crop_proposals(args.video, args.out, pose_model=args.pose_model,
                source_note=args.source_note, max_seconds=args.max_seconds,
                progress=lambda name, detail: print(detail.get("message", name.replace("_", " ")),
                                                    file=sys.stderr, flush=True))
            result = {"profile": "exercise-pose-crop-proposals-v1", "status": "source_evidence_only",
                "directory": str(destination), "simulation_ready": False,
                "next_step": "Use exercise-pose-recover with this package and the original video"}
        elif args.command == "exercise-pose-recover":
            from weft.pose_source_recovery import prepare_pose_source_recovery
            destination = prepare_pose_source_recovery(args.run, args.out, source_video=args.video,
                                                       max_seconds=args.max_seconds)
            result = {"profile": "exercise-pose-recovery-v1", "status": "source_evidence_only",
                "directory": str(destination), "simulation_ready": False,
                "new_inference": False, "next_step": "Use exercise-body --evidence-run on this preserved pose package"}
        elif args.command == "exercise-pose-recover-modes":
            from weft.pose_mode_source import prepare_pose_mode_recovery
            destination = prepare_pose_mode_recovery(args.primary_run, args.alternative_run, args.out,
                                                      source_video=args.video, max_seconds=args.max_seconds)
            result = {"profile": "exercise-pose-mode-recovery-v1", "status": "source_evidence_only",
                "directory": str(destination), "simulation_ready": False, "new_inference": False,
                "next_step": "Inspect retained decisions, then use exercise-body --evidence-run"}
        elif args.command == "exercise-body":
            from weft.exercise_body_source import prepare_exercise_body_source, prepare_exercise_body_source_from_evidence
            def body_progress(stage, detail):
                message = detail.get("message", stage.replace("_", " "))
                if stage == "tracking":
                    message += f" — {detail['decoded_frames']} decoded frames, {detail['sampled_frames']} sampled"
                print(message, file=sys.stderr, flush=True)
            if args.evidence_run:
                if args.native_timing:
                    raise ValueError("--native-timing requires --video; use --native-clock to bind saved evidence")
                if any(getattr(args, key) is not None for key in ("pose_model", "pose_mode", "pose_identity", "source_note", "max_pose_hypotheses", "person_region", "pose_sampling")):
                    raise ValueError("Saved body evidence inherits its exact source/model/options; overrides are not supported")
                destination = prepare_exercise_body_source_from_evidence(args.evidence_run, args.out,
                    max_seconds=30 if args.max_seconds is None else args.max_seconds, progress=body_progress,
                    **({"native_clock": args.native_clock} if args.native_clock is not None else {}))
            else:
                if args.native_clock is not None:
                    raise ValueError("--native-clock requires --evidence-run; use --native-timing to capture raw video timing")
                if not args.pose_model:
                    raise ValueError("--video requires --pose-model; fetch-pose-model can prepare the pinned local model")
                options = {key: getattr(args, key) for key in ("pose_mode", "pose_identity", "source_note", "max_pose_hypotheses", "person_region", "pose_sampling")
                           if getattr(args, key) is not None}
                if args.native_timing:
                    options["native_timing"] = True
                destination = prepare_exercise_body_source(args.video, args.out, pose_model=args.pose_model,
                    max_seconds=180 if args.max_seconds is None else args.max_seconds, progress=body_progress, **options)
            result = {"profile": "exercise-body-source-v1", "status": "construction_only",
                "directory": str(destination), "simulation_ready": False,
                "next_step": "Use shared-exercise-build --human-run with sealed equipment evidence and explicit camera/equipment configuration"}
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
        if args.command == "motion-forces":
            if result.get("status") != "diagnostic_only":
                parser.exit(1, f"weft: motion force check stopped; preserved output: {args.out}\n")
            print(f"Saved motion force report: {result['report_path']}", file=sys.stderr)
        if args.command == "lab-console" and result.get("recording", {}).get("export") == "failed":
            parser.exit(1, "weft: recording export failed; actions are preserved and recovery guidance is printed above\n")
        if args.command in {"shared-exercise-build", "shared-exercise-auto-build"} and result.get("status") != "ready":
            parser.exit(1, f"weft: exercise build {result.get('status', 'not_ready')}; preserved result: {Path(args.out) / 'result.json'}\n")
        if args.command == "shared-exercise-alternatives" and result.get("status") != "candidates_ready":
            parser.exit(1, f"weft: no admitted equipment candidate; inspect {Path(args.out) / 'result.json'}\n")
        if args.command == "shared-exercise-config":
            pin = result.get("manifest_sha256")
            if (result.get("profile") != "shared-exercise-config-package-v1" or result.get("sealed") is not True
                    or type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)
                    or any(type(result.get(key)) is not str or not result[key] for key in ("output_path", "report_path"))):
                parser.exit(1, f"weft: setup did not publish a complete package; inspect preserved output: {args.out}\n")
            if result.get("status") not in ("needs_input", "config_ready"):
                parser.exit(1, f"weft: setup {result.get('status', 'unavailable')}; open preserved report: {result['report_path']}\n")
            target = Path(result["output_path"])
            if result["status"] == "config_ready":
                inputs, options = result.get("build_inputs"), result.get("build_options")
                if (type(inputs) is not dict or set(inputs) != {"human_run", "evidence_run", "source_mappings", "config"}
                        or any(type(inputs[key]) is not str or not inputs[key] for key in ("human_run", "evidence_run", "config"))
                        or type(inputs["source_mappings"]) is not list or len(inputs["source_mappings"]) > 7
                        or any(type(path) is not str or not path for path in inputs["source_mappings"])
                        or type(options) is not dict or set(options) != {"body_scale", "root_assistance"}
                        or type(options["body_scale"]) not in (int, float) or not .5 <= options["body_scale"] <= 2
                        or type(options["root_assistance"]) is not bool or result.get("full_reference") is not True
                        or result.get("config_path") != inputs["config"]):
                    parser.exit(1, f"weft: ready setup lacks exact build inputs/options; inspect: {result['report_path']}\n")
                followup = [sys.executable, "-m", "weft", "shared-exercise-build", "--human-run", inputs["human_run"],
                    "--evidence-run", inputs["evidence_run"], "--config", inputs["config"],
                    "--body-scale", str(options["body_scale"]), "--full-reference",
                    "--out", str(target.with_name(target.name + "-build"))]
                for path in inputs["source_mappings"]: followup.extend(["--source-mapping", path])
                if options["root_assistance"]: followup.append("--root-assistance")
                instruction = "Configuration ready, not a validated world. Run the existing builder to check geometry and collisions:"
            else:
                followup = [sys.executable, "-m", "weft", "shared-exercise-config", "--resume", str(target),
                    "--manifest-sha256", pin, "--request", "PATH_TO_DOWNLOADED_REQUEST.json",
                    "--out", str(target.with_name(target.name + "-configured"))]
                instruction = "Download your edited request from the setup page; replace PATH_TO_DOWNLOADED_REQUEST.json below. Do not edit the sealed package:"
            print(f"Saved setup page: {result['report_path']}\n{instruction}\n" + shlex.join(followup), file=sys.stderr)
        if args.command == "body-reference-propose" and result.get("status") != "completed":
            if (result.get("error") or {}).get("stage") == "inspection" and result.get("proposal_manifest_sha256"):
                recovery = shlex.join([sys.executable, "-m", "weft", "body-reference-report",
                    "--run", str(Path(args.out) / "proposal"), "--manifest-sha256", result["proposal_manifest_sha256"],
                    "--out", str(Path(args.out).with_name(Path(args.out).name + "-report-recovery"))])
                print("Proposal saved; viewer generation failed. Do not repeat fitting.\n"
                      "Recover the saved viewer with a new output directory:\n" + recovery, file=sys.stderr)
            parser.exit(1, f"weft: body proposal workflow failed; preserved result: {Path(args.out) / 'result.json'}\n")
        if args.command == "stereo-ground" and result.get("status") != "conditional_plane":
            parser.exit(1, f"weft: ground unavailable; inspect preserved labels and report: {args.out}\n")
        if args.command == "floor-calibrate":
            pin = result.get("manifest_sha256")
            if (result.get("profile") != "floor-target-calibration-package-v1" or result.get("sealed") is not True
                    or type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)
                    or type(result.get("report_path")) is not str or not result["report_path"]):
                parser.exit(1, f"weft: calibration did not publish a complete package; inspect preserved output: {args.out}\n")
            if result.get("status") != "qualified_nominal":
                parser.exit(1, f"weft: calibration {result.get('status', 'unavailable')}; inspect alternatives: {result['report_path']}\n")
            if (type(result.get("output_path")) is not str or not result["output_path"]
                    or result.get("camera_path") != str(Path(result["output_path"]) / "camera.json")
                    or result.get("ground_path") != str(Path(result["output_path"]) / "ground.json")):
                parser.exit(1, f"weft: calibration lacks its exact nominal output pair; inspect: {result['report_path']}\n")
            print(f"Saved calibration report: {result['report_path']}\n"
                  "Nominal geometry only; no body or floor has been changed.\n"
                  f"Camera: {result['camera_path']}\nGround: {result['ground_path']}", file=sys.stderr)
        if args.command == "surface-anchors":
            if result.get("status") != "completed" or result.get("sealed") is not True:
                parser.exit(1, f"weft: surface anchors stopped; inspect preserved output: {args.out}\n")
            print(f"Saved conditional-coordinate report: {result['report_path']}", file=sys.stderr)
        if args.command == "surface-inputs":
            if result.get("status") != "completed" or result.get("sealed") is not True or not result.get("report_path"):
                parser.exit(1, f"weft: surface input collection stopped; inspect preserved output: {args.out}\n")
            print(f"Saved surface-input report: {result['report_path']}", file=sys.stderr)
        if args.command == "body-reference-trial" and result.get("inspection_path"):
            print(f"Saved comparison page: {result['inspection_path']}", file=sys.stderr)
        if args.command == "exercise-equipment-evidence":
            pin = result.get("manifest_sha256")
            if (result.get("profile") != "pose-equipment-silhouette-composition-v1"
                    or result.get("status") != "completed" or result.get("sealed") is not True
                    or type(result.get("output_path")) is not str or not result["output_path"]
                    or type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)):
                parser.exit(1, f"weft: pose/equipment composition stopped; inspect preserved output: {args.out}\n")
            target = Path(result["output_path"])
            followup = shlex.join([sys.executable, "-m", "weft", "equipment-support", "--run", str(target),
                "--manifest-sha256", pin, "--hand-points", "--fixed-camera",
                "--out", str(target.with_name(target.name + "-hand-support"))])
            print(f"Saved pose/equipment image evidence: {target}\n"
                  "Next, only if the source camera was fixed (not verified by this command):\n" + followup, file=sys.stderr)
        if args.command == "equipment-evidence-report":
            if result.get("status") != "completed" or result.get("sealed") is not True or not result.get("report_path"):
                parser.exit(1, f"weft: equipment evidence inspection stopped; inspect preserved output: {args.out}\n")
            print(f"Saved equipment-evidence report: {result['report_path']}", file=sys.stderr)
        if args.command == "equipment-support":
            if result.get("status") != "completed" or result.get("sealed") is not True or not result.get("report_path"):
                parser.exit(1, f"weft: equipment support check stopped; inspect preserved output: {args.out}\n")
            print(f"Saved equipment-support comparison: {result['report_path']}", file=sys.stderr)
        if args.command == "body-reference-trial" and result.get("status") != "completed":
            parser.exit(1, f"weft: reference controller trial stopped; preserved report: {Path(args.out) / 'report.json'}\n")
        if args.command == "spatial-scene" and result.get("status") != "constructed_hypothesis":
            parser.exit(1, f"weft: 3D construction stopped; inspect preserved output: {args.out}\n")
        if args.command == "shared-world-shape" and result.get("status") == "failed":
            parser.exit(1, f"weft: shape preparation failed; preserved report: {result['report_path']}\n")
        if args.command == "shared-exercise-fullspan" and result.get("status") == "failed":
            parser.exit(1, f"weft: full-reference preparation failed; preserved report: {result['directory']}/fullspan.json\n")
        if args.command == "qualify" and (result["status"] != "completed" or result["operationally_qualified"] != len(result["cases"])):
            parser.exit(1, f"weft: not all environments qualified; preserved report: {result['directory']}/report.json\n")
    except (ValueError, RuntimeError, TimeoutError, OSError, MemoryError) as error:
        parser.exit(1, f"weft: {error}\n")


if __name__ == "__main__":
    main()
