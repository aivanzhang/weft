"""Build source-linked, explicitly authored ramp sandboxes without fitting video labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from html import escape
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts
from weft.recording import Recorder, export_recording, load_recording, render_snapshot
from weft.scene import SCENE_UNITS, SceneEnv, validate_scene


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/phys101-ramp-source-01"
COAST = [0] * 120
INTERVENTION = [2] * 20 + [0] * 100
LIMITATIONS = [
    "Authored normalized 2D scene, not a calibrated reconstruction or a pixel-aligned camera view.",
    "Only the ramp angle and source identity come from provider metadata; all dimensions and placements are authored.",
    "Gravity 9.81 scene-widths/s² assumes one scene width is one meter; the video does not establish that scale.",
    "Masses are relative. The wooden block's reported 21.2 g is metadata, not a recovered mass calibration; target mass is unknown.",
    "Each shape has friction 0.5, giving pair coefficient 0.25; it is a hypothesis, not a measured material property.",
    "One friction coefficient models both sticking and sliding; no separate kinetic coefficient, rolling resistance or deformation.",
    "Simulation starts at rest without the hand. Both complete original videos remain available, including release.",
    "Arrows apply virtual forces; brake and the parking target are authored controls, not measured physical interventions.",
    "Qualitative agreement is a development check, not a fresh holdout, physical validation or evidence of general reconstruction.",
]


def make_scene(degrees):
    angle = math.radians(degrees)
    tangent = np.array([math.cos(angle), math.sin(angle)])
    normal = np.array([math.sin(angle), -math.cos(angle)])
    center = np.array([0.36, 0.6 - 0.25 * math.sin(angle) - 0.01 * math.cos(angle)])
    common = {"shape": "box", "mass": 1, "velocity": [0, 0], "angular_velocity": 0,
              "friction": 0.5, "provenance": {"geometry": "authored", "mass": "assumed relative unit",
              "friction": "assumed Pymunk shape coefficient; pair coefficients multiply"}}
    return validate_scene({"schema_version": 2, "profile": "planar-rigid-v2", "units": deepcopy(SCENE_UNITS),
        "boundary_friction": 0.5, "width": 1, "height": 0.8, "dt": 1 / 60,
        "gravity": [0, 9.81], "damping": 1, "restitution": 0,
        "objects": [
            common | {"id": "wood-block", "label": "Wooden block", "role": "active", "size": [0.06, 0.04],
                "angle": angle, "position": (center - 0.19 * tangent + 0.03001 * normal).tolist(), "color": "#AB7E48"},
            common | {"id": "green-box", "label": "Green target box", "role": "passive", "size": [0.07, 0.07],
                "angle": 0, "position": [0.70, 0.56499], "color": "#4A762E"},
            common | {"id": "ramp", "label": "Fixed inclined support", "role": "static", "size": [0.5, 0.02],
                "angle": angle, "position": center.tolist(), "color": "#665F59"},
            common | {"id": "table", "label": "Fixed table", "role": "static", "size": [0.94, 0.02],
                "angle": 0, "position": [0.5, 0.61], "color": "#B8ADA0"}],
        "controlled_id": "wood-block", "target": [0.88, 0.22],
        "task": {"position_tolerance": 0.025, "speed_tolerance": 0.05, "max_steps": 900, "acceleration": 2},
        "relationships": [
            {"type": "supported_by", "objects": ["wood-block", "ramp"], "status": "assumed"},
            {"type": "supported_by", "objects": ["green-box", "table"], "status": "assumed"},
            {"type": "possible_collision", "objects": ["wood-block", "green-box"], "status": "assumed"}],
        "assumptions": LIMITATIONS[:]})


def verify_replay(path):
    saved = load_recording(path)
    env = SceneEnv(saved["scene"])
    try:
        for event in saved["events"]:
            if event["type"] in {"start", "reset"}:
                env.reset(seed=event["seed"], options={"jitter": False})
            else:
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if (reward, terminated, truncated) != (event["reward"], event["terminated"], event["truncated"]):
                    raise ValueError("Recorded transition differs on exact replay")
            if env.snapshot() != event["snapshot"]:
                raise ValueError("Recorded pose, spin or contacts differ on exact replay")
        return len(saved["events"])
    finally:
        env.close()


def run_case(item):
    out, degrees = item
    out = Path(out)
    scene = json.loads((out / f"ramp-{degrees}/scene.json").read_text())
    result, finals = {}, {}
    for label, actions in (("coast", COAST), ("push", INTERVENTION)):
        env = SceneEnv(scene)
        env.reset(seed=1729, options={"jitter": False})
        path = out / f"ramp-{degrees}-{label}"
        recorder = Recorder(path, scene)
        recorder.start(env.snapshot(), seed=1729)
        initial = env.snapshot()
        timing, contacts, clamps = [], set(), 0
        started = time.perf_counter()
        try:
            for action in actions:
                begin = time.perf_counter()
                _, reward, terminated, truncated, _ = env.step(action)
                state = env.snapshot()
                timing.append((time.perf_counter() - begin) * 1000)
                recorder.append({"type": "step", "action": action, "reward": reward,
                    "terminated": terminated, "truncated": truncated, "snapshot": state})
                contacts.update(tuple(pair) for pair in state["contacts"])
                clamps += any(state[key] for key in ("speed_limited", "angular_speed_limited", "substep_limited"))
                if terminated or truncated:
                    raise ValueError("Unexpected early task termination; preserve the actual recorded interval")
            finals[label] = state
            Image.fromarray(render_snapshot(state, scene)).save(out / f"ramp-{degrees}-{label}-final.png")
        finally:
            recorder.close()
            env.close()
        elapsed = time.perf_counter() - started
        exported = export_recording(path)
        replayed = verify_replay(path)
        result[label] = {"recording": path.name, "completed_actions": len(actions), "exact_replay_states": replayed,
            "video_frames": exported["video_frames"], "contacts": sorted(contacts), "limited_steps": clamps,
            "block_displacement": math.dist(initial["objects"][0]["position"], state["objects"][0]["position"]),
            "target_displacement": math.dist(initial["objects"][1]["position"], state["objects"][1]["position"]),
            "integration_and_journal_seconds": elapsed,
            "step_and_snapshot_ms": {"median": float(np.median(timing)), "p95": float(np.percentile(timing, 95)),
                                     "max": max(timing), "sample_count": len(timing)},
            "performance_scope": "Local Python step/snapshot calls, excluding rendering, journal, HTTP and browser"}
    result["changed_action_distance"] = math.dist(finals["coast"]["objects"][0]["position"], finals["push"]["objects"][0]["position"])
    return degrees, result


def index_html(clips):
    sections = []
    for clip in clips:
        degree = clip["ramp_degrees"]
        sections.append(f'''<section><h2>{degree}° ramp</h2><div class="pair">
        <figure><figcaption>Original video · complete {clip['decoded_frames']} frames</figcaption>
        <video controls preload="metadata" src="../phys101-ramp-source-01/{escape(clip['file'])}"></video></figure>
        <figure><figcaption>Authored simulation · coast · 2 seconds</figcaption>
        <video controls preload="metadata" src="ramp-{degree}-coast/video.mp4"></video></figure></div>
        <p>The views are not calibrated, aligned or time-synchronized. The simulation omits the hand release.</p>
        <p><a href="ramp-{degree}-coast/replay.html">Inspect every simulated state</a> ·
        <a href="ramp-{degree}-push/replay.html">See the right-push intervention</a> ·
        <a href="ramp-{degree}/scene.json">Scene inputs</a> ·
        <a href="ramp-{degree}/scenario-provenance.json">Source and assumptions</a></p>
        <details><summary>Interact and record in your terminal</summary><pre>.venv-lab/bin/python -m weft scene-sandbox \\
  --run artifacts/oriented-ramp-sandbox-01/ramp-{degree} \\
  --record artifacts/my-ramp-{degree}-session</pre>
        <p>Use a new recording folder. Enter <code>coast 6</code>, <code>right 6</code>, <code>left 6</code>,
        <code>up 6</code>, <code>down 6</code>, <code>brake 6</code>, <code>reset</code> or <code>quit</code>.
        Each command advances up to six steps. Open <code>current-frame.png</code> for the current view;
        quitting saves a video and an offline replay. Nothing moves while you are typing.</p></details></section>''')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
    <title>Weft ramp sandbox</title><style>body{background:#F5EADC;color:#214F50;font:17px system-ui;margin:32px auto;
    max-width:1120px;padding:0 24px;line-height:1.55}h1,h2{line-height:1.15}a{color:#965644}section{border-top:1px solid #B8ADA0;
    padding:24px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}video{width:100%;margin-top:8px}
    pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#fff8ef}summary{cursor:pointer}
    @media(max-width:700px){.pair{grid-template-columns:1fr}}</style><h1>Weft ramp sandbox</h1>
    <p><strong>This is an authored simulation, not an accurate reconstruction of the video.</strong>
    Compare what is recorded, inspect the assumptions, then try virtual controls and record your session.</p>
    <p>Two hypotheses share the same assumed friction and dimensions. Only the provider-declared angle changes.
    Real scale, friction, mass ratio and camera geometry are unverified. No new paid call or policy training was used.</p>
    <p><a href="report.json">Verification results</a> · <a href="../meta-phys101-ramp-masks-01/index.html">Real-video tracking and masks</a></p>''' + "".join(sections) + "</html>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=int, choices=(10, 20), help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != ROOT / "artifacts" or out == SOURCE or not re.fullmatch(r"[a-z0-9-]+", out.name):
        raise ValueError("Use a new lowercase-slug artifacts directory outside the immutable source")
    if args.case is not None:
        degree, result = run_case((str(out), args.case))
        write_json(out / f"case-{degree}-result.json", result)
        return
    source_before = verify_artifacts(SOURCE)
    provenance = json.loads((ROOT / "data/general/phys101-ramps.source.json").read_text())
    for clip in provenance["clips"]:
        if digest(SOURCE / clip["file"]) != clip["sha256"]:
            raise ValueError("Source clip identity differs from the declared binding")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    for name, path in _implementation_files().items():
        target = out / "implementation-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    scenarios = {}
    for clip in provenance["clips"]:
        degree = clip["ramp_degrees"]
        path = out / f"ramp-{degree}"
        path.mkdir()
        scene = make_scene(degree)
        write_json(path / "scene.json", scene)
        write_json(path / "scenario-provenance.json", {"source": clip, "attribution": provenance["attribution"],
            "rights": provenance["license"], "source_reported_object": provenance["object"],
            "construction_inputs": ["Provider ramp angle", "Qualitative block/support/target roles"],
            "unused_for_construction": ["Comparison labels", "Saved model tracks and masks", "Metric trajectory or force measurements"],
            "limitations": LIMITATIONS, "source_frame_coverage": list(range(clip["decoded_frames"])),
            "simulation_to_source_frame_mapping": None})
        write_json(path / "report.json", {"status": "authored_scenario_frozen", "physics_ready": False,
            "physical_validation": "unverified", "created_at": local_time(), "scene_sha256": canonical_hash(scene)})
        scenarios[degree] = seal_suite(path)
    write_json(out / "pre-run.json", {"created_at": local_time(), "source": source_before, "scenarios": scenarios,
        "implementation": implementation, "workers": 2, "max_wall_seconds_per_case": 45,
        "paid_requests": 0, "training": False, "actions": {"coast": COAST, "push": INTERVENTION},
        "development_gates": {"static_10_max_displacement": 0.002, "sliding_20_min_displacement": 0.15,
            "sliding_20_block_target_contact": True, "changed_action_min_distance": 0.01, "limited_steps": 0},
        "scope": "Implementation, replay and qualitative development checks only; no physical accuracy score"})
    started = time.perf_counter()
    def launch(degree):
        with (out / f"case-{degree}.stdout").open("x") as stdout, (out / f"case-{degree}.stderr").open("x") as stderr:
            subprocess.run([sys.executable, str(out / Path(__file__).name), "--out", str(out), "--case", str(degree)],
                           stdout=stdout, stderr=stderr, cwd=ROOT, check=True, timeout=45)
        return degree, json.loads((out / f"case-{degree}-result.json").read_text())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(launch, (10, 20)))
    commands = "status\ncoast 6\nright 6\nbrake 6\nreset\nquit\n"
    (out / "terminal-input.txt").write_text(commands)
    with (out / "terminal.stdout").open("w") as stdout, (out / "terminal.stderr").open("w") as stderr:
        subprocess.run([sys.executable, "-m", "weft", "scene-sandbox", "--run", str(out / "ramp-20"),
            "--record", str(out / "terminal-session")], input=commands, text=True, stdout=stdout, stderr=stderr,
            cwd=ROOT, check=True, timeout=30)
    terminal_states = verify_replay(out / "terminal-session")
    if verify_artifacts(SOURCE) != source_before or any(verify_artifacts(out / f"ramp-{d}") != scenarios[d] for d in (10, 20)):
        raise ValueError("Frozen source or scenarios changed")
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError("Implementation changed during the study")
    gates = {"static_10": results[10]["coast"]["block_displacement"] <= 0.002,
        "sliding_20": results[20]["coast"]["block_displacement"] >= 0.15,
        "sliding_contact": ["green-box", "wood-block"] in results[20]["coast"]["contacts"],
        "changed_actions": all(results[d]["changed_action_distance"] >= 0.01 for d in (10, 20)),
        "no_clamps": all(results[d][label]["limited_steps"] == 0 for d in (10, 20) for label in ("coast", "push"))}
    write_json(out / "report.json", {"created_at": local_time(), "status": "completed_authored_sandbox_check",
        "physical_validation": "unverified", "physics_ready": False, "results": results, "development_gates": gates,
        "wall_seconds": time.perf_counter() - started, "terminal_exact_states": terminal_states,
        "all_sources_scenarios_implementation_unchanged": True, "paid_requests": 0, "training": False,
        "limitations": LIMITATIONS, "source_trajectory_error": None})
    (out / "index.html").write_text(index_html(provenance["clips"]).replace(
        "artifacts/oriented-ramp-sandbox-01", str(out.relative_to(ROOT))), encoding="utf-8")
    print(json.dumps({"development_gates": gates, "results": results, "terminal_exact_states": terminal_states,
        "verification": seal_suite(out)}, indent=2))


if __name__ == "__main__":
    main()
