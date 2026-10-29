"""Frozen four-source ankle candidate audit, direct actions and portable replays."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.foot_body import FootBody, audit_foot_body, foot_xml
from weft.foot_sandbox import FootEnv, console
from weft.human import HumanEnv
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.recording import render_snapshot, load_recording


ROOT = Path(__file__).resolve().parents[2]
CASES = ("single-leg-squat", "squat", "kettlebell-pushup", "fullbody-squat")


def distribution(values):
    return {"samples": len(values), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)), "maximum_ms": max(values)} if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/foot-body-01")
    args = parser.parse_args()
    out, source = Path(args.out).resolve(), ROOT / "artifacts/foot-reference-01"
    source_seal = verify_artifacts(source)
    if out.exists() or out.is_relative_to(source):
        raise ValueError("Use a new output outside the frozen foot-reference study")
    out.mkdir(parents=True)
    for path in [*sorted((ROOT / "src/weft").glob("*.py")), Path(__file__).resolve(),
                 ROOT / "tests/test_foot_body.py", ROOT / "requirements-lab.lock", ROOT / "pyproject.toml"]:
        target = out / "implementation-source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for filename in ("foot-body-tests.xml", "foot-body-console-tests.xml", "foot-body-final-tests.xml",
                     "foot-body-integration-tests.xml", "foot-body-replay-tests.xml"):
        shutil.copyfile(ROOT / ".local" / filename, out / filename)
    started, results, tiles = time.monotonic(), {}, []
    for case in CASES:
        folder = out / case
        inputs = folder / "inputs"
        inputs.mkdir(parents=True)
        for origin, filename in ((source / case / "inputs/human.json", "human.json"),
                                 (source / case / "foot-reference.json", "foot-reference.json")):
            shutil.copyfile(origin, inputs / filename)
        spec = json.loads((inputs / "human.json").read_text())
        reference = json.loads((inputs / "foot-reference.json").read_text())
        audit = audit_foot_body(spec, reference)
        write_json(folder / "candidate-audit.json", audit)
        result = {"status": audit["status"], "summary": audit["summary"],
                  "original_body_floor_failures": audit["baseline"]["summary"]["failed_source_poses"],
                  "interactive_start_available": audit["interactive_start_available"],
                  "source_raw_frames_retained": len(reference["frames"]), "physical_validation": "unverified", "promoted": False}
        if audit["status"] != "unavailable":
            (folder / "model.xml").write_text(foot_xml(spec, reference))
            original, candidate = HumanEnv(spec), FootBody(spec, reference)
            maximum_body_delta, maximum_foot_error = 0., 0.
            projections = []
            for i in range(len(spec["reference"])):
                state = candidate.reference_snapshot(i)
                if state is None:
                    projections.append({"body_reference_index": i, "status": "unavailable", "snapshot": None})
                    continue
                before = original.reference_snapshot(i)
                points = state["human"]["joints"]
                for name, point in before["human"]["joints"].items():
                    maximum_body_delta = max(maximum_body_delta, float(np.max(np.abs(np.array(points[name]) - point))))
                frame = candidate._foot_frames[spec["reference"][i]["time_s"]]
                for side in ("left", "right"):
                    for part in ("heel", "foot_index"):
                        expected = frame["feet"][side]["endpoint_geometry"][part + "_relative"]
                        actual = np.array(points[side + "_" + part]) - points[side + "_ankle"]
                        maximum_foot_error = max(maximum_foot_error, float(np.max(np.abs(actual - expected))))
                projections.append({"body_reference_index": i, "status": "kinematic_projection", "snapshot": state})
            write_json(folder / "candidate-source-poses.json", {"rows": projections, "interpolation": False})
            result.update(maximum_original_joint_position_change=maximum_body_delta,
                          maximum_native_foot_endpoint_error=maximum_foot_error)
            assert maximum_body_delta < 1e-12 and maximum_foot_error < 1e-12
            tile = Image.new("RGB", (960, 355), "#F5EADC")
            for column, index in enumerate((0, len(spec["reference"]) // 2, len(spec["reference"]) - 1)):
                state = candidate.reference_snapshot(index)
                if state is not None:
                    tile.paste(Image.fromarray(render_snapshot(state, candidate.scene, width=320)), (column * 320, 25))
                else:
                    ImageDraw.Draw(tile).text((column * 320 + 12, 140), "FOOT POSE UNAVAILABLE", fill="#214F50")
                ImageDraw.Draw(tile).text((column * 320 + 8, 7), f"{case}: body pose {index}", fill="#214F50")
            tile.save(folder / "source-poses.png")
            tiles.append(tile)
            original.close()
            candidate.close()
        if audit["interactive_start_available"]:
            commands = ["hold 6", "motor left_ankle_y .15 2", "reset", "release 6", "quit"]
            commands_iterator, messages = iter(commands), []
            result["recording"] = console(inputs / "human.json", inputs / "foot-reference.json", folder / "session",
                input_fn=lambda _: next(commands_iterator), output_fn=messages.append)
            write_json(folder / "console.json", {"commands": commands, "messages": messages})
            loaded = load_recording(folder / "session")
            assert all(len(e["action"]) == 28 for e in loaded["events"] if e["type"] == "step")
            env = FootEnv(spec, reference)
            env.reset(seed=1729)
            durations, render_times = [], []
            for i in range(180):
                tick = time.perf_counter()
                _, _, terminated, truncated, _ = env.step(env.hold_action())
                durations.append((time.perf_counter() - tick) * 1000)
                if i % 6 == 0:
                    tick = time.perf_counter()
                    Image.fromarray(env.render()).save(folder / "posture-final-frame.png")
                    render_times.append((time.perf_counter() - tick) * 1000)
                if terminated or truncated:
                    break
            result["posture_trial"] = {"metrics": env.metrics(), "steps": distribution(durations),
                                        "render_and_png": distribution(render_times),
                                        "scope": "One local bounded trial; not UI interaction, CPU/RSS or physical validation"}
            write_json(folder / "posture-final-state.json", env.snapshot())
            Image.fromarray(env.render()).save(folder / "posture-final-frame.png")
            env.close()
        results[case] = result
    if tiles:
        sheet = Image.new("RGB", (960, 355 * len(tiles)), "#F5EADC")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, (0, 355 * i))
        sheet.save(out / "candidate-source-sheet.png")
    write_json(out / "report.json", {"created_at": local_time(), "source_seal": source_seal,
        "source_manifest_sha256": digest(source / "manifest.json"), "cases": results,
        "runtime_seconds": time.monotonic() - started, "paid_requests": 0,
        "scope": "Frozen development cases; capsule feet and posture tasks are unverified candidates, not repaired anatomy"})
    print(json.dumps({"out": str(out), "cases": results, "seal": seal_suite(out)}, indent=2))


if __name__ == "__main__":
    main()
