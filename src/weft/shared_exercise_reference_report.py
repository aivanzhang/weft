"""A static entry point for a paired trial; delivery is not task success."""

from html import escape
import math
from pathlib import Path
import re


PROFILE = "shared-exercise-reference-trial-v1"
MAX_PAGE_BYTES = 256 * 1024
_MODES = {"authored": "Original controller", "reference_proposal": "Video-fitted controller"}


def _text(value, limit=240):
    return escape(str(value)[:limit], quote=True)


def _number(value, digits=3):
    return f"{value:.{digits}f}" if type(value) in (int, float) and math.isfinite(value) else "Unavailable"


def _identity(value):
    return f"<code>{value}</code>" if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else "Unavailable"


def _exists(root, relative):
    path = root
    for part in Path(relative).parts:
        path /= part
        if path.is_symlink():
            return False
    return path.is_file()


def _link(root, relative, label):
    # Only fixed owned paths reach this helper; report text is never a URL.
    return f'<a href="{relative}">{escape(label)}</a>' if _exists(root, relative) else ""


def _trial(root, row):
    mode = row["mode"]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    success = metrics.get("success")
    outcome = "Passed" if success is True else "Failed" if success is False else "Not evaluated"
    completed = metrics.get("completed_interval")
    interval = "Yes" if completed is True else "No" if completed is False else "Unavailable"
    proof = f"proofs/{mode}-action-replay.json"
    replay_result = row.get("action_replay")
    verified = isinstance(replay_result, dict) and replay_result.get("status") == "passed" and _exists(root, proof)
    replay = "Verified exact actions and states" if verified else "Not verified"
    failures = metrics.get("failed_gates", [])
    failures = failures if isinstance(failures, list) else []
    reasons = "<ul>" + "".join(f"<li>{_text(reason)}</li>" for reason in failures[:32]) + "</ul>" if failures else ""
    links = [_link(root, f"recordings/{mode}/{name}", label) for name, label in (
        ("replay.html", "Inspect 3D replay"), ("video.mp4", "Recorded video"),
        ("events.jsonl", "Actions and states"), ("manifest.json", "Recording hashes"))]
    links.append(_link(root, proof, "Exact replay check"))
    links = " · ".join(link for link in links if link) or "No completed replay artifacts available."
    return f'''<section aria-label="{_MODES[mode]}"><h3>{_MODES[mode]}</h3>
<p><strong>Task outcome: {outcome}</strong> · Full interval completed: {interval}</p>
<dl><dt>Actions</dt><dd>{_number(row.get('steps'), 0)}</dd>
<dt>Simulated time</dt><dd>{_number(metrics.get('simulated_duration_s'))} seconds</dd>
<dt>Mean reward</dt><dd>{_number(metrics.get('mean_reward'))}</dd>
<dt>Joint tracking error</dt><dd>{_number(metrics.get('angular_rmse_rad'))} radians</dd>
<dt>Exact replay</dt><dd>{replay}</dd></dl>{reasons}<p>{links}</p></section>'''


def write_reference_trial_report(out, report):
    """Write once from the runner's final result, without re-running any stage."""
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("profile") != PROFILE
            or report.get("status") not in {"completed", "failed"}):
        raise ValueError("Reference trial report requires a terminal result")
    rows = report.get("trials", [])
    if (not isinstance(rows, list) or len(rows) > 2 or any(not isinstance(row, dict)
            or row.get("mode") not in _MODES for row in rows)
            or len({row["mode"] for row in rows}) != len(rows)):
        raise ValueError("Reference trial report requires at most two distinct controllers")
    root = Path(out).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Reference trial report cannot write through symlinks")
    completed = report["status"] == "completed"
    heading = "Comparison saved" if completed else "Comparison stopped"
    delivery = "The comparison workflow finished. Review its saved artifacts and checks below." if completed else "Some stages did not finish. Preserved artifacts remain available below."
    same = report.get("same_initial_state")
    reset = "Identical initial states verified." if same is True else "Initial states differ." if same is False else "Matched initial states not yet verified."
    error = report.get("error")
    failure = f"<p role=\"alert\">Stopped at {_text(report.get('stage', 'unknown'))}: {_text(error.get('message', 'Unknown error'), 2000)}</p>" if isinstance(error, dict) else ""
    links = [_link(root, name, label) for name, label in (
        ("inputs/proposal-report.json", "Saved source-fit measurements"),
        ("inputs/input-receipt.json", "Input identities and checks"),
        ("run/scene.json", "Unchanged scene"), ("run/exercise-task.json", "Unchanged task"),
        ("run/reference-controller.json", "Fitted controller recipe"),
        ("report.json", "Complete trial results"))]
    documents = " · ".join(link for link in links if link) or "No completed input documents available."
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weft — controller comparison</title><style>
:root{{color-scheme:light}}body{{margin:0;background:#F5EADC;color:#214F50;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:32px 20px}}h1,h2,h3{{line-height:1.25}}
a{{color:#965644;text-underline-offset:3px}}a:focus-visible{{outline:3px solid currentColor;outline-offset:3px}}
section{{background:#fff9;padding:20px;border:1px solid #214F5033;border-radius:12px;margin:16px 0}}
.trials{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}dd{{margin:0}}code{{overflow-wrap:anywhere;font-size:.85em}}
</style></head><body><main><p>Weft · saved simulation comparison</p><h1>{heading}</h1>
<p>{delivery} <strong>This does not mean the simulated task passed.</strong></p>{failure}
<h2>What changed</h2><p>The video-fitted controller uses the proposed joint targets to choose actions.
The body, equipment, original task, scoring targets and scoring rules stay fixed. Root assistance stays explicit.
This is a programmed controller, not a learned policy or a repaired body.</p>
<p>{reset} No candidate is installed in your working sandbox.</p>
<h2>Inspect the result</h2><ol><li>Compare task outcomes below, including failed checks.</li>
<li>Open each 3D replay to inspect the recorded motion. Replay controls change the view, not the physics.</li>
<li>Use the saved measurements and exact action/state records to understand the result.</li></ol>
<div class="trials">{''.join(_trial(root, row) for row in rows)}</div>
<h2>Preserved evidence</h2><p>{documents}</p>
<p>Original scene: {_identity(report.get('scene_sha256'))}<br>Original task: {_identity(report.get('task_sha256'))}</p>
<p>These are simulated outcomes on exposed development evidence. Exact replay is not physical validation,
measured muscle activation, recovered grasp or evidence of general accuracy on new videos.</p>
</main></body></html>'''
    raw = page.encode("utf-8")
    if len(raw) > MAX_PAGE_BYTES:
        raise ValueError("Reference trial report exceeds its page limit")
    with (root / "index.html").open("xb") as stream:
        stream.write(raw)
    return "index.html"
