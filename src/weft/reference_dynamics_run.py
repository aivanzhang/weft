"""Bounded CLI diagnostics for source motion; completed labs stay immutable."""

from hashlib import sha256
import html
import io
from importlib.metadata import version
import json
import math
from pathlib import Path
import resource
import sys
import time

from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files


MAX_SOURCE_BYTES, MAX_MEMBER_BYTES = 256 << 20, 64 << 20
MAX_OUTPUT_BYTES, MAX_OUTPUT_FILES = 64 << 20, 160
MAX_RSS_BYTES = 512 << 20
PROFILE = "motion-force-check-v1"


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate force-check metadata key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Force-check metadata must be finite strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Force-check metadata must be an object")
    return value


def _encoded(value, limit=32 << 20):
    stream = io.BytesIO()
    for chunk in json.JSONEncoder(sort_keys=True, indent=2, allow_nan=False).iterencode(value):
        raw = chunk.encode()
        if stream.tell() + len(raw) + 1 > limit:
            raise ValueError("Force-check document exceeds its output bound")
        stream.write(raw)
    stream.write(b"\n")
    return stream.getvalue()


def _path(value):
    path = Path(value).absolute()
    if len(path.parts) > 64 or '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Force-check paths cannot traverse or follow symlinks")
    return path


def _rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def _inputs(run, check):
    manifest_raw = read_package_file(run, 'manifest.json', max_bytes=256 << 10)
    manifest = _json(manifest_raw)
    files = manifest.get('files')
    if (not isinstance(files, dict) or not 1 <= len(files) <= 512 or 'manifest.json' in files
            or not {'scene.json', 'exercise-task.json'} <= files.keys()
            or manifest.get('status') in {'failed', 'rejected', 'cancelled'} or 'failure.json' in files
            or any(not isinstance(pin, str) or len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin)
                   for pin in files.values())):
        raise ValueError("Force check requires a completed seal with scene and task, and at most 512 members")
    validate_package_layout([*files, 'manifest.json'])
    total, retained = 0, {'manifest.json': manifest_raw}
    for name, pin in files.items():
        check()
        raw = read_package_file(run, name, max_bytes=min(MAX_MEMBER_BYTES, max(1, MAX_SOURCE_BYTES - total)),
                                expected_sha256=pin)
        total += len(raw)
        if total > MAX_SOURCE_BYTES:
            raise ValueError("Force-check source exceeds 256 MiB")
        if name in {'scene.json', 'exercise-task.json'}:
            if len(raw) > 8 << 20:
                raise ValueError("Force-check scene/task exceeds 8 MiB")
            retained[name] = raw
    if read_package_file(run, 'manifest.json', max_bytes=256 << 10) != manifest_raw:
        raise ValueError("Source manifest changed while reading")
    check()
    scene, task = (_json(retained[name]) for name in ('scene.json', 'exercise-task.json'))
    if (scene.get('profile') != 'shared-world-v6' or scene.get('objects') != []
            or task.get('profile') != 'camera-shared-task-v1' or task.get('scene_sha256') != canonical_hash(scene)):
        raise ValueError("Force check currently supports human-only camera worlds with their unchanged task")
    return scene, task, retained, {'verified_files': len(files), 'manifest_sha256': sha256(manifest_raw).hexdigest()}


def _runtime():
    files, total = {}, 0
    for name, module in sorted(sys.modules.copy().items()):
        source = getattr(module, '__file__', None)
        if (name == 'weft' or name.startswith('weft.')) and source and source.endswith('.py'):
            path = _path(source)
            raw = read_package_file(path.parent, path.name, max_bytes=2 << 20)
            total += len(raw)
            if total > 8 << 20 or len(files) >= 128:
                raise ValueError("Force-check runtime snapshot exceeds its bound")
            files['implementation/' + name.replace('.', '/') + '.py'] = raw
    return files


def render_report(result):
    diagnostic = result['diagnostic']
    labels = {'interval_midpoint': 'Between saved poses (constant target speed)',
              'reset_transition_probe': 'Starting from rest (finite-window probe)',
              'knot_transition_probe': 'At pose changes (finite-window probes)'}
    rows = []
    for kind, summary in diagnostic['summary'].items():
        ratio = summary['maximum_force_limit_ratio']
        free = [row for row in diagnostic['samples'] if row['kind'] == kind and row['constraint_rows'] == 0]
        free_ratio = max((max(row['force_limit_ratio']) for row in free), default=None)
        rows.append(f'<tr><td>{labels[kind]}</td><td>{summary["samples"]}</td>'
            f'<td>{summary["over_limit_samples"]}</td><td>{summary["constraint_free_over_limit_samples"]}</td>'
            f'<td>{"unavailable" if ratio is None else f"{ratio:.3f}×"}</td>'
            f'<td>{"unavailable" if free_ratio is None else f"{free_ratio:.3f}×"}</td></tr>')
    worst = sorted(diagnostic['samples'], key=lambda row: max(row['force_limit_ratio']), reverse=True)[:12]
    details = []
    for row in worst:
        index = max(range(len(diagnostic['names'])), key=lambda i: row['force_limit_ratio'][i])
        details.append(f'<tr><td>{row["time_s"]:.3f}s</td><td>{labels[row["kind"]]}</td>'
            f'<td>{html.escape(diagnostic["names"][index])}</td><td>{row["force_limit_ratio"][index]:.3f}×</td>'
            f'<td>{row["constraint_rows"]}</td></tr>')
    limits = ''.join(f'<li>{html.escape(text)}</li>' for text in diagnostic['limitations'])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Weft motion force diagnostic</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}
p,li{{line-height:1.5}}</style><h1>Can this model follow the reconstructed movement?</h1>
<p>This is a source-target diagnostic, not a new controller or a physical measurement.
Values above <b>1×</b> need more force than the unchanged motor limit at that sampled state.</p>
<p>The body moves along straight lines between saved joint angles. Its requested speed jumps at pose changes.
The second and third rows approximate those speed changes over {diagnostic['transition_window_s']:.3f} seconds;
they do not replace or smooth the saved motion. Constraint-assisted results are not evidence of correct contact.</p>
<table><thead><tr><th>Sample</th><th>Count</th><th>Over motor limit</th><th>Over limit without constraints</th>
<th>Largest force / limit</th><th>Largest without constraints</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Largest sampled demands</h2><table><thead><tr><th>Source time</th><th>Sample</th><th>Motor</th>
<th>Force / limit</th><th>Constraint rows</th></tr></thead><tbody>{''.join(details)}</tbody></table>
<h2>What this does not establish</h2><ul>{limits}</ul>
<p><a href="diagnostic.json">Every force and source-time sample</a> · <a href="reference-input.json">Unchanged targets</a>
 · <a href="report.json">Inputs, runtime and resource record</a></p></html>'''



def run_motion_force_check(run, out, *, max_seconds=30):
    """CLI-only native work; budgets are cooperative, not a security/process sandbox."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not .01 <= max_seconds <= 60:
        raise ValueError("Force check allows .01–60 seconds")
    started = time.monotonic()
    def check():
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Force check exceeded its cooperative time limit")
        if _rss() > MAX_RSS_BYTES:
            raise MemoryError("Force check exceeded its cooperative peak-memory limit")

    run, out = _path(run), _path(out)
    if out.exists() or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError("Use a new force-check directory outside the source run")
    scene, task, originals, verified = _inputs(run, check)
    from weft.camera_shared_task import validate_camera_shared_task
    from weft.reference_dynamics import audit_reference_dynamics
    from weft.shared_world import SharedWorldEnv

    validate_camera_shared_task(task, scene)
    if task['control']['root_assistance'] != 'explicit_3_axis':
        raise ValueError("Force check requires explicitly assisted roots; disabled-root feasibility is not implemented")
    check()
    runtime = _runtime()
    versions = {'python': sys.version.split()[0], **{name: version(name) for name in ('numpy', 'mujoco', 'gymnasium')}}
    plan = {'profile': PROFILE, 'created_at': local_time(), 'source_run': str(run), 'source_manifest': verified,
        'scene_sha256': canonical_hash(scene), 'task_sha256': canonical_hash(task),
        'gates_sha256': task['gates_sha256'], 'runtime_versions': versions,
        'limits': {'seconds': max_seconds, 'peak_rss_bytes': MAX_RSS_BYTES, 'output_bytes': MAX_OUTPUT_BYTES,
                   'source_bytes': MAX_SOURCE_BYTES, 'source_members': 512, 'output_files': MAX_OUTPUT_FILES,
                   'reference_poses': 900, 'inverse_calls': 1798, 'physics_steps': 0},
        'budget_scope': 'Cooperative time/RSS checks between bounded reads and native operations, not hard preemption.',
        'physical_validation': 'unverified', 'api_cost_usd': 0}
    out.mkdir(parents=True, mode=0o700)
    files = {**{'inputs/' + name: raw for name, raw in originals.items()}, **runtime,
             'plan.json': _encoded(plan, 128 << 10)}
    write_package_files(out, files)
    env, diagnostic, error = None, None, None
    try:
        check()
        env = SharedWorldEnv(scene)
        check()
        if (env.human_motor_count != 22 or env.start_offset_s != 0
                or abs(env.horizon_steps * env.dt - (env._times[-1] - env._times[0])) > 1e-8):
            raise ValueError("Force check requires the complete nominal 22-axis source interval")
        reference = {'times': env._times.tolist(), 'positions': env._reference.T.tolist(),
            'names': list(env.human_action_names), 'limits': env.human_limits.tolist(), 'transition_s': env.dt}
        diagnostic = audit_reference_dynamics(env.model, reference['times'], reference['positions'],
            names=reference['names'], limits=reference['limits'], transition_s=reference['transition_s'], check_budget=check)
        if canonical_hash(env.scene) != canonical_hash(scene):
            raise ValueError("Diagnostic changed the source scene")
        _, _, current, after = _inputs(run, check)
        final_runtime = _runtime()
        if current != originals or after != verified or any(final_runtime.get(name) != raw for name, raw in runtime.items()):
            raise ValueError("Source or installed runtime changed during force checking")
        summary = {}
        for kind, row in diagnostic['summary'].items():
            free = [sample for sample in diagnostic['samples'] if sample['kind'] == kind and sample['constraint_rows'] == 0]
            summary[kind] = {**row, 'constraint_free_maximum_force_limit_ratio':
                            max((max(sample['force_limit_ratio']) for sample in free), default=None)}
        extra = {'reference-input.json': _encoded(reference), 'diagnostic.json': _encoded(diagnostic),
                 'index.html': render_report({'diagnostic': diagnostic}).encode(),
                 **{name: raw for name, raw in final_runtime.items() if name not in runtime}}
        check()
        if len(files) + len(extra) + 2 > MAX_OUTPUT_FILES or sum(map(len, [*files.values(), *extra.values()])) > MAX_OUTPUT_BYTES - (256 << 10):
            raise ValueError("Force check exceeds the package output bound")
        write_package_files(out, extra)
        files.update(extra)
    except (Exception, KeyboardInterrupt) as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:2000]}
    finally:
        if env is not None:
            env.close()
    report = {**plan, 'status': 'failed' if error else 'diagnostic_only', 'error': error,
        'source_and_task_unchanged': None if error else True,
        'summary': {} if error else summary, 'inverse_calls': None if error else diagnostic['inverse_calls'],
        'physics_steps': 0, 'inference_calls': 0, 'elapsed_seconds': time.monotonic() - started,
        'peak_rss_bytes': _rss(), 'report_path': str(out / ('report.json' if error else 'index.html')),
        'scope': 'Authored model force budgets only. Sampled force demands do not certify full-path feasibility or measured contact.'}
    report_raw = _encoded(report, 128 << 10)
    files['report.json'] = report_raw
    manifest = {'schema_version': 1, 'created_at': local_time(), 'status': report['status'],
                'files': {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    write_package_files(out, {'report.json': report_raw, 'manifest.json': _encoded(manifest, 128 << 10)})
    return report
