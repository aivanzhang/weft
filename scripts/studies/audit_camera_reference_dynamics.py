"""Inspect a sealed camera-world reference without stepping physics or changing targets."""

import argparse
import html
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import mujoco

from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.reference_dynamics import audit_reference_dynamics
from weft.shared_world import SharedWorldEnv


def safe_path(value):
    path = Path(value).absolute()
    if '..' in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Diagnostic paths cannot traverse or follow symlinks')
    return path.resolve()


def read(path, limit=8 << 20):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError('Diagnostic input is missing, linked or oversized')
    return json.loads(path.read_bytes())


def validate_inputs(run, out):
    run, out = safe_path(run), safe_path(out)
    if out.exists() or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError('Use a new output directory separate from the source run')
    manifest = read(run / 'manifest.json')
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not 1 <= len(files) <= 512:
        raise ValueError('Camera run requires a bounded nonempty seal')
    total = 0
    for name in files:
        if not isinstance(name, str):
            raise ValueError('Sealed paths must be strings')
        path = safe_path(run / name)
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError('Every sealed member must remain inside its source run')
        total += path.stat().st_size
        if total > 256 << 20:
            raise ValueError('Camera run exceeds the 256 MiB inspection bound')
    if not {'scene.json', 'exercise-task.json'} <= set(files):
        raise ValueError('Camera scene and unchanged task must both be sealed')
    verified = verify_artifacts(run)
    scene, task = read(run / 'scene.json'), read(run / 'exercise-task.json')
    if (not isinstance(scene, dict) or scene.get('profile') != 'shared-world-v6' or scene.get('objects') != []
            or not isinstance(task, dict) or task.get('profile') != 'camera-shared-task-v1'
            or task.get('scene_sha256') != canonical_hash(scene)):
        raise ValueError('Diagnostic requires a sealed, unmodified human-only camera world/task')
    return run, out, scene, task, verified


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


def run_audit(run, out):
    started = time.perf_counter()
    run, out, scene, task, verified = validate_inputs(run, out)
    env = SharedWorldEnv(scene)
    try:
        if (env.human_motor_count != 22 or env.start_offset_s != 0
                or abs(env.horizon_steps * env.dt - (env._times[-1] - env._times[0])) > 1e-8):
            raise ValueError('This diagnostic requires the complete nominal 22-axis source interval')
        reference = {'times': env._times.tolist(), 'positions': env._reference.T.tolist(),
            'names': list(env.human_action_names), 'limits': env.human_limits.tolist(), 'transition_s': env.dt}
        diagnostic = audit_reference_dynamics(env.model, reference['times'], reference['positions'],
            names=reference['names'], limits=reference['limits'], transition_s=reference['transition_s'])
        assert canonical_hash(env.scene) == canonical_hash(scene)
    finally:
        env.close()
    if verify_artifacts(run) != verified:
        raise ValueError('Source run changed during inspection')
    out.mkdir(parents=True)
    (out / 'inputs').mkdir()
    for name in ('scene.json', 'exercise-task.json', 'manifest.json'):
        shutil.copyfile(run / name, out / 'inputs' / name)
    write_json(out / 'reference-input.json', reference)
    write_json(out / 'diagnostic.json', diagnostic)
    implementation = out / 'implementation'; implementation.mkdir()
    for name, module in sorted(sys.modules.copy().items()):
        source = getattr(module, '__file__', None)
        if name.startswith('weft.') and source and source.endswith('.py'):
            target = implementation / (name.replace('.', '/') + '.py')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    shutil.copyfile(__file__, implementation / 'audit_camera_reference_dynamics.py')
    report = {'created_at': local_time(), 'profile': diagnostic['profile'], 'status': 'diagnostic_only',
        'source_run': str(run), 'source_manifest': verified, 'scene_sha256': canonical_hash(scene),
        'body_sha256': canonical_hash(scene['human_spec']), 'task_sha256': canonical_hash(task),
        'gates_sha256': task['gates_sha256'], 'mujoco_version': mujoco.__version__, 'summary': diagnostic['summary'],
        'inverse_calls': diagnostic['inverse_calls'], 'physics_steps': 0, 'inference_calls': 0, 'api_cost_usd': 0,
        'elapsed_seconds': time.perf_counter() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'physical_validation': 'unverified', 'source_and_task_unchanged': True}
    write_json(out / 'report.json', report)
    (out / 'index.html').write_text(render_report({'diagnostic': diagnostic}), encoding='utf-8')
    write_json(out / 'manifest.json', {'schema_version': 1, 'created_at': local_time(),
        'files': {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob('*')) if path.is_file()}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.run, args.out), sort_keys=True))


if __name__ == '__main__':
    main()
