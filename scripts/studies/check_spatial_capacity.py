"""Measure a bounded twelve-object workload without inference, training or external calls."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import resource
import shutil
import time

import mujoco
import numpy as np
from PIL import Image

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.spatial_rigid import PROFILE, UNITS, SpatialRigidEnv, render_snapshot, validate_scene, validate_snapshot


ROOT = Path(__file__).resolve().parents[2]


def fixture():
    common = {'shape': 'box', 'quaternion': [1, 0, 0, 0], 'velocity': [0, 0, 0],
              'angular_velocity': [0, 0, 0], 'mass': 1, 'friction': [.25, .001, .0001],
              'color': '#AB7E48', 'provenance': {'scope': 'synthetic capacity fixture, not physical evidence'}}
    objects = []
    for i in range(11):
        obj = deepcopy(common) | {'id': f'body-{i}', 'label': f'Body {i}',
            'role': 'active' if i == 0 else 'passive',
            'position': [.2+.2*(i%4), .2+.2*(i//4), .12+.03*(i%3)]}
        if i%2:
            obj.update(shape='sphere', radius=.02, color='#487653')
        else:
            obj['size'] = [.04, .04, .04]
        objects.append(obj)
    objects.append(deepcopy(common) | {'id': 'floor', 'label': 'Floor', 'role': 'static',
        'position': [.5, .4, .02], 'size': [.98, .78, .04]})
    return {'schema_version': 1, 'profile': PROFILE, 'units': deepcopy(UNITS),
        'bounds': [1, .8, .8], 'dt': 1/120, 'gravity': [0, 0, -9.81],
        'contact': {'time_constant': .01, 'damping_ratio': 1}, 'objects': objects,
        'controlled_id': 'body-0', 'target': [.9, .7, .06],
        'task': {'position_tolerance': .025, 'speed_tolerance': .05, 'acceleration': 5, 'max_steps': 900},
        'assumptions': ['Synthetic eleven-body drop onto one support; not a reconstruction or worst-case solver test']}


def stats(values):
    return {'count': len(values), 'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)), 'max': max(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT / 'artifacts':
        raise ValueError('Use a new top-level artifacts directory')
    scene = validate_scene(fixture())
    excess = deepcopy(scene)
    excess['objects'].append(deepcopy(scene['objects'][1]) | {'id': 'extra'})
    try:
        validate_scene(excess)
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError('Thirteen objects must be rejected')
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    write_json(out / 'scene.json', scene)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'implementation': implementation,
        'max_steps': 600, 'wall_limit_s': 15, 'workers': 1, 'paid_calls': 0,
        'python': platform.python_version(), 'mujoco': mujoco.__version__,
        'scope': 'In-process CPU workload; not browser, HTTP, idle or leak profiling'})
    env = SpatialRigidEnv(scene)
    env.reset(seed=19)
    first = env.snapshot()
    write_json(out / 'initial.json', first)
    timings, rows = [], []
    started, cpu_started = time.perf_counter(), time.process_time()
    for index in range(600):
        if time.perf_counter()-started > 15:
            raise TimeoutError('Capacity run exceeds its frozen wall limit')
        action = (2, 1, 3, 4, 5, 6, 7, 0)[index%8]
        step_started = time.perf_counter()
        observation, reward, terminated, truncated, _ = env.step(action)
        state = env.snapshot()
        payload = json.dumps(state, allow_nan=False, separators=(',', ':')).encode()
        timings.append((time.perf_counter()-step_started)*1000)
        validate_snapshot(state, scene)
        assert len(state['objects']) == 12 and observation.shape == (160,)
        assert env.observation_space.contains(observation) and np.isfinite(reward)
        assert not terminated and not truncated
        rows.append({'step': state['steps'], 'action': action, 'contacts': state['contact_count'],
            'substeps': state['substeps'], 'json_bytes': len(payload), 'step_snapshot_json_ms': timings[-1]})
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    write_json(out / 'final.json', state)
    render_started = time.perf_counter()
    preview = render_snapshot(state, scene, width=1280)
    render_ms = (time.perf_counter()-render_started)*1000
    Image.fromarray(preview).save(out / 'final.png')
    env.reset(seed=19)
    assert env.snapshot() == first
    env.close()
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Implementation changed during the workload')
    write_json(out / 'samples.json', rows)
    report = {'status': 'complete', 'created_at': local_time(), 'object_count': 12,
        'dynamic_bodies': 11, 'steps': len(rows), 'simulated_seconds': len(rows)*scene['dt'],
        'wall_seconds': wall, 'process_cpu_seconds': cpu, 'step_snapshot_json_ms': stats(timings),
        'maximum_contacts': max(row['contacts'] for row in rows),
        'maximum_substeps': max(row['substeps'] for row in rows),
        'maximum_json_bytes': max(row['json_bytes'] for row in rows), 'single_render_ms': render_ms,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'rss_scope': 'macOS lifetime peak including imports; no sustained leak claim',
        'exact_reset': True, 'thirteenth_object_rejection': rejected, 'paid_calls': 0,
        'limits': ['One local process and one synthetic fixture', 'Not worst-case contacts or speed',
                   'No browser/HTTP/idle measurement', 'No physical-accuracy evidence']}
    write_json(out / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
