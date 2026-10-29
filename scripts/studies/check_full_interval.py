"""Full-interval development checks; never replace physical evidence with synthetic passes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
import time

from weft.benchmark import run_suite, seal_suite
from weft.generic_vision import inspect_generic
from weft.io import digest, local_time, write_json
from weft.reconstruction_coverage import coverage_summary, plan_reconstruction
from weft.scene import validate_scene
from weft.scene_vision import fit_scene, _reconstruction_frames

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/full-interval-reconstruction-01'


def config(video, profile):
    world = {'controlled_id': 'actor', 'target': [.85, .3], 'gravity': [0, 0], 'damping': 1., 'restitution': .5}
    if profile == 'planar':
        return {**world, 'objects': [{'id': 'actor', 'hue_min': 20, 'hue_max': 40, 'role': 'active', 'shape': 'circle'}]}
    return {'world': world, 'tracking': {'schema_version': 1, 'source_sha256': digest(video),
        'start_frame': 0, 'end_frame': 199, 'objects': [{'id': 'actor', 'label': 'Synthetic texture',
        'box_xywh': [30, 40, 24, 28], 'role': 'active', 'shape': 'box', 'provenance': 'Analytic source rectangle'}]}}


def large_replay():
    target = OUT / 'large-replay'; target.mkdir()
    objects = [{'id': f'body{i}', 'label': f'Synthetic body {i}', 'role': 'active' if i == 0 else 'passive',
        'shape': 'circle', 'radius': .02, 'position': [.1 + .2 * (i % 4), .2 + .3 * (i // 4)],
        'velocity': [0., 0.], 'mass': 1., 'color': '#747474', 'provenance': {'scope': 'Synthetic stationary control'}}
        for i in range(8)]
    scene = validate_scene({'schema_version': 1, 'profile': 'planar-rigid-v1', 'width': 1., 'height': .7,
        'dt': 1/30, 'gravity': [0., 0.], 'damping': 1., 'restitution': .5, 'objects': objects,
        'controlled_id': 'body0', 'target': [.9, .3], 'relationships': [], 'assumptions': ['Synthetic performance control'],
        'task': {'position_tolerance': .06, 'speed_tolerance': .15, 'max_steps': 300, 'acceleration': 2.}})
    rows = [{'frame': index, 'time_s': index/30, 'object_id': obj['id'], 'id': f'{index}:{obj["id"]}',
             'x': obj['position'][0], 'y': obj['position'][1]} for index in range(1800) for obj in objects]
    started = time.monotonic()
    coverage, generation, development = plan_reconstruction(rows, [o['id'] for o in objects], [0, 1799], 30)
    evidence = {'fps': 30, 'width': 960, 'origin_time_s': 0., 'initialization_frame': 0,
                'generation': generation, 'development_validation': development, 'reconstruction_coverage': coverage}
    planned_seconds = time.monotonic() - started
    started = time.monotonic()
    frames = _reconstruction_frames(evidence, scene)
    replay_seconds = time.monotonic() - started
    assert len(frames) == 1800 and frames[-1]['simulation_time_s'] == 1799/30
    assert all(row['observed'][i]['position'] == row['predicted'][i]['position'] for row in frames for i in range(8))
    write_json(target / 'scene.json', scene)
    write_json(target / 'evidence.json', evidence)
    write_json(target / 'frames.json', frames)
    result = {'frames': 1800, 'objects': 8, 'positions_verified': 14400, 'physics_steps': 1799,
              'plan_seconds': planned_seconds, 'replay_seconds': replay_seconds,
              'serialized_frame_bytes': (target / 'frames.json').stat().st_size,
              'scope': 'One local maximum-frame/object stationary workload; not sustained UI, RSS, or general dynamics performance'}
    write_json(target / 'report.json', result)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    fixtures = runpy.run_path(str(ROOT / 'tests/test_reconstruction_coverage.py'))
    files = ['src/weft/reconstruction_coverage.py', 'src/weft/scene_vision.py', 'src/weft/generic_vision.py',
             'src/weft/ball_benchmark.py', 'src/weft/assets/lab.html', 'src/weft/assets/lab.js',
             'tests/test_reconstruction_coverage.py', 'tests/test_generic_vision.py', 'tests/lab_ui_smoke.mjs',
             'scripts/studies/check_full_interval.py']
    hashes = {name: digest(ROOT / name) for name in files}
    for name in files:
        path = OUT / 'implementation' / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    write_json(OUT / 'pre-run.json', {'created_at': local_time(), 'hypothesis': 'Every later observation reaches unchanged replay gates; no reset at gaps or shortened successful prefix',
        'bounds': {'lab_cases': 4, 'parallel_workers': 2, 'training_episodes_per_ready_case': 200,
                   'evaluation_episodes_per_policy': 10, 'worker_seconds_per_case': 360,
                   'late_motion_cases': 2, 'large_replay_frames': 1800, 'large_replay_objects': 8},
        'api_calls': 0, 'external_spend_usd': 0, 'recurring_services': 0,
        'implementation_hashes': hashes, 'scope': 'Development implementation check; original physical accuracy still unverified'})
    inputs = OUT / 'inputs'; inputs.mkdir()
    cases, negative = [], {}
    for profile in ('appearance', 'planar'):
        for changed in (False, True):
            name = profile + ('-late-motion' if changed else '-constant')
            video = inputs / (name + '.mp4')
            fixtures['write_late_motion'](video, profile, changed=changed)
            settings = config(video, profile)
            config_path = inputs / (name + '.json'); write_json(config_path, settings)
            if changed:
                run = OUT / name; run.mkdir()
                evidence, scene = fixtures['inspect_fixture'](video, run, profile)
                fit_scene(evidence, scene, run)
                report = json.loads((run / 'reconstruction.json').read_text())
                assert not report['replay_gate']['passes'] and len(report['frames']) == 200
                negative[name] = {'coverage': coverage_summary(evidence['reconstruction_coverage']),
                                  'replay_gate': report['replay_gate'], 'after': report['after']}
                continue
            cases.append({'id': name, 'profile': profile, 'video': str(video), 'config': str(config_path),
                'source_sha256': digest(video), 'source_kind': 'analytic_synthetic',
                'rights': 'Weft-authored deterministic test fixture; not a physical trial',
                'conditions': ['20-second constant translation', 'brief missing interval'],
                'episodes': 200, 'eval_episodes': 10, 'expected': {'status': 'ready'}, 'gates': [
                    {'metric': 'reconstruction_rmse_px', 'operator': 'max', 'value': 10},
                    {'metric': 'action_effect', 'operator': 'min', 'value': .001},
                    {'metric': 'recorded_transitions', 'operator': 'min', 'value': 12}]})
    original_config = ROOT / 'data/general/annotations/rolling-grayscale.scene.json'
    rolling = json.loads(original_config.read_text())
    for extended in (False, True):
        name = 'real-rolling-full-suffix' if extended else 'real-rolling-declared-pass'
        settings = deepcopy(rolling)
        if extended:
            settings['tracking']['end_frame'] = 156
            settings['source_label'] += ' New development attempt explicitly extends through source frame 156; no automatic reinitialization.'
        path = inputs / (name + '.json'); write_json(path, settings)
        cases.append({'id': name, 'profile': 'appearance', 'video': str(ROOT / 'data/general/rolling-ball-grayscale.avi'),
            'config': str(path), 'source_sha256': settings['tracking']['source_sha256'], 'source_kind': 'physical_recording',
            'rights': rolling['source_label'], 'conditions': ['Known CC0 grayscale derivative', 'Full declared interval, not new physical evidence'],
            'episodes': 200, 'eval_episodes': 10,
            'expected': {'status': 'rejected', 'error_contains': 'covers less than 80%'} if extended else {'status': 'ready'},
            'gates': [] if extended else [{'metric': 'reconstruction_rmse_px', 'operator': 'max', 'value': 10},
                                        {'metric': 'action_effect', 'operator': 'min', 'value': .001},
                                        {'metric': 'recorded_transitions', 'operator': 'min', 'value': 12}]})
    suite_path = OUT / 'suite.json'
    write_json(suite_path, {'schema_version': 1, 'name': 'Full-interval reconstruction development check',
        'purpose': 'Complete declared source intervals, bounded learning, action effects and recording. No physical accuracy promotion.', 'cases': cases})
    with ThreadPoolExecutor(max_workers=1) as executor:
        large = executor.submit(large_replay)
        suite = run_suite(suite_path, OUT / 'labs', workers=2)
        performance = large.result()
    assert hashes == {name: digest(ROOT / name) for name in files}
    write_json(OUT / 'report.json', {'created_at': local_time(), 'negative_controls': negative,
        'suite': suite, 'large_replay': performance, 'implementation_unchanged': True,
        'new_api_calls': 0, 'external_spend_usd': 0, 'scope': 'Full requested-frame coverage is now inspectable; no claim of general physical recovery'})
    print(json.dumps(seal_suite(OUT)), flush=True)


if __name__ == '__main__':
    main()
