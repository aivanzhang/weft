"""Bounded local timing of the early-motion proposer, not browser performance."""

from copy import deepcopy
from pathlib import Path
import shutil
import statistics
import time

from weft.benchmark import seal_suite
from weft.initial_motion import propose_initial_motion
from weft.io import write_json, local_time
from weft.scene import validate_scene

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/initial-motion-performance-01'


def main():
    OUT.mkdir(exist_ok=False)
    template = {'schema_version': 1, 'profile': 'planar-rigid-v1', 'width': 1., 'height': .7,
        'dt': 1/30, 'gravity': [0., 0.], 'damping': 1., 'restitution': .5,
        'controlled_id': 'body0', 'target': [.9, .3], 'relationships': [], 'assumptions': ['Synthetic performance control'],
        'task': {'position_tolerance': .06, 'speed_tolerance': .15, 'max_steps': 300, 'acceleration': 2.}}
    results = {}
    for name, count, samples, fps in [('small', 1, 20, 10), ('large', 8, 1080, 240)]:
        scene = deepcopy(template)
        scene['objects'] = [{'id': f'body{i}', 'label': f'Synthetic {i}', 'role': 'active' if i == 0 else 'passive',
            'shape': 'circle', 'radius': .02, 'position': [.1 + .2 * (i % 4), .2 + .3 * (i // 4)],
            'velocity': [.025, 0.], 'mass': 1., 'color': '#747474', 'provenance': {'scope': 'Analytic input'}} for i in range(count)]
        scene = validate_scene(scene)
        rows = [{'frame': index, 'time_s': index / fps, 'object_id': obj['id'], 'id': f'{index}:{obj["id"]}',
                 'x': obj['position'][0] + .02 * index / fps, 'y': obj['position'][1]}
                for index in range(samples) for obj in scene['objects']]
        times = []
        for _ in range(3):
            started = time.perf_counter()
            candidate, audit = propose_initial_motion(scene, rows, width=960, origin_time_s=0.)
            times.append(time.perf_counter() - started)
            assert candidate is not None and len(audit['changes']) == count
            assert all(abs(v[0] - .02) < 1e-10 and abs(v[1]) < 1e-10 for v in audit['changes'].values())
        write_json(OUT / (name + '-audit.json'), audit)
        results[name] = {'objects': count, 'construction_frames': samples, 'input_rows': len(rows),
                         'fps': fps, 'measurements_seconds': times, 'median_seconds': statistics.median(times),
                         'maximum_seconds': max(times), 'all_analytic_velocities_verified': True}
    shutil.copyfile(ROOT / 'src/weft/initial_motion.py', OUT / 'initial_motion.py')
    shutil.copyfile(__file__, OUT / 'measure_initial_motion.py')
    write_json(OUT / 'report.json', {'created_at': local_time(), 'cases': results, 'new_api_calls': 0,
        'scope': 'Three local calls per prepared workload; large case is the 60% construction prefix of 1800 frames with eight objects at 240 fps. Not sustained CPU/RSS or UI performance.'})
    print(seal_suite(OUT))


if __name__ == '__main__':
    main()
