"""Freeze source-only ramp labels, then run unchanged bounded appearance tracking."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual, validate_config
from weft.io import digest, local_time, write_json
from weft.observations import adapt_appearance
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/phys101-ramp-source-01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.is_relative_to(SOURCE): raise ValueError('Output must not alter source files')
    source_before = verify_artifacts(SOURCE)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out/'inputs'; inputs.mkdir()
    shutil.copyfile(__file__, inputs/Path(__file__).name)
    boxes = {10: [[193, 374, 75, 69], [939, 478, 122, 43]],
             20: [[200, 254, 77, 73], [938, 465, 118, 44]]}
    centers = {10: [(12, [230, 408], [1000, 500]), (20, [230, 408], [1000, 500]), (26, [230, 408], [1000, 500])],
               20: [(16, [389, 336], [997, 487]), (27, [824, 472], [997, 487]), (43, [965, 469], [1117, 488])]}
    configs, annotations = {}, {}
    for i, angle in enumerate((10, 20)):
        video = SOURCE/f'member-{i:02d}-Camera_1.mp4'
        objects = [{'id': identifier, 'label': label, 'box_xywh': box, 'role': 'unknown', 'shape': 'unknown',
                    'provenance': 'Manually estimated source-frame-8 appearance region; not physical geometry or a controlled role'}
                   for identifier, label, box in zip(('wood-block', 'green-box'), ('w_block_01 wooden block', 'Green cardboard box'), boxes[angle], strict=True)]
        config = validate_config({'schema_version': 1, 'source_sha256': digest(video), 'start_frame': 0,
            'end_frame': 26 if angle == 10 else 43, 'initialization_frame': 8, 'objects': objects})
        configs[angle] = config
        annotations[angle] = {'source_sha256': digest(video), 'coordinate_system': 'original_source_pixels',
            'uncertainty_radius_source_px': 12, 'annotator': 'Assistant visual source-only development labels',
            'scope': 'Approximate visible envelope centers, not physical centers of mass or independent laboratory truth',
            'annotations': [{'frame': frame, 'object_id': identifier, 'visible': True, 'center_xy': point}
                for frame, wood, green in centers[angle] for identifier, point in [('wood-block', wood), ('green-box', green)]]}
        write_json(inputs/f'config-{angle}.json', config)
        write_json(inputs/f'labels-{angle}.json', annotations[angle])
    for path in sorted((ROOT/'.local').glob('phys101-*.png')):
        shutil.copyfile(path, inputs/path.name)
    write_json(inputs/'report.json', {'created_at': local_time(), 'status': 'source_only_labels_frozen',
        'source_verification': source_before, 'initialization_frame': 8, 'full_source_intervals': [[0, 26], [0, 43]],
        'preinitialization': 'All source frames retained; object estimates before frame8 remain unsampled',
        'labels_per_clip': 6, 'label_uncertainty_source_px': 12, 'tracker_defaults_unchanged': True,
        'not_modeled': ['hand release', 'inclined supporting ramp', 'friction or physical scale', 'contact forces'],
        'paid_requests': 0})
    frozen = seal_suite(inputs)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': frozen, 'implementation': implementation,
        'workers': 2, 'timeout_seconds_each': 30, 'paid_requests': 0, 'training': False,
        'scope': 'Complete-source appearance tracking and separate sparse development grading; not dynamics reconstruction'})

    def run(item):
        index, angle = item
        started = time.monotonic()
        command = [sys.executable, '-m', 'weft.generic_vision', '--video', str(SOURCE/f'member-{index:02d}-Camera_1.mp4'),
                   '--config', str(inputs/f'config-{angle}.json'), '--out', str(out/f'tracking-{angle}')]
        with (out/f'{angle}.stdout').open('w') as stdout, (out/f'{angle}.stderr').open('w') as stderr:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=30, check=False)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = None
        return angle, {'exit_code': code, 'wall_seconds': time.monotonic()-started}

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(run, enumerate((10, 20))))
    for angle, result in results.items():
        if result['exit_code'] != 0: continue
        path = out/f'tracking-{angle}'
        trace = json.loads((path/'appearance-tracks.json').read_text())
        if trace['selected_interval'] != [0, configs[angle]['end_frame']]:
            raise ValueError('Tracking dropped source frames')
        grade = evaluate_manual(trace, annotations[angle])
        grade['within_label_uncertainty'] = sum(r.get('error_source_px', float('inf')) <= 12 for r in grade['rows'])
        write_json(path/'manual-evaluation.json', grade)
        observations = adapt_appearance(trace)
        write_json(path/'scene-observations.json', observations)
        result.update(decoded_frames=trace['decoded_frames'], physics_ready=trace['physics_ready'],
            per_object_status={obj['id']: dict(Counter(row['status'] for row in trace['observations'] if row['object_id'] == obj['id'])) for obj in trace['objects']},
            camera_check=trace['camera_check'], manual_grade=grade,
            full_observation_slots=len(observations['observations']))
    if verify_artifacts(SOURCE) != source_before or verify_artifacts(inputs) != frozen:
        raise ValueError('Frozen source or labels changed')
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Implementation changed during tracking')
    report = {'created_at': local_time(), 'status': 'bounded_tracking_study_completed', 'results': results,
        'wall_seconds': time.monotonic()-started, 'paid_requests': 0, 'scene_or_policy_created': False,
        'scope': 'Sparse manual development checks, not a fresh holdout or validated physical reconstruction',
        'all_sources_labels_implementation_unchanged': True}
    write_json(out/'report.json', report)
    print(json.dumps({'results': results, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
