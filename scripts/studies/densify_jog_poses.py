"""Bounded missing-frame inference on two construction views; no comparison inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from weft.benchmark import seal_suite
from weft.exercise_vision import model_provenance
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
POSES = [ROOT / f'artifacts/full-pose-comparison-01/jog-drone{i}' for i in (1, 2)]
VIDEOS = [ROOT / 'artifacts/multiview-jog-source-01/drone1.mp4',
          ROOT / 'artifacts/multiview-jog-extra-views-02/drone2.mp4']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    inputs = [*POSES, *(video.parent for video in VIDEOS)]
    if any(out.is_relative_to(path) for path in inputs):
        raise ValueError('Choose a new output outside sealed sources')
    before = {str(path): verify_artifacts(path) for path in inputs}
    model = ROOT / '.local/models/pose_landmarker_full.task'
    provenance = model_provenance(model)
    cached = [json.loads((path/'evidence.json').read_text()) for path in POSES]
    for evidence, video in zip(cached, VIDEOS, strict=True):
        if evidence['source_sha256'] != digest(video) or evidence['frame_count'] != 240 or len(evidence['frames']) != 120:
            raise ValueError('The bounded study requires the complete pinned 240-frame source and 120 cached samples')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': before, 'model': provenance,
        'implementation': implementation, 'workers': 2, 'timeout_seconds_each': 95,
        'maximum_new_inference_calls': 240, 'reused_observations': 240, 'stride': 1,
        'hypothesis': 'Additional original image samples can supply temporal anchors absent from the 15-Hz evidence',
        'excluded': ['drone3', 'camera calibration', 'provider joints', 'comparison labels', 'ground fitting'],
        'projected_external_cost_usd': 0, 'paid_requests': 0, 'default_changed': False,
        'no_pose_retry_or_identity_selection': True})
    def run(index):
        label = f'drone{index+1}'
        started = time.perf_counter()
        with (out/(label+'.stdout')).open('w') as stdout, (out/(label+'.stderr')).open('w') as stderr:
            try:
                child = subprocess.run([sys.executable, '-m', 'weft.pose_densify', '--run', str(POSES[index]),
                    '--video', str(VIDEOS[index]), '--model', str(model), '--out', str(out/label)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, timeout=95, check=False)
                code = child.returncode
            except subprocess.TimeoutExpired:
                code = None
        return label, {'exit_code': code, 'wall_seconds': time.perf_counter()-started}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = dict(pool.map(run, (0, 1)))
    elapsed = time.perf_counter()-start
    for index, label in enumerate(('drone1', 'drone2')):
        if results[label]['exit_code'] != 0: continue
        path = out/label
        results[label]['verification'] = verify_artifacts(path)
        results[label]['report'] = json.loads((path/'report.json').read_text())
        evidence = json.loads((path/'evidence.json').read_text())
        rows = {row['frame']: row for row in evidence['frames']}
        if sorted(rows) != list(range(240)) or any(canonical_hash(rows[row['frame']]) != canonical_hash(row)
                for row in cached[index]['frames']):
            raise ValueError('Densification changed a cached observation or omitted a source frame')
    if any(verify_artifacts(Path(path)) != value for path, value in before.items()):
        raise ValueError('A pinned source changed')
    if {name: digest(path) for name, path in _implementation_files().items()} != implementation:
        raise ValueError('Implementation changed during inference')
    complete = all(result['exit_code'] == 0 for result in results.values())
    report = {'created_at': local_time(), 'status': 'completed_densification' if complete else 'inference_failed',
        'results': results, 'wall_seconds': elapsed, 'paid_requests': 0, 'new_body_or_policy': False,
        'all_sources_implementation_unchanged': True,
        'scope': 'Two views of the same source; additional temporal samples, not new video diversity or physical validation'}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))
    if not complete: raise SystemExit(1)


if __name__ == '__main__':
    main()
