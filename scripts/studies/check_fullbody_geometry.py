"""Fresh full-body source geometry diagnosis, without floor or policy evaluation."""
import json
from pathlib import Path
import shutil
import signal
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.body_shape_diagnostic import diagnose_shape
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/fullbody-geometry-diagnosis-01'
RUN = ROOT / 'artifacts/fullbody-exercise-01/baseline/run'
VIDEO = ROOT / 'data/exercise/bodyweight-squats-danielflefil-500.mp4'


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second study bound')))
    signal.alarm(60)
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    verification = verify_artifacts(RUN)
    inputs = {name: digest(RUN / name) for name in ('human.json', 'pose-landmarks.json', 'root-motion.json', 'manifest.json')}
    code = ['src/weft/body_shape_diagnostic.py', 'src/weft/body_refinement.py', 'src/weft/human.py',
            'src/weft/mobile_human.py', 'src/weft/body_frame.py', 'src/weft/spatial_pose.py',
            'tests/test_body_shape_diagnostic.py', 'tests/test_whole_body_feasibility.py',
            'scripts/studies/check_fullbody_geometry.py']
    hashes = {name: digest(ROOT / name) for name in code}
    for name in code:
        destination = OUT / 'implementation' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    (OUT / 'inputs').mkdir()
    for name in inputs:
        shutil.copyfile(RUN / name, OUT / 'inputs' / name)
    spec = json.loads((RUN / 'human.json').read_text())
    raw = json.loads((RUN / 'pose-landmarks.json').read_text())
    root_audit = json.loads((RUN / 'root-motion.json').read_text())
    source_sha = digest(VIDEO)
    assert source_sha == spec['source_sha256'] == root_audit['source_sha256']
    cap = cv2.VideoCapture(str(VIDEO))
    fps, metadata_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded, dimensions = 0, set()
    while True:
        ok, image = cap.read()
        if not ok:
            break
        decoded += 1
        dimensions.add((image.shape[1], image.shape[0]))
        assert decoded <= 1800
    cap.release()
    assert decoded == metadata_frames == 68 and len(dimensions) == 1
    width, height = next(iter(dimensions))
    config = {'source_sha256': source_sha, 'source_relative_path': str(VIDEO.relative_to(ROOT)),
              'actual_decoded_frames': decoded, 'actual_dimensions': [width, height], 'fps': fps,
              'source_manifest_verification': verification, 'input_hashes': inputs, 'implementation_hashes': hashes,
              'metadata_adapter': 'Retained pose-landmarks.json unchanged; source_sha256 added only to in-memory adapter from verified original spec/video',
              'scale': root_audit['scale'], 'scope': 'Construction-source diagnostics only; heldout means shape-fit prefix split, not new physical ground truth or final holdout',
              'maximum_seconds': 60, 'new_api_calls': 0, 'new_cost_usd': 0}
    write_json(OUT / 'config.json', config)
    report = diagnose_shape(spec, raw | {'source_sha256': source_sha}, width=width, height=height,
                            fps=fps, frame_count=decoded, model_units_per_pixel=root_audit['scale']['model_units_per_pixel'])
    assert hashes == {name: digest(ROOT / name) for name in code}
    assert inputs == {name: digest(RUN / name) for name in inputs}
    report['study'] = config | {'runtime_seconds': time.monotonic() - started, 'inputs_unchanged': True, 'code_unchanged': True}
    report['per_joint_summary'] = {}
    for joint in report['rows'][0]['comparisons']['baseline']['world_endpoint_errors']:
        report['per_joint_summary'][joint] = {}
        for partition in ('training', 'heldout'):
            subset = [row for row in report['rows'] if row['partition'] == partition and row['status'] == 'compared']
            report['per_joint_summary'][joint][partition] = {
                variant: {coordinate: float(np.sqrt(np.mean([row['comparisons'][variant][coordinate + '_endpoint_errors'][joint] ** 2 for row in subset])))
                          for coordinate in ('world', 'pixel')} for variant in ('baseline', 'lengths', 'attachments', 'both')}
    report['largest_baseline_endpoint_errors'] = sorted(
        [{'frame': row['frame'], 'partition': row['partition'], 'joint': joint, 'error_model_units': error,
          'confidence': row['effective_confidence'][joint]}
         for row in report['rows'] if row['status'] == 'compared'
         for joint, error in row['comparisons']['baseline']['world_endpoint_errors'].items()],
        key=lambda row: row['error_model_units'], reverse=True)[:12]
    write_json(OUT / 'report.json', report)
    signal.alarm(0)
    print(json.dumps(seal_suite(OUT)))
    print(json.dumps({'summary': report['summary'], 'decomposition': report['decomposition_summary'],
                      'largest': report['largest_baseline_endpoint_errors'][:3]}, indent=2))


if __name__ == '__main__':
    main()
