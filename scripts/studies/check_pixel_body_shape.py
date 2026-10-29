"""Frozen four-source static pixel-length comparison; no dynamics or promotion."""

import json
from pathlib import Path
import shutil
import signal
import time

import cv2

from weft.benchmark import seal_suite
from weft.body_shape_diagnostic import diagnose_shape
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.pixel_body_shape import CONTRACT, fit_pixel_shape
from weft.whole_body_feasibility import audit_reference_feasibility


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/pixel-body-shape-01'
CASES = {'fullbody': ROOT / 'artifacts/fullbody-exercise-01/baseline/run'} | {
    name: ROOT / 'artifacts/mobile-exercise-e2e-10' / name / 'run'
    for name in ('single-leg-squat', 'squat', 'kettlebell-pushup')}


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second study bound')))
    signal.alarm(60)
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    files = ['src/weft/pixel_body_shape.py', 'src/weft/body_shape_diagnostic.py',
             'src/weft/body_refinement.py', 'src/weft/human.py', 'src/weft/mobile_human.py',
             'src/weft/body_frame.py', 'src/weft/spatial_pose.py', 'src/weft/whole_body_feasibility.py',
             'tests/test_pixel_body_shape.py', 'scripts/studies/check_pixel_body_shape.py']
    hashes = {name: digest(ROOT / name) for name in files}
    for name in files:
        dest = OUT / 'implementation' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    write_json(OUT / 'pre-run.json', {'created_at': local_time(), 'contract': CONTRACT,
        'cases': {name: str(path.relative_to(ROOT)) for name, path in CASES.items()},
        'implementation_hashes': hashes, 'analytic_tests_passed': 16, 'maximum_seconds': 60,
        'new_api_calls': 0, 'cost_usd': 0,
        'gate': 'No promotion; report all later-pose errors and fixed-root/floor feasibility. No dynamics. Independent geometry labels still required.',
        'chronology': 'Pixel-length hypothesis follows rejected world-length fits; all four sources are development data, not final holdouts.'})
    summaries = {}
    for name, run in CASES.items():
        out = OUT / name
        out.mkdir()
        verification = verify_artifacts(run)
        names = ['human.json', 'pose-landmarks.json', 'root-motion.json', 'source.json', 'manifest.json']
        input_hashes = {key: digest(run / key) for key in names}
        for key in names:
            shutil.copyfile(run / key, out / ('input-' + key))
        spec = json.loads((run / 'human.json').read_text())
        raw = json.loads((run / 'pose-landmarks.json').read_text())
        root = json.loads((run / 'root-motion.json').read_text())
        source = json.loads((run / 'source.json').read_text())
        video = run / source['artifact']
        assert digest(video) == spec['source_sha256'] == source['sha256']
        cap = cv2.VideoCapture(str(video))
        fps, decoded, dimensions = cap.get(cv2.CAP_PROP_FPS), 0, set()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                decoded += 1
                dimensions.add((frame.shape[1], frame.shape[0]))
                assert decoded <= 1800
        finally:
            cap.release()
        assert len(dimensions) == 1
        width, height = next(iter(dimensions))
        write_json(out / 'inputs.json', {'verification': verification, 'hashes': input_hashes,
            'decoded_frames': decoded, 'fps': fps, 'dimensions': [width, height],
            'source_video': str(video.relative_to(ROOT)), 'source_sha256': spec['source_sha256']})
        diagnosis = diagnose_shape(spec, raw | {'source_sha256': spec['source_sha256']}, width=width,
            height=height, fps=fps, frame_count=decoded,
            model_units_per_pixel=root['scale']['model_units_per_pixel'])
        write_json(out / 'diagnosis.json', diagnosis)
        result = fit_pixel_shape(spec, diagnosis)
        write_json(out / 'shape.json', result)
        write_json(out / 'candidate.json', result['candidate'])
        floors = {key: audit_reference_feasibility(value) for key, value in
                  [('baseline', spec), ('candidate', result['candidate'])]}
        write_json(out / 'floor-comparison.json', floors)
        assert input_hashes == {key: digest(run / key) for key in names}
        summaries[name] = {'projection': result['summary'], 'floor': {key: value['summary'] for key, value in floors.items()},
            'length_ratios': {key: value['bounded_ratio'] for key, value in result['estimates'].items()},
            'retained_poses': len(result['rows']), 'inputs_unchanged': True, 'promoted': False}
        print(json.dumps({name: summaries[name]}), flush=True)
    assert hashes == {name: digest(ROOT / name) for name in hashes}
    write_json(OUT / 'report.json', {'created_at': local_time(), 'cases': summaries,
        'seconds': time.monotonic() - started, 'implementation_unchanged': True,
        'new_api_calls': 0, 'cost_usd': 0, 'promoted': False,
        'scope': 'Development model-output comparisons and sampled collision consistency, not verified anatomy or physical transfer'})
    signal.alarm(0)
    print(json.dumps(seal_suite(OUT)), flush=True)


if __name__ == '__main__':
    main()
