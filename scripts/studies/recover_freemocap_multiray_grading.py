"""Finish the failed report export from verified predictions; do not repeat geometry."""

import argparse
import json
from pathlib import Path
import shutil
import time

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files
from weft.projected_joint_evaluation import grade_projected_joints


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    previous = ROOT/'artifacts/freemocap-multiray-verification-01'
    manifest = read(previous/'manifest.json')
    for name, expected in manifest['files'].items(): assert digest(previous/name) == expected
    protocol = read(previous/'pre-run.json'); failure = read(previous/'report.json')
    assert failure['status'] == 'grading_export_key_error_after_verification'
    assert digest(previous/'predictions-before-grading.json') == protocol['prediction_sha256']
    assert digest(ROOT/'src/weft/projected_joint_evaluation.py') == protocol['scorer_sha256']
    source = ROOT/'artifacts/freemocap-three-view-02'
    runtime = {n: digest(p) for n, p in _implementation_files().items()}
    assert runtime == read(source/'pre-run.json')['runtime_hashes']
    for name, record in protocol['source_manifests'].items():
        assert digest(ROOT/'artifacts'/name/'manifest.json') == record['sha256']
    observations_path = ROOT/'artifacts/freemocap-third-comparison-01/all-comparison-image-observations.json'
    assert digest(observations_path) == protocol['comparison_observations_sha256']
    observations = read(observations_path); predictions = read(previous/'predictions-before-grading.json')
    if out.parent != ROOT/'artifacts': raise ValueError('Use a new artifact directory')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    shutil.copyfile(previous/'predictions-before-grading.json', out/'predictions-before-grading.json')
    write_json(out/'pre-run.json', protocol | {'recovery_created_at': local_time(),
        'verified_parent': str(previous.relative_to(ROOT)), 'parent_manifest_sha256': digest(previous/'manifest.json'),
        'geometry_checks_repeated': False, 'new_fits': 0})
    summaries = {}; started = time.perf_counter()
    for name in ('complete_source', 'active_400_through_1059'):
        selected = [i for i, r in enumerate(observations) if name == 'complete_source' or 400 <= r['frame'] <= 1059]
        result = grade_projected_joints({k: [v[i] for i in selected] for k, v in predictions.items()},
                                      [observations[i] for i in selected], baseline='raw_two_view', limit_px=12.8)
        summaries[name] = {k: v for k, v in result.items() if k != 'all_slot_errors'}
        write_json(out/f'{name}-errors.json', result['all_slot_errors'])
    total = sum(p.stat().st_size for p in out.rglob('*') if p.is_file()); assert total < protocol['maximum_output_bytes']
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'recovered_two_view_pixel_development_comparison',
        'comparison': summaries, 'prior_verification': failure, 'complete_source_slots': 1108, 'joint_slots': 36564,
        'new_geometry_checks': 0, 'new_fits': 0, 'new_inference_calls': 0, 'new_video_decodes': 0, 'paid_calls': 0,
        'output_bytes': total, 'grading_seconds': time.perf_counter()-started, 'candidate_promoted': False,
        'body_built': False, 'physical_validation': 'unverified'})
    print(json.dumps({'comparison': summaries, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
