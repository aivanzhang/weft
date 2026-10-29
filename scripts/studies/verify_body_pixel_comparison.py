"""Check source-frame panels and preserved candidate bindings without repeating fits."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

from weft.benchmark import seal_suite
from weft.io import canonical_hash, digest, local_time, write_json
from weft.pipeline import _implementation_files


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/body-pixel-refinement-02'
SOURCE_SEAL = 'edc76d63dae053a5f92f62fb3c5bc460b8748cd4ef2a1a3d9feb1d0fc91218e3'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.is_relative_to(SOURCE):
        raise ValueError('Verification must not change the saved comparison')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    if digest(SOURCE / 'manifest.json') != SOURCE_SEAL:
        raise ValueError('Comparison manifest changed')
    plan, report = read(SOURCE / 'pre-fit-freeze.json'), read(SOURCE / 'report.json')
    if plan['implementation'] != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime differs from the frozen candidate inputs')
    groups, preservation, new_failures = {}, {}, {}
    for name, case in report['cases'].items():
        directory = SOURCE / name
        before, after = read(directory / 'original-human.json'), read(directory / 'construction/candidate-human.json')
        evidence, audit = read(directory / 'evidence.json'), read(directory / 'construction/refinement.json')
        if (canonical_hash(before) != audit['input_model_sha256'] or canonical_hash(after) != audit['output_model_sha256']
                or any(before.get(key) != after.get(key) for key in audit['fixed_fields'])
                or [row['time_s'] for row in before['reference']] != [row['time_s'] for row in after['reference']]):
            raise ValueError('Candidate lost its original model/time/assumption binding')
        for old, new in zip(before['reference_evidence'], after['reference_evidence'], strict=True):
            if any(old.get(key) != new.get(key) for key in ('evidence_id', 'time_s', 'joint_confidence')):
                raise ValueError('Original source identity/confidence changed')
        preservation[name] = {'reference_poses': len(before['reference']), 'source_samples': len(evidence['frames']),
                              'decoded_frames': evidence['decoded_frames'], 'all_fixed_fields_unchanged': True}
        group = groups.setdefault(plan['inputs'][name]['source']['artifact'], {'expected_decoded': evidence['decoded_frames'], 'panels': {}})
        for filename in case['panels']:
            frame = int(Path(filename).stem.rsplit('-', 1)[1])
            group['panels'].setdefault(frame, []).append(directory / filename)
        floor = read(directory / 'candidate-feasibility.json')
        by_time = {row['time_s']: row for row in floor['checks'] if row['kind'] == 'source_pose'}
        new_failures[name] = []
        for failure in case['floor']['changes']['new_floor_failures_on_paired']:
            if failure['kind'] != 'source_pose':
                continue
            fitted = audit['frames'][by_time[failure['time_s']]['source_reference_index']]
            new_failures[name].append({'frame': fitted['frame'], 'weighted_points': fitted['weighted_points'],
                'ankles': [{key: point[key] for key in ('joint', 'source_status', 'source_confidence', 'effective_fit_weight')}
                           for point in fitted['points'] if point['joint'].endswith('ankle')]})
        if name == 'heavy-squat':
            worst = max(by_time.values(), key=lambda row: row['max_penetration_model_units'])
            fitted = audit['frames'][worst['source_reference_index']]
            group['diagnostic_frame'] = fitted['frame']
            worst_case = {'frame': fitted['frame'], 'time_s': fitted['time_s'],
                          'max_penetration_model_units': worst['max_penetration_model_units'],
                          'ankles': [point for point in fitted['points'] if point['joint'].endswith('ankle')]}
    source_checks, checked_panels = {}, 0
    for source, group in groups.items():
        cap = cv2.VideoCapture(str(SOURCE / source))
        decoded = 0; matched = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok: break
                if decoded >= 1800:
                    raise ValueError('Source decoding exceeds the lab bound')
                if decoded in group['panels'] or decoded == group.get('diagnostic_frame'):
                    raw = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    if decoded == group.get('diagnostic_frame'):
                        raw.save(out / 'heavy-worst-source.png')
                    raw.thumbnail((640, 480))
                    for panel in group['panels'].get(decoded, []):
                        with Image.open(panel) as saved:
                            tile = saved.crop((0, 38, raw.width, 38 + raw.height))
                            if not np.array_equal(np.asarray(tile), np.asarray(raw)):
                                raise ValueError('Panel does not show its declared sequential source frame')
                        checked_panels += 1
                    matched.append(decoded)
                decoded += 1
        finally:
            cap.release()
        if decoded != group['expected_decoded'] or not set(group['panels']) <= set(matched):
            raise ValueError('Source frame count or panel coverage changed')
        source_checks[source] = {'decoded_frames': decoded, 'verified_panel_frame_indices': sorted(group['panels'])}
    if checked_panels != 15:
        raise ValueError('Every saved source panel must be checked')
    result = {'created_at': local_time(), 'status': 'bindings_and_source_panels_verified',
        'comparison_manifest_sha256': SOURCE_SEAL, 'preservation': preservation, 'source_checks': source_checks,
        'unique_decoded_source_frames': sum(row['decoded_frames'] for row in source_checks.values()),
        'source_panels_exactly_matched': checked_panels, 'new_source_floor_failures': new_failures,
        'heavy_worst_candidate': worst_case, 'new_fits': 0, 'new_inference_calls': 0, 'paid_requests': 0,
        'browser_visual_qa': False, 'physical_accuracy_verified': False,
        'scope': 'Source identity/coverage and model binding verification, not anatomical or contact validation'}
    write_json(out / 'report.json', result)
    print(json.dumps({'decoded_source_frames': result['unique_decoded_source_frames'],
                      'source_panels_exactly_matched': checked_panels, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
