"""Compare image-root estimates with frozen source-only labels, never calibrating to them."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


def verify_images(labels):
    wanted = {row['frame']: row for row in labels['annotations']}
    cap = cv2.VideoCapture(labels['source_path'])
    checked = []
    try:
        for index in range(max(wanted) + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Source ended before a frozen label frame')
            if index in wanted:
                actual = hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
                if actual != wanted[index]['decoded_rgb_sha256']:
                    raise ValueError('Frozen source image differs from a sequential decode')
                checked.append(index)
    finally:
        cap.release()
    return checked


def grade(labels, audit):
    observations = audit.get('observations', [])
    rows, paired = [], []
    for label in labels['annotations']:
        row = dict(label)
        nearest = min(observations, key=lambda item: (abs(item['frame'] - label['frame']), item['frame'])) if observations else None
        usable = nearest is not None and abs(nearest['frame'] - label['frame']) <= 1
        row['prediction_frame'] = nearest['frame'] if usable else None
        row['prediction_status'] = nearest['status'] if usable else 'no_sample_within_one_source_frame'
        predicted = nearest['pelvis_image_px'] if usable else None
        row['predicted_pelvis_image_px'] = predicted
        row['scope'] = 'Estimated image position; not observed 3D anatomy'
        if label['pelvis_midpoint_px'] is not None and predicted is not None:
            error = math.dist(predicted, label['pelvis_midpoint_px'])
            row.update(error_source_px=error, within_annotation_allowance=error <= label['pelvis_uncertainty_radius_px'],
                       torso_length_error_px=abs(nearest['torso_image_length_px'] - label['projected_torso_length_px']))
            paired.append(row)
        rows.append(row)
    # Relative scoring removes only the origin; no rotation, scale or camera is fitted.
    if paired:
        origin = paired[0]
        for row in paired[1:]:
            observed = np.subtract(row['pelvis_midpoint_px'], origin['pelvis_midpoint_px'])
            predicted = np.subtract(row['predicted_pelvis_image_px'], origin['predicted_pelvis_image_px'])
            row['relative_motion'] = {'origin_label_frame': origin['frame'],
                'label_displacement_px': observed.tolist(), 'predicted_displacement_px': predicted.tolist(),
                'anchored_error_px': float(np.linalg.norm(observed)),
                'mobile_proxy_error_px': float(np.linalg.norm(predicted - observed)),
                'annotation_displacement_allowance_px': row['pelvis_uncertainty_radius_px'] + origin['pelvis_uncertainty_radius_px']}
    visible = sum(row['pelvis_midpoint_px'] is not None for row in rows)
    return {'rows': rows, 'slots': len(rows), 'estimable_labels': visible, 'predictions_at_estimable_labels': len(paired),
            'within_annotation_allowance': sum(row.get('within_annotation_allowance', False) for row in rows),
            'estimates_at_unobservable_labels': sum(row['pelvis_midpoint_px'] is None and row['predicted_pelvis_image_px'] is not None for row in rows),
            'accepted_rmse_px': math.sqrt(sum(row['error_source_px']**2 for row in paired) / len(paired)) if paired else None,
            'root_reference_status': audit.get('status'),
            'root_quality_errors': audit.get('quality', {}).get('errors', [])}


def run(candidate, out):
    out.mkdir(parents=True, exist_ok=False)
    index_path = Path('data/root-motion-labels/index.json')
    index = json.loads(index_path.read_text())
    cases = []
    inputs = {str(index_path): digest(index_path)}
    for item in index['files']:
        path = Path(item['path'])
        if digest(path) != item['sha256']:
            raise ValueError('Frozen source labels changed')
        labels = json.loads(path.read_text())
        if digest(labels['source_path']) != labels['source_sha256']:
            raise ValueError('Video identity differs from frozen labels')
        audit_path = candidate / item['clip_id'] / 'run/root-motion.json'
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        if audit and audit['source_sha256'] != labels['source_sha256']:
            raise ValueError('Root estimate source differs from labels')
        result = grade(labels, audit)
        result.update(case=item['clip_id'], sequential_source_images_verified=verify_images(labels))
        cases.append(result)
        inputs[str(path)] = digest(path)
        if audit_path.exists():
            inputs[str(audit_path)] = digest(audit_path)
        shutil.copyfile(path, out / path.name)
    report = {'created_at': local_time(), 'cases': cases, 'inputs': inputs,
              'matching_rule': 'Nearest sample within one source frame; earlier frame wins ties; missing predictions retained',
              'scope': 'Frozen approximate 2D source-only development labels with subjective uncertainty. No metric calibration, hidden anatomy, contact-force or physical balance validation.'}
    write_json(out / 'report.json', report)
    shutil.copyfile(__file__, out / Path(__file__).name)
    print(seal_suite(out))
    for case in cases:
        print({key: value for key, value in case.items() if key not in ('rows', 'root_quality_errors')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.candidate, args.out)
