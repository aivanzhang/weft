"""Four-source opt-in foot references; source-only proxy labels never enter construction."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import shutil
import signal
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.foot_reference import build_foot_reference
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/foot-reference-01'
LABELS = ROOT / 'artifacts/foot-reference-source-labels-01'
CASES = {name: ROOT / 'artifacts/mobile-exercise-e2e-10' / name / 'run'
         for name in ('single-leg-squat', 'squat', 'kettlebell-pushup')}
CASES['fullbody-squat'] = ROOT / 'artifacts/fullbody-exercise-01/baseline/run'


def pairs_error(a, b):
    a, b = np.array(a), np.array(b)
    distance = min(float(np.linalg.norm(a - b, axis=1).mean()), float(np.linalg.norm(a - b[::-1], axis=1).mean()))
    av, bv = a[1] - a[0], b[1] - b[0]
    angle = float(np.degrees(np.arccos(np.clip(abs(av @ bv) / (np.linalg.norm(av) * np.linalg.norm(bv)), 0, 1))))
    return distance, angle


def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('60-second foot study bound')))
    signal.alarm(60)
    started = time.monotonic()
    OUT.mkdir(exist_ok=False)
    label_verification = verify_artifacts(LABELS)
    labels = json.loads((LABELS / 'labels.json').read_text())
    shutil.copytree(LABELS, OUT / 'frozen-source-labels')
    code = ['src/weft/foot_reference.py', 'src/weft/body_frame.py', 'tests/test_foot_reference.py',
            'scripts/studies/check_foot_reference.py']
    hashes = {name: digest(ROOT / name) for name in code}
    for name in code:
        target = OUT / 'implementation' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    summaries, comparison, tiles = {}, [], []
    for name, run in CASES.items():
        verified = verify_artifacts(run)
        destination = OUT / name
        (destination / 'inputs').mkdir(parents=True)
        names = ['human.json', 'evidence.json', 'pose-landmarks.json', 'root-motion.json', 'source.json', 'manifest.json']
        inputs = {filename: digest(run / filename) for filename in names}
        for filename in names:
            shutil.copyfile(run / filename, destination / 'inputs' / filename)
        spec = json.loads((run / 'human.json').read_text())
        evidence = json.loads((run / 'evidence.json').read_text())
        raw = json.loads((run / 'pose-landmarks.json').read_text())
        root_audit = json.loads((run / 'root-motion.json').read_text())
        video = next(path for path in run.glob('source.*') if path.suffix in ('.mp4', '.webm'))
        assert digest(video) == spec['source_sha256'] == evidence['source_sha256']
        selected = {row['frame']: row for row in labels['rows'] if row['case'] == name}
        decoded, images, sizes = 0, {}, set()
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, image = cap.read()
            if not ok:
                break
            sizes.add((image.shape[1], image.shape[0]))
            if decoded in selected:
                images[decoded] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            decoded += 1
            assert decoded <= 1800
        cap.release()
        assert decoded == evidence['decoded_frames'] and sizes == {(evidence['width'], evidence['height'])}
        metadata = {'source_sha256': spec['source_sha256'], 'width': evidence['width'], 'height': evidence['height'],
                    'fps': evidence['fps'], 'decoded_frames': decoded, 'dimensions_verified_from_decoding': True}
        reference = build_foot_reference(spec, raw | {'source_sha256': spec['source_sha256']}, metadata)
        write_json(destination / 'foot-reference.json', reference)
        frames = {row['frame']: row for row in reference['frames']}
        scale = root_audit['scale']['model_units_per_pixel']
        construction_errors = []
        maximum_length_spread = 0.
        for side, shape in reference['fixed_shapes'].items():
            distances = []
            for row in reference['frames']:
                foot = row['feet'][side]
                if foot['status'] != 'kinematic_reference':
                    continue
                geometry = foot['endpoint_geometry']
                heel, index = (np.array(geometry[key]) for key in ('heel_relative', 'foot_index_relative'))
                distances.append([np.linalg.norm(heel), np.linalg.norm(index), np.linalg.norm(index - heel)])
            if distances:
                maximum_length_spread = max(maximum_length_spread, float(np.ptp(distances, axis=0).max()))
        assert maximum_length_spread < 1e-12
        for frame, label in selected.items():
            assert label['source_sha256'] == spec['source_sha256']
            source_image = np.array(Image.open(LABELS / label['source_png']).convert('RGB'))
            np.testing.assert_array_equal(images[frame], source_image)
            image = Image.fromarray(images[frame])
            canvas = Image.new('RGB', (image.width, image.height + 64), '#ffffff')
            canvas.paste(image)
            draw = ImageDraw.Draw(canvas)
            row, predicted, raw_axes = frames.get(frame), {}, {}
            if row:
                for side, foot in row['feet'].items():
                    source = foot['raw_image_reference']
                    if source:
                        raw_axes[side] = [source['points_px']['heel'], source['points_px']['foot_index']]
                        if all(not flag for flag in foot['outside_image'].values()):
                            draw.line([tuple(point) for point in raw_axes[side]], fill='#218B4A', width=2)
                    if foot['status'] != 'kinematic_reference':
                        continue
                    anchor = np.array(source['points_px']['ankle'])
                    points = [anchor + np.array(foot['endpoint_geometry'][key])[[0, 2]] * [1, -1] / scale
                              for key in ('heel_relative', 'foot_index_relative')]
                    predicted[side] = [point.tolist() for point in points]
                    draw.line([tuple(point) for point in points], fill='#B000B5', width=2)
                    for point in points:
                        draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), outline='#B000B5', width=2)
                    error, angle = pairs_error(points, raw_axes[side])
                    construction_errors.append({'frame': frame, 'side': side, 'endpoint_distance_px': error,
                                                'axis_degrees': angle, 'scope': 'Source pose-model endpoints, not independent truth'})
            observations = [foot for foot in label['feet'] if foot['status'] == 'axis_estimated']
            assignments = []
            if predicted and observations:
                count = min(len(predicted), len(observations))
                best = None
                for slots in itertools.combinations(observations, count):
                    for sides in itertools.permutations(predicted, count):
                        current = [{'label_id': item['id'], 'model_side_hypothesis': side,
                                    'endpoint_distance_px': pairs_error(item['endpoints_px'], predicted[side])[0],
                                    'axis_error_degrees': pairs_error(item['endpoints_px'], predicted[side])[1],
                                    'label_endpoint_uncertainty_px': item['endpoint_uncertainty_px']}
                                   for item, side in zip(slots, sides)]
                        if best is None or sum(x['endpoint_distance_px'] for x in current) < sum(x['endpoint_distance_px'] for x in best):
                            best = current
                assignments = best
            for item in observations:
                draw.line([tuple(point) for point in item['endpoints_px']], fill='#E78A00', width=2)
            draw.text((6, image.height + 5), f'{name} frame {frame}: green=raw pose, purple=fixed foot', fill='#111111')
            draw.text((6, image.height + 23), 'orange=source silhouette proxy; no anatomical/contact ground truth', fill='#111111')
            draw.text((6, image.height + 41), 'Source-ankle anchored; frozen uncalibrated scale; missing feet not filled', fill='#111111')
            canvas.save(destination / f'overlay-{frame:04d}.png')
            thumbnail = canvas.copy()
            thumbnail.thumbnail((640, 520))
            tile = Image.new('RGB', (660, 540), '#ffffff')
            tile.paste(thumbnail, ((660 - thumbnail.width) // 2, 0))
            tiles.append(tile)
            comparison.append({'case': name, 'frame': frame, 'source_sha256': spec['source_sha256'],
                'labels': label['feet'], 'predicted_geometry': predicted, 'raw_pose_axes': raw_axes,
                'anonymous_assignments': assignments, 'unmatched_estimated_axis_labels': len(observations) - len(assignments),
                'unavailable_source_label_slots': sum(foot['status'] != 'axis_estimated' for foot in label['feet']),
                'unmatched_model_feet': len(predicted) - len(assignments),
                'scope': 'Anonymous proxy-axis diagnostic only; not anatomical identity verification or a passing score'})
        reasons = {side: dict(Counter(reason for row in reference['frames'] for reason in row['feet'][side]['reasons']))
                   for side in ('left', 'right')}
        summaries[name] = {'source_verification': verified, 'input_hashes': inputs,
            'source_sha256': spec['source_sha256'], 'coverage': reference['coverage'], 'fixed_shapes': reference['fixed_shapes'],
            'construction_cutoff_time_s': reference['construction_cutoff_time_s'], 'unavailable_reasons': reasons,
            'maximum_fixed_triangle_edge_spread': maximum_length_spread, 'construction_endpoint_comparisons': construction_errors,
            'projection_scope': 'Fixed geometry anchored to source-image ankle using inherited uncalibrated model scale, not actual HumanEnv ankle position',
            'model_units_per_pixel': scale, 'runtime_seconds': reference['runtime_seconds']}
        assert inputs == {filename: digest(run / filename) for filename in names}
    sheet = Image.new('RGB', (1320, 540 * 4), '#ffffff')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 660, (i // 2) * 540))
    sheet.save(OUT / 'foot-comparison-sheet.png')
    assert hashes == {name: digest(ROOT / name) for name in code}
    write_json(OUT / 'report.json', {'schema_version': 1, 'status': 'completed_diagnostic', 'cases': summaries,
        'source_label_verification': label_verification, 'source_label_comparisons': comparison,
        'labels_used_for_fit': False, 'implementation_hashes': hashes, 'original_inputs_unchanged': True,
        'physical_model_modified': False, 'physical_validation': 'unverified', 'physics_steps': 0,
        'new_api_calls': 0, 'new_cost_usd': 0, 'runtime_seconds': time.monotonic() - started,
        'scope': 'Opt-in kinematic foot endpoint artifacts, not colliders, soles, contact forces, calibrated anatomy or learned control'})
    signal.alarm(0)
    print(json.dumps(seal_suite(OUT)))
    print(json.dumps({name: {'coverage': row['coverage']['feet'], 'fixed_shapes': row['fixed_shapes'],
                            'unavailable_reasons': row['unavailable_reasons']} for name, row in summaries.items()}, indent=2))


if __name__ == '__main__':
    main()
