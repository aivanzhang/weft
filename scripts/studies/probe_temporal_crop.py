"""Probe one fixed temporal crop on a known bad frame and a second-view control."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from weft.benchmark import seal_suite
from weft.exercise_vision import _detector, _draw_pose, model_provenance, summarize_landmarks, summarize_result
from weft.io import canonical_hash, digest, local_time, write_json
from weft.multiview_pose import triangulate_pair
from weft.pipeline import _implementation_files, verify_artifacts


ROOT = Path(__file__).resolve().parents[2]
FRAME = 191


def read(path):
    return json.loads(path.read_text())


def crop_plan(evidence, frame):
    anchors = [row for row in evidence['frames'] if row['frame'] != frame
               and abs(row['time_s']-frame/evidence['fps']) <= .2
               and row['status'] in ('tracked', 'tracked_partial') and row['pose_count'] == 1]
    before = max((row for row in anchors if row['frame'] < frame), key=lambda row: row['frame'])
    after = min((row for row in anchors if row['frame'] > frame), key=lambda row: row['frame'])
    bounds = []
    for row in (before, after):
        points = np.asarray([[p['x']*evidence['width'], p['y']*evidence['height']]
                             for p in row['landmarks'] if min(p['visibility'], p['presence']) >= .5
                             and 0 <= p['x'] <= 1 and 0 <= p['y'] <= 1])
        if len(points) < 8 or not np.isfinite(points).all():
            raise ValueError('Temporal crop lacks source-supported anchor points')
        bounds.append([points.min(axis=0), points.max(axis=0)])
    low, high = np.min(np.array(bounds)[:, 0], axis=0), np.max(np.array(bounds)[:, 1], axis=0)
    padding = (high-low)*.25
    low = np.maximum([0, 0], np.floor(low-padding)).astype(int)
    high = np.minimum([evidence['width'], evidence['height']], np.ceil(high+padding)).astype(int)
    if np.any(high-low < 64):
        raise ValueError('Temporal crop is smaller than the supported model image')
    return {'frame': frame, 'box_xyxy': [*low.tolist(), *high.tolist()], 'padding_fraction': .25,
            'anchor_frames': [before['frame'], after['frame']],
            'anchor_hashes': [canonical_hash(before), canonical_hash(after)],
            'maximum_anchor_gap_seconds': .2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    study = ROOT/'artifacts/limb-correspondence-dense-jog-01'
    sources = [ROOT/'artifacts/multiview-jog-source-01', ROOT/'artifacts/multiview-jog-extra-views-02']
    inputs = [study, *sources]
    if any(out.is_relative_to(path) for path in inputs):
        raise ValueError('Use a new output outside sealed inputs')
    verified = {str(path): verify_artifacts(path) for path in inputs}
    views = [read(study/f'inputs/view-{i}.json') for i in (1, 2)]
    plans = [crop_plan(view['evidence'], FRAME) for view in views]
    model = ROOT/'.local/models/pose_landmarker_full.task'
    provenance = model_provenance(model)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    for name, path in _implementation_files().items():
        target = out/'implementation-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': verified, 'plans': plans,
        'model': provenance, 'implementation': implementation, 'maximum_inference_calls': 2,
        'selection': 'Known bad frame191 in view1; same frame in view2 is a control, not an automatic repair',
        'hypothesis': 'A single fixed neighbor-bounded crop can avoid the original image detector failure',
        'scope': 'Post-hoc development probe; no reference joints, third camera, padding search or promotion',
        'image_coordinate_documentation': 'https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python',
        'paid_requests': 0, 'projected_external_cost_usd': 0, 'maximum_wall_seconds': 30})
    start = time.monotonic()
    mp, detector = _detector(model, 'image')
    candidates, old = [], []
    try:
        for index, (view, plan, source) in enumerate(zip(views, plans, sources, strict=True), 1):
            evidence = view['evidence']
            original = next(row for row in evidence['frames'] if row['frame'] == FRAME)
            old.append(original)
            video = source/f'drone{index}.mp4'
            if digest(video) != evidence['source_sha256']:
                raise ValueError('Video does not match the frozen image evidence')
            capture = cv2.VideoCapture(str(video))
            try:
                for frame in range(evidence['frame_count']):
                    ok, bgr = capture.read()
                    if not ok or abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/evidence['fps']) > 1e-6:
                        raise ValueError('Source decoding or clock differs from the frozen interval')
                    if time.monotonic()-start > 30:
                        raise TimeoutError('Crop probe exceeded its local time bound')
                    if frame != FRAME:
                        continue
                    left, top, right, bottom = plan['box_xyxy']
                    crop = bgr[top:bottom, left:right]
                    cv2.imwrite(str(out/f'view-{index}-crop.png'), crop)
                    cv2.imwrite(str(out/f'view-{index}-source.jpg'), bgr)
                    begin = time.monotonic()
                    result = summarize_result(detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                    elapsed = time.monotonic()-begin
                    write_json(out/f'view-{index}-crop-result.json', result)
                    mapped = deepcopy(result)
                    if result.get('pose_count') == 1 and len(result.get('landmarks', [])) == 33:
                        for point in mapped['landmarks']:
                            point['x'] = (left+point['x']*(right-left))/evidence['width']
                            point['y'] = (top+point['y']*(bottom-top))/evidence['height']
                            point['z'] *= (right-left)/evidence['width']
                        mapped = summarize_landmarks(mapped['landmarks'], mapped['world_landmarks'])
                    mapped.update({key: original[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')})
                    mapped.update(inference_seconds=elapsed, crop_proposal=plan)
                    candidates.append(mapped)
                    write_json(out/f'view-{index}-mapped-proposal.json', mapped)
                    _draw_pose(bgr, mapped).save(out/f'view-{index}-mapped-pose.png')
                if capture.read()[0]:
                    raise ValueError('Source has extra frames outside its frozen interval')
            finally:
                capture.release()
    finally:
        detector.close()
    predictions, summaries = {}, {}
    for name, use in [('raw', (False, False)), ('first_only', (True, False)),
                      ('second_only', (False, True)), ('both', (True, True))]:
        pair = deepcopy(views)
        for index in (0, 1):
            if use[index]:
                pair[index]['evidence']['frames'][FRAME] = candidates[index]
        result = triangulate_pair(*pair)['frames'][FRAME]
        predictions[name] = result
        summaries[name] = {'hip_statuses': [result['joints'][j]['status'] for j in (23, 24)],
                          'estimated_joints': sum(j['status'] == 'estimated' for j in result['joints']),
                          'hip_reprojection_px': [result['joints'][j]['reprojection_px'] for j in (23, 24)]}
    write_json(out/'triangulation-proposals.json', predictions)
    if any(verify_artifacts(Path(path)) != value for path, value in verified.items()):
        raise ValueError('A source changed during the probe')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime implementation changed during the probe')
    report = {'created_at': local_time(), 'status': 'crop_probe_complete_no_promotion',
        'original_pose_statuses': [row['status'] for row in old],
        'proposed_pose_statuses': [row['status'] for row in candidates], 'hypotheses': summaries,
        'wall_seconds': time.monotonic()-start, 'paid_requests': 0, 'inference_calls': 2,
        'scope': 'Source-image re-inference only; not verified anatomy, fixed temporal correspondence or a new body'}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
