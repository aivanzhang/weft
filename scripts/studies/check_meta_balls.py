"""Offline, sealed three-ball comparison; never sends a provider request."""

import argparse
import json
import math
from pathlib import Path
import shutil

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts

START, END, INITIAL = 2580, 2760, 2620
NAMES = ('upper-passive', 'lower-passive', 'incoming-active')
LABEL_FRAMES = (2580, 2620, 2640, 2650, 2670, 2700, 2720, 2760)
BOXES = ([187,104,8,8], [189,111,8,8], [270,113,8,8])
SOURCE_HASH = 'f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887'
PROMPTS = [{'object_id': i+1, 'frame': INITIAL-START, 'roi': box} for i, box in enumerate(BOXES)]
ROOT = Path(__file__).resolve().parents[2]


def _json(path):
    return json.loads(Path(path).read_text())


def _sealed(root, names):
    verification = verify_artifacts(root)
    manifest = _json(Path(root) / 'manifest.json')
    if not set(names) <= manifest['files'].keys():
        raise ValueError('Required evidence is not covered by its manifest')
    return verification


def _center(row):
    box = row.get('box_xyxy_halfopen')
    if row.get('status') == 'missing':
        if box is not None:
            raise ValueError('Missing observation contains a position')
        return None
    if (row.get('status') != 'returned' or not isinstance(box, list) or len(box) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in box)
            or not 0 <= box[0] < box[2] <= 426 or not 0 <= box[1] < box[3] <= 240):
        raise ValueError('Invalid returned source box')
    return [(box[0]+box[2])/2, (box[1]+box[3])/2]


def grade(frames, manual):
    if (len(frames) != END-START+1 or [r['frame'] for r in frames] != list(range(END-START+1))
            or manual.get('objects') != list(NAMES) or manual.get('source_sha256') != SOURCE_HASH
            or [r['frame'] for r in manual['observations']] != list(LABEL_FRAMES)):
        raise ValueError('Comparison requires every pinned source frame and all 24 label slots')
    centers = {}
    for frame in frames:
        objects = frame.get('objects', [])
        if len(objects) != 3 or {r['object_id'] for r in objects} != {'1','2','3'}:
            raise ValueError('Each frame requires exactly the three configured object slots')
        for row in objects:
            if row.get('frame') != frame['frame']:
                raise ValueError('Object and frame identity disagree')
            centers[(frame['frame']+START, row['object_id'])] = _center(row)
    rows, errors, eligible, absent, unknown = [], [], 0, 0, 0
    for label in manual['observations']:
        if len(label['centers']) != 3 or len(label['visibility']) != 3:
            raise ValueError('Incomplete manual observation')
        for i, name in enumerate(NAMES):
            source_frame, expected = label['frame'], label['centers'][i]
            visibility = label['visibility'][i]
            if expected is not None and (not isinstance(expected, list) or len(expected) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in expected)):
                raise ValueError('Invalid manual center')
            if visibility == 'visible' and expected is None:
                raise ValueError('Visible manual center is missing')
            predicted = centers[(source_frame, str(i+1))]
            period = 'pre_initialization' if source_frame < INITIAL else 'initialization' if source_frame == INITIAL else 'post_initialization'
            scoreable = period == 'post_initialization' and visibility == 'visible' and expected is not None
            eligible += scoreable
            error = math.dist(predicted, expected) if scoreable and predicted is not None else None
            if error is not None:
                errors.append(error)
            absent_prediction = period == 'post_initialization' and visibility == 'out_of_frame' and predicted is not None
            unknown_prediction = visibility not in ('visible','out_of_frame') and predicted is not None
            absent += absent_prediction
            unknown += unknown_prediction
            rows.append({'source_frame': source_frame, 'relative_frame': source_frame-START,
                'object_id': str(i+1), 'source_object_id': name, 'source_center_px': expected,
                'source_visibility': visibility, 'source_identity_status': label.get('identity_status'),
                'period': period, 'accuracy_eligible': scoreable, 'prediction_center_px': predicted,
                'scored_error_px': error, 'within_5px_label_tolerance': error <= 5 if error is not None else None,
                'prediction_on_labeled_absence': absent_prediction, 'prediction_on_unknown_visibility': unknown_prediction})
    post = range(INITIAL+1, END+1)
    joint = [frame for frame in post if all(centers[(frame,str(i))] is not None for i in (1,2,3))]
    return {'rows': rows, 'manual_slots': len(rows), 'eligible_postinit_center_slots': eligible,
        'returned_eligible_center_slots': len(errors), 'missing_eligible_center_slots': eligible-len(errors),
        'eligible_center_coverage': len(errors)/eligible if eligible else None,
        'conditional_rmse_px': math.sqrt(sum(x*x for x in errors)/len(errors)) if errors else None,
        'conditional_max_error_px': max(errors) if errors else None, 'within_5px_label_tolerance': sum(x <= 5 for x in errors),
        'postinit_frames': len(post), 'all_three_returned_postinit_frames': len(joint),
        'all_three_returned_postinit_coverage': len(joint)/len(post), 'all_three_returned_source_frames': joint,
        'postinit_returned_object_slots': sum(centers[(f,str(i))] is not None for f in post for i in (1,2,3)),
        'postinit_object_slots': len(post)*3, 'predictions_on_labeled_absence': absent,
        'predictions_on_unknown_visibility': unknown,
        'center_convention': 'Half-open box geometric midpoint, matching saved ball ROI convention; not a decoded mask centroid',
        'coverage_scope': 'Returned proposals, not verified identities; all-three coverage includes frames with no independent labels',
        'scope': 'Development labels ±5 px; same-color post-contact identities inferred, camera drift, no mask or physical validation'}


def prior_frames(trace):
    if [r['frame'] for r in trace['frames']] != list(range(START, END+1)):
        raise ValueError('Prior trace source frame mapping differs')
    result = []
    for frame in trace['frames']:
        objects = {obj['object_id']: obj for obj in frame['objects']}
        if set(objects) != set(NAMES) or len(frame['objects']) != 3:
            raise ValueError('Prior trace object identities differ')
        rows = []
        for i, name in enumerate(NAMES):
            old = objects[name]
            box = old.get('box_xywh') if old['status'] in ('tracked','initialized') else None
            bounds = [box[0],box[1],box[0]+box[2],box[1]+box[3]] if box else None
            rows.append({'frame': frame['frame']-START, 'object_id': str(i+1),
                         'status': 'returned' if bounds else 'missing', 'box_xyxy_halfopen': bounds})
        result.append({'frame': frame['frame']-START, 'objects': rows})
    return result


def prepare(input_dir, out):
    input_dir, out = Path(input_dir).resolve(), Path(out).resolve()
    names = ['report.json','frame-mapping-and-pixels.json','object-map.json']
    verified = _sealed(input_dir, names)
    report = _json(input_dir / 'report.json')
    video = (input_dir / report['derivative_path']).resolve()
    if not video.is_relative_to(input_dir):
        raise ValueError('Derivative must remain inside the sealed input')
    _sealed(input_dir, [str(video.relative_to(input_dir))])
    if (report.get('source_sha256') != SOURCE_HASH or report.get('decoded_frames') != 181
            or report.get('dimensions') != [426,240] or report.get('selected_source_interval') != [START,END]
            or report.get('relative_interval') != [0,180] or report.get('exact_relative_pts_match') is not True
            or report.get('all_frame_mapping_verified') is not True or digest(video) != report['derivative_sha256']):
        raise ValueError('Video preparation differs from the frozen comparison')
    mapping = _json(input_dir / 'frame-mapping-and-pixels.json')
    if (len(mapping['frames']) != 181 or any(row['source_frame'] != START+i or row['relative_frame'] != i
            or row['source_pts']-mapping['source_pts_offset'] != row['relative_pts'] for i,row in enumerate(mapping['frames']))):
        raise ValueError('Frame/timestamp mapping differs')
    object_map = _json(input_dir / 'object-map.json')
    if len(object_map) != 3 or any(row['sam_id'] != i+1 or row['object_id'] != NAMES[i]
            or row['source_initialization_frame'] != INITIAL or row['relative_initialization_frame'] != INITIAL-START
            or row['source_box_xywh'] != BOXES[i] for i,row in enumerate(object_map)):
        raise ValueError('Prompt identity mapping differs')
    originals = {name: ROOT/'data/balls'/f'triple-collision-240p.{name}.json' for name in ('manual','initialization')}
    originals.update({'prior-comparison': ROOT/'artifacts/joint-appearance-02/triple/comparison.json',
                      'prior-trace': ROOT/'artifacts/joint-appearance-02/triple/appearance-tracks.json'})
    prior_verification = _sealed(ROOT/'artifacts/joint-appearance-02', ['triple/comparison.json','triple/appearance-tracks.json'])
    manual = _json(originals['manual'])
    initialization = _json(originals['initialization'])
    if (initialization.get('source_sha256') != SOURCE_HASH or initialization.get('initialization_frame') != INITIAL
            or [(obj['id'],obj['bbox_xywh']) for obj in initialization['objects']] != list(zip(NAMES,BOXES))):
        raise ValueError('Frozen initialization differs from the prompt mapping')
    prior = grade(prior_frames(_json(originals['prior-trace'])), manual)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out/'inputs'
    inputs.mkdir()
    for name,path in originals.items():
        shutil.copyfile(path, inputs/f'{name}.json')
    for name in names:
        shutil.copyfile(input_dir/name, inputs/('video-'+name))
    shutil.copyfile(input_dir/'manifest.json', inputs/'video-manifest.json')
    implementations = [Path(__file__).resolve(), ROOT/'tests/test_meta_ball_grading.py',
                       ROOT/'src/weft/meta_segmentation.py', ROOT/'src/weft/meta_segmentation_parser.py',
                       ROOT/'src/weft/segmentation_lab.py', ROOT/'src/weft/cli.py', ROOT/'src/weft/__main__.py',
                       ROOT/'tests/test_segmentation_lab.py', ROOT/'tests/test_meta_segmentation.py',
                       ROOT/'tests/test_meta_segmentation_parser.py']
    for path in implementations:
        shutil.copyfile(path, inputs/path.name)
    freeze = {'created_at': local_time(), 'input_bundle': str(input_dir), 'input_verification': verified,
        'video': str(video), 'video_sha256': digest(video), 'source_sha256': SOURCE_HASH,
        'source_interval': [START,END], 'initialization_source_frame': INITIAL, 'prompts': PROMPTS,
        'prior_verification': prior_verification, 'original_hashes': {str(p): digest(p) for p in originals.values()},
        'implementation_hashes': {str(p): digest(p) for p in implementations},
        'expected_provider_frame_cost_usd': .0362, 'conservative_three_object_envelope_usd': .1086,
        'reservation_usd': .15, 'labels_sent_to_provider': False, 'new_api_calls': 0}
    write_json(inputs/'freeze.json', freeze)
    write_json(inputs/'prior-regraded.json', prior)
    write_json(inputs/'report.json', {'status': 'frozen_before_request', 'api_calls': 0})
    seal = seal_suite(inputs)
    write_json(out/'report.json', {'status': 'prepared_no_api', 'inputs': seal, 'new_api_calls': 0})
    return freeze


def grade_saved(out):
    out = Path(out).resolve()
    if (out/'manifest.json').exists() or (out/'comparison.json').exists():
        raise ValueError('Comparison output already exists')
    inputs, tracking = out/'inputs', out/'tracking'
    _sealed(inputs, ['freeze.json','manual.json','prior-regraded.json','check_meta_balls.py'])
    freeze = _json(inputs/'freeze.json')
    if digest(Path(__file__)) != digest(inputs/'check_meta_balls.py'):
        raise ValueError('Grader changed after the pre-request freeze')
    tracking_verification = _sealed(tracking, ['call/segmentation.json','call/call.json','request.json','video.json','report.json'])
    call, result, request = (_json(tracking/name) for name in ('call/call.json','call/segmentation.json','request.json'))
    video = _json(tracking/'video.json')
    prompt = '<40f>' + ','.join(f'{i+1}<|box;x1={x};y1={y};x2={x+w-1};y2={y+h-1};w=426;h=240|>'
                               for i,(x,y,w,h) in enumerate(BOXES))
    content = request.get('input', [{}])[0].get('content', [])
    if (call.get('status') != 'completed' or call.get('video_sha256') != freeze['video_sha256']
            or call.get('initialization') != PROMPTS or result.get('initialization') != PROMPTS
            or result.get('status') != 'completed' or call.get('request_sha256') != digest(tracking/'request.json')
            or request.get('metadata', {}).get('propagation_direction') != 'forward'
            or video.get('source_sha256') != freeze['video_sha256']
            or (video.get('frame_count'),video.get('width'),video.get('height')) != (181,426,240)
            or len(content) != 2 or content[0] != {'type':'input_text','text':prompt}
            or content[1].get('type') != 'input_video'):
        raise ValueError('Saved request/result differs from the frozen forward-only study')
    manual = _json(inputs/'manual.json')
    comparison = {'sam': grade(result['frames'], manual), 'prior_joint_v2_regraded': _json(inputs/'prior-regraded.json'),
        'prior_legacy_denominator_note': 'Old saved 9/22 included five visible pre-init/init labels; compare the recomputed 17 eligible post-init center slots',
        'prior_trace_sha256': digest(inputs/'prior-trace.json'), 'prior_saved_grade_sha256': digest(inputs/'prior-comparison.json'),
        'source_mapping_sha256': digest(inputs/'video-frame-mapping-and-pixels.json'),
        'new_api_calls': 0, 'default_change': False, 'mask_accuracy': 'Not measured; masks remain opaque'}
    write_json(out/'comparison.json', comparison)
    write_json(out/'report.json', {'status': 'completed_offline_comparison', 'created_at': local_time(),
        'tracking_verification': tracking_verification, 'inputs_verification': verify_artifacts(inputs),
        'comparison': comparison, 'new_api_calls': 0, 'default_change': False})
    return seal_suite(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare','grade'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare' and args.input is None:
        parser.error('prepare requires --input')
    result = prepare(args.input,args.out) if args.mode == 'prepare' else grade_saved(args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
