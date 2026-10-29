"""One frozen full-clip Meta tracking comparison; initialization is not a test label."""

import argparse
import json
import math
from pathlib import Path
import shutil
import signal

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.meta_segmentation import segment_video
from weft.pipeline import verify_artifacts


def grade(frames, manual):
    if len(frames) != 354 or [f['frame'] for f in frames] != list(range(354)):
        raise ValueError('Comparison requires every original frame slot')
    rows, errors = [], []
    for label in manual['observations']:
        prediction = frames[label['frame']]
        box = prediction.get('box_xyxy_halfopen')
        # Inclusive pixel-center envelopes retain the original label convention.
        center = [(box[0] + box[2] - 1) / 2, (box[1] + box[3] - 1) / 2] if box else None
        error = math.dist(center, label['center_px']) if center and label['center_px'] else None
        scored = label['frame'] != 0 and error is not None
        if scored:
            errors.append(error)
        rows.append({**label, 'sam_status': prediction['status'], 'sam_box_center_px': center,
                     'initialization_not_scored': label['frame'] == 0,
                     'scored_error_px': error if scored else None,
                     'within_16px_uncertainty': error <= 16 if scored else None,
                     'unknown_crossfade_has_prediction': label['center_px'] is None and center is not None})
    return {'rows': rows, 'noninitialization_center_slots': 4, 'returned_center_slots': len(errors),
            'conditional_rmse_px': math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
            'within_16px_uncertainty': sum(e <= 16 for e in errors),
            'returned_all_frame_slots': sum(f.get('box_xyxy_halfopen') is not None for f in frames),
            'full_frame_slots': 354, 'mask_accuracy': 'unmeasured; opaque encoded masks retained, not decoded',
            'scope': 'Sparse development envelope centers, not physical identity or simulator validation'}


def render_comparison(out, frames, manual):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    source = Path('data/general/corgi-hand-crank-van.webm')
    if digest(source) != manual['source_sha256']:
        raise ValueError('Original source changed before visual comparison')
    labels = {r['frame']: r for r in manual['observations']}
    figure, axes = plt.subplots(3, 2, figsize=(16, 14))
    capture = cv2.VideoCapture(str(source))
    selected, index = 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in labels:
                axis = axes.flat[selected]
                axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prediction, label = frames[index], labels[index]
                box = prediction.get('box_xyxy_halfopen')
                if box:
                    axis.add_patch(Rectangle((box[0],box[1]), box[2]-box[0],box[3]-box[1],
                                             fill=False, edgecolor='lime', linewidth=2))
                if label['center_px'] is not None:
                    axis.add_patch(Circle(label['center_px'], 16, fill=False, edgecolor='cyan', linewidth=2))
                suffix = 'initialization, unscored' if index == 0 else ('crossfade, unknown center' if index == 140 else prediction['status'])
                axis.set_title(f'Original frame {index}: {suffix}')
                axis.axis('off')
                selected += 1
            index += 1
        if index != 354 or selected != 6:
            raise ValueError('Visual comparison did not preserve all six original slots')
        figure.suptitle('Green: Meta box | Cyan: approximate source center + 16 px uncertainty\nMasks not decoded; no physical identity or simulation validation')
        figure.tight_layout()
        figure.savefig(out / 'comparison.png', dpi=120)
    finally:
        capture.release()
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    verified = verify_artifacts(args.input)
    if not args.video.resolve().is_relative_to(args.input.resolve()) or digest(args.video) != args.sha256:
        raise ValueError('Video must belong to the verified input bundle')
    manifest = json.loads((args.input / 'manifest.json').read_text())
    if manifest['files'].get(str(args.video.resolve().relative_to(args.input.resolve()))) != args.sha256:
        raise ValueError('Video is not sealed in the input bundle')
    preparation = json.loads((args.input / 'report.json').read_text())
    if (preparation.get('derivative_sha256') != args.sha256 or preparation.get('decoded_frames') != 354
            or preparation.get('frame_dimensions') != [1920,1080]
            or preparation.get('source_frame0_roi') != [970,538,108,68]
            or preparation.get('source_sha256') != 'caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8'
            or preparation.get('all_frame_mapping_verified') is not True
            or preparation.get('presentation_timestamps_all_equal') is not True
            or preparation.get('inline_size_ready') is not True):
        raise ValueError('Verified preparation does not match this frozen comparison')
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = args.out / 'inputs'
    inputs.mkdir()
    originals = [Path('data/general/corgi-hand-crank-van.manual.json'),
                 Path('data/general/corgi-hand-crank-van.initialization.json'),
                 Path('data/general/corgi-hand-crank-van.source.json'),
                 Path('src/weft/meta_segmentation.py'), Path('src/weft/meta_segmentation_parser.py'),
                 Path('tests/test_meta_segmentation.py'), Path('tests/test_meta_segmentation_parser.py'),
                 Path(__file__)]
    for source in originals:
        shutil.copyfile(source, inputs / source.name)
    shutil.copyfile(args.input / 'report.json', inputs / 'video-preparation.json')
    shutil.copyfile(args.input / 'manifest.json', inputs / 'video-manifest.json')
    write_json(inputs / 'freeze.json', {'created_at': local_time(), 'video_sha256': args.sha256,
        'input_verification': verified, 'full_interval': [0, 353], 'size': [1920,1080], 'nominal_fps': 59,
        'roi': [970,538,108,68], 'source_sha256': 'caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8',
        'implementation': {str(p): digest(p) for p in originals},
        'expected_api_usd': .0708, 'reservation_usd': .15, 'retries': 0, 'recurring_work': False,
        'one_shot_process_deadline_seconds': 160,
        'labels_sent_to_model': False, 'mask_decoding': False, 'default_change': False})
    write_json(inputs / 'report.json', {'status': 'frozen_before_request'})
    seal_suite(inputs)
    report = {'created_at': local_time(), 'status': 'prepared', 'input_bundle': str(args.input),
              'input_verification': verified, 'video_sha256': args.sha256,
              'simulation_gate': 'Unchanged rejection: edited/moving camera, unidentified 3D contacts and forces',
              'default_change': False}
    try:
        def deadline(*unused):
            raise TimeoutError('One-shot study reached its 160-second process deadline')
        previous_alarm_handler = signal.signal(signal.SIGALRM, deadline)
        signal.alarm(160)
        report['call'] = segment_video(args.video, args.out / 'call', expected_sha256=args.sha256,
            frame_count=354, width=1920, height=1080, roi=[970,538,108,68])
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        result = json.loads((args.out / 'call/segmentation.json').read_text())
        manual = json.loads((inputs / 'corgi-hand-crank-van.manual.json').read_text())
        report['comparison'] = grade(result['frames'], manual)
        render_comparison(args.out, result['frames'], manual)
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed_no_retry', error_type=type(error).__name__)
        raise
    finally:
        signal.alarm(0)
        report['inputs_unchanged'] = all(digest(p) == digest(inputs / p.name) for p in originals)
        write_json(args.out / 'report.json', report)
        seal_suite(args.out)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
