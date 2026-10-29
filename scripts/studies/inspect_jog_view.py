"""Preserve one additional view's raw image/pose evidence without building or training a body."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_vision import inspect_exercise, model_provenance
from weft.io import digest, local_time, write_json
from weft.pipeline import _implementation_files, verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--view', choices=('drone2', 'drone3'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bundle, out = args.bundle.resolve(), args.out.resolve()
    if out.is_relative_to(bundle): raise ValueError('Keep view evidence outside the sealed source bundle')
    verification = verify_artifacts(bundle)
    video = bundle/(args.view+'.mp4')
    source_hash = digest(video)
    provenance = model_provenance(args.model)
    implementation = _implementation_files()
    hashes = {name: digest(path) for name, path in implementation.items()}
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'view': args.view,
        'source_bundle': str(bundle), 'source_verification': verification, 'video': str(video), 'source_sha256': source_hash,
        'model': provenance, 'pose_mode': 'image', 'pose_identity': 'raw',
        'maximum_source_frames': 240, 'inference_samples': 120, 'inference_seconds_bound': 180,
        'construction_role': 'fitting image evidence' if args.view == 'drone2' else 'development projection comparison only',
        'scope': 'Raw single-view observations; no body, camera fit, reference joint input or policy training',
        'paid_requests': 0})
    started = time.monotonic()
    clock, panels = [], []
    selected = [0, 120, 200, 239]
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if abs(fps-30000/1001) > 1e-9: raise ValueError('Extra view has an unexpected frame clock')
        for index in range(241):
            ok, bgr = capture.read()
            if not ok: break
            if index >= 240 or bgr.shape != (1080, 1920, 3): raise ValueError('Extra view dimensions or count differ')
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if abs(stamp-index/fps) > 1e-6: raise ValueError('Decoder clock differs from frame time')
            clock.append({'frame': index, 'time_s': index/fps, 'decoder_time_s': stamp})
            if index in selected:
                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((640, 360))
                panel = Image.new('RGB', (640, 400), '#F5EADC'); panel.paste(frame, (0, 40))
                ImageDraw.Draw(panel).text((10, 8), f'{args.view} / frame {index} / {stamp:.3f} s | Autelica (autelica.com), CC BY 4.0', fill='#214F50')
                panel.save(out/f'source-{index:04d}.png'); panels.append(panel)
    finally:
        capture.release()
    if len(clock) != 240: raise ValueError('Extra view does not cover its complete 240-frame interval')
    write_json(out/'source-clock.json', clock)
    strip = Image.new('RGB', (1280, 800))
    for index, panel in enumerate(panels): strip.paste(panel, ((index%2)*640, (index//2)*400))
    strip.save(out/'source-inspection.png')
    status, error = 'raw_pose_inference_completed', None
    try:
        evidence = inspect_exercise(video, out, args.model, pose_mode='image', pose_identity='raw')
    except UnsupportedVideo as failure:
        status, error = 'raw_pose_saved_legacy_body_gate_rejected', str(failure)
        evidence = json.loads((out/'evidence.json').read_text())
    if digest(video) != source_hash or verify_artifacts(bundle) != verification:
        raise RuntimeError('Source changed during inspection')
    if _implementation_files() != implementation or any(digest(path) != hashes[name] for name, path in implementation.items()):
        raise RuntimeError('Implementation changed during inference; evidence remains unsealed')
    for name, path in implementation.items():
        destination = out/'implementation'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if digest(destination) != hashes[name]: raise RuntimeError('Implementation changed during snapshot')
    write_json(out/'implementation.json', hashes)
    report = {'created_at': local_time(), 'status': status, 'legacy_body_gate_error': error,
        'view': args.view, 'source_sha256': source_hash, 'decoded_frames': len(clock),
        'sampled_frames': len(evidence['frames']), 'pose_statuses': dict(Counter(row['status'] for row in evidence['frames'])),
        'quality': evidence.get('quality'), 'runtime_seconds': time.monotonic()-started,
        'paid_requests': 0, 'new_body_or_policy': False, 'provider_joints_or_uv_used': False}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
