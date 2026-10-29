"""Inspect frozen worst/conflicting hip cases in their original construction images."""

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]
ASSOCIATIONS = ROOT/'artifacts/freemocap-joint-association-01'
JOINTS = ROOT/'artifacts/freemocap-camera-space-joints-01'
POSES = ROOT/'artifacts/freemocap-full-pose-01'
EVIDENCE = ROOT/'artifacts/freemocap-full-evidence-01'


def read(path):
    return json.loads(path.read_text())


def verified(root, name):
    path = root/name
    if digest(path) != read(root/'manifest.json')['files'][name]:
        raise ValueError('Frozen inspection input changed: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out.parent != ROOT/'artifacts':
        raise ValueError('Use a new top-level artifact directory')
    diagnostic = read(verified(ASSOCIATIONS, 'report.json'))
    selected = [diagnostic['cross_only_active_hip_frames'][0], 550, diagnostic['worst_active_hip_frame']]
    views = [read(verified(POSES, f'view-{i}/evidence.json')) for i in (1, 2)]
    poses = [{r['frame']: r for r in v['frames']} for v in views]
    joints = read(verified(JOINTS, 'camera-space-joints.json'))['frames']
    sources = read(verified(POSES, 'pre-run.json'))['views']
    if selected != [416, 550, 966]:
        raise ValueError('Frozen selection changed')
    out.mkdir(exist_ok=False); shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'selected_frames': selected,
        'selection': ['Largest raw-to-crossed hip error reduction among cross-only active samples',
            'Previously exposed standing reference', 'Largest active raw hip error'],
        'input_manifests': {p.name: digest(p/'manifest.json') for p in (ASSOCIATIONS, JOINTS, POSES, EVIDENCE)},
        'maximum_decoded_frames': 4, 'maximum_seconds': 30, 'new_inference': 0, 'new_geometry': 0,
        'reserved_camera_opened': False, 'paid_calls': 0,
        'scope': 'Development diagnosis only. Original image estimates are not anatomical ground truth.'})
    started = time.perf_counter(); decoded = 0; files = {}
    for view, source in enumerate(sources, 1):
        video = ROOT/source['path']
        if digest(video) != source['sha256']:
            raise ValueError('Video identity changed')
        cap = None
        try:
            for frame in selected:
                target = out/f'view-{view}-source-{frame:04d}.png'
                if (EVIDENCE/target.name).exists():
                    shutil.copyfile(verified(EVIDENCE, target.name), target)
                else:
                    if cap is None:
                        cap = cv2.VideoCapture(str(video))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame); ok, image = cap.read()
                    if (not ok or image.shape[:2] != (1280, 720)
                            or abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(frame+1)) > 1e-8
                            or abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/30) > 1e-7):
                        raise ValueError('Exact inspection frame unavailable')
                    if not cv2.imwrite(str(target), image):
                        raise ValueError('Inspection image export failed')
                    decoded += 1
                files[view, frame] = target
        finally:
            if cap is not None:
                cap.release()
    detail = []
    for frame in selected:
        canvas = Image.new('RGB', (1440, 1340), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'Frame {frame}: ORIGINAL image estimates; red=left, cyan=right; X=inconsistent stereo rays', fill='black')
        draw.text((12, 26), 'No label swap, retiming, coordinate repair, floor or body applied.', fill='black')
        for view in (1, 2):
            image = Image.open(files[view, frame]).convert('RGB'); ink = ImageDraw.Draw(image)
            row = poses[view-1][frame]
            for a, b in ((11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)):
                points = [row['landmarks'][j] for j in (a, b)]
                xy = [(p['x']*720, p['y']*1280) for p in points]
                ink.line(xy, fill='#ffff00', width=2)
            for joint in (11, 12, 23, 24, 25, 26, 27, 28):
                p = row['landmarks'][joint]; x, y = p['x']*720, p['y']*1280
                color = '#ff3434' if joint%2 else '#00ffff'
                ink.ellipse((x-5, y-5, x+5, y+5), outline=color, width=2)
                ink.text((x+7, y-4), str(joint), fill=color, stroke_width=1, stroke_fill='black')
                if joints[frame]['joints'][joint]['status'] == 'inconsistent_image_rays':
                    ink.line((x-8, y-8, x+8, y+8), fill=color, width=2)
                    ink.line((x-8, y+8, x+8, y-8), fill=color, width=2)
            canvas.paste(image, ((view-1)*720, 60))
            detail.append({'frame': frame, 'view': view, 'original_pose': row,
                'hip_rays': [joints[frame]['joints'][j] for j in (23, 24)],
                'source_image': files[view, frame].name, 'source_image_sha256': digest(files[view, frame])})
        canvas.save(out/f'original-hips-{frame:04d}.png')
    write_json(out/'selected-original-evidence.json', detail)
    if time.perf_counter()-started > 30:
        raise TimeoutError('Inspection exceeded its work envelope')
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'selected_source_evidence_exported',
        'selected_frames': selected, 'new_source_frames_decoded': decoded, 'cached_images_reused': 6-decoded,
        'runtime_seconds': time.perf_counter()-started, 'new_inference': 0, 'new_geometry': 0,
        'repair_applied': False, 'reserved_camera_opened': False, 'paid_calls': 0})
    print(json.dumps({'decoded': decoded, 'frames': selected, 'verification': seal_suite(out)}))


if __name__ == '__main__':
    main()
