"""Inspect saved image-coordinate failures without repairing or rerunning inference."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.camera_reference import JOINTS, project_world
from weft.io import digest, local_time, write_json


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'artifacts/multiview-jog-source-01'
    paths = [ROOT/'artifacts/multiview-jog-image-01/run/evidence.json',
             ROOT/'artifacts/multiview-jog-drone2-image-01/evidence.json', ROOT/'artifacts/multiview-jog-drone3-image-01/evidence.json']
    inputs = paths+[source/name for name in ('cameras.json', 'extrinsics.npz', 'body-coco17.npz', 'drone1.mp4')]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in inputs}
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': hashes,
        'scope': 'Post-hoc image-space diagnosis of an already observed failure, not a fresh holdout or fitted repair',
        'selected_frames': [200, 202], 'inference_calls': 0, 'paid_requests': 0})
    cameras = json.loads((source/'cameras.json').read_text())['cameras']
    with np.load(source/'extrinsics.npz', allow_pickle=False) as data: rotations, translations = data['R'], data['t']
    with np.load(source/'body-coco17.npz', allow_pickle=False) as data: joints, names = data['joints'], data['names'].tolist()
    poses = [{row['frame']: row for row in json.loads(path.read_text())['frames']} for path in paths]
    rows, selected = [], {}
    for view, camera in enumerate(cameras):
        for frame in range(240):
            _, projected, valid = project_world(joints[frame], rotations[view, frame], translations[view, frame], camera['K'], camera['radial_k1'])
            pose = poses[view].get(frame)
            points = pose.get('landmarks') if pose and pose.get('pose_count') == 1 else None
            for name, index in JOINTS.items():
                p = points[index] if points else None
                pixel = [p['x']*1920, p['y']*1080] if p else None
                confidence = min(p['visibility'], p['presence']) if p else None
                reference = projected[names.index(name)]
                row = {'view': view+1, 'frame': frame, 'joint': name, 'raw_image_px': pixel,
                       'raw_confidence': confidence, 'provider_projection_px': reference.tolist() if valid[names.index(name)] else None,
                       'distance_px': float(np.linalg.norm(np.asarray(pixel)-reference)) if pixel is not None and valid[names.index(name)] else None}
                rows.append(row)
                if view == 0 and frame in (200, 202): selected[frame, name] = row
    write_json(out/'image-comparison.json', {'joint_slots': len(rows), 'rows': rows})
    numbers = {}
    for frame in (200, 202):
        values = {name: selected[frame, name] for name in JOINTS}
        shoulder_y = np.mean([values[f'{side}_shoulder']['raw_image_px'][1] for side in ('left', 'right')])
        hip_y = np.mean([values[f'{side}_hip']['raw_image_px'][1] for side in ('left', 'right')])
        ankle_y = np.mean([values[f'{side}_ankle']['raw_image_px'][1] for side in ('left', 'right')])
        numbers[frame] = {'raw_shoulder_y_px': float(shoulder_y), 'raw_hip_y_px': float(hip_y), 'raw_ankle_y_px': float(ankle_y),
                          'shoulders_below_hips_in_image': bool(shoulder_y > hip_y), 'ankles_above_hips_in_image': bool(ankle_y < hip_y)}
    pixels = np.array([row[key] for row in selected.values() for key in ('raw_image_px', 'provider_projection_px') if row[key] is not None])
    low = np.maximum(0, np.floor(pixels.min(axis=0)-50)).astype(int)
    high = np.minimum([1920, 1080], np.ceil(pixels.max(axis=0)+50)).astype(int)
    scale = min(460/(high[0]-low[0]), 500/(high[1]-low[1]))
    sheet = Image.new('RGB', (960, 610), '#F5EADC')
    capture = cv2.VideoCapture(str(source/'drone1.mp4'))
    try:
        for frame in range(203):
            ok, bgr = capture.read()
            if not ok: raise ValueError('Source ended before diagnostic frames')
            if frame not in (200, 202): continue
            cropped = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).crop((*low, *high))
            cropped = cropped.resize((round(cropped.width*scale), round(cropped.height*scale)))
            draw = ImageDraw.Draw(cropped)
            for joint, label, color in [('left_shoulder', 'LS', '#e53935'), ('right_shoulder', 'RS', '#e53935'),
                ('left_hip', 'LH', '#0a9f44'), ('right_hip', 'RH', '#0a9f44'), ('left_ankle', 'LA', '#1468ec'), ('right_ankle', 'RA', '#1468ec')]:
                x, y = (np.asarray(selected[frame, joint]['raw_image_px'])-low)*scale
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline='white')
                draw.text((x+6, y-7), label, fill=color, stroke_width=1, stroke_fill='white')
            offset = 0 if frame == 200 else 480
            sheet.paste(cropped, (offset+10, 80))
            label = ImageDraw.Draw(sheet)
            label.text((offset+10, 12), f'Original Lite image estimates / frame {frame}', fill='#214F50')
            label.text((offset+10, 32), 'Red: shoulders | Green: hips | Blue: ankles', fill='#214F50')
            label.text((offset+10, 52), 'Labels are raw model outputs, not manual corrections.', fill='#214F50')
    finally:
        capture.release()
    ImageDraw.Draw(sheet).text((10, 590), 'Autelica (autelica.com), CC BY 4.0 | Shared crop/scale; no inference or pose repair', fill='#214F50')
    sheet.save(out/'raw-image-labels.png')
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in inputs}: raise RuntimeError('Diagnostic inputs changed')
    report = {'created_at': local_time(), 'image_joint_slots': len(rows), 'selected_raw_image_geometry': numbers,
        'scope': 'The observed jogging failure already exists in raw image labels; this does not certify calibration or synchronization',
        'image_y_axis': 'increases downward', 'repair_applied': False, 'paid_requests': 0}
    write_json(out/'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
