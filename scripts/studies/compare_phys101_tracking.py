"""Compare frozen model IDs with pre-request source labels; never reassign by error."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.generic_vision import evaluate_manual
from weft.io import local_time, write_json
from weft.pipeline import verify_artifacts


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    local = Path('artifacts/phys101-ramp-tracking-01')
    live = Path('artifacts/meta-phys101-ramp-live-01')
    evidence = Path('artifacts/meta-phys101-ramp-evidence-01')
    masks = Path('artifacts/meta-phys101-ramp-masks-01')
    inputs = [local, live, evidence, masks]
    before = {str(p): verify_artifacts(p) for p in inputs}
    out = args.out.resolve()
    if any(out.is_relative_to(p.resolve()) for p in inputs): raise ValueError('Use a new output outside sealed inputs')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    labels = read(local/'inputs/labels-20.json')
    trace = read(local/'tracking-20/appearance-tracks.json')
    sam = read(evidence/'segmentation-evidence.json')
    call = read(live/'call/call.json')
    video = read(live/'video.json')
    if trace['source_sha256'] != video['source_sha256'] or labels['source_sha256'] != video['source_sha256']:
        raise ValueError('Comparison sources differ')
    mapping = {'1': 'wood-block', '2': 'green-box'}
    converted, slots = [], []
    local_by_key = {(row['frame'], row['object_id']): row for row in trace['observations']}
    for row in sam['observations']:
        name = mapping[row['raw_model_id']]
        available = row['proposal_status'] == 'returned' and row['geometry'] is not None
        status = ('pre_initialization' if row['phase'] == 'pre_initialization' else
                  'initialized' if row['phase'] == 'initialization' else 'tracked' if available else 'missing')
        center = row['geometry']['center_px'] if available else [None, None]
        converted.append({'frame': row['frame'], 'object_id': name, 'status': status, 'x_px': center[0], 'y_px': center[1]})
        old = local_by_key[(row['frame'], name)]
        slots.append({'frame': row['frame'], 'time_s': row['time_s'], 'object_id': name, 'raw_model_id': row['raw_model_id'],
            'model_status': status, 'model_center_source_px': center if available else None,
            'local_status': old['status'], 'identity': row['identity']})
    if len(slots) != 88 or {r['frame'] for r in slots} != set(range(44)):
        raise ValueError('Comparison must retain both objects and every source frame')
    model_trace = {**trace, 'width': video['width'], 'height': video['height'], 'observations': converted,
                   'seconds': call['seconds'], 'camera_check': {'result': 'unknown_no_camera_estimation'}}
    grades = {name: evaluate_manual(value, labels) for name, value in [('local', trace), ('sam', model_trace)]}
    for grade in grades.values():
        grade['within_12px_label_uncertainty'] = sum(row.get('error_source_px', float('inf')) <= 12 for row in grade['rows'])
    paired = [(a, b) for a, b in zip(grades['local']['rows'], grades['sam']['rows'], strict=True)
              if a['status'] == b['status'] == 'tracked']
    common = {'slots': len(paired), 'jointly_accepted_objects': sorted({a['object_id'] for a, _ in paired}),
              'local_rmse_px': float(np.sqrt(np.mean([a['error_source_px']**2 for a, _ in paired]))) if paired else None,
              'sam_rmse_px': float(np.sqrt(np.mean([b['error_source_px']**2 for _, b in paired]))) if paired else None}
    write_json(out/'all-slots.json', slots)
    write_json(out/'manual-comparison.json', {'grades': grades, 'paired_common': common,
        'scope': 'Sparse source-only development labels; initialization is excluded and missing rows are not zero error'})
    capture = cv2.VideoCapture(str(live/'source.mp4'))
    images = []
    try:
        for frame in range(44):
            ok, bgr = capture.read()
            if not ok: raise ValueError('Source ended before complete inspection')
            if frame not in (16, 27, 43): continue
            image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((960, 540))
            draw = ImageDraw.Draw(image)
            for row in sam['observations']:
                if row['frame'] != frame or row['geometry'] is None: continue
                x1, y1, x2, y2 = [v/2 for v in row['geometry']['box_xyxy_halfopen']]
                draw.rectangle((x1, y1, x2, y2), outline='#00d7ff', width=2)
                draw.text((x1, y1-16), mapping[row['raw_model_id']], fill='#00d7ff', stroke_width=1, stroke_fill='#000000')
            for label in labels['annotations']:
                if label['frame'] != frame: continue
                x, y = [v/2 for v in label['center_xy']]
                draw.ellipse((x-6, y-6, x+6, y+6), outline='#ff9e32', width=2)
            panel = Image.new('RGB', (960, 646), '#F5EADC'); panel.paste(image, (0, 64))
            draw = ImageDraw.Draw(panel)
            draw.text((12, 10), f'Physics 101 / 20-degree ramp / frame {frame} | unchanged source and model IDs', fill='#214F50')
            draw.text((12, 32), 'Cyan: SAM boxes | Orange: frozen approximate centers with 12-source-pixel uncertainty', fill='#214F50')
            draw.text((12, 616), 'Wu et al., BMVC 2016. Model tracking, not validated contact forces or a simulated environment.', fill='#214F50')
            filename = f'frame-{frame:04d}.png'; panel.save(out/filename); images.append(filename)
    finally:
        capture.release()
    report = {'created_at': local_time(), 'input_verification': before, 'source_frames': 44, 'object_frame_slots': len(slots),
        'per_object': {name: {kind: dict(Counter(r[kind+'_status'] for r in slots if r['object_id'] == name))
                             for kind in ('local', 'model')} for name in mapping.values()},
        'manual_grades': {name: {k: v for k, v in grade.items() if k not in ('rows', 'annotation_sha256')} for name, grade in grades.items()},
        'paired_common': common, 'identity_groups': sam['identity_groups'],
        'mask_counts': read(masks/'report.json')['counts'], 'paid_requests_this_comparison': 0,
        'source_call_cost_usd': call['cost_usd'], 'source_call_seconds': call['seconds'],
        'physics_ready': False, 'source_video_altered': False, 'all_inputs_unchanged': True}
    if any(verify_artifacts(Path(p)) != value for p, value in before.items()): raise ValueError('Frozen input changed')
    write_json(out/'report.json', report)
    figures = ''.join(f'<figure><img src="{name}" alt="Source and inferred boxes at {name}"></figure>' for name in images)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Real ramp tracking comparison</title>'
        '<style>body{font:17px system-ui;background:#F5EADC;color:#214F50;max-width:960px;margin:2rem auto;padding:1rem}'
        'img{max-width:100%;height:auto}figure{margin:1rem 0}</style><h1>Real ramp tracking comparison</h1>'
        '<p>Tracking is not an executable physical environment. Source, starting boxes and comparison labels remain unchanged.</p>'
        '<p><a href="../meta-phys101-ramp-masks-01/index.html">Inspect every frame and silhouette</a> · '
        '<a href="report.json">Results and limitations</a> · <a href="manual-comparison.json">Every comparison label</a></p>'+figures)
    print(json.dumps({'grades': report['manual_grades'], 'paired_common': common, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
