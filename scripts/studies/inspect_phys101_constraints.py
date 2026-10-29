"""Inspect construction clips while keeping the preselected repeat's pixels unopened."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts


SOURCE = Path('artifacts/phys101-constraints-source-01')
PLAN = Path('artifacts/phys101-constraints-plan-01')


def inspect(item):
    out, source, planned = item
    video = SOURCE / source['file']
    if digest(video) != source['sha256']:
        raise ValueError('Source identity changed')
    if planned['purpose'] == 'reserved-repeat':
        capture = cv2.VideoCapture(str(video))
        try:
            metadata = {key: capture.get(prop) for key, prop in [('width', cv2.CAP_PROP_FRAME_WIDTH),
                ('height', cv2.CAP_PROP_FRAME_HEIGHT), ('fps', cv2.CAP_PROP_FPS), ('frame_count', cv2.CAP_PROP_FRAME_COUNT)]}
        finally:
            capture.release()
        return source | {'purpose': planned['purpose'], 'metadata': metadata,
                         'frame_read_calls': 0, 'pixels_inspected': False}
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not 1 <= count <= 1800 or not 0 < fps <= 120 or max(width, height) > 4096:
        raise ValueError('Source exceeds inspection envelope')
    selected = sorted(set(np.linspace(0, count-1, 12).round().astype(int).tolist()))
    sheet = Image.new('RGB', (1280, 3 * (round(320 * height / width) + 30) + 48), '#F5EADC')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), 'Physics101 / ' + planned['purpose'] + ' / complete original source', fill='#214F50')
    saved, times = 0, []
    try:
        for frame in range(count):
            ok, bgr = capture.read()
            if not ok or bgr.shape[:2] != (height, width):
                raise ValueError('Source decode differs from container')
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if times and stamp <= times[-1]:
                raise ValueError('Non-increasing source clock')
            times.append(stamp)
            if frame in selected:
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                preview = out / f'{planned["purpose"]}-{frame:04d}.png'
                if preview.exists():
                    if not np.array_equal(np.asarray(Image.open(preview)), np.asarray(image)):
                        raise ValueError('Reused source preview differs from exact decoded pixels')
                else:
                    image.save(preview)
                image = image.resize((320, round(320 * height / width)))
                x, y = saved % 4 * 320, 48 + saved // 4 * (image.height + 30)
                sheet.paste(image, (x, y))
                draw.text((x + 6, y + image.height + 6), f'Frame {frame} / {stamp:.3f} s', fill='#214F50')
                saved += 1
        if capture.read()[0]:
            raise ValueError('Source contains extra frames')
    finally:
        capture.release()
    filename = f'{planned["purpose"]}-sheet.png'
    if (out / filename).exists():
        if not np.array_equal(np.asarray(Image.open(out / filename)), np.asarray(sheet)):
            raise ValueError('Reused source sheet differs from declared frames')
    else:
        sheet.save(out / filename)
    return source | {'purpose': planned['purpose'], 'frames_decoded': len(times), 'pixels_inspected': True,
        'width': width, 'height': height, 'fps': fps, 'timestamps_s': times,
        'maximum_nominal_clock_error_s': float(np.max(np.abs(np.asarray(times) - np.arange(count) / fps))),
        'sheet': filename, 'selected_frames': selected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reuse-previews', type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.parent != Path('artifacts').resolve():
        raise ValueError('Use a new top-level artifacts directory')
    verification = {str(p): verify_artifacts(p) for p in (SOURCE, PLAN)}
    source = json.loads((SOURCE / 'report.json').read_text())['extracted']
    plan = json.loads((PLAN / 'plan.json').read_text())['selected_members']
    if [r['member'] for r in source] != [r['name'] for r in plan]:
        raise ValueError('Acquisition differs from frozen plan')
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / Path(__file__).name)
    if args.reuse_previews:
        previous = args.reuse_previews.resolve()
        verification['reused_previews'] = verify_artifacts(previous)
        old = json.loads((previous / 'pre-run.json').read_text())
        if old['source_verification'] != {key: value for key, value in verification.items() if key != 'reused_previews'}:
            raise ValueError('Reusable previews have different inputs')
        for image in previous.glob('*.png'):
            shutil.copyfile(image, out / image.name)
    cv2.setNumThreads(1)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'source_verification': verification,
        'workers': 2, 'maximum_frames_per_source': 1800, 'maximum_source_dimension': 4096,
        'reserved_repeat': 'Container metadata only; no decoded frames, screenshots, tracking or fitting',
        'paid_calls': 0, 'model_inference': False})
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(inspect, [(out, a, b) for a, b in zip(source, plan, strict=True)]))
    report = {'status': 'construction_sources_inspected_repeat_unopened', 'created_at': local_time(),
        'runtime_seconds': time.perf_counter() - started, 'sources': results, 'source_verification': verification,
        'new_paid_calls': 0, 'tracking': False, 'physical_validation': False}
    write_json(out / 'report.json', report)
    figures = ''.join(f'<h2>{row["purpose"]}</h2><img src="{row["sheet"]}" alt="Original source frames">'
                      for row in results if row['pixels_inspected'])
    (out / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Additional physical observations</title>'
        '<style>body{font:17px system-ui;max-width:1280px;margin:32px auto;background:#F5EADC;color:#214F50;padding:20px}img{max-width:100%}</style>'
        '<h1>Additional physical observations</h1><p>Original construction sources, not simulations. The separate repeat remains unopened. '
        'Wu et al., Physics101 (2016); local/private research copies.</p>' + figures + '</html>')
    print(json.dumps({'clips': [{k: v for k, v in r.items() if k != 'timestamps_s'} for r in results],
                      'verification': seal_suite(out), 'runtime_seconds': report['runtime_seconds']}))


if __name__ == '__main__':
    main()
