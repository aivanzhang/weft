"""Decode complete preselected ramp clips and retain source-only inspection sheets."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import local_time, write_json
from weft.pipeline import verify_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = Path('artifacts/phys101-ramp-source-01').resolve()
    before = verify_artifacts(source)
    out = args.out.resolve()
    if out.is_relative_to(source): raise ValueError('Output must not alter source files')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'input_verification': before,
        'maximum_frames_per_video': 1800, 'videos': 2, 'paid_requests': 0,
        'selection': 'Full first-trial front-camera clips chosen before source inspection; no trimming'})
    results = []
    for index, angle in enumerate((10, 20)):
        video = source/f'member-{index:02d}-Camera_1.mp4'
        capture = cv2.VideoCapture(str(video))
        fps, advertised = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0 < fps <= 120 or not 1 <= advertised <= 1800 or max(width, height) > 4096:
            raise ValueError('Unexpected source dimensions/frame bounds')
        selected = sorted(set(np.linspace(0, advertised-1, 12).round().astype(int).tolist()))
        sheet = Image.new('RGB', (1280, 3*(round(320*height/width)+32)+48), '#F5EADC')
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), f'Physics 101 / w_block_01 / {angle} degrees / complete original front-camera clip', fill='#214F50')
        stamps, saved = [], 0
        try:
            for frame in range(1801):
                ok, bgr = capture.read()
                if not ok: break
                if frame == 1800: raise ValueError('Decoded source exceeds frame bound')
                stamp = capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                if stamps and stamp <= stamps[-1]: raise ValueError('Source decoder timestamps are not increasing')
                stamps.append(stamp)
                if frame in selected:
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    if frame in (0, advertised-1): image.save(out/f'ramp-{angle}-frame-{frame:04d}.png')
                    image = image.resize((320, round(320*height/width)))
                    x, y = (saved%4)*320, 48+(saved//4)*(image.height+32)
                    sheet.paste(image, (x, y))
                    draw.text((x+6, y+image.height+6), f'Frame {frame} / {stamp:.3f} s', fill='#214F50')
                    saved += 1
        finally:
            capture.release()
        if len(stamps) != advertised: raise ValueError('Declared and actual frame counts disagree')
        sheet.save(out/f'ramp-{angle}-sheet.png')
        results.append({'ramp_degrees': angle, 'video': str(video), 'width': width, 'height': height,
            'fps': fps, 'advertised_frames': advertised, 'decoded_frames': len(stamps),
            'first_time_s': stamps[0], 'last_time_s': stamps[-1], 'timestamps_s': stamps,
            'maximum_nominal_clock_error_s': float(np.max(np.abs(np.asarray(stamps)-np.arange(len(stamps))/fps))),
            'sheet': f'ramp-{angle}-sheet.png', 'selected_frames': selected})
    if verify_artifacts(source) != before: raise ValueError('Source files changed')
    write_json(out/'report.json', {'created_at': local_time(), 'status': 'complete_sources_decoded',
        'clips': results, 'input_verification': before, 'paid_requests': 0, 'physics_validated': False,
        'scope': 'Source-only inspection; no tracking, parameter fitting, simulation or training'})
    print(json.dumps({'clips': [{k: v for k, v in row.items() if k != 'timestamps_s'} for row in results],
                      'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
