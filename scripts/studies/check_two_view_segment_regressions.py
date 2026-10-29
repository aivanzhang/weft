"""Inspect regressions in sealed segment candidates without changing their geometry."""

import argparse
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw

from weft.benchmark import seal_suite
from weft.io import local_time, write_json
from weft.pipeline import verify_artifacts
from weft.recording import render_snapshot


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    study = Path('artifacts/two-view-segment-jog-study-01')
    runs = {name: Path(f'artifacts/{prefix}-jog-lab-01-both/run')
            for name, prefix in [('v1', 'two-view-body'), ('v2', 'two-view-segment')]}
    inputs = {'comparison': study, **runs}
    before = {name: verify_artifacts(path) for name, path in inputs.items()}
    out = args.out.resolve()
    if any(out.is_relative_to(path.resolve()) for path in inputs.values()):
        raise ValueError('Inspection output must not alter a sealed input')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out/Path(__file__).name)
    rows = [row for row in read(study/'comparison.json')['rows'] if row['body_available']]
    summary = {}
    for name in ('directions', 'both'):
        deltas = [{'frame': row['frame'], 'joint': row['joint'],
                   'before_m': row['errors']['v1_'+name]['absolute_m'],
                   'after_m': row['errors'][name]['absolute_m'],
                   'delta_m': row['errors'][name]['absolute_m']-row['errors']['v1_'+name]['absolute_m']}
                  for row in rows]
        summary[name] = {'compared': len(deltas),
            'improved': sum(row['delta_m'] < -1e-12 for row in deltas),
            'worsened': sum(row['delta_m'] > 1e-12 for row in deltas),
            'unchanged': sum(abs(row['delta_m']) <= 1e-12 for row in deltas),
            'largest_regression': max(deltas, key=lambda row: row['delta_m'])}
        write_json(out/f'{name}-deltas.json', deltas)
    frame = summary['both']['largest_regression']['frame']
    evidence = read(runs['v2']/'evidence.json')
    capture = cv2.VideoCapture(str(runs['v2']/'source.mp4'))
    try:
        for _ in range(frame+1):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('Source ended before inspection frame')
        if abs(capture.get(cv2.CAP_PROP_POS_MSEC)/1000-frame/evidence['fps']) > 1e-6:
            raise ValueError('Decoded clock differs from retained evidence')
        source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).resize((480, 270))
    finally:
        capture.release()
    panel = Image.new('RGB', (1440, 680), '#F5EADC')
    draw = ImageDraw.Draw(panel)
    draw.text((16, 12), f'Frame {frame}: largest ankle regression after upper-segment correction', fill='#214F50')
    panel.paste(source, (0, 160))
    draw.text((16, 46), 'Original source: Autelica (autelica.com), CC BY 4.0', fill='#214F50')
    audits = {}
    for index, (name, run) in enumerate(runs.items(), 1):
        reference = read(run/'human-reference.json')['frames']
        snapshot = next(row for row in reference if round(row['source_time_s']*evidence['fps']) == frame)
        panel.paste(Image.fromarray(render_snapshot(snapshot, read(run/'scene.json'), width=480)), (index*480, 80))
        draw.text((index*480+16, 46), name+': '+('complete limbs only' if name == 'v1' else 'supported upper segment'), fill='#214F50')
        audits[name] = next(row for row in read(run/'two-view-body.json')['frames'] if row['frame'] == frame)
    draw.text((16, 610), 'Body views follow the body, not the source camera. These are kinematic poses, not physical trials.', fill='#214F50')
    draw.text((16, 632), 'The ankle has inconsistent image rays. Its bend and twist remain priors; no missing observation is filled.', fill='#965644')
    draw.text((16, 654), 'A better average score does not validate this pose. Illustrative muscles are not measured activation.', fill='#965644')
    panel.save(out/f'regression-{frame:04d}.png')
    write_json(out/'frame-audits.json', audits)
    if {name: verify_artifacts(path) for name, path in inputs.items()} != before:
        raise ValueError('A sealed input changed during inspection')
    write_json(out/'report.json', {'created_at': local_time(), 'input_verification': before,
        'comparison': summary, 'inspection_frame': frame, 'all_inputs_unchanged': True,
        'scope': 'Post-hoc development diagnosis, not a fresh holdout or candidate selection',
        'geometry_changed': False, 'paid_requests': 0})
    print(json.dumps({'summary': summary, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
