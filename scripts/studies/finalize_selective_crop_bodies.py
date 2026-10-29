"""Finish the saved body study without repeating completed inference or learning."""

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import time

from weft.benchmark import seal_suite
from weft.errors import UnsupportedVideo
from weft.human import build_human
from weft.io import digest, local_time, write_json
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from check_body_refinement import limb_predictions
from check_crop_diversity import CLIPS, LIMB_LABELS, deltas


ROOT = Path(__file__).resolve().parents[2]
POSES = ROOT/'artifacts/selective-crop-videos-03'
BODIES = ROOT/'artifacts/selective-crop-bodies-01'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if any(out.is_relative_to(source) for source in (POSES, BODIES)):
        raise ValueError('Use a new output outside the original studies')
    inputs = {str(source): verify_artifacts(source) for source in (POSES, BODIES)}
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    if read(BODIES/'pre-run.json')['implementation'] != implementation:
        raise ValueError('Original body construction/runtime changed')
    out.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    for path in (Path(__file__), Path(__file__).with_name('check_body_refinement.py'), Path(__file__).with_name('check_crop_diversity.py')):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'inputs': inputs,
        'implementation': implementation, 'labels_sha256': digest(LIMB_LABELS),
        'scope': 'Reuse two completed bodies/recordings and two pose rejections; test the previously unattempted selected jumping-jacks body',
        'maximum_new_body_builds': 1, 'new_training': False, 'new_inference': False, 'paid_requests': 0})
    clip = 'jumping-jacks-burpees'; attempt = out/clip; attempt.mkdir()
    old_root = read(BODIES/clip/'whole_body/root-motion.json')
    if old_root['status'] == 'ready': raise ValueError('The recorded failure no longer matches this recovery plan')
    try:
        build_human(read(POSES/clip/'selected/evidence.json'), attempt, body_model='mobile')
    except UnsupportedVideo as error:
        candidate = {'status': 'body_rejected', 'body_constructed': False, 'original_errors': old_root['quality']['errors'],
            'selected_errors': [str(error)], 'root_audit_artifact': str(attempt/'root-motion.json'),
            'scope': 'Whole interval retained; no anchored fallback, trimmed gaps or invented observations'}
    else:
        write_json(attempt/'report.json', {'status': 'candidate_body_constructed_needs_remaining_checks',
            'body_constructed': True, 'scope': 'Stop before comparison or promotion; run remaining controls/recording checks'})
        seal_suite(attempt)
        raise RuntimeError('Selected body unexpectedly succeeded; retain it and complete remaining checks separately')
    write_json(attempt/'report.json', candidate); seal_suite(attempt)
    reports = {name: read(BODIES/name/'report.json') for name in CLIPS if name != clip}
    reports[clip] = candidate
    write_json(out/'construction.json', {'created_at': local_time(), 'status': 'all_body_outcomes_fixed_before_scoring',
        'cases': {name: {'status': report['status'], 'body_constructed': report['body_constructed']} for name, report in reports.items()}})
    labels = load_annotations(LIMB_LABELS, project_root=ROOT)
    if digest(LIMB_LABELS) != read(out/'pre-run.json')['labels_sha256']:
        raise ValueError('Development labels changed')
    scores = {}
    for key, path in (('before', 'whole_body'), ('after', 'selected_lab/run')):
        specs = {name: read(BODIES/name/path/'human.json') for name in CLIPS if reports[name]['body_constructed']}
        predictions, matching = limb_predictions(specs, labels)
        write_json(out/(key+'-predictions.json'), {'predictions': predictions, 'matching': matching,
            'projection': 'Fixed simulation x and -z; unchanged 1/23-second matching; rejected bodies remain missing'})
        scores[key] = grade_limb_projection(labels, predictions)
    changes = deltas(scores['before']['observations'], scores['after']['observations'], 'within_tolerance')
    write_json(out/'body-label-comparison.json', {'scores': scores, 'changes': changes})
    new_root = read(attempt/'root-motion.json')
    gaps = {key: {'quality_errors': root['quality']['errors'],
        'unfilled_gaps': len([row for row in root['quality']['gaps'] if row['status'] == 'unfilled']),
        'long_reference_intervals': len(root['quality']['long_reference_intervals']),
        'maximum_reference_gap_seconds': max(row['seconds'] for row in root['quality']['long_reference_intervals'])}
        for key, root in (('before', old_root), ('after', new_root))}
    for source, expected in inputs.items():
        if verify_artifacts(Path(source)) != expected: raise ValueError('A saved study changed during recovery')
    if implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Runtime changed during recovery')
    report = {'status': 'all_five_body_outcomes_compared', 'created_at': local_time(),
        'cases': reports, 'body_label_summaries': {key: score['summary'] for key, score in scores.items()},
        'body_label_changes': {key: value for key, value in changes.items() if key != 'paired_rows'},
        'jumping_jacks_root_gaps': gaps, 'completed_labs': {name: str(BODIES/name/'selected_lab')
            for name in CLIPS if reports[name]['body_constructed']},
        'all_crop_anatomy_and_contact_unverified': True,
        'new_training': False, 'new_inference': False, 'paid_requests': 0,
        'runtime_seconds': time.monotonic()-started,
        'scope': 'Two playable candidates, two competing-hypothesis rejections and one root-gap rejection; not general physical accuracy'}
    write_json(out/'report.json', report)
    page = ['<!doctype html><meta charset="utf-8"><title>Exercise sandbox results</title>',
        '<style>body{font:16px system-ui;max-width:1440px;margin:2rem auto;padding:0 1rem;background:#F5EADC;color:#214F50}img{width:100%;height:auto}pre{overflow:auto}section{margin:3rem 0}a{color:#965644}</style>',
        '<h1>Exercise sandbox results</h1><p>Two complete interactive candidates; three rejected clips. Muscle visuals are illustrative. Anatomy, contacts and changed physical actions are not verified.</p>']
    for name in CLIPS:
        case = reports[name]; page.append('<section><h2>'+html.escape(name)+'</h2>')
        if not case['body_constructed']:
            page.append('<p>Not built: '+html.escape('; '.join(case['selected_errors']))+'</p></section>'); continue
        command = f'.venv-lab/bin/python -m weft lab --resume {BODIES/name/"selected_lab"}'
        recording = os.path.relpath(case['interaction']['recording'], out)
        page.append('<p>Start locally:</p><pre>'+html.escape(command)+'</pre>')
        page.append(f'<p><a href="{recording}/replay.html">Recorded interaction</a> · <a href="{recording}/video.mp4">Recorded video</a></p>')
        page.append(f'<p>{case["selected_body_poses"]} body poses; {case["floor_after"]["failed_source_poses"]} source-pose floor failures. Exact recording replay is verified, not physical accuracy.</p>')
        for frame in case['accepted_image_frames']:
            path = os.path.relpath(BODIES/name/f'comparison-{frame:04d}.png', out)
            page.append(f'<img loading="lazy" src="{path}" alt="Source and body estimates for frame {frame}">')
        page.append('</section>')
    (out/'index.html').write_text('\n'.join(page))
    print(json.dumps({'verification': seal_suite(out), 'body_label_summaries': report['body_label_summaries'],
        'body_label_changes': report['body_label_changes'], 'root_gaps': gaps,
        'completed_labs': report['completed_labs'], 'runtime_seconds': report['runtime_seconds']}, indent=2))


if __name__ == '__main__': main()
