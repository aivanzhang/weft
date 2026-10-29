"""Build, compare and record full-video pose candidates without promoting anatomy."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.benchmark import probe_interaction, seal_suite
from weft.errors import UnsupportedVideo
from weft.exercise_source import describe_source
from weft.human import HumanEnv, build_human, evaluate_reference
from weft.human_learning import train_human
from weft.human_task import propose_joint_task
from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import HumanSession, Lab
from weft.limb_fidelity import grade_limb_projection, load_annotations
from weft.pipeline import _implementation_files, verify_artifacts
from weft.recording import load_recording, render_snapshot, verify_recording
from weft.whole_body_feasibility import audit_reference_feasibility
from check_body_refinement import limb_predictions
from check_crop_diversity import CLIPS, LIMB_LABELS, deltas


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/selective-crop-videos-03'
EPISODES, EVALUATIONS, SEED = 100, 20, 7311


def read(path):
    return json.loads(path.read_text())


def panels(out, clip, before, after, frames):
    envs = [HumanEnv(spec) for spec in (before, after)]
    indices = [{int(row['evidence_id'].rsplit(':', 1)[1]): i for i, row in enumerate(spec['reference_evidence'])}
               for spec in (before, after)]
    try:
        for frame in frames:
            source = Image.open(SOURCE/clip/f'source-{frame:04d}.png').convert('RGB')
            tiles = [source]
            for env, index in zip(envs, indices):
                if frame in index:
                    tiles.append(Image.fromarray(render_snapshot(env.reference_snapshot(index[frame]), env.scene, width=480)))
                else:
                    blank = Image.new('RGB', (480, 360), '#F5EADC')
                    ImageDraw.Draw(blank).text((20, 20), 'No usable original pose at this frame', fill='#214F50')
                    tiles.append(blank)
            panel = Image.new('RGB', (1440, 420), '#F5EADC'); draw = ImageDraw.Draw(panel)
            for i, (label, tile) in enumerate(zip(('Source video', 'Original body estimate', 'Selective-crop body estimate'), tiles)):
                draw.text((i*480+10, 10), f'{label} | frame {frame}', fill='#214F50')
                tile.thumbnail((480, 360)); panel.paste(tile, (i*480, 35))
            draw.text((10, 400), 'Inferred body and authored floor/forces. Muscle bands are illustrative, not measured.', fill='#214F50')
            panel.save(out/f'comparison-{frame:04d}.png')
    finally:
        for env in envs: env.close()


def floor_changes(before, after):
    old = {(row['kind'], row['time_s']): row for row in before['checks']}
    new = {(row['kind'], row['time_s']): row for row in after['checks']}
    paired = sorted(old.keys() & new.keys())
    regressions = [{'kind': kind, 'time_s': stamp, 'old': old[(kind, stamp)]['max_penetration_model_units'],
                    'new': new[(kind, stamp)]['max_penetration_model_units']} for kind, stamp in paired
                   if old[(kind, stamp)]['passes_floor_limit'] and not new[(kind, stamp)]['passes_floor_limit']]
    recoveries = sum(not old[k]['passes_floor_limit'] and new[k]['passes_floor_limit'] for k in paired)
    return {'paired_checks': len(paired), 'new_checks': len(new.keys()-old.keys()),
        'missing_old_checks': len(old.keys()-new.keys()), 'new_floor_failures_on_paired': regressions,
        'recovered_floor_checks': recoveries,
        'scope': 'Matched timestamps and fixed floor limit; pose-derived proportions/root fitting can change the entire body'}


def build_case(out, clip):
    started = time.monotonic(); directory = out/clip; directory.mkdir()
    report = read(SOURCE/clip/'report.json')
    before_evidence = read(SOURCE/clip/'whole_image/evidence.json')
    after_evidence = read(SOURCE/clip/'selected/evidence.json')
    if not before_evidence['quality']['passes'] or not after_evidence['quality']['passes']:
        write_json(directory/'report.json', {'status': 'perception_rejected', 'clip': clip,
            'original_errors': before_evidence['quality']['errors'], 'selected_errors': after_evidence['quality']['errors'],
            'source_frame_count': after_evidence['decoded_frames'], 'retained_sample_count': len(after_evidence['frames']),
            'body_constructed': False, 'scope': 'Competing hypotheses are not collapsed or silently excluded'})
        seal_suite(directory); return
    baseline = directory/'whole_body'; baseline.mkdir()
    try:
        before = build_human(before_evidence, baseline, body_model='mobile')
    except UnsupportedVideo as original_error:
        write_json(baseline/'report.json', {'status': 'body_rejected', 'error': str(original_error)})
        seal_suite(baseline)
        attempt = directory/'selected_body_attempt'; attempt.mkdir()
        try:
            build_human(after_evidence, attempt, body_model='mobile')
        except UnsupportedVideo as selected_error:
            failure = {'status': 'body_rejected', 'clip': clip, 'body_constructed': False,
                'original_errors': [str(original_error)], 'selected_errors': [str(selected_error)],
                'scope': 'Both complete candidates attempted; no shortened interval or anchored fallback'}
            write_json(attempt/'report.json', failure); seal_suite(attempt)
            write_json(directory/'report.json', failure); seal_suite(directory)
            return
        write_json(attempt/'report.json', {'status': 'candidate_body_constructed_needs_remaining_checks',
            'scope': 'Preserve this successful construction and complete its controls/recording before promotion'})
        seal_suite(attempt)
        raise RuntimeError('Original body rejected but selected body built; retain it and complete remaining checks separately')
    before_floor = audit_reference_feasibility(before)
    write_json(baseline/'reference-feasibility.json', before_floor)
    write_json(baseline/'report.json', {'status': 'baseline_body_constructed', 'source': str(SOURCE/clip/'whole_image'),
        'reference_frames': len(before['reference']), 'reference_feasibility': before_floor['summary'], 'physical_validation': 'unverified'})
    baseline_verification = seal_suite(baseline)
    lab_root = directory/'selected_lab'; lab_root.mkdir(); run = lab_root/'run'; run.mkdir()
    input_run = SOURCE/clip/'selected'; input_manifest = read(input_run/'manifest.json')
    for name, checksum in input_manifest['files'].items():
        destination = run/('input-pose-report.json' if name == 'report.json' else name)
        shutil.copyfile(input_run/name, destination)
        if digest(destination) != checksum: raise ValueError('Pose input changed while copying')
    shutil.copyfile(input_run/'manifest.json', run/'input-pose-manifest.json')
    for name in ('crop-plan.json', 'decisions.json', 'crop-observations.jsonl'):
        shutil.copyfile(SOURCE/clip/name, run/name)
    video = ROOT/'data/exercise'/(clip+'.webm'); shutil.copyfile(video, run/'source.webm')
    if digest(run/'source.webm') != after_evidence['source_sha256']: raise ValueError('Source bytes changed')
    source = describe_source(after_evidence['source_sha256'])
    if source is None: raise ValueError('Expected attributed exercise source is not registered')
    write_json(run/'source.json', source | {'artifact': 'source.webm', 'sha256': after_evidence['source_sha256']})
    unchanged = before_evidence['frames'] == after_evidence['frames']
    if unchanged:
        for name in ('human.json', 'human.xml', 'body-frame.json', 'spatial-pose.json', 'root-motion.json', 'floor-assumption.json'):
            shutil.copyfile(baseline/name, run/name)
        after, after_floor = deepcopy(before), deepcopy(before_floor)
    else:
        after = build_human(after_evidence, run, body_model='mobile')
        after_floor = audit_reference_feasibility(after)
    write_json(run/'reference-feasibility.json', after_floor)
    task = propose_joint_task(after_evidence, after)
    write_json(run/'task-selection.json', task)
    settings = {'schema_version': 1, 'profile': 'exercise', 'video': str(video), 'pose_mode': 'image',
        'pose_identity': 'raw', 'body_model': 'mobile', 'episodes': EPISODES, 'eval_episodes': EVALUATIONS,
        'seed': SEED, 'pose_refinement': 'single-view-selective-crop-v1', 'source_pose_study': str(SOURCE),
        'created_at': local_time(), 'bounds': {'wall_seconds': 180, 'external_spend_usd': 0, 'recurring_services': 0}}
    write_json(run/'run-config.json', settings); write_json(run/'model.json', after_evidence['model'])
    env = HumanEnv(after)
    try:
        env.select_joint(task['selected_joint']); env.reset(seed=0, options={'jitter': False})
        env.scene['joint_names'] = list(env.joint_names)
        write_json(run/'scene.json', env.scene); write_json(run/'task.json', env.scene['task'])
        indices = list(range(0, len(after['reference']), max(1, math.ceil(len(after['reference'])/120))))
        references = [env.reference_snapshot(i) for i in indices]
        write_json(run/'human-reference.json', {'source_indices': indices, 'frames': references,
            'scope': 'Sampled kinematic inspection; all source-linked body references remain in human.json'})
        images = [Image.fromarray(render_snapshot(state, env.scene, width=480))
                  for state in references[::max(1, math.ceil(len(references)/32))]]
        images[0].save(run/'reconstruction.gif', save_all=True, append_images=images[1:], duration=150, loop=0)
    finally:
        env.close()
    dynamic = evaluate_reference(after, run)
    learning = train_human(after, run, episodes=EPISODES, eval_episodes=EVALUATIONS, seed=SEED,
                           selected_joint=task['selected_joint'])
    result = {'status': 'executable_unpromoted_candidate', 'profile': 'exercise', 'source': source,
        'learning': learning, 'task_selection': task, 'pose_quality': after_evidence['quality'],
        'reference_tracking': dynamic, 'reference_feasibility': {'status': after_floor['status'], 'summary': after_floor['summary']},
        'reference_frames': len(after['reference']), 'assumptions': after['assumptions'],
        'seconds': time.monotonic()-started, 'physical_validation': 'unverified',
        'cost': {'external_api_usd': 0, 'cloud_usd': 0}, 'limits': [
            'Crop consistency does not establish anatomy; source poses and body geometry remain estimates',
            'Authored floor, masses, support and forces; contact failures remain visible',
            'Muscle bands are illustrative motor-command proxies, not measured activation',
            'Learning targets one selected joint, not the complete exercise']}
    write_json(run/'report.json', result)
    write_json(run/'reconstruction.json', {'scope': 'Mobile inferred body with authored floor/forces; no physical accuracy claim',
        'reference_frames': len(after['reference']), 'physical_validation': 'unverified'})
    write_json(run/'repair-ledger.json', {'proposals': [], 'scope': 'No physical model repair promoted; image decisions are separate'})
    implementation = read(out/'pre-run.json')['implementation']
    for name, path in _implementation_files().items():
        destination = run/'implementation-source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if digest(destination) != implementation[name]: raise ValueError('Runtime changed during construction')
    write_json(run/'implementation.json', implementation)
    run_verification = seal_suite(run)
    write_json(lab_root/'lab-config.json', settings)
    events = [{'index': i, 'stage': stage, 'time': local_time(), 'message': message} for i, (stage, message) in enumerate([
        ('source', 'Reused complete source-pinned pose study; original observations retained'),
        ('observe', 'Selective crops retain missing and competing hypotheses'),
        ('world', 'Built a mobile body with illustrative muscles and authored floor/forces'),
        ('evaluate', 'Reference floor audit: '+after_floor['status']),
        ('train', 'Learned one authored joint target; not whole-exercise imitation'),
        ('ready', 'Interactive candidate ready; physical reconstruction not verified')])]
    write_json(lab_root/'status.json', {'status': 'ready', 'events': events, 'result': result})
    lab = Lab(lab_root)
    try:
        metrics, interaction = probe_interaction(lab)
    finally:
        lab.close()
    recording = Path(interaction['recording']); events = load_recording(recording)['events']
    session = HumanSession(run)
    try:
        for event in events:
            if event['type'] in ('start', 'reset'):
                state = session.reset(event['seed'])
            else:
                reward, terminated, truncated, _ = session.step(event['action'])
                if (reward, terminated, truncated) != (event['reward'], event['terminated'], event['truncated']):
                    raise ValueError('Recorded reward or completion flags differ on exact replay')
                state = session.snapshot()
            if canonical_hash(state) != canonical_hash(event['snapshot']): raise ValueError('Recorded state differs on exact replay')
    finally:
        session.env.close()
    panels(directory, clip, before, after, report['accepted_frames'])
    if verify_artifacts(run) != run_verification or verify_artifacts(baseline) != baseline_verification:
        raise ValueError('Body inputs changed during interaction or comparison')
    summary = {'status': 'body_controls_recording_verified', 'clip': clip, 'body_constructed': True,
        'whole_body_poses': len(before['reference']), 'selected_body_poses': len(after['reference']),
        'unchanged_body_reused': unchanged, 'selected_joint': task['selected_joint'],
        'floor_before': before_floor['summary'], 'floor_after': after_floor['summary'],
        'floor_changes': floor_changes(before_floor, after_floor), 'dynamic_reference_tracking': dynamic,
        'learning': learning, 'interaction_metrics': metrics, 'interaction': interaction,
        'recording': verify_recording(recording), 'all_states_rewards_flags_replayed_exactly': True,
        'replayed_events': len(events), 'accepted_image_frames': report['accepted_frames'],
        'crop_frames_without_fixed_image_labels': [frame for frame in report['accepted_frames'] if frame != 40 or clip != 'jumping-jacks-burpees'],
        'wall_seconds': time.monotonic()-started, 'paid_requests': 0, 'physical_validation': 'unverified'}
    write_json(directory/'report.json', summary); seal_suite(directory)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', choices=CLIPS); args = parser.parse_args(); out = args.out.resolve()
    if args.worker:
        build_case(out, args.worker); return
    if out.is_relative_to(SOURCE): raise ValueError('Use a fresh output outside the frozen pose study')
    verification = verify_artifacts(SOURCE)
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    out.mkdir(parents=True, exist_ok=False)
    for path in (Path(__file__), Path(__file__).with_name('check_body_refinement.py'), Path(__file__).with_name('check_crop_diversity.py')):
        shutil.copyfile(path, out/path.name)
    write_json(out/'pre-run.json', {'created_at': local_time(), 'source_verification': verification,
        'implementation': implementation, 'labels_sha256': digest(LIMB_LABELS), 'maximum_workers': 2,
        'process_timeout_seconds': 180, 'maximum_body_models': 6, 'maximum_training_episodes': 3*EPISODES,
        'evaluation_trials_per_policy_per_body': EVALUATIONS, 'training_seed': SEED,
        'paid_requests': 0, 'new_model_inference': False,
        'scope': 'All quality-accepted full-video candidates plus explicit rejected cases; no floor/label-based selection'})
    started = time.monotonic()
    def worker(clip):
        with (out/(clip+'.log')).open('x') as log:
            result = subprocess.run([sys.executable, str(Path(__file__)), '--out', str(out), '--worker', clip],
                cwd=ROOT, env=os.environ | {'MPLCONFIGDIR': str(ROOT/'.local/matplotlib')},
                stdout=log, stderr=subprocess.STDOUT, timeout=180)
        return clip, result.returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = dict(pool.map(worker, CLIPS))
    write_json(out/'worker-results.json', outcomes)
    if any(outcomes.values()):
        write_json(out/'report.json', {'status': 'incomplete_workers', 'outcomes': outcomes, 'paid_requests': 0})
        seal_suite(out)
        raise RuntimeError('Preserved incomplete body work; reuse completed construction and training')
    reports = {clip: read(out/clip/'report.json') for clip in CLIPS}
    write_json(out/'construction.json', {'status': 'all_bodies_frozen_before_label_comparison',
        'created_at': local_time(), 'cases': {clip: verify_artifacts(out/clip) for clip in CLIPS}})
    labels = load_annotations(LIMB_LABELS, project_root=ROOT)
    if digest(LIMB_LABELS) != read(out/'pre-run.json')['labels_sha256']: raise ValueError('Comparison labels changed')
    scores = {}
    for mode in ('whole_body', 'selected_lab/run'):
        specs = {clip: read(out/clip/mode/'human.json') for clip in CLIPS if reports[clip]['body_constructed']}
        predictions, matching = limb_predictions(specs, labels)
        key = 'before' if mode == 'whole_body' else 'after'
        write_json(out/(key+'-predictions.json'), {'predictions': predictions, 'matching': matching,
            'projection': 'Simulation x and -z, no camera or label fit; unchanged 1/23-second nearest match'})
        scores[key] = grade_limb_projection(labels, predictions)
    changes = deltas(scores['before']['observations'], scores['after']['observations'], 'within_tolerance')
    write_json(out/'body-label-comparison.json', {'scores': scores, 'changes': changes})
    if verify_artifacts(SOURCE) != verification or implementation != {name: digest(path) for name, path in _implementation_files().items()}:
        raise ValueError('Frozen pose input or runtime changed')
    result = {'status': 'complete_body_candidate_comparison', 'created_at': local_time(), 'cases': reports,
        'body_label_summaries': {name: group['summary'] for name, group in scores.items()},
        'body_label_changes': {key: value for key, value in changes.items() if key != 'paired_rows'},
        'wall_seconds': time.monotonic()-started, 'paid_requests': 0, 'default_changed': False,
        'scope': 'Playable inferred candidates and exact recording; physical accuracy/contact remain unverified'}
    write_json(out/'report.json', result)
    rows = ['<!doctype html><meta charset="utf-8"><title>Exercise candidate comparison</title>',
        '<style>body{font:16px system-ui;max-width:1440px;margin:2rem auto;padding:0 1rem;background:#F5EADC;color:#214F50}img{width:100%;height:auto}pre{overflow:auto}section{margin:3rem 0}a{color:#965644}</style>',
        '<h1>Exercise candidate comparison</h1><p>Interactive bodies with illustrative muscles. Pose, anatomy, ground contact and forces are estimates or assumptions, not verified physical measurements.</p>']
    for clip, report in reports.items():
        rows.append('<section><h2>'+html.escape(clip)+'</h2>')
        if not report['body_constructed']:
            rows.append('<p>Not built: '+html.escape('; '.join(report['selected_errors']))+'</p></section>'); continue
        command = f'.venv-lab/bin/python -m weft lab --resume {out/clip/"selected_lab"}'
        relative_recording = Path(report['interaction']['recording']).relative_to(out)
        rows.append('<p>Run locally:</p><pre>'+html.escape(command)+'</pre>')
        rows.append(f'<p><a href="{relative_recording}/replay.html">Recorded interaction</a> · <a href="{relative_recording}/video.mp4">Recorded video</a></p>')
        rows.append('<p>Floor checks failing: '+str(report['floor_after']['failed_source_poses'])+'/'+str(report['floor_after']['source_pose_checks'])+' source poses. Physical accuracy remains unverified.</p>')
        for frame in report['accepted_image_frames']:
            rows.append(f'<img loading="lazy" src="{clip}/comparison-{frame:04d}.png" alt="Source and body estimates for frame {frame}">')
        rows.append('</section>')
    (out/'index.html').write_text('\n'.join(rows))
    print(json.dumps({'verification': seal_suite(out), 'body_label_summaries': result['body_label_summaries'],
        'body_label_changes': result['body_label_changes'], 'cases': {clip: {'status': row['status'],
        'selected_body_poses': row.get('selected_body_poses'), 'floor': row.get('floor_after'),
        'replayed_events': row.get('replayed_events')} for clip, row in reports.items()},
        'wall_seconds': result['wall_seconds']}, indent=2))


if __name__ == '__main__': main()
