"""Build and record the admitted Heavy squat while comparing unchanged body checks."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image, ImageDraw

from weft.benchmark import probe_interaction, seal_suite
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
from check_crop_diversity import LIMB_LABELS, deltas
from check_selective_crop_bodies import floor_changes


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/heavy-exercise-comparison-01/squat'
BASELINE = ROOT / 'artifacts/selective-crop-bodies-01/squat/selected_lab/run'
VIDEO = ROOT / 'data/exercise/squat.webm'
INSPECT = [0, 50, 100, 150, 200]


def read(path):
    return json.loads(path.read_text())


def save_panels(out, before, after):
    envs = [HumanEnv(spec) for spec in (before, after)]
    lookup = [{int(row['evidence_id'].rsplit(':', 1)[1]): index for index, row in enumerate(spec['reference_evidence'])}
              for spec in (before, after)]
    cap = cv2.VideoCapture(str(VIDEO))
    try:
        for frame in range(213):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError('Source preview decoding ended early')
            if frame not in INSPECT:
                continue
            tiles = [Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]
            for env, indices in zip(envs, lookup):
                if frame in indices:
                    tile = Image.fromarray(render_snapshot(env.reference_snapshot(indices[frame]), env.scene, width=480))
                else:
                    tile = Image.new('RGB', (480, 360), '#F5EADC')
                    ImageDraw.Draw(tile).text((20, 20), 'No usable body pose at this frame', fill='#214F50')
                tiles.append(tile)
            panel = Image.new('RGB', (1440, 430), '#F5EADC')
            draw = ImageDraw.Draw(panel)
            for index, (label, tile) in enumerate(zip(('Source', 'Full-model body', 'Heavy-model body'), tiles)):
                draw.text((index * 480 + 10, 10), f'{label} | frame {frame}', fill='#214F50')
                tile.thumbnail((480, 360)); panel.paste(tile, (index * 480, 40))
            draw.text((10, 410), 'Inferred bodies and authored physics. Muscle bands are illustrative.', fill='#214F50')
            panel.save(out / f'comparison-{frame:04d}.png')
        if cap.read()[0]:
            raise ValueError('Source preview exceeds the frozen interval')
    finally:
        cap.release()
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if any(out.is_relative_to(path) for path in (SOURCE, BASELINE)):
        raise ValueError('Use a fresh output outside saved runs')
    verified = {str(path): verify_artifacts(path) for path in (SOURCE, BASELINE)}
    evidence = read(SOURCE / 'evidence.json')
    if not evidence['quality']['passes'] or read(SOURCE / 'root-motion.json')['status'] != 'ready':
        raise ValueError('Retain all perception and root admission gates')
    if digest(VIDEO) != evidence['source_sha256']:
        raise ValueError('Original source bytes changed')
    implementation = {name: digest(path) for name, path in _implementation_files().items()}
    label_hash = digest(LIMB_LABELS)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('build_heavy_squat_lab.py', 'check_selective_crop_bodies.py', 'check_body_refinement.py', 'check_crop_diversity.py'):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    write_json(out / 'pre-run.json', {'created_at': local_time(), 'source_verifications': verified,
        'implementation': implementation, 'labels_sha256': label_hash, 'inspection_frames': INSPECT,
        'maximum_training_episodes': 100, 'evaluation_trials_per_policy': 20, 'seed': 7311,
        'new_inference_calls': 0, 'paid_requests': 0,
        'scope': 'Only Heavy squat passes both source gates; no new inference or model selection'})
    started = time.monotonic()
    lab_root = out / 'lab'; run = lab_root / 'run'; run.mkdir(parents=True)
    for name, expected in read(SOURCE / 'manifest.json')['files'].items():
        destination = run / ('input-pose-report.json' if name == 'report.json' else name)
        shutil.copyfile(SOURCE / name, destination)
        if digest(destination) != expected:
            raise ValueError('Copied pose input changed')
    shutil.copyfile(SOURCE / 'manifest.json', run / 'input-pose-manifest.json')
    shutil.copyfile(VIDEO, run / 'source.webm')
    source = describe_source(evidence['source_sha256'])
    write_json(run / 'source.json', source | {'artifact': 'source.webm', 'sha256': evidence['source_sha256']})
    spec = build_human(evidence, run, body_model='mobile')
    floor = audit_reference_feasibility(spec)
    write_json(run / 'reference-feasibility.json', floor)
    task = propose_joint_task(evidence, spec)
    write_json(run / 'task-selection.json', task)
    settings = {'schema_version': 1, 'profile': 'exercise', 'video': str(VIDEO), 'body_model': 'mobile',
        'pose_mode': 'image', 'pose_identity': 'raw', 'pose_model': str(ROOT / '.local/models/pose_landmarker_heavy.task'),
        'episodes': 100, 'eval_episodes': 20, 'seed': 7311, 'created_at': local_time()}
    write_json(run / 'run-config.json', settings); write_json(run / 'model.json', evidence['model'])
    env = HumanEnv(spec)
    try:
        env.select_joint(task['selected_joint']); env.reset(seed=0, options={'jitter': False})
        env.scene['joint_names'] = list(env.joint_names)
        write_json(run / 'scene.json', env.scene); write_json(run / 'task.json', env.scene['task'])
        references = [env.reference_snapshot(i) for i in range(len(spec['reference']))]
        write_json(run / 'human-reference.json', {'source_indices': list(range(len(references))), 'frames': references,
            'scope': 'Kinematic inspection, not learned execution or measured anatomy'})
        images = [Image.fromarray(render_snapshot(state, env.scene, width=480))
            for state in references[::max(1, math.ceil(len(references) / 32))]]
        images[0].save(run / 'reconstruction.gif', save_all=True, append_images=images[1:], duration=150, loop=0)
    finally:
        env.close()
    dynamics = evaluate_reference(spec, run)
    learning = train_human(spec, run, episodes=100, eval_episodes=20, seed=7311, selected_joint=task['selected_joint'])
    result = {'profile': 'exercise', 'status': 'executable_unpromoted_candidate', 'source': source,
        'learning': learning, 'task_selection': task, 'pose_quality': evidence['quality'],
        'reference_frames': len(spec['reference']), 'assumptions': spec['assumptions'],
        'reference_tracking': dynamics, 'reference_feasibility': {'status': floor['status'], 'summary': floor['summary']},
        'physical_validation': 'unverified', 'limits': ['Single-joint learning, not the full exercise',
            'Image/body estimates and authored floor, scale, masses and support', 'Illustrative muscles, not measured activation']}
    write_json(run / 'report.json', result)
    write_json(run / 'reconstruction.json', {'scope': 'Inferred mobile body; inspect unchanged floor and source comparisons',
        'reference_frames': len(spec['reference']), 'physical_validation': 'unverified'})
    write_json(run / 'repair-ledger.json', {'proposals': [], 'scope': 'No physical repair or model promotion'})
    for name, path in _implementation_files().items():
        target = run / 'implementation-source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_json(run / 'implementation.json', implementation)
    run_verified = seal_suite(run)
    write_json(lab_root / 'lab-config.json', settings)
    events = [{'index': i, 'time': local_time(), 'stage': stage, 'message': message} for i, (stage, message) in enumerate([
        ('observe', 'Reused complete source-pinned Heavy IMAGE estimates; missing poses retained'),
        ('world', 'Built a mobile body with authored physics and illustrative muscles'),
        ('evaluate', 'Floor checks: ' + floor['status']),
        ('train', 'Learned one joint-target task, not whole-exercise behavior'),
        ('ready', 'Candidate ready to inspect, control and record; physical fidelity not established')])]
    write_json(lab_root / 'status.json', {'status': 'ready', 'events': events, 'result': result})
    lab = Lab(lab_root)
    try:
        metrics, interaction = probe_interaction(lab)
    finally:
        lab.close()
    recording = Path(interaction['recording'])
    recorded = load_recording(recording)['events']
    session = HumanSession(run)
    try:
        for event in recorded:
            if event['type'] in ('start', 'reset'):
                state = session.reset(event['seed'])
            else:
                reward, terminated, truncated, _ = session.step(event['action'])
                if (reward, terminated, truncated) != (event['reward'], event['terminated'], event['truncated']):
                    raise ValueError('Replay changed reward or completion flags')
                state = session.snapshot()
            if canonical_hash(state) != canonical_hash(event['snapshot']):
                raise ValueError('Recorded state did not replay exactly')
    finally:
        session.env.close()
    before = read(BASELINE / 'human.json')
    before_floor = read(BASELINE / 'reference-feasibility.json')
    if digest(LIMB_LABELS) != label_hash:
        raise ValueError('Fixed comparison labels changed')
    labels = load_annotations(LIMB_LABELS, project_root=ROOT)
    labels['frames'] = [row for row in labels['frames'] if row['clip_id'] == 'squat']
    scores = {}
    for mode, model in (('full', before), ('heavy', spec)):
        predictions, matching = limb_predictions({'squat': model}, labels)
        scores[mode] = grade_limb_projection(labels, predictions)
        write_json(out / (mode + '-body-predictions.json'), {'predictions': predictions, 'matching': matching})
    changes = deltas(scores['full']['observations'], scores['heavy']['observations'], 'within_tolerance')
    write_json(out / 'body-scores.json', {'scores': scores, 'changes': changes})
    save_panels(out, before, spec)
    if (verify_artifacts(run) != run_verified or any(verify_artifacts(Path(path)) != value for path, value in verified.items())
            or implementation != {name: digest(path) for name, path in _implementation_files().items()}):
        raise ValueError('Frozen body input or runtime changed')
    report = {'status': 'heavy_squat_body_recording_compared', 'created_at': local_time(),
        'body_poses': {'full': len(before['reference']), 'heavy': len(spec['reference'])},
        'floor': {'full': before_floor['summary'], 'heavy': floor['summary'], 'changes': floor_changes(before_floor, floor)},
        'body_scores': {mode: score['summary'] for mode, score in scores.items()},
        'body_score_changes': {key: value for key, value in changes.items() if key != 'paired_rows'},
        'task_joint': task['selected_joint'], 'learning': learning, 'dynamic_reference': dynamics,
        'interaction_metrics': metrics, 'interaction': interaction, 'recording': verify_recording(recording),
        'replayed_events': len(recorded), 'all_states_rewards_flags_replayed_exactly': True,
        'wall_seconds': time.monotonic() - started, 'paid_requests': 0, 'new_inference_calls': 0,
        'physical_accuracy_verified': False,
        'launch': '.venv-lab/bin/python -m weft lab --resume ' + str(lab_root.relative_to(ROOT))}
    write_json(out / 'report.json', report)
    print(json.dumps({'report': report, 'verification': seal_suite(out)}, indent=2))


if __name__ == '__main__':
    main()
