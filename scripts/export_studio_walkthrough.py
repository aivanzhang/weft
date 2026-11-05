"""Export allowlisted studio exhibits from existing evidence, without rerunning a model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from weft.portal_studio import recorded_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Choose an empty output directory')
    lab = root / '.local/verify-video-01/portal/9eadcc9655864070b0039fa3feb93fbc/lab'
    run = lab / 'run'
    def read(path):
        return json.loads(path.read_text())
    def write(name, data):
        (out / name).write_text(json.dumps(data, separators=(',', ':'), ensure_ascii=False) + '\n')
    def excerpt(name, source, keys):
        data = read(source)
        write(name, {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'presentation': 'Selected fields from saved evidence; not a new evaluation.',
                     **{key: data[key] for key in keys if key in data}})
    def asset(name, label, kind='json'):
        return {'name': name, 'label': label, 'kind': kind}
    excerpt('body-evidence.json', run / 'evidence.json',
            ['profile', 'source_sha256', 'model', 'sample_rate_fps', 'frame_count', 'quality', 'scope', 'assumptions'])
    excerpt('body-construction.json', run / 'reconstruction.json',
            ['profile', 'simulation_ready', 'physical_validation', 'scope', 'reference_interval', 'readiness'])
    excerpt('body-task.json', run / 'exercise-task.json',
            ['profile', 'gate_profile', 'gates_sha256', 'interval', 'control', 'source_scope'])
    sheet = lab.parent / 'saved-body/world/tracks-contact-sheet.png'
    shutil.copyfile(sheet, out / 'pose-sheet.png')
    for prefix, session in [('body', 'session-9f8a60b20993'), ('character', 'session-1de315688ae1')]:
        folder = lab / 'sessions' / session
        write(prefix + '-model.json', recorded_model(folder / 'events.jsonl'))
        events = (folder / 'events.jsonl').read_text().splitlines()
        final = json.loads(events[-1])['snapshot']['task_metrics']
        write(prefix + '-outcome.json', {'metrics': final, 'source_sha256': hashlib.sha256((folder / 'events.jsonl').read_bytes()).hexdigest(),
                                       'scope': 'Saved simulated task outcome, not physical validation.'})
    excerpt('character-appearance.json', lab / 'sessions/session-1de315688ae1/appearance.json',
            ['profile', 'changes', 'status', 'scope', 'display_scope', 'events_sha256', 'appearance_events_sha256'])
    common = [
        {'id': 'intake', 'title': 'Start with what the camera saw.', 'description': 'Track the person through the clip. Keep the joint estimates, confidence and gaps so the reconstruction can be checked against its source.',
         'assets': [asset('pose-sheet.png', 'Tracked movement', 'image'), asset('body-evidence.json', 'Pose evidence')]},
        {'id': 'world', 'title': 'Give the movement a body.', 'description': 'Build an articulated skeleton with joint limits and a ground plane. The result can run in a physics engine; depth and contact remain model assumptions.',
         'assets': [asset('body-model.json', 'Recorded skeleton'), asset('body-construction.json', 'Construction report')]},
        {'id': 'task', 'title': 'Decide what a good result means.', 'description': 'Attach a tracking task before running the controller. Fixed scoring rules make it possible to compare results without moving the goalposts.',
         'assets': [asset('body-task.json', 'Task and scoring contract')]},
        {'id': 'rollout', 'title': 'Run it. Keep the result.', 'description': 'An assisted reference controller follows the observed motion. Save the outcome and the recording, including the checks that passed or failed. This is programmed control, not a learned policy.',
         'assets': [asset('body-outcome.json', 'Recorded task results')]},
    ]
    for identity in ['waiters-squat-panda', 'waiters-squat-created']:
        steps = json.loads(json.dumps(common))
        character = identity.endswith('created')
        if character:
            steps[1]['assets'][0] = asset('character-model.json', 'Recorded skeleton')
            steps[3]['assets'][0] = asset('character-outcome.json', 'Recorded task results')
            steps.append({'id': 'avatar', 'title': 'Change the character, not the experiment.',
                          'description': 'A generated panda takes the place of the robot during the recording. The simulated body, controller and score continue unchanged. The video above is the excerpt after that swap.',
                          'assets': [asset('character-appearance.json', 'Appearance-change record')]})
        write(identity + '.json', {'model': 'character-model.json' if character else 'body-model.json',
                                   'model_label': 'Reconstructed skeleton', 'steps': steps})
    catalog = json.loads((root / 'website/environments/catalog.js').read_text().split(' = ', 1)[1].rstrip(';\n'))[0]
    segment = next(s for s in catalog['segments'] if s['primary'])
    write('rolling-model.json', {'kind': 'rolling', 'world': catalog['models'][0]['world'], 'segment': segment,
                                 'scope': 'Existing one-dimensional fit. Push controls are authored extensions, not recovered actions.'})
    write('rolling-evidence.json', {'source_manifest_sha256': catalog['manifest_sha256'],
                                    'models': catalog['models'], 'segments': catalog['segments']})
    write('rolling-motion.json', {'model': 'rolling-model.json', 'model_label': 'Reconstructed motion', 'steps': [
        {'id': 'intake', 'title': 'Follow the ball.', 'description': 'Keep the measured passes and their timestamps. The initial input above shows the same pass as the result, both at quarter speed.', 'assets': []},
        {'id': 'world', 'title': 'Fit the simplest useful model.', 'description': 'Estimate position, speed and drag along the table. This is a one-dimensional motion model, not reconstructed 3D geometry.', 'assets': [asset('rolling-evidence.json', 'Measured passes and model stages')]},
        {'id': 'rollout', 'title': 'Make the model executable.', 'description': 'Replay the fitted motion or try an authored push. The recording above uses the existing fit; this presentation does not run a new evaluation.', 'assets': [asset('rolling-model.json', 'Executable model parameters')]},
    ]})
    print(json.dumps({'files': len(list(out.iterdir())), 'bytes': sum(p.stat().st_size for p in out.iterdir())}))


if __name__ == '__main__':
    main()
