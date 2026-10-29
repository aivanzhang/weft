"""Explicit terminal actions for sealed planar hypotheses, with portable recordings."""

import json
from hashlib import sha256
from pathlib import Path

from PIL import Image

from weft.io import canonical_hash, write_json
from weft.pipeline import verify_artifacts
from weft.recording import Recorder, export_recording, render_snapshot
from weft.scene import ACTIONS, SceneEnv


HELP = ('Commands: coast/left/right/up/down/brake [1–6], reset, status, help, quit. '
        'The optional number is simulation steps, not an action number. Up decreases image y. '
        'Nothing advances until you enter a movement command; an ended task requires reset.')
SPATIAL_HELP = ('Commands: coast/left/right/forward/back/up/down/brake [1–6], reset, status, help, quit. '
                'Left/right move x, forward/back move depth, and up/down apply vertical forces. '
                'Forces are bounded; up is not automatic flight. Nothing advances while you type.')


def command_action(command, *, actions=ACTIONS, help_text=HELP):
    parts = command.split()
    if len(command) > 256 or len(parts) not in (1, 2) or parts[0] not in actions:
        raise ValueError(help_text)
    try:
        count = int(parts[1]) if len(parts) == 2 else 1
    except ValueError:
        raise ValueError('Use an integer from 1 to 6 steps per command') from None
    if not 1 <= count <= 6:
        raise ValueError('Use 1–6 steps per command')
    return actions.index(parts[0]), count


def _load_scene(run):
    manifest_path = run / 'manifest.json'
    if manifest_path.is_symlink() or manifest_path.stat().st_size > 4_000_000:
        raise ValueError('Scene console requires a bounded sealed manifest')
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files') if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or 'scene.json' not in files or len(files) > 4096:
        raise ValueError('scene.json must belong to the sealed run')
    verification = verify_artifacts(run)

    def read(name):
        path = run / name
        if name not in files or path.is_symlink() or not path.resolve().is_relative_to(run) or path.stat().st_size > 64_000_000:
            raise ValueError('Console input must be bounded and sealed: ' + name)
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != files[name]:
            raise ValueError('Sealed console input changed after verification: ' + name)
        return json.loads(raw)

    scene = read('scene.json')
    has_report, has_candidate = (run / 'report.json').exists(), (run / 'candidate.json').exists()
    report = read('report.json') if has_report else None
    candidate = read('candidate.json') if has_candidate else None
    if has_report and not isinstance(report, dict):
        raise ValueError('Export report must be an object')
    if has_candidate:
        if (not isinstance(report, dict) or report.get('status') != 'completed_diagnostic_replay'
                or not isinstance(candidate, dict) or candidate.get('status') != 'executable_hypothesis'
                or candidate.get('profile') != 'segmentation-planar-candidate-v1'
                or any(item.get('physical_validation') != 'unverified' or item.get('physics_ready') is not False
                       for item in (candidate, report))):
            raise ValueError('Candidate export must be executable, completed, and explicitly physically unverified')
        if report.get('candidate_sha256') != canonical_hash(candidate) or candidate.get('scene') != scene:
            raise ValueError('Candidate, scene, and export report identities differ')
        label = 'Segmentation-derived executable hypothesis. Physical validation: unverified. Controls, forces and task are authored.'
    else:
        if isinstance(report, dict) and ('candidate_sha256' in report or 'assumptions_sha256' in report):
            raise ValueError('Candidate export is missing its sealed candidate.json')
        label = 'Authored planar simulation. Sealed files verify integrity, not physical accuracy; physical validation is unverified.'
        if scene.get('profile') == 'spatial-rigid-v1':
            label = '3D rigid-object hypothesis. Depth, materials, controls and task are assumed; physical validation is unverified.'
    return scene, label, verification


def console(run, recording, *, seed=1729, input_fn=input, output_fn=print):
    run, recording = Path(run).resolve(), Path(recording).resolve()
    if recording.is_relative_to(run):
        raise ValueError('Keep recordings outside the immutable scene run')
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError('Seed must be an integer between 0 and 2**32−1')
    if recording.exists():
        raise FileExistsError('Use a new recording directory; existing recordings are preserved')
    scene, label, verification = _load_scene(run)
    if scene.get('profile') == 'spatial-rigid-v1':
        from weft.spatial_rigid import ACTIONS as spatial_actions, SpatialRigidEnv
        env, actions, help_text = SpatialRigidEnv(scene), spatial_actions, SPATIAL_HELP
    else:
        env, actions, help_text = SceneEnv(scene), ACTIONS, HELP
    recorder = Recorder(recording, env.scene, profile='spatial-rigid' if scene.get('profile') == 'spatial-rigid-v1' else 'planar')
    done, failure = False, None

    def preview():
        state = env.snapshot()
        Image.fromarray(render_snapshot(state, env.scene, width=640)).save(recording / 'current-frame.png')
        controlled = next(item for item in state['objects'] if item['id'] == state['controlled_id'])
        output_fn(json.dumps({'step': state['steps'], 'time_s': state['time_s'],
                             'controlled_id': state['controlled_id'], 'position': controlled['position'],
                             'velocity': controlled['velocity'], 'target': state['target'],
                             'success': state['success'], 'reset_required': done,
                             'contacts': state['contacts']}, allow_nan=False))

    try:
        env.reset(seed=seed, options={'jitter': False})
        recorder.start(env.snapshot(), seed=seed)
        write_json(recording / 'console-input.json', {'run_manifest_sha256': verification['manifest_sha256'],
                   'scene_sha256': canonical_hash(scene), 'reset_options': {'jitter': False}, 'scope': label})
        output_fn(label)
        output_fn(help_text)
        output_fn('Current image: ' + str(recording / 'current-frame.png'))
        preview()
        while True:
            try:
                command = input_fn('scene> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if command == 'quit':
                break
            if command == 'help':
                output_fn(help_text)
                continue
            if command == 'status':
                preview()
                continue
            if command != 'reset':
                try:
                    action, count = command_action(command, actions=actions, help_text=help_text)
                    if done:
                        raise ValueError('Task ended; reset before another action')
                except ValueError as error:
                    output_fn(str(error))
                    continue
            try:
                recorder.check_capacity()
                if command == 'reset':
                    env.reset(seed=seed, options={'jitter': False})
                    recorder.append({'type': 'reset', 'seed': seed, 'snapshot': env.snapshot()})
                    done = False
                else:
                    for _ in range(count):
                        recorder.check_capacity()
                        _, reward, terminated, truncated, _ = env.step(action)
                        recorder.append({'type': 'step', 'action': action, 'reward': float(reward),
                                         'terminated': bool(terminated), 'truncated': bool(truncated),
                                         'snapshot': env.snapshot()})
                        done = terminated or truncated
                        if done:
                            break
                preview()
            except (RuntimeError, ValueError) as error:
                failure = {'reason': str(error), 'command': command,
                           'completed_recorded_steps': recorder.summary()['steps'],
                           'scope': 'Stopped at a recording limit or failed action; no terminal state fabricated'}
                output_fn(json.dumps(failure))
                break
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()
    if failure:
        metadata = json.loads((recording / 'recording.json').read_text())
        write_json(recording / 'recording.json', metadata | {'failed_attempt': failure})
    result = export_recording(recording)
    output_fn('Saved replay: ' + result['replay'])
    output_fn('Saved video: ' + result['video'])
    return result | {'failed_attempt': failure} if failure else result
