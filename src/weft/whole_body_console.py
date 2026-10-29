"""Terminal control with explicit forces, a saved journal and portable visual replay."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from weft.io import canonical_hash, write_json
from weft.pipeline import verify_artifacts
from weft.recording import Recorder, export_recording, render_snapshot
from weft.whole_body import WholeBodyEnv
from weft.whole_body_learning import policy_action


HELP = ('Commands: learned [1–6], follow [1–6], release [1–6], '
        'motor NAME VALUE [1–6], reset, status, help, quit. '
        'VALUE is −1 to 1; a single-motor command leaves every other motor at zero. '
        'Follow is authored feedback, not learned control. Completed steps are recorded; '
        'failed attempts are retained separately without inventing a final pose.')


def command_action(command, env, policy):
    parts = command.split()
    if not parts or parts[0] not in {'learned', 'follow', 'release', 'motor'}:
        raise ValueError(HELP)
    motor = parts[0] == 'motor'
    required = 3 if motor else 1
    if len(parts) not in (required, required + 1):
        raise ValueError(HELP)
    steps = int(parts[-1]) if len(parts) == required + 1 else 1
    if not 1 <= steps <= 6:
        raise ValueError('Use 1–6 steps per command')
    if parts[0] == 'learned':
        return lambda: policy_action(policy, env), steps
    if parts[0] == 'follow':
        return lambda: np.clip(env.feature_matrix().sum(axis=1), -1, 1), steps
    action = np.zeros(22)
    if motor:
        if parts[1] not in env.action_names:
            raise ValueError('Unknown motor; use: ' + ', '.join(env.action_names))
        value = float(parts[2])
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('Motor value must be finite and between −1 and 1')
        action[env.action_names.index(parts[1])] = value
    return lambda: action.copy(), steps


def console(run, recording, *, seed=1729, input_fn=input, output_fn=print):
    run, recording = Path(run).resolve(), Path(recording).resolve()
    if recording.is_relative_to(run):
        raise ValueError('Keep recordings outside the immutable learning run')
    verify_artifacts(run)
    spec = json.loads((run / 'human.json').read_text())
    policy = json.loads((run / 'policy.json').read_text())
    settings = json.loads((run / 'settings.json').read_text())
    if policy.get('spec_sha256') != canonical_hash(spec):
        raise ValueError('Policy belongs to a different body model')
    partition = settings['partition']
    env = WholeBodyEnv(spec, start_time_s=partition['evaluation_start_times_s'][0],
                       horizon_steps=partition['horizon_steps'])
    recorder = Recorder(recording, env.scene, profile='exercise')
    done, failure = False, None

    def preview():
        snapshot = env.snapshot()
        Image.fromarray(render_snapshot(snapshot, env.scene, width=640)).save(recording / 'current-frame.png')
        forces = snapshot['human']['motor_forces']
        output_fn(json.dumps({**env.metrics(), 'virtual_root_force_n': forces[-3:],
                              'maximum_joint_torque_nm': max(abs(v) for v in forces[:19])}, allow_nan=False))
    try:
        env.reset(seed=seed)
        policy_action(policy, env)
        recorder.start(env.snapshot(), seed=seed)
        output_fn(HELP)
        output_fn('Motors: ' + ', '.join(env.action_names))
        output_fn('Live frame (updated after each command): ' + str(recording / 'current-frame.png'))
        preview()
        while True:
            try:
                command = input_fn('body> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if command == 'quit':
                break
            if command == 'help':
                output_fn(HELP)
                continue
            if command == 'status':
                output_fn(json.dumps(env.metrics(), allow_nan=False))
                continue
            if command == 'reset':
                env.reset(seed=seed)
                recorder.append({'type': 'reset', 'seed': seed, 'snapshot': env.snapshot()})
                done = False
                preview()
                continue
            try:
                if done:
                    raise ValueError('Episode ended; reset before another action')
                choose, steps = command_action(command, env, policy)
                for _ in range(steps):
                    recorder.check_capacity()
                    action = choose()
                    try:
                        _, reward, terminated, truncated, _ = env.step(action)
                    except RuntimeError as error:
                        failure = {'reason': str(error), 'attempted_action': action.tolist(),
                                   'command': command, 'completed_recorded_steps': recorder.summary()['steps'],
                                   'scope': 'Failed action attempt; no valid terminal snapshot was fabricated.'}
                        break
                    recorder.append({'type': 'step', 'action': action.tolist(), 'reward': float(reward),
                                     'terminated': bool(terminated), 'truncated': bool(truncated),
                                     'snapshot': env.snapshot()})
                    done = terminated or truncated
                    if done:
                        break
                if failure:
                    output_fn(json.dumps(failure))
                    break
                preview()
            except ValueError as error:
                output_fn(str(error))
    finally:
        if recorder._started is not None:
            recorder.close()
        env.close()
    if failure:
        metadata = json.loads((recording / 'recording.json').read_text())
        write_json(recording / 'recording.json', metadata | {'failed_attempt': failure})
    result = export_recording(recording)
    return result | {'failed_attempt': failure} if failure else result
