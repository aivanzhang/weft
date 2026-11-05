"""Bounded, read-only presentation of saved simulation evidence."""
import hashlib
import json
import math
import re
from pathlib import Path

PUBLIC_FILES = frozenset({
    'rolling-motion.json', 'waiters-squat-panda.json', 'waiters-squat-created.json',
    'body-model.json', 'character-model.json', 'rolling-model.json', 'pose-sheet.png',
    'body-evidence.json', 'body-construction.json', 'body-task.json', 'body-outcome.json',
    'character-outcome.json', 'character-appearance.json', 'rolling-evidence.json',
}) | frozenset(
    identity + suffix + '.json'
    for identity in ('bodyweight-squats', 'incline-press', 'deadlift', 'jumping-jacks',
                     'overhead-squat', 'waiters-squat', 'sots-press', 'lunge-twist', 'single-leg-squat')
    for suffix in ('', '-model', '-evidence')
)

PUBLIC_FILES = PUBLIC_FILES | frozenset(
    identity + '-' + kind + '-poster.jpg'
    for identity in ('rolling-motion', 'bodyweight-squats', 'incline-press', 'deadlift', 'jumping-jacks',
                     'overhead-squat', 'waiters-squat', 'sots-press', 'lunge-twist', 'single-leg-squat')
    for kind in ('source', 'result')
)


def public_file(project, name):
    if name not in PUBLIC_FILES:
        raise ValueError('Unknown studio asset')
    base = Path(project).resolve() / 'assets/demo/studio'
    path = base / name
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError('Studio asset unavailable')
    return path


def recorded_model(path):
    """Export segment endpoints, never executable code or a new physical claim."""
    if path.stat().st_size > 32_000_000:
        raise ValueError('Recording exceeds preview limit')
    frames = []
    segment_ids = None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for index in range(1802):
            raw = stream.readline(262145)
            if not raw:
                break
            if index == 1801 or len(raw) > 262144:
                raise ValueError('Recording exceeds preview limit')
            digest.update(raw)
            event = json.loads(raw)
            snapshot = event.get('snapshot', {})
            state = snapshot.get('engine_state', snapshot)
            segments = state.get('human', {}).get('segments', [])
            if not segments:
                continue
            if len(segments) > 128:
                raise ValueError('Invalid recorded body')
            ids = [segment.get('id') for segment in segments]
            if segment_ids is None:
                segment_ids = ids
            elif ids != segment_ids:
                raise ValueError('Recorded body segment order changed')
            lines = []
            for segment in segments:
                points = [segment['a'], segment['b']]
                if any(len(p) != 3 or any(not isinstance(v, (float, int)) or not math.isfinite(v)
                                          or abs(v) > 10000 for v in p) for p in points):
                    raise ValueError('Invalid recorded body')
                lines.append([[round(v, 5) for v in p] for p in points])
            frames.append({'time': round(float(state.get('time_s', 0)), 5), 'segments': lines})
    if not frames:
        raise ValueError('No recorded skeleton available')
    stride = max(1, math.ceil(len(frames) / 360))
    selected = frames[::stride]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])
    return {'kind': 'skeleton', 'frames': selected, 'segment_ids': segment_ids if all(isinstance(v, str) for v in segment_ids) else [], 'source_sha256': digest.hexdigest(),
            'source_frames': len(frames), 'preview_stride': stride,
            'scope': 'Recorded simulated skeleton. Camera and playback controls do not change the physics.'}


def snapshot_model(states):
    """Bound the presentation copy without changing the source frames."""
    if not states or len(states) > 2000:
        raise ValueError('Unsupported preview frame count')
    indices = list(range(0, len(states), max(1, math.ceil(len(states) / 360))))
    if indices[-1] != len(states) - 1:
        indices.append(len(states) - 1)
    frames, ids = [], None
    for index in indices:
        state = states[index].get('engine_state', states[index])
        segments = state.get('human', {}).get('segments', [])
        if not 1 <= len(segments) <= 128:
            raise ValueError('No bounded skeleton in preview')
        current_ids = [segment.get('id') for segment in segments]
        if ids is None:
            ids = current_ids
        if current_ids != ids:
            raise ValueError('Preview segment order changed')
        lines = [[segment['a'], segment['b']] for segment in segments]
        if any(len(p) != 3 or any(not isinstance(v, (float, int)) or not math.isfinite(v)
                                  or abs(v) > 10000 for v in p) for line in lines for p in line):
            raise ValueError('Invalid preview coordinates')
        moment = float(state.get('time_s', 0))
        if not math.isfinite(moment):
            raise ValueError('Invalid preview time')
        frames.append({'time': moment, 'segments': lines})
    return {'kind': 'skeleton', 'frames': frames,
            'segment_ids': ids if all(isinstance(value, str) for value in ids) else [],
            'scope': 'Saved model state; not measured anatomy or physical validation.'}


def visual_file(root, project, relative, kind, label):
    """Only existing files inside the exact project can enter a stage preview."""
    if not isinstance(project, str) or not re.fullmatch(r'[a-f0-9]{32}', project):
        return []
    base = Path(root).resolve() / project
    path = base / relative
    if (path.resolve() != path or not path.is_relative_to(base) or not path.is_file()
            or path.stat().st_size > 100_000_000):
        return []
    return [{'kind': kind, 'url': f'/api/projects/{project}/files/{relative}', 'label': label}]


def save_reference_preview(root, project):
    if not isinstance(project, str) or not re.fullmatch(r"[a-f0-9]{32}", project):
        return []
    source = Path(root).resolve() / project / 'lab/run/human-reference.json'
    if source.resolve() != source or not source.is_file() or source.stat().st_size > 32_000_000:
        return []
    data = json.loads(source.read_text())
    model = snapshot_model(data.get('frames', []))
    model['scope'] = 'Reconstructed kinematic reference, not a controller rollout.'
    target = source.with_name('studio-reference-preview.json')
    if target.is_symlink():
        return []
    target.write_text(json.dumps(model, separators=(',', ':')) + '\n')
    return visual_file(root, project, 'lab/run/studio-reference-preview.json', 'model',
                       'Reconstructed body · saved reference movement')


def recording_visuals(root, record):
    """The physics and appearance renders share a session, never a video URL."""
    project = record.get('project')
    session = Path(record.get('video') or '').parent
    base = Path(root).resolve() / str(project) / 'lab/sessions'
    if session.resolve() != session or session.parent != base:
        return {}
    return {
        'rollout': visual_file(root, project, f'lab/sessions/{session.name}/video.mp4', 'video',
                              'Physics replay · simulated body and controller motion'),
        'avatar': visual_file(root, project, f'lab/sessions/{session.name}/avatar-video.mp4', 'video',
                             'Character rendering · the same session with appearance changes'),
    }
