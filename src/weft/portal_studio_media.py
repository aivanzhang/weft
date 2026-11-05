"""Prepare bounded presentation assets off the request path; never rerun the pipeline."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

from weft.portal_studio import recording_visuals, save_reference_preview, visual_file


def _json(path, limit=4_000_000):
    path = Path(path)
    if path.resolve() != path or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Presentation source is unavailable or exceeds its bound')
    return json.loads(path.read_text())


def poster(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() != source or not source.is_file() or source.stat().st_size > 100_000_000:
        return None
    if target.is_symlink():
        raise ValueError('Presentation output cannot be a symlink')
    if target.is_file():
        return target
    executable = shutil.which('ffmpeg')
    if not executable:
        from imageio_ffmpeg import get_ffmpeg_exe
        executable = get_ffmpeg_exe()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.poster-', dir=target.parent) as folder:
        temporary = Path(folder) / 'frame.jpg'
        subprocess.run([executable, '-nostdin', '-v', 'error', '-threads', '1', '-ss', '0.1',
                        '-protocol_whitelist', 'file,pipe', '-i', str(source), '-frames:v', '1',
                        '-vf', 'scale=960:-2', '-q:v', '3', '-threads', '1', str(temporary)],
                       check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if not temporary.is_file() or temporary.stat().st_size > 2_000_000:
            raise ValueError('Video poster was not produced within its bound')
        temporary.replace(target)
    return target


def _character_package(root, run_dir, record):
    from weft.avatar_assets import load_imported_avatar
    step = next((step for step in record.get('steps', []) if step['id'] == 'create'), {})
    metrics = step.get('metrics') or {}
    name = metrics.get('superimposed') or metrics.get('imported') or metrics.get('character')
    preset = record.get('character') or {}
    library_hash = None
    library = Path(__file__).resolve().parents[2] / '.local/avatar-library'
    for sidecar in sorted(library.glob('*.json'))[:16]:
        if _json(sidecar, 128_000).get('display_name') != name:
            continue
        delivery = sidecar.with_suffix('.zip')
        if delivery.resolve() != delivery or not delivery.is_file() or delivery.stat().st_size > 32_000_000:
            continue
        with zipfile.ZipFile(delivery) as bundle:
            if bundle.getinfo('model.glb').file_size <= 32_000_000:
                library_hash = hashlib.sha256(bundle.read('model.glb')).hexdigest()
        break
    archive = run_dir / 'artifacts/create.zip'
    catalog = None
    if archive.is_file() and not archive.is_symlink():
        with zipfile.ZipFile(archive) as bundle:
            if 'catalog.json' in bundle.namelist() and bundle.getinfo('catalog.json').file_size < 1_000_000:
                catalog = json.loads(bundle.read('catalog.json'))
    catalog = catalog or _json(root / 'avatars/catalog.json', 1_000_000)
    for entry in catalog.get('entries', []):
        folder = root / 'avatars' / entry['directory']
        if folder.resolve() != folder or not folder.is_relative_to(root / 'avatars'):
            continue
        original = _json(folder / 'originals/avatar.json', 128_000)
        imported = _json(folder / 'manifest.json', 128_000)
        matches = name and original.get('display_name') == name
        matches = matches or (library_hash and imported.get('files', {}).get('originals/model.glb') == library_hash)
        matches = matches or (preset.get('id') and f'/builtin/{preset["id"]}/' in folder.as_posix())
        if matches:
            return load_imported_avatar(folder), folder
    return None, None


def _motion_samples(path, fps=15, max_frames=240):
    """Bound replay input and sample the saved clock without running physics."""
    import math
    if path.resolve() != path or not path.is_file() or path.stat().st_size > 64_000_000:
        raise ValueError('Character recording exceeds its preview bound')
    digest, samples, origin, previous = hashlib.sha256(), [], None, -math.inf
    with path.open('rb') as stream:
        for index in range(2001):
            raw = stream.readline(262145)
            if not raw:
                break
            if index == 2000 or len(raw) > 262144:
                raise ValueError('Character recording exceeds its preview bound')
            digest.update(raw)
            snapshot = json.loads(raw).get('snapshot')
            if not snapshot:
                continue
            state = snapshot.get('engine_state', snapshot)
            moment = float(state.get('time_s', 0))
            if not math.isfinite(moment) or moment < previous:
                raise ValueError('Character recording has an invalid clock')
            previous = moment
            if origin is None:
                origin = moment
            while len(samples) < max_frames and moment + 1e-6 >= origin + len(samples) / fps:
                samples.append(snapshot)
    if len(samples) < 2:
        raise ValueError('No saved character movement is available')
    return samples, digest.hexdigest()


def character_preview(root, run_dir, record, session):
    """Render the selected asset on saved motion; leave the scored recording intact."""
    import time
    import numpy as np
    from PIL import Image
    from imageio_ffmpeg import write_frames
    from weft.avatar_offscreen import AvatarOffscreen
    package, _ = _character_package(root, run_dir, record)
    if package is None:
        return []
    samples, source_hash = _motion_samples(session / 'events.jsonl')
    key = package.source_manifest_sha256
    project = record['project']
    stem = f'studio-character-{key[:12]}-{source_hash[:12]}'
    output = root / project / 'lab/run'
    target, still, receipt = (output / (stem + suffix) for suffix in ('.mp4', '.jpg', '.json'))
    if any(path.is_symlink() for path in (target, still, receipt)):
        raise ValueError('Character preview cannot follow a symlink')
    if not all(path.is_file() for path in (target, still, receipt)):
        scene = _json(session / 'scene.json', 8_000_000)
        settings = {'asset': key, 'overlay': False, 'show_muscles': False,
                    'camera': {'yaw': .5, 'zoom': 1.5, 'follow': True}}
        deadline = time.monotonic() + 120
        with tempfile.TemporaryDirectory(prefix='.character-', dir=output) as folder:
            temporary = Path(folder) / 'motion.mp4'
            renderer = AvatarOffscreen(scene.get('engine_scene', scene), {key: package}, width=1440, height=560)
            writer = None
            try:
                writer = write_frames(str(temporary), (480, 560), fps=15, codec='libx264',
                                      pix_fmt_out='yuv420p', macro_block_size=1,
                                      output_params=['-threads', '1', '-movflags', '+faststart'])
                writer.send(None)
                for index, snapshot in enumerate(samples):
                    if time.monotonic() > deadline:
                        raise ValueError('Character preview exceeded its render deadline')
                    frame = np.ascontiguousarray(renderer.render(snapshot, settings)[:, 960:])
                    if index == 0:
                        Image.fromarray(frame).save(Path(folder) / 'poster.jpg', quality=88)
                    writer.send(frame)
            finally:
                try:
                    if writer is not None:
                        writer.close()
                finally:
                    renderer.close()
            if not temporary.is_file() or not 0 < temporary.stat().st_size <= 20_000_000:
                raise ValueError('Character video exceeded its output bound')
            temporary.replace(target)
            (Path(folder) / 'poster.jpg').replace(still)
            receipt.write_text(json.dumps({'asset_sha256': key, 'render_sha256': package.render_sha256,
                'source_sha256': source_hash, 'frames': len(samples), 'fps': 15,
                'scope': 'Selected character rendered on saved motion; not a new scored trial.'}, indent=2) + '\n')
    visuals = visual_file(root, project, 'lab/run/' + target.name, 'video', 'Character preview')
    if visuals:
        visuals[0]['poster'] = f'/api/projects/{project}/files/lab/run/{still.name}'
    return visuals


def prepare_run(root, identity, record=None):
    root = Path(root).resolve()
    if not re.fullmatch(r'run-[a-zA-Z0-9_-]{1,60}', identity):
        raise ValueError('Unknown presentation run')
    run_dir = root / 'demo' / identity
    record = record or _json(run_dir / 'record.json')
    project = record.get('project')
    if not isinstance(project, str) or not re.fullmatch(r'[a-f0-9]{32}', project):
        return {}
    base = root / project
    session = Path(record.get('video') or (str(Path(record['session']) / 'avatar-video.mp4') if record.get('session') else '')).parent
    if session.resolve() != session or session.parent != base / 'lab/sessions':
        return {}
    prepared = {'steps': {}, 'notes': []}
    source = base / 'lab/run/source.mp4'
    source_poster = base / 'lab/run/studio-source-poster.jpg'
    if poster(source, source_poster):
        prepared['source_poster_url'] = f'/api/projects/{project}/files/lab/run/{source_poster.name}'
    prepared['steps']['intake'] = visual_file(root, project, 'lab/run/source.mp4', 'video', 'The original movement')
    if prepared['steps']['intake']:
        prepared['steps']['intake'][0]['poster'] = prepared.get('source_poster_url')
    try:
        parents = _json(run_dir / 'projects.json', 16_000)
        for parent in parents[:4]:
            tracked = visual_file(root, parent, 'perception/preflight/body_source/tracks-contact-sheet.png',
                                  'image', 'The movement, frame by frame')
            if tracked:
                prepared['steps']['intake'] += tracked
                break
    except (OSError, ValueError):
        pass
    try:
        prepared['steps']['world'] = save_reference_preview(root, project)
    except (OSError, ValueError, KeyError, TypeError) as error:
        prepared['notes'].append('Model: ' + str(error)[:160])
    try:
        task = _json(base / 'lab/run/exercise-task.json')
        prepared['steps']['task'] = [{'kind': 'facts', 'label': 'What the controller is asked to do', 'items': [
            ['Goal', 'Follow the reconstructed movement'],
            ['Length', str((task.get('interval') or {}).get('horizon_steps', '—')) + ' control steps'],
            ['Scoring rules', task.get('gate_profile', 'Saved task contract')],
        ]}]
    except (OSError, ValueError):
        pass
    recordings = recording_visuals(root, {**record, 'video': str(session / 'avatar-video.mp4')})
    for step, filename in [('rollout', 'video.mp4'), ('avatar', 'avatar-video.mp4')]:
        assets = recordings.get(step, [])
        if assets:
            target = session / (Path(filename).stem + '-poster.jpg')
            if poster(session / filename, target):
                assets[0]['poster'] = f'/api/projects/{project}/files/lab/sessions/{session.name}/{target.name}'
            prepared['steps'][step] = assets
    if recordings.get('avatar'):
        prepared['poster_url'] = recordings['avatar'][0].get('poster')
    try:
        review = _json(base / 'muse/advice/hypotheses.json').get('advice', {})
        hypotheses = review.get('hypotheses') or []
        prepared['steps']['muse'] = [{'kind': 'facts', 'label': 'What Meta noticed', 'items': [
            ['Summary', review.get('summary') or 'Saved review'],
            *[['Suggestion', str(item.get('claim') or item.get('summary') or item.get('title') or item)[:600]]
              for item in hypotheses[:4] if isinstance(item, dict)],
        ]}]
    except (OSError, ValueError):
        pass
    prepared['steps']['create'] = prepared['steps'].get('avatar', [])
    (run_dir / 'presentation.json').write_text(json.dumps(prepared, indent=2) + '\n')
    return prepared


def read_presentation(root, identity):
    if not re.fullmatch(r'run-[a-zA-Z0-9_-]{1,60}', identity or ''):
        return {}
    try:
        prepared = _json(Path(root).resolve() / 'demo' / identity / 'presentation.json', 128_000)
        steps = prepared.setdefault('steps', {})
        steps['create'] = steps.get('avatar', [])
        return prepared
    except (OSError, ValueError):
        return {}
