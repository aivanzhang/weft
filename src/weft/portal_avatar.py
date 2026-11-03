"""Bounded avatar deliveries and appearance-only adapters for admitted body labs."""
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import stat
import time
from uuid import uuid4
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED

from weft.avatar_assets import import_avatar_model, MAX_INPUT_BYTES
from weft.avatar_lab import AvatarCatalog, DEFAULT_APPEARANCE, validate_appearance
from weft.avatar_recording import AppearanceRecorder
from weft.recording import Recorder
from weft.shared_world_packaging import read_package_file, write_package_files

MAX_UPLOAD_BYTES = MAX_INPUT_BYTES
MAX_MAPPING_BYTES = 128 * 1024
BUILTINS = {'muse-panda': ('weft-muse-panda-v1', 'b32e5cc08510035aa7c778dff33635ac8850af42536ac13daac9bfb92954f012'),
            'robot': ('weft-humanoid-robot-v1', '48cc6156fff0200b76c24c4ef785113da2389bdbb0924e121ba45e69ac7cf6cd')}
SUPPORTED = {'shared-world-v1', 'shared-world-v2', 'shared-world-v3'}


def _deadline(seconds):
    if type(seconds) not in (int, float) or not 0 < seconds <= 60 or not math.isfinite(seconds):
        raise ValueError('Avatar preparation requires a finite bound at most 60 seconds')
    return time.monotonic() + seconds


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Avatar preparation deadline exceeded; partial files preserved')
    return remaining


def _json(raw):
    from weft.spatial_scene_inputs import _json as strict_json
    return strict_json(raw, MAX_MAPPING_BYTES)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def import_avatar_zip(raw, out, *, max_seconds=15):
    """Worker-only intake: a self-contained GLB and explicit rights/rig mapping, never code."""
    deadline = _deadline(max_seconds)
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_UPLOAD_BYTES:
        raise ValueError('Avatar ZIP exceeds 32 MiB')
    with ZipFile(BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 2 or {entry.filename for entry in entries} != {'model.glb', 'mapping.json'}:
            raise ValueError('Avatar ZIP must contain only model.glb and mapping.json at its root')
        total = 0
        for entry in entries:
            mode = entry.external_attr >> 16
            if (entry.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                    or entry.flag_bits & 1 or entry.compress_type not in (ZIP_STORED, ZIP_DEFLATED)):
                raise ValueError('Avatar ZIP contains an unsupported member')
            maximum = MAX_MAPPING_BYTES if entry.filename == 'mapping.json' else MAX_INPUT_BYTES
            total += entry.file_size
            if not 0 < entry.file_size <= maximum or total > MAX_INPUT_BYTES:
                raise ValueError('Expanded avatar ZIP exceeds its input bound')
        payload = {entry.filename: archive.read(entry) for entry in entries}
    _remaining(deadline)
    metadata = _json(payload['mapping.json'])
    out = Path(out).absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    (out / 'input').mkdir()
    write_package_files(out / 'input', payload)
    result = import_avatar_model(out / 'input/model.glb', metadata, out / 'asset',
                                 max_seconds=min(10, _remaining(deadline)))
    receipt = {'profile': 'portal-avatar-import-v1', 'upload_sha256': sha256(raw).hexdigest(),
               'input_sha256': {name: sha256(value).hexdigest() for name, value in payload.items()},
               'asset': result, 'scope': 'Appearance only; authored animation never controls physics'}
    write_package_files(out, {'receipt.json': _bytes(receipt)})
    return receipt


def prepare_builtin_avatars(out, *, project, max_seconds=30):
    """Import immutable deliveries with explicit v2 role-convention correction."""
    deadline = _deadline(max_seconds)
    out, project = Path(out).absolute(), Path(project).absolute()
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    results = {}
    for name, (directory, expected) in BUILTINS.items():
        source = project / 'tools/avatar-assets/outputs' / directory
        raw = read_package_file(source, 'model.glb', max_bytes=MAX_INPUT_BYTES, expected_sha256=expected)
        original = read_package_file(source, 'mapping.json', max_bytes=MAX_MAPPING_BYTES)
        metadata = _json(original)
        mapping = metadata['retargeting']
        def role(body):
            if body.startswith('left_'): return 'right_' + body[5:]
            if body.startswith('right_'): return 'left_' + body[6:]
            return body
        metadata['retargeting'] = {key: deepcopy(mapping[key]) for key in ('root_node', 'scale', 'orientation_xyzw')}
        metadata['retargeting'].update(profile='weft-avatar-retarget-v2',
            bindings=[{'node': item['node'], 'body': role(item['body'])} for item in mapping['bindings']])
        metadata['provenance']['weft_delivery_bridge'] = {
            'source_model_sha256': expected, 'source_mapping_sha256': sha256(original).hexdigest(),
            'scope': 'New mapping only. Left/right roles align negative-X Weft left with positive-X authored naming; original asset hierarchy and proportions retained. Uniform scale remains authored, not calibrated. Baked clips are unused.'}
        metadata['asset_id'] += '-runtime-v2'
        stage = out / name
        (stage / 'input').mkdir(parents=True)
        write_package_files(stage / 'input', {'model.glb': raw, 'mapping.json': _bytes(metadata)})
        results[name] = import_avatar_model(stage / 'input/model.glb', metadata, stage / 'asset',
                                           max_seconds=min(10, _remaining(deadline)))
    write_package_files(out, {'receipt.json': _bytes({'profile': 'portal-avatar-builtins-v1', 'assets': results})})
    return results


class PortalAppearanceRecorder(AppearanceRecorder):
    """Reuse appearance journaling while admitting the same existing v3 physical recorder."""
    def __init__(self, root, scene, *, profile, catalog, settings, executor):
        if scene.get('engine_scene', scene).get('profile') not in SUPPORTED:
            raise ValueError('Avatar recording is not qualified for this body profile')
        Recorder.__init__(self, root, scene, profile=profile)
        self.catalog = catalog
        self.settings = validate_appearance(settings, catalog.packages)
        self._appearance_executor = executor
        self._appearance_stream = None
        self._appearance_count = self._appearance_bytes = 0
        self._last_physical_hash = None
        self.asset_future = None


class PortalAvatarLab:
    """Compose with one already admitted Lab; never replace its session or physical export."""
    def __init__(self, lab, catalog):
        session = lab.load_session()
        if (session.scene.get('profile') not in SUPPORTED
                or getattr(session, 'control_profile', None) in ('shared_world_exploration', 'moving_equipment_task')):
            raise ValueError('Avatar recording currently supports ordinary shared body labs only')
        if lab.recorder is not None or (lab.export_future is not None and not lab.export_future.done()):
            raise ValueError('Stop and export before attaching an avatar catalog')
        if not isinstance(catalog, AvatarCatalog):
            raise ValueError('A worker-prepared avatar catalog is required')
        self._lab = lab
        self._closed_bridge = False
        lab.avatars = catalog
        lab.appearance = validate_appearance(DEFAULT_APPEARANCE, catalog.packages)

    def __getattr__(self, name):
        return getattr(self._lab, name)

    def close(self):
        if not self._closed_bridge:
            self._closed_bridge = True
            self._lab.close()

    def control(self, path, data):
        if path != '/api/record/start' or not self._lab.avatars.packages:
            return self._lab.control(path, data)
        from weft.lab import _shared_exercise
        lab = self._lab
        if data:
            raise ValueError('Avatar recording uses the current task and seed; no overrides')
        if lab.recorder is not None:
            raise ValueError('A recording is already active')
        if lab.export_future is not None and not lab.export_future.done():
            raise ValueError('Wait for the current export to finish')
        session = lab.load_session()
        snapshot = session.reset(lab.seed)
        scene = session.recording_scene() if _shared_exercise(session) else session.scene
        name = 'session-' + uuid4().hex[:12]
        recorder = PortalAppearanceRecorder(lab.root / 'sessions' / name, scene,
            profile='shared-exercise' if _shared_exercise(session) else 'shared-world',
            catalog=lab.avatars, settings=lab.appearance, executor=lab.executor)
        try:
            recorder.start(snapshot, seed=lab.seed)
        except Exception:
            if recorder._stream is not None and not recorder._closed:
                recorder.close()
            raise
        lab.recorder = recorder
        return {'recording': True, 'directory': name, 'snapshot': snapshot}
