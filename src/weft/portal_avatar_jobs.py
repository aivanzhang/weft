"""One bounded avatar preparation worker; request paths read cached metadata only."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import threading
from uuid import uuid4

from weft.avatar_lab import AvatarCatalog
from weft.portal_avatar import MAX_UPLOAD_BYTES, PortalAvatarLab, import_avatar_zip, prepare_builtin_avatars
from weft.shared_world_packaging import read_package_file, write_package_files

RESERVE_PER_ASSET = 97 * 1024 * 1024
# Two builtins plus library picks and created characters; the catalog, offscreen and
# replay caps below track the same total of eight.
MAX_CUSTOM = 6
MAX_ATTACHMENTS = 3


def _bytes(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > 128 * 1024:
        raise ValueError('Avatar registry exceeds its metadata bound')
    return raw


class PortalAvatarManager:
    def __init__(self, root, project):
        self.root = Path(root).absolute()
        self.project = Path(project).absolute()
        self.root.mkdir(mode=0o700, exist_ok=True, parents=True)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('Avatar library must not use symbolic links')
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='weft-avatar')
        self._catalog = None
        self._entries = []
        self._attachments = {}
        self._closed = False
        self._state = {'status': 'running', 'operation': 'restore', 'error': None, 'reserved_bytes': 0}
        self._future = self._executor.submit(self._restore)

    def _publish(self, state):
        with self._lock:
            self._state.update(state)
            from weft.lab_worker import atomic_json
            atomic_json(self.root / 'status.json', self._state)

    def _load_entries(self, entries):
        if not isinstance(entries, list) or len(entries) > 8:
            raise ValueError('Avatar registry exceeds eight assets')
        seen = set()
        directories = []
        for row in entries:
            if (not isinstance(row, dict) or set(row) != {'kind', 'directory', 'manifest_sha256', 'key'}
                    or row['kind'] not in ('builtin', 'custom')
                    or not isinstance(row['directory'], str)
                    or not re.fullmatch(r'operations/[0-9a-f]{32}/(?:builtin/(?:muse-panda|robot)/asset|custom/asset)', row['directory'])
                    or row['directory'] in seen
                    or any(not isinstance(row[k], str) or not re.fullmatch('[0-9a-f]{64}', row[k]) for k in ('manifest_sha256', 'key'))):
                raise ValueError('Avatar registry entry is invalid')
            seen.add(row['directory'])
            directory = self.root / row['directory']
            read_package_file(directory, 'manifest.json', max_bytes=128*1024, expected_sha256=row['manifest_sha256'])
            directories.append(directory)
        if sum(row['kind'] == 'custom' for row in entries) > MAX_CUSTOM or sum(row['kind'] == 'builtin' for row in entries) not in (0, 2):
            raise ValueError('Avatar registry count differs from two builtins and six custom slots')
        catalog = AvatarCatalog(directories)
        if [row['key'] for row in entries] != list(catalog.packages):
            raise ValueError('Avatar registry identity differs from its imported source')
        return catalog

    def _restore(self):
        try:
            previous = None
            if (self.root / 'status.json').exists():
                from weft.spatial_scene_inputs import _json
                previous = _json(read_package_file(self.root, 'status.json', max_bytes=128*1024),128*1024)
            index = self.root / 'catalog.json'
            if index.exists() or index.is_symlink():
                from weft.spatial_scene_inputs import _json
                raw = read_package_file(self.root, 'catalog.json', max_bytes=128*1024)
                saved = _json(raw, 128*1024)
                if not isinstance(saved, dict) or set(saved) != {'profile', 'entries'} or saved['profile'] != 'portal-avatar-catalog-v1':
                    raise ValueError('Avatar catalog profile is invalid')
                entries = saved['entries']; catalog = self._load_entries(entries)
            else:
                entries = []; catalog = AvatarCatalog()
            with self._lock:
                self._entries = entries; self._catalog = catalog
            restored = {'status': 'ready' if entries else 'not_prepared', 'operation': None, 'error': None}
            if isinstance(previous, dict) and previous.get('status') in ('running', 'failed', 'interrupted'):
                restored.update(status='interrupted' if previous['status'] == 'running' else previous['status'],
                    error='Previous avatar work interrupted; partial files preserved, no automatic retry.' if previous['status'] == 'running' else str(previous.get('error', ''))[:1000])
            self._publish(restored)
        except Exception as error:
            self._publish({'status': 'failed', 'operation': 'restore', 'error': str(error)[:1000]})

    def _available(self):
        if self._closed:
            raise ValueError('Avatar manager is closed')
        if self._future is not None and not self._future.done():
            raise ValueError('Avatar preparation is already running')
        if self._catalog is None:
            raise ValueError('Avatar catalog could not be restored; preserved files require review')

    def status(self):
        with self._lock:
            choices = deepcopy(list(self._catalog.entries)) if self._catalog is not None else []
            return deepcopy(self._state) | {'builtins_ready': sum(row['kind'] == 'builtin' for row in self._entries) == 2,
                'custom_count': sum(row['kind'] == 'custom' for row in self._entries), 'choices': choices, 'custom_slots_remaining': MAX_CUSTOM - sum(row['kind'] == 'custom' for row in self._entries),
                'attachments': {key: {'status': row['status'], 'error': row.get('error')} for key, row in self._attachments.items()},
                'scope': 'Appearance only. Imported animation never changes physics, rewards or task checks.'}

    def reserve_for(self, raw=None):
        with self._lock:
            self._available()
            if raw is not None and (type(raw) is not bytes or not 0 < len(raw) <= MAX_UPLOAD_BYTES):
                raise ValueError('Avatar upload requires ZIP bytes up to 32 MiB')
            if raw is None and any(row['kind'] == 'builtin' for row in self._entries):
                return 0
            if raw is not None and sum(row['kind'] == 'custom' for row in self._entries) >= MAX_CUSTOM:
                raise ValueError('This portal already retains its six custom avatar slots')
            return RESERVE_PER_ASSET * (2 if raw is None else 1)

    def submit(self, raw=None):
        with self._lock:
            self._available()
            if raw is not None and (type(raw) is not bytes or not 0 < len(raw) <= MAX_UPLOAD_BYTES):
                raise ValueError('Avatar upload requires ZIP bytes up to 32 MiB')
            if raw is None and any(row['kind'] == 'builtin' for row in self._entries):
                return self.status()
            if raw is not None and sum(row['kind'] == 'custom' for row in self._entries) >= MAX_CUSTOM:
                raise ValueError('This portal already retains its six custom avatar slots')
            operation = uuid4().hex
            self._state = {'status': 'running', 'operation': 'builtins' if raw is None else 'import',
                           'operation_id': operation, 'error': None, 'reserved_bytes': RESERVE_PER_ASSET * (2 if raw is None else 1)}
            self._future = self._executor.submit(self._prepare, operation, raw)
            return self.status()

    def _prepare(self, operation, raw):
        directory = self.root / 'operations' / operation
        try:
            directory.mkdir(parents=True, mode=0o700)
            self._publish({})
            if raw is None:
                result = prepare_builtin_avatars(directory / 'builtin', project=self.project)
                new = list(result.values()); kind = 'builtin'
            else:
                result = import_avatar_zip(raw, directory / 'custom')
                new = [result['asset']]; kind = 'custom'
            entries = deepcopy(self._entries)
            for item in new:
                entries.append({'kind': kind, 'directory': Path(item['directory']).relative_to(self.root).as_posix(),
                                'manifest_sha256': item['manifest_sha256'], 'key': item['source_manifest_sha256']})
            catalog = self._load_entries(entries)
            from weft.lab_worker import atomic_json
            write_package_files(directory, {'result.json': _bytes({'status': 'ready', 'assets': new})})
            atomic_json(self.root / 'catalog.json', {'profile': 'portal-avatar-catalog-v1', 'entries': entries})
            with self._lock:
                self._entries = entries; self._catalog = catalog
            self._publish({'status': 'ready', 'error': None, 'reserved_bytes': 0})
        except Exception as error:
            try:
                if directory.is_dir():
                    write_package_files(directory, {'failure.json': _bytes({'status': 'failed', 'error': str(error)[:1000], 'partial_files_preserved': True})})
            finally:
                self._publish({'status': 'failed', 'error': str(error)[:1000], 'reserved_bytes': 0})

    def attach_async(self, identity, lab):
        if not isinstance(identity, str) or not re.fullmatch('[0-9a-f]{32}', identity):
            raise ValueError('Avatar attachment requires an exact project ID')
        with self._lock:
            self._available()
            if not self._catalog.packages:
                raise ValueError('Prepare or import an avatar before attaching it')
            self._attachments = {key: row for key, row in self._attachments.items()
                                 if not getattr(row.get('lab'), '_closed_bridge', False)}
            if identity not in self._attachments and len(self._attachments) >= MAX_ATTACHMENTS:
                raise ValueError('Close an attached sandbox before attaching another')
            prior = self._attachments.get(identity, {}).get('lab')
            self._attachments[identity] = {'status': 'running', **({'lab': prior} if prior is not None else {})}
            self._state.update(status='running', operation='attach', error=None, reserved_bytes=0)
            self._future = self._executor.submit(self._attach, identity, lab, self._catalog)
            return self.status()

    def _attach(self, identity, lab, catalog):
        try:
            self._publish({})
            base = lab._lab if isinstance(lab, PortalAvatarLab) else lab
            with base.lock:
                wrapped = PortalAvatarLab(base, catalog)
            with self._lock:
                self._attachments[identity] = {'status': 'ready', 'lab': wrapped}
            self._publish({'status': 'ready', 'error': None})
        except Exception as error:
            with self._lock:
                self._attachments[identity] = {**self._attachments.get(identity, {}), 'status': 'failed', 'error': str(error)[:1000]}
            self._publish({'status': 'failed', 'error': str(error)[:1000]})

    def attached(self, identity):
        with self._lock:
            row = self._attachments.get(identity, {})
            lab = row.get('lab')
            return lab if lab is not None and not getattr(lab, '_closed_bridge', False) else None

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
