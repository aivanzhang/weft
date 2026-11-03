"""Admit immutable saved body/equipment inputs once; never reconstruct media or dynamics."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
import time

from weft.io import canonical_hash
from weft.shared_world_packaging import read_package_file, validate_package_layout
from weft.world_evidence import _check_tree, _freeze, _nonfinite, _thaw, _unique_object


PROFILE = 'moving-equipment-inputs-v1'
MAX_BYTES, MAX_MEMBER_BYTES, MAX_FILES, MAX_SECONDS = 128 << 20, 64 << 20, 512, 30.
_TOKEN = object()
_LAYOUTS = (('inputs/human-source/source-manifest.json', 'inputs/human-source/'),
            ('inputs/body-source-manifest.json', ''))
_RUNTIME = ('moving_equipment_inputs.py', 'moving_equipment_evidence.py', 'equipment_root_binding.py',
            'exercise_body_source.py', 'source_clock.py', 'source_clock_capture.py', 'root_motion.py',
            'shared_human_projection.py', 'shared_world.py', 'shared_world_evidence.py',
            'shared_world_packaging.py', 'shared_world_pipeline.py', 'equipment_box_consensus.py',
            'segmentation_observations.py', 'observations.py', 'world_evidence.py', 'human.py',
            'pose_capture_recovery.py', 'pose_source_recovery.py', 'pose_mode_source.py', 'io.py',
            'shared_world_reconstruction.py', 'exercise_vision.py', 'exercise_source.py', 'video_io.py',
            'pose_identity.py', 'pose_crop.py', 'pose_crop_selection.py', 'pose_mode_recovery.py',
            'mobile_human.py', 'body_frame.py', 'spatial_pose.py', 'whole_body.py', 'spatial_rigid.py',
            'shared_world_observation.py', 'source_equivalence.py', 'native_interval.py', 'dynamic_native_interval.py',
            'person_inventory.py', 'errors.py')


def _require(value, message):
    if not value:
        raise ValueError('Moving equipment inputs: ' + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Moving equipment inputs exceeded its aggregate deadline')


def _deadline(max_seconds):
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             'use a finite positive bound of at most thirty seconds')
    return time.monotonic() + max_seconds


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    _require(isinstance(value, dict), 'JSON object required')
    return value


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _manifest(raw):
    value = _json(raw)
    members = value.get('files')
    _require(type(members) is dict and 0 < len(members) <= MAX_FILES, 'bounded original member seal required')
    validate_package_layout(members)
    _require('manifest.json' not in members and all(type(pin) is str and len(pin) == 64
             and all(c in '0123456789abcdef' for c in pin) for pin in members.values()), 'exact safe member hashes required')
    _require(value.get('status') not in {'failed', 'rejected', 'cancelled', 'interrupted'}, 'failed source package is not admissible')
    return value, members


def _layout(members):
    choices = [item for item in _LAYOUTS if item[0] in members]
    _require(len(choices) == 1, 'one original sealed body layout required')
    return choices[0]


def _body_paths(base_members, original, prefix, marker):
    names = marker.get('artifacts')
    source = marker.get('source_artifact')
    _require(type(names) is dict and type(source) is str, 'original body artifact inventory required')
    wanted = set(names) | {'body-source.json', source, 'root-motion.json'}
    validate_package_layout(wanted)
    paths = {}
    for name in sorted(wanted):
        candidates = [path for path in dict.fromkeys((prefix + name, name)) if path in base_members]
        _require(name in original and candidates and all(base_members[path] == original[name] for path in candidates),
                 'retained body member differs from its original seal: ' + name)
        paths[name] = candidates[0]
    return paths


@dataclass(frozen=True, init=False)
class MovingEquipmentInputs:
    base_scene_bytes: bytes
    human_bytes: bytes
    construction_evidence_bytes: bytes
    motion_evidence_bytes: bytes
    admission_receipt_bytes: bytes
    root_proof: object
    observation_documents: object
    source_files: object
    _views: object
    _token: object
    _origins: object

    @property
    def base_scene(self): return _thaw(self._views['base_scene'])
    @property
    def human(self): return _thaw(self._views['human'])
    @property
    def source_binding(self): return _thaw(self._views['source_binding'])
    @property
    def body_binding(self): return _thaw(self._views['body_binding'])
    @property
    def body_references(self): return _thaw(self._views['body_references'])
    @property
    def motion_evidence(self): return self._views['motion_evidence']
    @property
    def identity_context(self): return None


def validate_moving_equipment_inputs(packet, *, deadline=None):
    """In-process admission is reusable; external bytes must pass admit_moving_equipment_files."""
    if deadline is not None:
        _require(type(deadline) in (int, float) and math.isfinite(deadline), 'finite deadline required')
        _check(deadline)
    _require(type(packet) is MovingEquipmentInputs and getattr(packet, '_token', None) is _TOKEN,
             'use an immutable packet returned by original saved-input admission')
    return packet


def verify_moving_equipment_inputs_unchanged(packet, *, deadline):
    """Recheck originals once after a candidate batch, never once per candidate or action."""
    validate_moving_equipment_inputs(packet, deadline=deadline)
    total = 0
    for name, (root, relative, signature) in packet._origins.items():
        _check(deadline)
        path = Path(root) / relative
        _require(not any(part.is_symlink() for part in (path, *path.parents)), 'original path became a symlink')
        state = path.stat()
        _require((state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns, state.st_ctime_ns) == signature,
                 'original source changed after admission')
        content = read_package_file(root, relative, max_bytes=len(packet.source_files[name]),
                                    expected_sha256=sha256(packet.source_files[name]).hexdigest())
        total += len(content)
    _check(deadline)
    return {'verified_files': len(packet._origins), 'verified_bytes': total,
            'scope': 'original_files_and_exact_hashes' if packet._origins else 'immutable_in_memory_inputs_only'}


def _admit(source_files, deadline):
    from weft.exercise_body_source import validate_body_source_files
    from weft.equipment_root_binding import derive_body_binding
    from weft.moving_equipment_evidence import prepare_motion_evidence
    from weft.segmentation_observations import adapt_segmentation
    from weft.shared_human_projection import validate_projection
    from weft.shared_world import validate_shared_world
    from weft.shared_world_evidence import build_shared_world_binding
    from weft.shared_world_reconstruction import _construction_index

    _require(isinstance(source_files, (dict, MappingProxyType)) and 0 < len(source_files) <= MAX_FILES,
             'bounded fixed source inventory required')
    validate_package_layout(source_files)
    _require(all(type(raw) is bytes and 0 < len(raw) <= MAX_MEMBER_BYTES for raw in source_files.values())
             and sum(map(len, source_files.values())) <= MAX_BYTES, 'complete retained source byte bound')
    files = dict(source_files)
    manifests = {}
    for label in ('base', 'equipment'):
        _, manifests[label] = _manifest(files[label + '/manifest.json'])
    for name, raw in files.items():
        _check(deadline)
        label, relative = name.split('/', 1)
        _require(label in manifests, 'only fixed base/equipment namespaces are admitted')
        if relative != 'manifest.json':
            _require(manifests[label].get(relative) == sha256(raw).hexdigest(), 'retained source member hash differs: ' + name)
    runtime = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in _RUNTIME}
    base_raw, human_raw = files['base/scene.json'], files['base/human.json']
    base = _json(base_raw)
    _require(base.get('profile') in ('shared-world-v1', 'shared-world-v3'), 'original seed/person-only base profile required')
    _require(validate_shared_world(base) == base and base['source_binding']['mode'] == 'source_qualified'
             and not base['source_binding'].get('source_equivalences'), 'unchanged same-source base required')
    original_name, prefix = _layout(manifests['base'])
    original_value, original = _manifest(files['base/' + original_name])
    _require(original_value.get('profile') == 'exercise-body-source-v1'
             and original_value.get('status') == 'construction_only' and original_value.get('simulation_ready') is False,
             'original construction-only body seal required')
    marker = _json(files['base/' + prefix + 'body-source.json'])
    paths = _body_paths(manifests['base'], original, prefix, marker)
    body_files = {name: files['base/' + path] for name, path in paths.items()}
    _require(original.get('human.json') == sha256(human_raw).hexdigest() and body_files['human.json'] == human_raw,
             'base human sidecar differs from original body bytes')
    _check(deadline)
    human, observations = validate_body_source_files(body_files, deadline=deadline)
    _check(deadline)
    _require('native_clock' in human, 'this input profile requires complete native body custody')
    if 'human_provenance' in base:
        validate_projection(base['human_spec'], base['human_provenance'], human_raw)
    else:
        _require(canonical_hash(base['human_spec']) == canonical_hash(human), 'base changes original full human')
    human_entity = next(row for row in base['source_binding']['entity_bindings'] if row['world_entity_id'] == 'human')
    human_sha = sha256(body_files['observations.json']).hexdigest()
    _require(human_entity['observation_document_sha256'] == human_sha, 'base body observation identity differs')
    observation_path = base['source_binding']['evidence_documents'][human_sha]
    _require(files['base/' + observation_path] == body_files['observations.json'], 'base observation bytes differ from original body')
    base_ids = ['human', *(obj['id'] for obj in base['objects'])]
    old_binding, _ = build_shared_world_binding({human_sha: body_files['observations.json']},
        source_sha256=human['source_sha256'], world_ids=base_ids,
        entity_refs={name: (human_sha, human_entity['observation_entity_id']) if name == 'human' else None for name in base_ids}, deadline=deadline)
    old_binding['evidence_documents'] = base['source_binding']['evidence_documents']
    _require(old_binding == base['source_binding'], 'base source binding differs from exact original body evidence')
    equipment_raw = files['equipment/scene-observations.json']
    equipment = _json(equipment_raw)
    segmentation = _json(files['equipment/segmentation-evidence.json'])
    _require(canonical_hash(adapt_segmentation(segmentation)) == canonical_hash(equipment),
             'equipment observations differ from the complete original segmentation adapter output')
    _require(equipment['source']['sha256'] == human['source_sha256'], 'equipment and body must retain the same exact source')
    equipment_sha = sha256(equipment_raw).hexdigest()
    docs = {human_sha: body_files['observations.json'], equipment_sha: equipment_raw}
    ids = [obj['id'] for obj in equipment['objects']]
    _require(1 <= len(ids) <= 12, 'complete equipment inventory bound')
    refs = {'human': (human_sha, human_entity['observation_entity_id']),
            **{f'raw-{i}': (equipment_sha, identifier) for i, identifier in enumerate(ids)}}
    binding, construction = build_shared_world_binding(docs, source_sha256=human['source_sha256'],
        world_ids=list(refs), entity_refs=refs, deadline=deadline)
    indexed, documents = _construction_index(construction, binding)
    root_proof = {name: body_files[name] for name in ('reconstruction-evidence.json', 'root-motion.json')}
    body_binding, frames = derive_body_binding(base['human_spec'], human_raw, binding['entity_bindings'][0], indexed, documents,
                                             root_proof=root_proof)
    del indexed, documents, observations, segmentation
    body_refs = [{'index': i, 'source_frame': frame, **{key: human['reference_evidence'][i][key]
                  for key in ('evidence_id', 'time_s')}} for i, frame in enumerate(frames)]
    _check(deadline)
    motion = prepare_motion_evidence(construction, source_binding=binding, body_references=body_refs,
        observation_documents={equipment_sha: equipment_raw}, max_seconds=min(10., deadline - time.monotonic()))
    receipt = {'schema_version': 1, 'profile': PROFILE, 'source_sha256': human['source_sha256'],
        'files': {name: sha256(raw).hexdigest() for name, raw in files.items()}, 'runtime_sha256': runtime,
        'motion_evidence_sha256': sha256(motion).hexdigest(), 'body_binding_sha256': canonical_hash(body_binding),
        'source_binding_sha256': canonical_hash(binding), 'construction_evidence_sha256': sha256(construction).hexdigest(),
        'scope': 'Original saved body/native/root and segmentation-adapter custody; not physical identity, geometry, initial state, contact or task admission'}
    _check(deadline)
    _require(all(sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == pin for name, pin in runtime.items()),
             'admission runtime changed')
    packet = object.__new__(MovingEquipmentInputs)
    values = dict(base_scene_bytes=base_raw, human_bytes=human_raw, construction_evidence_bytes=construction,
        motion_evidence_bytes=motion, admission_receipt_bytes=_bytes(receipt), root_proof=MappingProxyType(root_proof),
        observation_documents=MappingProxyType(docs), source_files=MappingProxyType(files),
        _views=_freeze({'base_scene': base, 'human': human, 'source_binding': binding, 'body_binding': body_binding,
                       'body_references': body_refs, 'motion_evidence': json.loads(motion)}), _token=_TOKEN,
        _origins=MappingProxyType({}))
    for key, value in values.items(): object.__setattr__(packet, key, value)
    return packet


def admit_moving_equipment_files(source_files, *, max_seconds=30):
    """Revalidate external package bytes once before any candidate generation or playback."""
    try:
        return _admit(source_files, _deadline(max_seconds))
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError('Moving equipment inputs: incomplete or malformed source custody') from error


def load_moving_equipment_inputs(base_run, evidence_run, *, base_manifest_sha256, evidence_manifest_sha256,
                                 source_mappings=(), max_seconds=30):
    """Read fixed sealed members once; no source scanning, media decoding or inference."""
    deadline = _deadline(max_seconds)
    _require(source_mappings == () or source_mappings == [], 'mapped media require a separately qualified input profile')
    roots = {'base': Path(base_run).absolute(), 'equipment': Path(evidence_run).absolute()}
    files, manifests, signatures = {}, {}, {}
    total = 0
    def read(label, name, maximum=MAX_MEMBER_BYTES):
        nonlocal total
        key = label + '/' + name
        if key in files: return files[key]
        _check(deadline)
        path = roots[label] / name
        _require(not any(p.is_symlink() for p in (path, *path.parents)) and '..' not in path.parts,
                 'source paths cannot traverse symlinks or parents')
        before = path.stat()
        _require(len(files) + 1 <= MAX_FILES and before.st_size <= min(maximum, MAX_BYTES - total), 'source inventory bound')
        pin = manifests[label][name] if name != 'manifest.json' else None
        content = read_package_file(roots[label], name, max_bytes=min(maximum, MAX_BYTES - total), expected_sha256=pin)
        after = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        _require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'source changed during read')
        files[key] = content; signatures[key] = signature; total += len(content)
        return content
    for label, pin in [('base', base_manifest_sha256), ('equipment', evidence_manifest_sha256)]:
        raw = read(label, 'manifest.json', 8 << 20)
        _require(sha256(raw).hexdigest() == pin, 'caller-pinned original package seal differs')
        _, manifests[label] = _manifest(raw)
    base = _json(read('base', 'scene.json', 16 << 20))
    read('base', 'human.json', 8 << 20)
    original_name, prefix = _layout(manifests['base'])
    _, original = _manifest(read('base', original_name, 8 << 20))
    marker = _json(read('base', prefix + 'body-source.json', 1 << 20))
    for path in _body_paths(manifests['base'], original, prefix, marker).values(): read('base', path)
    for path in base['source_binding']['evidence_documents'].values(): read('base', path)
    for name in ('scene-observations.json', 'segmentation-evidence.json'): read('equipment', name)
    _check(deadline)
    packet = _admit(files, deadline)
    for key, expected in signatures.items():
        label, name = key.split('/', 1); current = (roots[label] / name).stat()
        _require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) == expected,
                 'source changed before admission publication')
    object.__setattr__(packet, '_origins', MappingProxyType({key: (str(roots[key.split('/', 1)[0]]),
        key.split('/', 1)[1], signature) for key, signature in signatures.items()}))
    _check(deadline)
    return packet
