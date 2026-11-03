"""Complete image-evidence ledger; no motion fit, initialization or physical admission."""

from collections.abc import Mapping
import hashlib
import json
import math
import time

from weft.equipment_box_consensus import _clock, _prepare
from weft.io import canonical_hash
from weft.world_evidence import _thaw, load_world_evidence


PROFILE = 'moving-equipment-evidence-v1'
MAX_INPUT_BYTES, MAX_OUTPUT_BYTES = 64 << 20, 16 << 20
MAX_SECONDS, MAX_SLOTS = 10., 21600
PHASES = frozenset({'pre_initialization', 'initialization', 'post_initialization'})
SCOPE = ('Structural construction-image ledger only; not source-world admission, depth, physical identity, '
         'initial state, contact, grasp, trajectory or a task. Returned pre-initialization proposals retain '
         'their provider phase, not verified backward physical observations.')
LIMITS = {'maximum_input_bytes': MAX_INPUT_BYTES, 'maximum_output_bytes': MAX_OUTPUT_BYTES,
          'maximum_seconds': MAX_SECONDS, 'maximum_slots': MAX_SLOTS,
          'measurements': 'Exact image box/center only; other source channels remain in sealed inputs',
          'withheld_payloads': 'Not accessed for ledger fields; original documents remain hash-bound',
          'missing_samples': 'Retained without interpolation, selection or absence claims',
          'source_qualification': 'Requires later original body/root/media/source-equivalence custody checks'}


def _require(value, message):
    if not value:
        raise ValueError('Moving equipment evidence: ' + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Moving equipment evidence exceeded its aggregate time bound')


def _encode(value, maximum, deadline):
    output = bytearray()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        _check(deadline)
        part = chunk.encode()
        _require(len(output) + len(part) <= maximum, 'complete output exceeds its byte bound')
        output.extend(part)
    return bytes(output)


def _phase(record, row):
    if record['kind'] != 'model_segmentation_proposal':
        return None
    payload = record['payload']
    _require(all(payload.get(key) == row[key] for key in ('frame', 'time_s', 'object_id'))
             and payload.get('id') == record['id'], 'provider phase record identity differs')
    raw_id, phase, status = payload.get('raw_model_id'), payload.get('phase'), payload.get('proposal_status')
    _require(type(raw_id) is str and 1 <= len(raw_id) <= 80 and raw_id.strip() == raw_id,
             'provider raw identity must be an explicit bounded string')
    _require(type(phase) is str and phase in PHASES and status in ('returned', 'missing'),
             'provider phase or proposal status is unsupported')
    _require((status == 'returned') == (record['status'] == 'inferred'),
             'provider status differs from its exact evidence status')
    return {'evidence_id': record['id'], 'raw_model_id': raw_id, 'phase': phase, 'proposal_status': status}


def _prepare_motion_evidence(construction_evidence, source_binding, body_references, observation_documents, max_seconds):
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= MAX_SECONDS,
             'use a finite positive time bound of at most ten seconds')
    deadline = time.monotonic() + max_seconds
    _require(type(construction_evidence) is bytes and isinstance(observation_documents, Mapping)
             and 1 <= len(observation_documents) <= 8, 'exact construction and bounded original documents required')
    _require(all(type(raw) is bytes for raw in observation_documents.values()), 'original documents require immutable bytes')
    remaining = MAX_INPUT_BYTES - len(construction_evidence) - sum(map(len, observation_documents.values()))
    _require(remaining > 0, 'aggregate input byte bound')
    encoded_binding = _encode(source_binding, remaining, deadline)
    encoded_refs = _encode(body_references, remaining - len(encoded_binding), deadline)
    _require(len(encoded_refs) <= remaining - len(encoded_binding), 'aggregate input byte bound')
    object_ids = [row['world_entity_id'] for row in source_binding['entity_bindings'] if row['world_entity_id'] != 'human']
    entities, sources, rows, references, _ = _prepare(construction_evidence, source_binding, body_references, object_ids, deadline)
    human = entities['human']
    human_source = sources[human['observation_document_sha256']]
    count = source_binding['total_source_frames']
    required = {sha for sha, source in sources.items() if any(
        (sha, obj['id']) != (human['observation_document_sha256'], human['observation_entity_id']) for obj in source['objects'])}
    _require(set(observation_documents) == required, 'raw documents must cover exactly all non-body source inventories')
    slot_count = sum(max(count, source['source']['metadata_frames']) * len(source['objects']) for source in sources.values())
    _require(slot_count <= MAX_SLOTS, 'complete source inventory exceeds the slot bound')
    registry = load_world_evidence(observation_documents)
    _check(deadline)
    projected = json.loads(registry.construction_bytes())
    for source in projected['sources']:
        _require(canonical_hash(source) == canonical_hash(sources[source['document_sha256']]),
                 'original document construction projection differs')
        _check(deadline)
    raw_evidence = {source.digest: {row['id']: row for row in source.document['evidence']} for source in registry._sources}
    result_sources = []
    for sha, source in sorted(sources.items()):
        tracks = []
        for obj in source['objects']:
            _check(deadline)
            entity_ref = {'document_sha256': sha, 'object_id': obj['id']}
            world_ids = sorted(name for name, bound in entities.items()
                               if (bound['observation_document_sha256'], bound['observation_entity_id']) == (sha, obj['id']))
            slots = []
            for frame in range(max(count, source['source']['metadata_frames'])):
                _check(deadline)
                ref = references.get(frame)
                slot = {'source_frame': frame, 'source_time_s': None, 'body_time_s': None,
                        'body_reference': None if ref is None else dict(ref), 'observation_id': None,
                        'partition': None, 'admission': 'not_decoded', 'visibility': None,
                        'evidence_refs': [], 'measurements': {}, 'phase_records': [], 'unavailable_reasons': []}
                if frame >= count:
                    slot['unavailable_reasons'].append('metadata_only_not_decoded')
                    slots.append(slot)
                    continue
                row = rows[(sha, obj['id'], frame)]
                slot.update(source_time_s=row['time_s'], body_time_s=_clock(human_source, frame),
                            observation_id=row['id'], partition=row['partition'], admission=row['admission'],
                            visibility=row['visibility'], evidence_refs=_thaw(row['evidence_refs']))
                if ref is None:
                    slot['unavailable_reasons'].append('body_reference_unavailable')
                if row['admission'] != 'included':
                    slot['unavailable_reasons'].append('partition_withheld')
                else:
                    slot['measurements'] = {name: _thaw(value) for name, value in row['measurements'].items() if name in ('box', 'center')}
                    if row['visibility'] not in ('detected', 'partial'):
                        slot['unavailable_reasons'].append('source_visibility_unavailable')
                    if not slot['measurements']:
                        slot['unavailable_reasons'].append('image_geometry_unavailable')
                    if sha in raw_evidence:
                        for evidence_ref in row['evidence_refs']:
                            phase = _phase(raw_evidence[sha][evidence_ref['evidence_id']], row)
                            if phase is not None:
                                slot['phase_records'].append(phase)
                    if not slot['phase_records']:
                        slot['unavailable_reasons'].append('provider_phase_unavailable')
                slots.append(slot)
            tracks.append({'entity_ref': entity_ref, 'world_entity_ids': world_ids, 'slots': slots})
        result_sources.append({'document_sha256': sha, 'source': _thaw(source['source']), 'time': _thaw(source['time']),
                               'image': _thaw(source['image']), 'relationships': _thaw(source['relationships']), 'tracks': tracks})
    value = {'schema_version': 1, 'profile': PROFILE,
             'construction_evidence_sha256': hashlib.sha256(construction_evidence).hexdigest(),
             'source_binding_sha256': canonical_hash(source_binding), 'body_references_sha256': canonical_hash(body_references),
             'observation_document_sha256s': sorted(observation_documents), 'source_sha256': source_binding['source_sha256'],
             'decoded_frames': count, 'body_references': body_references, 'sources': result_sources, 'scope': SCOPE, 'limits': LIMITS}
    return _encode(value, MAX_OUTPUT_BYTES, deadline)


def prepare_motion_evidence(construction_evidence: bytes, *, source_binding, body_references, observation_documents, max_seconds=10):
    """Return immutable typed bytes; callers retain the exact original sealed input packages."""
    try:
        return _prepare_motion_evidence(construction_evidence, source_binding, body_references, observation_documents, max_seconds)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError('Moving equipment evidence: malformed bounded source input') from error


def validate_motion_evidence(receipt: bytes, **inputs):
    """Recompute from exact source inputs; a hash-only or shape-only receipt is insufficient."""
    _require(type(receipt) is bytes and 0 < len(receipt) <= MAX_OUTPUT_BYTES, 'bounded immutable receipt required')
    _require(receipt == prepare_motion_evidence(**inputs), 'saved ledger differs from exact source rederivation')
    return receipt
