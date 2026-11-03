"""Native source transfer preserves bounded gaps, not measured motion or contact."""

from copy import deepcopy
import hashlib
import json

import pytest

from test_root_motion import evidence
from weft.equipment_root_binding import derive_body_binding, validate_body_mapping
from weft.io import canonical_hash
from weft.observations import adapt_exercise
from weft.root_motion import build_root_reference
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_reconstruction import _construction_index
from weft.source_clock import PROFILE, clock_summary, derive_evidence


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


def fixture(*, missing=(4, 5), fps=20, identity_raw=False):
    original = evidence(dx=2, fps=fps)
    original['decoded_frames'] = 10
    for frame in original['frames']:
        frame['decoder_timestamp_s'] = frame['frame'] / fps
    for index in missing:
        for point in (11, 12):
            original['frames'][index]['landmarks'][point]['y'] = .49
    if identity_raw:
        for frame in original['frames']:
            frame['identity_raw'] = deepcopy(frame)
    digest = hashlib.sha256(raw(original)).hexdigest()
    native = {'schema_version': 1, 'profile': PROFILE, 'source_sha256': original['source_sha256'],
        'evidence_sha256': digest, 'evidence_canonical_sha256': canonical_hash(original),
        'metadata_frames': 10, 'decoded_frames': 10, 'nominal_fps': fps, 'time_base': '1/1000',
        'native_origin_pts': 0, 'native_to_reconstruction_offset_s': 0., 'decoder_minus_native_s': 0.,
        'capture_clock_status': 'unknown', 'detector_clock_status': 'preserved_nominal_frame_index_divided_by_fps',
        'rows': [{'frame': i, 'pts': round(i * 1000 / fps), 'decoder_timestamp_s': i / fps} for i in range(10)]}
    view = derive_evidence(original, native)
    audit = build_root_reference(view)
    human = {'profile': 'exercise-v3', 'source_sha256': original['source_sha256'], 'native_clock': clock_summary(native),
        'lengths': original['lengths'], 'mobility': {'floor_height': 0},
        'reference_evidence': deepcopy(view['reference']), 'reference': [{'time_s': r['time_s']} for r in view['reference']],
        'root_reference': [{'time_s': r['time_s'], 'translation': r['translation']} for r in audit['reference']]}
    document = raw(adapt_exercise(view)); sha = hashlib.sha256(document).hexdigest()
    binding, construction = build_shared_world_binding({sha: document}, source_sha256=original['source_sha256'],
        world_ids=['human'], entity_refs={'human': (sha, 'person-0')})
    indexed, documents = _construction_index(construction, binding)
    return human, raw(human), binding['entity_bindings'][0], indexed, documents, {
        'reconstruction-evidence.json': raw(view), 'root-motion.json': raw(audit)}


def test_original_qualified_scale_and_all_roots_are_preserved_without_mutation():
    human, human_raw, entity, indexed, documents, proof = fixture()
    before = deepcopy((human, entity, indexed, documents, proof))
    contract, frames = derive_body_binding(human, human_raw, entity, indexed, documents, root_proof=proof)
    assert contract['root_scale_unchanged']['samples'] == 8
    assert frames == list(range(10))
    assert [r['status'] for r in contract['reference_derivations']].count('interpolated_reference_only') == 2
    assert all(r['observed_status'] == 'degenerate_torso_projection' for r in contract['reference_derivations'][4:6])
    scale, _, checked = validate_body_mapping(human, contract, entity, indexed, documents, root_proof=proof)
    assert scale == .005 and checked == frames
    assert (human, entity, indexed, documents, proof) == before


@pytest.mark.parametrize('mutation', ['scale', 'subset', 'derivation', 'extra', 'proof_hash', 'reference_status', 'boolean'])
def test_closed_binding_rejects_changed_recipe(mutation):
    h, raw_h, ent, idx, docs, proof = fixture()
    binding, _ = derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)
    if mutation == 'scale': binding['root_scale_unchanged']['model_units_per_pixel'] *= 1.01
    elif mutation == 'subset': binding['root_scale_unchanged']['evidence_ids'].pop()
    elif mutation == 'derivation': binding['reference_derivations'][4]['derivation_evidence_ids'].reverse()
    elif mutation == 'extra': binding['unreviewed'] = True
    elif mutation == 'proof_hash': binding['root_proof_sha256']['root-motion.json'] = 'b' * 64
    elif mutation == 'boolean': binding['preserve_all_source_references_and_clocks'] = 1
    else: binding['reference_derivations'][4]['status'] = 'observed_image_estimate'
    with pytest.raises(ValueError): validate_body_mapping(h, binding, ent, idx, docs, root_proof=proof)


@pytest.mark.parametrize('mutation', ['human_root', 'human_time', 'source_time', 'measurement', 'withheld', 'native', 'audit', 'floor'])
def test_original_source_and_body_joins_cannot_be_rewritten(mutation):
    h, raw_h, ent, idx, docs, proof = fixture()
    row = idx[(ent['observation_document_sha256'], ent['observation_entity_id'], 4)]
    if mutation == 'human_root': h['root_reference'][4]['translation'][0] += .001
    elif mutation == 'human_time': h['reference'][4]['time_s'] += .001
    elif mutation == 'source_time': row['time_s'] += .001
    elif mutation == 'measurement': row['measurements']['landmarks']['value'][11]['position'][1] += 1
    elif mutation == 'withheld': row['partition'] = 'final_holdout'
    elif mutation == 'native': h['native_clock']['binding_sha256'] = 'f' * 64
    elif mutation == 'floor': h['mobility']['floor_height'] = .1
    else:
        audit = json.loads(proof['root-motion.json']); audit['reference'][4]['status'] = 'observed_image_estimate'
        proof['root-motion.json'] = raw(audit)
    with pytest.raises(ValueError): derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)


@pytest.mark.parametrize('missing,fps', [((3, 4, 5), 20), ((4, 5), 10), ((0,), 20)])
def test_original_gap_and_first_origin_gates_remain_unchanged(missing, fps):
    h, raw_h, ent, idx, docs, proof = fixture(missing=missing, fps=fps)
    with pytest.raises(ValueError, match='original ready source recipe'):
        derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)


def test_proof_aggregate_bound_and_exact_byte_identity():
    h, raw_h, ent, idx, docs, proof = fixture()
    binding, _ = derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)
    proof['root-motion.json'] += b' '
    with pytest.raises(ValueError, match='byte identity'):
        validate_body_mapping(h, binding, ent, idx, docs, root_proof=proof)
    proof['root-motion.json'] = b' ' * (20 * 1024 * 1024)
    with pytest.raises(ValueError, match='20 MiB'):
        derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)


def test_pose_count_bound_precedes_measurement_walk():
    h, raw_h, ent, idx, docs, proof = fixture()
    view = json.loads(proof['reconstruction-evidence.json'])
    view['frames'] = [{}] * 901
    proof['reconstruction-evidence.json'] = raw(view)
    with pytest.raises(ValueError, match='900 original pose samples'):
        derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)


def test_only_known_native_degeneracy_selects_versioned_transfer():
    from weft.equipment_proposal import _needs_root_transfer
    h, raw_h, ent, idx, docs, _ = fixture()
    assert _needs_root_transfer(h, raw_h, ent, idx, docs) is True
    del h['native_clock']
    with pytest.raises(ValueError, match='Torso scale is degenerate'):
        _needs_root_transfer(h, raw_h, ent, idx, docs)


def test_previously_admitted_identity_recomputed_native_body_keeps_legacy_path():
    from weft.equipment_proposal import _needs_root_transfer
    h, raw_h, ent, idx, docs, proof = fixture(missing=(), identity_raw=True)
    assert _needs_root_transfer(h, raw_h, ent, idx, docs) is False
    with pytest.raises(ValueError, match='Identity-recomputed'):
        derive_body_binding(h, raw_h, ent, idx, docs, root_proof=proof)


def test_unrelated_legacy_source_failure_cannot_trigger_v2():
    from weft.equipment_proposal import _needs_root_transfer
    h, raw_h, ent, idx, docs, _ = fixture()
    idx[(ent['observation_document_sha256'], ent['observation_entity_id'], 0)]['partition'] = 'final_holdout'
    with pytest.raises(ValueError, match='unavailable or withheld'):
        _needs_root_transfer(h, raw_h, ent, idx, docs)
