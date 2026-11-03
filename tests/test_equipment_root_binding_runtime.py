"""Versioned root transfer preserves legacy rejection and exact sealed input custody."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_equipment_root_binding import fixture
from weft.equipment_root_binding import derive_body_binding, proof_hashes
from weft import shared_world_reconstruction as reconstruction
from weft.shared_world_pipeline import read_equipment_root_proof


def test_runtime_dispatch_preserves_all_original_roots_and_interpolation():
    human, raw, entity, indexed, documents, proof = fixture()
    binding, frames = derive_body_binding(human, raw, entity, indexed, documents, root_proof=proof)
    before = deepcopy((human, binding, indexed, documents, proof))
    scale, origin, actual = reconstruction._body_mapping(
        human, {'body_binding': binding}, entity, indexed, documents, root_proof=proof)
    assert scale == .005 and origin.tolist() == [320., 240.] and actual == frames == list(range(10))
    assert [r['status'] for r in binding['reference_derivations']].count('interpolated_reference_only') == 2
    assert (human, binding, indexed, documents, proof) == before
    human['root_reference'][4]['translation'][0] += .001
    with pytest.raises(ValueError, match='translation'):
        reconstruction._body_mapping(human, {'body_binding': binding}, entity, indexed, documents, root_proof=proof)


@pytest.mark.parametrize('missing', [(), (4, 5)])
def test_legacy_mapping_keeps_direct_observation_gate(missing):
    human, raw, entity, indexed, documents, proof = fixture(missing=missing)
    binding, _ = derive_body_binding(human, raw, entity, indexed, documents, root_proof=proof)
    for name in ('profile', 'root_proof_sha256', 'reference_derivations'):
        binding.pop(name)
    if missing:
        with pytest.raises(ValueError, match='Torso scale is degenerate'):
            reconstruction._body_mapping(human, {'body_binding': binding}, entity, indexed, documents)
    else:
        scale, origin, frames = reconstruction._body_mapping(human, {'body_binding': binding}, entity, indexed, documents)
        assert scale == .005 and origin.tolist() == [320., 240.] and frames == list(range(10))
    with pytest.raises(ValueError, match='Legacy'):
        reconstruction._body_mapping(human, {'body_binding': binding}, entity, indexed, documents, root_proof=proof)


@pytest.mark.parametrize('damage', ['missing_proof', 'unknown_profile', 'changed_scale'])
def test_versioned_dispatch_cannot_fallback_or_rewrite_scale(damage):
    human, raw, entity, indexed, documents, proof = fixture()
    binding, _ = derive_body_binding(human, raw, entity, indexed, documents, root_proof=proof)
    if damage == 'missing_proof':
        proof = None
    elif damage == 'unknown_profile':
        binding['profile'] = 'unreviewed-v3'
    else:
        binding['root_scale_unchanged']['model_units_per_pixel'] *= 1.001
    with pytest.raises(ValueError):
        reconstruction._body_mapping(human, {'body_binding': binding}, entity, indexed, documents, root_proof=proof)


def test_public_wrapper_passes_exact_proof_without_mutation(monkeypatch):
    proof = {'root-motion.json': b'audit', 'reconstruction-evidence.json': b'evidence'}
    def construct(*args, **kwargs):
        assert kwargs['root_proof'] is proof
        return {'fixture': True}
    monkeypatch.setattr(reconstruction, '_construct_equipment_world', construct)
    assert reconstruction.construct_equipment_world({}, {}, source_binding={}, construction_evidence=b'',
        human_document=b'', target_object_id='equipment', root_proof=proof) == {'fixture': True}


def retained(tmp_path, layout='seed'):
    proof = {'root-motion.json': b'{"audit":1}', 'reconstruction-evidence.json': b'{"pose":1}'}
    original = {'profile': 'exercise-body-source-v1', 'status': 'construction_only',
                'simulation_ready': False, 'files': {**proof_hashes(proof), 'human.json': 'a'*64}}
    prefix = 'inputs/human-source/' if layout == 'seed' else ''
    manifest_name = 'inputs/human-source/source-manifest.json' if layout == 'seed' else 'inputs/body-source-manifest.json'
    contents = {manifest_name: json.dumps(original).encode(), 'root-motion.json': proof['root-motion.json'],
                prefix+'reconstruction-evidence.json': proof['reconstruction-evidence.json'], prefix+'native-clock.json': b'{}'}
    for name, raw in contents.items():
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return proof, {name: sha256(raw).hexdigest() for name, raw in contents.items()}, manifest_name


@pytest.mark.parametrize('layout', ['seed', 'human_only'])
def test_loader_retains_exact_original_manifest_join(tmp_path, layout):
    proof, declared, manifest = retained(tmp_path, layout)
    seen = []
    def read(name, maximum):
        raw = (tmp_path/name).read_bytes()
        assert len(raw) <= maximum
        seen.append(name)
        return raw
    actual = read_equipment_root_proof(tmp_path, declared, expected=proof_hashes(proof), human_sha256='a'*64, read_file=read)
    assert actual == proof and len(seen) == 3 and seen[0] == manifest


@pytest.mark.parametrize('damage', ['missing', 'changed', 'outer_pin', 'original_pin', 'proposal_pin',
                                   'unsealed_manifest', 'symlink', 'ambiguous', 'unexpected_member'])
def test_loader_rejects_missing_tampered_or_unsealed_proof(tmp_path, damage):
    proof, declared, manifest_name = retained(tmp_path)
    expected = proof_hashes(proof)
    if damage == 'missing':
        declared.pop('root-motion.json')
    elif damage == 'changed':
        (tmp_path/'root-motion.json').write_bytes(b'{}')
    elif damage == 'outer_pin':
        declared['root-motion.json'] = 'f'*64
    elif damage == 'original_pin':
        manifest = json.loads((tmp_path/manifest_name).read_bytes())
        manifest['files']['root-motion.json'] = 'f'*64
        raw = json.dumps(manifest).encode()
        (tmp_path/manifest_name).write_bytes(raw)
        declared[manifest_name] = sha256(raw).hexdigest()
    elif damage == 'proposal_pin':
        expected['root-motion.json'] = 'f'*64
    elif damage == 'unsealed_manifest':
        declared.pop(manifest_name)
    elif damage == 'symlink':
        (tmp_path/'root-motion.json').rename(tmp_path/'retained-root.json')
        (tmp_path/'root-motion.json').symlink_to(tmp_path/'retained-root.json')
    elif damage == 'ambiguous':
        declared['reconstruction-evidence.json'] = expected['reconstruction-evidence.json']
    else:
        expected['extra.json'] = 'f'*64
    with pytest.raises(ValueError):
        read_equipment_root_proof(tmp_path, declared, expected=expected, human_sha256='a'*64)


def test_loader_legacy_omission_and_versioned_missing_proof_are_distinct(tmp_path):
    assert read_equipment_root_proof(tmp_path, {}) is None
    with pytest.raises(ValueError, match='native body layout'):
        read_equipment_root_proof(tmp_path, {}, expected={name: 'a'*64 for name in ('root-motion.json', 'reconstruction-evidence.json')})


def test_custom_reader_cannot_substitute_manifest_or_exceed_proof_budget(tmp_path, monkeypatch):
    proof, declared, _ = retained(tmp_path)
    with pytest.raises(ValueError, match='outer seal'):
        read_equipment_root_proof(tmp_path, declared, human_sha256='a'*64, read_file=lambda *_: b'{}')
    import weft.equipment_root_binding as helper
    monkeypatch.setattr(helper, 'MAX_PROOF_BYTES', sum(map(len, proof.values()))-1)
    with pytest.raises(ValueError):
        read_equipment_root_proof(tmp_path, declared, human_sha256='a'*64)


@pytest.mark.parametrize('pin', [None, 'a'*63, 'b'*64])
def test_root_proof_cannot_bind_another_original_human(tmp_path, pin):
    _, declared, _ = retained(tmp_path)
    with pytest.raises(ValueError):
        read_equipment_root_proof(tmp_path, declared, human_sha256=pin)
