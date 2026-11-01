"""Real selected-byte custody with the independent report validator stubbed explicitly."""

from hashlib import sha256
import json

import pytest

from test_camera_reference_inputs import packages, metadata_only
from test_shared_exercise_reference_inputs import reseal
from test_shared_exercise_reference_inputs import packages as legacy_packages
from test_whole_body_reference_report import encode
from weft import camera_reference_inputs as camera
from weft import camera_multiview_inputs as secondary
from weft import whole_body_reference_report as report
from weft import shared_exercise_reference_inputs as shared


def multiview_packages(tmp_path, monkeypatch):
    proposal, source, task, *_ = packages(tmp_path)
    seal = json.loads((proposal / 'manifest.json').read_bytes())
    docs = {name: json.loads((proposal / name).read_bytes()) for name in seal['files']
            if name != 'unselected-missing.json'}
    saved, plan = docs['report.json'], docs['plan.json']
    saved['profile'] = camera.MULTIVIEW_METHOD
    saved['multiview_images'] = {'scope': 'Report validator is a routing stub in this custody fixture'}
    docs['angle-proposal.json'].update(profile='camera-body-multiview-angle-proposal-v1',
                                     additional_view={'scope': 'Report validation is separately tested'})
    plan['proposal_objective'] = camera.MULTIVIEW_OBJECTIVE
    source_pin = docs['inputs/human.json']['source_sha256']
    other = 'f' * 64
    observation = {'scope': 'Authored custody input; no source reconstruction', 'source_sha256': other}
    observation_pin = sha256(encode(observation)).hexdigest()
    config = {'schema_version': 1, 'profile': secondary.PROFILE,
        'primary_source_sha256': source_pin, 'secondary_source_sha256': other,
        'observations': [{'path': 'authored-observation.json', 'sha256': observation_pin}],
        'world_alignment': None, 'correspondence': {
            'primary_entity_ref': {'document_sha256': 'd' * 64, 'object_id': 'person'},
            'secondary_entity_ref': {'document_sha256': observation_pin, 'object_id': 'person'},
            'status': 'assumed', 'method': 'Authored test relation', 'uncertainty': 'Unverified'},
        'image_scale': .01, 'weight': 2.}
    docs.update({secondary.CONFIG_MEMBER: config, secondary.CAMERA_MEMBER: {'source_sha256': other, 'fixture': 'camera'},
        secondary.EVIDENCE_MEMBER: {'source_sha256': other, 'fixture': 'pose'},
        'inputs/camera-secondary-observation-00.json': observation})
    files = {name: encode(value) for name, value in docs.items()}
    descriptor = secondary._descriptor(config, files[secondary.CONFIG_MEMBER], files)
    plan['secondary_view'] = descriptor
    names = secondary.camera_multiview_member_names(descriptor)
    plan['input_files'].update({name: sha256(files[name]).hexdigest() for name in names})
    proof, body_manifest = docs['inputs/camera-inputs-proof.json'], docs['inputs/body-manifest.json']
    for original, alias in camera._SECONDARY_ALIASES.items():
        raw = files[alias]
        proof['selected_members'][original] = {'bytes': len(raw), 'sha256': sha256(raw).hexdigest(), 'copied_member': alias}
        body_manifest['files'][original] = sha256(raw).hexdigest()
        path = source / original; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    for name in {*report._CAMERA_EXTRAS, *camera._SECONDARY_ALIASES.values()}:
        plan['source']['copied_files'][name] = sha256(encode(docs[name])).hexdigest()
    for name, value in docs.items():
        path = proposal / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(encode(value))
    seal['files'] = {name: sha256(encode(value)).hexdigest() for name, value in docs.items()}
    (proposal / 'manifest.json').write_bytes(encode(seal))
    original_seal = json.loads((source / 'manifest.json').read_bytes())
    original_seal['files'].update({name: sha256(files[alias]).hexdigest() for name, alias in camera._SECONDARY_ALIASES.items()})
    (source / 'manifest.json').write_bytes(encode(original_seal))
    calls = []
    def model(documents, hashes, check):
        check(); calls.append((set(documents), dict(hashes)))
        return {'source_sha256': source_pin, 'source_references_without_geometry': 0, 'grid': [True]}
    monkeypatch.setattr(report, '_model', model)
    return proposal, source, task, descriptor, calls


def test_selected_secondary_bytes_bind_original_source_without_inventing_body(tmp_path, monkeypatch):
    proposal, source, task, descriptor, calls = multiview_packages(tmp_path, monkeypatch)
    value = camera.load_camera_reference_inputs(proposal, source, task)
    receipt = value.receipt()
    assert receipt['profile'] == camera.MULTIVIEW_PROFILE and len(calls) == 1
    assert secondary.camera_multiview_member_names(descriptor) <= calls[0][0]
    assert value.identities['secondary_camera_sha256'] == descriptor['camera_sha256']
    assert value.identities['secondary_observation_sha256s'] == descriptor['observation_sha256s']
    assert value.original_human_bytes == (source / 'human.json').read_bytes()
    assert receipt['input_bytes_read'] == 2 * sum(row['bytes'] for row in receipt['selected_files'])
    assert not hasattr(value, 'candidate_human_bytes') and receipt['body_installed'] is False
    for original, alias in camera._SECONDARY_ALIASES.items():
        assert value.proposal_members[alias] == (source / original).read_bytes()


@pytest.mark.parametrize('kind', ['missing_selected', 'missing_source', 'raw_source_change',
    'proof_rebind', 'descriptor', 'old_method', 'aggregate'])
def test_multiview_custody_rejects_resealed_missing_or_changed_inputs(tmp_path, monkeypatch, kind):
    proposal, source, task, descriptor, calls = multiview_packages(tmp_path, monkeypatch)
    options = {}
    if kind == 'missing_selected':
        seal = json.loads((proposal / 'manifest.json').read_bytes())
        del seal['files']['inputs/camera-secondary-observation-00.json']
        (proposal / 'manifest.json').write_bytes(encode(seal))
    elif kind == 'missing_source':
        seal = json.loads((source / 'manifest.json').read_bytes())
        del seal['files']['inputs/two-view-root/evidence.json']
        (source / 'manifest.json').write_bytes(encode(seal))
    elif kind == 'raw_source_change':
        name = 'inputs/two-view-root/camera.json'
        reseal(source, name, (source / name).read_bytes() + b' ')
    elif kind == 'proof_rebind':
        name = 'inputs/camera-inputs-proof.json'
        proof = json.loads((proposal / name).read_bytes())
        proof['selected_members']['inputs/two-view-root/camera.json']['sha256'] = '0' * 64
        reseal(proposal, name, proof)
        plan = json.loads((proposal / 'plan.json').read_bytes())
        plan['source']['copied_files'][name] = sha256(encode(proof)).hexdigest()
        reseal(proposal, 'plan.json', plan)
    elif kind == 'descriptor':
        plan = json.loads((proposal / 'plan.json').read_bytes()); plan['secondary_view']['weight'] = 3.
        reseal(proposal, 'plan.json', plan)
    elif kind == 'old_method':
        saved = json.loads((proposal / 'report.json').read_bytes()); saved['profile'] = report.CAMERA_METHOD
        reseal(proposal, 'report.json', saved)
    else: options['max_input_bytes'] = 1024
    with pytest.raises(ValueError): camera.load_camera_reference_inputs(proposal, source, task, **options)
    if kind not in {'missing_source', 'raw_source_change'}: assert not calls


@pytest.mark.parametrize('family', ['multiview', 'single_camera', 'legacy', 'mixed', 'failed_gate'])
def test_extra_read_allowance_requires_passing_sealed_multiview_family(tmp_path, monkeypatch, family):
    if family == 'legacy':
        proposal, source, task, *_ = legacy_packages(tmp_path)
    elif family == 'single_camera':
        proposal, source, task, *_ = packages(tmp_path)
    else:
        proposal, source, task, *_ = multiview_packages(tmp_path, monkeypatch)
    if family == 'mixed':
        path = proposal / 'manifest.json'; seal = json.loads(path.read_bytes())
        seal['files']['candidate-human.json'] = '0' * 64; path.write_bytes(encode(seal))
    elif family == 'failed_gate':
        saved = json.loads((proposal / 'report.json').read_bytes())
        saved['gates']['camera_projection_domain'] = False
        reseal(proposal, 'report.json', saved)
    first = (proposal / 'manifest.json').stat().st_size
    classified = first + (proposal / 'report.json').stat().st_size
    reads, reader = [], shared.read_package_file
    def read(directory, name, **kwargs):
        reads.append(name); return reader(directory, name, **kwargs)
    monkeypatch.setattr(shared, 'read_package_file', read)
    options = dict(allow_camera=True, max_input_bytes=2 * (first + 512),
                   camera_max_input_bytes=2 * (classified + 1), multiview_max_input_bytes=2 << 20)
    if family == 'multiview':
        value = shared.load_reference_inputs(proposal, source, task, **options)
        receipt = value.receipt()
        assert receipt['profile'] == camera.MULTIVIEW_PROFILE
        assert receipt['maximum_input_bytes'] == 2 << 20
        assert receipt['input_bytes_read'] == 2 * sum(row['bytes'] for row in receipt['selected_files'])
    else:
        with pytest.raises(ValueError): shared.load_reference_inputs(proposal, source, task, **options)
        if family == 'mixed': assert reads == ['manifest.json']
        if family == 'failed_gate': assert reads == ['manifest.json', 'report.json']


@pytest.mark.parametrize('options', [
    {'multiview_max_input_bytes': 80 << 20},
    {'allow_camera': True, 'multiview_max_input_bytes': (80 << 20) + 1},
])
def test_multiview_cap_rejects_invalid_options_before_reads(monkeypatch, options):
    monkeypatch.setattr(shared, 'read_package_file', lambda *a, **k: pytest.fail('invalid cap read an input'))
    with pytest.raises(ValueError, match='bound'):
        shared.load_reference_inputs('unused', 'unused', 'unused', **options)
