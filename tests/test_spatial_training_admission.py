from copy import deepcopy
import json
from types import SimpleNamespace

import mujoco
import pytest

from weft import spatial_lab as lab
from weft import spatial_learning as learning
from weft.io import canonical_hash, digest, write_json


class Admitted(Exception):
    pass


@pytest.fixture(autouse=True)
def no_native_or_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Admission tests must stop before native construction, rollout or training')

    monkeypatch.setattr(mujoco, 'MjModel', SimpleNamespace(from_xml_string=forbidden, from_xml_path=forbidden))
    for name in ('MjData', 'mj_step', 'mj_forward', 'mj_resetData'):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(lab, 'SpatialRigidEnv', forbidden)
    monkeypatch.setattr(lab, '_probe_controls', forbidden)
    monkeypatch.setattr(learning, 'SpatialRigidEnv', forbidden)
    monkeypatch.setattr(learning, '_rollout_episode', forbidden)


def seal(root, *, manifest=None, excluded=()):
    document = {'schema_version': 1, **(manifest or {}), 'files': {
        path.relative_to(root).as_posix(): digest(path) for path in root.rglob('*')
        if path.is_file() and path != root / 'manifest.json' and path.relative_to(root).as_posix() not in excluded}}
    write_json(root / 'manifest.json', document)


def bundle(tmp_path, *, status='constructed_hypothesis'):
    root = tmp_path / 'source'; root.mkdir()
    scene = lab.authored_spatial_scene()
    candidate = {'scene': deepcopy(scene), 'physical_validation': 'unverified', 'physics_ready': False}
    report = {'candidate_sha256': canonical_hash(candidate), 'scene_sha256': canonical_hash(scene),
              'physical_validation': 'unverified', 'physics_ready': False}
    if status is not None:
        candidate['status'] = status
        report.update(status=status, candidate_sha256=canonical_hash(candidate))
    write_json(root / 'scene.json', scene)
    write_json(root / 'candidate.json', candidate)
    write_json(root / 'report.json', report)
    seal(root, manifest={'profile': 'spatial-scene-construction-v1', 'status': status,
                         'simulation_ready': False, 'physical_validation': 'unverified'} if status else None)
    return root, scene


def call(route, source, out):
    if route == 'training':
        return learning.train_spatial_policy(source, out, generations=1, population=4, eval_episodes=4, max_seconds=1)
    return lab.prepare_spatial_lab(source, out)


def change(root, name, **updates):
    document = json.loads((root / name).read_bytes())
    document.update(updates)
    write_json(root / name, document)


@pytest.mark.parametrize('route', ['training', 'lab'])
@pytest.mark.parametrize('kind', ['unsealed_scene', 'tampered_scene', 'unsealed_report', 'unsealed_candidate',
                                 'dangling_report', 'failed_manifest', 'incomplete_manifest', 'failed_report',
                                 'partial_report', 'prepared_report', 'failed_candidate', 'running_candidate',
                                 'missing_candidate', 'candidate_hash', 'candidate_scene', 'report_scene_hash',
                                 'candidate_scene_hash', 'false_physical_claim', 'false_physics_ready',
                                 'unknown_status'])
def test_unadmitted_sources_stop_before_any_output_or_model(tmp_path, route, kind):
    source, scene = bundle(tmp_path)
    out = tmp_path / 'output'
    if kind.startswith('unsealed_'):
        seal(source, excluded=(kind.removeprefix('unsealed_') + '.json',))
    elif kind == 'tampered_scene':
        change(source, 'scene.json', target=[.5, .5, .5])
    elif kind == 'dangling_report':
        (source / 'report.json').unlink()
        seal(source)
        (source / 'report.json').symlink_to('missing.json')
    elif kind in {'failed_manifest', 'incomplete_manifest'}:
        seal(source, manifest={'status': kind.removesuffix('_manifest')})
    else:
        if kind in {'failed_report', 'partial_report', 'prepared_report'}:
            change(source, 'report.json', status=kind.removesuffix('_report'))
        elif kind in {'failed_candidate', 'running_candidate'}:
            change(source, 'candidate.json', status=kind.removesuffix('_candidate'))
        elif kind == 'missing_candidate':
            (source / 'candidate.json').unlink()
        elif kind == 'candidate_hash':
            change(source, 'report.json', candidate_sha256='a' * 64)
        elif kind == 'candidate_scene':
            other_scene = {**scene, 'target': [.5, .5, .5]}
            change(source, 'candidate.json', scene=other_scene)
            change(source, 'report.json', candidate_sha256=canonical_hash(json.loads((source / 'candidate.json').read_bytes())))
        elif kind == 'report_scene_hash':
            change(source, 'report.json', scene_sha256='a' * 64)
        elif kind == 'candidate_scene_hash':
            change(source, 'candidate.json', scene_sha256='a' * 64)
        elif kind == 'false_physical_claim':
            change(source, 'report.json', physical_validation='verified')
        elif kind == 'false_physics_ready':
            change(source, 'candidate.json', physics_ready=True)
        else:
            change(source, 'candidate.json', status='unrecognized_stage')
        seal(source)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file() and not p.is_symlink()}
    with pytest.raises(ValueError):
        call(route, source, out)
    assert not out.exists()
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file() and not p.is_symlink()}


@pytest.mark.parametrize('route', ['training', 'lab'])
@pytest.mark.parametrize('kind', ['sealed', 'unsealed', 'dangling'])
def test_current_failure_marker_is_never_rescued_by_completed_metadata(tmp_path, route, kind):
    source, _ = bundle(tmp_path)
    if kind == 'dangling':
        (source / 'failure.json').symlink_to('missing.json')
    else:
        write_json(source / 'failure.json', {'status': 'failed', 'error': 'Preserved finalization failure'})
        if kind == 'sealed':
            seal(source)
    out = tmp_path / 'output'
    with pytest.raises(ValueError, match='failure.json'):
        call(route, source, out)
    assert not out.exists()


@pytest.mark.parametrize('route', ['training', 'lab'])
@pytest.mark.parametrize('kind', ['legacy_minimal', 'legacy_fixture', 'legacy_candidate', 'ramp_hypothesis',
                                 'constructed_hypothesis', 'nested_failed_alternative'])
def test_known_complete_and_legacy_sources_keep_the_exact_scene(tmp_path, monkeypatch, route, kind):
    source, scene = bundle(tmp_path, status=None if kind.startswith('legacy') else 'constructed_hypothesis')
    if kind in {'legacy_minimal', 'legacy_fixture', 'ramp_hypothesis'}:
        (source / 'candidate.json').unlink()
        (source / 'report.json').unlink()
        if kind != 'legacy_minimal':
            write_json(source / 'report.json', {'physical_validation': 'unverified', 'physics_ready': False,
                'status': 'authored_analytic_fixture' if kind == 'legacy_fixture' else 'frozen_source_linked_spatial_hypothesis'})
        seal(source)
    elif kind == 'nested_failed_alternative':
        (source / 'inputs/rejected').mkdir(parents=True)
        write_json(source / 'inputs/rejected/failure.json', {'status': 'failed'})
        write_json(source / 'inputs/rejected/report.json', {'status': 'partial'})
        seal(source)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'output'

    def admitted(actual):
        assert actual == scene
        if route == 'training':
            settings = json.loads((out / 'settings.json').read_bytes())
            assert settings['source_provenance'] == {'source_directory': str(source),
                'source_manifest_sha256': digest(source / 'manifest.json'),
                'source_scene_sha256': digest(source / 'scene.json')}
            assert json.loads((out / 'scene.json').read_bytes()) == scene
            assert not (out / 'policy.json').exists()
        else:
            assert not out.exists()
        raise Admitted()

    monkeypatch.setattr(learning if route == 'training' else lab,
                        'SpatialRigidEnv' if route == 'training' else '_probe_controls', admitted)
    with pytest.raises(Admitted):
        call(route, source, out)
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_raw_authored_scene_training_stays_explicit_and_copies_without_mutation(tmp_path, monkeypatch):
    scene = lab.authored_spatial_scene()
    original = deepcopy(scene)
    out = tmp_path / 'authored-training'

    def admitted(actual):
        assert actual == original and actual is not scene
        settings = json.loads((out / 'settings.json').read_bytes())
        assert settings['source_provenance'] is None
        assert settings['scene_sha256'] == canonical_hash(original)
        raise Admitted()

    monkeypatch.setattr(learning, 'SpatialRigidEnv', admitted)
    with pytest.raises(Admitted):
        call('training', scene, out)
    assert scene == original


@pytest.mark.parametrize('route', ['training', 'lab'])
@pytest.mark.parametrize('kind', ['scene', 'failure_marker'])
def test_changed_scene_after_bundle_verification_is_not_read_unsealed(tmp_path, monkeypatch, route, kind):
    source, _ = bundle(tmp_path)
    out = tmp_path / 'output'
    actual = lab._read_sealed_bundle

    def changed(*args, **kwargs):
        result = actual(*args, **kwargs)
        if kind == 'scene':
            change(source, 'scene.json', target=[.5, .5, .5])
        else:
            write_json(source / 'failure.json', {'status': 'failed'})
        return result

    monkeypatch.setattr(lab, '_read_sealed_bundle', changed)
    with pytest.raises(ValueError, match='changed after verification' if kind == 'scene' else 'failure.json'):
        call(route, source, out)
    assert not out.exists()
