import json

import pytest

from weft import spatial_scene_pipeline as pipeline
from weft.io import canonical_hash
from weft.observations import adapt_tracks
from test_observations import planar as source_observations
from test_spatial_scene_inputs import bundle, encoded, identity, prohibit_simulation, replace_member


RUNTIME_FILE_SET = frozenset(pipeline.RUNTIME_FILES)


@pytest.fixture(autouse=True)
def bounded_runtime(tmp_path, monkeypatch):
    root = tmp_path / 'trusted-runtime'; root.mkdir()
    (root / 'constructor.py').write_bytes(b'# frozen fixture constructor\n')
    (root / 'validator.py').write_bytes(b'# frozen fixture validator\n')
    (root / 'assets').mkdir()
    (root / 'assets/palette.json').write_bytes(b'{"colors":{"ink":"#214F50"}}\n')
    (root / 'avatar_replay.js').write_bytes(b'Unrelated asset must not enter the package')
    monkeypatch.setattr(pipeline, 'RUNTIME_ROOT', root)
    monkeypatch.setattr(pipeline, 'RUNTIME_FILES', ('constructor.py', 'validator.py', 'assets/palette.json'))
    monkeypatch.setattr(pipeline, 'version', lambda name: 'fixture-version')
    return root


def case(tmp_path, *, nested=False):
    source, scene_artifact, config, _ = bundle(tmp_path, nested=nested, legacy=nested)
    config_path = tmp_path / 'authored.json'; config_path.write_bytes(config)
    return source, scene_artifact, config_path, tmp_path / 'output'


def run(source, artifact, config, out, **options):
    return pipeline.prepare_spatial_scene(source, config, out, scene_artifact=artifact,
        expected_manifest_sha256=identity((source / 'manifest.json').read_bytes()), **options)


def checked_manifest(out):
    manifest = json.loads((out / 'manifest.json').read_bytes())
    actual = {path.relative_to(out).as_posix(): identity(path.read_bytes())
              for path in out.rglob('*') if path.is_file() and path != out / 'manifest.json'}
    assert manifest['files'] == actual
    return manifest


def test_runtime_snapshot_includes_the_imported_palette_but_no_avatar_assets(tmp_path, bounded_runtime):
    assert RUNTIME_FILE_SET == {'__init__.py', 'spatial_scene_pipeline.py', 'spatial_scene_inputs.py',
        'scene.py', 'spatial_rigid.py', 'brand.py', 'assets/palette.json', 'io.py',
        'shared_world_packaging.py', 'shared_world_evidence.py', 'world_evidence.py',
        'source_equivalence.py', 'observations.py'}
    source, artifact, config, out = case(tmp_path)
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis'
    assert (out / 'runtime/weft/assets/palette.json').read_bytes() == (bounded_runtime / 'assets/palette.json').read_bytes()
    assert not (out / 'runtime/weft/avatar_replay.js').exists()
    assert result['runtime_sha256']['assets/palette.json'] == identity((bounded_runtime / 'assets/palette.json').read_bytes())


@pytest.mark.parametrize('nested', [False, True])
def test_candidate_package_archives_every_byte_without_installing_planar_policy(tmp_path, nested):
    source, artifact, config, out = case(tmp_path, nested=nested)
    replace_member(source, 'policy.json', {'fixture': 'old planar policy must remain archived'})
    replace_member(source, 'training.json', {'episodes': 10})
    replace_member(source, 'rl-evaluation.json', {'successes': 8})
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    config_before = config.read_bytes()
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis' and result['profile'] == 'spatial-rigid-v1'
    assert result['simulation_ready'] is False and result['physics_ready'] is False
    assert result['learned_policy_available'] is False and result['next_step'] == 'lab'
    assert result['source_archive_members'] == len(before) - 1
    assert result['source_archive_bytes'] == sum(map(len, before.values()))
    assert result['readback']['source_and_config_bytes'] == 3 * sum(map(len, before.values())) + 2 * len(config_before)
    assert result['readback']['source_and_config_bytes'] <= pipeline.MAX_INPUT_BYTES
    assert result['presentation']['source']['status'] == 'unavailable'
    assert all(row['status'] == 'unavailable' for row in result['presentation']['previews'])
    assert {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()} == before
    assert config.read_bytes() == config_before
    for name, raw in before.items():
        assert (out / 'inputs/planar' / name).read_bytes() == raw
    assert (out / 'inputs/original-scene.json').read_bytes() == before[artifact]
    assert (out / 'inputs/authored-config.json').read_bytes() == config_before
    for name in ('policy.json', 'training.json', 'rl-evaluation.json', 'source.webm'):
        assert not (out / name).exists()
    assert not list((out / 'runtime').rglob('avatar*'))
    scene = json.loads((out / 'scene.json').read_bytes())
    candidate = json.loads((out / 'candidate.json').read_bytes())
    report = json.loads((out / 'report.json').read_bytes())
    assert candidate['scene'] == scene and report['candidate_sha256'] == canonical_hash(candidate)
    assert result['scene_sha256'] == canonical_hash(scene)
    assert not {'policy', 'training', 'rl-evaluation'} & set(candidate)
    assert result['counts']['model_compilations'] == result['counts']['environment_constructions'] == result['counts']['physics_steps'] == 0
    manifest = checked_manifest(out)
    assert manifest['status'] == 'constructed_hypothesis'
    assert result['manifest_sha256'] == identity((out / 'manifest.json').read_bytes())
    assert result['output_files'] == len(manifest['files']) + 1
    assert result['output_bytes'] == sum(p.stat().st_size for p in out.rglob('*') if p.is_file())


@pytest.mark.parametrize('kind', ['ordinary_planar_legacy', 'explicit_standard', 'explicit_other_basename', 'selected_child'])
def test_original_media_uses_only_exact_known_binding_with_explicit_normalization(tmp_path, kind):
    source, artifact, config, out = case(tmp_path, nested=kind == 'selected_child')
    prefix = 'cases/Run_7/' if kind == 'selected_child' else ''
    original_name = 'bound-clip.MP4' if kind == 'explicit_other_basename' else 'source.avi'
    media = b'Exact source bytes: never decoded or re-encoded'
    replace_member(source, prefix + original_name, media)
    metadata = {'sha256': identity(media), 'original_name': 'rolling-ball-grayscale.avi',
                'rights': 'User-provided; not independently verified', 'scope': 'User-provided local input'}
    if kind != 'ordinary_planar_legacy':
        metadata['artifact'] = original_name
    raw_metadata = encoded(metadata)
    replace_member(source, prefix + 'source.json', raw_metadata)
    for name in ('tracks.gif', 'tracks-contact-sheet.png'):
        replace_member(source, prefix + name, b'Exact saved preview bytes; not decoded')
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis'
    published_name = 'source.mp4' if kind == 'explicit_other_basename' else 'source.avi'
    assert (out / published_name).read_bytes() == media
    assert json.loads((out / 'source.json').read_bytes()) == {**metadata, 'artifact': published_name}
    assert (out / 'inputs/planar' / (prefix + 'source.json')).read_bytes() == raw_metadata
    assert (out / 'inputs/planar' / (prefix + original_name)).read_bytes() == media
    presentation = result['presentation']['source']
    assert presentation['status'] == 'available' and presentation['sha256'] == identity(media)
    assert presentation['source_media_artifacts'] == [prefix + original_name]
    assert presentation['original_metadata_sha256'] == identity(raw_metadata)
    assert presentation['published_metadata_sha256'] == identity((out / 'source.json').read_bytes())
    changed = kind in ('ordinary_planar_legacy', 'explicit_other_basename')
    assert (presentation['metadata_derivative'] is not None) is changed
    assert all(row['status'] == 'available' for row in result['presentation']['previews'])
    checked_manifest(out)


@pytest.mark.parametrize('kind', ['unsafe_original', 'unknown_legacy', 'unsupported_suffix', 'missing_media',
                                 'wrong_hash', 'remote_artifact', 'ambiguous_sources'])
def test_media_mismatch_never_searches_for_a_fallback(tmp_path, kind):
    source, artifact, config, out = case(tmp_path, nested=kind == 'ambiguous_sources')
    media = b'Bound video'
    replace_member(source, 'source.avi', media)
    doc = {'sha256': identity(media), 'original_name': 'rolling-ball-grayscale.avi',
           'rights': 'User-provided', 'scope': 'User-provided local input'}
    if kind == 'unsafe_original':
        doc['original_name'] = '../rolling-ball-grayscale.avi'
    elif kind == 'unknown_legacy':
        doc['unknown'] = 'not the known schema'
    elif kind == 'unsupported_suffix':
        doc['original_name'] = 'source.txt'
    elif kind == 'missing_media':
        doc['original_name'] = 'source.mkv'
    elif kind == 'wrong_hash':
        doc['sha256'] = 'b' * 64
    elif kind == 'remote_artifact':
        doc['artifact'] = 'https://example.invalid/video.avi'
    else:
        other = b'Different explicitly bound source'
        replace_member(source, 'cases/Run_7/source.avi', other)
        replace_member(source, 'cases/Run_7/source.json', {**doc, 'sha256': identity(other)})
    replace_member(source, 'source.json', doc)
    result = run(source, artifact, config, out)
    assert result['status'] == 'failed' and result['next_step'] == 'inspect_saved_failure'
    assert result['stage'] == 'selecting_saved_presentation'
    assert not (out / 'source.json').exists() and not (out / 'source.avi').exists()
    assert checked_manifest(out)['status'] == 'failed'


def test_ambiguous_preview_remains_archived_without_top_level_selection(tmp_path):
    source, artifact, config, out = case(tmp_path, nested=True)
    replace_member(source, 'tracks.gif', b'Root preview')
    replace_member(source, 'cases/Run_7/tracks.gif', b'Child preview')
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis' and not (out / 'tracks.gif').exists()
    assert result['presentation']['previews'][0]['status'] == 'unavailable'
    assert (out / 'inputs/planar/tracks.gif').read_bytes() == b'Root preview'
    assert (out / 'inputs/planar/cases/Run_7/tracks.gif').read_bytes() == b'Child preview'


def source_documents():
    evidence, tracks = source_observations()
    evidence['schema_version'] = 1
    observations = adapt_tracks(evidence, profile='planar', tracks=tracks)
    return evidence, observations


@pytest.mark.parametrize('nested', [False, True])
def test_source_documents_keep_exact_original_ids_clocks_coordinates_and_unknowns(tmp_path, nested):
    source, artifact, config, out = case(tmp_path, nested=nested)
    documents = dict(zip(('evidence.json', 'observations.json'), source_documents()))
    for name, document in documents.items():
        replace_member(source, name, document)
        if nested:
            replace_member(source, 'cases/Run_7/' + name, document)
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis'
    for row in result['presentation']['documents']:
        name = row['artifact']
        assert row['status'] == 'available'
        assert row['source_artifacts'] == [name] + (['cases/Run_7/' + name] if nested else [])
        assert row['archive_artifacts'] == ['inputs/planar/' + path for path in row['source_artifacts']]
        assert row['sha256'] == identity(encoded(documents[name]))
        assert (out / name).read_bytes() == encoded(documents[name])
        assert 'not observations of the authored 3D scene' in row['scope']
    saved = json.loads((out / 'observations.json').read_bytes())
    assert saved['time']['basis'] == 'frame_index_divided_by_fps'
    assert saved['image'] == {'width_px': 200, 'height_px': 100}
    assert saved['observations'][-1]['visibility'] == 'not_detected'
    assert saved['observations'][-1]['measurements'] == {}
    assert saved['camera']['calibration']['value'] is None
    assert {obj['id'] for obj in saved['objects']} == {'ball', 'passive', 'wall'}
    checked_manifest(out)


@pytest.mark.parametrize('kind', ['ambiguous', 'malformed', 'foreign_media', 'foreign_pair', 'unknown_profile', 'wrong_adapter'])
def test_unsupported_source_documents_stay_archived_without_relabeling(tmp_path, kind):
    source, artifact, config, out = case(tmp_path, nested=kind == 'ambiguous')
    evidence, observations = source_documents()
    if kind == 'malformed':
        observations['observations'][-1]['measurements'] = {'invented': {}}
    elif kind == 'foreign_pair':
        evidence['source_sha256'] = 'b' * 64
    elif kind == 'unknown_profile':
        evidence['profile'] = 'unknown-3d-source'
    elif kind == 'wrong_adapter':
        observations['provenance']['adapter'] = 'exercise'
    elif kind == 'foreign_media':
        media = b'Different bound video'
        replace_member(source, 'source.avi', media)
        replace_member(source, 'source.json', {'artifact': 'source.avi', 'sha256': identity(media)})
    replace_member(source, 'evidence.json', evidence)
    replace_member(source, 'observations.json', observations)
    if kind == 'ambiguous':
        replace_member(source, 'cases/Run_7/observations.json', {**observations, 'limits': ['Distinct saved document']})
    result = run(source, artifact, config, out)
    assert result['status'] == 'constructed_hypothesis'
    rows = {row['artifact']: row for row in result['presentation']['documents']}
    missing = ({'evidence.json', 'observations.json'} if kind in {'foreign_media', 'foreign_pair'}
               else {'evidence.json'} if kind == 'unknown_profile' else {'observations.json'})
    for name in missing:
        assert rows[name]['status'] == 'unavailable' and rows[name]['reason']
        assert not (out / name).exists()
        assert (out / 'inputs/planar' / name).read_bytes() == (source / name).read_bytes()
    checked_manifest(out)


@pytest.mark.parametrize('kind', ['source', 'config', 'runtime', 'failure_marker'])
def test_final_input_readback_rejects_change_and_retains_original_copies(tmp_path, monkeypatch, bounded_runtime, kind):
    source, artifact, config, out = case(tmp_path)
    original = (source / 'all-slots.json').read_bytes()
    actual = pipeline._recheck

    def changing(*args):
        target = {'source': source / 'all-slots.json', 'config': config,
                  'runtime': bounded_runtime / 'validator.py', 'failure_marker': source / 'failure.json'}[kind]
        target.write_bytes(b'changed')
        return actual(*args)

    monkeypatch.setattr(pipeline, '_recheck', changing)
    result = run(source, artifact, config, out)
    assert result['status'] == 'failed' and result['stage'] == 'input_readback'
    assert (out / 'inputs/planar/all-slots.json').read_bytes() == original
    assert json.loads((out / 'report.json').read_bytes())['status'] == 'failed'
    assert checked_manifest(out)['status'] == 'failed'


@pytest.mark.parametrize('kind', ['copy', 'seal', 'output_tamper'])
def test_partial_failure_never_overwrites_or_emits_a_successful_seal(tmp_path, monkeypatch, kind):
    source, artifact, config, out = case(tmp_path)
    if kind == 'copy':
        actual = pipeline.write_package_files

        def write(directory, files):
            if 'candidate.json' in files:
                raise OSError('Injected copy failure')
            return actual(directory, files)

        monkeypatch.setattr(pipeline, 'write_package_files', write)
    elif kind == 'seal':
        monkeypatch.setattr(pipeline, '_seal', lambda *a, **k: (_ for _ in ()).throw(OSError('Injected seal failure')))
    else:
        actual = pipeline._inventory

        def inventory(directory, deadline, expected=None):
            if expected is not None:
                (directory / 'scene.json').write_bytes(b'changed')
            return actual(directory, deadline, expected)

        monkeypatch.setattr(pipeline, '_inventory', inventory)
    result = run(source, artifact, config, out)
    assert result['status'] == 'failed'
    assert (out / 'inputs/planar/all-slots.json').read_bytes() == (source / 'all-slots.json').read_bytes()
    if kind == 'copy':
        assert checked_manifest(out)['status'] == 'failed'
        assert json.loads((out / 'report.json').read_bytes())['status'] == 'failed'
    else:
        assert not (out / 'manifest.json').exists()
        assert json.loads((out / 'failure.json').read_bytes())['status'] == 'failed'
        assert json.loads((out / 'report.json').read_bytes())['status'] == 'constructed_hypothesis'


@pytest.mark.parametrize('kind', ['existing', 'inside_source', 'source_inside_output', 'config_inside_output', 'symlink'])
def test_new_output_cannot_overlap_inputs_or_replace_evidence(tmp_path, kind):
    source, artifact, config, out = case(tmp_path)
    if kind == 'existing':
        out.mkdir(); (out / 'preserved.txt').write_bytes(b'unchanged')
    elif kind == 'inside_source':
        out = source / 'new'
    elif kind == 'source_inside_output':
        out = tmp_path
    elif kind == 'config_inside_output':
        out = tmp_path / 'config-parent'; out.mkdir(); config = out / 'config.json'; config.write_bytes(b'{}')
    else:
        out.symlink_to(source, target_is_directory=True)
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        run(source, artifact, config, out)
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    if kind == 'existing':
        assert (out / 'preserved.txt').read_bytes() == b'unchanged'


@pytest.mark.parametrize('kind', ['input', 'runtime', 'output', 'files', 'deadline', 'invalid_seconds'])
def test_resource_bounds_stop_without_simulation(tmp_path, monkeypatch, kind):
    source, artifact, config, out = case(tmp_path)
    if kind == 'invalid_seconds':
        with pytest.raises(ValueError, match='twenty'):
            run(source, artifact, config, out, max_seconds=21)
        assert not out.exists()
        return
    if kind == 'input':
        monkeypatch.setattr(pipeline, 'MAX_INPUT_BYTES', 1024)
    elif kind == 'runtime':
        monkeypatch.setattr(pipeline, 'MAX_RUNTIME_BYTES', 8)
    elif kind == 'output':
        monkeypatch.setattr(pipeline, 'MAX_OUTPUT_BYTES', 20000)
    elif kind == 'files':
        monkeypatch.setattr(pipeline, 'MAX_OUTPUT_FILES', 5)
    else:
        ticks = iter((0., 21.))
        monkeypatch.setattr(pipeline.time, 'monotonic', lambda: next(ticks, 21.))
    result = run(source, artifact, config, out)
    assert result['status'] == 'failed'
    assert (out / 'report.json').exists()
    if (out / 'manifest.json').exists():
        assert checked_manifest(out)['status'] == 'failed'
