"""Wrap actual analytic exports in audit-contract fixtures; never call them avatars."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets import fur_attachment, fur_bindings
from avatar_assets.safety import artifacts, sha256, write_json
from avatar_assets.three_d import _GLB


def inspect_body(glb, report):
    identity = report['portable_fur_identity']
    matches = [n for n in glb.doc['nodes'] if n.get('name') == identity['source_node']]
    assert len(matches) == 1
    node = matches[0]; skin = glb.doc['skins'][node['skin']]
    names = [glb.doc['nodes'][i]['name'] for i in skin['joints']]
    expected = np.asarray(report['native_body_points'], dtype=np.float32)[:, [0, 2, 1]]
    expected[:, 2] *= -1
    seen, faces = set(), []
    for primitive in glb.doc['meshes'][node['mesh']]['primitives']:
        attrs = primitive['attributes']
        ids = np.asarray(glb.accessor(attrs['_AVATAR_FUR_VERTEX_ID'], 'SCALAR', (5126,))).ravel().astype(int)
        positions = np.asarray(glb.accessor(attrs['POSITION'], 'VEC3', (5126,)))
        assert np.array_equal(positions, expected[ids])
        joints = np.asarray(glb.accessor(attrs['JOINTS_0'], 'VEC4', (5121, 5123)), dtype=int)
        weights = np.asarray(glb.accessor(attrs['WEIGHTS_0'], 'VEC4', (5126,)))
        for source_id, joint_row, weight_row in zip(ids, joints, weights):
            actual = {}
            for j, w in zip(joint_row, weight_row):
                if w:
                    actual[names[j]] = actual.get(names[j], 0.)+float(w)
            assert actual == report['native_body_weights'][source_id]
        indices = np.asarray(glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125))).ravel().astype(int)
        faces.extend(ids[indices].reshape(-1, 3).tolist()); seen.update(map(int, ids))
    assert seen == set(range(len(expected)))
    assert {tuple(row) for row in faces} == {tuple(row) for row in report['native_body_faces']}
    assert [a['name'] for a in glb.doc['animations']] == ['flex_probe']
    return {'body_points_exact_float32_basis': True, 'body_named_weights_exact': True,
            'body_oriented_id_triangles_exact': True, 'body_vertex_ids': sorted(seen)}


request = json.loads((HERE/'cpu-request.json').read_text())
assert sha256(Path(__file__)) == request['script_sha256']
for row in request['inputs']:
    assert sha256(ROOT/row['path']) == row['sha256']
out = ROOT/request['output']; assert out.is_dir()
packs = out/'packs'; packs.mkdir()
results = []
for profile in ('legacy', 'combed-v1', 'tuft-v1'):
    origin = ROOT/'outputs/portable-attachment-native-fixture-v1'/profile
    fixture = json.loads((origin/'binding-report.json').read_text())
    glb = _GLB(origin/'model.glb')
    body_proof = inspect_body(glb, fixture)
    pack = packs/profile; pack.mkdir(); (pack/'sources').mkdir()
    shutil.copyfile(origin/'model.glb', pack/'model.glb')
    fur_bindings.copy_bindings(origin, pack, fixture['portable_fur'], 4)
    source = {key: fixture[key] for key in ('native_body_points', 'native_body_faces', 'native_body_weights')}
    source.update(analytic_fixture_not_avatar=True, originating_native_report_sha256=sha256(origin/'binding-report.json'))
    write_json(pack/'source-fixture.json', source)
    source_hash = sha256(pack/'source-fixture.json')
    design = {'schema_version': 1, 'id': 'analytic-native-'+profile, 'source': 'source-fixture.json',
              'source_sha256': source_hash, 'states': [{'id': 'flex_probe'}],
              'portable_fur': fixture['config'], 'analytic_fixture_not_public_finish': True}
    write_json(pack/'sources/finish-design.json', design)
    finish = {'schema_version': 1, 'source_sha256': source_hash, 'source_unchanged': True,
              'states': design['states'], 'portable_fur': fixture['portable_fur'],
              'portable_fur_identity': fixture['portable_fur_identity'],
              'analytic_fixture_not_public_finish': True, 'independent_fixture_body_check': body_proof}
    write_json(pack/'finish-report.json', finish)
    write_json(pack/'manifest.json', {'schema_version': 1, 'id': design['id'],
               'status': 'needs_visual_review', 'source_sha256': source_hash, 'states': design['states'],
               'finish': finish, 'analytic_fixture_not_avatar': True, 'artifacts': artifacts(pack)})
    intent = {'schema_version': 1, 'source_sha256': source_hash,
              'design_sha256': sha256(pack/'sources/finish-design.json'), 'model_sha256': sha256(pack/'model.glb'),
              'state_ids': ['flex_probe']}
    write_json(out/(profile+'-intent.json'), intent)
    report = fur_attachment.audit_finish_pack(pack, intent)
    write_json(out/(profile+'-report.json'), report)
    fur_attachment.validate_report(pack, intent, report)
    assert report['status'] == 'measured'
    assert report['passed'] is (profile == 'tuft-v1')
    assert report['roots'] == 4 and report['metrics']['root_pose_checks'] == 100
    if profile != 'tuft-v1':
        assert report['metrics']['max_named_weight_error'] < 1e-6
        assert report['metrics']['max_attachment_error_m'] > 1e-5
    results.append({'profile': profile, 'report_sha256': sha256(out/(profile+'-report.json')),
                    'source_model_sha256': sha256(origin/'model.glb'), 'passed': report['passed'],
                    'metrics': report['metrics'], 'body_proof': body_proof,
                    'native_fixture_warnings_retained': True})
for row in request['inputs']:
    assert sha256(ROOT/row['path']) == row['sha256']
write_json(out/'result.json', {'schema_version': 1, 'status': 'passed_expected_native_fixture_outcomes',
    'results': results, 'source_pins_unchanged': True, 'analytic_not_generated_avatars': True,
    'new_native_calls': 0, 'new_provider_calls': 0, 'new_pipeline_role_calls': 0,
    'appearance_accepted': False})
print(json.dumps({'status': 'passed_expected_native_fixture_outcomes',
                  'profiles': [(r['profile'], r['passed'], r['metrics']['max_attachment_error_m']) for r in results]}))
