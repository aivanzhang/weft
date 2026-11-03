"""Compact metadata preserves controls and exact original provenance, not physical truth."""
from copy import deepcopy
import json

import pytest

from weft.shared_human_projection import create_projection, validate_projection, validate_runtime_human


def full_human(source_sha256='a'*64):
    names=('torso_twist','torso_lateral','torso_flex')+tuple(f'{side}_{joint}' for side in ('left','right') for joint in ('shoulder_twist','shoulder_lateral','shoulder','elbow','hip_twist','hip_lateral','hip','knee'))
    ids=[f'{source_sha256}:frame:{i}' for i in (0,3,6)];times=[0.,.1,.2];identity=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
    return {'schema_version':3,'profile':'exercise-v3','source_sha256':source_sha256,'dt':1/30,
        'lengths':{'torso':.5,'upper_arm':.3,'forearm':.25,'thigh':.42,'shin':.42,'shoulder_width':.4,'hip_width':.26},
        'controller':'bias_compensated_pd','motor_limits':{name:30. for name in names},
        'mobility':{'schema_version':1,'base_height':1.,'root_mass':.1,'slide_limits':[-5.,5.],
            'floor_height':0.,'clearance':.005,'kp_per_mass':60.,'kd_per_mass':12.,'force_limit_weight_multiple':3.,'maximum_force':2000.,'push_acceleration':6.},
        'reference':[{'time_s':t,'joints':{name:.1*t for name in names}} for t in times],
        'root_reference':[{'time_s':t,'translation':[0.,0.,0.]} for t in times],
        'reference_evidence':[{'time_s':t,'evidence_id':identifier,'joint_confidence':{name:.9 for name in names},'identity_status':'analytical_unverified','pose_status':'tracked'} for t,identifier in zip(times,ids)],
        'body_frame':{'schema_version':1,'status':'inferred','rotation':identity,'evidence_ids':ids,
            'reference':[{'time_s':t,'id':identifier,'rotation':deepcopy(identity)} for t,identifier in zip(times,ids)],
            'scope':'Analytical orientation fixture'},
        'reconstruction_coverage':{'physical_validation':'unverified','spatial_pose':{'scope':'Retained analytical audit'}},
        'assumptions':['Analytical model and timings only; no measured anatomy, source identity or physical accuracy'],
        'native_clock':{'schema_version':1,'profile':'source-native-clock-summary-v1','source_sha256':source_sha256,
            'evidence_sha256':'b'*64,'evidence_canonical_sha256':'c'*64,'binding_sha256':'d'*64,
            'metadata_frames':30,'decoded_frames':30,'nominal_fps':30.,'time_base':'1/30000','native_origin_pts':0,
            'native_to_reconstruction_offset_s':0.,'decoder_minus_native_s':0.,'capture_clock_status':'unknown',
            'detector_clock_status':'preserved_nominal_frame_index_divided_by_fps','reconstruction_time_basis':'native_pts_minus_first_pts'}}


def raw_human(value=None):
    return (json.dumps(full_human() if value is None else value,sort_keys=True,indent=2)+'\n').encode()


def test_projection_preserves_every_runtime_value_and_raw_input():
    full=full_human();before=deepcopy(full);raw=raw_human(full)
    compact,descriptor=create_projection(raw)
    assert validate_projection(compact,descriptor,raw)==compact
    assert validate_runtime_human(compact)==compact and full==before
    for field in set(full)-{'reconstruction_coverage','reference_evidence','body_frame'}:
        assert compact[field]==full[field]
    assert compact['body_frame']=={k:v for k,v in full['body_frame'].items() if k!='reference'}
    assert compact['reference_evidence']==[{k:r[k] for k in ('time_s','evidence_id')} for r in full['reference_evidence']]


@pytest.mark.parametrize('damage',['path','full_pin','runtime_pin','count','clock','reference','rotation','missing_raw','unknown_field'])
def test_projection_tamper_fails_exact_join(damage):
    raw=raw_human();compact,descriptor=create_projection(raw)
    if damage=='path':descriptor['artifact']='../human.json'
    elif damage=='full_pin':descriptor['full_human_sha256']='0'*64
    elif damage=='runtime_pin':descriptor['runtime_human_sha256']='0'*64
    elif damage=='count':descriptor['reference_count']=2
    elif damage=='clock':compact['native_clock']['decoder_minus_native_s']=.1
    elif damage=='reference':compact['reference'][1]['time_s']+=.01
    elif damage=='rotation':compact['body_frame']['rotation'][0][0]=2
    elif damage=='missing_raw':raw=b'{}'
    else:descriptor['verified_physical']=True
    with pytest.raises(ValueError):validate_projection(compact,descriptor,raw)


def test_full_body_validator_cannot_treat_runtime_projection_as_source():
    from weft.human import validate_human
    full=full_human();assert validate_human(full)==full
    compact,_=create_projection(raw_human(full))
    with pytest.raises(ValueError,match='sidecar'):validate_human(compact)


def test_camera_body_and_missing_reference_metadata_are_not_projected():
    full=full_human();full['camera_reference']={}
    with pytest.raises(ValueError):create_projection(raw_human(full))
    full=full_human();full['reference_evidence'].pop()
    with pytest.raises(ValueError):create_projection(raw_human(full))


def _source_scene(root):
    from test_compact_human_consumer_joins import projected_fixture
    from test_shared_world_reconstruction import fixture
    from test_shared_world_evidence import save_sources
    from test_shared_world_packaging import seal
    scene, _, options = projected_fixture()
    _, _, _, docs = fixture(include_documents=True)
    save_sources(root, scene, docs)
    (root / 'human.json').write_bytes(options['human_document'])
    seal(root)
    return scene, options['human_document']


def test_full_sidecar_is_verified_and_preserved_by_normal_package_collector(tmp_path):
    from weft.shared_world_packaging import collect_shared_source_files
    from weft.shared_world_evidence import verify_shared_world_sources
    scene, original = _source_scene(tmp_path)
    pinned = collect_shared_source_files(scene, tmp_path)
    assert pinned.files['human.json'] == original
    assert pinned.report['human_provenance']['verified']
    target = tmp_path / 'copy';target.mkdir()
    pinned.copy_to(target)
    assert (target / 'human.json').read_bytes() == original
    assert verify_shared_world_sources(scene, target)['human_provenance']['verified']


@pytest.mark.parametrize('damage', ['missing', 'changed', 'symlink', 'outer_pin', 'source', 'omitted_invalid_rotation'])
def test_source_qualification_refuses_incomplete_or_invalid_full_sidecar(tmp_path, damage):
    from weft.shared_world_packaging import collect_shared_source_files
    scene, original = _source_scene(tmp_path)
    path = tmp_path / 'human.json'
    if damage == 'missing':path.rename(tmp_path / 'preserved.json')
    elif damage == 'changed':path.write_bytes(original + b' ')
    elif damage == 'symlink':
        path.rename(tmp_path / 'preserved.json');path.symlink_to('preserved.json')
    elif damage == 'outer_pin':
        manifest = json.loads((tmp_path / 'manifest.json').read_bytes())
        manifest['files'].pop('human.json')
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    elif damage == 'source':scene['human_provenance']['source_sha256']='b'*64
    else:
        full = json.loads(original)
        full['body_frame']['reference']=[{'time_s':0.,'id':'bad','rotation':[[0.,0.,0.]]*3}]
        original = raw_human(full);path.write_bytes(original)
        scene['human_spec'],scene['human_provenance']=create_projection(original)
    with pytest.raises(ValueError):collect_shared_source_files(scene,tmp_path)


def test_original_and_projected_xml_reset_and_actions_are_exactly_equal(tmp_path):
    import numpy as np
    from weft.shared_world import SharedWorldEnv, shared_world_xml, validate_shared_world
    projected, raw = _source_scene(tmp_path)
    original = deepcopy(projected);original.pop('human_provenance');original['human_spec']=json.loads(raw)
    original=validate_shared_world(original);projected=validate_shared_world(projected)
    assert shared_world_xml(original)==shared_world_xml(projected)
    assert original['action_contract']==projected['action_contract']
    a,b=SharedWorldEnv(original),SharedWorldEnv(projected)
    try:
        oa,ia=a.reset(seed=7);ob,ib=b.reset(seed=7)
        np.testing.assert_array_equal(oa,ob)
        for index in range(6):
            action=np.zeros(a.action_space.shape);action[index % len(action)]=.1
            ra,rb=a.step(action),b.step(action)
            np.testing.assert_array_equal(ra[0],rb[0]);assert ra[1:4]==rb[1:4]
            np.testing.assert_array_equal(a.data.qpos,b.data.qpos)
            np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
            assert ra[4]==rb[4]
    finally:a.close();b.close()


def test_projection_does_not_raise_generic_scene_bound_or_change_small_scene(tmp_path):
    from weft.shared_human_projection import compact_scene_if_needed
    from weft.shared_world import validate_shared_world
    scene, raw = _source_scene(tmp_path)
    full=deepcopy(scene);full.pop('human_provenance');full['human_spec']=json.loads(raw)
    assert compact_scene_if_needed(full,raw) is full
    scene['human_spec']['assumptions']=['x'*6000]*100
    # A new consistent pin cannot turn oversized runtime data into admitted metadata.
    from weft.shared_human_projection import _canonical
    scene['human_provenance']['runtime_human_sha256']=_canonical(scene['human_spec'])
    with pytest.raises(ValueError,match='500'):
        validate_shared_world(scene)


def test_oversized_audit_externalization_keeps_full_scene_under_unchanged_bound(tmp_path):
    from weft.shared_human_projection import compact_scene_if_needed
    from weft.shared_world import validate_shared_world
    scene, original = _source_scene(tmp_path)
    full=json.loads(original);full['reconstruction_coverage']['audit']=['x'*600]*900
    original=raw_human(full)
    scene.pop('human_provenance');scene['human_spec']=full
    assert len(json.dumps(scene,separators=(',',':')).encode())>500_000
    projected=compact_scene_if_needed(scene,original)
    assert len(json.dumps(projected,separators=(',',':')).encode())<500_000
    assert validate_projection(projected['human_spec'],projected['human_provenance'],original)==projected['human_spec']
    assert validate_shared_world(projected)['human_spec']==projected['human_spec']
    assert scene['human_spec']==full and 'human_provenance' not in scene


from test_capture_recovery_seed import analytical_recovered_body


def test_seed_rejects_projection_implementation_changed_after_use(analytical_recovered_body,tmp_path,monkeypatch):
    from weft import shared_human_projection as projection, shared_world_lab as lab
    original_read=lab.read_package_file
    reads=[]
    def checked_read(directory,name,**kwargs):
        raw=original_read(directory,name,**kwargs)
        if name=='shared_human_projection.py':
            reads.append(name)
            if len(reads)>1:return raw+b'\n# concurrent edit\n'
        return raw
    def force_projection(scene,raw):
        assert reads==['shared_human_projection.py'], 'Implementation must be captured before projection executes'
        scene=deepcopy(scene)
        scene['human_spec'],scene['human_provenance']=create_projection(raw)
        return scene
    monkeypatch.setattr(lab,'read_package_file',checked_read)
    monkeypatch.setattr(projection,'compact_scene_if_needed',force_projection)
    with pytest.raises(ValueError,match='projection implementation changed'):
        lab.prepare_shared_world_seed(analytical_recovered_body,tmp_path/'seed')
    assert len(reads)==2
    assert not (tmp_path/'seed/manifest.json').exists()
