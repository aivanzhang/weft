"""Exact precursor-member custody with deferred construction-only image comparison."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only
from test_pose_surface_annotations import declared_raw_pose, fixture as original_fixture, raw


@pytest.fixture
def api():
    return importlib.import_module('weft.pose_precursor_custody')


def documents(*,context_change=None,precursor_change=None,audit_change=None,count=7):
    source_inputs=original_fixture(count,names=('left_wrist','right_wrist'),pose_change=declared_raw_pose)
    context=json.loads(source_inputs['pose_document'])
    context['model']={'name':'Analytical original model metadata','sha256':'b'*64}
    context['scope']='Authored source fixture; no inference or authenticated anatomy'
    precursor={key:deepcopy(context[key]) for key in ('frames','model','scope')}
    audit={'schema_version':1,'mode':'temporal','source_sha256':context['source_sha256'],
           'raw_artifact':'pose-landmarks.json','method':'short-island-bilateral-continuity-v1',
           'implementation_sha256':'d'*64,'repaired_evidence_ids':[],'decisions':[],
           'unavailable_frames':[{'evidence_id':context['frames'][0]['id'],'reason':'insufficient_torso_pair_separation'}],
           'scope':'Initial global left/right identity remains unverified'}
    if precursor_change: precursor_change(precursor)
    precursor_raw=raw(precursor)
    context['pose_identity']={'mode':'temporal','audit_artifact':'identity-audit.json','repaired_frames':0,
                              'raw_artifact_sha256':sha256(precursor_raw).hexdigest()}
    audit['raw_artifact_sha256']=context['pose_identity']['raw_artifact_sha256']
    if context_change: context_change(context)
    if audit_change: audit_change(audit)
    context_raw,audit_raw=raw(context),raw(audit)
    manifest={'schema':1,'files':{'evidence.json':sha256(context_raw).hexdigest(),
        'pose-landmarks.json':sha256(precursor_raw).hexdigest(),'identity-audit.json':sha256(audit_raw).hexdigest()}}
    return {'context_pose_document':context_raw,'raw_pose_document':precursor_raw,
        'identity_audit_document':audit_raw,'manifest_document':raw(manifest)}


def test_complete_metadata_deferred_payload_and_exact_audit_custody(api,monkeypatch):
    args=documents(); custody=api.prepare_pose_precursor_custody(**args)
    meta=custody.describe()
    assert meta['counts']=={'original_source_slots':7,'context_pose_rows':3,'raw_pose_rows':3}
    assert meta['mode']=='temporal' and meta['recorded_repaired_frames']==0 and meta['audit_unavailable_frames']==1
    assert custody.frame(0)['audit_identity_unavailable_reason']=='insufficient_torso_pair_separation'
    assert custody.frame(1)['reason']=='context_pose_not_sampled'
    assert custody.frame(6)['reason']=='metadata_only_not_decoded'
    assert dict(custody.input_documents)['evidence.json']==args['context_pose_document']
    assert meta['identity_authenticated'] is False and meta['contact']=='unknown'
    working=custody._begin(args['context_pose_document'])
    first=working.qualify(0); assert first['status']=='matched_saved_precursor'
    assert first['raw_image_payload_sha256']==first['context_image_payload_sha256']
    first['reason']='caller mutation'
    assert working.qualify(0)['reason'] is None and working.image_comparisons==1
    summary,rows=working.finish()
    assert json.loads(summary)['qualification']['requested_frames']==1 and len(rows)==7
    assert json.loads(rows[2])['reason']=='not_queried_canonical_unqualified_or_unsampled'
    assert working.context==working.raw==working.cache=={}
    with pytest.raises(ValueError,match='closed'): working.qualify(0)


@pytest.mark.parametrize('kind,reason',[
    ('coordinate','original_image_coordinates_or_scores_differ'),
    ('score','original_image_coordinates_or_scores_differ'),
    ('clock','original_frame_or_clock_differs'),('missing_clock','precursor_decoder_clock_missing'),
    ('identity','unsupported_precursor_identity_status'),('unknown_status','unavailable_precursor_pose_status'),
    ('multiple','unavailable_or_ambiguous_model_hypothesis'),('bool_count','unavailable_or_ambiguous_model_hypothesis'),
    ('derived','derived_or_crop_pose_requires_other_proof'),('crop','derived_or_crop_pose_requires_other_proof'),
    ('missing_raw','raw_precursor_slot_missing'),('shape','missing_33_precursor_image_landmarks'),
])
def test_zero_repairs_never_substitute_for_exact_original_array_match(api,kind,reason):
    def context(value):
        row=value['frames'][0]
        if kind=='coordinate': row['landmarks'][15]['x']+=.01
        if kind=='score': row['landmarks'][15]['visibility']-=.1
        if kind=='clock': row['decoder_timestamp_s']+=.01
        if kind=='missing_clock': row['decoder_timestamp_s']=None
        if kind=='identity': row['identity_status']='unknown'
        if kind=='unknown_status': row['status']='future_unqualified_pose'
        if kind=='multiple': row['pose_count']=2
        if kind=='bool_count': row['pose_count']=True
        if kind=='derived': row['identity_recompute_required']=False
        if kind=='crop': row['image_refinement']=None
    def precursor(value):
        if kind=='missing_raw': value['frames'].pop(0)
        if kind=='shape': value['frames'][0]['landmarks']='unavailable'
    args=documents(context_change=context,precursor_change=precursor)
    custody=api.prepare_pose_precursor_custody(**args); working=custody._begin(args['context_pose_document'])
    result=working.qualify(0)
    assert result['reason']==reason and result['status']=='unavailable' and result['contact']=='unknown'
    summary,rows=working.finish(); assert len(rows)==7
    assert json.loads(summary)['recorded_repaired_frames']==0


@pytest.mark.parametrize('kind,message',[
    ('member','member hashes'),('model','model metadata'),('source','source/frame/clock'),
    ('audit_source','audit raw/source/mode'),('mode','identity declaration'),('audit_mode','audit raw/source/mode'),
    ('repair_count','repaired identity denominator'),('boolean_count','source dimensions/counts'),
    ('unknown_method','known recorded temporal'),
])
def test_contradictory_member_source_model_or_audit_context_fails_closed(api,kind,message):
    def context(value):
        if kind=='model': value['model']['sha256']='f'*64
        if kind=='source': value['source_sha256']='f'*64
        if kind=='mode': value['pose_identity']['mode']='future_mode'
        if kind=='repair_count': value['pose_identity']['repaired_frames']=1
        if kind=='boolean_count': value['decoded_frames']=True
    def audit(value):
        if kind=='audit_source': value['source_sha256']='f'*64
        if kind=='audit_mode': value['mode']='raw'
        if kind=='unknown_method': value['method']='unknown'
    args=documents(context_change=context,audit_change=audit)
    if kind=='member':
        manifest=json.loads(args['manifest_document']); manifest['files']['evidence.json']='f'*64
        args['manifest_document']=raw(manifest)
    with pytest.raises(ValueError,match=message): api.prepare_pose_precursor_custody(**args)


def test_reported_repaired_row_abstains_even_if_arrays_happen_to_match(api):
    args=documents(context_change=lambda d:d['pose_identity'].update(repaired_frames=1),
        audit_change=lambda d:d['repaired_evidence_ids'].append(f"{d['source_sha256']}:frame:0"))
    custody=api.prepare_pose_precursor_custody(**args); working=custody._begin(args['context_pose_document'])
    assert working.qualify(0)['reason']=='recorded_identity_repair_not_original_coordinates'
    assert working.image_comparisons==0


@pytest.mark.parametrize('sentinel',[{'pixel':[35,33]},['coordinate_sentinel',35,33]])
def test_raw_audit_method_cannot_export_coordinate_payload(api,sentinel):
    args=documents(context_change=lambda d:d['pose_identity'].update(mode='raw'),
        audit_change=lambda d:d.update(mode='raw',method=sentinel))
    with pytest.raises(ValueError,match='bounded scalar audit method'):
        result=api.prepare_pose_precursor_custody(**args)
        assert result.describe()['audit_method']==sentinel


@pytest.mark.parametrize('method',[None,'raw_model_identity_retained','missing'])
def test_raw_audit_method_preserves_optional_scalar_label_without_identity_claim(api,method):
    def audit(value):
        value['mode']='raw'
        if method=='missing': value.pop('method')
        else: value['method']=method
    args=documents(context_change=lambda d:d['pose_identity'].update(mode='raw'),audit_change=audit)
    result=api.prepare_pose_precursor_custody(**args); meta=result.describe()
    assert meta['audit_method']==(None if method=='missing' else method)
    assert meta['mode']=='raw' and meta['identity_authenticated'] is False and meta['contact']=='unknown'
    assert dict(result.input_documents)['identity-audit.json']==args['identity_audit_document']


@pytest.mark.parametrize('method',[True,0,'',' ','x'*161])
def test_raw_audit_method_rejects_wrong_or_unbounded_scalar_types(api,method):
    args=documents(context_change=lambda d:d['pose_identity'].update(mode='raw'),
        audit_change=lambda d:d.update(mode='raw',method=method))
    with pytest.raises(ValueError,match='bounded scalar audit method'):
        api.prepare_pose_precursor_custody(**args)


def test_temporal_audit_method_keeps_its_known_method_requirement(api):
    args=documents(); result=api.prepare_pose_precursor_custody(**args)
    assert result.describe()['audit_method']=='short-island-bilateral-continuity-v1'
    assert result.describe()['mode']=='temporal'
    for method in (None,'raw_model_identity_retained'):
        with pytest.raises(ValueError,match='known recorded temporal'):
            api.prepare_pose_precursor_custody(**documents(audit_change=lambda d:d.update(method=method)))


@pytest.mark.parametrize('field,message',[
    ('context_status','bounded scalar original pose labels'),('raw_identity','bounded scalar original pose labels'),
    ('unavailable_reason','audit unavailable record'),('fps','source dimensions/counts'),
])
def test_other_public_metadata_sentinels_fail_closed(api,field,message):
    sentinel={'pixel':[35,33]}
    def context(value):
        if field=='context_status': value['frames'][0]['status']=sentinel
        if field=='fps': value['fps']=sentinel
    def precursor(value):
        if field=='raw_identity': value['frames'][0]['identity_status']=sentinel
    def audit(value):
        if field=='unavailable_reason': value['unavailable_frames'][0]['reason']=sentinel
    args=documents(context_change=context,precursor_change=precursor,audit_change=audit)
    with pytest.raises(ValueError,match=message): api.prepare_pose_precursor_custody(**args)


def test_unrequested_bad_payload_is_not_compared_or_normalized(api):
    args=documents(precursor_change=lambda d:d['frames'][0].update(landmarks={'coordinate_sentinel':[999,999]}))
    custody=api.prepare_pose_precursor_custody(**args); working=custody._begin(args['context_pose_document'])
    working.qualify(2)
    summary,rows=working.finish()
    assert json.loads(summary)['qualification']['image_payload_comparisons']==1
    assert json.loads(rows[0])['representation']=='not_read_unqualified_source'
    assert 'coordinate_sentinel' not in b''.join(rows).decode()


def test_original_labels_and_derived_marker_names_survive_without_payload_leak(api):
    args=documents(precursor_change=lambda d:d['frames'][0].update(identity_raw={'coordinate_sentinel':[500,600]}))
    custody=api.prepare_pose_precursor_custody(**args); row=custody.frame(0)
    assert row['raw_metadata']['identity_status']=='single_model_hypothesis_not_verified_person_count'
    assert row['raw_metadata']['status']=='tracked' and row['raw_metadata']['pose_count']==1
    assert row['raw_metadata']['derivation_markers']==['identity_raw']
    assert 'coordinate_sentinel' not in json.dumps(row)
    assert row['audit_mode']=='temporal' and row['audit_identity_unavailable_reason']=='insufficient_torso_pair_separation'


def test_prepare_byte_budget_precedes_json_parsing(api,monkeypatch):
    args=documents(); monkeypatch.setattr(api,'MAX_INPUT_BYTES',1); monkeypatch.setattr(api,'_document',denied)
    with pytest.raises(ValueError,match='sixteen MiB'): api.prepare_pose_precursor_custody(**args)


def test_late_qualification_and_cache_exhaustion_do_not_publish_partial_rows(api,monkeypatch):
    args=documents(); custody=api.prepare_pose_precursor_custody(**args)
    working=custody._begin(args['context_pose_document'])
    monkeypatch.setattr(api,'MAX_CACHE_BYTES',working.cache_bytes+1)
    with pytest.raises(ValueError,match='cache'): working.qualify(0)
    assert working.cache=={}
    original=time.monotonic
    monkeypatch.setattr(api.time,'monotonic',lambda:original()+10)
    with pytest.raises(TimeoutError,match='cooperative deadline'): working.finish()


def test_context_byte_pin_cannot_be_replaced_by_equal_reserialization(api):
    args=documents(); custody=api.prepare_pose_precursor_custody(**args)
    with pytest.raises(ValueError,match='context bytes'):
        custody._begin(json.dumps(json.loads(args['context_pose_document']),indent=1).encode())


def test_representative_preparation_qualification_and_cached_inspection(api,record_property):
    args=documents(count=61); tick=time.perf_counter(); custody=api.prepare_pose_precursor_custody(**args)
    working=custody._begin(args['context_pose_document'])
    for frame in range(0,60,2): working.qualify(frame)
    summary,rows=working.finish(); elapsed=time.perf_counter()-tick
    tick=time.perf_counter()
    for index in range(1000): custody.frame(index%61)
    query=time.perf_counter()-tick
    record_property('prepare_and_30_compare_seconds',elapsed)
    record_property('cached_1000_metadata_queries_seconds',query)
    record_property('qualification',json.dumps(json.loads(summary)['qualification'],sort_keys=True))
    assert len(rows)==61 and elapsed<api.MAX_SECONDS
