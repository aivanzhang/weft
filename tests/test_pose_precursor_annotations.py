"""Actual annotation composition preserves default bytes and excluded-source boundaries."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json

import pytest

from test_multiview_landmark_evidence import denied, local_only
from test_pose_surface_annotations import declared_raw_pose, fixture as original_fixture, raw
from test_pose_precursor_custody import documents


def inputs(documents,observation_change=None):
    context=json.loads(documents['context_pose_document'])
    observations=importlib.import_module('weft.observations').adapt_exercise(context)
    if observation_change: observation_change(observations)
    encoded=raw(observations); pin=sha256(encoded).hexdigest()
    channels=[{'id':name,'entity_ref':{'document_sha256':pin,'object_id':'person-0'},'point_role':name,
               'surface_id':'unknown-surface'} for name in ('left_wrist','right_wrist')]
    return {'observation_documents':{pin:encoded},'pose_document':documents['context_pose_document'],
            'source_sha256':context['source_sha256'],'channels':channels,
            'landmark_bindings':[{'channel_id':name,'landmark_name':name,'uncertainty_px':2.} for name in ('left_wrist','right_wrist')],
            'minimum_confidence':.5,'uncertainty_method':'Authored two-pixel hypothesis',
            'uncertainty_note':'Not calibrated from confidence or validated as physical contact'}


@pytest.fixture
def api(): return importlib.import_module('weft.pose_surface_annotations')


def test_actual_annotation_path_uses_precursor_without_relabeling_temporal_identity(api):
    source=documents(); args=inputs(source)
    custody=importlib.import_module('weft.pose_precursor_custody').prepare_pose_precursor_custody(**source)
    result=api.prepare_pose_surface_annotations(**args,precursor_custody=custody)
    assert result.pose_document==source['context_pose_document'] and result.precursor_custody is custody
    assert json.loads(result.pose_document)['pose_identity']['mode']=='temporal'
    first=result.frame(0); point=first['channels'][0]
    assert point['status']=='copied_source_image_estimate' and point['pixel']==pytest.approx([35.,33.2])
    assert point['representation']=='matched_saved_precursor_model_estimate_identity_unverified'
    assert first['precursor_custody']['audit_identity_unavailable_reason']=='insufficient_torso_pair_separation'
    assert not first['precursor_custody']['identity_authenticated'] and point['contact']=='unknown'
    annotation=json.loads(result.annotation_document()); provenance=json.loads(annotation['provenance'])
    assert provenance['precursor_custody']['mode']=='temporal' and provenance['precursor_custody']['audit_unavailable_frames']==1
    assert provenance['precursor_custody']['raw_sha256']==sha256(source['raw_pose_document']).hexdigest()
    assert provenance['precursor_custody']['audit_sha256']==sha256(source['identity_audit_document']).hexdigest()
    meta=result.describe(); assert meta['counts']['retained_source_frames']==7 and meta['counts']['channel_slots']==14
    assert meta['precursor_custody']['qualification']['requested_frames']==3
    assert meta['precursor_custody']['qualification']['image_payload_comparisons']==3
    assert meta['retained_precursor_cache_bytes']>0 and not meta['raw_identity_authenticated']


@pytest.mark.parametrize('reason',['holdout','canonical_payload_mismatch','canonical_pin_mismatch'])
def test_unqualified_rows_do_not_compare_precursor_landmarks(api,monkeypatch,reason):
    source=documents(precursor_change=lambda d:d['frames'][0].update(landmarks={'withheld_coordinate_sentinel':[500,600]}))
    def change(observation):
        if reason=='holdout': observation['observations'][0]['partition']='audit'
        if reason=='canonical_payload_mismatch': observation['evidence'][0]['payload']['landmarks']='not_same_payload'
        if reason=='canonical_pin_mismatch': observation['provenance']['input_evidence_sha256']='f'*64
    args=inputs(source,change); module=importlib.import_module('weft.pose_precursor_custody')
    custody=module.prepare_pose_precursor_custody(**source)
    original=module._Qualification.qualify; requested=[]
    def qualified(self,frame):
        assert frame!=0
        requested.append(frame); return original(self,frame)
    monkeypatch.setattr(module._Qualification,'qualify',qualified)
    result=api.prepare_pose_surface_annotations(**args,precursor_custody=custody)
    assert result.frame(0)['channels'][0]['pixel'] is None
    assert result.frame(0)['precursor_custody']['representation']=='not_read_unqualified_source'
    assert 'withheld_coordinate_sentinel' not in result.annotation_document().decode()
    if reason=='canonical_pin_mismatch': assert requested==[]


def test_zero_repair_coordinate_change_remains_unknown_in_actual_annotations(api):
    source=documents(context_change=lambda d:d['frames'][0]['landmarks'][15].update(x=.9))
    custody=importlib.import_module('weft.pose_precursor_custody').prepare_pose_precursor_custody(**source)
    result=api.prepare_pose_surface_annotations(**inputs(source),precursor_custody=custody)
    assert all(point['pixel'] is None and point['reason']=='original_image_coordinates_or_scores_differ' for point in result.frame(0)['channels'])
    assert json.loads(result.annotation_document())['annotations'][0]['status']=='unknown'


def test_wrong_prepared_context_rejected_without_implicit_hash_relabeling(api):
    source=documents(); other=documents(context_change=lambda d:d.update(scope=d['scope']+'; separate supplied context'))
    custody=importlib.import_module('weft.pose_precursor_custody').prepare_pose_precursor_custody(**source)
    with pytest.raises(ValueError,match='different exact context'):
        api.prepare_pose_surface_annotations(**inputs(other),precursor_custody=custody)


def test_original_start_no_pose_is_retained_and_never_replaced(api):
    def empty(row): row.update(status='no_pose',pose_count=0,identity_status='no_model_hypothesis',landmarks=[])
    source=documents(context_change=lambda d:empty(d['frames'][0]),precursor_change=lambda d:empty(d['frames'][0]))
    custody=importlib.import_module('weft.pose_precursor_custody').prepare_pose_precursor_custody(**source)
    result=api.prepare_pose_surface_annotations(**inputs(source),precursor_custody=custody)
    assert result.frame(0)['channels'][0]['reason']=='source_entity_unavailable'
    assert result.frame(0)['precursor_custody']['status']=='not_queried'
    assert result.frame(2)['channels'][0]['pixel'] is not None
    assert result.describe()['counts']['retained_source_frames']==7


def test_no_opt_in_default_serialized_outputs_match_published_module_exactly(api):
    # Captured from published a4d29c84f46f0be6, before the opt-in custody integration.
    expected=[
        ('899a01537865269b623f116d0b6600a06a764d88dadc994482cb382fc8fc063a','a386229108f650f088584b44844ff5f9fe66f37307f98bc236e895e71505f479','dba9a4d7f822beb7ae84581952d68e9df8708025bd0e19d06a4fb7620dc3440d'),
        ('65bca52d03adc21c101125cade90693b4fc1a9d527444e51543a61115abb415a','d6b05b9d581862eeb435ba2f9373f2e7931f9b6abc55039d7de9926dc8fb45e6','0e73d71d7ea1df38a90f8d1e78c3fd91abb802f3b93f625c680095ed63408cd1'),
        ('a2bd67d346a69d8018425f2e39bc2500ee00f244e41f072d7e9c7051932e175e','df08ee538e1a89913d7865d2e77f7930cbffe79076ce9f9c841f14f5a969b80a','988939bae9a486f0f705e56a6936ea5ad6c8ccdacc6c35f9f093858491de2f0c')]
    cases=[original_fixture(),original_fixture(pose_change=declared_raw_pose),inputs(documents())]
    for args,pins in zip(cases,expected,strict=True):
        new=api.prepare_pose_surface_annotations(**deepcopy(args))
        assert tuple(sha256(value).hexdigest() for value in (new._annotation,new._metadata,b'\n'.join(new._rows)))==pins
        assert new.precursor_custody is None
