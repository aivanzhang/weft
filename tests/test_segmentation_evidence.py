"""Synthetic sealed-run controls; no provider calls or development score inputs."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from weft.io import digest, write_json
from weft.segmentation_evidence import CONTRACT, adapt_tracking_run


def seal(root):
    write_json(root/'manifest.json',{'files':{str(path.relative_to(root)):digest(path)
               for path in root.rglob('*') if path.is_file() and path.name!='manifest.json'}})


def returned(frame, identifier, box):
    return {'frame':frame,'object_id':str(identifier),'status':'returned','box_xyxy_halfopen':box,
        'box_xywh':[box[0],box[1],box[2]-box[0],box[3]-box[1]],'confidence':.8,'box_confidence':.8,
        'mask_confidence':.6,'mask_encoded':'!abcdec=.5[]','mask_dimensions':[4,4],'mask_encoding':'one_bit',
        'mask_status':'opaque_not_decoded_or_codec_validated','source_dimensions':[100,80],
        'unknown_provider_detail':{'retained':'verbatim'}}


def missing(frame, identifier):
    row=returned(frame,identifier,[0,0,4,4])
    row.update(status='missing',box_xyxy_halfopen=None,box_xywh=None,confidence=None,box_confidence=None,
               mask_confidence=None,mask_encoded=None,mask_dimensions=None,mask_encoding=None,mask_status='not_returned')
    return row


def fixture(root, boxes=None, *, initial=0, times=None, ids=(1,2)):
    root.mkdir(parents=True)
    (root/'call').mkdir()
    (root/'source.mp4').write_bytes(b'synthetic container bytes; decoder metadata is authored fixture')
    video_hash=digest(root/'source.mp4')
    boxes=boxes if boxes is not None else [[[0,0,6,8],[30,0,36,8]],[[0,0,6,8],[11,0,17,8]],[[0,0,6,8],[50,0,56,8]]]
    count=len(boxes)
    times=times if times is not None else [index*.017 for index in range(count)]
    prompts=[{'object_id':identifier,'frame':initial,'roi':[0,0,6,8]} for identifier in ids]
    frames=[{'frame':index,'objects':[returned(index,identifier,box) if box else missing(index,identifier)
                                     for identifier,box in zip(ids,row)]} for index,row in enumerate(boxes)]
    write_json(root/'video.json',{'source_sha256':video_hash,'width':100,'height':80,'frame_count':count,
                                'decoder_timestamps_s':times})
    write_json(root/'call/call.json',{'model':'sam-3.1','video_sha256':video_hash,'status':'completed',
               'initialization':prompts,'response_id':'fixture-response','request_sha256':'a'*64})
    write_json(root/'call/segmentation.json',{'status':'completed','object_ids':[str(i) for i in ids],
        'initialization':prompts,'response_id':'fixture-response','frames':frames,'raw_text':'synthetic text'})
    write_json(root/'report.json',{'objects':prompts})
    seal(root)
    return root


def mutate(root, name, fn):
    path=root/name
    value=json.loads(path.read_text())
    fn(value)
    write_json(path,value)
    seal(root)


def test_exact_irregular_times_raw_ids_masks_and_unknown_fields_are_preserved(tmp_path):
    root=fixture(tmp_path/'run',times=[0,.016,.033])
    before={path:digest(path) for path in root.rglob('*') if path.is_file()}
    evidence=adapt_tracking_run(root)
    assert evidence['time']['timestamps_s']==[0,.016,.033]
    assert evidence['source']['original_source_sha256'] is None
    assert [row['time_s'] for row in evidence['observations']]==[0,0,.016,.016,.033,.033]
    row=evidence['observations'][0]
    assert row['id']==f"{evidence['source']['submitted_video_sha256']}:frame:0:model-object:1"
    assert row['raw_model_id']=='1' and row['geometry']['status']=='inferred'
    assert row['geometry']['center_px']==[3,4]
    assert row['raw_model_output']['mask_encoded']=='!abcdec=.5[]'
    assert row['raw_model_output']['unknown_provider_detail']=={'retained':'verbatim'}
    assert row['mask_provenance']['encoded_sha256']==sha256(b'!abcdec=.5[]').hexdigest()
    assert row['confidence']=={'box':.8,'mask':.6}
    assert evidence['physics_ready'] is False
    assert all(obj['kind'] is None and obj['role'] is None for obj in evidence['objects'])
    assert all(digest(path)==value for path,value in before.items())
    row['raw_model_output']['mask_encoded']='mutated'
    assert adapt_tracking_run(root)['observations'][0]['raw_model_output']['mask_encoded']=='!abcdec=.5[]'


def test_frozen_edge_gap_boundary_and_persistent_group_do_not_change_positions(tmp_path):
    root=fixture(tmp_path/'run')
    evidence=adapt_tracking_run(root)
    assert len(evidence['identity_groups'])==1
    group=evidence['identity_groups'][0]
    assert group['onset_frame']==1 and group['raw_model_ids']==['1','2']
    assert evidence['frames'][0]['identity_groups']==[]
    assert evidence['frames'][2]['identity_groups']==[group['id']]
    event=evidence['proximity_events'][0]
    assert event['edge_gap_px']==event['threshold_px']==5
    assert event['type']=='possible_association_ambiguity' and event['physical_contact_verified'] is False
    assert evidence['observations'][-1]['geometry']['center_px']==[53,4]
    assert evidence['observations'][-1]['identity']['physical_identity_verified'] is False
    assert evidence['frames'][2]['anonymous_location_set'][1]['center_px']==[53,4]
    mutate(root,'call/segmentation.json',lambda value: value['frames'][1]['objects'].__setitem__(1,returned(1,2,[12,0,18,8])))
    assert adapt_tracking_run(root)['identity_groups']==[]


def test_group_merges_preserve_onsets_and_no_future_information_in_earlier_frames(tmp_path):
    boxes=[[[0,0,6,8],[11,0,17,8],[60,0,66,8]],
           [[0,0,6,8],[30,0,36,8],[41,0,47,8]],
           [[0,0,6,8],[30,0,36,8],[70,0,76,8]]]
    evidence=adapt_tracking_run(fixture(tmp_path/'run',boxes,ids=(1,2,3)))
    first,second=evidence['identity_groups']
    assert first['raw_model_ids']==['1','2'] and first['onset_frame']==0
    assert second['raw_model_ids']==['1','2','3'] and second['onset_frame']==1
    assert second['predecessor_group_ids']==[first['id']]
    assert evidence['frames'][0]['identity_groups']==[first['id']]
    assert evidence['frames'][2]['identity_groups']==[second['id']]
    assert evidence['observations'][2]['identity']['unresolved_group_id'] is None


def test_missing_preinit_and_reappearance_never_manufacture_geometry_or_continuity(tmp_path):
    boxes=[[None,None],[[0,0,6,8],[30,0,36,8]],[None,[30,0,36,8]],[[70,40,76,48],[30,0,36,8]]]
    evidence=adapt_tracking_run(fixture(tmp_path/'run',boxes,initial=1))
    assert [row['phase'] for row in evidence['observations'][::2]]==['pre_initialization','initialization','post_initialization','post_initialization']
    assert evidence['observations'][0]['identity']['gap_seen'] is False
    assert evidence['observations'][4]['geometry'] is None
    assert evidence['observations'][4]['mask_provenance'] is None
    assert evidence['observations'][6]['identity']['gap_seen'] is True
    assert evidence['observations'][6]['geometry']['center_px']==[73,44]
    assert evidence['frames'][0]['anonymous_location_set']==[]
    assert len(evidence['frames'][2]['anonymous_location_set'])==1
    assert evidence['identity_groups']==[]


def test_duplicate_model_regions_remain_two_proposals_with_ambiguity_not_relabeling(tmp_path):
    evidence=adapt_tracking_run(fixture(tmp_path/'run',[[[10,10,16,18],[10,10,16,18]]]))
    assert [row['raw_model_id'] for row in evidence['observations']]==['1','2']
    assert len(evidence['frames'][0]['anonymous_location_set'])==2
    assert evidence['proximity_events'][0]['edge_gap_px']==0
    assert len(evidence['identity_groups'])==1


@pytest.mark.parametrize('name,mutation',[
    ('video.json',lambda v:v.update(frame_count=501)),
    ('video.json',lambda v:v.update(decoder_timestamps_s=[0,.02,.01])),
    ('video.json',lambda v:v.update(decoder_timestamps_s=[0,.02])),
    ('video.json',lambda v:v.update(source_sha256='0'*64)),
    ('call/call.json',lambda v:v.update(status='failed')),
    ('call/call.json',lambda v:v.update(response_id='different')),
    ('call/call.json',lambda v:v['initialization'][0].update(frame=1)),
    ('call/segmentation.json',lambda v:v['frames'].pop()),
    ('call/segmentation.json',lambda v:v['frames'][0].update(frame=True)),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'].pop()),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(object_id='2')),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(source_dimensions=[99,80])),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(box_confidence=1.1)),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(box_xywh=[0,0,7,8])),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(status='missing')),
    ('call/segmentation.json',lambda v:v['frames'][0]['objects'][0].update(mask_encoded='decoded?')),
])
def test_sealed_but_semantically_conflicting_inputs_fail_closed(tmp_path,name,mutation):
    root=fixture(tmp_path/'run')
    mutate(root,name,mutation)
    with pytest.raises(ValueError):
        adapt_tracking_run(root)


def test_tampering_unsealed_required_file_and_path_escape_are_rejected(tmp_path):
    root=fixture(tmp_path/'run')
    (root/'call/segmentation.json').write_text('{}')
    with pytest.raises(ValueError,match='integrity'):
        adapt_tracking_run(root)
    seal(root)
    mutate(root,'manifest.json',lambda v:v['files'].pop('call/segmentation.json'))
    # mutate reseals manifests, so remove the entry directly for this check.
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['files'].pop('call/segmentation.json')
    write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='Required'):
        adapt_tracking_run(root)
    manifest['files']['../outside']='0'*64
    manifest['files']['call/segmentation.json']=digest(root/'call/segmentation.json')
    write_json(root/'manifest.json',manifest)
    with pytest.raises(ValueError,match='path'):
        adapt_tracking_run(root)


def test_resource_envelopes_apply_before_unbounded_work(tmp_path,monkeypatch):
    root=fixture(tmp_path/'run')
    monkeypatch.setitem(CONTRACT,'maximum_bundle_bytes',10)
    with pytest.raises(ValueError,match='byte bound'):
        adapt_tracking_run(root)


def test_maximum_500_frames_eight_objects_has_bounded_pair_checks(tmp_path):
    boxes=[[[i*10,20,i*10+4,24] for i in range(8)] for _ in range(500)]
    evidence=adapt_tracking_run(fixture(tmp_path/'run',boxes,ids=tuple(range(1,9))))
    assert len(evidence['observations'])==4000
    assert len(evidence['frames'])==500
    assert evidence['provenance']['pair_checks']==14000
    assert evidence['proximity_events']==[]


def mapping_bundle(root, run, *, legacy=False):
    root.mkdir()
    (root/'original.webm').write_bytes(b'original synthetic source')
    (root/'submitted.mp4').write_bytes((run/'source.mp4').read_bytes())
    metadata=json.loads((run/'video.json').read_text())
    report={'source_sha256':digest(root/'original.webm'),'derivative_sha256':digest(root/'submitted.mp4'),
            'frame_dimensions' if legacy else 'dimensions':[100,80],'all_frame_mapping_verified':True}
    if legacy:
        report['source_frame0_roi']=[0,0,6,8]
        prompt='<0f>1<|box;x1=0;y1=0;x2=5;y2=7;w=100;h=80|>'
        write_json(root/'request-template.json',{'model':'sam-3.1','metadata':{'propagation_direction':'forward'},
            'input':[{'content':[{'type':'input_text','text':prompt},{'type':'input_video','video_url':'template'}]}]})
    offset=0 if legacy else 1000
    rows=[]
    for index,time_s in enumerate(metadata['decoder_timestamps_s']):
        rows.append({'source_frame':index if legacy else 60+index,
                     'derivative_frame' if legacy else 'relative_frame':index,
                     'source_pts':offset+round(time_s*1000),
                     'derivative_pts' if legacy else 'relative_pts':round(time_s*1000)})
    mapping={'time_base':'1/1000','frames':rows}
    if not legacy:
        mapping['source_pts_offset']=offset
    write_json(root/('frame-mapping.json' if legacy else 'frame-mapping-and-pixels.json'),mapping)
    write_json(root/'report.json',report)
    seal(root)
    return root


def test_verified_mapping_preserves_original_and_submitted_source_identities(tmp_path):
    root=fixture(tmp_path/'run',times=[0,.016,.033])
    mapping=mapping_bundle(tmp_path/'mapping',root)
    evidence=adapt_tracking_run(root,source_mapping_bundle=mapping)
    original_hash=digest(mapping/'original.webm')
    assert evidence['source']['original_source_sha256']==original_hash
    assert evidence['source']['submitted_video_sha256']!=original_hash
    assert evidence['observations'][-1]['frame']==2
    assert evidence['observations'][-1]['time_s']==.033
    assert evidence['observations'][-1]['original_source']=={'sha256':original_hash,'frame':62,'time_s':1.033}
    assert ':frame:2:' in evidence['observations'][-1]['id']
    assert evidence['provenance']['source_mapping']['mapping_format']=='frame-mapping-and-pixels.json'


def legacy_run(root):
    root=fixture(root,[[[0,0,6,8]],[[20,0,26,8]]],ids=(1,),times=[0,.017])
    for name in ('call/call.json','call/segmentation.json'):
        mutate(root,name,lambda value:value.pop('initialization'))
    mutate(root,'report.json',lambda value:value.pop('objects'))
    def flatten(value):
        value.pop('object_ids')
        value['frames']=[row['objects'][0] for row in value['frames']]
    mutate(root,'call/segmentation.json',flatten)
    return root


def test_legacy_single_object_requires_pinned_sealed_prompt_not_frame_zero_inference(tmp_path):
    root=legacy_run(tmp_path/'run')
    with pytest.raises(ValueError,match='Initialization provenance'):
        adapt_tracking_run(root)
    mapping=mapping_bundle(tmp_path/'mapping',root,legacy=True)
    evidence=adapt_tracking_run(root,source_mapping_bundle=mapping)
    assert evidence['objects'][0]['initialization']=={'frame':0,'roi':[0,0,6,8]}
    assert evidence['observations'][0]['phase']=='initialization'
    assert evidence['observations'][1]['original_source']['frame']==1
    assert evidence['provenance']['source_mapping']['mapping_format']=='frame-mapping.json'
    mutate(mapping,'request-template.json',lambda value:value['input'][0]['content'][0].update(text='a concept instead of a box'))
    with pytest.raises(ValueError,match='Legacy prompt'):
        adapt_tracking_run(root,source_mapping_bundle=mapping)


@pytest.mark.parametrize('name,mutation',[
    ('report.json',lambda v:v.update(derivative_sha256='0'*64)),
    ('report.json',lambda v:v.update(source_sha256='0'*64)),
    ('report.json',lambda v:v.update(all_frame_mapping_verified=False)),
    ('frame-mapping-and-pixels.json',lambda v:v['frames'][1].update(relative_pts=16)),
    ('frame-mapping-and-pixels.json',lambda v:v['frames'][1].update(source_frame=60)),
    ('frame-mapping-and-pixels.json',lambda v:v['frames'][1].update(relative_frame=True)),
    ('frame-mapping-and-pixels.json',lambda v:v.update(time_base='1/0')),
])
def test_mapping_does_not_fabricate_or_rebase_source_time(tmp_path,name,mutation):
    root=fixture(tmp_path/'run')
    mapping=mapping_bundle(tmp_path/'mapping',root)
    mutate(mapping,name,mutation)
    with pytest.raises(ValueError):
        adapt_tracking_run(root,source_mapping_bundle=mapping)


def test_ambiguity_threshold_is_scale_relative_not_a_fixed_pixel_radius(tmp_path):
    base=fixture(tmp_path/'base',[[[0,0,6,8],[11,0,17,8]]])
    scaled=fixture(tmp_path/'scaled',[[[0,0,12,16],[22,0,34,16]]])
    a,b=adapt_tracking_run(base),adapt_tracking_run(scaled)
    assert len(a['proximity_events'])==len(b['proximity_events'])==1
    assert b['proximity_events'][0]['threshold_px']==2*a['proximity_events'][0]['threshold_px']
    assert b['proximity_events'][0]['edge_gap_px']==2*a['proximity_events'][0]['edge_gap_px']
