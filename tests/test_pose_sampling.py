"""Dense source observations retain clocks, alternatives and ordinary quality gates."""
from copy import deepcopy
import json

import pytest

from test_exercise_body_source import VIDEO, model_metadata, seal, forbid_runtime
from test_exercise_vision import inference_result, stub_detector, video
from weft import exercise_body_source as source, exercise_vision as vision
from weft.io import canonical_hash, digest, write_json


@pytest.mark.parametrize('mode',[None,True,1,'dense','ALL_FRAMES',{}])
def test_unknown_sampling_rejected_before_input_access(tmp_path,mode):
    with pytest.raises(ValueError,match='Pose sampling'):
        vision.inspect_exercise('missing',tmp_path,'missing',pose_sampling=mode)
    with pytest.raises(ValueError,match='Pose sampling'):
        source.prepare_exercise_body_source('missing',tmp_path/'out',pose_model='missing',pose_sampling=mode)
    assert not list(tmp_path.iterdir())


def test_default_contract_omits_new_fields_and_preserves_stride():
    assert source._options('video','raw') == {'pose_mode':'video','pose_identity':'raw'}
    assert vision.pose_sampling_metadata('bounded') is None
    assert vision.pose_sampling_stride('bounded',30.00131,458)==3
    assert vision.pose_sampling_stride('all_frames',30.00131,458)==1


@pytest.mark.parametrize('count,decoded',[(901,901),(900,901),(True,1),(10,0)])
def test_dense_metadata_and_actual_decode_both_bounded(count,decoded):
    with pytest.raises(ValueError,match='900'):
        vision.pose_sampling_stride('all_frames',30,count,decoded)


def test_dense_inference_observes_every_original_frame_and_keeps_detector_clock(tmp_path,monkeypatch):
    original=tmp_path/'source.mp4';video(original)
    detector=stub_detector(monkeypatch,lambda i:inference_result(i*.03))
    rows=[]
    evidence=vision.inspect_exercise(original,tmp_path,'model',pose_sampling='all_frames',decoder_clock_sink=rows.append)
    assert [r['frame'] for r in evidence['frames']]==list(range(30))
    assert len(rows)==30 and detector.timestamps==[round(i*1000/30) for i in range(30)]
    assert evidence['inference_stride']==1 and evidence['sample_rate_fps']==30
    assert evidence['pose_sampling']==evidence['model']['pose_sampling']==vision.pose_sampling_metadata('all_frames')
    assert evidence['model']['maximum_pose_hypotheses']==2
    assert all(r['time_s']==r['frame']/30 and r['decoder_timestamp_s']==rows[r['frame']]['decoder_timestamp_s'] for r in evidence['frames'])
    assert json.loads((tmp_path/'pose-landmarks.json').read_text())['frames']==evidence['frames']


def test_dense_does_not_choose_between_competing_poses(tmp_path,monkeypatch):
    original=tmp_path/'source.mp4';video(original)
    stub_detector(monkeypatch,lambda i:inference_result(i*.03,count=2 if i==11 else 1))
    with pytest.raises(vision.UnsupportedVideo,match='competing'):
        vision.inspect_exercise(original,tmp_path,'model',pose_sampling='all_frames')
    raw=json.loads((tmp_path/'pose-landmarks.json').read_text())
    assert len(raw['frames'])==30 and len(raw['frames'][11]['poses'])==2
    assert raw['frames'][11]['status']=='multiple_people'


def dense_run(root):
    root.mkdir();(root/'source.webm').write_bytes(VIDEO)
    pin=digest(root/'source.webm');metadata=vision.pose_sampling_metadata('all_frames')
    model=model_metadata()|{'pose_sampling':metadata}
    evidence={'schema_version':1,'profile':'exercise-pose-v1','source_sha256':pin,'width':640,'height':480,
        'fps':30.,'frame_count':30,'decoded_frames':30,'inference_stride':1,'sample_rate_fps':30.,
        'pose_sampling':metadata,'model':model,'assumptions':vision.ASSUMPTIONS[:],'scope':'Analytical dense fixture only','frames':[]}
    for i in range(30):
        row=vision.summarize_result(inference_result(i*.01))
        row.update(id=f'{pin}:frame:{i}',frame=i,time_s=i/30,decoder_timestamp_s=i/30+.003,inference_seconds=0.)
        evidence['frames'].append(row)
    write_json(root/'pose-landmarks.json',{'frames':evidence['frames'],'model':model,'scope':evidence['scope']})
    vision.resolve_pose_identity(evidence,root,'raw');vision.finalize_evidence(evidence,root)
    write_json(root/'source.json',{'artifact':'source.webm','sha256':pin})
    write_json(root/'run-config.json',{'profile':'exercise','pose_mode':'video','pose_identity':'raw','pose_sampling':'all_frames'})
    write_json(root/'model.json',model);seal(root)
    return root


def test_dense_native_saved_body_preserves_all_raw_slots(tmp_path,monkeypatch):
    run=dense_run(tmp_path/'pose');forbid_runtime(monkeypatch)
    e=json.loads((run/'evidence.json').read_text());pin=digest(run/'source.webm')
    clock={'schema_version':1,'profile':'source-native-clock-v1','source_sha256':pin,
        'evidence_sha256':digest(run/'evidence.json'),'evidence_canonical_sha256':canonical_hash(e),
        'metadata_frames':30,'decoded_frames':30,'nominal_fps':30.,'time_base':'1/30000','native_origin_pts':90,
        'native_to_reconstruction_offset_s':-.003,'decoder_minus_native_s':0.,'capture_clock_status':'unknown',
        'detector_clock_status':'preserved_nominal_frame_index_divided_by_fps',
        'rows':[{'frame':i,'pts':90+1000*i,'decoder_timestamp_s':i/30+.003} for i in range(30)]}
    clock_path=tmp_path/'native.json';write_json(clock_path,clock)
    out=source.prepare_exercise_body_source_from_evidence(run,tmp_path/'body',native_clock=clock_path)
    files={p.relative_to(out).as_posix():p.read_bytes() for p in out.rglob('*') if p.is_file()}
    body,observations=source.validate_body_source_files(files)
    assert files['evidence.json']==(run/'evidence.json').read_bytes()
    assert len(observations['observations'])==30 and observations['schema_version']==4
    assert json.loads(files['body-source.json'])['pose_options']['pose_sampling']=='all_frames'
    assert body['profile']=='exercise-v3'


@pytest.mark.parametrize('damage',['model_marker','evidence_marker','config_marker','stride','missing_frame','receipt_mutation','sample_rate'])
def test_dense_sampling_tampering_rejected(tmp_path,damage):
    run=dense_run(tmp_path/'pose');files={name:(run/name).read_bytes() for name in source.POSE_FILES}
    name={'model_marker':'model.json','evidence_marker':'evidence.json','config_marker':'run-config.json',
          'stride':'evidence.json','missing_frame':'evidence.json','receipt_mutation':'evidence.json','sample_rate':'evidence.json'}[damage]
    value=json.loads(files[name])
    if damage.endswith('_marker'):value.pop('pose_sampling')
    elif damage=='stride':value['inference_stride']=2
    elif damage=='missing_frame':value['frames'].pop(11)
    elif damage=='sample_rate':value['sample_rate_fps']=15.
    else:value['pose_sampling']['maximum_samples']=1000
    files[name]=json.dumps(value).encode()
    with pytest.raises(ValueError):
        source._validated_pose(files,source._options('video','raw',pose_sampling='all_frames'),digest(run/'source.webm'))


def test_underreported_metadata_stops_before_inference_901_and_preserves_actual_read(tmp_path,monkeypatch):
    import numpy as np
    source_path=tmp_path/'underreported.mp4';source_path.write_bytes(b'Analytical decoder fixture')
    class Capture:
        index=0
        closed=False
        def isOpened(self):return True
        def get(self,key):
            return {vision.cv2.CAP_PROP_FPS:30.,vision.cv2.CAP_PROP_FRAME_COUNT:900.,
                    vision.cv2.CAP_PROP_FRAME_WIDTH:64.,vision.cv2.CAP_PROP_FRAME_HEIGHT:64.,
                    vision.cv2.CAP_PROP_POS_MSEC:max(0,self.index-1)*1000/30}[key]
        def read(self):
            self.index+=1
            return (True,np.zeros((64,64,3),dtype=np.uint8)) if self.index<=901 else (False,None)
        def release(self):self.closed=True
    capture=Capture();monkeypatch.setattr(vision.cv2,'VideoCapture',lambda _:capture)
    detector=stub_detector(monkeypatch,lambda _:inference_result(count=0))
    rows=[]
    with pytest.raises(ValueError,match='900 decoded-frame'):
        vision.inspect_exercise(source_path,tmp_path,'model',pose_sampling='all_frames',decoder_clock_sink=rows.append)
    saved=json.loads((tmp_path/'evidence.json').read_text())
    assert saved['decoded_frames']==901 and saved['frame_count']==900
    assert len(saved['frames'])==detector.sample==900
    assert len(rows)==901 and rows[-1]['frame']==900
    assert 'quality' not in saved and capture.closed and detector.closed
