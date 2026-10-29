from copy import deepcopy

import cv2
import numpy as np
import pytest

from weft.io import digest
from weft.recording import Recorder, load_recording
from weft.scene_console import console
from weft.spatial_rigid import SpatialRigidEnv, render_snapshot, validate_snapshot
from test_scene_console import sealed_run
from test_spatial_rigid import scene


def test_all_spatial_controls_export_and_replay_without_mutating_inputs(tmp_path):
    value=scene();run=sealed_run(tmp_path/'run',value)
    before={p.name:digest(p) for p in run.iterdir()}
    commands=iter(['status','help','coast 2','left 2','right 2','forward 2','back 2','up 2','down 2','brake 2','reset','quit'])
    output=[];recording=tmp_path/'recording'
    result=console(run,recording,input_fn=lambda _:next(commands),output_fn=output.append)
    saved=load_recording(recording)
    assert result['steps']==16 and result['resets']==1 and result['video_frames']==18
    assert [e['action'] for e in saved['events'] if e['type']=='step']==[i for i in range(8) for _ in range(2)]
    assert any('3D rigid-object hypothesis' in line for line in output)
    assert any('forward/back move depth' in line for line in output)
    env=SpatialRigidEnv(value)
    for event in saved['events']:
        if event['type'] in {'start','reset'}:env.reset(seed=event['seed'],options={'jitter':False})
        else:
            _,reward,terminated,truncated,_=env.step(event['action'])
            assert (reward,terminated,truncated)==(event['reward'],event['terminated'],event['truncated'])
        assert env.snapshot()==event['snapshot']
    assert before=={p.name:digest(p) for p in run.iterdir()}
    page=(recording/'replay.html').read_text()
    assert '__WEFT_DATA__' not in page and 'new Worker(url)' in page
    assert '3D recorded simulation' in page and 'unverified' in page
    cap=cv2.VideoCapture(str(recording/'video.mp4'));frames=0
    while cap.read()[0]:frames+=1
    cap.release();assert frames==18


@pytest.mark.parametrize('change',[
    lambda s:s.pop('units'),lambda s:s.update(contact_count=1),lambda s:s.update(time_s=.1),
    lambda s:s['objects'][1]['position'].__setitem__(0,.6),lambda s:s.update(physical_validation='verified'),
    lambda s:s['objects'][0].pop('quaternion'),
])
def test_invalid_spatial_recordings_are_rejected_before_directory_creation(tmp_path,change):
    env=SpatialRigidEnv(scene());env.reset();state=deepcopy(env.snapshot());change(state)
    recorder=Recorder(tmp_path/'recording',env.scene)
    with pytest.raises(ValueError):recorder.start(state)
    assert not (tmp_path/'recording').exists()


def test_quaternion_roundoff_is_preserved_but_nonunit_orientation_is_rejected():
    env=SpatialRigidEnv(scene());env.reset();state=env.snapshot()
    state['objects'][0]['quaternion']=[1.0000000000000002,0,0,0]
    validate_snapshot(state,env.scene)
    assert state['objects'][0]['quaternion'][0]==1.0000000000000002
    state['objects'][0]['quaternion'][0]=1.0001
    with pytest.raises(ValueError):validate_snapshot(state,env.scene)


def test_support_wireframes_leave_the_actor_visible_in_all_three_views():
    value=scene();value['objects'][0]['position']=[.45,.4,.06501]
    env=SpatialRigidEnv(value);env.reset();before=env.snapshot()
    image=render_snapshot(before,env.scene,width=960)
    for pane in range(3):
        pixels=image[:326,pane*320:(pane+1)*320]
        assert np.all(pixels==[171,126,72],axis=2).sum()>20
    assert env.snapshot()==before
