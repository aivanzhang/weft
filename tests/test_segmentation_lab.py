import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from weft import segmentation_lab as lab
from weft.benchmark import seal_suite
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (96,64))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((64,96,3),i*10,np.uint8))
    writer.release()
    return path


def test_offline_preparation_never_calls_api_and_seals_exact_video(video,tmp_path,monkeypatch):
    monkeypatch.setattr(lab,'segment_video',lambda *a,**k: pytest.fail('Unexpected paid call'))
    result=lab.track_video(video,tmp_path/'prepared',[[1,2,10,10,8,8]])
    assert result['status']=='prepared_not_sent'
    assert result['video']['frame_count']==10
    assert result['video']['decoder_timestamps_s']==pytest.approx([i/10 for i in range(10)])
    assert digest(video)==digest(tmp_path/'prepared/source.mp4')
    payload=json.loads((tmp_path/'prepared/request.json').read_text())
    assert payload['input'][0]['content'][0]['text'].startswith('<2f>1')
    assert verify_artifacts(tmp_path/'prepared')['verified_files']==4


@pytest.mark.parametrize('budget',[None,0,.149,float('nan'),True])
def test_execution_requires_explicit_reservation_allowance(video,tmp_path,monkeypatch,budget):
    monkeypatch.setattr(lab,'segment_video',lambda *a,**k: pytest.fail('Unexpected paid call'))
    with pytest.raises(ValueError,match='budget'):
        lab.track_video(video,tmp_path/'bad',[[1,0,10,10,8,8]],execute=True,max_cost_usd=budget)
    assert not (tmp_path/'bad').exists()


def test_offline_view_preserves_original_seal_and_missing_slots(video,tmp_path):
    run=tmp_path/'run';(run/'call').mkdir(parents=True)
    record={'status':'completed','video_sha256':digest(video)}
    write_json(run/'call/call.json',record)
    write_json(run/'call/segmentation.json',{'frames':[{'frame':i,'objects':[
        {'object_id':1,'status':'missing','box_xyxy_halfopen':None,'box_confidence':None}]} for i in range(10)]})
    write_json(run/'report.json',{'status':'fixture'});original=seal_suite(run)
    result=lab.prepare_tracking_view(run,video,tmp_path/'view')
    assert result['new_api_calls']==0 and verify_artifacts(run)==original
    page=(tmp_path/'view/index.html').read_text()
    assert 'No API calls' in page and '"status": "missing"' in page
    assert 'fetch(' not in page and 'https://' not in page
    assert verify_artifacts(tmp_path/'view')['verified_files']==6


def test_source_identity_mismatch_rejects_view_before_output(video,tmp_path):
    run=tmp_path/'run';(run/'call').mkdir(parents=True)
    write_json(run/'call/call.json',{'status':'completed','video_sha256':'0'*64})
    write_json(run/'report.json',{'status':'fixture'});seal_suite(run)
    with pytest.raises(ValueError,match='exact video'):
        lab.prepare_tracking_view(run,video,tmp_path/'view')
    assert not (tmp_path/'view').exists()
