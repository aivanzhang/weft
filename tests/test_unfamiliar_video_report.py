"""Portable report escaping and all-case bounds; fixture bytes are not decoded video."""
import importlib.util,json
from pathlib import Path
from hashlib import sha256
import pytest
s=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'scripts/studies/render_unfamiliar_video_report.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def fixture(tmp_path):
    rows=[];cases=[];(tmp_path/'normalized').mkdir()
    for i in range(3):
        name='case'+str(i);video=tmp_path/(name+'.mp4');video.write_bytes(bytes([i])*10)
        p=tmp_path/(name+'.json');p.write_text(json.dumps({'title':'<script>bad</script>','author':'Person','source_page':'https://example.com/source','license_url':'https://example.com/license','license':'CC0'}))
        cases.append({'id':name,'upload':str(video),'upload_sha256':sha256(video.read_bytes()).hexdigest(),'original_sha256':'a'*64,'provenance':str(p)})
        rows.append({'id':name,'status':'failed','stage':'preflight','error':'<img src=x>','decoded_frames':12,'usable_frames':1,'sampled_frames':6,'usable_fraction':1/6,'seconds':1,'peak_rss_bytes':1000000,'status_counts':{'multiple_people':5},'competing_frames':[],'raw_pose_sha256':'b'*64,'raw_evidence_sha256':'c'*64})
        (tmp_path/'normalized'/(name+'-receipt.json')).write_text(json.dumps({'max_decoder_timestamp_difference_s':0}))
    (tmp_path/'evaluation-plan.json').write_text(json.dumps({'cases':cases}));(tmp_path/'assessment.json').write_text(json.dumps({'cases':rows}));return cases


def test_offline_report_preserves_three_failures_and_escapes(tmp_path):
    fixture(tmp_path);got=m.render(tmp_path,tmp_path/'report.html');raw=(tmp_path/'report.html').read_text()
    assert got['embedded_videos']==3 and got['network_dependencies']==0
    assert raw.count('data:video/mp4;base64,')==3 and '0/3 reached' in raw
    assert '<script>bad' not in raw and '&lt;script&gt;bad' in raw
    assert '<img src=x>' not in raw


def test_changed_video_refuses_report(tmp_path):
    cases=fixture(tmp_path);Path(cases[1]['upload']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):m.render(tmp_path,tmp_path/'report.html')
    assert not (tmp_path/'report.html').exists()


def test_size_bound_never_omits_case(tmp_path,monkeypatch):
    fixture(tmp_path);monkeypatch.setattr(m,'MAX_VIDEOS',21)
    with pytest.raises(ValueError,match='no cases omitted'):m.render(tmp_path,tmp_path/'report.html')
    assert not (tmp_path/'report.html').exists()


def test_other_outcomes_cannot_be_mislabeled_as_refusals(tmp_path):
    fixture(tmp_path);p=tmp_path/'assessment.json';d=json.loads(p.read_bytes());d['cases'][0]['status']='complete';p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='actual preflight refusals'):m.render(tmp_path,tmp_path/'report.html')
