"""Frame-coded software fixtures check decoding boundaries, not physical reconstruction."""
import cv2
import numpy as np
import pytest

import weft.video_interval as intervals
from weft.io import digest
from weft.video_interval import IntervalVideo


class FakeCapture:
    def __init__(self, *, actual_frames=100, metadata=None):
        self.values={cv2.CAP_PROP_FPS:30.,cv2.CAP_PROP_FRAME_COUNT:100.,cv2.CAP_PROP_FRAME_WIDTH:96.,cv2.CAP_PROP_FRAME_HEIGHT:64.}
        self.values.update(metadata or {})
        self.actual_frames=actual_frames;self.position=0;self.grabs=0;self.reads=0;self.releases=0
        self.opened=True;self.on_grab=None;self.on_read=None
        self.positions=None;self.timestamps=None;self.bad_shape=False

    def isOpened(self):return self.opened

    def get(self,key):
        if key==cv2.CAP_PROP_POS_FRAMES:
            return self.positions(self.position) if self.positions else float(self.position)
        if key==cv2.CAP_PROP_POS_MSEC:
            return self.timestamps(self.position-1) if self.timestamps else max(0,self.position-1)*1000/self.values[cv2.CAP_PROP_FPS]
        return self.values[key]

    def grab(self):
        self.grabs+=1
        if self.position>=self.actual_frames:return False
        self.position+=1
        if self.on_grab:self.on_grab()
        return True

    def read(self):
        self.reads+=1
        if self.position>=self.actual_frames:return False,None
        shape=(64,96) if self.bad_shape else (64,96,3)
        frame=np.full(shape,self.position%256,np.uint8);self.position+=1
        if self.on_read:self.on_read()
        return True,frame

    def release(self):self.releases+=1


def mocked_reader(tmp_path,monkeypatch,first=3,last=6,*,capture=None,timeout_seconds=120):
    source=tmp_path/'source.bin';source.write_bytes(b'fixed fake source for decoder boundary tests')
    capture=capture or FakeCapture()
    monkeypatch.setattr(intervals.cv2,'VideoCapture',lambda path:capture)
    return IntervalVideo(source,first,last,digest(source),timeout_seconds=timeout_seconds),capture,source


def test_frame_coded_real_video_preserves_absolute_indices_and_never_reads_tail(tmp_path,monkeypatch):
    source=tmp_path/'coded.avi';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'FFV1'),24,(96,64))
    assert writer.isOpened()
    for index in range(80):writer.write(np.full((64,96,3),index,np.uint8))
    writer.release()
    factory=cv2.VideoCapture
    class Spy:
        def __init__(self,path):self.inner=factory(path);self.grabs=0;self.reads=0;self.releases=0
        def isOpened(self):return self.inner.isOpened()
        def get(self,key):return self.inner.get(key)
        def grab(self):self.grabs+=1;return self.inner.grab()
        def read(self):self.reads+=1;return self.inner.read()
        def release(self):self.releases+=1;return self.inner.release()
    capture=Spy(str(source));monkeypatch.setattr(intervals.cv2,'VideoCapture',lambda path:capture)
    reader=IntervalVideo(source,12,20,digest(source))
    with reader:
        assert reader.metadata['frame_count']==80
        assert reader.metadata['fps']==24
        observed=[(index,int(frame[0,0,0])) for index,frame in reader]
    assert observed==[(index,index) for index in range(12,21)]
    assert capture.grabs==12 and capture.reads==9 and capture.releases==1
    audit=reader.audit
    assert audit['selected_interval']==[12,20] and audit['prefix_advanced_frames']==12
    assert audit['selected_frames_complete'] is True and audit['decoded_frames']==9
    assert audit['source_complete'] is False and audit['post_window_frames_decoded']==0
    assert audit['post_window_frames_metadata']==59 and audit['post_window_frame_count_actual'] is None
    assert audit['source_frame_count_actual'] is None
    assert audit['decoder_ordinals_complete'] is True and audit['selected_ordinal_skips']==0
    assert audit['encoded_frame_skips'] is None
    assert audit['encoded_frame_completeness']=='not_independently_verified'
    assert 'selected_frame_skips' not in audit
    assert audit['timestamp_cfr_check']['status']=='verified_cfr_cadence'
    assert [row['source_frame'] for row in audit['decoder_timestamps']]==list(range(12,21))
    assert audit['decoder_timestamps'][0]['decoder_time_ms']==pytest.approx(500)
    assert audit['decoder_timestamps'][-1]['nominal_time_s']==pytest.approx(20/24)


def test_long_metadata_source_does_not_certify_or_decode_unselected_tail(tmp_path,monkeypatch):
    cap=FakeCapture(actual_frames=3,metadata={cv2.CAP_PROP_FRAME_COUNT:216000,cv2.CAP_PROP_FPS:60})
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2,capture=cap)
    with reader:assert len(list(reader))==3
    assert cap.reads==3 and cap.grabs==0
    assert reader.audit['metadata_frames']==216000
    assert reader.audit['decoded_frames']==3 and reader.audit['source_complete'] is False


def test_stream_is_lazy_single_pass_and_does_not_reuse_frames(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch)
    with reader:
        stream=iter(reader)
        assert cap.grabs==0 and cap.reads==0
        first_index,first=next(stream)
        assert first_index==3 and cap.reads==1
        rest=list(stream)
        assert int(first[0,0,0])==3 and len(rest)==3
        with pytest.raises(RuntimeError,match='one streaming pass'):iter(reader)
    with pytest.raises(RuntimeError,match='open context'):iter(reader)
    with pytest.raises(RuntimeError,match='reopened'):reader.__enter__()


@pytest.mark.parametrize('actual,first,last,expected_prefix,expected_reads',[(2,4,6,2,0),(5,3,8,3,3)])
def test_early_eof_never_becomes_success(tmp_path,monkeypatch,actual,first,last,expected_prefix,expected_reads):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,first,last,capture=FakeCapture(actual_frames=actual))
    with pytest.raises(ValueError,match='EOF'):
        with reader:list(reader)
    assert cap.releases==1 and cap.reads==expected_reads
    assert reader.audit['prefix_advanced_frames']==expected_prefix
    assert reader.audit['selected_frames_complete'] is False
    assert reader.audit['state']=='failed' and reader.audit['source_complete'] is False


def test_early_consumer_stop_raises_and_releases(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='completely consumed'):
        with reader:
            for _ in reader:break
    assert cap.releases==1 and cap.reads==1
    assert reader.audit['selected_frames_complete'] is False


def test_consumer_exception_is_preserved_and_capture_released(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch)
    with pytest.raises(LookupError,match='consumer failure'):
        with reader:
            for _ in reader:raise LookupError('consumer failure')
    assert cap.releases==1 and reader.audit['failure']['type']=='LookupError'


def test_suppressed_decode_failure_cannot_close_as_success(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,capture=FakeCapture(actual_frames=4))
    with pytest.raises(ValueError,match='previously failed'):
        with reader:
            with pytest.raises(ValueError,match='EOF'):list(reader)
    assert cap.releases==1


def test_deadline_records_completed_prefix_advance_before_cleanup(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,first=1,last=2,timeout_seconds=1)
    clock=[0.];monkeypatch.setattr(intervals.time,'monotonic',lambda:clock[0])
    cap.on_grab=lambda:clock.__setitem__(0,2.)
    with pytest.raises(TimeoutError):
        with reader:list(reader)
    assert cap.releases==1 and reader.audit['prefix_advanced_frames']==1
    assert reader.audit['decoded_frames']==0 and reader.audit['selected_frames_complete'] is False


def test_source_hash_is_checked_before_opening_decoder(tmp_path,monkeypatch):
    path=tmp_path/'source';path.write_bytes(b'different source')
    opened=[];monkeypatch.setattr(intervals.cv2,'VideoCapture',lambda path:opened.append(path))
    with pytest.raises(ValueError,match='checksum'):
        with IntervalVideo(path,0,2,'0'*64):pass
    assert not opened


def test_source_change_during_decode_invalidates_interval(tmp_path,monkeypatch):
    reader,cap,path=mocked_reader(tmp_path,monkeypatch,0,1)
    cap.on_read=lambda:path.write_bytes(b'changed bytes during decode')
    with pytest.raises(ValueError,match='source changed'):
        with reader:list(reader)
    assert cap.releases==1 and reader.audit['selected_frames_complete'] is False


def test_source_change_after_final_yield_is_checked_before_context_success(tmp_path,monkeypatch):
    reader,cap,path=mocked_reader(tmp_path,monkeypatch,0,1)
    with pytest.raises(ValueError,match='source changed'):
        with reader:
            list(reader)
            path.write_bytes(b'changed after selected frames were decoded')
    assert cap.releases==1 and reader.audit['state']=='failed'
    assert reader.audit['source_stat_unchanged'] is False


def test_missing_and_oversized_sources_fail_without_decoder(tmp_path,monkeypatch):
    opened=[];monkeypatch.setattr(intervals.cv2,'VideoCapture',lambda path:opened.append(path))
    with pytest.raises(FileNotFoundError):
        with IntervalVideo(tmp_path/'missing',0,1,'0'*64):pass
    large=tmp_path/'large'
    with large.open('wb') as stream:stream.truncate(100000001)
    with pytest.raises(ValueError,match='100 MB'):
        with IntervalVideo(large,0,1,'0'*64):pass
    assert not opened


@pytest.mark.parametrize('first,last,sha,timeout',[
    (-1,2,'0'*64,120),(2,1,'0'*64,120),(True,2,'0'*64,120),(0,2.0,'0'*64,120),
    (20001,20002,'0'*64,120),(0,1800,'0'*64,120),(0,1,'invalid',120),
    (0,1,'A'*64,120),(0,1,'0'*64,0),(0,1,'0'*64,121),(0,1,'0'*64,float('inf')),
])
def test_invalid_interval_arguments_fail_without_io(first,last,sha,timeout):
    with pytest.raises(ValueError):IntervalVideo('unused',first,last,sha,timeout_seconds=timeout)


@pytest.mark.parametrize('metadata',[
    {cv2.CAP_PROP_FPS:0},{cv2.CAP_PROP_FPS:241},{cv2.CAP_PROP_FPS:float('nan')},
    {cv2.CAP_PROP_FRAME_COUNT:216001},{cv2.CAP_PROP_FRAME_COUNT:108001,cv2.CAP_PROP_FPS:30},
    {cv2.CAP_PROP_FRAME_COUNT:10.5},{cv2.CAP_PROP_FRAME_WIDTH:4097},{cv2.CAP_PROP_FRAME_HEIGHT:2161},
    {cv2.CAP_PROP_FRAME_WIDTH:32},{cv2.CAP_PROP_FRAME_HEIGHT:float('inf')},
])
def test_metadata_bounds_release_capture(tmp_path,monkeypatch,metadata):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,1,capture=FakeCapture(metadata=metadata))
    with pytest.raises(ValueError):
        with reader:pass
    assert cap.releases==1 and reader.audit['selected_frames_complete'] is False


def test_selected_duration_bound_and_underreported_metadata_not_repaired(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,60,capture=FakeCapture(metadata={cv2.CAP_PROP_FPS:1}))
    with pytest.raises(ValueError,match='60 nominal seconds'):
        with reader:pass
    assert cap.releases==1
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,10,capture=FakeCapture(metadata={cv2.CAP_PROP_FRAME_COUNT:10}))
    with pytest.raises(ValueError,match='advertised source'):
        with reader:pass
    assert cap.reads==0 and cap.releases==1


def test_decoder_position_jump_is_not_silently_renumbered(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2)
    cap.positions=lambda pos:pos+1 if pos else 0
    with pytest.raises(ValueError,match='jumped or reordered'):
        with reader:list(reader)
    assert cap.releases==1 and reader.audit['decoded_frames']==0


def test_raw_timestamps_are_distinct_from_nominal_fps_times(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,2,4)
    stamps={2:250.,3:283.333,4:316.667};cap.timestamps=lambda index:stamps[index]
    with reader:list(reader)
    rows=reader.audit['decoder_timestamps']
    assert [row['decoder_time_ms'] for row in rows]==[250,283.333,316.667]
    assert rows[0]['nominal_time_s']==pytest.approx(2/30)


def test_unavailable_and_repeated_timestamps_are_explicit(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2)
    cap.timestamps=lambda index:float('nan');cap.positions=lambda pos:float('nan')
    with reader:list(reader)
    assert all(row['decoder_time_ms'] is None for row in reader.audit['decoder_timestamps'])
    assert reader.audit['decoder_positions_unavailable']==4
    assert reader.audit['timestamp_cfr_check']['status']=='unavailable'
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2)
    cap.timestamps=lambda index:0.
    with reader:list(reader)
    assert reader.audit['decoder_timestamp_repeats']==2
    assert reader.audit['timestamp_cfr_check']['status']=='unavailable_all_zero'
    assert reader.audit['timestamp_cfr_check']['complete_timestamp_coverage'] is False


def test_forward_timestamp_gap_fails_even_when_decoder_positions_advance_normally(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,4)
    cap.timestamps=lambda index:(index+int(index>=2))*1000/30
    with pytest.raises(ValueError,match='timestamp gap'):
        with reader:list(reader)
    audit=reader.audit
    assert cap.position==3 and cap.reads==3 and cap.releases==1
    assert audit['decoded_frames']==2 and audit['selected_frames_complete'] is False
    assert audit['timestamp_cfr_check']['status']=='failed'
    assert audit['timestamp_cfr_check']['failure']['ordinal_distance']==1
    assert audit['decoder_timestamps'][-1]['decoder_next_frame_position']==3
    assert audit['decoder_timestamps'][-1]['decoder_time_ms']==100
    assert audit['decoder_timestamps'][-1]['accepted_for_stream'] is False


def test_repeated_positive_timestamp_stops_decode(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,4)
    cap.timestamps=lambda index:100.
    with pytest.raises(ValueError,match='repeats a usable'):
        with reader:list(reader)
    assert cap.reads==2 and cap.releases==1
    assert reader.audit['decoder_timestamp_repeats']==1


def test_initial_repeated_zeros_do_not_retroactively_become_verified_timing(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,4)
    cap.timestamps=lambda index:0. if index<2 else index*1000/30
    with pytest.raises(ValueError,match='Repeated initial zero'):
        with reader:list(reader)
    assert cap.reads==3 and reader.audit['timestamp_cfr_check']['status']=='failed'
    assert reader.audit['selected_frames_complete'] is False


@pytest.mark.parametrize('fps',[24.,30.,60.,240.])
def test_millisecond_timestamp_quantization_passes_predeclared_tolerance(tmp_path,monkeypatch,fps):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,2,10,capture=FakeCapture(metadata={cv2.CAP_PROP_FPS:fps}))
    cap.timestamps=lambda index:float(int(index*1000/fps))
    with reader:list(reader)
    check=reader.audit['timestamp_cfr_check']
    assert check['status']=='verified_cfr_cadence'
    assert check['tolerance_ms']==pytest.approx(max(2.,.1*1000/fps))
    assert check['maximum_step_error_ms']<=1


def test_missing_timestamps_preserve_unverified_coverage_and_use_actual_ordinal_gap(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,4)
    cap.timestamps=lambda index:float('nan') if index==2 else 100+index*1000/30
    with reader:list(reader)
    check=reader.audit['timestamp_cfr_check']
    assert check['status']=='unverified_missing_timestamps'
    assert check['complete_timestamp_coverage'] is False
    assert check['unavailable_timestamps']==1 and check['checked_intervals_spanning_missing']==1
    assert reader.audit['decoder_ordinals_complete'] is True
    assert reader.audit['encoded_frame_skips'] is None


def test_missing_timestamp_cannot_hide_a_forward_clock_gap(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,4)
    cap.timestamps=lambda index:float('nan') if index==2 else 100+(index+int(index>=3))*1000/30
    with pytest.raises(ValueError,match='timestamp gap'):
        with reader:list(reader)
    failure=reader.audit['timestamp_cfr_check']['failure']
    assert failure['ordinal_distance']==2
    assert failure['expected_delta_ms']==pytest.approx(2000/30)
    assert cap.reads==4


def test_single_frame_has_ordinal_coverage_but_cannot_verify_cadence(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,3,3)
    with reader:list(reader)
    assert reader.audit['decoder_ordinals_complete'] is True
    assert reader.audit['timestamp_cfr_check']['status']=='unverified_insufficient_samples'


def test_backward_timestamps_and_invalid_decoded_shape_fail(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2)
    cap.timestamps=lambda index:100-index*10
    with pytest.raises(ValueError,match='timestamps move backward'):
        with reader:list(reader)
    assert cap.releases==1
    reader,cap,_=mocked_reader(tmp_path,monkeypatch,0,2)
    cap.bad_shape=True
    with pytest.raises(ValueError,match='BGR frame'):
        with reader:list(reader)
    assert cap.releases==1


def test_unopenable_decoder_releases_and_reader_requires_context(tmp_path,monkeypatch):
    reader,cap,_=mocked_reader(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='context'):iter(reader)
    cap.opened=False
    with pytest.raises(ValueError,match='opened'):
        with reader:pass
    assert cap.releases==1
