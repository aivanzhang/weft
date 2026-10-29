"""Explicit, offline-first video tracking inspection; never promotes boxes to physics."""

import json
import math
from pathlib import Path
import shutil
import time

import cv2

from weft.benchmark import seal_suite
from weft.io import digest, local_time, write_json
from weft.meta_segmentation import build_request, segment_video


def inspect_video(video):
    video = Path(video)
    if video.suffix.lower() != '.mp4' or not 1 <= video.stat().st_size <= 37_000_000:
        raise ValueError('Use an MP4 of at most 37 MB; conversion is never automatic')
    source_hash = digest(video)
    capture = cv2.VideoCapture(str(video))
    started = time.monotonic()
    timestamps = []
    try:
        if not capture.isOpened():
            raise ValueError('Cannot decode input MP4')
        width, height, count = [capture.get(prop) for prop in
                               (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT)]
        if (any(not math.isfinite(x) or x != int(x) for x in (width,height,count))
                or not 1 <= count <= 500 or not 16 <= width <= 4096 or not 16 <= height <= 4096
                or width * height > 8_847_360):
            raise ValueError('Video must contain 1–500 frames within the supported size envelope')
        for index in range(int(count) + 1):
            if time.monotonic() - started > 30:
                raise TimeoutError('Local decode exceeded its between-frame time bound')
            ok, frame = capture.read()
            if not ok:
                break
            if index >= count or frame.shape[:2] != (int(height), int(width)):
                raise ValueError('Decoded video differs from container count or dimensions')
            stamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(stamp) or stamp < 0 or (timestamps and stamp <= timestamps[-1]):
                raise ValueError('Decoder did not expose strictly increasing frame timestamps')
            timestamps.append(stamp)
        if len(timestamps) != count or digest(video) != source_hash:
            raise ValueError('Input changed or decoded frame count is incomplete')
        return {'source_sha256': source_hash, 'width': int(width), 'height': int(height),
                'frame_count': int(count), 'decoder_timestamps_s': timestamps,
                'timestamp_scope': 'Decoder presentation times; not independent encoded-container verification',
                'seconds': time.monotonic() - started}
    finally:
        capture.release()


def _viewer(out, metadata, result):
    # Only bounded box metadata enters the viewer; raw masks remain in the evidence file.
    frames = [{'frame': row['frame'], 'objects': [{key: item.get(key) for key in
               ('object_id', 'status', 'box_xyxy_halfopen', 'box_confidence')}
               for item in row.get('objects', [row])]} for row in result['frames']]
    data = json.dumps({'frames': frames, 'times': metadata['decoder_timestamps_s'],
                       'width': metadata['width'], 'height': metadata['height']}).replace('<', '\\u003c')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Weft — tracking inspection</title><style>
body{font:16px system-ui;margin:24px auto;padding:0 16px;max-width:1000px;background:#F5EADC;color:#214F50}
button,a{color:inherit}button{padding:8px;margin:4px}canvas{width:100%;background:#111}video{display:none}
input{width:100%}pre{white-space:pre-wrap}small{display:block;margin:12px 0}
</style><h1>Tracking inspection</h1><p>Green boxes are Meta proposals, not verified objects or physics.</p>
<video id="video" src="source.mp4" muted playsinline preload="metadata"></video><canvas id="canvas"></canvas>
<div><button id="previous">Previous frame</button><button id="next">Next frame</button><span id="position"></span></div>
<label for="frame">Source frame</label><input id="frame" type="range" min="0" step="1" value="0">
<small>Video seeking is browser-dependent; the displayed image can differ slightly from the selected frame.
Missing boxes stay missing. Masks are retained but not decoded. No API calls occur in this viewer.</small>
<pre id="details"></pre><p><a href="report.json">Run report</a> · <a href="call/segmentation.json">All model output</a></p>
<script>const data=DATA;const video=document.getElementById('video'),canvas=document.getElementById('canvas'),
slider=document.getElementById('frame'),ctx=canvas.getContext('2d');canvas.width=data.width;canvas.height=data.height;
slider.max=data.frames.length-1;let selected=0;
function draw(){if(video.readyState<2)return;ctx.drawImage(video,0,0,canvas.width,canvas.height);
ctx.strokeStyle='#00ff70';ctx.fillStyle='#00ff70';ctx.lineWidth=2;ctx.font='18px system-ui';
for(const o of data.frames[selected].objects){const b=o.box_xyxy_halfopen;if(b){ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.fillText('Object '+o.object_id,b[0],Math.max(20,b[1]-5));}}
document.getElementById('position').textContent='Frame '+selected+' / '+slider.max;
document.getElementById('details').textContent=JSON.stringify(data.frames[selected],null,2);}
function select(n){selected=Math.min(data.frames.length-1,Math.max(0,n));slider.value=selected;
document.getElementById('position').textContent='Seeking frame '+selected;
const target=data.times[selected];if(Math.abs(video.currentTime-target)<1e-6){draw();}else{video.currentTime=target;}}
slider.addEventListener('input',()=>select(Number(slider.value)));video.addEventListener('seeked',draw);
video.addEventListener('loadeddata',()=>select(0));document.getElementById('previous').onclick=()=>select(selected-1);
document.getElementById('next').onclick=()=>select(selected+1);</script></html>'''
    (out / 'index.html').write_text(page.replace('DATA', data))


def track_video(video, out, boxes, *, execute=False, max_cost_usd=None, project_root=None):
    if not isinstance(boxes, list) or not 1 <= len(boxes) <= 8 or any(
            not isinstance(b, list) or len(b) != 6 or any(type(n) is not int for n in b) for b in boxes):
        raise ValueError('Provide 1–8 boxes: object ID, initialization frame, x, y, width, height')
    metadata = inspect_video(video)
    objects = [{'object_id': b[0], 'frame': b[1], 'roi': b[2:]} for b in boxes]
    payload = build_request(video, expected_sha256=metadata['source_sha256'], frame_count=metadata['frame_count'],
                            width=metadata['width'], height=metadata['height'], objects=objects)
    estimate = max(50, metadata['frame_count']) * len(objects) * .20 / 1000
    reservation = max(.15, estimate)
    if execute and (type(max_cost_usd) not in (int,float) or not math.isfinite(max_cost_usd) or max_cost_usd < reservation):
        raise ValueError(f'Explicit per-request budget must cover ${reservation:.6f}; shared budget also applies')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(video, out / 'source.mp4')
    if digest(out / 'source.mp4') != metadata['source_sha256']:
        raise ValueError('Input changed while copying the inspected video')
    write_json(out / 'video.json', metadata)
    (out / 'request.json').write_bytes(payload)
    report = {'status': 'prepared_not_sent', 'created_at': local_time(), 'video': metadata,
              'objects': objects, 'request_sha256': digest(out / 'request.json'), 'execute': execute,
              'conservative_estimate_usd': estimate, 'reservation_usd': reservation,
              'estimate_assumption': 'Frame rate multiplied by object count as conservative uncertainty envelope; settle provider frame usage',
              'scope': 'Tracking proposals only; no simulator, physical validation, or default changes'}
    try:
        if execute:
            report['call'] = segment_video(out / 'source.mp4', out / 'call',
                expected_sha256=metadata['source_sha256'], frame_count=metadata['frame_count'],
                width=metadata['width'], height=metadata['height'], objects=objects, project_root=project_root)
            result = json.loads((out / 'call/segmentation.json').read_text())
            _viewer(out, metadata, result)
            report['status'] = 'completed_tracking_not_physics'
    except Exception as error:
        report.update(status='failed_no_retry', error_type=type(error).__name__)
        raise
    finally:
        write_json(out / 'report.json', report)
        seal_suite(out)
    return report


def prepare_tracking_view(run, video, out):
    from weft.pipeline import verify_artifacts

    run, out = Path(run), Path(out)
    verification = verify_artifacts(run)
    metadata = inspect_video(video)
    record = json.loads((run / 'call/call.json').read_text())
    if record.get('status') != 'completed' or record.get('video_sha256') != metadata['source_sha256']:
        raise ValueError('Viewer requires completed tracking for this exact video')
    result = json.loads((run / 'call/segmentation.json').read_text())
    if len(result['frames']) != metadata['frame_count']:
        raise ValueError('Tracking and video frame counts differ')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'call').mkdir()
    shutil.copyfile(video, out / 'source.mp4')
    if digest(out / 'source.mp4') != metadata['source_sha256']:
        raise ValueError('Video changed during viewer preparation')
    for name in ('call.json','segmentation.json'):
        shutil.copyfile(run / 'call' / name, out / 'call' / name)
    write_json(out / 'video.json', metadata)
    _viewer(out, metadata, result)
    report = {'status': 'saved_tracking_view', 'created_at': local_time(), 'source_run': str(run),
              'source_verification': verification, 'source_call': record, 'video': metadata,
              'new_api_calls': 0, 'scope': 'Offline boxes; no physics or mask accuracy claim'}
    write_json(out / 'report.json', report)
    seal_suite(out)
    return report
