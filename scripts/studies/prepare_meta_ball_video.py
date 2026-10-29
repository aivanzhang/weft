"""Prepare the frozen real three-ball interval without inference or label-derived prompts."""

from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import argparse
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/meta-three-ball-input-01'
SOURCE = ROOT / 'data/balls/triple-collision-240p.webm'
FF = ROOT / '.venv-lab/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
EXPECTED = 'f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887'
SIZE = 426 * 240 * 3 // 2


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()


def save(name, data):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + '\n')


def execute(name, arguments):
    command = [str(FF), '-hide_banner', '-nostdin', '-n', *arguments]
    save(name + '-command.json', {'command': command, 'timeout_seconds': 120})
    started = time.monotonic()
    with (OUT / (name + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    save(name + '-result.json', {'returncode': result.returncode, 'seconds': time.monotonic()-started})
    if result.returncode:
        raise ValueError(f'{name} failed; evidence preserved')


def rows(name):
    return [[v.strip() for v in line.split(',')] for line in (OUT / name).read_text().splitlines()
            if not line.startswith('#')]


def metrics(histogram):
    count = int(histogram.sum())
    values = np.arange(256, dtype=np.int64)
    mse = float(np.dot(histogram, values * values) / count)
    return {'mae': float(np.dot(histogram, values) / count), 'rmse': math.sqrt(mse),
            'maximum_absolute_error': int(np.flatnonzero(histogram)[-1]),
            'p95_absolute_error': int(np.searchsorted(np.cumsum(histogram), math.ceil(.95 * count))),
            'psnr_db': 10 * math.log10(255**2 / mse) if mse else None, 'zero_mse': mse == 0,
            'sample_count': count}


def main(reuse_source=None):
    if digest(SOURCE) != EXPECTED:
        raise ValueError('Frozen source changed')
    initialization = json.loads(SOURCE.with_suffix('.initialization.json').read_text())
    scene = json.loads(SOURCE.with_suffix('.scene.json').read_text())
    if [initialization['start_frame'], initialization['end_frame_inclusive']] != [2580, 2760]:
        raise ValueError('Frozen interval changed')
    if initialization['initialization_frame'] != 2620:
        raise ValueError('Frozen initialization changed')
    OUT.mkdir(exist_ok=False)
    for path in [SOURCE, *[SOURCE.with_suffix(suffix) for suffix in
                          ('.initialization.json', '.scene.json', '.manual.json', '.source.json')]]:
        shutil.copyfile(path, OUT / path.name)
    shutil.copyfile(__file__, OUT / 'prepare_meta_ball_video.py')
    (OUT / 'ffmpeg-version.txt').write_bytes(subprocess.run([str(FF), '-version'], capture_output=True,
                                                          timeout=10, check=True).stdout)
    ids, tokens = [], []
    for index, obj in enumerate(initialization['objects'], 1):
        x, y, width, height = obj['bbox_xywh']
        other = next(r for r in scene['tracking']['objects'] if r['id'] == obj['id'])
        if other['box_xywh'] != obj['bbox_xywh']:
            raise ValueError('Original initialization and scene disagree')
        ids.append({'sam_id': index, 'object_id': obj['id'], 'source_initialization_frame': 2620,
                    'relative_initialization_frame': 40, 'source_box_xywh': obj['bbox_xywh'],
                    'inclusive_box': [x, y, x+width-1, y+height-1]})
        tokens.append(f'{index}<|box;x1={x};y1={y};x2={x+width-1};y2={y+height-1};w=426;h=240|>')
    request = {'model': 'sam-3.1', 'stream': True, 'store': False,
               'input': [{'type': 'message', 'role': 'user', 'content': [
                   {'type': 'input_text', 'text': '<40f>' + ','.join(tokens)},
                   {'type': 'input_video', 'video_url': 'data:video/mp4;base64,'}]}],
               'metadata': {'propagation_direction': 'forward', 'include_confidence': 'true', 'mask_encoding': 'one_bit'}}
    save('request-template.json', request)
    save('object-map.json', ids)
    save('preparation-freeze.json', {'created_at': now(), 'source_sha256': EXPECTED,
         'source_interval_inclusive': [2580, 2760], 'source_to_relative_frame': 'source_frame - 2580',
         'original_files': {p.name: digest(p) for p in OUT.iterdir() if p.is_file()},
         'encoder': 'Explicitly lossy libx264 CRF18, medium, threads2; native yuv420p',
         'request_prompt_source': 'Frozen initialization only; manual labels copied for later independent scoring',
         'full_source_duration_limit_not_bypassed': 'This is the previously frozen bounded real-time interval, not full212s source',
         'api_calls': 0})
    trim = 'trim=start_frame=2580:end_frame=2761'
    common = ['-map', '0:v:0', '-an', '-sn', '-dn', '-vf', trim, '-frames:v', '181',
              '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p', '-threads:v', '1', '-fps_mode', 'passthrough',
              '-enc_time_base:v', 'demux']
    if reuse_source is not None:
        for name in ['source-interval.yuv', 'source-frames.sha256', 'source-extraction-command.json',
                     'source-extraction.log', 'source-extraction-result.json']:
            shutil.copyfile(reuse_source / name, OUT / name)
        save('reused-source.json', {'path': str(reuse_source), 'source_sha256': EXPECTED,
                                   'raw_sha256': digest(OUT / 'source-interval.yuv')})
    else:
        execute('source-extraction', ['-threads', '1', '-copyts', '-i', str(SOURCE),
                *common, '-f', 'rawvideo', str(OUT / 'source-interval.yuv'),
                *common, '-f', 'framehash', '-hash', 'sha256', str(OUT / 'source-frames.sha256')])
    source_rows = rows('source-frames.sha256')
    if len(source_rows) != 181:
        raise ValueError('Source extraction did not retain181 frames')
    offset = int(source_rows[0][2])
    derivative = OUT / 'three-ball-crf18.mp4'
    unfinalized = OUT / 'three-ball-crf18-unfinalized.mp4'
    execute('encode', ['-threads', '2', '-copyts', '-i', str(SOURCE), '-map', '0:v:0', '-an', '-sn', '-dn',
            '-vf', trim + ',setpts=PTS-STARTPTS', '-frames:v', '181', '-c:v', 'libx264', '-crf', '18',
            '-preset', 'medium', '-pix_fmt', 'yuv420p', '-threads:v', '2', '-bf', '0', '-fps_mode', 'passthrough',
            '-enc_time_base:v', 'demux', '-video_track_timescale', '1000', '-movflags', '+faststart', str(unfinalized)])
    # The trim filter leaves a zero-duration last sample; remux without changing encoded pixels or PTS.
    execute('remux', ['-ignore_editlist', '1', '-i', str(unfinalized), '-map', '0:v:0', '-c:v', 'copy',
            '-bsf:v', 'setts=duration=16', '-video_track_timescale', '1000', '-movflags', '+faststart', str(derivative)])
    common = ['-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p',
              '-threads:v', '1', '-fps_mode', 'passthrough', '-enc_time_base:v', 'demux']
    execute('derivative-decode', ['-threads', '1', '-copyts', '-i', str(derivative),
            *common, '-f', 'rawvideo', str(OUT / 'derivative-interval.yuv'),
            *common, '-f', 'framehash', '-hash', 'sha256', str(OUT / 'derivative-frames.sha256')])
    derived_rows = rows('derivative-frames.sha256')
    if len(derived_rows) != 181 or any(int(a[2])-offset != int(b[2]) for a, b in zip(source_rows, derived_rows)):
        raise ValueError('Frame count or exact timestamp mapping changed')
    histogram = np.zeros(256, dtype=np.int64)
    comparisons = []
    with (OUT / 'source-interval.yuv').open('rb') as original, (OUT / 'derivative-interval.yuv').open('rb') as derived:
        for i, (a, b) in enumerate(zip(source_rows, derived_rows)):
            before, after = original.read(SIZE), derived.read(SIZE)
            if len(before) != SIZE or len(after) != SIZE or sha256(before).hexdigest() != a[5] or sha256(after).hexdigest() != b[5]:
                raise ValueError('Raw frame shape/hash differs from independent framehash output')
            difference = np.abs(np.frombuffer(before, dtype=np.uint8).astype(np.int16) -
                                np.frombuffer(after, dtype=np.uint8).astype(np.int16))
            counts = np.bincount(difference, minlength=256)
            histogram += counts
            comparisons.append({'source_frame': 2580+i, 'relative_frame': i, 'source_pts': int(a[2]),
                'relative_pts': int(b[2]), 'source_pixel_sha256': a[5], 'derivative_pixel_sha256': b[5],
                'source_duration_ticks': int(a[3]), 'derivative_duration_ticks': int(b[3]), **metrics(counts)})
        if original.read(1) or derived.read(1):
            raise ValueError('Unexpected frame suffix')
    save('frame-mapping-and-pixels.json', {'time_base': '1/1000', 'source_pts_offset': offset,
         'metric_space': 'Native8-bit YUV420p equally weighted samples, not perceptual or segmentation accuracy',
         'frames': comparisons, 'aggregate': metrics(histogram)})
    for path, name in [(SOURCE, 'source-metadata.log'), (derivative, 'derivative-metadata.log')]:
        (OUT / name).write_bytes(subprocess.run([str(FF), '-hide_banner', '-i', str(path)],
                                               capture_output=True, timeout=10).stderr)
    request_size = len(json.dumps(request, separators=(',', ':')).encode()) + 4*((derivative.stat().st_size+2)//3)
    report = {'status': 'ready_local_preparation_only' if request_size < 50000000 else 'blocked_inline_size',
        'created_at': now(), 'source_sha256': EXPECTED, 'derivative_path': derivative.name,
        'derivative_sha256': digest(derivative), 'derivative_bytes': derivative.stat().st_size,
        'compact_inline_json_bytes': request_size, 'inline_limit_bytes': 50000000,
        'selected_source_interval': [2580, 2760], 'relative_interval': [0, 180], 'decoded_frames': 181,
        'dimensions': [426, 240], 'native_to_analysis_coordinate_scale': [1, 1],
        'source_sample_aspect_ratio': '427:426, preserved; coordinates address426×240 native raster',
        'nominal_fps': 60, 'time_base': '1/1000', 'source_pts_offset': offset,
        'exact_relative_pts_match': True, 'first_source_pts': int(source_rows[0][2]),
        'last_source_pts': int(source_rows[-1][2]), 'last_relative_pts': int(derived_rows[-1][2]),
        'source_duration_ticks': dict(Counter(a[3] for a in source_rows)),
        'derivative_duration_ticks': dict(Counter(a[3] for a in derived_rows)),
        'timing_note': 'Only constant initialPTS offset removed; exact16/17ms sourcegaps preserved, not synthesizedconstantfps',
        'objects': ids, 'encoding': 'Explicitly lossy CRF18; no resize or crop; audio omitted',
        'pixel_metrics': metrics(histogram), 'no_hidden_suffix_frames': True, 'api_calls': 0,
        'prompt_direction': 'forward: all3 objects initializedatrelative40; preceding0..39 retained aspreinitialization, untracked',
        'provider_estimate_usd': .0362, 'conservative_three_object_envelope_usd': .1086,
        'proposed_shared_ledger_reservation_usd': .15, 'reservation_made': False}
    save('report.json', report)
    save('manifest.json', {'schema': 1, 'created_at': now(), 'files': {
         str(p.relative_to(OUT)): digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}})
    from weft.pipeline import verify_artifacts
    print(json.dumps({'report': report, 'verification': verify_artifacts(OUT)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--reuse-source', type=Path)
    args = parser.parse_args()
    OUT = args.out.resolve()
    main(args.reuse_source.resolve() if args.reuse_source else None)
