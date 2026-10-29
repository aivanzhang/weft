"""Preserve one complete GIF cycle and verify lossless RGB video conversion."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/exercise-fullbody-source-01/candidate-02'
SOURCE = ROOT / 'data/exercise/bodyweight-squats-danielflefil-500.gif'
DESTINATION = SOURCE.with_suffix('.mp4')
FF = ROOT / '.venv-lab/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name, data):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def timeout(*_):
    raise TimeoutError('Conversion exceeded120 seconds')


def main():
    source = Image.open(SOURCE)
    if digest(SOURCE) != '14f1febdd9827951c0099ae2df6288fa5afb7b83c1fb99882f7505486982af85':
        raise ValueError('Source GIF changed')
    if source.n_frames != 68 or source.size != (500, 500):
        raise ValueError('Source shape/count changed')
    for index in range(68):
        source.seek(index)
        if source.info['duration'] != 40:
            raise ValueError('Nonuniform GIF timing must not be resampled')
    command = [str(FF), '-hide_banner', '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', '500x500', '-framerate', '25', '-i', 'pipe:0', '-an',
               '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'medium', '-pix_fmt', 'rgb24',
               '-threads:v', '2', '-bf', '0', '-movflags', '+faststart', str(DESTINATION)]
    save('conversion-plan.json', {'created_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
         'command': command, 'source_sha256': digest(SOURCE), 'frozen_manual_sha256': digest(OUT / 'manual.json'),
         'duration_ms_by_frame': [40]*68, 'full_one_cycle': True, 'frame_count': 68,
         'uniform_duration_justifies25fps': True, 'script_sha256': digest(Path(__file__)),
         'timeout_seconds': 120, 'source_frame_duplication': False})
    expected = []
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    started = time.monotonic()
    process = None
    try:
        with (OUT / 'conversion.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            for index in range(68):
                source.seek(index)
                pixels = source.convert('RGB').tobytes()
                expected.append(sha256(pixels).hexdigest())
                process.stdin.write(pixels)
            process.stdin.close()
            if process.wait(timeout=120):
                raise ValueError('Video conversion failed')
    finally:
        signal.alarm(0)
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
    capture = cv2.VideoCapture(str(DESTINATION))
    decoded = []
    while len(decoded) <= 68:
        ok, frame = capture.read()
        if not ok:
            break
        index = len(decoded)
        rgb_hash = sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
        decoded.append({'frame': index, 'original_gif_time_ms': index*40, 'decoded_video_time_ms': timestamp,
                        'source_rgb_sha256': expected[index] if index < 68 else None,
                        'decoded_rgb_sha256': rgb_hash})
    capture.release()
    equal = len(decoded) == 68 and all(r['source_rgb_sha256'] == r['decoded_rgb_sha256'] and
                                     abs(r['original_gif_time_ms']-r['decoded_video_time_ms']) < 1e-6 for r in decoded)
    save('conversion-verification.json', {'status': 'verified' if equal else 'failed',
         'source_sha256': digest(SOURCE), 'derivative_sha256': digest(DESTINATION),
         'derivative_path': str(DESTINATION.relative_to(ROOT)), 'derivative_bytes': DESTINATION.stat().st_size,
         'decoded_frames': len(decoded), 'all_rgb_pixels_and_timestamps_equal': equal,
         'duration_ms': 2720, 'fps': 25, 'dimensions': [500, 500], 'frames': decoded,
         'seconds': time.monotonic()-started, 'pose_or_model_calls': 0})
    if not equal:
        raise ValueError('Lossless pixels/count/timestamp verification failed; preserve output')
    print(json.dumps({'video': str(DESTINATION), 'sha256': digest(DESTINATION),
                      'bytes': DESTINATION.stat().st_size, 'frames': len(decoded), 'verified': equal}))


if __name__ == '__main__':
    main()
