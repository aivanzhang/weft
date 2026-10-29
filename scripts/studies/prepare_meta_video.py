"""Create an explicitly selected format derivative; never submit media to an API."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import time
import argparse
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/meta-toy-van-input-02'
SOURCE = ROOT / 'data/general/corgi-hand-crank-van.webm'
FFMPEG = ROOT / '.venv-lab/lib/python3.14/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, data):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def run(crf):
    if digest(SOURCE) != 'caad84aeda62860cbd682caac2a8a1e21e6f12f715615c9c56193dc5926aebc8':
        raise ValueError('Original source changed')
    derivative = OUT / ('toy-van-lossless.mp4' if crf == 0 else 'toy-van-crf18.mp4')
    command = [str(FFMPEG), '-hide_banner', '-nostdin', '-n', '-threads', '2', '-copyts', '-i', str(SOURCE),
               '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-threads:v', '2', '-bf', '0', '-fps_mode', 'passthrough',
               '-enc_time_base:v', 'demux', '-video_track_timescale', '1000', '-movflags', '+faststart',
               str(derivative)]
    save('execution.json', {'created_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
                           'command': command, 'timeout_seconds': 120, 'threads': 2,
                           'source_sha256': digest(SOURCE), 'ffmpeg_sha256': digest(FFMPEG),
                           'script_sha256': digest(Path(__file__)),
                           'encoding': 'lossless' if crf == 0 else 'explicitly authorized lossy CRF18',
                           'audio': 'Not included: segmentation derivative contains original video stream only',
                           'source_hash_command': [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-threads', '2',
                              '-copyts', '-i', str(SOURCE), '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'rawvideo',
                              '-pix_fmt', 'yuv420p', '-threads:v', '2', '-fps_mode', 'passthrough', '-enc_time_base:v',
                              'demux', '-f', 'framehash', '-hash', 'sha256', str(OUT / 'source-frames.sha256')]})
    started = time.monotonic()
    with (OUT / 'encode.log').open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    save('encode-result.json', {'returncode': result.returncode, 'seconds': time.monotonic()-started})
    if result.returncode:
        raise RuntimeError('Encoding failed; preserve evidence')
    hash_command = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-threads', '2', '-copyts', '-i', str(derivative),
                    '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'rawvideo', '-pix_fmt', 'yuv420p', '-threads:v', '2',
                    '-fps_mode', 'passthrough', '-enc_time_base:v', 'demux', '-f', 'framehash', '-hash', 'sha256',
                    str(OUT / 'derivative-frames.sha256')]
    save('verification-command.json', hash_command)
    with (OUT / 'decode.log').open('x') as log:
        subprocess.run(hash_command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    print(json.dumps({'derivative': str(derivative), 'sha256': digest(derivative), 'bytes': derivative.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crf', type=int, choices=[0, 18], required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    run(args.crf)
