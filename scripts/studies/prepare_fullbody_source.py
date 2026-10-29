"""Freeze original-source samples without running pose or other model inference."""

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/exercise/bodyweight-squats-criles23.webm'
OUT = ROOT / 'artifacts/exercise-fullbody-source-01'


def write(name, data):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def main():
    source_hash = sha256(SOURCE.read_bytes()).hexdigest()
    capture = cv2.VideoCapture(str(SOURCE))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not 1 <= count <= 1800 or not 0 < fps <= 60 or count / fps > 60:
        raise ValueError('Candidate exceeds full-clip bounds; preserve without silent interval selection')
    fractions = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    indices = [int(q * (count - 1)) for q in fractions]
    annotation_indices = [int(q * (count-1)) for q in [.1, .3, .5, .7, .9]]
    write('pre-view-selection.json', {'created_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
          'source_sha256': source_hash, 'metadata_frames': count, 'fps': fps, 'dimensions': [width, height],
          'selected_interval_inclusive': [0, count-1], 'sample_fractions': fractions,
          'sample_frames': indices, 'annotation_frames': annotation_indices,
          'method': 'floor(fraction*(metadata_frame_count-1)), frozen before viewing pixels',
          'no_model_inference': True})
    frames_dir = OUT / 'frames'
    frames_dir.mkdir()
    decoded = 0
    saved, tiles = [], []
    while decoded <= 1800:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded in indices:
            path = frames_dir / f'{decoded:06d}.png'
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.save(path)
            saved.append({'frame': decoded, 'source_time_ms': capture.get(cv2.CAP_PROP_POS_MSEC),
                          'path': str(path.relative_to(OUT)), 'png_sha256': sha256(path.read_bytes()).hexdigest(),
                          'rgb_pixel_sha256': sha256(image.tobytes()).hexdigest()})
            thumb = image.copy()
            thumb.thumbnail((480, 270))
            tile = Image.new('RGB', (480, 296), 'white')
            tile.paste(thumb, (0, 26))
            ImageDraw.Draw(tile).text((5, 5), f'Original frame {decoded} | {decoded/fps:.2f}s', fill='black')
            tiles.append(tile)
        decoded += 1
    capture.release()
    sheet = Image.new('RGB', (480*3, 296*4), 'white')
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (480*(i%3), 296*(i//3)))
    sheet.save(OUT / 'contact-sheet.png')
    write('decode.json', {'decoded_frames': decoded, 'metadata_frames': count,
                         'exact_count_match': decoded == count, 'full_source_decoded': decoded <= 1800,
                         'samples': saved, 'source_sha256': source_hash})
    print(json.dumps({'source_sha256': source_hash, 'decoded_frames': decoded, 'metadata_frames': count,
                      'fps': fps, 'annotation_frames': annotation_indices, 'bytes': SOURCE.stat().st_size}))


if __name__ == '__main__':
    main()
