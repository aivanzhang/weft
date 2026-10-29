"""Freeze source-only camera landmarks; never read registration or object tracks."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2


VIDEO = Path('artifacts/meta-three-ball-input-01/attempt-05/three-ball-crf18.mp4')
VIDEO_SHA = 'd963bb8e3c4941921fb2dba03be34c5969e073ce6fd4c6dedfd8d719e1e3ddcb'
FRAMES = [40, 61, 90, 124, 150, 180]
LANDMARKS = [
    ('sheet_upper_left', 'Upper-left outer corner of white printed table sheet', [59, 20]),
    ('sheet_upper_right', 'Upper-right outer corner of white printed table sheet', [378, 18]),
    ('yellow_label_upper_left', 'Upper-left rounded corner of lower-left yellow printed label', [76, 213]),
    ('yellow_label_upper_right', 'Upper-right rounded corner of lower-left yellow printed label', [150, 212]),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seal-existing', action='store_true')
    args = parser.parse_args()
    out = args.out
    assert sha(VIDEO) == VIDEO_SHA
    if not args.seal_existing:
        out.mkdir(parents=True, exist_ok=False)
        (out/'frames').mkdir()
        cap = cv2.VideoCapture(str(VIDEO))
        rows, count = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert count < 181 and frame.shape == (240, 426, 3)
            if count in FRAMES:
                path = out/f'frames/frame-{count:03d}.png'
                assert cv2.imwrite(str(path), frame)
                rows.append({'relative_frame': count, 'absolute_source_frame': 2580+count,
                             'timestamp_s': cap.get(cv2.CAP_PROP_POS_MSEC)/1000,
                             'dimensions': [426, 240],
                             'decoded_bgr_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
                             'path': str(path.relative_to(out)), 'png_sha256': sha(path)})
            count += 1
        cap.release()
        assert count == 181 and len(rows) == 6
        write(out/'extraction.json', {'video': str(VIDEO), 'video_sha256': VIDEO_SHA,
              'decoded_frames': count, 'selected_relative_frames': FRAMES, 'frames': rows})
    extraction = json.loads((out/'extraction.json').read_text())
    assert extraction['video_sha256'] == VIDEO_SHA
    assert extraction['selected_relative_frames'] == FRAMES
    assert [row['relative_frame'] for row in extraction['frames']] == FRAMES
    for row in extraction['frames']:
        path = out/row['path']
        assert sha(path) == row['png_sha256']
        image = cv2.imread(str(path))
        assert image.shape == (240, 426, 3)
        assert hashlib.sha256(image.tobytes()).hexdigest() == row['decoded_bgr_sha256']
    write(out/'annotations.json', {
        'schema_version': 1, 'status': 'frozen_source_only',
        'frozen_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
        'video_sha256': VIDEO_SHA, 'anchor_relative_frame': 40,
        'coordinate_system': 'Native decoded 426x240 raster; x right, y down; integer pixel centers',
        'annotation_method': 'Assistant visual estimates from original decoded frames and nearest-neighbor inspection patches; not expert truth',
        'uncertainty_definition': 'Conservative +/- pixel extent independently in x and y; not a statistical confidence interval',
        'selection': 'All six requested frames retained. Landmarks selected from reference frame pixels, before registration outputs.',
        'landmarks': [{'id': identity, 'description': description, 'role': 'apparently fixed printed table background'} for identity, description, _ in LANDMARKS],
        'frames': [{'relative_frame': frame, 'absolute_source_frame': frame+2580,
                    'landmarks': [{'id': identity, 'xy': xy, 'uncertainty_xy_px': [2, 2],
                                   'visibility': 'visible', 'confidence': 'assistant_approximate',
                                   'note': 'Low-resolution soft or rounded edge; no subpixel precision claimed'}
                                  for identity, _, xy in LANDMARKS]} for frame in FRAMES],
        'limitations': ['All four are printed sheet features, not surveyed physical control points.',
                       'Sheet appears stationary relative to wood; its physical attachment is not verified.',
                       'Equal rounded pixel labels do not establish zero camera motion; small drift remains unresolved within uncertainty.',
                       'Two bottom landmarks are close together; homography conditioning is weaker than four widely separated corners.',
                       'Shadowed lower-right yellow-label candidate was rejected during source-only selection; its inspection patch remains retained.',
                       'No object tracks, camera transforms, candidate scores or model outputs were read.'],
    })
    write(out/'provenance.json', {
        'submitted_video': str(VIDEO), 'submitted_video_sha256': VIDEO_SHA,
        'submitted_video_bytes': VIDEO.stat().st_size,
        'original_video_sha256': 'f2d9f8125a197ec135dcc4d95e5b1b615c875f097c12ee272d83f8b97baa0887',
        'original_interval': [2580, 2760], 'relative_interval': [0, 180],
        'preparation_report': 'artifacts/meta-three-ball-input-01/report.json',
        'encoding': 'Exact submitted CRF18 derivative, not the original WebM; no additional video encoding',
        'decoder': {'name': 'OpenCV', 'version': cv2.__version__},
        'construction_access_boundary': 'Keep annotations sealed from registration generation until candidate freeze.',
        'missing_slots': 0, 'retained_slots': 24,
    })
    script = Path(__file__)
    (out/'freeze_camera_source_labels.py').write_bytes(script.read_bytes())
    files = [{'path': str(p.relative_to(out)), 'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'manifest.json', {'schema_version': 1, 'scope': 'Source-only camera labels', 'files': files})
    print(json.dumps({'path': str(out), 'manifest_sha256': sha(out/'manifest.json'), 'files': len(files)}))


if __name__ == '__main__':
    main()
