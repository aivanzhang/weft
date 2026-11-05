"""Curated comparisons remain decodable, aligned and traceable to saved evidence."""
import hashlib
import json
from pathlib import Path

import cv2

from weft.portal_demo import GALLERY_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def test_ten_distinct_comparisons_have_matching_playback_windows():
    assert len(GALLERY_SEEDS) == 10
    assert GALLERY_SEEDS[0]['id'] == 'bodyweight-squats'
    assert GALLERY_SEEDS[-1]['id'] == 'rolling-motion'
    assert len({row['id'] for row in GALLERY_SEEDS}) == 10
    assert len({row['source_video'] for row in GALLERY_SEEDS}) == 10
    assert 'panda' not in json.dumps(GALLERY_SEEDS).lower()
    for row in GALLERY_SEEDS:
        durations = []
        for key in ('source_video', 'video'):
            capture = cv2.VideoCapture(str(ROOT / row[key]))
            try:
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                durations.append(count / capture.get(cv2.CAP_PROP_FPS))
                for position in (0, count // 2, count - 1):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ok, frame = capture.read()
                    assert ok and frame.size, (row['id'], key, position)
                if key == 'video' and row['id'] != 'rolling-motion':
                    assert frame.shape[:2] == (640, 640)
            finally:
                capture.release()
        assert abs(durations[0] - durations[1]) < .08, row['id']
        assert row['source_credit'] and row['scope'] and row['result_note']
        assert row['source_page'].startswith('https://')


def test_motion_receipts_preserve_hashes_and_disclose_reference_replay():
    sources = set()
    for row in (seed for seed in GALLERY_SEEDS if seed['id'] != 'rolling-motion'):
        directory = (ROOT / row['video']).parent
        receipt = json.loads((directory / 'receipt.json').read_text())
        assert receipt['input'] == row['id']
        assert 'no new simulation or learned motion' in receipt['presentation']
        assert len(receipt['reference_sha256']) == 64
        assert receipt['source_start_s'] >= 0
        sources.add(receipt['source_sha256'])
        for name, expected in receipt['files'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    assert len(sources) == 9
    # This excerpt follows an earlier admitted window in the submitted footage.
    jumping = json.loads((ROOT / 'assets/demo/motion-showcase/jumping-jacks/receipt.json').read_text())
    assert jumping['source_start_s'] == 21.44


def test_rolling_preview_retains_its_presentation_manifest():
    directory = ROOT / 'assets/demo/previews'
    manifest = json.loads((directory / 'manifest.json').read_text())
    for row in manifest['files']:
        if not row['output'].startswith('rolling-'):
            continue
        path = directory / row['output']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert row.get('source_sha256') or row.get('source_manifest_sha256')
        assert row.get('filters') or row.get('playback_speed')
