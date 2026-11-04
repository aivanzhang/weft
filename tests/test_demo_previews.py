"""Curated previews remain decodable, paired and traceable to saved evidence."""
import hashlib
import json
from pathlib import Path

import cv2

from weft.portal_demo import GALLERY_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def test_curated_previews_have_distinct_purposes_and_matching_playback_windows():
    assert [row['category'] for row in GALLERY_SEEDS] == ['Object motion', 'Body control', 'Character transfer']
    for row in GALLERY_SEEDS:
        durations = []
        for key in ('source_video', 'video'):
            capture = cv2.VideoCapture(str(ROOT / row[key]))
            try:
                ok, frame = capture.read()
                assert ok and frame.size
                durations.append(capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS))
                if key == 'video' and row['category'] != 'Object motion':
                    assert (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (360, 420)
            finally:
                capture.release()
        assert abs(durations[0] - durations[1]) < .05
        assert row['source_credit'] and row['scope'] and row['result_note']


def test_preview_manifest_preserves_hashes_and_declares_presentation_changes():
    directory = ROOT / 'assets/demo/previews'
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert len(manifest['files']) == 6
    for row in manifest['files']:
        path = directory / row['output']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        assert path.stat().st_size == row['bytes']
        assert row.get('source_sha256') or row.get('source_manifest_sha256')
        assert row.get('filters') or row.get('playback_speed')
