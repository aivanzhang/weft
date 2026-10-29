import json
from pathlib import Path

import pytest

from weft.exercise_source import SAMPLES, describe_source, fetch_exercise_sample


def test_new_exercise_sources_have_matching_pins_attribution_and_download_limits():
    for name in ('kettlebell-pushup', 'lunge-twist', 'jumping-jacks-burpees'):
        source = SAMPLES[name]
        assert Path('data/exercise', source['file']).is_file(), 'Test fixtures must exist; tests never download'
        result = fetch_exercise_sample(name=name)
        sidecar = json.loads(Path(result['provenance']).read_text())
        assert sidecar['sha256'] == source['sha256']
        assert sidecar['bytes'] == source['bytes']
        assert source['license'] == 'CC BY-SA 4.0'
        assert describe_source(source['sha256']) == source


def test_source_fetch_preserves_a_conflicting_existing_file(tmp_path):
    source = SAMPLES['lunge-twist']
    path = tmp_path / source['file']
    path.write_bytes(b'preserved conflicting input')
    with pytest.raises(ValueError, match='preserved without overwrite'):
        fetch_exercise_sample(out=tmp_path, name='lunge-twist')
    assert path.read_bytes() == b'preserved conflicting input'
