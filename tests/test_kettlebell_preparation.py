import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("prepare_kettlebell_tracking", Path(__file__).parents[1] / "scripts/studies/prepare_kettlebell_tracking.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decoded(times):
    return {"frames": [{"frame": index, "decoder_time_s": time} for index, time in enumerate(times)]}


@pytest.mark.parametrize("origin", [0, 3, 121])
def test_nonzero_native_origins_are_measured_not_removed(origin):
    native = [{"pts": origin + value} for value in (0, 33, 67)]
    result = MODULE.clock_relationship(native, decoded([0., .033, .067]))
    assert result["native_minus_decoder_pts"] == origin
    assert result["native_first_pts"] == origin
    assert result["decoder_pts"] == [0, 33, 67]
    assert native == [{"pts": origin + value} for value in (0, 33, 67)]


@pytest.mark.parametrize("times", [[0., .034, .067], [0., .0335, .067], [0., .033, .033],
                                   [0., float("nan"), .067], [0., .033], [.004, .037, .071]])
def test_clock_drift_repair_nonfinite_and_incomplete_sequences_are_rejected(times):
    with pytest.raises(ValueError):
        MODULE.clock_relationship([{"pts": value} for value in (3, 36, 70)], decoded(times))


def test_full_source_framehash_accepts_nonzero_native_start(tmp_path):
    path = tmp_path / "frames.sha256"
    rows = [f"0, {3 + round(i * 1000 / 30)}, {3 + round(i * 1000 / 30)}, 33, 460800, {'a' * 64}" for i in range(257)]
    path.write_text("#tb 0: 1/1000\n#dimensions 0: 640x480\n" + "\n".join(rows))
    result = MODULE.read_framehash(path)
    assert len(result) == 257 and result[0]["pts"] == 3 and result[-1]["pts"] == 8536


def test_framehash_input_size_is_bounded_before_parsing(tmp_path):
    path = tmp_path / "oversized.sha256"
    path.write_bytes(b" " * 128_001)
    with pytest.raises(ValueError, match="envelope"):
        MODULE.read_framehash(path)
