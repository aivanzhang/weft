"""Export labels distinguish absent equipment without changing legacy captions or states."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_recording as recording
from weft.shared_exercise_task import HUMAN_ONLY_TASK_PROFILE, PROFILE


@pytest.fixture
def captions(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_a, **_k: pytest.fail("Caption tests cannot advance physics"))
    original, text = recording.ImageDraw.Draw, []
    def draw(image):
        result = original(image)
        native = result.text
        def capture(xy, value, **options):
            text.append(value)
            return native(xy, value, **options)
        result.text = capture
        return result
    monkeypatch.setattr(recording.ImageDraw, "Draw", draw)
    return text


def snapshot(profile, state):
    return {"engine_state_status": "last_valid" if state == "engine_failure" else "current",
            "task_metrics": {"profile": profile, "success": state == "passed",
                "completed_interval": state in ("passed", "task_failed"),
                "failure": "Recorded failure" if state == "engine_failure" else None,
                "simulated_duration_s": .123456,
                "equipment": [] if profile == HUMAN_ONLY_TASK_PROFILE else [
                    {"displacement_model_units": .012345}, {"displacement_model_units": .001}]}}


@pytest.mark.parametrize("state,label", [
    ("running", "IN PROGRESS"), ("passed", "PASSED"),
    ("task_failed", "NOT PASSED"), ("engine_failure", "NOT PASSED"),
])
def test_person_caption_marks_equipment_and_contact_inapplicable(captions, state, label):
    image = np.full((120, 640, 3), (13, 47, 191), dtype=np.uint8)
    saved = snapshot(HUMAN_ONLY_TASK_PROFILE, state)
    before = deepcopy(saved)
    rendered = recording._annotate_frame(image, saved)
    assert captions[0] == f"PERSON-ONLY TASK | {label} | equipment/contact: N/A"
    expected = ("Last valid geometry; simulation failed at 0.123s." if state == "engine_failure" else
                "Person and ground only. Simulated task success is not verified biomechanics.")
    assert captions[1] == expected
    assert "drift" not in " ".join(captions) and "Passive equipment" not in " ".join(captions)
    assert rendered.shape == (184, 640, 3)
    np.testing.assert_array_equal(rendered[:120], image)
    assert saved == before


@pytest.mark.parametrize("state,label", [
    ("running", "IN PROGRESS"), ("passed", "PASSED"), ("engine_failure", "NOT PASSED"),
])
def test_legacy_caption_strings_and_rounded_drift_remain_exact(captions, state, label):
    saved = snapshot(PROFILE, state)
    before = deepcopy(saved)
    recording._annotate_frame(np.zeros((120, 640, 3), dtype=np.uint8), saved)
    assert captions[0] == f"HUMAN-ONLY TASK | {label} | maximum equipment drift 0.012 model units"
    assert captions[1] == ("Last valid geometry; simulation failed at 0.123s." if state == "engine_failure" else
        "Passive equipment; no object forces. Simulated task success is not verified grasp or biomechanics.")
    assert saved == before
