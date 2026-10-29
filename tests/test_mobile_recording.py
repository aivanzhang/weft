"""Mobile recordings preserve simulated root forces separately from muscle illustration."""

from copy import deepcopy
import json

import numpy as np
import pytest

from weft.recording import Recorder, export_recording, load_recording, render_snapshot


def mobile_fixture():
    scene = {"schema_version": 1, "profile": "exercise-v3", "width": 1, "height": 1,
             "dt": 1/30, "objects": [{"id": f"joint-{i}"} for i in range(19)]}
    state = {"steps": 0, "time_s": 0.,
             "objects": [{"id": row["id"], "position": [0., 0.], "velocity": [0., 0.]} for row in scene["objects"]],
             "human": {"height": 1.8, "origin": [0., 0., 0.],
                       "segments": [{"id": "torso", "a": [0., 0., .8], "b": [0., 0., 1.4], "radius": .1}],
                       "muscles": [{"name": f"band-{i}", "a": [-.05, 0., .9], "b": [.05, 0., 1.3],
                                    "activation": .25, "status": "illustrative"} for i in range(38)],
                       "angles": {f"joint-{i}": 0. for i in range(19)},
                       "mobility": {"support_enabled": True, "translation": [.1, 0., -.02],
                                    "linear_velocity": [.2, 0., -.1], "support_force": [10., 0., 150.],
                                    "floor_height": 0., "contacts": [{"geom": "left_foot", "position": [.2, .1, 0.],
                                        "distance": -.001, "normal_force": 120.}],
                                    "contact_count": 1, "total_normal_force": 120., "max_penetration": .001,
                                    "scope": "simulated_floor_contact_and_virtual_support_not_measured_forces"}}}
    return scene, state


def event(state, action):
    return {"type": "step", "action": action, "reward": 0., "terminated": False, "truncated": False,
            "snapshot": deepcopy(state)}


def test_mobile_commands_roundtrip_and_export_preserve_full_state(tmp_path):
    scene, state = mobile_fixture()
    root = tmp_path / "mobile"
    recorder = Recorder(root, scene)
    recorder.start(state)
    states = [deepcopy(state)]
    for action in range(6, 10):
        state["steps"] += 1
        state["time_s"] += scene["dt"]
        state["human"]["mobility"]["support_enabled"] = action != 7
        state["human"]["mobility"]["support_force"] = [float(action), 0., 100.]
        recorder.append(event(state, action))
        states.append(deepcopy(state))
    recorder.close()
    loaded = load_recording(root)
    assert [row["action"] for row in loaded["events"][1:]] == [6, 7, 8, 9]
    assert [row["snapshot"] for row in loaded["events"]] == states
    assert all(len(row["snapshot"]["human"]["angles"]) == 19 for row in loaded["events"])
    assert all(len(row["snapshot"]["human"]["muscles"]) == 38 for row in loaded["events"])
    result = export_recording(root)
    assert result["video_frames"] == 5 and result["steps"] == 4
    html = (root / "replay.html").read_text()
    assert "SUPPORT ON" in html and "SUPPORT OFF" in html and "ANCHORED" in html
    assert "h.mobility?.floor_height" in html and "contact.position" in html
    assert "not muscle activation" in html
    assert all(muscle["activation"] == .25 for row in loaded["events"] for muscle in row["snapshot"]["human"]["muscles"])


@pytest.mark.parametrize("action", [-1, 10, True, 9.0])
def test_mobile_bad_actions_do_not_write(tmp_path, action):
    scene, state = mobile_fixture()
    recorder = Recorder(tmp_path / "bad-action", scene)
    recorder.start(state)
    before = (recorder.root / "events.jsonl").read_bytes()
    with pytest.raises(ValueError, match="0 to 9"):
        recorder.append(event(state, action))
    assert (recorder.root / "events.jsonl").read_bytes() == before
    recorder.close()


@pytest.mark.parametrize("profile", ["exercise-v1", "exercise-v2", "planar-rigid-v1"])
def test_old_profiles_still_reject_mobile_commands(tmp_path, profile):
    scene, state = mobile_fixture()
    scene["profile"] = profile
    scene["objects"] = scene["objects"][:1]
    state["objects"] = state["objects"][:1]
    state["human"].pop("mobility")
    recorder = Recorder(tmp_path / "legacy", scene)
    recorder.start(state)
    with pytest.raises(ValueError, match="0 to 5"):
        recorder.append(event(state, 6))
    recorder.append(event(state, 5))
    recorder.close()
    assert load_recording(recorder.root)["events"][1]["action"] == 5


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("translation"),
    lambda m: m.update(extra="unknown"),
    lambda m: m.update(support_enabled=1),
    lambda m: m.update(translation=[float("nan"), 0., 0.]),
    lambda m: m.update(translation=[101., 0., 0.]),
    lambda m: m.update(linear_velocity=[101., 0., 0.]),
    lambda m: m.update(support_force=[2001., 0., 0.]),
    lambda m: m.update(floor_height=.1),
    lambda m: m.update(contacts=m["contacts"]*33, contact_count=33),
    lambda m: m.update(contact_count=0),
    lambda m: m.update(contact_count=129),
    lambda m: m.update(contact_count=True),
    lambda m: m.update(total_normal_force=-1.),
    lambda m: m.update(total_normal_force=float("inf")),
    lambda m: m.update(max_penetration=-.01),
    lambda m: m.update(scope=""),
    lambda m: m["contacts"][0].update(position=[0., 0.]),
    lambda m: m["contacts"][0].update(distance=float("nan")),
    lambda m: m["contacts"][0].update(geom="x"*101),
    lambda m: m["contacts"][0].update(normal_force=-1.),
    lambda m: m["contacts"][0].update(normal_force=1e10),
])
def test_invalid_mobile_snapshot_rejected_before_journal_creation(tmp_path, mutate):
    scene, state = mobile_fixture()
    mutate(state["human"]["mobility"])
    recorder = Recorder(tmp_path / "invalid", scene)
    with pytest.raises(ValueError):
        recorder.start(state)
    assert not recorder.root.exists()


@pytest.mark.parametrize("missing", ["human", "mobility"])
def test_v3_requires_complete_human_and_mobility(tmp_path, missing):
    scene, state = mobile_fixture()
    if missing == "human":
        state.pop("human")
    else:
        state["human"].pop("mobility")
    with pytest.raises(ValueError):
        Recorder(tmp_path / "missing", scene).start(state)
    with pytest.raises(ValueError):
        render_snapshot(state, scene)


def test_contact_truncation_keeps_total_count_and_force(tmp_path):
    scene, state = mobile_fixture()
    state["human"]["mobility"].update(contact_count=40, contacts=state["human"]["mobility"]["contacts"]*32,
                                        total_normal_force=5000.)
    recorder = Recorder(tmp_path / "truncated-contacts", scene)
    recorder.start(state)
    recorder.close()
    mobility = load_recording(recorder.root)["events"][0]["snapshot"]["human"]["mobility"]
    assert len(mobility["contacts"]) == 32 and mobility["contact_count"] == 40
    assert mobility["total_normal_force"] == 5000.


def test_load_revalidates_mobile_commands_and_required_state(tmp_path):
    scene, state = mobile_fixture()
    recorder = Recorder(tmp_path / "load-validation", scene)
    recorder.start(state)
    recorder.append(event(state, 9))
    recorder.close()
    journal = recorder.root / "events.jsonl"
    original = journal.read_text()
    rows = [json.loads(line) for line in original.splitlines()]
    rows[1]["action"] = 10
    journal.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="0 to 9"):
        load_recording(recorder.root)
    rows[1]["action"] = 9
    rows[1]["snapshot"]["human"].pop("mobility")
    journal.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="mobility"):
        load_recording(recorder.root)


def test_contact_render_changes_geometry_not_muscle_activation():
    scene, state = mobile_fixture()
    before = render_snapshot(state, scene)
    altered = deepcopy(state)
    altered["human"]["mobility"].update(contacts=[], contact_count=0, total_normal_force=0., max_penetration=0.)
    after = render_snapshot(altered, scene)
    assert np.any(before[:576] != after[:576])
    assert state["human"]["muscles"] == altered["human"]["muscles"]
    assert state["human"]["angles"] == altered["human"]["angles"]


def test_mobile_view_bounds_allow_full_root_span_without_relaxing_legacy_or_coordinates(tmp_path):
    scene, state = mobile_fixture()
    state["human"].update(view_height=30., view_origin=[0., 0., 1.])
    recorder = Recorder(tmp_path / "wide-mobile-view", scene)
    recorder.start(state)
    recorder.close()
    assert load_recording(recorder.root)["events"][0]["snapshot"]["human"]["view_height"] == 30.
    invalid = deepcopy(state)
    invalid["human"]["view_height"] = 30.01
    with pytest.raises(ValueError, match="view height"):
        Recorder(tmp_path / "oversize-mobile-view", scene).start(invalid)
    invalid = deepcopy(state)
    invalid["human"]["view_origin"] = [101., 0., 0.]
    with pytest.raises(ValueError, match="view origin"):
        Recorder(tmp_path / "invalid-mobile-origin", scene).start(invalid)
    legacy = deepcopy(scene)
    legacy["profile"] = "exercise-v2"
    state["human"].pop("mobility")
    state["human"]["view_height"] = 10.01
    with pytest.raises(ValueError, match="view height"):
        Recorder(tmp_path / "oversize-legacy-view", legacy).start(state)


def test_inclined_floor_recording_preserves_plane_and_renders_without_changing_body(tmp_path):
    scene, state = mobile_fixture()
    baseline = render_snapshot(state, scene)
    before = deepcopy(state["human"])
    state["human"]["mobility"].update(floor_normal=[0., .6, .8], floor_scope="Inferred orientation; authored gravity alignment")
    tilted = render_snapshot(state, scene)
    assert np.any(baseline[:576] != tilted[:576])
    assert state["human"]["segments"] == before["segments"]
    recorder = Recorder(tmp_path / "inclined", scene)
    recorder.start(state)
    recorder.append(event(state, 0))
    recorder.close()
    loaded = load_recording(recorder.root)
    assert loaded["events"][0]["snapshot"] == state
    result = export_recording(recorder.root)
    assert result["video_frames"] == 2
    assert "h.mobility?.floor_normal" in (recorder.root / "replay.html").read_text()


@pytest.mark.parametrize("fields", [
    {"floor_normal": [0., .6, .8]},
    {"floor_scope": "Missing normal"},
    {"floor_normal": [0., .6, .8], "floor_scope": ""},
    {"floor_normal": [0., .6, .7], "floor_scope": "Non-unit"},
    {"floor_normal": [0., .6, -.8], "floor_scope": "Down"},
    {"floor_normal": [0., 1., 0.], "floor_scope": "Vertical"},
    {"floor_normal": [0., float('nan'), 1.], "floor_scope": "NaN"},
])
def test_inclined_floor_snapshot_rejects_incomplete_or_invalid_plane_before_write(tmp_path, fields):
    scene, state = mobile_fixture()
    state["human"]["mobility"].update(fields)
    recorder = Recorder(tmp_path / "invalid-plane", scene)
    with pytest.raises(ValueError):
        recorder.start(state)
    assert not recorder.root.exists()
