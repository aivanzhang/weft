"""Appearance selection cannot alter physics or insert actions into a recording."""

from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace
import time

import pytest

from test_avatar_assets import factory
from test_shared_exercise_lab import prepared
from weft.avatar_assets import import_avatar_package
from weft.avatar_lab import AvatarCatalog, DEFAULT_APPEARANCE, validate_appearance
from weft.avatar_recording import load_appearance, settings_by_event, _saved_catalog
from weft.io import digest
from weft.lab import Lab, LabHandler
from weft.recording import load_recording


def imported(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    source, _ = factory(tmp_path, glb=True)
    out = tmp_path / "import"
    import_avatar_package(source, out)
    return out


def look(key=None, **kwargs):
    return deepcopy(DEFAULT_APPEARANCE) | {"asset": key} | kwargs


def test_catalog_serves_only_verified_bytes_and_no_paths(tmp_path):
    source = imported(tmp_path)
    catalog = AvatarCatalog([source])
    key = next(iter(catalog.packages))
    data = catalog.summary(look(key))
    assert str(tmp_path) not in json.dumps(data)
    raw, mime, sha = catalog.resource(key, "render.json")
    assert mime == "application/json" and sha == catalog.packages[key].render_sha256
    assert raw == (source / "render.json").read_bytes()
    image, mime, _ = catalog.resource(key, "texture-0")
    assert mime == "image/png" and image.startswith(b"\x89PNG")
    for name in ("../render.json", "texture--1", "texture-8", "originals/model.glb"):
        with pytest.raises(ValueError):
            catalog.resource(key, name)
    with pytest.raises(ValueError, match="unique"):
        AvatarCatalog([source, source])
    with pytest.raises(ValueError, match="eight"):
        AvatarCatalog([source] * 9)
    # Eight distinct directories must pass the count gate (uniqueness fails later,
    # proving the ninth-item refusal above came from the count, not the message).
    with pytest.raises(ValueError, match="unique"):
        AvatarCatalog([source] * 8)


@pytest.mark.parametrize("change", [
    {"asset": "missing"}, {"overlay": 1}, {"show_muscles": None}, {"physics": {}},
    {"camera": {"yaw": float("nan"), "zoom": 1, "follow": False}},
    {"camera": {"yaw": 0, "zoom": 4, "follow": False}},
    {"camera": {"yaw": 0, "zoom": 1, "follow": 1}},
])
def test_invalid_visual_settings_are_rejected(change):
    with pytest.raises(ValueError):
        validate_appearance(look() | change, {})


def test_same_actions_states_rewards_and_journal_under_appearance_changes(tmp_path):
    model = imported(tmp_path / "avatar")
    (tmp_path / "physics").mkdir()
    target, _, _ = prepared(tmp_path / "physics")
    labs = [Lab(target), Lab(target, avatars=[model])]
    records = []
    before = {p.relative_to(target / "run").as_posix(): digest(p) for p in (target / "run").rglob("*") if p.is_file()}
    try:
        key = next(iter(labs[1].avatars.packages))
        for lab in labs:
            lab.control("/api/reset", {"seed": 37})
            records.append(lab.control("/api/record/start", {}))
        assert records[0]["snapshot"] == records[1]["snapshot"]
        for index in range(3):
            selection = look(key if index % 2 == 0 else None, overlay=index == 0,
                             camera={"yaw": index / 2, "zoom": 1 + index / 2, "follow": index == 2})
            state_before = deepcopy(labs[1].session.snapshot())
            labs[1].control("/api/appearance", selection)
            assert labs[1].session.snapshot() == state_before
            payload = {"action": [.25] + [0.] * 18}
            a, b = (lab.control("/api/step", payload) for lab in labs)
            assert a == b
        for lab in labs:
            lab.recorder.close()
        labs[1].recorder.asset_future.result(timeout=10)
        roots = [target / "sessions" / item["directory"] for item in records]
        assert (roots[0] / "events.jsonl").read_bytes() == (roots[1] / "events.jsonl").read_bytes()
        assert (roots[0] / "scene.json").read_bytes() == (roots[1] / "scene.json").read_bytes()
        recording = load_recording(roots[1])
        sidecar = load_appearance(roots[1], recording)
        assert [row["after_event_index"] for row in sidecar["changes"]] == [0, 0, 1, 2]
        sequence = list(settings_by_event(sidecar["changes"], len(recording["events"])))
        assert [row["asset"] for row in sequence] == [key, None, key, key]
        packages, viewer = _saved_catalog(roots[1], sidecar["assets"], time.monotonic() + 10)
        assert packages[key].render_sha256 == labs[1].avatars.packages[key].render_sha256
        assert len(viewer) == 3
        assert before == {p.relative_to(target / "run").as_posix(): digest(p) for p in (target / "run").rglob("*") if p.is_file()}
        change_path = roots[1] / "appearance-events.jsonl"
        changed = [json.loads(line) for line in change_path.read_text().splitlines()]
        changed[1]["after_event_sha256"] = "0" * 64
        change_path.write_text("\n".join(json.dumps(row) for row in changed) + "\n")
        meta_path = roots[1] / "appearance.json"
        meta = json.loads(meta_path.read_text()); meta["appearance_events_sha256"] = digest(change_path)
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="physical event"):
            load_appearance(roots[1], recording)
    finally:
        for lab in labs:
            lab.close()


def test_resource_endpoint_requires_existing_host_origin_and_token(tmp_path):
    catalog = AvatarCatalog([imported(tmp_path)])
    key = next(iter(catalog.packages))
    for authorized in (False, True):
        handler = object.__new__(LabHandler)
        handler.server = SimpleNamespace(server_port=12345, lab=SimpleNamespace(token="secret", avatars=catalog))
        handler.path = f"/api/avatar/{key}/render.json"
        handler.headers = {"Host": "127.0.0.1:12345"}
        if authorized:
            handler.headers["Authorization"] = "Bearer secret"
        replies = []
        handler.reply = lambda value, code=200: replies.append((code, value))
        handler.response_headers = lambda code, mime, size: replies.append((code, mime, size))
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        handler.wfile = BytesIO()
        handler.do_GET()
        assert replies[0][0] == (200 if authorized else 401)
        assert bool(handler.wfile.getvalue()) is authorized
