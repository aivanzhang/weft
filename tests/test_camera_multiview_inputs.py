"""Secondary inputs are exact local copies, not a new source acquisition path."""

from copy import deepcopy
from hashlib import sha256
import json
import socket
import time

import pytest

from weft import camera_multiview_inputs as inputs


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def pin(value):
    return sha256(value).hexdigest()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Secondary snapshot attempted network work")
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)


def source_config(tmp_path, *, alignment=True, observations=1):
    files = {inputs.CAMERA_MEMBER: raw({"source_sha256": "b" * 64, "kind": "camera"}),
             inputs.EVIDENCE_MEMBER: raw({"source_sha256": "b" * 64, "kind": "evidence"})}
    documents = [raw({"observation": index}) for index in range(observations)]
    for index, document in enumerate(documents):
        (tmp_path / f"observation-{index}.json").write_bytes(document)
    mapping = raw({"mapping": "authored fixture, not world validation"})
    if alignment:
        (tmp_path / "alignment.json").write_bytes(mapping)
    correspondence = {"primary_entity_ref": {"document_sha256": "c" * 64, "object_id": "person"},
        "secondary_entity_ref": {"document_sha256": pin(documents[0]), "object_id": "person"},
        "status": "assumed", "method": "Authored fixture identity", "uncertainty": "Not verified identity"}
    config = {"schema_version": 1, "profile": inputs.PROFILE, "primary_source_sha256": "a" * 64,
        "secondary_source_sha256": "b" * 64, "image_scale": .003, "weight": 2.5, "correspondence": correspondence,
        "observations": [{"path": f"observation-{index}.json", "sha256": pin(document)}
                         for index, document in enumerate(documents)],
        "world_alignment": {"path": "alignment.json", "sha256": pin(mapping)} if alignment else None}
    path = tmp_path / "config.json"
    path.write_bytes(raw(config))
    return path, files, config


def prepare(path, files):
    return inputs.prepare_camera_multiview_inputs(path, source_files=files, source_sha256="a" * 64,
                                                 deadline=time.monotonic() + 5)


@pytest.mark.parametrize("alignment, count", [(True, 1), (False, 8)])
def test_exact_snapshot_preserves_existing_camera_copies_and_optional_unknown_mapping(tmp_path, alignment, count):
    path, source, config = source_config(tmp_path, alignment=alignment, observations=count)
    before = deepcopy(source)
    files, descriptor = prepare(path, source)
    assert source == before
    assert set(files) == inputs.camera_multiview_member_names(descriptor)
    assert descriptor["config_sha256"] == pin(path.read_bytes())
    assert descriptor["observation_sha256s"] == [entry["sha256"] for entry in config["observations"]]
    assert files[inputs.CAMERA_MEMBER] is source[inputs.CAMERA_MEMBER]
    assert files[inputs.EVIDENCE_MEMBER] is source[inputs.EVIDENCE_MEMBER]
    assert inputs.validate_camera_multiview_snapshot(files, descriptor, source_sha256="a" * 64) == config
    files["inputs/human.json"] = raw({"source_sha256": "a" * 64})
    options = inputs.prepare_camera_multiview_options(files, descriptor)["camera_secondary_view"]
    assert options["camera_document"] == before[inputs.CAMERA_MEMBER]
    assert options["evidence_document"] == before[inputs.EVIDENCE_MEMBER]
    assert options["world_alignment_document"] == files.get(inputs.ALIGNMENT_MEMBER)
    assert options["image_scale"] == .003 and options["weight"] == 2.5
    assert options["correspondence"]["status"] == "assumed"


@pytest.mark.parametrize("change", [
    lambda config: config.update(schema_version=True),
    lambda config: config.update(primary_source_sha256="b" * 64),
    lambda config: config.update(image_scale=True),
    lambda config: config.update(image_scale=0),
    lambda config: config.update(weight=1001),
    lambda config: config.update(observations=[]),
    lambda config: config["observations"].append(config["observations"][0]),
    lambda config: config["observations"][0].update(path="../outside.json"),
    lambda config: config["observations"][0].update(path="/outside.json"),
    lambda config: config["observations"][0].update(sha256="A" * 64),
    lambda config: config["correspondence"].update(status="measured"),
    lambda config: config["correspondence"].update(uncertainty=""),
    lambda config: config.update(camera={"path": "unrelated.json", "sha256": "d" * 64}),
])
def test_ambiguous_or_unbounded_config_is_rejected_before_member_reads(tmp_path, monkeypatch, change):
    path, source, config = source_config(tmp_path)
    change(config)
    path.write_bytes(raw(config))
    from weft import shared_world_packaging as packaging
    original, reads = packaging.read_package_file, []
    def read(root, name, **kwargs):
        reads.append(name)
        return original(root, name, **kwargs)
    monkeypatch.setattr(packaging, "read_package_file", read)
    with pytest.raises(ValueError):
        prepare(path, source)
    assert reads == ["config.json"]


@pytest.mark.parametrize("document", [b'{"profile":1,"profile":2}', b'{"scale":NaN}', b'[]'])
def test_noncanonical_json_fails(document):
    with pytest.raises(ValueError):
        inputs.parse_camera_multiview_config(document)


@pytest.mark.parametrize("member", [inputs.CAMERA_MEMBER, inputs.EVIDENCE_MEMBER,
    "inputs/camera-secondary-observation-00.json", inputs.ALIGNMENT_MEMBER, inputs.CONFIG_MEMBER])
def test_frozen_member_mutation_cannot_reuse_descriptor(tmp_path, member):
    path, source, _ = source_config(tmp_path)
    files, descriptor = prepare(path, source)
    files[member] += b" "
    with pytest.raises(ValueError):
        inputs.validate_camera_multiview_snapshot(files, descriptor)


def test_source_join_and_missing_original_camera_fail(tmp_path):
    path, source, config = source_config(tmp_path)
    source[inputs.CAMERA_MEMBER] = raw({"source_sha256": "d" * 64})
    with pytest.raises(ValueError, match="secondary source differs"):
        prepare(path, source)
    del source[inputs.CAMERA_MEMBER]
    with pytest.raises(ValueError, match="original sealed body"):
        prepare(path, source)
    config["primary_source_sha256"] = "d" * 64
    path.write_bytes(raw(config))
    with pytest.raises(ValueError, match="primary source differs"):
        prepare(path, source)


def test_pinned_observation_and_symlink_are_rejected(tmp_path):
    path, source, _ = source_config(tmp_path)
    member = tmp_path / "observation-0.json"
    member.write_bytes(b'{"changed":true}')
    with pytest.raises(ValueError):
        prepare(path, source)
    member.rename(tmp_path / "original.json")
    member.symlink_to(tmp_path / "original.json")
    with pytest.raises((ValueError, OSError)):
        prepare(path, source)


def test_deadline_and_byte_limits_stop_before_expensive_preparation(tmp_path, monkeypatch):
    path, source, _ = source_config(tmp_path)
    with pytest.raises(TimeoutError):
        inputs.prepare_camera_multiview_inputs(path, source_files=source, source_sha256="a" * 64, deadline=0)
    files, descriptor = prepare(path, source)
    monkeypatch.setattr(inputs, "MAX_INPUT_BYTES", sum(map(len, files.values())) - 1)
    with pytest.raises(ValueError, match="byte bound"):
        inputs.validate_camera_multiview_snapshot(files, descriptor)
    with pytest.raises(ValueError):
        prepare(path, source)


def test_descriptor_inventory_matches_standalone_worker_preimport_contract(tmp_path):
    from weft.whole_body_reference_worker import _secondary_members
    path, source, _ = source_config(tmp_path, observations=8)
    files, descriptor = prepare(path, source)
    assert _secondary_members(descriptor) == {name: pin(value) for name, value in files.items()}
    for field in ("observation_sha256s", "world_alignment_sha256", "image_scale", "camera_sha256"):
        changed = descriptor | {field: True}
        with pytest.raises(ValueError):
            inputs.camera_multiview_member_names(changed)
        with pytest.raises(ValueError):
            _secondary_members(changed)


def test_unselected_alignment_in_broader_source_package_is_never_consumed(tmp_path):
    path, source, _ = source_config(tmp_path, alignment=False)
    files, descriptor = prepare(path, source)
    files.update({"inputs/human.json": raw({"source_sha256": "a" * 64}),
                  inputs.ALIGNMENT_MEMBER: b'{"unselected":"must not be read"}'})
    options = inputs.prepare_camera_multiview_options(files, descriptor)["camera_secondary_view"]
    assert options["world_alignment_document"] is None
