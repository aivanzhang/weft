"""Pure source joins and objective/kinematic boundary; no native engine or source study."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import math
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module("weft.source_surface_interval_construction")


def raw(value):
    return json.dumps(value, allow_nan=False, sort_keys=True).encode()


def fixture(api, count=4, channels=1, source_changes=None, body_indices=None):
    fixtures = importlib.import_module("test_source_surface_envelopes")
    data = fixtures.fixture(count, channels)
    source = data[0]["source"]["sha256"]
    for index, (record, observation) in enumerate(zip(data[0]["evidence"], data[0]["observations"], strict=True)):
        record["id"] = f"{source}:frame:{index}"
        observation["evidence_ids"] = [record["id"]]
    if source_changes is not None:
        source_changes(data)
    inputs = fixtures.encoded(data)
    envelopes = importlib.import_module("weft.source_surface_envelopes").prepare_source_surface_envelopes(**inputs)
    indices = tuple(range(count)) if body_indices is None else body_indices
    body = {"schema_version": 3, "profile": "exercise-v3", "source_sha256": source, "dt": 1/30,
            "reference": [{"time_s": i/30, "joints": {name: 0. for name in api.NAMES}} for i in indices],
            "reference_evidence": [{"evidence_id": f"{source}:frame:{i}", "time_s": i/30} for i in indices],
            "root_reference": [{"time_s": i/30, "translation": [0., 0., 0.]} for i in indices]}
    body_bytes = raw(body)
    claim = {"status": "assumed", "method": "Explicit analytical model-to-world fixture",
             "uncertainty": "Authored units and scale, not physical calibration", "evidence_refs": []}
    metadata = envelopes.describe()
    mapping = {"schema_version": 1, "profile": "world-alignment-v1",
        "from_frame": {"id": "body:"+api.canonical_hash(body), "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"},
        "to_frame": metadata["world_frame"],
        "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]},
        "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
        "claims": {"spatial": deepcopy(claim), "clock": deepcopy(claim)}}
    points = ("left_ankle", "right_ankle")
    binding = {"schema_version": 1, "profile": api.BINDING_PROFILE, "source_sha256": source,
        "body_sha256": sha256(body_bytes).hexdigest(), "envelope_bindings": metadata["bindings"],
        "human_entity_ref": metadata["channels"][0]["entity_ref"], "body_to_world": mapping,
        "channels": [{"channel_id": channel["id"], "body_point": points[index % 2], "status": "assumed",
                      "method": "Authored correspondence to named core-body endpoint",
                      "uncertainty": "A toe annotation does not measure an ankle; fixture assumes correspondence explicitly"}
                     for index, channel in enumerate(metadata["channels"])]}
    return envelopes, body, binding


def prepare(api, data=None, **changes):
    envelopes, body, binding = fixture(api) if data is None else data
    return api.prepare_surface_interval_objective(envelopes, **({"body_bytes": raw(body), "binding_bytes": raw(binding)} | changes))


def repin(api, data):
    _, body, binding = data
    binding["body_sha256"] = sha256(raw(body)).hexdigest()
    binding["body_to_world"]["from_frame"]["id"] = "body:"+api.canonical_hash(body)
    return data


def evaluate(objective, position, **changes):
    args = {"body_sha256": objective.describe()["body_sha256"], "body_time_s": 0.,
            "coordinate_frame": objective.coordinate_frame, "positions": {"left_ankle": position}}
    return objective.evaluate(0, **(args | changes))


def test_closed_enclosure_zero_inside_not_only_at_center(api):
    objective = prepare(api)
    bound = objective.frame(0)["source_envelopes"]["anchors"][0]["world_envelope_m"]
    for point in ([.13, -.10, .21], [.19, -.06, .59], bound["minimum"], bound["maximum"]):
        result = evaluate(objective, point)
        assert result["objective_m2"] == 0.
        assert result["channels"][0]["signed_outside_m"] == [0., 0., 0.]
        assert result["channels"][0]["active_axes"] == [False, False, False]
    description = objective.describe()
    assert description["contact"] == description["load"] == description["grasp"] == "unknown"
    assert description["confidence"] is None
    assert description["geometry_modified"] is description["body_or_floor_installed"] is False
    assert description["optimizer_installation"] == "not_installed_no_existing_fitter_callback"
    assert description["original_start_frame"] == 0


def test_signed_outside_distance_uses_nearest_bound_not_midpoint(api):
    objective = prepare(api)
    result = evaluate(objective, [.1, -.2, .8])
    residual = np.array([-.026, -.092, .2])
    np.testing.assert_allclose(result["channels"][0]["signed_outside_m"], residual, atol=1e-15)
    assert result["objective_m2"] == pytest.approx(.5*float(residual @ residual))
    assert result["channels"][0]["distance_to_outer_envelope_m"] == pytest.approx(float(np.linalg.norm(residual)))
    assert result["gradient_m2_per_rad"] is None


def test_rotation_scale_translation_and_derivative_units_chain(api):
    data = fixture(api)
    data[2]["body_to_world"]["spatial"] = {"scale": 2., "rotation_wxyz": [math.sqrt(.5), 0., 0., math.sqrt(.5)],
                                           "translation": [.01, -.02, .03]}
    objective = prepare(api, data)
    point = np.array([.2, .3, .7])
    jacobian = np.array([[1., .2], [.4, -.3], [.1, 2.]])
    arguments = {"jacobians": {"left_ankle": jacobian}, "parameter_names": [api.NAMES[0], api.NAMES[1]]}
    result = evaluate(objective, point, **arguments)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(result["channels"][0]["world_position_m"], 2*rotation @ point + [.01, -.02, .03], atol=1e-15)
    np.testing.assert_allclose(result["channels"][0]["jacobian_m_per_rad"], 2*rotation @ jacobian, atol=1e-15)
    step = 1e-6
    numerical = [(evaluate(objective, point + step*jacobian[:, i])["objective_m2"]
                  - evaluate(objective, point - step*jacobian[:, i])["objective_m2"])/(2*step) for i in range(2)]
    np.testing.assert_allclose(result["gradient_m2_per_rad"], numerical, atol=1e-9)
    np.testing.assert_allclose(result["gauss_newton_m2_per_rad2"], (2*rotation @ jacobian).T @ (2*rotation @ jacobian), atol=1e-14)


def test_boundary_convention_zero_derivative_without_moving_the_box(api):
    objective = prepare(api)
    upper = objective.frame(0)["source_envelopes"]["anchors"][0]["world_envelope_m"]["maximum"]
    result = evaluate(objective, upper, jacobians={"left_ankle": [[1.], [1.], [1.]]}, parameter_names=[api.NAMES[0]])
    assert result["channels"][0]["jacobian_m_per_rad"] == [[0.], [0.], [0.]]
    assert result["gradient_m2_per_rad"] == [0.] and result["gauss_newton_m2_per_rad2"] == [[0.]]


@pytest.mark.parametrize("change,reason", [
    (lambda d: d.update(body_sha256="b"*64), "binding_body_or_source_identity_differs"),
    (lambda d: d.update(source_sha256="b"*64), "binding_body_or_source_identity_differs"),
    (lambda d: d["envelope_bindings"].update(camera_sha256="b"*64), "binding_envelope_inputs_differ"),
    (lambda d: d["body_to_world"]["from_frame"].update(id="other-body"), "incompatible_body_world_coordinate_frames"),
    (lambda d: d["body_to_world"]["from_frame"].update(length_unit="m"), "incompatible_body_world_coordinate_frames"),
    (lambda d: d["body_to_world"]["to_frame"].update(id="other-world"), "incompatible_body_world_coordinate_frames"),
    (lambda d: d.update(body_to_world=None), "missing_or_unknown_body_world_alignment"),
    (lambda d: d["body_to_world"]["clock"].update(target_origin_s=.1), "body_world_source_clock_differs"),
    (lambda d: d["human_entity_ref"].update(object_id="not-person"), "human_entity_not_in_exact_source"),
    (lambda d: d.update(channels=[]), "missing_or_unknown_endpoint_correspondence"),
])
def test_incompatible_or_missing_joins_abstain_without_selecting_later_frame(api, change, reason):
    data = fixture(api)
    change(data[2])
    objective = prepare(api, data)
    assert objective.frame_count == 4 and objective.describe()["original_start_frame"] == 0
    assert objective.frame(0)["channels"][0]["reason"] == reason
    result = evaluate(objective, object())
    assert result["objective_m2"] is None and result["counts"] == {"retained_channels": 1, "evaluated_channels": 0, "unavailable_channels": 1}


def test_missing_binding_and_unknown_coordinate_claim_are_useful_unavailable_results(api):
    data = fixture(api)
    missing = prepare(api, data, binding_bytes=None)
    assert missing.describe()["binding_unavailable_reason"] == "missing_explicit_body_binding"
    mapping = data[2]["body_to_world"]
    mapping["spatial"] = None
    mapping["claims"]["spatial"]["status"] = "unknown"
    unknown = prepare(api, data)
    assert unknown.frame(0)["channels"][0]["reason"] == "missing_or_unknown_body_world_alignment"


def test_same_video_hash_does_not_authorize_another_source_entity(api):
    def alter(data):
        other = deepcopy(data[0]["objects"][0]); other["id"] = "other"
        data[0]["objects"].append(other)
        for old in list(data[0]["observations"]):
            new = deepcopy(old); new.update(id=f"{data[0]['source']['sha256']}:observation:{old['frame']}:other",
                                            object_id="other", evidence_ids=[])
            data[0]["observations"].append(new)
        data[0]["observations"].sort(key=lambda row: row["frame"])
    data = fixture(api, source_changes=alter)
    data[2]["human_entity_ref"]["object_id"] = "other"
    assert prepare(api, data).frame(0)["channels"][0]["reason"] == "channel_entity_differs_from_body_entity"


def test_body_source_evidence_and_original_clocks_cannot_be_relabelled(api):
    data = fixture(api)
    for row in data[1]["reference_evidence"]:
        row["evidence_id"] = row["evidence_id"].replace("a"*64, "b"*64)
    with pytest.raises(ValueError, match="source-frame identity"):
        prepare(api, repin(api, data))
    data = fixture(api)
    for key in ("reference", "reference_evidence", "root_reference"):
        data[1][key][0]["time_s"] = .001
    data[2]["body_to_world"]["clock"]["target_origin_s"] = -.001
    objective = prepare(api, repin(api, data))
    assert objective.frame(0)["channels"][0]["reason"] == "body_reference_source_clock_differs"


def test_unsampled_excluded_occluded_and_metadata_only_denominators_survive(api):
    def alter(data):
        data[0]["source"]["metadata_frames"] = 7
        data[1]["frame_count"] = 7
        last = deepcopy(data[1]["frames"][-1]); last.update(frame=6, time_s=6/30)
        data[1]["frames"].append(last)
        data[0]["observations"][1].update(partition="unsampled", visibility="not_sampled", evidence_ids=[])
        data[0]["observations"][2]["partition"] = "audit"
        data[3]["annotations"][3].update(status="occluded", pixel="must not be read", uncertainty_px="must not be read")
    data = fixture(api, count=6, source_changes=alter, body_indices=(0, 2, 3, 4, 5))
    objective = prepare(api, data)
    reasons = [objective.frame(i)["channels"][0]["reason"] for i in range(7)]
    assert reasons == [None, "source_not_sampled", "excluded_evidence_partition", "annotation_occluded", None, None, "metadata_only_not_decoded"]
    assert objective.describe()["counts"]["retained_channel_slots"] == 7
    assert objective.evaluate(2, body_sha256=sha256(raw(data[1])).hexdigest(), body_time_s=2/30,
                              coordinate_frame=objective.coordinate_frame,
                              positions=object(), jacobians=object(), parameter_names=object())["objective_m2"] is None
    assert objective.evaluate(6, body_sha256=sha256(raw(data[1])).hexdigest(), body_time_s=None,
                              coordinate_frame=objective.coordinate_frame,
                              positions=object())["channels"][0]["reason"] == "metadata_only_not_decoded"


def test_missing_body_reference_does_not_interpolate_or_advance_original_start(api):
    data = fixture(api, body_indices=(0, 2, 3))
    data[2]["channels"][0].update(body_point=None, status="unknown")
    objective = prepare(api, data)
    assert objective.describe()["original_start_statuses"][0]["status"] == "unavailable"
    data[2]["channels"][0].update(body_point="left_ankle", status="assumed")
    objective = prepare(api, data)
    assert objective.frame(1)["channels"][0]["reason"] == "original_body_reference_unavailable"
    assert objective.describe()["original_start_frame"] == 0


def test_partial_candidate_queries_keep_every_channel_and_missing_derivative(api):
    objective = prepare(api, fixture(api, channels=2))
    missing = evaluate(objective, [.1, -.2, .8])
    assert missing["counts"] == {"retained_channels": 2, "evaluated_channels": 1, "unavailable_channels": 1}
    assert missing["channels"][1]["reason"] == "missing_candidate_body_point"
    partial = evaluate(objective, None, positions={"left_ankle": [.1, -.2, .8], "right_ankle": [.1, -.2, .8]},
                       jacobians={"left_ankle": [[1.], [1.], [1.]]}, parameter_names=[api.NAMES[0]])
    assert partial["counts"]["evaluated_channels"] == 2
    assert partial["gradient_m2_per_rad"] is None and partial["gauss_newton_m2_per_rad2"] is None


@pytest.mark.parametrize("changes", [
    {"body_sha256": "b"*64}, {"body_time_s": .001}, {"body_time_s": True},
    {"coordinate_frame": {"id": "other-body", "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"}},
    {"positions": {"left_toe": [0., 0., 0.]}},
    {"positions": {"left_ankle": [0., float("nan"), 0.]}},
    {"positions": {"left_ankle": [0., True, 0.]}},
    {"jacobians": {"left_ankle": [[1.], [1.], [1.]]}, "parameter_names": ["root_z"]},
    {"jacobians": {"left_ankle": [[1., 2.], [1., 2.], [1., 2.]]}, "parameter_names": ["torso_twist", "torso_twist"]},
])
def test_candidate_identity_names_clock_and_numeric_guards(api, changes):
    with pytest.raises(ValueError):
        evaluate(prepare(api), [.1, -.2, .8], **changes)


def test_unknown_or_observed_correspondence_cannot_become_measured_geometry(api):
    data = fixture(api)
    data[2]["channels"][0]["status"] = "observed"
    with pytest.raises(ValueError, match="explicit assumed"):
        prepare(api, data)


def test_inputs_outputs_and_cached_queries_cannot_mutate_preparation(api, monkeypatch):
    data = fixture(api)
    before = (raw(data[1]), raw(data[2]), data[0].frame(0))
    objective = prepare(api, data)
    expected = evaluate(objective, [.1, -.2, .8])
    data[1]["reference"][0]["joints"][api.NAMES[0]] = 1.
    data[2]["body_to_world"]["spatial"]["translation"][0] = 999.
    output = objective.frame(0); output["source_envelopes"]["anchors"][0]["world_envelope_m"]["minimum"][0] = -999.
    output = objective.describe(); output["binding"]["channels"][0]["body_point"] = "right_wrist"
    monkeypatch.setattr(api, "_sources", denied)
    monkeypatch.setattr(api, "load_world_alignment", denied)
    monkeypatch.setattr(api, "canonical_hash", denied)
    monkeypatch.setattr(api, "sha256", denied)
    assert evaluate(objective, [.1, -.2, .8]) == expected
    assert objective.body_bytes == before[0] and objective.binding_bytes == before[1]
    assert data[0].frame(0) == before[2]


def test_cheap_byte_row_and_cooperative_bounds(api, monkeypatch):
    data = fixture(api)
    with pytest.raises(ValueError, match="bounded exact"):
        prepare(api, data, body_bytes=b"x"*(api.MAX_BODY_BYTES+1))
    with pytest.raises(ValueError, match="bounded exact"):
        prepare(api, data, binding_bytes=b"x"*(api.MAX_BINDING_BYTES+1))
    body = data[1]
    for key in ("reference", "reference_evidence", "root_reference"):
        body[key] = body[key][:1]*901
    with pytest.raises(ValueError, match="bounded body"):
        prepare(api, repin(api, data))
    data = fixture(api)
    monkeypatch.setattr(api, "MAX_SECONDS", 0.)
    with pytest.raises(TimeoutError):
        prepare(api, data)


def test_result_cache_and_query_deadline_fail_without_dropping_channels(api, monkeypatch):
    data = fixture(api)
    objective = prepare(api, data)
    monkeypatch.setattr(api, "MAX_QUERY_SECONDS", 0.)
    with pytest.raises(TimeoutError):
        evaluate(objective, [.1, -.2, .8])
    monkeypatch.setattr(api, "MAX_CACHE_BYTES", 100)
    with pytest.raises(ValueError, match="cache exceeds"):
        prepare(api, data)


def test_actual_kinematic_producer_routes_pelvis_position_and_derivative(api, monkeypatch):
    objective = prepare(api)
    body = json.loads(objective.body_bytes)
    count, parameters = len(api.POINTS), len(api.NAMES)
    point_names = tuple(api.POINTS)
    pelvis = count
    base = np.zeros((count+1, 3))
    base[pelvis] = [1., -2., 3.]
    for index in range(count):
        base[index] = [.3 + index*.005, -.2, .8]
    derivatives = np.zeros((count+1, 3, parameters))
    derivatives[pelvis, :, 0] = [.4, -.3, .2]
    for index in range(count):
        derivatives[index, :, 0] = [1.+index*.01, .2, -.1]
        derivatives[index, :, 1] = [-.2, .3, .4]
    calls = {"forward": 0, "jacobian": 0}
    fake = ModuleType("mujoco")

    def forward(model, state):
        assert model.body_sha256 == objective.describe()["body_sha256"]
        state.site_xpos[:] = base + np.einsum("ijk,k->ij", derivatives, state.qpos)
        calls["forward"] += 1

    def jac_site(model, state, destination, rotation, site):
        destination[:] = derivatives[site]
        calls["jacobian"] += 1

    fake.mj_forward, fake.mj_jacSite = forward, jac_site
    fake_human = ModuleType("weft.human"); fake_human.HumanEnv = denied
    guarded = builtins.__import__

    def imported(name, *args, **kwargs):
        if name == "mujoco":
            return fake
        if name == "weft.human":
            return fake_human
        return guarded(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", imported)
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    monkeypatch.setitem(sys.modules, "weft.human", fake_human)
    # Load the unchanged producer under a private name; stub imports cannot poison later tests.
    spec = importlib.util.spec_from_file_location("_weft_interval_analytic_kinematics", Path(api.__file__).with_name("body_refinement.py"))
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    model = SimpleNamespace(nv=parameters, body_sha256=sha256(objective.body_bytes).hexdigest())
    env = SimpleNamespace(model=model, spec=body, joint_names=api.NAMES, _q_array=np.arange(parameters),
                          _v_array=np.arange(parameters), _site_ids=dict(zip(point_names, range(count))) | {"pelvis": pelvis})
    state = SimpleNamespace(qpos=np.zeros(parameters), site_xpos=np.zeros((count+1, 3)))
    angles = np.array([body["reference"][0]["joints"][name] for name in env.joint_names])
    relative, jacobian = producer._kinematics(env, state, angles, True)
    result = objective.evaluate_kinematics(0, body_sha256=model.body_sha256, body_time_s=body["reference"][0]["time_s"],
        coordinate_frame=objective.coordinate_frame,
        point_names=tuple(producer.JOINT_LANDMARKS), joint_names=env.joint_names, relative_positions=relative,
        pelvis_position=state.site_xpos[pelvis], relative_jacobian=jacobian, pelvis_jacobian=derivatives[pelvis])
    index = point_names.index("left_ankle")
    np.testing.assert_allclose(result["channels"][0]["world_position_m"], base[index], atol=1e-15)
    np.testing.assert_allclose(result["channels"][0]["jacobian_m_per_rad"], derivatives[index], atol=1e-15)
    assert calls == {"forward": 1, "jacobian": 13}
    assert result["parameter_names"] == list(env.joint_names)
    with pytest.raises(ValueError, match="pelvis Jacobian"):
        objective.evaluate_kinematics(0, body_sha256=model.body_sha256, body_time_s=0., point_names=point_names,
            coordinate_frame=objective.coordinate_frame,
            joint_names=env.joint_names, relative_positions=relative, pelvis_position=state.site_xpos[pelvis], relative_jacobian=jacobian)


def test_representative_preparation_and_cached_query_workload(api, record_property):
    data = fixture(api, count=600, channels=2)
    start = time.monotonic(); objective = prepare(api, data); prepared = time.monotonic()-start
    digest = objective.describe()["body_sha256"]
    start = time.monotonic()
    for index in range(60):
        output = objective.evaluate(index, body_sha256=digest, body_time_s=index/30,
            coordinate_frame=objective.coordinate_frame,
            positions={"left_ankle": [.1, -.2, .8], "right_ankle": [.1, -.2, .8]},
            jacobians={name: np.ones((3, 19)) for name in ("left_ankle", "right_ankle")}, parameter_names=api.NAMES)
        assert output["counts"]["evaluated_channels"] == 2
    queried = time.monotonic()-start
    assert objective.describe()["counts"]["retained_channel_slots"] == 1200
    record_property("preparation_seconds", prepared)
    record_property("sixty_cached_full_jacobian_queries_seconds", queried)
    record_property("cached_join_bytes", objective.describe()["cached_join_bytes"])
    record_property("source_frames", 600)
    record_property("retained_channel_slots", 1200)
